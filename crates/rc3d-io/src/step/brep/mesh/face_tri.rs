//! Constrained Delaunay face triangulation. T2.3-T2.4

use std::collections::HashMap;
use spade::handles::FixedVertexHandle;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation as _};
use rc3d_core::math::Vec3;
use rc3d_core::utils::hash::f32x3_quantized_bits;
use crate::step::brep::topo::{EdgeKey, FaceKey};
use crate::step::brep::registry::BRepRegistry;
use crate::step::mesh_result::MeshResult;
use super::edge_disc::EdgePolygon;

/// Triangulate a face's UV domain with edge polygons as constraints.
///
/// Builds a constrained Delaunay triangulation (CDT) in the face's UV parameter
/// space using the edge polygon 2D points as boundary constraints.  Each
/// triangle's UV centroid is lifted to 3D via `SurfaceGeom::d0` to produce the
/// output mesh.
pub fn triangulate_face(
    face_key: FaceKey,
    reg: &BRepRegistry,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
) -> Option<MeshResult> {
    let face = reg.faces.get(face_key)?;
    let wire = reg.wires.get(face.outer_wire)?;

    // ── Collect unique UV positions ──────────────────────────────
    let mut uv_points: Vec<(f64, f64)> = Vec::new();
    let mut uv_to_idx: HashMap<(u64, u64), usize> = HashMap::new();
    let mut edge_vertex_indices: Vec<Vec<usize>> = Vec::new();

    for &(ek, _orient) in &wire.edges {
        if let Some(poly) = edge_polygons.get(&ek) {
            if let Some(pcurve_pts) = poly.params_2d.get(&face_key) {
                let mut indices = Vec::new();
                for &(_t, (u, v)) in pcurve_pts {
                    let key = (
                        (u as f64 * 1e6) as u64,
                        (v as f64 * 1e6) as u64,
                    );
                    let idx = *uv_to_idx.entry(key).or_insert_with(|| {
                        let i = uv_points.len();
                        uv_points.push((u as f64, v as f64));
                        i
                    });
                    indices.push(idx);
                }
                edge_vertex_indices.push(indices);
            }
        }
    }

    if uv_points.len() < 3 {
        return None;
    }

    // ── Build CDT ────────────────────────────────────────────────
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::default();

    // Insert all UV points
    let mut fixed_handles: Vec<FixedVertexHandle> = Vec::new();
    for &(u, v) in &uv_points {
        let h = cdt.insert(Point2::new(u, v)).ok()?;
        fixed_handles.push(h);
    }

    // Add constrained edges — wire edges joined end-to-end form the
    // closed boundary (corner vertices deduplicate, so the last point
    // of edge N and first point of edge N+1 share an index).
    for edge_indices in &edge_vertex_indices {
        for w in edge_indices.windows(2) {
            if w[0] != w[1] {
                cdt.add_constraint(fixed_handles[w[0]], fixed_handles[w[1]]);
            }
        }
    }

    // ── Extract triangles from CDT ──────────────────────────────
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut indices: Vec<i32> = Vec::new();
    let mut pos_map: HashMap<[u32; 3], i32> = HashMap::new();

    // Build a reverse-lookup: FixedVertexHandle → insertion index
    let handle_to_idx: HashMap<FixedVertexHandle, usize> = fixed_handles
        .iter()
        .enumerate()
        .map(|(i, &h)| (h, i))
        .collect();

    for tri_face in cdt.inner_faces() {
        let vs = tri_face.vertices();
        let fv0 = vs[0].fix();
        let fv1 = vs[1].fix();
        let fv2 = vs[2].fix();

        let i0 = handle_to_idx.get(&fv0)?;
        let i1 = handle_to_idx.get(&fv1)?;
        let i2 = handle_to_idx.get(&fv2)?;

        let u0 = uv_points[*i0];
        let u1 = uv_points[*i1];
        let u2 = uv_points[*i2];

        // Evaluate 3D positions from surface
        let p0 = face.surface.d0(u0.0 as f32, u0.1 as f32);
        let p1 = face.surface.d0(u1.0 as f32, u1.1 as f32);
        let p2 = face.surface.d0(u2.0 as f32, u2.1 as f32);

        // Compute analytic normal at triangle centroid
        let uc = ((u0.0 + u1.0 + u2.0) / 3.0) as f32;
        let vc = ((u0.1 + u1.1 + u2.1) / 3.0) as f32;
        let mut n = face.surface.normal(uc, vc);
        if !face.same_sense {
            n = -n;
        }

        let idx0 = get_or_insert(&p0, n, &mut vertices, &mut normals, &mut pos_map);
        let idx1 = get_or_insert(&p1, n, &mut vertices, &mut normals, &mut pos_map);
        let idx2 = get_or_insert(&p2, n, &mut vertices, &mut normals, &mut pos_map);

        indices.extend_from_slice(&[idx0, idx1, idx2, -1]);
    }

    if vertices.is_empty() {
        None
    } else {
        Some(MeshResult {
            vertices,
            indices,
            normals,
        })
    }
}

/// Deduplicate vertex insertion using spatial hash.
fn get_or_insert(
    pos: &Vec3,
    normal: Vec3,
    vertices: &mut Vec<Vec3>,
    normals: &mut Vec<Vec3>,
    pos_map: &mut HashMap<[u32; 3], i32>,
) -> i32 {
    let hash = f32x3_quantized_bits([pos.x, pos.y, pos.z]);
    if let Some(&idx) = pos_map.get(&hash) {
        return idx;
    }
    let idx = vertices.len() as i32;
    vertices.push(*pos);
    normals.push(normal);
    pos_map.insert(hash, idx);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::topo::*;
    use crate::step::brep::registry::BRepRegistry;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use super::super::edge_disc::{discretize_all_edges, EdgeDiscConfig};

    #[test]
    fn test_triangulate_plane_face() {
        let mut reg = BRepRegistry::new();

        // Four corners of a unit square in the XY plane
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 1.0, 0.0), 1e-4);

        // Face: the plane z=0
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: reg.wires.insert(BRepWire { edges: vec![] }),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
        });

        // 3D edge curves (square boundary)
        let c0 = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
        };
        let c1 = CurveGeom::Line {
            origin: Vec3::new(1.0, 0.0, 0.0),
            direction: Vec3::new(0.0, 1.0, 0.0),
        };
        let c2 = CurveGeom::Line {
            origin: Vec3::new(1.0, 1.0, 0.0),
            direction: Vec3::new(-1.0, 0.0, 0.0),
        };
        let c3 = CurveGeom::Line {
            origin: Vec3::new(0.0, 1.0, 0.0),
            direction: Vec3::new(0.0, -1.0, 0.0),
        };

        // PCURVEs map t∈[0,1] → (u,v) for this face.
        // e0: (0,0)→(1,0)   e1: (1,0)→(1,1)
        // e2: (1,1)→(0,1)   e3: (0,1)→(0,0)
        let pc0 = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
        };
        let pc1 = CurveGeom::Line {
            origin: Vec3::new(1.0, 0.0, 0.0),
            direction: Vec3::new(0.0, 1.0, 0.0),
        };
        let pc2 = CurveGeom::Line {
            origin: Vec3::new(1.0, 1.0, 0.0),
            direction: Vec3::new(-1.0, 0.0, 0.0),
        };
        let pc3 = CurveGeom::Line {
            origin: Vec3::new(0.0, 1.0, 0.0),
            direction: Vec3::new(0.0, -1.0, 0.0),
        };

        let e0 = reg.add_edge_with_pcurve(v0, v1, c0, 1e-4, face_key, pc0);
        let e1 = reg.add_edge_with_pcurve(v1, v2, c1, 1e-4, face_key, pc1);
        let e2 = reg.add_edge_with_pcurve(v2, v3, c2, 1e-4, face_key, pc2);
        let e3 = reg.add_edge_with_pcurve(v3, v0, c3, 1e-4, face_key, pc3);

        // Update the face's outer wire
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = reg.wires.insert(BRepWire {
                edges: vec![
                    (e0, Orientation::Forward),
                    (e1, Orientation::Forward),
                    (e2, Orientation::Forward),
                    (e3, Orientation::Forward),
                ],
            });
        }

        let edge_polygons = discretize_all_edges(&reg, &EdgeDiscConfig::default());
        let mesh = triangulate_face(face_key, &reg, &edge_polygons).unwrap();

        assert!(!mesh.vertices.is_empty(), "should produce vertices");
        assert!(mesh.indices.len() >= 12, "at least 2 triangles (4 indices each)");
        // All vertices should lie on z=0
        for v in &mesh.vertices {
            assert!((v.z - 0.0).abs() < 1e-4, "vertex should be on the plane z=0");
        }
    }

    #[test]
    fn test_degenerate_too_few_points() {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);

        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: reg.wires.insert(BRepWire { edges: vec![] }),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
        });

        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let e0 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, face_key, line.clone());

        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = reg.wires.insert(BRepWire {
                edges: vec![(e0, Orientation::Forward)],
            });
        }

        // Only one edge → only collinear UV points → not enough for a face
        let edge_polygons = discretize_all_edges(&reg, &EdgeDiscConfig::default());
        let result = triangulate_face(face_key, &reg, &edge_polygons);
        assert!(result.is_none(), "single edge should not produce a triangulation");
    }
}
