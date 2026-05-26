//! Constrained Delaunay triangulation for trimmed faces (spade CDT, earcut fallback).
//! OCC BRepMesh_Delaun stand-in: CDT-first + insert-time Steiner refinement.

use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;
use rc3d_core::utils::hash::f32x3_quantized_bits;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use super::face_fill::{effective_min_size, FaceFillConfig};
use super::face_uv::{point_in_trim, FaceUvLoops};
use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::topo::BRepFace;

fn insert_uv(
    cdt: &mut ConstrainedDelaunayTriangulation<Point2<f64>>,
    uv: (f32, f32),
    gi: usize,
    handles: &mut Vec<spade::handles::FixedVertexHandle>,
    uv_to_gi: &mut HashMap<(u64, u64), usize>,
) -> Option<spade::handles::FixedVertexHandle> {
    let pt = Point2::new(uv.0 as f64, uv.1 as f64);
    let Ok(h) = cdt.insert(pt) else {
        return None;
    };
    let key = ((uv.0 * 1e6) as u64, (uv.1 * 1e6) as u64);
    uv_to_gi.insert(key, gi);
    handles.push(h);
    Some(h)
}

/// CDT-first triangulation with insert-time Steiner refinement.
/// Returns (flat list of global vertex indices per triangle, max_chord_error).
pub fn triangulate_uv_cdt_with_steiner(
    loops: &FaceUvLoops,
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    config: &FaceFillConfig,
) -> (Vec<usize>, f32) {
    if loops.outer.boundary.len() < 3 {
        return (Vec::new(), 0.0);
    }

    // Build initial CDT
    let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
        ConstrainedDelaunayTriangulation::new();
    let mut handles: Vec<spade::handles::FixedVertexHandle> = Vec::new();
    let mut uv_to_gi: HashMap<(u64, u64), usize> = HashMap::new();

    // Insert outer boundary
    let outer_start = handles.len();
    for v in &loops.outer.boundary {
        if insert_uv(&mut cdt, v.uv, v.global_idx, &mut handles, &mut uv_to_gi).is_none() {
            return (Vec::new(), 0.0);
        }
    }
    // Add outer constraints
    let n_outer = loops.outer.boundary.len();
    for i in 0..n_outer {
        let a = handles[outer_start + i];
        let b = handles[outer_start + (i + 1) % n_outer];
        let _ = cdt.try_add_constraint(a, b);
    }

    // Insert inner boundaries with constraints
    for inner in &loops.inners {
        let start = handles.len();
        for v in &inner.boundary {
            if insert_uv(&mut cdt, v.uv, v.global_idx, &mut handles, &mut uv_to_gi).is_none() {
                return (Vec::new(), 0.0);
            }
        }
        let n = inner.boundary.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = handles[start + i];
            let b = handles[start + (i + 1) % n];
            let before = cdt.num_constraints();
            let added = cdt.try_add_constraint(a, b);
            if added.is_empty() && cdt.num_constraints() == before && !cdt.exists_constraint(a, b) {
                return (Vec::new(), 0.0);
            }
        }
    }

    // Steiner insertion loop (OCC BRepMesh_Delaun node insertion)
    let mut max_chord = 0.0f32;
    if config.enable_interior && config.deflection_interior > 0.0 {
        let min_sz = effective_min_size(config);
        for _iter in 0..config.max_adapt_iterations {
            let mut splits: Vec<(f64, f64)> = Vec::new();
            for face_h in cdt.inner_faces() {
                let verts: Vec<_> = face_h
                    .vertices()
                    .iter()
                    .map(|v| v.fix().index())
                    .collect();
                if verts.len() != 3 {
                    continue;
                }
                let i0 = verts[0];
                let i1 = verts[1];
                let i2 = verts[2];

                let uv0 = (
                    cdt.vertex(handles[i0]).position().x as f32,
                    cdt.vertex(handles[i0]).position().y as f32,
                );
                let uv1 = (
                    cdt.vertex(handles[i1]).position().x as f32,
                    cdt.vertex(handles[i1]).position().y as f32,
                );
                let uv2 = (
                    cdt.vertex(handles[i2]).position().x as f32,
                    cdt.vertex(handles[i2]).position().y as f32,
                );

                let p0 = face.surface.d0_native(uv0.0, uv0.1);
                let p1 = face.surface.d0_native(uv1.0, uv1.1);
                let p2 = face.surface.d0_native(uv2.0, uv2.1);

                let mut tri_split = false;
                for (a, b, uva, uvb) in [
                    (&p0, &p1, uv0, uv1),
                    (&p1, &p2, uv1, uv2),
                    (&p2, &p0, uv2, uv0),
                ] {
                    let edge_len = (*a - *b).length();
                    let mid_uv = ((uva.0 + uvb.0) * 0.5, (uva.1 + uvb.1) * 0.5);
                    let on_surf = face.surface.d0_native(mid_uv.0, mid_uv.1);
                    let dev = ((*a + *b) * 0.5 - on_surf).length();
                    max_chord = max_chord.max(dev);
                    if dev > config.deflection_interior && edge_len > min_sz {
                        tri_split = true;
                    }
                    // Angular deflection
                    let na = face.surface.normal_native(uva.0, uva.1);
                    let nb = face.surface.normal_native(uvb.0, uvb.1);
                    let angle = na.normalize().dot(nb.normalize()).max(-1.0).min(1.0).acos();
                    if angle > config.angular_deflection && edge_len > min_sz {
                        tri_split = true;
                    }
                }
                if tri_split {
                    let cu = (uv0.0 + uv1.0 + uv2.0) / 3.0;
                    let cv = (uv0.1 + uv1.1 + uv2.1) / 3.0;
                    splits.push((cu as f64, cv as f64));
                }
            }
            if splits.is_empty() {
                break;
            }
            // Dedup and insert Steiner points
            let mut dedup = HashSet::new();
            for (u, v) in splits {
                let key = ((u * 1e4) as u64, (v * 1e4) as u64);
                if dedup.contains(&key) {
                    continue;
                }
                dedup.insert(key);
                let pt_3d = face.surface.d0_native(u as f32, v as f32);
                let hash = f32x3_quantized_bits([pt_3d.x, pt_3d.y, pt_3d.z]);
                let gi = *pos_to_idx.entry(hash).or_insert_with(|| {
                    let i = global_vertices.len();
                    global_vertices.push(pt_3d);
                    let mut n = face.surface.normal_native(u as f32, v as f32);
                    if !face.same_sense {
                        n = -n;
                    }
                    global_normals.push(n);
                    i
                });
                insert_uv(
                    &mut cdt,
                    (u as f32, v as f32),
                    gi,
                    &mut handles,
                    &mut uv_to_gi,
                );
            }
        }
    }

    // Extract triangles inside trim domain
    let outer_uv: Vec<(f32, f32)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
    let inner_uv: Vec<Vec<(f32, f32)>> = loops
        .inners
        .iter()
        .map(|l| l.boundary.iter().map(|v| v.uv).collect())
        .collect();

    let mut tris = Vec::new();
    for face_h in cdt.inner_faces() {
        let verts: Vec<_> = face_h
            .vertices()
            .iter()
            .map(|v| v.fix().index())
            .collect();
        if verts.len() != 3 {
            continue;
        }
        let uv0 = (
            cdt.vertex(handles[verts[0]]).position().x as f32,
            cdt.vertex(handles[verts[0]]).position().y as f32,
        );
        let uv1 = (
            cdt.vertex(handles[verts[1]]).position().x as f32,
            cdt.vertex(handles[verts[1]]).position().y as f32,
        );
        let uv2 = (
            cdt.vertex(handles[verts[2]]).position().x as f32,
            cdt.vertex(handles[verts[2]]).position().y as f32,
        );
        let cu = (uv0.0 + uv1.0 + uv2.0) / 3.0;
        let cv = (uv0.1 + uv1.1 + uv2.1) / 3.0;
        if point_in_trim(cu, cv, &outer_uv, &inner_uv) {
            let key0 = ((uv0.0 * 1e6) as u64, (uv0.1 * 1e6) as u64);
            let key1 = ((uv1.0 * 1e6) as u64, (uv1.1 * 1e6) as u64);
            let key2 = ((uv2.0 * 1e6) as u64, (uv2.1 * 1e6) as u64);
            if let (Some(&gi0), Some(&gi1), Some(&gi2)) =
                (uv_to_gi.get(&key0), uv_to_gi.get(&key1), uv_to_gi.get(&key2))
            {
                tris.push(gi0);
                tris.push(gi1);
                tris.push(gi2);
            }
        }
    }

    (tris, max_chord)
}

/// Backward-compat: UV-only CDT without surface/refinement (no Steiner points).
/// Use `triangulate_uv_cdt_with_steiner` for the full pipeline.
pub fn triangulate_uv_cdt(loops: &FaceUvLoops) -> Option<Vec<usize>> {
    if loops.outer.boundary.len() < 3 {
        return None;
    }
    let face = BRepFace {
        surface: SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        },
        outer_wire: Default::default(),
        inner_wires: vec![],
        same_sense: true,
        tolerance: 1e-4,
        seam_edges: vec![],
        color: None,
    };
    let mut verts = Vec::new();
    let mut norms = Vec::new();
    let mut pmap = HashMap::new();
    let config = FaceFillConfig {
        enable_interior: false,
        ..Default::default()
    };
    let (tris, _) = triangulate_uv_cdt_with_steiner(
        loops, &face, &mut verts, &mut norms, &mut pmap, &config,
    );
    if tris.is_empty() {
        None
    } else {
        Some(tris)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::face_uv::{UvLoop, UvSource, UvVertex};

    #[test]
    fn steiner_split_on_curved_surface() {
        let surface = SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 10.0 };
        let face = BRepFace {
            surface,
            outer_wire: Default::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
        };
        // Four points on a sphere radius=10 at equator band (v-native ≈ PI/2)
        // Native UV: Sphere::native_uv_to_d0 maps u→u/TAU, v→v/PI
        // u=0, v≈PI/2 → d0(0, 0.5) = (10, 0, 0)
        // u≈PI/2, v≈PI/2 → d0(0.25, 0.5) = (0, 10, 0)
        // Below equator: v≈PI/2+1 → d0(0.25, (PI/2+1)/PI) = (0, ~5.4, ~-8.4)
        let outer = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 0, uv: (0.0, 1.5708) },
                UvVertex { global_idx: 1, uv: (1.5708, 1.5708) },
                UvVertex { global_idx: 2, uv: (1.5708, 2.5708) },
                UvVertex { global_idx: 3, uv: (0.0, 2.5708) },
            ],
        };
        let loops = FaceUvLoops { outer, inners: vec![], uv_source: UvSource::Pcurve };

        let mut verts = vec![
            face.surface.d0_native(0.0, 1.5708),
            face.surface.d0_native(1.5708, 1.5708),
            face.surface.d0_native(1.5708, 2.5708),
            face.surface.d0_native(0.0, 2.5708),
        ];
        let mut norms = vec![Vec3::Z; 4];
        let mut pos_map = HashMap::new();
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(f32x3_quantized_bits([v.x, v.y, v.z]), i);
        }

        let config_loose = FaceFillConfig {
            enable_interior: true,
            deflection_interior: 0.5,
            min_size: 0.1,
            min_size_relative: 0.0,
            shell_min_size: 0.0,
            max_adapt_iterations: 4,
            angular_deflection: std::f32::consts::PI,
        };

        let (tris_loose, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, &mut verts, &mut norms, &mut pos_map, &config_loose,
        );
        let verts_before = verts.len();

        let config_tight = FaceFillConfig {
            deflection_interior: 0.01,
            max_adapt_iterations: 4,
            ..config_loose.clone()
        };
        let (tris_tight, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, &mut verts, &mut norms, &mut pos_map, &config_tight,
        );
        assert!(verts.len() > verts_before,
            "tight deflection should insert Steiner points on sphere, verts {} -> {}",
            verts_before, verts.len());
        assert!(tris_tight.len() > tris_loose.len(),
            "Steiner points should produce more triangles");
    }

    #[test]
    fn cdt_holed_rect_no_triangles_inside_hole() {
        let outer = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 0, uv: (0.0, 0.0) },
                UvVertex { global_idx: 1, uv: (10.0, 0.0) },
                UvVertex { global_idx: 2, uv: (10.0, 10.0) },
                UvVertex { global_idx: 3, uv: (0.0, 10.0) },
            ],
        };
        let inner = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 4, uv: (3.0, 3.0) },
                UvVertex { global_idx: 5, uv: (7.0, 3.0) },
                UvVertex { global_idx: 6, uv: (7.0, 7.0) },
                UvVertex { global_idx: 7, uv: (3.0, 7.0) },
            ],
        };
        let loops = FaceUvLoops {
            outer,
            inners: vec![inner],
            uv_source: UvSource::Pcurve,
        };

        let tris = triangulate_uv_cdt(&loops).expect("CDT should succeed");
        assert!(tris.len() >= 3);

        let gi_to_uv: std::collections::HashMap<usize, (f32, f32)> = loops
            .outer
            .boundary
            .iter()
            .chain(loops.inners[0].boundary.iter())
            .map(|v| (v.global_idx, v.uv))
            .collect();

        let outer_uv: Vec<(f32, f32)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
        let inner_uv: Vec<Vec<(f32, f32)>> = loops
            .inners
            .iter()
            .map(|l| l.boundary.iter().map(|v| v.uv).collect())
            .collect();

        for chunk in tris.chunks(3) {
            let c = (
                (gi_to_uv[&chunk[0]].0 + gi_to_uv[&chunk[1]].0 + gi_to_uv[&chunk[2]].0) / 3.0,
                (gi_to_uv[&chunk[0]].1 + gi_to_uv[&chunk[1]].1 + gi_to_uv[&chunk[2]].1) / 3.0,
            );
            let in_hole = c.0 > 3.5 && c.0 < 6.5 && c.1 > 3.5 && c.1 < 6.5;
            assert!(!in_hole, "triangle centroid {:?} should not lie in hole", c);
            assert!(point_in_trim(c.0, c.1, &outer_uv, &inner_uv));
        }
    }
}
