//! Shared boundary vertex pool with topological edge-sample welding.
//!
//! Phase P3: equivalent edges (same `v_lo`/`v_hi`, different `EdgeKey`) share
//! mesh vertices via unified arc-length parameter along the physical edge.

use std::collections::HashMap;

use rc3d_core::math::Vec3;

use crate::store::BRepStore;
use crate::topo::{EdgeKey, VertexKey};

use super::boundary::register_boundary_point;
use super::edge_disc::EdgePolygon;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EdgeSampleKey {
    v_lo: VertexKey,
    v_hi: VertexKey,
    t_q: u32,
}

fn quantize_t(t: f32) -> u32 {
    (t.clamp(0.0, 1.0) * 1_000_000.0).round() as u32
}

/// Collect edge groups that share the same canonical vertex pair.
pub fn equivalent_edge_groups(reg: &BRepStore) -> Vec<Vec<EdgeKey>> {
    let mut out = Vec::new();
    for keys in reg.edge_hash_index.values() {
        if keys.len() > 1 {
            out.push(keys.clone());
        }
    }
    out
}

fn pick_canonical_polygon<'a>(
    group: &[EdgeKey],
    edge_polygons: &'a HashMap<EdgeKey, EdgePolygon>,
) -> &'a EdgePolygon {
    group
        .iter()
        .max_by_key(|ek| edge_polygons.get(ek).map(|p| p.params_3d.len()).unwrap_or(0))
        .and_then(|ek| edge_polygons.get(ek))
        .expect("group edge must have polygon")
}

/// Arc-length fraction [0,1] of `pt` along a polyline (piecewise linear).
fn arc_length_fraction(polyline: &[(f32, Vec3)], pt: Vec3) -> f32 {
    if polyline.len() < 2 {
        return 0.0;
    }

    let mut seg_lens: Vec<f32> = Vec::with_capacity(polyline.len() - 1);
    let mut total = 0.0f32;
    for w in polyline.windows(2) {
        let len = (w[1].1 - w[0].1).length();
        seg_lens.push(len);
        total += len;
    }
    if total <= 1e-12 {
        return 0.0;
    }

    let mut best_s = 0.0f32;
    let mut best_d2 = f32::MAX;
    let mut cum = 0.0f32;

    for (i, &seg_len) in seg_lens.iter().enumerate() {
        let a = polyline[i].1;
        let b = polyline[i + 1].1;
        let ab = b - a;
        let len2 = ab.length_squared();
        if len2 > 1e-20 {
            let t = ((pt - a).dot(ab) / len2).clamp(0.0, 1.0);
            let closest = a + ab * t;
            let d2 = (pt - closest).length_squared();
            let s = (cum + t * seg_len) / total;
            if d2 < best_d2 {
                best_d2 = d2;
                best_s = s;
            }
        }
        cum += seg_len;
    }
    best_s
}

fn unified_arc_fraction(
    reg: &BRepStore,
    ek: EdgeKey,
    pt: Vec3,
    group: &[EdgeKey],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
) -> f32 {
    if group.len() <= 1 {
        if let Some(edge) = reg.edges.get(ek) {
            let p_lo = reg.vertices.get(edge.v_low).map(|v| v.position).unwrap_or(Vec3::ZERO);
            let p_hi = reg.vertices.get(edge.v_high).map(|v| v.position).unwrap_or(Vec3::ZERO);
            let chord = p_hi - p_lo;
            let len2 = chord.length_squared();
            if len2 > 1e-20 {
                let t = ((pt - p_lo).dot(chord) / len2).clamp(0.0, 1.0);
                return t;
            }
        }
        return 0.0;
    }

    let canonical = pick_canonical_polygon(group, edge_polygons);
    arc_length_fraction(&canonical.params_3d, pt)
}

/// Maximum 3D gap between welded samples on equivalent edge duplicates.
pub fn max_equivalent_edge_gap(
    reg: &BRepStore,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &HashMap<(EdgeKey, usize), usize>,
    vertices: &[Vec3],
    weld_tol: f32,
) -> f32 {
    let mut max_gap = 0.0f32;
    for group in equivalent_edge_groups(reg) {
        if group.len() < 2 {
            continue;
        }
        let canonical = pick_canonical_polygon(&group, edge_polygons);
        let n = canonical.params_3d.len();
        if n < 2 {
            continue;
        }

        let mut arc_to_idx: HashMap<u32, usize> = HashMap::new();
        for &ek in &group {
            let Some(poly) = edge_polygons.get(&ek) else {
                continue;
            };
            for (pi, &(_t, pt)) in poly.params_3d.iter().enumerate() {
                if pi == 0 || pi + 1 == poly.params_3d.len() {
                    continue;
                }
                let arc_s = unified_arc_fraction(reg, ek, pt, &group, edge_polygons);
                let key = quantize_t(arc_s);
                let Some(idx) = edge_boundary_idx.get(&(ek, pi)) else {
                    continue;
                };
                let idx = *idx;
                arc_to_idx
                    .entry(key)
                    .and_modify(|prev| {
                        let gap = (vertices[idx] - vertices[*prev]).length();
                        if gap > max_gap {
                            max_gap = gap;
                        }
                    })
                    .or_insert(idx);
            }
        }
    }
    let _ = weld_tol;
    max_gap
}

/// Build a shared boundary vertex pool keyed by `(EdgeKey, sample_index)`.
pub fn build_edge_boundary_pool(
    reg: &BRepStore,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    vertex_mesh_idx: &HashMap<VertexKey, usize>,
) -> HashMap<(EdgeKey, usize), usize> {
    let mut edge_boundary_idx: HashMap<(EdgeKey, usize), usize> = HashMap::new();
    let mut topo_sample_idx: HashMap<EdgeSampleKey, usize> = HashMap::new();

    let groups: Vec<Vec<EdgeKey>> = equivalent_edge_groups(reg);
    let ek_to_group: HashMap<EdgeKey, usize> = groups
        .iter()
        .enumerate()
        .flat_map(|(gi, g)| g.iter().map(move |&ek| (ek, gi)))
        .collect();

    for (&ek, poly) in edge_polygons {
        let edge = reg.edges.get(ek);
        let n = poly.params_3d.len();
        let group_owned;
        let group: &[EdgeKey] = if let Some(&gi) = ek_to_group.get(&ek) {
            &groups[gi]
        } else {
            group_owned = vec![ek];
            &group_owned
        };

        for (pi, &(t, pt)) in poly.params_3d.iter().enumerate() {
            let idx = if let Some(e) = edge {
                if e.v_low != e.v_high {
                    if pi == 0 {
                        vertex_mesh_idx.get(&e.v_low).copied()
                    } else if pi + 1 == n {
                        vertex_mesh_idx.get(&e.v_high).copied()
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let idx = idx.unwrap_or_else(|| {
                if let Some(e) = edge {
                    let arc_s = if group.len() > 1 {
                        unified_arc_fraction(reg, ek, pt, group, edge_polygons)
                    } else {
                        t
                    };
                    let key = EdgeSampleKey {
                        v_lo: e.v_low,
                        v_hi: e.v_high,
                        t_q: quantize_t(arc_s),
                    };
                    if let Some(&existing) = topo_sample_idx.get(&key) {
                        return existing;
                    }
                    let created = register_boundary_point(
                        pt,
                        global_vertices,
                        global_normals,
                        pos_to_idx,
                    );
                    topo_sample_idx.insert(key, created);
                    created
                } else {
                    register_boundary_point(
                        pt,
                        global_vertices,
                        global_normals,
                        pos_to_idx,
                    )
                }
            });

            edge_boundary_idx.insert((ek, pi), idx);
        }
    }

    edge_boundary_idx
}

/// Discretize edges and measure max positional gap between welded equivalent-edge samples.
pub fn measure_equivalent_edge_weld_gap(
    reg: &BRepStore,
    config: &super::edge_disc::EdgeDiscConfig,
) -> f32 {
    let edge_polygons = super::edge_disc::discretize_all_edges(reg, config);
    let mut global_vertices: Vec<Vec3> = Vec::new();
    let mut global_normals: Vec<Vec3> = Vec::new();
    let mut pos_to_idx: HashMap<[u32; 3], usize> = HashMap::new();
    let mut vertex_mesh_idx: HashMap<VertexKey, usize> = HashMap::new();
    for (vk, v) in reg.vertices.iter() {
        let idx = register_boundary_point(
            v.position,
            &mut global_vertices,
            &mut global_normals,
            &mut pos_to_idx,
        );
        vertex_mesh_idx.insert(vk, idx);
    }
    let bidx = build_edge_boundary_pool(
        reg,
        &edge_polygons,
        &mut global_vertices,
        &mut global_normals,
        &mut pos_to_idx,
        &vertex_mesh_idx,
    );
    max_equivalent_edge_gap(reg, &edge_polygons, &bidx, &global_vertices, 1e-3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepWire, Orientation};

    fn build_two_face_shared_duplicate_edges() -> (BRepStore, crate::topo::FaceKey, crate::topo::FaceKey) {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(10.0, 0.0, 0.0), 1e-4);
        let _v2 = reg.find_or_add_vertex(Vec3::new(10.0, 10.0, 0.0), 1e-4);
        let _v3 = reg.find_or_add_vertex(Vec3::new(0.0, 10.0, 0.0), 1e-4);

        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };

        let make_face = |reg: &mut BRepStore| {
            let w = reg.wires.insert(BRepWire { edges: vec![] });
            reg.faces.insert(BRepFace {
                surface: plane.clone(),
                outer_wire: w,
                inner_wires: vec![],
                same_sense: true,
                tolerance: 1e-4,
                seam_edges: vec![],
                color: None,
                degenerated_edges: vec![],
            })
        };

        let f0 = make_face(&mut reg);
        let f1 = make_face(&mut reg);

        let curve_a = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::new(10.0, 0.0, 0.0),
        };
        let curve_b = CurveGeom::Polyline {
            points: vec![
                Vec3::ZERO,
                Vec3::new(5.0, 0.0, 0.5),
                Vec3::new(10.0, 0.0, 0.0),
            ],
        };
        let pcurve_a = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(10.0, 0.0, 0.0),
        };
        let pcurve_b = CurveGeom::Line {
            origin: Vec3::new(0.0, 10.0, 0.0),
            direction: Vec3::new(10.0, 0.0, 0.0),
        };

        // Two EdgeKeys for the same geometric edge (STEP duplicate-edge case).
        let ek_a = reg.add_edge_with_pcurve(v0, v1, curve_a, 1e-4, f0, pcurve_a);
        let ek_b = reg.add_edge_with_pcurve(v0, v1, curve_b, 1e-4, f1, pcurve_b);
        assert_ne!(ek_a, ek_b);

        let w0 = reg.wires.insert(BRepWire {
            edges: vec![(ek_a, Orientation::Forward)],
        });
        let w1 = reg.wires.insert(BRepWire {
            edges: vec![(ek_b, Orientation::Forward)],
        });
        if let Some(f) = reg.faces.get_mut(f0) {
            f.outer_wire = w0;
        }
        if let Some(f) = reg.faces.get_mut(f1) {
            f.outer_wire = w1;
        }

        (reg, f0, f1)
    }

    #[test]
    fn duplicate_edge_keys_weld_by_arc_length() {
        let (reg, f0, f1) = build_two_face_shared_duplicate_edges();
        let shared = reg.find_shared_edges(f0, f1);
        assert!(
            shared.is_empty(),
            "duplicate EdgeKeys are not visible via find_shared_edges"
        );

        let groups = equivalent_edge_groups(&reg);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);

        let polys = super::super::edge_disc::discretize_all_edges(&reg, &super::super::edge_disc::EdgeDiscConfig::default());
        let mut verts = Vec::new();
        let mut norms = Vec::new();
        let mut pos_to_idx = HashMap::new();
        let mut vertex_mesh_idx = HashMap::new();
        for (vk, v) in reg.vertices.iter() {
            let idx = register_boundary_point(v.position, &mut verts, &mut norms, &mut pos_to_idx);
            vertex_mesh_idx.insert(vk, idx);
        }
        let bidx = build_edge_boundary_pool(
            &reg,
            &polys,
            &mut verts,
            &mut norms,
            &mut pos_to_idx,
            &vertex_mesh_idx,
        );

        let gap = max_equivalent_edge_gap(&reg, &polys, &bidx, &verts, 1e-3);
        assert!(
            gap < 1e-4,
            "equivalent duplicate edges should weld to same vertices, gap={gap}"
        );
    }
}
