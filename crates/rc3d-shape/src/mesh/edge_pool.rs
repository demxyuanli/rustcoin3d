//! Shared boundary vertex pool with topological edge-sample welding.
//!
//! Phase P3: equivalent edges (same `v_lo`/`v_hi`, different `EdgeKey`) share
//! mesh vertices via unified arc-length parameter along the physical edge.

use std::collections::HashMap;

use rc3d_core::math::{Real, PVec3};

use crate::geom::curve2d::Curve2d;
use crate::geom::curve_eval::find_param_on_curve;
use crate::geom::SurfaceGeom;
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, ShellKey, VertexKey};

use super::edge_disc::{eval_pcurve_on_surface, EdgePolygon};

/// Boundary vertex index keyed by (face, edge, sample) — OCC BRepAdaptor_Curve(E, F).
pub type FaceEdgeBoundaryIdx = HashMap<(FaceKey, EdgeKey, usize), usize>;

use super::boundary::{register_boundary_point_indexed, BoundaryPosIndex};

/// True when two edges between the same vertices describe the same 3D curve (STEP duplicate),
/// not a distinct curve on an adjacent offset/revolution face (wall thickness apart).
fn curves_should_weld(reg: &BRepStore, ek_a: EdgeKey, ek_b: EdgeKey) -> bool {
    let (Some(ea), Some(eb)) = (reg.edges.get(ek_a), reg.edges.get(ek_b)) else {
        return false;
    };
    let p_lo = reg.vertices.get(ea.v_low).map(|v| v.position).unwrap_or(PVec3::ZERO);
    let p_hi = reg.vertices.get(ea.v_high).map(|v| v.position).unwrap_or(PVec3::ZERO);
    let tol = ea.tolerance.max(eb.tolerance);
    let chord = (p_hi - p_lo).length().max(tol);
    let tight = (chord * 0.01).max(1e-4) + tol * 10.0;
    let mid_a = ea.curve.d0(0.5);
    let mid_b = eb.curve.d0(0.5);
    if (mid_a - mid_b).length() <= tight {
        return true;
    }
    // STEP duplicate-edge variants (polyline vs line) can differ at mid by a few % of chord.
    let loose = chord * 0.055 + tol * 10.0;
    let t_b = find_param_on_curve(&eb.curve, mid_a);
    if (mid_a - eb.curve.d0(t_b)).length() <= loose {
        return true;
    }
    let t_a = find_param_on_curve(&ea.curve, mid_b);
    (mid_b - ea.curve.d0(t_a)).length() <= loose
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EdgeSampleKey {
    v_lo: VertexKey,
    v_hi: VertexKey,
    t_q: u32,
}

fn quantize_t(t: Real) -> u32 {
    (t.clamp(0.0, 1.0) * 1_000_000.0).round() as u32
}

/// Collect edge groups that share the same vertex pair AND the same 3D curve geometry.
/// Offset/revolution wall pairs share vertices but must not weld (different radii).
pub fn equivalent_edge_groups(reg: &BRepStore) -> Vec<Vec<EdgeKey>> {
    let mut out = Vec::new();
    for keys in reg.edge_hash_index.values() {
        if keys.len() <= 1 {
            continue;
        }
        let mut clusters: Vec<Vec<EdgeKey>> = Vec::new();
        for &ek in keys {
            let mut placed = false;
            for cluster in &mut clusters {
                if cluster
                    .iter()
                    .any(|&other| curves_should_weld(reg, ek, other))
                {
                    cluster.push(ek);
                    placed = true;
                    break;
                }
            }
            if !placed {
                clusters.push(vec![ek]);
            }
        }
        for cluster in clusters {
            if cluster.len() > 1 {
                out.push(cluster);
            }
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
fn arc_length_fraction(polyline: &[(Real, PVec3)], pt: PVec3) -> Real {
    if polyline.len() < 2 {
        return 0.0;
    }

    let mut seg_lens: Vec<Real> = Vec::with_capacity(polyline.len() - 1);
    let mut total = 0.0_f64;
    for w in polyline.windows(2) {
        let len = (w[1].1 - w[0].1).length();
        seg_lens.push(len);
        total += len;
    }
    if total <= 1e-12 {
        return 0.0;
    }

    let mut best_s = 0.0_f64;
    let mut best_d2 = f64::MAX;
    let mut cum = 0.0_f64;

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
    pt: PVec3,
    group: &[EdgeKey],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
) -> Real {
    if group.len() <= 1 {
        if let Some(edge) = reg.edges.get(ek) {
            let p_lo = reg.vertices.get(edge.v_low).map(|v| v.position).unwrap_or(PVec3::ZERO);
            let p_hi = reg.vertices.get(edge.v_high).map(|v| v.position).unwrap_or(PVec3::ZERO);
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
    vertices: &[PVec3],
    weld_tol: Real,
) -> Real {
    let mut max_gap = 0.0_f64;
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
    global_vertices: &mut Vec<PVec3>,
    global_normals: &mut Vec<PVec3>,
    pos_to_idx: &mut BoundaryPosIndex,
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
                    let created = register_boundary_point_indexed(
                        pt,
                        global_vertices,
                        global_normals,
                        pos_to_idx,
                    );
                    topo_sample_idx.insert(key, created);
                    created
                } else {
                    register_boundary_point_indexed(
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

/// Per-face boundary pool: 3D points from each face's PCURVE on that face's surface.
pub fn build_face_boundary_pool(
    reg: &BRepStore,
    shell_key: ShellKey,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    global_vertices: &mut Vec<PVec3>,
    global_normals: &mut Vec<PVec3>,
    pos_to_idx: &mut BoundaryPosIndex,
    vertex_mesh_idx: &HashMap<VertexKey, usize>,
) -> FaceEdgeBoundaryIdx {
    let mut out: FaceEdgeBoundaryIdx = HashMap::new();
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return out,
    };

    for &(face_key, _) in &shell.faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        let wires: Vec<crate::topo::WireKey> = std::iter::once(face.outer_wire)
            .chain(face.inner_wires.iter().copied())
            .collect();
        for wire_key in wires {
            let wire = match reg.wires.get(wire_key) {
                Some(w) => w,
                None => continue,
            };
            for &(ek, _) in &wire.edges {
                let Some(edge) = reg.edges.get(ek) else {
                    continue;
                };
                let Some(pcurve) = edge.pcurves.get(&face_key) else {
                    continue;
                };
                let Some(poly) = edge_polygons.get(&ek) else {
                    continue;
                };
                let surface = &face.surface;
                let n = poly.params_3d.len();
                for (pi, &(t, _)) in poly.params_3d.iter().enumerate() {
                    let pt = boundary_point_on_face(
                        face_key,
                        edge,
                        pcurve,
                        surface,
                        pi,
                        n,
                        t,
                        poly,
                        reg,
                    );
                    let gi = if edge.v_low != edge.v_high {
                        if pi == 0 {
                            vertex_mesh_idx.get(&edge.v_low).copied()
                        } else if pi + 1 == n {
                            vertex_mesh_idx.get(&edge.v_high).copied()
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let gi = gi.unwrap_or_else(|| {
                        register_boundary_point_indexed(
                            pt,
                            global_vertices,
                            global_normals,
                            pos_to_idx,
                        )
                    });
                    out.insert((face_key, ek, pi), gi);
                }
            }
        }
    }
    out
}

/// 3D point on `face_key` for boundary sample `pi` (OCC BRepAdaptor_Curve(E, F)).
fn boundary_point_on_face(
    face_key: FaceKey,
    edge: &crate::topo::BRepEdge,
    pcurve: &Curve2d,
    surface: &SurfaceGeom,
    pi: usize,
    _n: usize,
    t: Real,
    poly: &EdgePolygon,
    _reg: &BRepStore,
) -> PVec3 {
    if let Some(pts) = poly.params_2d.get(&face_key) {
        if let Some(&(_, (u, v))) = pts.get(pi) {
            return surface.d0_native(u, v);
        }
    }
    let _ = edge;
    eval_pcurve_on_surface(pcurve, surface, t)
}

/// Max distance between per-face boundary samples and the face surface (PCURVE UV → `d0_native`).
pub fn measure_face_boundary_surface_gap(
    reg: &BRepStore,
    shell_key: ShellKey,
    config: &super::edge_disc::EdgeDiscConfig,
) -> Real {
    let edge_polygons = super::edge_disc::discretize_all_edges(reg, config);
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return 0.0,
    };
    let mut max_gap = 0.0_f64;

    for &(face_key, _) in &shell.faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        let wires: Vec<crate::topo::WireKey> = std::iter::once(face.outer_wire)
            .chain(face.inner_wires.iter().copied())
            .collect();
        for wire_key in wires {
            let wire = match reg.wires.get(wire_key) {
                Some(w) => w,
                None => continue,
            };
            for &(ek, _) in &wire.edges {
                let edge = match reg.edges.get(ek) {
                    Some(e) => e,
                    None => continue,
                };
                let Some(pcurve) = edge.pcurves.get(&face_key) else {
                    continue;
                };
                let poly = match edge_polygons.get(&ek) {
                    Some(p) => p,
                    None => continue,
                };
                let n = poly.params_3d.len();
                for (pi, &(t, _)) in poly.params_3d.iter().enumerate() {
                    let intended = boundary_point_on_face(
                        face_key, edge, pcurve, &face.surface, pi, n, t, poly, reg,
                    );
                    max_gap = max_gap.max(surface_gap_at_pcurve_uv(
                        &face.surface, intended, pcurve, t,
                    ));
                }
            }
        }
    }
    max_gap
}

fn surface_gap_at_pcurve_uv(surface: &SurfaceGeom, pt: PVec3, pcurve: &Curve2d, t: Real) -> Real {
    let uv = pcurve.d0(t);
    let mut best = (pt - surface.d0_native(uv.0, uv.1)).length();
    if matches!(surface, SurfaceGeom::Revolution { .. }) {
        const TAU: Real = std::f64::consts::TAU;
        if uv.0 <= 1.0 + 1e-4 {
            best = best.min((pt - surface.d0_native(uv.0 * TAU, uv.1)).length());
        }
        if uv.0 >= TAU * 0.25 {
            let un = uv.0 / TAU;
            if (un - uv.0).abs() > 1e-6 {
                best = best.min((pt - surface.d0_native(un, uv.1)).length());
            }
        }
    }
    if let Some(pu) = surface.native_u_period() {
        for shift in [-1.0_f64, 1.0] {
            best = best.min((pt - surface.d0_native(uv.0 + shift * pu, uv.1)).length());
        }
    }
    if let Some(pv) = surface.native_v_period() {
        for shift in [-1.0_f64, 1.0] {
            best = best.min((pt - surface.d0_native(uv.0, uv.1 + shift * pv)).length());
        }
    }
    best
}

/// Discretize edges and measure max positional gap between welded equivalent-edge samples.
pub fn measure_equivalent_edge_weld_gap(
    reg: &BRepStore,
    config: &super::edge_disc::EdgeDiscConfig,
) -> Real {
    let edge_polygons = super::edge_disc::discretize_all_edges(reg, config);
    let mut global_vertices: Vec<PVec3> = Vec::new();
    let mut global_normals: Vec<PVec3> = Vec::new();
    let mut pos_to_idx =
        BoundaryPosIndex::with_cell_size(crate::mesh::boundary::BOUNDARY_DEDUP_TOLERANCE);
    let mut vertex_mesh_idx: HashMap<VertexKey, usize> = HashMap::new();
    for (vk, v) in reg.vertices.iter() {
        let idx = register_boundary_point_indexed(
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
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(10.0, 0.0, 0.0), 1e-4);
        let _v2 = reg.find_or_add_vertex(PVec3::new(10.0, 10.0, 0.0), 1e-4);
        let _v3 = reg.find_or_add_vertex(PVec3::new(0.0, 10.0, 0.0), 1e-4);

        let plane = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
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
            origin: PVec3::ZERO,
            direction: PVec3::new(10.0, 0.0, 0.0),
        };
        let curve_b = CurveGeom::Polyline {
            points: vec![
                PVec3::ZERO,
                PVec3::new(5.0, 0.0, 0.5),
                PVec3::new(10.0, 0.0, 0.0),
            ],
        };
        let pcurve_a = Curve2d::Line { origin: (0.0, 0.0), direction: (10.0, 0.0) };
        let pcurve_b = Curve2d::Line { origin: (0.0, 10.0), direction: (10.0, 0.0) };

        // Two EdgeKeys for the same geometric edge (STEP duplicate-edge case).
        let ek_a = reg.add_edge_with_pcurve(v0, v1, curve_a, 1e-4, f0, pcurve_a, true);
        let ek_b = reg.add_edge_with_pcurve(v0, v1, curve_b, 1e-4, f1, pcurve_b, true);
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
        let mut pos_to_idx =
            BoundaryPosIndex::with_cell_size(crate::mesh::boundary::BOUNDARY_DEDUP_TOLERANCE);
        let mut vertex_mesh_idx = HashMap::new();
        for (vk, v) in reg.vertices.iter() {
            let idx = register_boundary_point_indexed(v.position, &mut verts, &mut norms, &mut pos_to_idx);
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
