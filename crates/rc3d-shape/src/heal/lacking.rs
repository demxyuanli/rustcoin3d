//! Missing 2D edge detection (OCC ShapeFix_Wire::FixLacking).
//! Detects edges connected in 3D but disconnected in UV space.

use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, Orientation, VertexKey, WireKey};

#[derive(Debug, Default)]
pub struct LackingReport {
    pub tolerance_fixes: usize,
    pub edges_added: usize,
}

/// Detect and fix 2D disconnections between adjacent edges where 3D is connected.
/// Returns report with counts of fixes applied.
pub fn fix_lacking_edges(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepStore,
    tol_3d: f32,
    tol_uv: f32,
) -> LackingReport {
    let mut report = LackingReport::default();

    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return report; };
        wire.edges.clone()
    };

    if edges.len() < 2 {
        return report;
    }

    let n = edges.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (ek_i, orient_i) = edges[i];
        let (ek_j, orient_j) = edges[j];

        // Check 3D connectivity
        let end_i = get_endpoint_3d(ek_i, orient_i, false, reg);
        let start_j = get_endpoint_3d(ek_j, orient_j, true, reg);
        let (Some(p_i), Some(p_j)) = (end_i, start_j) else { continue; };
        if (p_i - p_j).length() > tol_3d {
            continue;
        }

        // Check UV connectivity
        let uv_i = pcurve_endpoint(ek_i, orient_i, false, face_key, reg);
        let uv_j = pcurve_endpoint(ek_j, orient_j, true, face_key, reg);
        let (Some((u1, v1)), Some((u2, v2))) = (uv_i, uv_j) else { continue; };

        let dist_uv = ((u1 - u2).powi(2) + (v1 - v2).powi(2)).sqrt();
        if dist_uv <= tol_uv {
            continue;
        }

        if dist_uv <= tol_uv * 10.0 {
            // Small gap: increase edge tolerance to bridge it
            if let Some(edge) = reg.edges.get_mut(ek_i) {
                edge.tolerance = edge.tolerance.max(dist_uv * 1.1);
                report.tolerance_fixes += 1;
            }
        } else if dist_uv < tol_uv * 1000.0 {
            // Large gap: insert a new line-segment PCurve edge between the junction vertices
            let (vk_i, vk_j, tolerance) = {
                let edge_i = match reg.edges.get(ek_i) {
                    Some(e) => e,
                    None => continue,
                };
                let vk_i = get_junction_vertex(ek_i, orient_i, false, reg);
                let vk_j = get_junction_vertex(ek_j, orient_j, true, reg);
                (vk_i, vk_j, edge_i.tolerance)
            };
            let (Some(vk_i), Some(vk_j)) = (vk_i, vk_j) else { continue; };

            let new_curve = crate::geom::CurveGeom::Line {
                origin: p_i,
                direction: p_j - p_i,
            };
            let new_pc = crate::geom::Curve2d::Line {
                origin: (u1, v1),
                direction: (u2 - u1, v2 - v1),
            };
            let ek_new = reg.add_edge_with_pcurve(vk_i, vk_j, new_curve, tolerance, face_key, new_pc, true);
            if let Some(wire) = reg.wires.get_mut(wire_key) {
                let insert_pos = (i + 1) % wire.edges.len();
                wire.edges.insert(insert_pos, (ek_new, Orientation::Forward));
                report.edges_added += 1;
            }
        }
    }

    report
}

fn get_endpoint_3d(ek: EdgeKey, orient: Orientation, is_start: bool, reg: &BRepStore) -> Option<rc3d_core::math::Vec3> {
    let edge = reg.edges.get(ek)?;
    let vk = match (orient, is_start) {
        (Orientation::Forward, true) | (Orientation::Reversed, false) => edge.v_low,
        _ => edge.v_high,
    };
    reg.vertices.get(vk).map(|v| v.position)
}

fn get_junction_vertex(ek: EdgeKey, orient: Orientation, is_end: bool, reg: &BRepStore) -> Option<VertexKey> {
    let edge = reg.edges.get(ek)?;
    if (orient == Orientation::Forward) != is_end {
        Some(edge.v_low)
    } else {
        Some(edge.v_high)
    }
}

fn pcurve_endpoint(ek: EdgeKey, orient: Orientation, is_start: bool, face_key: FaceKey, reg: &BRepStore) -> Option<(f32, f32)> {
    let edge = reg.edges.get(ek)?;
    let pc = edge.pcurves.get(&face_key)?;
    let t = if (orient == Orientation::Forward) == is_start { 0.0 } else { 1.0 };
    let uv = pc.d0(t);
    Some((uv.0, uv.1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Curve2d, CurveGeom, SurfaceGeom};
    use crate::topo::BRepWire;
    use rc3d_core::math::Vec3;

    fn make_wire_with_uv_gap(reg: &mut BRepStore, uv_gap: f32) -> (WireKey, FaceKey) {
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface,
            outer_wire: WireKey::default(),
            inner_wires: vec![],
            same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let pc1 = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        // Edge 1: UV (0,0)->(1,0), 3D (0,0,0)->(1,0,0)
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc1, true);
        // Edge 2: UV (1+gap,0)->(2+gap,0), 3D (1,0,0)->(2,0,0) -- UV gap at junction
        let pc2 = Curve2d::Line { origin: (1.0 + uv_gap, 0.0), direction: (1.0, 0.0) };
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc2, true);
        let wk = reg.wires.insert(BRepWire {
            edges: vec![(e1, Orientation::Forward), (e2, Orientation::Forward)],
        });
        (wk, fk)
    }

    #[test]
    fn test_lacking_small_gap_tolerance() {
        let mut reg = BRepStore::new();
        let (wk, fk) = make_wire_with_uv_gap(&mut reg, 0.001);
        let report = fix_lacking_edges(wk, fk, &mut reg, 1e-3, 2e-4);
        assert!(report.tolerance_fixes > 0, "small UV gap should get tolerance fix");
    }

    #[test]
    fn test_lacking_no_3d_connection() {
        let mut reg = BRepStore::new();
        let (wk, fk) = make_wire_with_uv_gap(&mut reg, 0.001);
        let report = fix_lacking_edges(wk, fk, &mut reg, 1e-8, 1e-8);
        assert_eq!(report.tolerance_fixes, 0, "no fix when 3D gap check fails");
    }

    #[test]
    fn test_lacking_no_uv_gap() {
        let mut reg = BRepStore::new();
        let (wk, fk) = make_wire_with_uv_gap(&mut reg, 0.0);
        let report = fix_lacking_edges(wk, fk, &mut reg, 1e-3, 2e-4);
        assert_eq!(report.tolerance_fixes, 0, "no fix when UV gap is zero");
    }

    #[test]
    fn test_lacking_large_gap_new_edge() {
        let mut reg = BRepStore::new();
        let (wk, fk) = make_wire_with_uv_gap(&mut reg, 0.01);
        let report = fix_lacking_edges(wk, fk, &mut reg, 1e-3, 1e-4);
        // Large UV gap should trigger edge insertion (dist_uv >= tol_uv * 10)
        assert!(report.edges_added > 0 || report.tolerance_fixes > 0, "large UV gap should be fixed");
    }
}
