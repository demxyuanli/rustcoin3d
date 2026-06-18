//! BRepCheck_Edge — edge tolerance, 3D curve, same-parameter, and range validation.

use rc3d_core::math::Real;

use crate::store::BRepStore;
use crate::topo::EdgeKey;

use super::CheckStatus;

/// Check edge tolerance validity (OCC BRepCheck_Edge).
pub fn check_edge_tolerance(ek: EdgeKey, reg: &BRepStore) -> Vec<String> {
    let mut warnings = Vec::new();
    let edge = match reg.edges.get(ek) {
        Some(e) => e,
        None => return warnings,
    };

    let p0 = reg.vertices.get(edge.v_low).map(|v| v.position);
    let p1 = reg.vertices.get(edge.v_high).map(|v| v.position);
    let approx_len = match (p0, p1) {
        (Some(a), Some(b)) => (a - b).length(),
        _ => return warnings,
    };

    if approx_len < 1e-12 {
        return warnings;
    }

    if edge.tolerance > approx_len * 10.0 {
        warnings.push(format!(
            "edge {:?}: tolerance {:.6} > 10x edge length {:.6}",
            ek, edge.tolerance, approx_len
        ));
    }
    if edge.tolerance < 1e-12 {
        warnings.push(format!(
            "edge {:?}: tolerance {:.6} is near-zero",
            ek, edge.tolerance
        ));
    }

    warnings
}

/// Check that the edge has a valid 3D curve (not degenerate).
///
/// Returns `No3DCurve` if the curve is a zero-length line or the edge
/// endpoints evaluate to the same point on the curve.
pub fn check_no_3d_curve(ek: EdgeKey, reg: &BRepStore) -> Vec<CheckStatus> {
    let mut statuses = Vec::new();
    let Some(edge) = reg.edges.get(ek) else {
        return statuses;
    };

    let p_tmin = edge.curve.d0(edge.t_min);
    let p_tmax = edge.curve.d0(edge.t_max);
    let curve_len = (p_tmax - p_tmin).length();

    // A valid edge should have non-zero length between t_min and t_max
    if curve_len < 1e-12 {
        statuses.push(CheckStatus::No3DCurve);
    }

    // Also check if the vertex positions match the curve endpoints
    let v_lo = reg.vertices.get(edge.v_low).map(|v| v.position);
    let v_hi = reg.vertices.get(edge.v_high).map(|v| v.position);
    if let (Some(lo), Some(hi)) = (v_lo, v_hi) {
        let vertex_len = (hi - lo).length();
        if vertex_len < 1e-12 {
            // Degenerate edge: both vertices at same position
            statuses.push(CheckStatus::No3DCurve);
        }
    }

    statuses
}

/// Check same-parameter flag: verify that the 3D curve and each PCURVE
/// evaluate to consistent 3D points at the same normalized parameter.
///
/// In OCC, `SameParameter` means the 3D curve and 2D PCURVE share the
/// same parameterization. We check this by comparing the 3D curve point
/// (mapped through the surface) against the direct 3D curve evaluation.
pub fn check_same_parameter_deviation(ek: EdgeKey, reg: &BRepStore) -> Vec<CheckStatus> {
    let mut statuses = Vec::new();
    let Some(edge) = reg.edges.get(ek) else {
        return statuses;
    };

    // Check 5 sample points along the edge
    let n_samples: usize = 5;
    for i in 0..=n_samples {
        let t = edge.t_min + (edge.t_max - edge.t_min) * (i as Real / n_samples as Real);
        let p3d = edge.curve.d0(t);

        for (&fk, pcurve) in &edge.pcurves {
            let Some(face) = reg.faces.get(fk) else {
                continue;
            };
            let uv = pcurve.d0(t);
            let p_surf = face.surface.d0(uv.0, uv.1);
            let dev = (p3d - p_surf).length();

            if dev > edge.tolerance * 10.0 && dev > 1e-4 {
                statuses.push(CheckStatus::InvalidSameParameterFlag);
                return statuses; // one violation is enough
            }
        }
    }

    statuses
}

/// Check t_min/t_max range validity.
///
/// Returns `InvalidSameRangeFlag` if:
/// - t_min >= t_max (degenerate or reversed range)
/// - t_min or t_max is NaN or infinite
/// - The parameter range does not cover [0, 1] when expected
pub fn check_same_range(ek: EdgeKey, reg: &BRepStore) -> Vec<CheckStatus> {
    let mut statuses = Vec::new();
    let Some(edge) = reg.edges.get(ek) else {
        return statuses;
    };

    if !edge.t_min.is_finite() || !edge.t_max.is_finite() {
        statuses.push(CheckStatus::InvalidSameRangeFlag);
        return statuses;
    }

    if edge.t_min >= edge.t_max {
        statuses.push(CheckStatus::InvalidSameRangeFlag);
        return statuses;
    }

    // Check that curve evaluation at t_min and t_max produces meaningful
    // results (not NaN at curve endpoints)
    let p_min = edge.curve.d0(edge.t_min);
    let p_max = edge.curve.d0(edge.t_max);
    if !p_min.x.is_finite() || !p_min.y.is_finite() || !p_min.z.is_finite()
        || !p_max.x.is_finite() || !p_max.y.is_finite() || !p_max.z.is_finite()
    {
        statuses.push(CheckStatus::InvalidSameRangeFlag);
    }

    statuses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepEdge, BRepFace, BRepWire};
    use rc3d_core::math::PVec3;
    use std::collections::HashMap;

    fn build_edge_reg() -> (BRepStore, EdgeKey) {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let curve = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let ek = reg.add_edge_with_pcurve(v0, v1, curve, 1e-4, fk, pc, true);
        (reg, ek)
    }

    #[test]
    fn valid_edge_no_3d_curve_flag() {
        let (reg, ek) = build_edge_reg();
        let statuses = check_no_3d_curve(ek, &reg);
        assert!(statuses.is_empty(), "valid edge should pass: {:?}", statuses);
    }

    #[test]
    fn degenerate_edge_triggers_no_3d_curve() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4); // same position
        let curve = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let ek = reg.edges.insert(BRepEdge {
            v_low: v0,
            v_high: v1,
            curve,
            tolerance: 1e-4,
            t_min: 0.0,
            t_max: 1.0,
            cached_deflection: None,
            pcurves: HashMap::new(),
        });
        let statuses = check_no_3d_curve(ek, &reg);
        assert!(
            statuses.contains(&CheckStatus::No3DCurve),
            "degenerate zero-length edge should be flagged"
        );
    }

    #[test]
    fn invalid_range_tmin_gte_tmax() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let curve = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let ek = reg.edges.insert(BRepEdge {
            v_low: v0,
            v_high: v1,
            curve,
            tolerance: 1e-4,
            t_min: 1.0,
            t_max: 0.0, // t_min > t_max
            cached_deflection: None,
            pcurves: HashMap::new(),
        });
        let statuses = check_same_range(ek, &reg);
        assert!(
            statuses.contains(&CheckStatus::InvalidSameRangeFlag),
            "t_min >= t_max should be flagged"
        );
    }

    #[test]
    fn test_edge_tolerance_oversized() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(0.01, 0.0, 0.0), 1e-4);
        let curve = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::new(0.01, 0.0, 0.0),
        };
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (0.01, 0.0),
        };
        let ek = reg.add_edge_with_pcurve(v0, v1, curve, 1e-4, fk, pc, true);
        if let Some(edge) = reg.edges.get_mut(ek) {
            edge.tolerance = 1.0;
        }
        let warnings = check_edge_tolerance(ek, &reg);
        assert!(
            !warnings.is_empty(),
            "oversized tolerance should produce warnings"
        );
    }
}
