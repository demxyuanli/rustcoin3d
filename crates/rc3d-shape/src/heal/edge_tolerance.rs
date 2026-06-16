//! Auto-fix edge tolerances based on PCurve-to-3D geometric deviation.
//!
//! Samples the PCurve at N uniform points, evaluates the surface at those UV
//! coordinates to get 3D positions, and compares with the edge's 3D curve at
//! the same parameter t. Sets `edge.tolerance` to encompass the max deviation.
//!
//! OCC alignment: ShapeFix_Edge::FixSameParameter + ShapeFix_Edge::FixVertexTolerance

use rc3d_core::math::Real;
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, ShellKey};
use crate::topo_iter;

/// Report from edge tolerance auto-fix pass.
#[derive(Debug, Default)]
pub struct EdgeToleranceReport {
    pub edges_checked: usize,
    pub tolerances_increased: usize,
    pub tolerances_decreased: usize,
    pub max_adjustment: Real,
}

/// Auto-fix a single edge's tolerance based on PCurve-to-3D deviation.
///
/// For each face referencing this edge, samples the PCurve at `n_samples`
/// uniform t values, evaluates the face surface to get 3D positions, and
/// compares with the edge's 3D curve. Sets edge tolerance to the max
/// deviation (clamped to [min_tolerance, max_tolerance]).
///
/// Returns `Some(new_tolerance)` if the tolerance changed, `None` otherwise.
pub fn auto_fix_edge_tolerance(
    ek: EdgeKey,
    reg: &mut BRepStore,
    min_tolerance: Real,
    max_tolerance: Real,
    n_samples: usize,
) -> Option<Real> {
    let edge = reg.edges.get(ek)?;
    let face_keys: Vec<FaceKey> = edge.pcurves.keys().copied().collect();

    if face_keys.is_empty() {
        return None;
    }

    let mut max_dev = 0.0_f64;

    for fk in &face_keys {
        let face = match reg.faces.get(*fk) {
            Some(f) => f,
            None => continue,
        };
        let pcurve = match edge.pcurves.get(fk) {
            Some(pc) => pc,
            None => continue,
        };

        for i in 0..=n_samples {
            let t = i as Real / n_samples as Real;
            let uv = pcurve.d0(t);

            // Evaluate surface at PCURVE UV to get 3D position
            let pcurve_3d = face.surface.d0_native(uv.0, uv.1);

            // Compare with edge's 3D curve at same t
            let curve_3d = edge.curve.d0(t);

            let dev = (pcurve_3d - curve_3d).length();
            max_dev = max_dev.max(dev);
        }
    }

    let new_tolerance = max_dev.max(min_tolerance).min(max_tolerance);

    if let Some(edge_mut) = reg.edges.get_mut(ek) {
        let old = edge_mut.tolerance;
        if (new_tolerance - old).abs() > 1e-10 {
            edge_mut.tolerance = new_tolerance;
            return Some(new_tolerance);
        }
    }

    None
}

/// Run auto-fix on all edges in a shell.
///
/// OCC: ShapeFix_Shell applies ShapeFix_Edge to each edge.
pub fn auto_fix_shell_edge_tolerances(
    shell_key: ShellKey,
    reg: &mut BRepStore,
    min_tolerance: Real,
    max_tolerance: Real,
) -> EdgeToleranceReport {
    let mut report = EdgeToleranceReport::default();
    let edges = topo_iter::iter_edges_of_shell(shell_key, reg);
    report.edges_checked = edges.len();

    for ek in &edges {
        if let Some(new_tol) =
            auto_fix_edge_tolerance(*ek, reg, min_tolerance, max_tolerance, 16)
        {
            // Re-read to get the old tolerance for reporting
            if let Some(edge) = reg.edges.get(*ek) {
                let adj = (new_tol - edge.tolerance).abs()
                    .max((edge.tolerance - new_tol).abs());
                report.max_adjustment = report.max_adjustment.max(adj);
                if new_tol > edge.tolerance {
                    report.tolerances_increased += 1;
                } else {
                    report.tolerances_decreased += 1;
                }
            }
        }
    }
    report
}

/// Collect all unique edge keys from a shell's faces and wires.
pub(crate) fn collect_shell_edges(
    reg: &BRepStore,
    shell_key: crate::topo::ShellKey,
) -> Vec<crate::topo::EdgeKey> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let Some(shell) = reg.shells.get(shell_key) else { return out; };
    for &(fk, _) in &shell.faces {
        let Some(face) = reg.faces.get(fk) else { continue; };
        let wires = std::iter::once(face.outer_wire).chain(face.inner_wires.iter().copied());
        for wk in wires {
            let Some(wire) = reg.wires.get(wk) else { continue; };
            for &(ek, _) in &wire.edges {
                if seen.insert(ek) { out.push(ek); }
            }
        }
        for &ek in &face.seam_edges { if seen.insert(ek) { out.push(ek); } }
        for &ek in &face.degenerated_edges { if seen.insert(ek) { out.push(ek); } }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::store::BRepStore;
    use crate::topo::*;
    use rc3d_core::math::PVec3;

    #[test]
    fn test_auto_fix_increases_tolerance_for_mismatched_pcurve() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let v0 = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.0), 1e-6);
        let v1 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-6);

        let curve_3d = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        // PCurve with intentional error: y=0.1 instead of y=0.0
        let pcurve = Curve2d::Line {
            origin: (0.0, 0.1),
            direction: (1.0, -0.1),
        };

        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let ek = reg.edges.insert(BRepEdge {
            curve: curve_3d,
            tolerance: 1e-6,
            v_low: v0,
            v_high: v1,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: [(fk, pcurve)].into(),
        });

        let new_tol = auto_fix_edge_tolerance(ek, &mut reg, 1e-6, 0.1, 32);
        assert!(new_tol.is_some(), "should adjust tolerance for mismatched PCurve");
        assert!(
            new_tol.unwrap() > 1e-6,
            "mismatched PCurve should increase tolerance"
        );
    }

    #[test]
    fn test_auto_fix_exact_pcurve_preserves_tight_tolerance() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-6);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-6);

        let curve_3d = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        // Exact PCurve matches 3D curve
        let pcurve = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };

        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let ek = reg.edges.insert(BRepEdge {
            curve: curve_3d,
            tolerance: 1e-6,
            v_low: v0,
            v_high: v1,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: [(fk, pcurve)].into(),
        });

        let new_tol = auto_fix_edge_tolerance(ek, &mut reg, 1e-6, 0.1, 16);
        // Exact PCurve means deviation ~0, tolerance should stay at min_tolerance (1e-6)
        // Since current tolerance == min_tolerance, no change needed → None
        assert!(new_tol.is_none(), "exact PCurve should not change tolerance");
    }
}
