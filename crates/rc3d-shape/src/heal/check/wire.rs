//! BRepCheck_Wire — closure, self-intersection, redundant edge validation.

use rc3d_core::math::Real;
use std::collections::HashSet;

use crate::store::BRepStore;
use crate::topo::{FaceKey, Orientation, WireKey};

use super::CheckStatus;

/// Check wire closure: measure the gap between the last edge's endpoint
/// and the first edge's start point.
///
/// Returns `Some(gap)` in 3D world units, or `None` if the wire is empty.
///
/// OCC alignment: BRepCheck_Wire::Closed()
pub fn check_wire_closed(wire_key: WireKey, reg: &BRepStore) -> Option<Real> {
    let wire = reg.wires.get(wire_key)?;
    if wire.edges.is_empty() {
        return None;
    }
    let n = wire.edges.len();
    let (first_ek, first_orient) = wire.edges[0];
    let (last_ek, last_orient) = wire.edges[n - 1];

    let first_edge = reg.edges.get(first_ek)?;
    let last_edge = reg.edges.get(last_ek)?;

    let first_start = if first_orient == Orientation::Forward {
        reg.vertices.get(first_edge.v_low)?.position
    } else {
        reg.vertices.get(first_edge.v_high)?.position
    };
    let last_end = if last_orient == Orientation::Forward {
        reg.vertices.get(last_edge.v_high)?.position
    } else {
        reg.vertices.get(last_edge.v_low)?.position
    };

    let gap = (last_end - first_start).length();
    Some(gap)
}

/// Check all wires of a face for closure and log gaps above tolerance.
pub fn check_face_wire_gaps(
    face_key: FaceKey,
    reg: &BRepStore,
    tolerance: Real,
) -> Vec<(WireKey, Real)> {
    let mut gaps = Vec::new();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return gaps,
    };
    for wk in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
        if let Some(gap) = check_wire_closed(*wk, reg) {
            if gap > tolerance {
                gaps.push((*wk, gap));
            }
        }
    }
    gaps
}

/// Check UV boundary self-intersection for a face.
pub fn check_uv_self_intersection(face_key: FaceKey, reg: &BRepStore) -> Vec<String> {
    let mut warnings = Vec::new();
    let Some(face) = reg.faces.get(face_key) else {
        return warnings;
    };
    let Some(wire) = reg.wires.get(face.outer_wire) else {
        return warnings;
    };
    let mut segments: Vec<((Real, Real), (Real, Real))> = Vec::new();
    for &(ek, _) in &wire.edges {
        let Some(edge) = reg.edges.get(ek) else {
            continue;
        };
        let Some(pcurve) = edge.pcurves.get(&face_key) else {
            continue;
        };
        let p0 = pcurve.d0(0.0);
        let p1 = pcurve.d0(1.0);
        segments.push(((p0.0, p0.1), (p1.0, p1.1)));
    }
    let n = segments.len();
    for i in 0..n {
        for j in (i + 2)..n {
            if i == 0 && j == n - 1 {
                continue;
            } // adjacent at loop closure
            if segments_intersect_2d(segments[i], segments[j]) {
                warnings.push(format!(
                    "face {:?}: UV boundary self-intersection between edge {} and edge {}",
                    face_key, i, j
                ));
            }
        }
    }
    warnings
}

fn segments_intersect_2d(
    a: ((Real, Real), (Real, Real)),
    b: ((Real, Real), (Real, Real)),
) -> bool {
    let ((ax0, ay0), (ax1, ay1)) = a;
    let ((bx0, by0), (bx1, by1)) = b;
    // Scale epsilon to coordinate magnitude for Real validity
    let extent = (ax0.abs()
        + ax1.abs()
        + bx0.abs()
        + bx1.abs()
        + ay0.abs()
        + ay1.abs()
        + by0.abs()
        + by1.abs())
        / 8.0;
    let eps = extent.max(1.0) * 1e-6;
    let d = (ax1 - ax0) * (by1 - by0) - (ay1 - ay0) * (bx1 - bx0);
    if d.abs() < eps {
        return false; // parallel or near-parallel
    }
    let t = ((bx0 - ax0) * (by1 - by0) - (by0 - ay0) * (bx1 - bx0)) / d;
    let u = ((bx0 - ax0) * (ay1 - ay0) - (by0 - ay0) * (ax1 - ax0)) / d;
    let edge_eps = eps;
    t > edge_eps && t < 1.0 - edge_eps && u > edge_eps && u < 1.0 - edge_eps
}

/// Check for redundant edges in a wire (same edge appearing multiple times).
///
/// An edge listed twice in the same wire is an OCC `BRepCheck_RedundantEdge`
/// violation and indicates a duplicated topology entry.
pub fn check_redundant_edge(wire_key: WireKey, reg: &BRepStore) -> Vec<CheckStatus> {
    let mut statuses = Vec::new();
    let Some(wire) = reg.wires.get(wire_key) else {
        return statuses;
    };

    let mut seen = HashSet::new();
    for &(ek, _) in &wire.edges {
        if !seen.insert(ek) {
            statuses.push(CheckStatus::RedundantEdge);
            break;
        }
    }

    statuses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepWire, Orientation};
    use rc3d_core::math::PVec3;

    #[test]
    fn test_check_wire_closed_detects_open_wire() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let v2 = reg.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), 1e-4);
        let line = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let fk = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: PVec3::ZERO,
                normal: PVec3::Z,
                u_dir: PVec3::X,
            },
            outer_wire: Default::default(),
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
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc.clone(), true);
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc, true);
        let wk = reg.wires.insert(BRepWire {
            edges: vec![(e1, Orientation::Forward), (e2, Orientation::Forward)],
        });
        // Wire: v0 -> v1 -> v2 -- not closed (v2 != v0)
        let gap = check_wire_closed(wk, &reg);
        assert!(gap.is_some());
        assert!(gap.unwrap() > 0.1, "open wire should have measurable gap");
    }

    #[test]
    fn test_redundant_edge_detected() {
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
        // Wire with duplicate edge
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (ek, Orientation::Forward),
            (ek, Orientation::Forward), // duplicate
        ];
        let statuses = check_redundant_edge(wk, &reg);
        assert!(
            statuses.contains(&CheckStatus::RedundantEdge),
            "duplicate edge should be flagged"
        );
    }

    #[test]
    fn test_no_redundant_edge_on_valid_wire() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let v2 = reg.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), 1e-4);
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
        let e1 = reg.add_edge_with_pcurve(v0, v1, curve.clone(), 1e-4, fk, pc.clone(), true);
        let e2 = reg.add_edge_with_pcurve(v1, v2, curve.clone(), 1e-4, fk, pc, true);
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
        ];
        let statuses = check_redundant_edge(wk, &reg);
        assert!(
            statuses.is_empty(),
            "wire with unique edges should have no redundant flag"
        );
    }
}
