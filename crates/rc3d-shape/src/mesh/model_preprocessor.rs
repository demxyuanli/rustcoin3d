//! Wire-level preprocessor for B-Rep meshing (OCC BRepMesh_ModelPreProcessor).
//!
//! Analyzes face UV loops for self-intersection and open-wire conditions
//! before Delaunay triangulation. Flagging bad loops prevents CDT failures.
//!
//! ## Algorithm
//! 1. For each face: collect the outer wire + inner wires as UV segments
//! 2. Check for self-intersection: if any non-adjacent UV segments cross
//! 3. Check for open wire: if start ≠ end vertex in any loop
//! 4. Generate a preprocessor report with flagged face keys

use std::collections::HashSet;

use crate::mesh_result::MeshResult;
use crate::store::BRepStore;
use crate::topo::{FaceKey, Orientation, ShellKey, WireKey};
use rc3d_core::math::{Real, PVec3};

/// Status flags per face after preprocessing.
#[derive(Debug, Clone, Default)]
pub struct FaceWireStatus {
    /// At least one wire (outer or inner) is self-intersecting.
    pub self_intersecting: bool,
    /// At least one wire is open (start ≠ end).
    pub open_wire: bool,
    /// Total number of wires analyzed for this face.
    pub wire_count: usize,
}

/// Result of the model preprocessing phase.
#[derive(Debug, Clone, Default)]
pub struct PreprocessorReport {
    /// Per-face wire status (populated for every face in the shell).
    pub face_status: Vec<(FaceKey, FaceWireStatus)>,
    /// Faces flagged as self-intersecting.
    pub self_intersecting_faces: Vec<FaceKey>,
    /// Faces flagged with open wires.
    pub open_wire_faces: Vec<FaceKey>,
    /// Total faces processed.
    pub total_faces: usize,
}

/// Run the model preprocessor on a shell before meshing.
///
/// Returns a report with per-face wire status. Callers should skip or
/// simplify faces flagged as self-intersecting / open-wire.
pub fn preprocess_shell_wires(
    shell_key: ShellKey,
    reg: &BRepStore,
) -> PreprocessorReport {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return PreprocessorReport::default(),
    };

    let mut report = PreprocessorReport::default();
    report.total_faces = shell.faces.len();

    for &(fk, _orient) in &shell.faces {
        let status = analyze_face_wires(fk, reg);
        if status.self_intersecting {
            report.self_intersecting_faces.push(fk);
        }
        if status.open_wire {
            report.open_wire_faces.push(fk);
        }
        report.face_status.push((fk, status));
    }

    if !report.self_intersecting_faces.is_empty() {
        log::warn!(
            "[mesh preprocessor] {} self-intersecting wire(s) detected: {:?}",
            report.self_intersecting_faces.len(),
            report.self_intersecting_faces.iter().take(5).collect::<Vec<_>>(),
        );
    }
    if !report.open_wire_faces.is_empty() {
        log::warn!(
            "[mesh preprocessor] {} open wire(s) detected: {:?}",
            report.open_wire_faces.len(),
            report.open_wire_faces.iter().take(5).collect::<Vec<_>>(),
        );
    }

    report
}

/// Analyze all wires of a face for self-intersection and openness.
fn analyze_face_wires(fk: FaceKey, reg: &BRepStore) -> FaceWireStatus {
    let face = match reg.faces.get(fk) {
        Some(f) => f,
        None => return FaceWireStatus { open_wire: true, ..Default::default() },
    };

    let all_wires: Vec<WireKey> = std::iter::once(face.outer_wire)
        .chain(face.inner_wires.iter().copied())
        .collect();
    let wire_count = all_wires.len();

    let mut status = FaceWireStatus {
        wire_count,
        ..Default::default()
    };

    for wk in all_wires {
        let segments = collect_wire_segments_uv(wk, fk, reg);
        if segments.is_empty() {
            // Wire with no valid edges → treat as open
            status.open_wire = true;
            continue;
        }
        // Check closure: last endpoint must connect to first
        let first_start = (segments[0].0, segments[0].1);
        let last_end = segments.last().map(|s| (s.2, s.3)).unwrap_or(first_start);
        if (last_end.0 - first_start.0).abs() > 1e-10
            || (last_end.1 - first_start.1).abs() > 1e-10
        {
            status.open_wire = true;
        }
        // Check self-intersection: any non-adjacent segment pair intersects
        if wire_has_self_intersection(&segments) {
            status.self_intersecting = true;
        }
    }

    status
}

/// A UV line segment: (x0, y0) to (x1, y1), with the edge key for adjacency check.
type UvSegment = (Real, Real, Real, Real);

/// Collect all UV line segments from a wire into a flat Vec.
fn collect_wire_segments_uv(
    wk: WireKey,
    fk: FaceKey,
    reg: &BRepStore,
) -> Vec<UvSegment> {
    let wire = match reg.wires.get(wk) {
        Some(w) => w,
        None => return vec![],
    };

    let mut segments = Vec::with_capacity(wire.edges.len());
    for &(ek, orient) in &wire.edges {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };
        let pcurve = match edge.pcurves.get(&fk) {
            Some(pc) => pc,
            None => continue,
        };
        let (start, end) = if orient == Orientation::Forward {
            (pcurve.d0(0.0), pcurve.d0(1.0))
        } else {
            (pcurve.d0(1.0), pcurve.d0(0.0))
        };
        segments.push((start.0, start.1, end.0, end.1));
    }
    segments
}

/// Check if a wire has self-intersecting segments.
/// Non-adjacent segments (not sharing a vertex) that cross indicate self-intersection.
fn wire_has_self_intersection(segments: &[UvSegment]) -> bool {
    let n = segments.len();
    if n < 3 {
        return false; // triangle wire cannot self-intersect
    }
    for i in 0..n {
        for j in (i + 2)..n {
            // Skip adjacent segments (wrapping: first and last are adjacent)
            if j == i + 1 {
                continue;
            }
            if i == 0 && j == n - 1 {
                continue; // adjacent through wire closure
            }
            if segments_intersect_2d(segments[i], segments[j]) {
                return true;
            }
        }
    }
    false
}

/// Check if two 2D segments a(p0-p1) and b(q0-q1) intersect.
fn segments_intersect_2d(a: UvSegment, b: UvSegment) -> bool {
    let o1 = orient_2d(a.0, a.1, a.2, a.3, b.0, b.1);
    let o2 = orient_2d(a.0, a.1, a.2, a.3, b.2, b.3);
    let o3 = orient_2d(b.0, b.1, b.2, b.3, a.0, a.1);
    let o4 = orient_2d(b.0, b.1, b.2, b.3, a.2, a.3);

    // General case: segments straddle each other
    if o1 * o2 < 0.0 && o3 * o4 < 0.0 {
        return true;
    }
    false
}

/// 2D orientation: cross product of (b-a) × (c-a).
/// Positive → counter-clockwise, negative → clockwise, zero → collinear.
fn orient_2d(ax: Real, ay: Real, bx: Real, by: Real, cx: Real, cy: Real) -> Real {
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::BRepStore;
    use crate::topo::*;
    use crate::geom::{Curve2d, CurveGeom, SurfaceGeom};
    use rc3d_core::math::PVec3;

    fn make_face_with_square_wire(reg: &mut BRepStore) -> (FaceKey, WireKey) {
        let surf = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface: surf,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-6,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        // Build a simple square wire with 4 edges
        let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        for i in 0..4 {
            let j = (i + 1) % 4;
            let v0 = reg.find_or_add_vertex(
                PVec3::new(corners[i].0, corners[i].1, 0.0), 1e-6);
            let v1 = reg.find_or_add_vertex(
                PVec3::new(corners[j].0, corners[j].1, 0.0), 1e-6);
            let pc = Curve2d::Line {
                origin: (corners[i].0, corners[i].1),
                direction: (corners[j].0 - corners[i].0, corners[j].1 - corners[i].1),
            };
            let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-6, fk, pc, true);
            reg.wires.get_mut(wk).unwrap().edges.push((ek, Orientation::Forward));
        }
        (fk, wk)
    }

    #[test]
    fn test_square_wire_no_self_intersection() {
        let mut reg = BRepStore::new();
        let (fk, _wk) = make_face_with_square_wire(&mut reg);
        let sk = reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let report = preprocess_shell_wires(sk, &reg);
        assert_eq!(report.total_faces, 1);
        assert!(report.self_intersecting_faces.is_empty());
        assert!(report.open_wire_faces.is_empty());
    }

    #[test]
    fn test_segments_intersect_crossing() {
        // Crossing segments
        assert!(segments_intersect_2d((0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 0.0)));
    }

    #[test]
    fn test_segments_intersect_parallel() {
        // Parallel segments — no intersection
        assert!(!segments_intersect_2d((0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 1.0, 1.0)));
    }

    #[test]
    fn test_empty_shell_no_faces() {
        let mut reg = BRepStore::new();
        let sk = reg.shells.insert(BRepShell {
            faces: vec![],
            closed: false,
            step_id: None,
        });
        let report = preprocess_shell_wires(sk, &reg);
        assert_eq!(report.total_faces, 0);
    }
}
