//! Face-edge and edge-edge intersection for the boolean pipeline.
//!
//! Finds where 3D curves cross surfaces (edge-face) and where two curves
//! cross each other (edge-edge). Used in B-rep boolean operations to detect
//! vertices where edges from one body touch faces or edges of another body.

use rc3d_core::math::Vec3;
use crate::geom::{CurveGeom, SurfaceGeom, project::project_point_on_curve};
use crate::topo::{EdgeKey, FaceKey, WireKey};
use crate::store::BRepStore;

// ── Hit types ────────────────────────────────────────────────────

/// Hit point from an edge-face intersection.
#[derive(Debug, Clone)]
pub struct EdgeFaceHit {
    /// Parameter on edge curve [0,1].
    pub t_edge: f32,
    /// 3D intersection point.
    pub point: Vec3,
    /// UV on face surface (native parameters).
    pub uv_face: (f32, f32),
}

/// Hit point from an edge-edge intersection.
#[derive(Debug, Clone)]
pub struct EdgeEdgeHit {
    /// Parameter on edge A [0,1].
    pub t_a: f32,
    /// Parameter on edge B [0,1].
    pub t_b: f32,
    /// 3D intersection point.
    pub point: Vec3,
}

// ── Edge-Face intersection ───────────────────────────────────────

/// Find intersections between a 3D curve and a face surface.
///
/// Samples the curve adaptively, projects each sample onto the surface,
/// finds crossing points where the curve passes through the surface,
/// then bisect-refines each crossing.
pub fn intersect_edge_face(
    edge_curve: &CurveGeom,
    surface: &SurfaceGeom,
    tolerance: f32,
) -> Vec<EdgeFaceHit> {
    let samples = edge_curve.sample_adaptive(0.0, 1.0, tolerance);

    // Compute signed distance of each sample to the surface.
    // For planes this is exact (signed); for other surfaces we use
    // projection distance.
    let dists: Vec<f32> = samples.iter().map(|&(_, p)| {
        surface_distance(surface, p)
    }).collect();

    // Find zero-crossings and refine each via bisection.
    let mut hits: Vec<EdgeFaceHit> = Vec::new();
    for i in 0..samples.len().saturating_sub(1) {
        let da = dists[i];
        let db = dists[i + 1];

        // Need a sign change (or one endpoint on the surface).
        if da * db > 0.0 && da.abs() > tolerance && db.abs() > tolerance {
            continue;
        }

        // Bisect to find the crossing between samples[i] and samples[i+1].
        if let Some(hit) = bisect_edge_face(
            edge_curve, surface,
            (samples[i].0, samples[i].1),
            (samples[i + 1].0, samples[i + 1].1),
            tolerance,
        ) {
            // Deduplicate: skip if close to an existing hit in parameter space.
            let dup = hits.iter().any(|h| (h.t_edge - hit.t_edge).abs() < tolerance);
            if !dup {
                hits.push(hit);
            }
        }
    }

    hits
}

/// Signed distance from a point to a surface.
///
/// For planes: exact signed distance (positive on normal side).
/// For other surfaces: distance from point to its nearest projection on the
/// surface, signed by the surface normal (positive = outside, negative = inside).
fn surface_distance(surface: &SurfaceGeom, point: Vec3) -> f32 {
    match surface {
        SurfaceGeom::Plane { origin, normal, .. } => {
            (point - *origin).dot(*normal)
        }
        _ => {
            // Use `project` (always succeeds analytically) rather than
            // `inverse_native_uv` (rejects far points).
            let uv = match surface.project(point) {
                Some(uv) => uv,
                None => return 1e6,
            };
            let proj = surface.d0_native(uv.0, uv.1);
            let diff = point - proj;
            let dist = diff.length();
            let n = surface.normal_native(uv.0, uv.1);
            if diff.dot(n) >= 0.0 { dist } else { -dist }
        }
    }
}

/// Bisect between two curve samples to find where the curve crosses the surface.
fn bisect_edge_face(
    curve: &CurveGeom,
    surface: &SurfaceGeom,
    s0: (f32, Vec3),
    s1: (f32, Vec3),
    tolerance: f32,
) -> Option<EdgeFaceHit> {
    let (mut t0, _) = s0;
    let (mut t1, _) = s1;
    let mut d0 = surface_distance(surface, curve.d0(t0));
    let d1 = surface_distance(surface, curve.d0(t1));

    // Must have sign change.
    if d0 * d1 > 0.0 {
        return None;
    }

    // Bisection iterations.
    const MAX_ITER: usize = 32;
    for _ in 0..MAX_ITER {
        let tm = (t0 + t1) * 0.5;
        let pm = curve.d0(tm);
        let dm = surface_distance(surface, pm);

        if dm.abs() <= tolerance {
            let uv = surface.inverse_native_uv(pm, tolerance * 100.0)?;
            return Some(EdgeFaceHit { t_edge: tm, point: pm, uv_face: uv });
        }

        if d0 * dm < 0.0 {
            t1 = tm;
        } else {
            t0 = tm;
            d0 = dm;
        }
    }

    // Converged close enough.
    let tm = (t0 + t1) * 0.5;
    let pm = curve.d0(tm);
    let dm = surface_distance(surface, pm);
    if dm.abs() <= tolerance * 10.0 {
        let uv = surface.inverse_native_uv(pm, tolerance * 100.0)?;
        return Some(EdgeFaceHit { t_edge: tm, point: pm, uv_face: uv });
    }

    None
}

// ── Edge-Edge intersection ───────────────────────────────────────

/// Find intersections between two 3D edge curves.
///
/// Uses adaptive sampling of curve A, then projects each sample onto curve B.
pub fn intersect_edge_edge(
    edge_a: &CurveGeom,
    edge_b: &CurveGeom,
    tolerance: f32,
) -> Vec<EdgeEdgeHit> {
    let samples_a = edge_a.sample_adaptive(0.0, 1.0, tolerance);
    let tol_sq = tolerance * tolerance;

    let mut hits: Vec<EdgeEdgeHit> = Vec::new();

    for &(t_a, p_a) in &samples_a {
        let projections = project_point_on_curve(edge_b, p_a);

        for (t_b, dist_sq) in projections {
            if dist_sq > tol_sq {
                continue;
            }
            if !(-1e-6..=1.0 + 1e-6).contains(&t_b) {
                continue;
            }
            let t_b_clamped = t_b.clamp(0.0, 1.0);
            let p_b = edge_b.d0(t_b_clamped);
            let midpoint = (p_a + p_b) * 0.5;

            // Deduplicate: skip if too close to an existing hit in t_a.
            let too_close = hits.iter().any(|h| (h.t_a - t_a).abs() < tolerance);
            if too_close {
                continue;
            }

            hits.push(EdgeEdgeHit {
                t_a,
                t_b: t_b_clamped,
                point: midpoint,
            });
        }
    }

    hits
}

// ── Wire-level driver ────────────────────────────────────────────

/// Find all edge-face intersections for edges in a wire against a face.
pub fn intersect_wire_face(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &BRepStore,
    tolerance: f32,
) -> Vec<(EdgeKey, Vec<EdgeFaceHit>)> {
    let wire = match reg.wires.get(wire_key) {
        Some(w) => w,
        None => return vec![],
    };
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return vec![],
    };

    let mut results = Vec::new();
    for &(ek, _orient) in &wire.edges {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };
        let hits = intersect_edge_face(&edge.curve, &face.surface, tolerance);
        if !hits.is_empty() {
            results.push((ek, hits));
        }
    }
    results
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepWire};
    use rc3d_core::math::Vec3;

    /// Helper: create a BRepFace with a dummy wire in a fresh BRepStore.
    fn make_face_with_store(surface: SurfaceGeom) -> (BRepStore, crate::topo::FaceKey) {
        let mut reg = BRepStore::new();
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
        (reg, fk)
    }

    // ── Edge-Face tests ──────────────────────────────────────────

    #[test]
    fn test_intersect_line_through_plane() {
        // Line along Z from (-1) to (+1), passing through the z=0 plane.
        let line = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, -1.0),
            direction: Vec3::new(0.0, 0.0, 2.0), // t=0 → z=-1, t=1 → z=+1
        };
        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let hits = intersect_edge_face(&line, &plane, 1e-4);
        assert_eq!(hits.len(), 1, "Line through plane should produce 1 hit");
        let h = &hits[0];
        assert!((h.t_edge - 0.5).abs() < 1e-3, "Hit should be near t=0.5, got {}", h.t_edge);
        assert!(h.point.z.abs() < 1e-3, "Hit point should be near z=0, got {}", h.point.z);
    }

    #[test]
    fn test_intersect_line_miss_plane() {
        // Line parallel to and above the z=0 plane (at z=1).
        let line = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 1.0),
            direction: Vec3::X, // travels along X, stays at z=1
        };
        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let hits = intersect_edge_face(&line, &plane, 1e-4);
        assert!(hits.is_empty(), "Line above plane should produce 0 hits, got {}", hits.len());
    }

    #[test]
    fn test_intersect_line_through_cylinder() {
        // Cylinder along Z axis, radius 1.0, centered at origin.
        let cylinder = SurfaceGeom::Cylinder {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 1.0,
            x_dir: Vec3::X,
            y_dir: Vec3::Y,
        };
        // Line along X from (-2, 0, 0) to (+2, 0, 0) — passes through at x=-1 and x=+1.
        let line = CurveGeom::Line {
            origin: Vec3::new(-2.0, 0.0, 0.0),
            direction: Vec3::new(4.0, 0.0, 0.0),
        };
        let hits = intersect_edge_face(&line, &cylinder, 1e-3);
        assert_eq!(hits.len(), 2,
            "Line through cylinder should produce 2 hits (entry + exit), got {}", hits.len());

        // First hit near x=-1 (t ~ 0.25), second near x=+1 (t ~ 0.75).
        let t0 = hits[0].t_edge.min(hits[1].t_edge);
        let t1 = hits[0].t_edge.max(hits[1].t_edge);
        assert!((t0 - 0.25).abs() < 0.05, "First hit t should be ~0.25, got {t0}");
        assert!((t1 - 0.75).abs() < 0.05, "Second hit t should be ~0.75, got {t1}");
    }

    // ── Edge-Edge tests ──────────────────────────────────────────

    #[test]
    fn test_intersect_two_crossing_lines() {
        // Line A: along X from (-1, 0, 0) to (1, 0, 0).
        let line_a = CurveGeom::Line {
            origin: Vec3::new(-1.0, 0.0, 0.0),
            direction: Vec3::new(2.0, 0.0, 0.0),
        };
        // Line B: along Y from (0, -1, 0) to (0, 1, 0).
        let line_b = CurveGeom::Line {
            origin: Vec3::new(0.0, -1.0, 0.0),
            direction: Vec3::new(0.0, 2.0, 0.0),
        };
        let hits = intersect_edge_edge(&line_a, &line_b, 1e-3);
        assert_eq!(hits.len(), 1, "Crossing lines should produce 1 hit, got {}", hits.len());
        let h = &hits[0];
        assert!(h.point.x.abs() < 0.01, "Hit x should be ~0, got {}", h.point.x);
        assert!(h.point.y.abs() < 0.01, "Hit y should be ~0, got {}", h.point.y);
    }

    #[test]
    fn test_intersect_parallel_lines() {
        // Two parallel lines along X, offset in Y.
        let line_a = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
        };
        let line_b = CurveGeom::Line {
            origin: Vec3::new(0.0, 1.0, 0.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
        };
        let hits = intersect_edge_edge(&line_a, &line_b, 1e-3);
        assert!(hits.is_empty(), "Parallel lines should produce 0 hits, got {}", hits.len());
    }
}
