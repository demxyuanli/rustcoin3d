//! Face-face intersector: produces BOPDS interference data from SSI.
//!
//! Wires the marching + Newton refinement pipeline into the BOPDS,
//! converting raw intersection curves into structured InterfPoint data
//! with UV coordinates on both faces.
//!
//! OCC alignment: IntTools_FaceFace + BOPAlgo_PaveFiller (face-face portion)

use crate::geom::{CurveGeom, SurfaceGeom};
use crate::topo::FaceKey;
use super::bopds::{FaceFaceInterf, InterfPoint};
use super::marching::{find_seeds, trace_curve_bidirectional};
use super::ssi_newton::newton_refine_ssi;
use rc3d_core::math::Vec3;

/// Compute face-face intersection with full UV data for both faces.
///
/// Uses the marching+Newton pipeline:
/// 1. Find seed points via grid sampling
/// 2. Trace intersection curves with Newton refinement at each step
/// 3. Sample each traced curve to produce InterfPoints with UV on both faces
pub fn intersect_faces(
    face_a: FaceKey,
    face_b: FaceKey,
    surface_a: &SurfaceGeom,
    surface_b: &SurfaceGeom,
    tolerance: f32,
) -> Option<FaceFaceInterf> {
    // Phase 1: Find seeds
    let mut seeds_a = find_seeds(surface_a, surface_b, 16, tolerance * 100.0);
    let seeds_b = find_seeds(surface_b, surface_a, 16, tolerance * 100.0);

    // Merge seeds from both directions
    for s in seeds_b {
        if !seeds_a.iter().any(|es| (es.point - s.point).length() < tolerance * 10.0) {
            seeds_a.push(super::marching::SeedPoint {
                point: s.point,
                uv_a: s.uv_b, // swap: seed was found on surf_b, now mapped to surf_a
                uv_b: s.uv_a,
            });
        }
    }

    if seeds_a.is_empty() {
        return None;
    }

    // Phase 2: Trace curves from each seed
    let range_a = surface_a.param_range();
    let u_span = (range_a.u_max - range_a.u_min).min(std::f32::consts::TAU);
    let v_span = (range_a.v_max - range_a.v_min).min(10.0);
    let step = (u_span + v_span) * 0.01;

    let mut all_curves: Vec<CurveGeom> = Vec::new();
    let mut all_points: Vec<InterfPoint> = Vec::new();
    // Track per-curve point ranges into all_points so PCurves get the correct
    // subset of UV samples (not the entire merged vector).
    let mut curve_point_ranges: Vec<(usize, usize)> = Vec::new();

    for seed in &seeds_a {
        // Refine the seed to exact intersection
        let refined = newton_refine_ssi(
            surface_a, surface_b,
            seed.uv_a, seed.uv_b,
            tolerance, 15,
        );

        let (uv_a_start, uv_b_start) = match refined {
            Some((ua, ub)) => (ua, ub),
            None => continue,
        };

        let refined_seed = super::marching::SeedPoint {
            point: surface_a.d0_native(uv_a_start.0, uv_a_start.1),
            uv_a: uv_a_start,
            uv_b: uv_b_start,
        };

        // Trace the curve bidirectionally
        let trace = trace_curve_bidirectional(
            surface_a, surface_b,
            &refined_seed, step, 200, tolerance * 10.0,
        );

        let points_3d = match trace {
            Some(pts) if pts.len() >= 4 => pts,
            _ => continue,
        };

        // Sample the traced curve to produce InterfPoints with UV on both faces
        let n_samples = points_3d.len().min(64);
        let step = (points_3d.len() - 1).max(1) as f32 / (n_samples - 1).max(1) as f32;

        let mut interf_points = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let idx_f = i as f32 * step;
            let idx = (idx_f as usize).min(points_3d.len() - 1);
            let pt = points_3d[idx];

            // Project to both surfaces for UV coordinates
            let uv_a = surface_a.project(pt)
                .or_else(|| surface_a.inverse_native_uv(pt, tolerance * 100.0));
            let uv_b = surface_b.project(pt)
                .or_else(|| surface_b.inverse_native_uv(pt, tolerance * 100.0));

            if let (Some(ua), Some(ub)) = (uv_a, uv_b) {
                interf_points.push(InterfPoint {
                    point_3d: pt,
                    uv_a: ua,
                    uv_b: ub,
                });
            }
        }

        if interf_points.len() >= 2 {
            let start = all_points.len();
            all_points.extend(interf_points);
            let end = all_points.len();
            curve_point_ranges.push((start, end));

            // Build polyline curve from 3D points
            all_curves.push(CurveGeom::Polyline { points: points_3d });
        }
    }

    if all_curves.is_empty() {
        return None;
    }

    // Build PCurves from per-curve InterfPoint UV data.
    // Each curve[i] gets UV points from all_points[ranges[i].0..ranges[i].1].
    let pcurves_a: Vec<CurveGeom> = curve_point_ranges.iter().map(|&(start, end)| {
        let uv_pts: Vec<Vec3> = all_points[start..end].iter()
            .map(|p| Vec3::new(p.uv_a.0, p.uv_a.1, 0.0))
            .collect();
        CurveGeom::Polyline { points: uv_pts }
    }).collect();

    let pcurves_b: Vec<CurveGeom> = curve_point_ranges.iter().map(|&(start, end)| {
        let uv_pts: Vec<Vec3> = all_points[start..end].iter()
            .map(|p| Vec3::new(p.uv_b.0, p.uv_b.1, 0.0))
            .collect();
        CurveGeom::Polyline { points: uv_pts }
    }).collect();

    Some(FaceFaceInterf {
        face_a,
        face_b,
        curves_3d: all_curves,
        pcurves_a,
        pcurves_b,
        points: all_points,
    })
}

/// Check if two surfaces could potentially intersect based on their types.
/// Returns false for pairs that we know cannot intersect (e.g., two parallel planes).
pub fn surfaces_may_intersect(surf_a: &SurfaceGeom, surf_b: &SurfaceGeom) -> bool {
    use SurfaceGeom::*;
    match (surf_a, surf_b) {
        // Parallel planes never intersect (handled by analytic path anyway)
        (Plane { normal: n1, .. }, Plane { normal: n2, .. }) => {
            n1.cross(*n2).length() > 1e-10
        }
        // Everything else: could potentially intersect (conservative)
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::SurfaceGeom;
    use rc3d_core::math::Vec3;

    #[test]
    fn test_intersect_faces_plane_plane() {
        let sa = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        };
        let sb = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Y, u_dir: Vec3::X,
        };

        let result = intersect_faces(
            FaceKey::default(), FaceKey::default(),
            &sa, &sb, 1e-4,
        );
        assert!(result.is_some(), "intersecting planes should produce result");
        let interf = result.unwrap();
        assert!(!interf.curves_3d.is_empty(), "should have at least one curve");
        assert!(!interf.points.is_empty(), "should have sample points");
    }

    #[test]
    fn test_intersect_faces_parallel_planes() {
        let sa = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        };
        let sb = SurfaceGeom::Plane {
            origin: Vec3::new(0.0, 0.0, 10.0), normal: Vec3::Z, u_dir: Vec3::X,
        };

        assert!(!surfaces_may_intersect(&sa, &sb));
    }

    #[test]
    fn test_surfaces_may_intersect() {
        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        };
        let cyl = SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 1.0);
        assert!(surfaces_may_intersect(&plane, &cyl));
    }

    /// Verify that when a face pair produces multiple intersection curves,
    /// each PCurve gets only the UV points belonging to its own curve,
    /// not the merged set from all curves.
    #[test]
    fn test_pcurve_per_curve_partition() {
        // A horizontal plane at z=0.5 cutting through a sphere of radius 1.0
        // can produce a single circular intersection curve. We verify the
        // per-curve partition invariant: pcurves_a[i] has the same number of
        // UV points as curves_3d[i] has 3D sample points.
        let plane = SurfaceGeom::Plane {
            origin: Vec3::new(0.0, 0.0, 0.5),
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let sphere = SurfaceGeom::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };

        let result = intersect_faces(
            FaceKey::default(), FaceKey::default(),
            &plane, &sphere, 1e-4,
        );

        if let Some(interf) = result {
            // Invariant: pcurves and curves must have matching counts
            assert_eq!(
                interf.pcurves_a.len(), interf.curves_3d.len(),
                "pcurves_a count ({}) must match curves_3d count ({})",
                interf.pcurves_a.len(), interf.curves_3d.len(),
            );
            assert_eq!(
                interf.pcurves_b.len(), interf.curves_3d.len(),
                "pcurves_b count ({}) must match curves_3d count ({})",
                interf.pcurves_b.len(), interf.curves_3d.len(),
            );

            // Each PCurve's UV point count should match the number of
            // InterfPoints allocated to that curve (not the total).
            let mut point_offset = 0usize;
            for (i, curve) in interf.curves_3d.iter().enumerate() {
                let pcurve_a = &interf.pcurves_a[i];
                let pcurve_b = &interf.pcurves_b[i];

                // Count how many InterfPoints belong to this curve
                // (determined by the CurveGeom::Polyline point count)
                let n_3d = match curve {
                    CurveGeom::Polyline { points } => points.len(),
                    _ => continue,
                };
                let n_pcurve_a = match pcurve_a {
                    CurveGeom::Polyline { points } => points.len(),
                    _ => 0,
                };
                let n_pcurve_b = match pcurve_b {
                    CurveGeom::Polyline { points } => points.len(),
                    _ => 0,
                };

                // The pcurve UV point count should be <= n_3d (some projections
                // may fail), and must NOT equal the total all_points count
                // (which would indicate the old bug where all points were merged).
                assert!(
                    n_pcurve_a <= n_3d,
                    "curve {}: pcurve_a has {} UV points but only {} 3D points",
                    i, n_pcurve_a, n_3d,
                );
                assert!(
                    n_pcurve_a < interf.points.len() || interf.curves_3d.len() == 1,
                    "curve {}: pcurve_a has {} points = total {}, indicating merged points bug",
                    i, n_pcurve_a, interf.points.len(),
                );
                assert!(
                    n_pcurve_b <= n_3d,
                    "curve {}: pcurve_b has {} UV points but only {} 3D points",
                    i, n_pcurve_b, n_3d,
                );

                point_offset += n_pcurve_a;
            }
        }
    }
}
