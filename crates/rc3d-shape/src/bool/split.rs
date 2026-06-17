//! Face splitting along intersection curves (B-Rep native).

use rc3d_core::math::{Real, PVec3};
use crate::store::BRepStore;
use crate::topo::{FaceKey, ShellKey, BRepFace};

/// A curve where two faces intersect, parameterized on both surfaces.
#[derive(Debug, Clone)]
pub struct BRepIntersectionCurve {
    pub points_3d: Vec<PVec3>,
    pub params_a: Vec<(Real, Real)>,
    pub params_b: Vec<(Real, Real)>,
    pub face_a: FaceKey,
    pub face_b: FaceKey,
}

#[derive(Debug, Clone)]
pub struct SplitFaceRegion {
    pub original_face: FaceKey,
    pub sub_faces: Vec<SubFaceRegion>,
}

#[derive(Debug, Clone)]
pub struct SubFaceRegion {
    pub uv_boundary: Vec<Vec<(Real, Real)>>,
    pub interior_point: (Real, Real),
    pub interior_point_3d: PVec3,
    pub original_face: FaceKey,
}

/// Compute B-Rep intersection curves from FaceIntersectionResult.
///
/// Uses pre-computed PCurves from the intersection when available
/// (face_intersector pathway), falling back to surface re-projection
/// for the legacy compute_intersections_brep pathway.
pub fn compute_brep_intersection_curves(
    intersections: &[super::intersect::FaceIntersectionResult],
    reg: &BRepStore,
) -> Vec<BRepIntersectionCurve> {
    let mut out = Vec::new();
    // Reusable buffers to avoid per-curve heap allocation in the hot loop.
    let mut pts_3d = Vec::new();
    let mut params_a = Vec::new();
    let mut params_b = Vec::new();

    for fi in intersections {
        let face_a = match reg.faces.get(fi.face_a) { Some(f) => f, None => continue };
        let face_b = match reg.faces.get(fi.face_b) { Some(f) => f, None => continue };

        for (i, curve) in fi.curves_3d.iter().enumerate() {
            pts_3d.clear();
            params_a.clear();
            params_b.clear();

            // Try pre-computed PCurves first (from face_intersector pathway)
            let pcurve_a = fi.pcurves_on_a.get(i);
            let pcurve_b = fi.pcurves_on_b.get(i);

            if let (Some(pc_a), Some(pc_b)) = (pcurve_a, pcurve_b) {
                // Extract UV point pairs from the pre-computed PCurves
                // PCurves are polylines in UV space: d0(t) returns PVec3(u, v, 0)
                let n_samples = 64;
                pts_3d.reserve(n_samples);
                params_a.reserve(n_samples);
                params_b.reserve(n_samples);
                for j in 0..n_samples {
                    let t = j as Real / (n_samples - 1).max(1) as Real;
                    let uva = pc_a.d0(t);
                    let uvb = pc_b.d0(t);
                    pts_3d.push(curve.d0(t));
                    params_a.push((uva.x, uva.y));
                    params_b.push((uvb.x, uvb.y));
                }
                if pts_3d.len() >= 2 {
                    out.push(BRepIntersectionCurve {
                        points_3d: std::mem::take(&mut pts_3d),
                        params_a: std::mem::take(&mut params_a),
                        params_b: std::mem::take(&mut params_b),
                        face_a: fi.face_a, face_b: fi.face_b,
                    });
                    continue;
                }
            }

            // Fallback: re-sample 3D curve and project to both surfaces
            let samples = sample_intersection_curve(curve, 32);
            if samples.is_empty() { continue; }

            pts_3d.reserve(samples.len());
            params_a.reserve(samples.len());
            params_b.reserve(samples.len());

            for &pt in &samples {
                if let Some(uv_a) = face_a.surface.project(pt) {
                    if let Some(uv_b) = face_b.surface.project(pt) {
                        pts_3d.push(pt);
                        params_a.push(uv_a);
                        params_b.push(uv_b);
                    }
                }
            }

            if pts_3d.len() >= 2 {
                out.push(BRepIntersectionCurve {
                    points_3d: std::mem::take(&mut pts_3d),
                    params_a: std::mem::take(&mut params_a),
                    params_b: std::mem::take(&mut params_b),
                    face_a: fi.face_a, face_b: fi.face_b,
                });
            }
        }
    }
    out
}

fn sample_intersection_curve(curve: &crate::geom::CurveGeom, n: usize) -> Vec<PVec3> {
    // All CurveGeom variants are parameterized over t in [0, 1].
    // Use the universal d0(t) evaluator to support every curve type.
    (0..n).map(|i| {
        let t = i as Real / (n - 1).max(1) as Real;
        curve.d0(t)
    }).collect()
}

/// Build SplitFaceRegions from BopDS intersection data (PaveBlock-driven).
///
/// OCC: BOPAlgo_BuilderFace region construction — uses InterfPoint UVs
/// from face-face intersections as interior points for sub-regions,
/// instead of UV geometric marching.
pub fn split_faces_from_bopds(
    shells: &[ShellKey],
    bopds: &super::bopds::BopDS,
    reg: &BRepStore,
) -> Vec<SplitFaceRegion> {
    let mut results = Vec::new();
    for &sk in shells {
        let shell = match reg.shells.get(sk) { Some(s) => s, None => continue };
        for &(face_key, _) in &shell.faces {
            let face = match reg.faces.get(face_key) { Some(f) => f, None => continue };

            // Collect all InterfPoint UVs on this face from BOPDS
            let interfs = bopds.interfs_for_face(face_key);
            let mut uv_points: Vec<(Real, Real)> = Vec::new();
            let mut uv_boundary_hint: Vec<Vec<(Real, Real)>> = Vec::new();

            for interf in &interfs {
                let is_face_a = interf.face_a == face_key;
                // Build UV boundary from intersection points on this face
                let pts_on_face: Vec<(Real, Real)> = interf.points.iter().map(|p| {
                    if is_face_a { p.uv_a } else { p.uv_b }
                }).collect();
                if pts_on_face.len() >= 2 {
                    uv_boundary_hint.push(pts_on_face.clone());
                }
                uv_points.extend(pts_on_face);
            }

            if uv_points.is_empty() {
                // No intersection on this face — whole face region
                results.push(SplitFaceRegion {
                    original_face: face_key,
                    sub_faces: vec![whole_face_region(face_key, face)],
                });
                continue;
            }

            // Use intersection UV boundary as the sub-region descriptor
            let sub_faces = if uv_boundary_hint.is_empty() {
                vec![whole_face_region(face_key, face)]
            } else {
                uv_boundary_hint.iter().map(|boundary| {
                    let interior = boundary.get(boundary.len() / 2).copied().unwrap_or((0.0, 0.0));
                    let (un, vn) = face.surface.native_uv_to_d0(interior.0, interior.1);
                    SubFaceRegion {
                        uv_boundary: vec![boundary.clone()],
                        interior_point: interior,
                        interior_point_3d: face.surface.d0(un, vn),
                        original_face: face_key,
                    }
                }).collect()
            };

            results.push(SplitFaceRegion {
                original_face: face_key,
                sub_faces,
            });
        }
    }
    results
}

pub fn split_all_faces_brep(
    shells: &[ShellKey],
    curves: &[BRepIntersectionCurve],
    reg: &BRepStore,
) -> Vec<SplitFaceRegion> {
    let mut results = Vec::new();
    for &sk in shells {
        let shell = match reg.shells.get(sk) { Some(s) => s, None => continue };
        for &(face_key, _) in &shell.faces {
            let face_curves: Vec<&BRepIntersectionCurve> = curves.iter()
                .filter(|c| c.face_a == face_key || c.face_b == face_key)
                .collect();
            let regions = split_face_along_curves(face_key, &face_curves, reg);
            results.push(SplitFaceRegion { original_face: face_key, sub_faces: regions });
        }
    }
    results
}

pub fn split_face_along_curves(
    face_key: FaceKey,
    curves: &[&BRepIntersectionCurve],
    reg: &BRepStore,
) -> Vec<SubFaceRegion> {
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return vec![],
    };
    if curves.is_empty() {
        return vec![whole_face_region(face_key, face)];
    }
    // Start with the whole face, then each curve further subdivides
    let mut regions = vec![whole_face_region(face_key, face)];
    for curve in curves {
        let mut next = Vec::new();
        for region in &regions {
            let sub = split_face_single_curve(face_key, face, curve, reg);
            if sub.len() > 1 {
                next.extend(sub);
            } else {
                next.push(region.clone());
            }
        }
        regions = next;
    }
    regions
}

fn whole_face_region(face_key: FaceKey, face: &BRepFace) -> SubFaceRegion {
    let range = face.surface.param_range();
    let mid_u = (range.u_min + range.u_max) * 0.5;
    let mid_v = (range.v_min + range.v_max) * 0.5;
    let (un, vn) = face.surface.native_uv_to_d0(mid_u, mid_v);
    let interior_3d = face.surface.d0(un, vn);
    SubFaceRegion {
        uv_boundary: vec![],
        interior_point: (mid_u, mid_v),
        interior_point_3d: interior_3d,
        original_face: face_key,
    }
}

fn split_face_single_curve(
    face_key: FaceKey,
    face: &BRepFace,
    curve: &BRepIntersectionCurve,
    _reg: &BRepStore,
) -> Vec<SubFaceRegion> {
    let params = if curve.face_a == face_key { &curve.params_a } else { &curve.params_b };
    if params.len() < 2 {
        return vec![whole_face_region(face_key, face)];
    }

    let mid_idx = params.len() / 2;
    let (cu, cv) = params[mid_idx];

    let tangent = if mid_idx > 0 && mid_idx + 1 < params.len() {
        let (u0, v0) = params[mid_idx - 1];
        let (u1, v1) = params[mid_idx + 1];
        (u1 - u0, v1 - v0)
    } else if mid_idx > 0 {
        let (u0, v0) = params[mid_idx - 1];
        (cu - u0, cv - v0)
    } else {
        let (u1, v1) = params[1];
        (u1 - cu, v1 - cv)
    };

    let t_len = (tangent.0 * tangent.0 + tangent.1 * tangent.1).sqrt();
    if t_len < 1e-12 {
        return vec![whole_face_region(face_key, face)];
    }

    let perp = (-tangent.1 / t_len, tangent.0 / t_len);
    let offset = 0.01;
    let range = face.surface.param_range();
    let du = perp.0 * offset * range.u_span();
    let dv = perp.1 * offset * range.v_span();
    let side_a_uv = (cu + du, cv + dv);
    let side_b_uv = (cu - du, cv - dv);

    let (un_a, vn_a) = face.surface.native_uv_to_d0(side_a_uv.0, side_a_uv.1);
    let (un_b, vn_b) = face.surface.native_uv_to_d0(side_b_uv.0, side_b_uv.1);

    vec![
        SubFaceRegion {
            uv_boundary: vec![params.to_vec()],
            interior_point: side_a_uv,
            interior_point_3d: face.surface.d0(un_a, vn_a),
            original_face: face_key,
        },
        SubFaceRegion {
            uv_boundary: vec![params.iter().rev().copied().collect()],
            interior_point: side_b_uv,
            interior_point_3d: face.surface.d0(un_b, vn_b),
            original_face: face_key,
        },
    ]
}

/// Create a new B-Rep face from a UV boundary region of an existing face (P3 sub-face creation).
/// Adds vertices, edges, and a wire. The new face shares the original surface geometry.
pub fn create_sub_face(
    original_face: FaceKey,
    uv_boundary: &[(Real, Real)],
    reg: &mut BRepStore,
) -> Option<FaceKey> {
    let original = reg.faces.get(original_face)?;
    if uv_boundary.len() < 3 { return None; }

    let mut edges = Vec::new();
    for i in 0..uv_boundary.len() {
        let p0 = uv_boundary[i];
        let p1 = uv_boundary[(i + 1) % uv_boundary.len()];
        let (un0, vn0) = original.surface.native_uv_to_d0(p0.0, p0.1);
        let (un1, vn1) = original.surface.native_uv_to_d0(p1.0, p1.1);
        let start = original.surface.d0(un0, vn0);
        let end = original.surface.d0(un1, vn1);
        let vk0 = reg.vertices.insert(crate::topo::BRepVertex { position: start, tolerance: original.tolerance });
        let vk1 = reg.vertices.insert(crate::topo::BRepVertex { position: end, tolerance: original.tolerance });
        let ek = reg.edges.insert(crate::topo::BRepEdge {
            curve: crate::geom::CurveGeom::Line { origin: start, direction: (end - start).normalize() },
            tolerance: original.tolerance,
            v_low: vk0,
            v_high: vk1,
            t_min: 0.0,
            t_max: 1.0,
            cached_deflection: None,
            pcurves: std::collections::HashMap::new(),
        });
        edges.push((ek, crate::topo::Orientation::Forward));
    }
    let wire = reg.wires.insert(crate::topo::BRepWire { edges });
    Some(reg.faces.insert(crate::topo::BRepFace {
        surface: original.surface.clone(),
        outer_wire: wire,
        inner_wires: vec![],
        same_sense: original.same_sense,
        tolerance: original.tolerance,
        seam_edges: vec![],
        color: original.color,
        degenerated_edges: vec![],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::SurfaceGeom;
    use crate::store::BRepStore;
    use crate::topo::BRepWire;

    #[test]
    fn unsplit_face_returns_one_region() {
        let mut reg = BRepStore::new();
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X },
            outer_wire: wire, inner_wires: vec![],
            same_sense: true, tolerance: 1e-6, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let result = split_face_along_curves(fk, &[], &reg);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].original_face, fk);
    }

    #[test]
    fn create_sub_face_from_triangle_uv() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-6,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let uv_tri = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let sub_fk = create_sub_face(fk, &uv_tri, &mut reg);
        assert!(sub_fk.is_some(), "should create sub-face from UV triangle");
        if let Some(sub) = sub_fk {
            let face = reg.faces.get(sub).unwrap();
            assert!(matches!(&face.surface, SurfaceGeom::Plane { .. }));
        }
    }

    #[test]
    fn test_sample_bspline_intersection_curve() {
        use crate::geom::CurveGeom;
        let curve = CurveGeom::BSpline {
            degree: 3,
            control_points: vec![
                PVec3::new(0.0, 0.0, 0.0),
                PVec3::new(1.0, 2.0, 0.0),
                PVec3::new(2.0, 2.0, 0.0),
                PVec3::new(3.0, 0.0, 0.0),
            ],
            knots: vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            weights: None,
        };
        let samples = sample_intersection_curve(&curve, 32);
        assert_eq!(samples.len(), 32, "should produce exactly 32 samples");
        assert!((samples[0] - PVec3::new(0.0, 0.0, 0.0)).length() < 1e-4,
            "first sample should be near start control point");
        assert!((samples[31] - PVec3::new(3.0, 0.0, 0.0)).length() < 1e-4,
            "last sample should be near end control point");
    }

    #[test]
    fn test_sample_polyline_intersection_curve() {
        use crate::geom::CurveGeom;
        let curve = CurveGeom::Polyline {
            points: vec![PVec3::ZERO, PVec3::X, PVec3::new(1.0, 1.0, 0.0)],
        };
        let samples = sample_intersection_curve(&curve, 10);
        assert_eq!(samples.len(), 10, "should produce 10 samples for polyline");
    }
}
