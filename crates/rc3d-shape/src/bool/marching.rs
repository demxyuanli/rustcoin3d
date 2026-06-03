//! NURBS-NURBS intersection via adaptive marching (OCC IntPatch_ImpPrmIntersection).

use rc3d_core::math::Vec3;
use crate::geom::SurfaceGeom;

/// A seed point for curve tracing: 3D point + UV params on both surfaces.
#[derive(Debug, Clone)]
pub struct SeedPoint {
    pub point: Vec3,
    pub uv_a: (f32, f32),
    pub uv_b: (f32, f32),
}

/// Find seed points by grid-sampling both surfaces.
/// Returns candidate proximity regions where surface distance < proximity_tol.
pub fn find_seeds(
    surf_a: &SurfaceGeom,
    surf_b: &SurfaceGeom,
    grid_res: usize,
    proximity_tol: f32,
) -> Vec<SeedPoint> {
    let range_a = surf_a.param_range();
    let mut seeds = Vec::new();

    for i in 0..=grid_res {
        let u = range_a.u_min + (range_a.u_max - range_a.u_min) * i as f32 / grid_res as f32;
        for j in 0..=grid_res {
            let v = range_a.v_min + (range_a.v_max - range_a.v_min) * j as f32 / grid_res as f32;
            let (un, vn) = surf_a.native_uv_to_d0(u, v);
            let pt = surf_a.d0(un, vn);
            if let Some(uv_b) = surf_b.project(pt) {
                let (un_b, vn_b) = surf_b.native_uv_to_d0(uv_b.0, uv_b.1);
                let proj_pt = surf_b.d0(un_b, vn_b);
                let dist = (pt - proj_pt).length();
                if dist < proximity_tol {
                    seeds.push(SeedPoint { point: pt, uv_a: (u, v), uv_b });
                }
            }
        }
    }
    seeds
}

/// Trace an intersection curve from a seed point.
pub fn trace_curve(
    surf_a: &SurfaceGeom,
    surf_b: &SurfaceGeom,
    seed: SeedPoint,
    step_size: f32,
    max_steps: usize,
    tolerance: f32,
) -> Option<Vec<Vec3>> {
    let range_a = surf_a.param_range();
    let mut points = vec![seed.point];
    let mut current_uv_a = seed.uv_a;

    for _ in 0..max_steps {
        let (un_a, vn_a) = surf_a.native_uv_to_d0(current_uv_a.0, current_uv_a.1);
        let (du_a, dv_a) = surf_a.d1(un_a, vn_a);
        let normal_a = du_a.cross(dv_a).normalize();

        let tangent = if let Some(uv_b) = surf_b.project(*points.last().unwrap()) {
            let (un_b, vn_b) = surf_b.native_uv_to_d0(uv_b.0, uv_b.1);
            let (du_b, dv_b) = surf_b.d1(un_b, vn_b);
            let normal_b = du_b.cross(dv_b).normalize();
            let t = normal_a.cross(normal_b);
            if t.length() < tolerance { break; }
            t.normalize()
        } else {
            break;
        };

        let next_pt = points.last().unwrap() + tangent * step_size;
        let Some(uv_a_next) = surf_a.project(next_pt) else { break };

        let (un_a_n, vn_a_n) = surf_a.native_uv_to_d0(uv_a_next.0, uv_a_next.1);
        let corrected = surf_a.d0(un_a_n, vn_a_n);
        points.push(corrected);
        current_uv_a = uv_a_next;

        if uv_a_next.0 < range_a.u_min || uv_a_next.0 > range_a.u_max
            || uv_a_next.1 < range_a.v_min || uv_a_next.1 > range_a.v_max
        {
            break;
        }
    }

    if points.len() >= 2 { Some(points) } else { None }
}

/// Trace an intersection curve bidirectionally from a seed point.
pub fn trace_curve_bidirectional(
    surf_a: &SurfaceGeom,
    surf_b: &SurfaceGeom,
    seed: &SeedPoint,
    step_size: f32,
    max_steps: usize,
    tolerance: f32,
) -> Option<Vec<Vec3>> {
    // Trace forward
    let forward = trace_curve_in_dir(surf_a, surf_b, seed, step_size, max_steps, tolerance, true);
    // Trace backward
    let backward = trace_curve_in_dir(surf_a, surf_b, seed, step_size, max_steps, tolerance, false);

    match (forward, backward) {
        (Some(fwd), Some(bwd)) => {
            // Combine: reverse backward, skip duplicate seed point, append forward
            let mut result: Vec<Vec3> = bwd.into_iter().rev().collect();
            result.pop(); // remove duplicate seed point
            result.extend(fwd);
            Some(result)
        }
        (Some(fwd), None) => Some(fwd),
        (None, Some(bwd)) => {
            let result: Vec<Vec3> = bwd.into_iter().rev().collect();
            Some(result)
        }
        (None, None) => None,
    }
}

/// Trace in one direction (forward=true) or opposite (forward=false)
fn trace_curve_in_dir(
    surf_a: &SurfaceGeom,
    surf_b: &SurfaceGeom,
    seed: &SeedPoint,
    step_size: f32,
    max_steps: usize,
    tolerance: f32,
    forward: bool,
) -> Option<Vec<Vec3>> {
    let range_a = surf_a.param_range();
    let mut points = vec![seed.point];
    let mut current_uv_a = seed.uv_a;
    let sign = if forward { 1.0 } else { -1.0 };

    for _ in 0..max_steps {
        let (un_a, vn_a) = surf_a.native_uv_to_d0(current_uv_a.0, current_uv_a.1);
        let (du_a, dv_a) = surf_a.d1(un_a, vn_a);
        let normal_a = du_a.cross(dv_a).normalize();

        let tangent = if let Some(uv_b) = surf_b.project(*points.last().unwrap()) {
            let (un_b, vn_b) = surf_b.native_uv_to_d0(uv_b.0, uv_b.1);
            let (du_b, dv_b) = surf_b.d1(un_b, vn_b);
            let normal_b = du_b.cross(dv_b).normalize();
            let t = normal_a.cross(normal_b) * sign;
            if t.length() < tolerance { break; }
            t.normalize()
        } else {
            break;
        };

        let next_pt = points.last().unwrap() + tangent * step_size;
        let Some(uv_a_next) = surf_a.project(next_pt) else { break };

        let (un_a_n, vn_a_n) = surf_a.native_uv_to_d0(uv_a_next.0, uv_a_next.1);
        let corrected = surf_a.d0(un_a_n, vn_a_n);
        points.push(corrected);
        current_uv_a = uv_a_next;

        if uv_a_next.0 < range_a.u_min || uv_a_next.0 > range_a.u_max
            || uv_a_next.1 < range_a.v_min || uv_a_next.1 > range_a.v_max
        {
            break;
        }
    }

    if points.len() >= 2 { Some(points) } else { None }
}

fn step_size_for_surfaces(surf_a: &SurfaceGeom, surf_b: &SurfaceGeom) -> f32 {
    let range_a = surf_a.param_range();
    let range_b = surf_b.param_range();
    let diag_a = ((range_a.u_max - range_a.u_min).powi(2)
        + (range_a.v_max - range_a.v_min).powi(2))
    .sqrt();
    let diag_b = ((range_b.u_max - range_b.u_min).powi(2)
        + (range_b.v_max - range_b.v_min).powi(2))
    .sqrt();
    diag_a.min(diag_b) * 0.01
}

fn cluster_directions(dirs: Vec<Vec3>, threshold: f32) -> Vec<Vec3> {
    let mut result = Vec::new();
    let mut used = vec![false; dirs.len()];
    for i in 0..dirs.len() {
        if used[i] { continue; }
        let mut avg = dirs[i];
        let mut count = 1;
        for j in (i + 1)..dirs.len() {
            if used[j] { continue; }
            if avg.dot(dirs[j]) > threshold {
                avg += dirs[j];
                count += 1;
                used[j] = true;
            }
        }
        if count > 0 {
            result.push((avg / count as f32).normalize());
        }
    }
    result
}

/// Check if a point is a bifurcation point where multiple intersection branches meet.
/// Returns alternative tangent directions if bifurcation is detected.
pub fn detect_bifurcation(
    surf_a: &SurfaceGeom,
    surf_b: &SurfaceGeom,
    point: Vec3,
    current_tangent: Vec3,
    uv_a: (f32, f32),
    tolerance: f32,
) -> Vec<Vec3> {
    // Sample a small circle around the current point in the tangent plane
    // of surface A, project to surface B, and check for multiple valid tangents
    let (un_a, vn_a) = surf_a.native_uv_to_d0(uv_a.0, uv_a.1);
    let (du_a, dv_a) = surf_a.d1(un_a, vn_a);
    let normal_a = du_a.cross(dv_a).normalize();

    let perp1 = if current_tangent.cross(normal_a).length() > 1e-10 {
        current_tangent.cross(normal_a).normalize()
    } else {
        // Fallback: use surface U direction
        du_a.normalize()
    };
    let perp2 = normal_a.cross(perp1).normalize();

    let mut alt_tangents = Vec::new();
    let n_samples = 36;
    let radius = step_size_for_surfaces(surf_a, surf_b) * 2.0;

    for i in 0..n_samples {
        let angle = std::f32::consts::TAU * i as f32 / n_samples as f32;
        let sample_dir = perp1 * angle.cos() + perp2 * angle.sin();
        // Skip directions close to current tangent
        if sample_dir.dot(current_tangent).abs() > 0.9 { continue; }

        let sample_pt = point + sample_dir * radius;
        if let Some(uv_b) = surf_b.project(sample_pt) {
            let (un_b, vn_b) = surf_b.native_uv_to_d0(uv_b.0, uv_b.1);
            let proj = surf_b.d0(un_b, vn_b);
            if (proj - sample_pt).length() < tolerance {
                // This direction also lies on the intersection — potential branch
                alt_tangents.push(sample_dir);
            }
        }
    }

    // Cluster nearby directions to avoid duplicates
    cluster_directions(alt_tangents, 0.3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::SurfaceGeom;

    #[test]
    fn marching_seeds_found_for_intersecting_planes() {
        let s1 = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let s2 = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Y, u_dir: Vec3::X };
        let seeds = find_seeds(&s1, &s2, 8, 0.01);
        assert!(!seeds.is_empty(), "intersecting planes should produce seeds");
    }

    #[test]
    fn trace_curve_produces_points() {
        let s1 = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let s2 = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Y, u_dir: Vec3::X };
        let seeds = find_seeds(&s1, &s2, 8, 0.01);
        if let Some(seed) = seeds.first() {
            let pts = trace_curve(&s1, &s2, seed.clone(), 0.05, 50, 1e-4);
            assert!(pts.is_some());
            assert!(pts.unwrap().len() >= 2);
        }
    }
}
