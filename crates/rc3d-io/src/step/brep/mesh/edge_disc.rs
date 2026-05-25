//! Adaptive edge discretization. T2.1-T2.2

use std::collections::HashMap;
use rc3d_core::math::Vec3;
use crate::step::brep::topo::{EdgeKey, FaceKey};
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::geom::CurveGeom;

/// Configuration for edge discretization.
#[derive(Debug, Clone)]
pub struct EdgeDiscConfig {
    pub deflection: f32,
    pub angle_deflection: f32,
    pub min_points: usize,
    pub max_points: usize,
}

impl Default for EdgeDiscConfig {
    fn default() -> Self {
        Self { deflection: 0.1, angle_deflection: 0.1, min_points: 2, max_points: 256 }
    }
}

/// A discretized edge polygon: 3D points + per-face 2D (UV) points.
#[derive(Debug, Clone)]
pub struct EdgePolygon {
    pub params_3d: Vec<(f32, Vec3)>,
    pub params_2d: HashMap<FaceKey, Vec<(f32, (f32, f32))>>,
}

/// Discretize all unique edges in a registry. Shared edges are discretized once.
pub fn discretize_all_edges(
    reg: &BRepRegistry,
    config: &EdgeDiscConfig,
) -> HashMap<EdgeKey, EdgePolygon> {
    let mut result = HashMap::new();
    for (ek, _edge) in reg.edges.iter() {
        if !result.contains_key(&ek) {
            let poly = discretize_edge(ek, reg, config);
            result.insert(ek, poly);
        }
    }
    result
}

/// Discretize a single edge: 3D curve + PCURVEs for each face.
pub fn discretize_edge(
    ek: EdgeKey,
    reg: &BRepRegistry,
    config: &EdgeDiscConfig,
) -> EdgePolygon {
    let edge = match reg.edges.get(ek) {
        Some(e) => e,
        None => return EdgePolygon { params_3d: vec![], params_2d: HashMap::new() },
    };

    let params_3d = sample_curve_adaptive(&edge.curve, 0.0, 1.0, config);

    let mut params_2d = HashMap::new();
    for (&face_key, pcurve) in &edge.pcurves {
        let pts_2d = sample_pcurve_adaptive(pcurve, 0.0, 1.0, config);
        params_2d.insert(face_key, pts_2d);
    }

    EdgePolygon { params_3d, params_2d }
}

/// Adaptive sampling of a 3D curve.
fn sample_curve_adaptive(
    curve: &CurveGeom, t0: f32, t1: f32, config: &EdgeDiscConfig,
) -> Vec<(f32, Vec3)> {
    let mut params = Vec::new();

    // Seed with uniform samples. Ensure at least 3 points for closed curves
    // (where start ≈ end — 2 points would produce identical endpoints, hiding curvature).
    let p0 = curve.d0(t0);
    let p1 = curve.d0(t1);
    let is_closed = (p1 - p0).length() < 1e-6;
    let n_seed = if is_closed { config.min_points.max(4) } else { 1 }; // 1 → 2 seed pts for lines
    for i in 0..=n_seed {
        let t = t0 + (t1 - t0) * i as f32 / n_seed as f32;
        params.push((t, curve.d0(t)));
    }

    // Refine: split segments where chordal deviation exceeds threshold
    let mut iterations = 0;
    let max_iter = 10;
    while iterations < max_iter && params.len() < config.max_points {
        let mut new_params = Vec::with_capacity(params.len() * 2);
        let mut any_split = false;

        for w in params.windows(2) {
            let (t_a, p_a) = w[0];
            let (t_b, p_b) = w[1];
            new_params.push((t_a, p_a));

            let t_mid = (t_a + t_b) * 0.5;
            let p_mid = curve.d0(t_mid);
            let chordal_dev = (p_mid - (p_a + p_b) * 0.5).length();

            // Also check angular deviation
            let d1_a = curve.d1(t_a);
            let d1_b = curve.d1(t_b);
            let angle_dev = if d1_a.length() > 1e-10 && d1_b.length() > 1e-10 {
                (d1_a.normalize().dot(d1_b.normalize())).acos().abs()
            } else { 0.0 };

            if chordal_dev > config.deflection || angle_dev > config.angle_deflection {
                new_params.push((t_mid, p_mid));
                any_split = true;
            }
        }
        new_params.push(params.last().copied().unwrap());
        params = new_params;

        if !any_split { break; }
        iterations += 1;
    }

    params
}

/// Adaptive sampling of a 2D PCURVE, returning (t, (u,v)) pairs.
fn sample_pcurve_adaptive(
    pcurve: &CurveGeom, t0: f32, t1: f32, config: &EdgeDiscConfig,
) -> Vec<(f32, (f32, f32))> {
    let params_3d = sample_curve_adaptive(pcurve, t0, t1, config);
    params_3d.into_iter().map(|(t, p)| (t, (p.x, p.y))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::CurveGeom;

    #[test]
    fn test_discretize_line_minimal() {
        let curve = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::new(10.0, 0.0, 0.0) };
        let config = EdgeDiscConfig::default();
        let pts = sample_curve_adaptive(&curve, 0.0, 1.0, &config);
        assert!(pts.len() >= 2, "line needs at least 2 pts, got {}", pts.len()); // n_seed=1 → 2 seed pts
        // Line should not need extra refinement (no curvature)
        assert!(pts.len() <= 3);
    }

    #[test]
    fn test_discretize_circle_needs_more_points() {
        let curve = CurveGeom::Circle { center: Vec3::ZERO, axis: Vec3::Z, radius: 1.0 };
        let config = EdgeDiscConfig::default();
        let pts = sample_curve_adaptive(&curve, 0.0, 1.0, &config);
        // Circle needs more points due to curvature
        assert!(pts.len() >= 4);
    }
}
