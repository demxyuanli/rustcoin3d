//! Adaptive edge discretization. T2.1-T2.2
//!
//! OCCT-aligned: BRepAdaptor_Curve(E,F) evaluates PCurve on surface;
//! BRep_Tool::Curve(E) is the fallback when no PCURVE is available.

use std::collections::HashMap;
use rc3d_core::math::Vec3;
use crate::topo::{EdgeKey, FaceKey, BRepEdge};
use crate::store::BRepRegistry;
use crate::geom::{CurveGeom, SurfaceGeom, normalize_edge_curve_to_vertices};

/// Configuration for edge discretization.
#[derive(Debug, Clone)]
pub struct EdgeDiscConfig {
    pub deflection: f32,
    pub angle_deflection: f32,
    pub min_points: usize,
    pub max_points: usize,
    pub relative_deflection: bool,
}

impl Default for EdgeDiscConfig {
    fn default() -> Self {
        Self {
            deflection: 0.01,
            angle_deflection: 0.15,
            min_points: 2,
            max_points: 128,
            relative_deflection: false,
        }
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

/// Discretize a single edge: PCurve-on-surface (preferred) + per-face UV params.
/// Estimate arc length of a curve by sampling.
fn estimate_curve_length(curve: &crate::geom::CurveGeom) -> f32 {
    let n = 64;
    let mut total = 0.0f32;
    let mut prev = curve.d0(0.0);
    for i in 1..=n {
        let t = i as f32 / n as f32;
        let p = curve.d0(t);
        total += (p - prev).length();
        prev = p;
    }
    total
}

pub fn discretize_edge(
    ek: EdgeKey,
    reg: &BRepRegistry,
    config: &EdgeDiscConfig,
) -> EdgePolygon {
    let edge = match reg.edges.get(ek) {
        Some(e) => e,
        None => return EdgePolygon { params_3d: vec![], params_2d: HashMap::new() },
    };

    let chord_len = (edge.curve.d0(1.0) - edge.curve.d0(0.0)).length();
    // For closed curves (circles etc.), chord is zero; use arc length estimate.
    let curve_len = if chord_len < 1e-10 {
        estimate_curve_length(&edge.curve).max(1.0)
    } else {
        chord_len
    };
    let effective_deflection = if config.relative_deflection {
        curve_len * config.deflection
    } else {
        config.deflection
    };
    let ec = EdgeDiscConfig {
        deflection: effective_deflection,
        angle_deflection: config.angle_deflection,
        min_points: config.min_points,
        max_points: config.max_points,
        relative_deflection: config.relative_deflection,
    };

    let params_3d = if is_seam_or_isoparam_edge(ek, edge, reg) {
        if let Some(exact) = sample_polyline_on_surface_exact(edge, reg) {
            exact
        } else if let Some((pcurve, surface)) = primary_pcurve_on_surface(edge, reg) {
            if is_usable_pcurve(pcurve) {
                sample_pcurve_on_surface(pcurve, surface, &ec)
            } else {
                let mesh_curve = mesh_curve_for_edge(edge, reg);
                sample_curve_adaptive(&mesh_curve, 0.0, 1.0, &ec)
            }
        } else {
            let mesh_curve = mesh_curve_for_edge(edge, reg);
            sample_curve_adaptive(&mesh_curve, 0.0, 1.0, &ec)
        }
    } else if let Some((pcurve, surface)) = primary_pcurve_on_surface(edge, reg) {
        if is_usable_pcurve(pcurve) {
            sample_pcurve_on_surface(pcurve, surface, &ec)
        } else {
            let mesh_curve = mesh_curve_for_edge(edge, reg);
            sample_curve_adaptive(&mesh_curve, 0.0, 1.0, &ec)
        }
    } else {
        let mesh_curve = mesh_curve_for_edge(edge, reg);
        sample_curve_adaptive(&mesh_curve, 0.0, 1.0, &ec)
    };

    let mut params_2d = HashMap::new();
    for (&face_key, pcurve) in &edge.pcurves {
        if reg.faces.get(face_key).is_some() {
            let pts_2d: Vec<(f32, (f32, f32))> = params_3d
                .iter()
                .map(|&(t, _)| {
                    let uv = pcurve.d0(t);
                    (t, (uv.x, uv.y))
                })
                .collect();
            params_2d.insert(face_key, pts_2d);
        }
    }

    EdgePolygon { params_3d, params_2d }
}

/// True for healed seam edges or closed isoparam edges (v_low == v_high).
fn is_seam_or_isoparam_edge(ek: EdgeKey, edge: &BRepEdge, reg: &BRepRegistry) -> bool {
    if edge.v_low == edge.v_high {
        return true;
    }
    reg.faces
        .iter()
        .any(|(_, face)| face.seam_edges.contains(&ek))
}

/// Use stored polyline knots exactly (seam / isoparam edges aligned with surface mesh grid).
fn sample_polyline_on_surface_exact(
    edge: &BRepEdge,
    reg: &BRepRegistry,
) -> Option<Vec<(f32, Vec3)>> {
    let mut face_keys: Vec<FaceKey> = edge.pcurves.keys().copied().collect();
    face_keys.sort_unstable();
    let face_key = *face_keys.first()?;
    let pcurve = edge.pcurves.get(&face_key)?;
    let face = reg.faces.get(face_key)?;
    let CurveGeom::Polyline { points: uv_pts } = pcurve else {
        return None;
    };
    if uv_pts.len() < 2 {
        return None;
    }

    let n = uv_pts.len();
    let denom = (n - 1).max(1) as f32;

    if let CurveGeom::Polyline { points: pts_3d } = &edge.curve {
        if pts_3d.len() == n {
            return Some(
                pts_3d
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (i as f32 / denom, *p))
                    .collect(),
            );
        }
    }

    Some(
        uv_pts
            .iter()
            .enumerate()
            .map(|(i, uv)| {
                let t = i as f32 / denom;
                (t, face.surface.d0_native(uv.x, uv.y))
            })
            .collect(),
    )
}

/// Pick the first face PCURVE pair for OCCT-style BRepAdaptor_Curve(E, F).
fn primary_pcurve_on_surface<'a>(
    edge: &'a BRepEdge,
    reg: &'a BRepRegistry,
) -> Option<(&'a CurveGeom, &'a SurfaceGeom)> {
    let mut keys: Vec<FaceKey> = edge.pcurves.keys().copied().collect();
    keys.sort_unstable();
    let face_key = keys.first()?;
    let pcurve = edge.pcurves.get(face_key)?;
    let face = reg.faces.get(*face_key)?;
    Some((pcurve, &face.surface))
}

fn is_usable_pcurve(pcurve: &CurveGeom) -> bool {
    let p0 = pcurve.d0(0.0);
    let p1 = pcurve.d0(1.0);
    if (p1 - p0).length_squared() >= 1e-12 {
        return true;
    }
    // Closed curve (p0 ≈ p1): check midpoint extent to distinguish
    // valid periodic curves (BSpline, Circle) from truly degenerate ones.
    let pmid = pcurve.d0(0.5);
    (pmid - p0).length_squared() >= 1e-12
}

/// Evaluate PCurve on surface at t (OCCT BRepAdaptor_Curve with face context).
fn eval_pcurve_on_surface(pcurve: &CurveGeom, surface: &SurfaceGeom, t: f32) -> Vec3 {
    let uv = pcurve.d0(t);
    surface.d0_native(uv.x, uv.y)
}

fn eval_pcurve_on_surface_d1(pcurve: &CurveGeom, surface: &SurfaceGeom, t: f32) -> Vec3 {
    let uv = pcurve.d0(t);
    let duv = pcurve.d1(t);
    let (su, sv) = surface.d1_native(uv.x, uv.y);
    su * duv.x + sv * duv.y
}

/// Adaptive sampling via PCurve composed with surface evaluator.
fn sample_pcurve_on_surface(
    pcurve: &CurveGeom,
    surface: &SurfaceGeom,
    config: &EdgeDiscConfig,
) -> Vec<(f32, Vec3)> {
    let mut params = Vec::new();

    let p0 = eval_pcurve_on_surface(pcurve, surface, 0.0);
    let p1 = eval_pcurve_on_surface(pcurve, surface, 1.0);
    let is_closed = (p1 - p0).length() < 1e-6;
    let n_seed = if is_closed { config.min_points.max(4) } else { 1 };
    for i in 0..=n_seed {
        let t = i as f32 / n_seed as f32;
        params.push((t, eval_pcurve_on_surface(pcurve, surface, t)));
    }

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
            let p_mid = eval_pcurve_on_surface(pcurve, surface, t_mid);
            let chordal_dev = (p_mid - (p_a + p_b) * 0.5).length();
            let seg_len = (p_b - p_a).length();

            let d1_a = eval_pcurve_on_surface_d1(pcurve, surface, t_a);
            let d1_b = eval_pcurve_on_surface_d1(pcurve, surface, t_b);
            let angle_dev = if d1_a.length() > 1e-10 && d1_b.length() > 1e-10 {
                (d1_a.normalize().dot(d1_b.normalize())).acos().abs()
            } else {
                0.0
            };

            let kappa = pcurve.curvature(t_mid);
            let curvature_dev = kappa * seg_len;

            if chordal_dev > config.deflection
                || angle_dev > config.angle_deflection
                || curvature_dev > config.angle_deflection
            {
                new_params.push((t_mid, p_mid));
                any_split = true;
            }
        }
        new_params.push(params.last().copied().unwrap());
        params = new_params;

        if !any_split {
            break;
        }
        iterations += 1;
    }

    params
}

/// Curve used for mesh discretization: always spans `v_low` → `v_high` at t=0..1.
fn mesh_curve_for_edge(edge: &BRepEdge, reg: &BRepRegistry) -> CurveGeom {
    if edge.v_low == edge.v_high {
        return edge.curve.clone();
    }
    let Some(v_lo) = reg.vertices.get(edge.v_low) else {
        return edge.curve.clone();
    };
    let Some(v_hi) = reg.vertices.get(edge.v_high) else {
        return edge.curve.clone();
    };
    normalize_edge_curve_to_vertices(
        edge.curve.clone(),
        v_lo.position,
        v_hi.position,
        edge.tolerance,
    )
}

/// Adaptive sampling of a 3D curve.
fn sample_curve_adaptive(
    curve: &CurveGeom, t0: f32, t1: f32, config: &EdgeDiscConfig,
) -> Vec<(f32, Vec3)> {
    let mut params = Vec::new();

    let p0 = curve.d0(t0);
    let p1 = curve.d0(t1);
    let is_closed = (p1 - p0).length() < 1e-6;
    let n_seed = if is_closed { config.min_points.max(4) } else { 1 };
    for i in 0..=n_seed {
        let t = t0 + (t1 - t0) * i as f32 / n_seed as f32;
        params.push((t, curve.d0(t)));
    }

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
            let seg_len = (p_b - p_a).length();

            let d1_a = curve.d1(t_a);
            let d1_b = curve.d1(t_b);
            let angle_dev = if d1_a.length() > 1e-10 && d1_b.length() > 1e-10 {
                (d1_a.normalize().dot(d1_b.normalize())).acos().abs()
            } else { 0.0 };

            let kappa = curve.curvature(t_mid);
            let curvature_dev = kappa * seg_len;

            if chordal_dev > config.deflection
                || angle_dev > config.angle_deflection
                || curvature_dev > config.angle_deflection
            {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::BRepFace;
    use crate::store::BRepRegistry;

    #[test]
    fn test_discretize_line_minimal() {
        let curve = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::new(10.0, 0.0, 0.0) };
        let config = EdgeDiscConfig::default();
        let pts = sample_curve_adaptive(&curve, 0.0, 1.0, &config);
        assert!(pts.len() >= 2, "line needs at least 2 pts, got {}", pts.len());
        assert!(pts.len() <= 3);
    }

    #[test]
    fn test_discretize_circle_needs_more_points() {
        let curve = CurveGeom::circle(Vec3::ZERO, Vec3::Z, 1.0);
        let config = EdgeDiscConfig::default();
        let pts = sample_curve_adaptive(&curve, 0.0, 1.0, &config);
        assert!(pts.len() >= 4);
    }

    #[test]
    fn test_pcurve_on_plane_matches_3d_line() {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(10.0, 0.0, 0.0), 1e-4);
        let wire = reg.wires.insert(crate::topo::BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let curve_3d = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::new(10.0, 0.0, 0.0) };
        let pcurve = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(10.0, 0.0, 0.0),
        };
        let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve);

        let config = EdgeDiscConfig::default();
        let poly = discretize_edge(ek, &reg, &config);
        assert!(poly.params_3d.len() >= 2);
        let end = poly.params_3d.last().unwrap().1;
        assert!((end.x - 10.0).abs() < 0.01, "expected x~10, got {:?}", end);
        let uv_end = poly.params_2d.get(&face_key).unwrap().last().unwrap().1;
        assert!((uv_end.0 - 10.0).abs() < 0.01);
    }
}
