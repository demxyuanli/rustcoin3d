//! Adaptive edge discretization. T2.1-T2.2
//!
//! OCCT-aligned: BRepAdaptor_Curve(E,F) evaluates PCurve on surface;
//! BRep_Tool::Curve(E) is the fallback when no PCURVE is available.

use std::collections::HashMap;
use rc3d_core::math::Vec3;
use crate::topo::{EdgeKey, FaceKey, BRepEdge};
use crate::store::BRepStore;
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
            max_points: 512,
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

impl EdgePolygon {
    /// Refine the polygon by inserting midpoints between consecutive segments.
    pub fn subdivide(&mut self) {
        if self.params_3d.len() < 2 {
            return;
        }
        let mut new_3d = Vec::with_capacity(self.params_3d.len() * 2 - 1);
        for w in self.params_3d.windows(2) {
            let (t0, p0) = w[0];
            let (t1, p1) = w[1];
            new_3d.push((t0, p0));
            let t_mid = (t0 + t1) * 0.5;
            let p_mid = (p0 + p1) * 0.5;
            new_3d.push((t_mid, p_mid));
        }
        if let Some(&last) = self.params_3d.last() {
            new_3d.push(last);
        }
        self.params_3d = new_3d;

        for (_fk, pts) in self.params_2d.iter_mut() {
            if pts.len() < 2 {
                continue;
            }
            let mut new_2d = Vec::with_capacity(pts.len() * 2 - 1);
            for w in pts.windows(2) {
                let (t0, uv0) = w[0];
                let (t1, uv1) = w[1];
                new_2d.push((t0, uv0));
                let t_mid = (t0 + t1) * 0.5;
                let uv_mid = ((uv0.0 + uv1.0) * 0.5, (uv0.1 + uv1.1) * 0.5);
                new_2d.push((t_mid, uv_mid));
            }
            if let Some(&last) = pts.last() {
                new_2d.push(last);
            }
            *pts = new_2d;
        }
    }
}

/// Discretize all unique edges in a registry. Shared edges are discretized once.
pub fn discretize_all_edges(
    reg: &BRepStore,
    config: &EdgeDiscConfig,
) -> HashMap<EdgeKey, EdgePolygon> {
    let mut result = HashMap::new();
    for (ek, _edge) in reg.edges.iter() {
        result.entry(ek).or_insert_with(|| {
            let poly = discretize_edge(ek, reg, config);
            poly
        });
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
    reg: &BRepStore,
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
            cap_polyline_params(exact, config.max_points)
        } else if let Some((pcurve, surface, ss)) = primary_pcurve_on_surface(edge, reg) {
            if matches!(&edge.curve, CurveGeom::Line { .. }) && pcurve_is_line(pcurve) {
                sample_straight_pcurve(pcurve, surface, ss)
            } else if is_usable_pcurve(pcurve) {
                sample_pcurve_on_surface(pcurve, surface, &ec, ss)
            } else {
                let mesh_curve = mesh_curve_for_edge(edge, reg);
                sample_curve_adaptive(&mesh_curve, 0.0, 1.0, &ec)
            }
        } else {
            let mesh_curve = mesh_curve_for_edge(edge, reg);
            sample_curve_adaptive(&mesh_curve, 0.0, 1.0, &ec)
        }
    } else if let Some((pcurve, surface, ss)) = primary_pcurve_on_surface(edge, reg) {
        if matches!(&edge.curve, CurveGeom::Line { .. }) && pcurve_is_line(pcurve) {
            sample_straight_pcurve(pcurve, surface, ss)
        } else if is_usable_pcurve(pcurve) {
            sample_pcurve_on_surface(pcurve, surface, &ec, ss)
        } else {
            let mesh_curve = mesh_curve_for_edge(edge, reg);
            sample_curve_adaptive(&mesh_curve, 0.0, 1.0, &ec)
        }
    } else {
        let mesh_curve = mesh_curve_for_edge(edge, reg);
        sample_curve_adaptive(&mesh_curve, 0.0, 1.0, &ec)
    };

    let mut params_2d = HashMap::new();
    for (&face_key, (pcurve, same_sense)) in &edge.pcurves {
        if reg.faces.get(face_key).is_some() {
            let pts_2d: Vec<(f32, (f32, f32))> = params_3d
                .iter()
                .map(|&(t, _)| {
                    let pc_t = if *same_sense { t } else { 1.0 - t };
                    let uv = pcurve.d0(pc_t);
                    (t, (uv.0, uv.1))
                })
                .collect();
            params_2d.insert(face_key, pts_2d);
        }
    }

    EdgePolygon { params_3d, params_2d }
}

/// Polyline for viewport edge overlay: per-face PCURVE when a single face owns the edge,
/// otherwise the stored 3D topological curve (avoids primary-face bias on offset pairs).
pub fn discretize_edge_overlay(
    ek: EdgeKey,
    reg: &BRepStore,
    config: &EdgeDiscConfig,
) -> Vec<Vec3> {
    let edge = match reg.edges.get(ek) {
        Some(e) => e,
        None => return Vec::new(),
    };
    if edge.pcurves.len() <= 1 {
        return discretize_edge(ek, reg, config)
            .params_3d
            .into_iter()
            .map(|(_, p)| p)
            .collect();
    }
    let chord_len = (edge.curve.d0(1.0) - edge.curve.d0(0.0)).length();
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
    let mesh_curve = mesh_curve_for_edge(edge, reg);
    sample_curve_adaptive(&mesh_curve, 0.0, 1.0, &ec)
        .into_iter()
        .map(|(_, p)| p)
        .collect()
}

/// True for healed seam edges or closed isoparam edges (v_low == v_high).
fn is_seam_or_isoparam_edge(ek: EdgeKey, edge: &BRepEdge, reg: &BRepStore) -> bool {
    if edge.v_low == edge.v_high {
        return true;
    }
    reg.faces
        .iter()
        .any(|(_, face)| face.seam_edges.contains(&ek))
}

/// Uniformly subsample dense polylines (STEP POLYLINE / B-spline control nets).
fn cap_polyline_params(params: Vec<(f32, Vec3)>, max_points: usize) -> Vec<(f32, Vec3)> {
    if params.len() <= max_points || max_points < 2 {
        return params;
    }
    let n = params.len();
    (0..max_points)
        .map(|i| {
            let src = i * (n - 1) / (max_points - 1);
            params[src]
        })
        .collect()
}

/// Use stored polyline knots exactly (seam / isoparam edges aligned with surface mesh grid).
fn sample_polyline_on_surface_exact(
    edge: &BRepEdge,
    reg: &BRepStore,
) -> Option<Vec<(f32, Vec3)>> {
    let mut face_keys: Vec<FaceKey> = edge.pcurves.keys().copied().collect();
    face_keys.sort_unstable();
    let face_key = *face_keys.first()?;
    let (pcurve, _same_sense) = edge.pcurves.get(&face_key)?;
    let face = reg.faces.get(face_key)?;
    let crate::geom::Curve2d::Polyline { points: uv_pts } = pcurve else {
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
                (t, face.surface.d0_native(uv.0, uv.1))
            })
            .collect(),
    )
}

/// Pick the first face PCURVE pair for OCCT-style BRepAdaptor_Curve(E, F).
fn primary_pcurve_on_surface<'a>(
    edge: &'a BRepEdge,
    reg: &'a BRepStore,
) -> Option<(&'a crate::geom::Curve2d, &'a SurfaceGeom, bool)> {
    let mut keys: Vec<FaceKey> = edge.pcurves.keys().copied().collect();
    keys.sort_unstable();
    let face_key = keys.first()?;
    let (pcurve, same_sense) = edge.pcurves.get(face_key)?;
    let face = reg.faces.get(*face_key)?;
    Some((pcurve, &face.surface, *same_sense))
}

fn is_usable_pcurve(pcurve: &crate::geom::Curve2d) -> bool {
    let p0 = pcurve.d0(0.0);
    let p1 = pcurve.d0(1.0);
    let d02 = (p1.0 - p0.0).powi(2) + (p1.1 - p0.1).powi(2);
    if d02 >= 1e-12 {
        return true;
    }
    // Closed curve (p0 ≈ p1): check midpoint extent to distinguish
    // valid periodic curves (BSpline, Circle) from truly degenerate ones.
    let pmid = pcurve.d0(0.5);
    let dm2 = (pmid.0 - p0.0).powi(2) + (pmid.1 - p0.1).powi(2);
    dm2 >= 1e-12
}

pub use crate::geom::eval_pcurve_on_surface;

fn pcurve_is_line(pcurve: &crate::geom::Curve2d) -> bool {
    match pcurve {
        crate::geom::Curve2d::Line { .. } => true,
        crate::geom::Curve2d::Trimmed { basis, .. } => pcurve_is_line(basis),
        _ => false,
    }
}

fn sample_straight_pcurve(
    pcurve: &crate::geom::Curve2d,
    surface: &SurfaceGeom,
    same_sense: bool,
) -> Vec<(f32, Vec3)> {
    let t0 = if same_sense { 0.0 } else { 1.0 };
    let t1 = if same_sense { 1.0 } else { 0.0 };
    vec![
        (0.0, eval_pcurve_on_surface(pcurve, surface, t0)),
        (1.0, eval_pcurve_on_surface(pcurve, surface, t1)),
    ]
}

fn eval_pcurve_on_surface_d1(pcurve: &crate::geom::Curve2d, surface: &SurfaceGeom, t: f32) -> Vec3 {
    let uv = pcurve.d0(t);
    let duv = pcurve.d1(t).1;
    let (su, sv) = surface.d1_native(uv.0, uv.1);
    su * duv.0 + sv * duv.1
}

/// Adaptive sampling via PCurve composed with surface evaluator.
fn sample_pcurve_on_surface(
    pcurve: &crate::geom::Curve2d,
    surface: &SurfaceGeom,
    config: &EdgeDiscConfig,
    same_sense: bool,
) -> Vec<(f32, Vec3)> {
    let pc_t = |t: f32| if same_sense { t } else { 1.0 - t };
    let mut params = Vec::new();

    let p0 = eval_pcurve_on_surface(pcurve, surface, pc_t(0.0));
    let p1 = eval_pcurve_on_surface(pcurve, surface, pc_t(1.0));
    let is_closed = (p1 - p0).length() < 1e-6;
    let n_seed = if is_closed { config.min_points.max(4) } else { 1 };
    for i in 0..=n_seed {
        let t = i as f32 / n_seed as f32;
        params.push((t, eval_pcurve_on_surface(pcurve, surface, pc_t(t))));
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
            let p_mid = eval_pcurve_on_surface(pcurve, surface, pc_t(t_mid));
            let chordal_dev = (p_mid - (p_a + p_b) * 0.5).length();
            let seg_len = (p_b - p_a).length();

            let d1_a = eval_pcurve_on_surface_d1(pcurve, surface, pc_t(t_a));
            let d1_b = eval_pcurve_on_surface_d1(pcurve, surface, pc_t(t_b));
            let angle_dev = if d1_a.length() > 1e-10 && d1_b.length() > 1e-10 {
                (d1_a.normalize().dot(d1_b.normalize())).acos().abs()
            } else {
                0.0
            };

            let kappa = 0.0; // Curve2d has no curvature() — curvature in UV space is
            // handled through chordal/angular deviation on the 3D surface.
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
fn mesh_curve_for_edge(edge: &BRepEdge, reg: &BRepStore) -> CurveGeom {
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
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::BRepFace;
    use crate::store::BRepStore;

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
        let mut reg = BRepStore::new();
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
        let pcurve = Curve2d::Line { origin: (0.0, 0.0), direction: (10.0, 0.0) };
        let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, (pcurve, true));

        let config = EdgeDiscConfig::default();
        let poly = discretize_edge(ek, &reg, &config);
        assert!(poly.params_3d.len() >= 2);
        let end = poly.params_3d.last().unwrap().1;
        assert!((end.x - 10.0).abs() < 0.01, "expected x~10, got {:?}", end);
        let uv_end = poly.params_2d.get(&face_key).unwrap().last().unwrap().1;
        assert!((uv_end.0 - 10.0).abs() < 0.01);
    }
}
