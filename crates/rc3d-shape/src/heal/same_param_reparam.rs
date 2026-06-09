//! OCC BRepLib::SameParameter — iterative PCurve reparameterization.
//!
//! Given an edge with a 3D curve and a PCurve on a face surface,
//! adjusts the PCurve parameters so that for all t ∈ [0,1]:
//!   | surface(pcurve(t)) - curve3d(t) | < tolerance

use rc3d_core::math::Vec3;

use crate::geom::{Curve2d, CurveGeom, SurfaceGeom};
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey};

/// Result of reparameterizing one PCurve.
#[derive(Debug, Clone)]
pub struct ReparamResult {
    pub edge_key: EdgeKey,
    pub face_key: FaceKey,
    pub max_deviation_before: f32,
    pub max_deviation_after: f32,
    pub converged: bool,
    pub iterations: usize,
}

/// Reparameterize PCurves for all edges in given shells.
pub fn same_parameter_reparam(
    reg: &mut BRepStore,
    shell_keys: &[crate::topo::ShellKey],
    tolerance: f32,
    max_iterations: usize,
) -> (Vec<ReparamResult>, usize) {
    let mut results = Vec::new();
    let mut adjusted = 0usize;

    for &sk in shell_keys {
        let edges = crate::heal::edge_tolerance::collect_shell_edges(reg, sk);
        for ek in edges {
            if let Some(edge) = reg.edges.get(ek) {
                let face_keys: Vec<FaceKey> = edge.pcurves.keys().copied().collect();
                for fk in face_keys {
                    if let Some(result) = reparam_one_edge(reg, ek, fk, tolerance, max_iterations) {
                        if result.max_deviation_after < result.max_deviation_before {
                            adjusted += 1;
                        }
                        results.push(result);
                    }
                }
            }
        }
    }

    (results, adjusted)
}

fn reparam_one_edge(
    reg: &mut BRepStore,
    ek: EdgeKey,
    fk: FaceKey,
    tolerance: f32,
    max_iterations: usize,
) -> Option<ReparamResult> {
    let edge = reg.edges.get(ek)?;
    let (pcurve, same_sense) = edge.pcurves.get(&fk)?.clone();
    let face = reg.faces.get(fk)?;
    let surface = &face.surface;
    let curve_3d = &edge.curve;

    let sample_count = 32usize;
    let before = max_deviation(curve_3d, &pcurve, surface, sample_count);

    if before <= tolerance {
        return Some(ReparamResult {
            edge_key: ek, face_key: fk,
            max_deviation_before: before, max_deviation_after: before,
            converged: true, iterations: 0,
        });
    }

    let mut current_pc = pcurve;
    let mut converged = false;
    let mut iters = 0usize;

    for i in 0..max_iterations {
        iters = i + 1;
        let samples = sample_deviations(curve_3d, &current_pc, surface, sample_count);
        if let Some(adjusted) = adjust_pcurve(&current_pc, &samples, surface) {
            current_pc = adjusted;
        } else {
            break;
        }
        let after = max_deviation(curve_3d, &current_pc, surface, sample_count);
        if after <= tolerance {
            converged = true;
            break;
        }
    }

    let after = max_deviation(curve_3d, &current_pc, surface, sample_count);
    if let Some(edge_mut) = reg.edges.get_mut(ek) {
        edge_mut.pcurves.insert(fk, (current_pc, same_sense));
    }

    Some(ReparamResult {
        edge_key: ek, face_key: fk,
        max_deviation_before: before, max_deviation_after: after,
        converged, iterations: iters,
    })
}

struct DeviationSample {
    #[allow(dead_code)] t: f32,
    pt_3d: Vec3,
    #[allow(dead_code)] uv: (f32, f32),
    dev: f32,
}

#[allow(dead_code)]
fn sample_deviations(
    curve_3d: &CurveGeom, pcurve: &Curve2d, surface: &SurfaceGeom, n: usize,
) -> Vec<DeviationSample> {
    (0..=n).map(|i| {
        let t = i as f32 / n as f32;
        let pt_3d = curve_3d.d0(t);
        let uv = pcurve.d0(t);
        let on_surf = surface.d0_native(uv.0, uv.1);
        DeviationSample { t, pt_3d, uv, dev: (on_surf - pt_3d).length() }
    }).collect()
}

fn max_deviation(curve_3d: &CurveGeom, pcurve: &Curve2d, surface: &SurfaceGeom, n: usize) -> f32 {
    sample_deviations(curve_3d, pcurve, surface, n)
        .iter().map(|s| s.dev).fold(0.0f32, f32::max)
}

fn adjust_pcurve(
    pcurve: &Curve2d, samples: &[DeviationSample], surface: &SurfaceGeom,
) -> Option<Curve2d> {
    match pcurve {
        Curve2d::Line { .. } => adjust_line_pcurve(samples, surface),
        Curve2d::Polyline { .. } => adjust_polyline_pcurve(pcurve, samples, surface),
        Curve2d::BSpline { .. } => adjust_bspline_pcurve(pcurve, samples, surface),
        Curve2d::Circle { .. } | Curve2d::Ellipse { .. } => {
            adjust_circle_pcurve(samples, surface)
        }
        Curve2d::Trimmed { .. } | Curve2d::Composite { .. } => {
            adjust_composite_pcurve(samples, surface)
        }
    }
}

fn adjust_line_pcurve(samples: &[DeviationSample], surface: &SurfaceGeom) -> Option<Curve2d> {
    let first = &samples[0];
    let last = &samples[samples.len() - 1];
    let uv_start = surface.project(first.pt_3d)?;
    let uv_end = surface.project(last.pt_3d)?;
    let dir = (uv_end.0 - uv_start.0, uv_end.1 - uv_start.1);
    if dir.0.abs() < 1e-12 && dir.1.abs() < 1e-12 { return None; }
    Some(Curve2d::Line { origin: uv_start, direction: dir })
}

fn adjust_polyline_pcurve(
    pcurve: &Curve2d, samples: &[DeviationSample], surface: &SurfaceGeom,
) -> Option<Curve2d> {
    let points = match pcurve {
        Curve2d::Polyline { points } => points,
        _ => return None,
    };
    if points.len() < 2 || samples.is_empty() { return None; }
    let step = (samples.len().saturating_sub(1) / points.len().saturating_sub(1)).max(1);
    let mut new_pts = Vec::with_capacity(points.len());
    for (i, &orig) in points.iter().enumerate() {
        let si = (i * step).min(samples.len() - 1);
        new_pts.push(surface.project(samples[si].pt_3d).unwrap_or(orig));
    }
    Some(Curve2d::Polyline { points: new_pts })
}

fn adjust_bspline_pcurve(
    pcurve: &Curve2d, samples: &[DeviationSample], surface: &SurfaceGeom,
) -> Option<Curve2d> {
    let (degree, control_points, knots, weights) = match pcurve {
        Curve2d::BSpline { degree, control_points, knots, weights } =>
            (*degree, control_points, knots, weights),
        _ => return None,
    };
    let n = control_points.len();
    if n < 2 { return None; }
    let first_3d = &samples[0].pt_3d;
    let last_3d = &samples[samples.len() - 1].pt_3d;
    let uv_first = surface.project(*first_3d)?;
    let uv_last = surface.project(*last_3d)?;
    let mut new_cps = control_points.clone();
    new_cps[0] = uv_first;
    new_cps[n - 1] = uv_last;
    let step = (samples.len().saturating_sub(1) / n.saturating_sub(1)).max(1);
    for i in 1..n - 1 {
        let si = (i * step).min(samples.len() - 1);
        if let Some(uv) = surface.project(samples[si].pt_3d) {
            new_cps[i] = ((new_cps[i].0 + uv.0) * 0.5, (new_cps[i].1 + uv.1) * 0.5);
        }
    }
    Some(Curve2d::BSpline { degree, control_points: new_cps, knots: knots.clone(), weights: weights.clone() })
}

fn adjust_circle_pcurve(samples: &[DeviationSample], surface: &SurfaceGeom) -> Option<Curve2d> {
    let n = 16usize;
    let pts: Vec<(f32, f32)> = (0..=n).filter_map(|i| {
        let t = i as f32 / n as f32;
        let si = (t * (samples.len() - 1) as f32) as usize;
        surface.project(samples[si.min(samples.len() - 1)].pt_3d)
    }).collect();
    if pts.len() < 2 { return None; }
    Some(Curve2d::Polyline { points: pts })
}

fn adjust_composite_pcurve(samples: &[DeviationSample], surface: &SurfaceGeom) -> Option<Curve2d> {
    let pts: Vec<(f32, f32)> = samples.iter()
        .filter_map(|s| surface.project(s.pt_3d)).collect();
    if pts.len() < 2 { return None; }
    Some(Curve2d::Polyline { points: pts })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::geom::CurveGeom;
    use crate::geom::SurfaceGeom;
    use crate::geom::curve2d::Curve2d;
    use crate::store::BRepStore;
    use crate::topo::{BRepEdge, BRepFace, BRepShell, BRepWire, Orientation, ShellKey};

    fn make_planar_shell(reg: &mut BRepStore, uv_shift: f32) -> (ShellKey, EdgeKey, FaceKey) {
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let fk = reg.faces.insert(BRepFace {
            surface, outer_wire: wire, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![],
            color: None, degenerated_edges: vec![],
        });
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-6);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-6);
        let mut pcurves = HashMap::new();
        // PCurve is shifted: UV(uv_shift, 0) instead of UV(0, 0)
        // surface.d0_native(uv_shift, 0) = (uv_shift, 0, 0) ≠ curve3d.d0(0) = (0, 0, 0)
        pcurves.insert(fk, (Curve2d::Line { origin: (uv_shift, 0.0), direction: (1.0, 0.0) }, true));
        let ek = reg.edges.insert(BRepEdge {
            v_low: v0, v_high: v1,
            curve: CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X },
            tolerance: 1e-6, t_min: 0.0, t_max: 1.0, pcurves,
        });
        reg.wires.get_mut(wire).unwrap().edges.push((ek, Orientation::Forward));
        let sk = reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)], closed: false, step_id: None,
        });
        (sk, ek, fk)
    }

    #[test]
    fn test_reparam_line_pcurve_alignment() {
        let mut reg = BRepStore::new();
        let (sk, ek, _) = make_planar_shell(&mut reg, 0.05);
        let (results, adjusted) = same_parameter_reparam(&mut reg, &[sk], 1e-4, 5);
        assert!(adjusted > 0, "should adjust misaligned PCurve");
        let result = results.iter().find(|r| r.edge_key == ek).unwrap();
        assert!(result.max_deviation_after < result.max_deviation_before,
            "after={:.6} < before={:.6}", result.max_deviation_after, result.max_deviation_before);
    }

    #[test]
    fn test_reparam_already_aligned() {
        let mut reg = BRepStore::new();
        let (sk, ek, _) = make_planar_shell(&mut reg, 0.0);
        let (results, _) = same_parameter_reparam(&mut reg, &[sk], 1e-4, 5);
        let result = results.iter().find(|r| r.edge_key == ek).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }
}
