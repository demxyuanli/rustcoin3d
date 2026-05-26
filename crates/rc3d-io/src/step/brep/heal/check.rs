//! Minimal BRepCheck topology validation (OCC BRepCheck_Analyzer subset).

use std::collections::HashMap;

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{EdgeKey, FaceKey, ShellKey};

#[derive(Debug, Clone, Default)]
pub struct CheckReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    /// Faces that failed validation and should not be meshed.
    pub failed_faces: Vec<FaceKey>,
}

impl CheckReport {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn merge(&mut self, other: CheckReport) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}

/// Validate shell topology before meshing.
pub fn check_shell(shell_key: ShellKey, reg: &BRepRegistry) -> CheckReport {
    let mut report = CheckReport::default();
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => {
            report.errors.push(format!("shell {:?} not found", shell_key));
            return report;
        }
    };

    for &(face_key, _) in &shell.faces {
        check_face(face_key, reg, &mut report);
    }

    let nm_warnings = check_non_manifold(&shell.faces, reg);
    report.warnings.extend(nm_warnings);

    for &(face_key, _) in &shell.faces {
        let si_warnings = check_uv_self_intersection(face_key, reg);
        report.warnings.extend(si_warnings);
    }

    report
}

fn check_face(face_key: FaceKey, reg: &BRepRegistry, report: &mut CheckReport) {
    let errors_before = report.errors.len();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => {
            report.errors.push(format!("face {:?} not found", face_key));
            return;
        }
    };

    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => {
            report
                .errors
                .push(format!("face {:?} missing outer wire", face_key));
            return;
        }
    };

    if wire.edges.is_empty() && face.seam_edges.is_empty() {
        report.errors.push(format!("face {:?} has zero area (empty wire, no seams)", face_key));
        return;
    }

    let mut first_v = None;
    let mut last_v = None;
    for &(ek, orient) in &wire.edges {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => {
                report.errors.push(format!("face {:?} wire references missing edge", face_key));
                continue;
            }
        };
        let (v_start, v_end) = if orient == crate::step::brep::topo::Orientation::Forward {
            (edge.v_low, edge.v_high)
        } else {
            (edge.v_high, edge.v_low)
        };
        if first_v.is_none() {
            first_v = Some(v_start);
        }
        if let Some(lv) = last_v {
            if lv != v_start {
                report.errors.push(format!(
                    "face {:?} outer wire not closed at edge {:?}",
                    face_key, ek
                ));
            }
        }
        last_v = Some(v_end);

        if edge.pcurves.get(&face_key).is_none() && !face.seam_edges.contains(&ek) {
            report.warnings.push(format!(
                "face {:?} edge {:?} has no PCURVE",
                face_key, ek
            ));
        }
    }

    if let (Some(fv), Some(lv)) = (first_v, last_v) {
        if fv != lv {
            let gap = vertex_gap(fv, lv, reg);
            let tol = face.tolerance.max(1e-4);
            if gap > tol {
                report.errors.push(format!(
                    "face {:?} outer wire open: gap {:.6}",
                    face_key, gap
                ));
            } else {
                report.warnings.push(format!(
                    "face {:?} outer wire not closed (gap {:.6} within tol {:.6})",
                    face_key, gap, tol
                ));
            }
        }
    }

    for &seam_ek in &face.seam_edges {
        let edge = match reg.edges.get(seam_ek) {
            Some(e) => e,
            None => {
                report.errors.push(format!(
                    "face {:?} seam edge {:?} missing",
                    face_key, seam_ek
                ));
                continue;
            }
        };
        if edge.v_low != edge.v_high {
            report.warnings.push(format!(
                "face {:?} seam edge {:?} is not closed (v_low != v_high)",
                face_key, seam_ek
            ));
        }
    }

    // Edge tolerance checks
    for &(ek, _) in &wire.edges {
        report.warnings.extend(check_edge_tolerance(ek, reg));
    }

    // Surface singularity detection
    report.warnings.extend(check_surface_singularities(face_key, reg));

    // Parameter range validity
    report.warnings.extend(check_parameter_range(face_key, reg));

    // Wire orientation consistency
    report.warnings.extend(check_wire_orientation(face_key, reg));

    if report.errors.len() > errors_before {
        report.failed_faces.push(face_key);
    }
}

fn vertex_gap(
    a: crate::step::brep::topo::VertexKey,
    b: crate::step::brep::topo::VertexKey,
    reg: &BRepRegistry,
) -> f32 {
    let pa = reg.vertices.get(a).map(|v| v.position);
    let pb = reg.vertices.get(b).map(|v| v.position);
    match (pa, pb) {
        (Some(a), Some(b)) => (a - b).length(),
        _ => f32::MAX,
    }
}

fn check_non_manifold(
    face_keys: &[(FaceKey, crate::step::brep::topo::Orientation)],
    reg: &BRepRegistry,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let mut edge_face_count: HashMap<EdgeKey, usize> = HashMap::new();
    for &(face_key, _) in face_keys {
        let Some(face) = reg.faces.get(face_key) else {
            continue;
        };
        for wire_key in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
            let Some(wire) = reg.wires.get(*wire_key) else {
                continue;
            };
            for &(ek, _) in &wire.edges {
                *edge_face_count.entry(ek).or_default() += 1;
            }
        }
    }
    for (ek, count) in edge_face_count {
        if count > 2 {
            warnings.push(format!(
                "edge {:?} is non-manifold (shared by {} faces)",
                ek, count
            ));
        }
    }
    warnings
}

fn check_uv_self_intersection(face_key: FaceKey, reg: &BRepRegistry) -> Vec<String> {
    let mut warnings = Vec::new();
    let Some(face) = reg.faces.get(face_key) else {
        return warnings;
    };
    let Some(wire) = reg.wires.get(face.outer_wire) else {
        return warnings;
    };
    let mut segments: Vec<((f32, f32), (f32, f32))> = Vec::new();
    for &(ek, _) in &wire.edges {
        let Some(edge) = reg.edges.get(ek) else {
            continue;
        };
        let Some(pcurve) = edge.pcurves.get(&face_key) else {
            continue;
        };
        let p0 = pcurve.d0(0.0);
        let p1 = pcurve.d0(1.0);
        segments.push(((p0.x, p0.y), (p1.x, p1.y)));
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
    a: ((f32, f32), (f32, f32)),
    b: ((f32, f32), (f32, f32)),
) -> bool {
    let ((ax0, ay0), (ax1, ay1)) = a;
    let ((bx0, by0), (bx1, by1)) = b;
    // Scale epsilon to coordinate magnitude for f32 validity
    let extent = (ax0.abs() + ax1.abs() + bx0.abs() + bx1.abs()
        + ay0.abs() + ay1.abs() + by0.abs() + by1.abs()) / 8.0;
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

/// Check edge tolerance validity (OCC BRepCheck_Edge).
fn check_edge_tolerance(ek: EdgeKey, reg: &BRepRegistry) -> Vec<String> {
    let mut warnings = Vec::new();
    let edge = match reg.edges.get(ek) {
        Some(e) => e,
        None => return warnings,
    };

    let p0 = reg.vertices.get(edge.v_low).map(|v| v.position);
    let p1 = reg.vertices.get(edge.v_high).map(|v| v.position);
    let approx_len = match (p0, p1) {
        (Some(a), Some(b)) => (a - b).length(),
        _ => return warnings,
    };

    if approx_len < 1e-12 {
        return warnings;
    }

    if edge.tolerance > approx_len * 10.0 {
        warnings.push(format!(
            "edge {:?}: tolerance {:.6} > 10x edge length {:.6}",
            ek, edge.tolerance, approx_len
        ));
    }
    if edge.tolerance < 1e-12 {
        warnings.push(format!("edge {:?}: tolerance {:.6} is near-zero", ek, edge.tolerance));
    }

    warnings
}

/// Check for surface singularities on the trim boundary (OCC BRepCheck_Face).
fn check_surface_singularities(
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return warnings,
    };

    match &face.surface {
        crate::step::brep::geom::SurfaceGeom::Sphere { .. } => {
            let wire = match reg.wires.get(face.outer_wire) {
                Some(w) => w,
                None => return warnings,
            };
            for &(ek, _) in &wire.edges {
                let edge = match reg.edges.get(ek) {
                    Some(e) => e,
                    None => continue,
                };
                if let Some(pc) = edge.pcurves.get(&face_key) {
                    for t in [0.0, 1.0] {
                        let uv = pc.d0(t);
                        if (uv.y.abs() - std::f32::consts::FRAC_PI_2).abs() < 0.01 {
                            warnings.push(format!(
                                "face {:?}: potential degeneracy near sphere pole at u={:.3}, v={:.3}",
                                face_key, uv.x, uv.y
                            ));
                        }
                    }
                }
            }
        }
        crate::step::brep::geom::SurfaceGeom::Cone { .. } => {
            let wire = match reg.wires.get(face.outer_wire) {
                Some(w) => w,
                None => return warnings,
            };
            for &(ek, _) in &wire.edges {
                let edge = match reg.edges.get(ek) {
                    Some(e) => e,
                    None => continue,
                };
                for &vk in &[edge.v_low, edge.v_high] {
                    if let Some(v) = reg.vertices.get(vk) {
                        if let crate::step::brep::geom::SurfaceGeom::Cone { apex, .. } = &face.surface {
                            if (v.position - *apex).length() < face.tolerance * 10.0 {
                                warnings.push(format!(
                                    "face {:?}: potential degeneracy at cone apex", face_key
                                ));
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }

    warnings
}

/// Check that PCurve UV coordinates fall within the surface's natural domain.
fn check_parameter_range(
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return warnings,
    };

    let (u_range, v_range): ((f32, f32), (f32, f32)) = match &face.surface {
        crate::step::brep::geom::SurfaceGeom::BSpline(nurbs) => {
            let uk = &nurbs.knots_u;
            let vk = &nurbs.knots_v;
            ((uk[0], uk[uk.len() - 1]), (vk[0], vk[vk.len() - 1]))
        }
        _ => ((0.0, 0.0), (0.0, 0.0)),
    };

    if u_range == (0.0, 0.0) && v_range == (0.0, 0.0) {
        return warnings;
    }

    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => return warnings,
    };
    for &(ek, _) in &wire.edges {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };
        if let Some(pc) = edge.pcurves.get(&face_key) {
            for t in [0.0, 1.0] {
                let uv = pc.d0(t);
                let margin = 0.1;
                if uv.x < u_range.0 - margin || uv.x > u_range.1 + margin {
                    warnings.push(format!(
                        "face {:?} edge {:?}: PCurve U={:.6} outside surface U range [{:.3}, {:.3}]",
                        face_key, ek, uv.x, u_range.0, u_range.1
                    ));
                }
                if uv.y < v_range.0 - margin || uv.y > v_range.1 + margin {
                    warnings.push(format!(
                        "face {:?} edge {:?}: PCurve V={:.6} outside surface V range [{:.3}, {:.3}]",
                        face_key, ek, uv.y, v_range.0, v_range.1
                    ));
                }
            }
        }
    }

    warnings
}

/// Check wire orientation consistency using signed area in UV space.
fn check_wire_orientation(
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return warnings,
    };

    if let Some(wire) = reg.wires.get(face.outer_wire) {
        let uv_points = collect_wire_uv_polygon(wire, face_key, reg);
        if uv_points.len() >= 3 {
            let area = signed_area_2d(&uv_points);
            let expected_positive = face.same_sense;
            if (area > 0.0) != expected_positive {
                warnings.push(format!(
                    "face {:?}: outer wire orientation inconsistent (signed_area={:.6}, same_sense={})",
                    face_key, area, expected_positive
                ));
            }
        }
    }

    for &inner_wire_key in &face.inner_wires {
        if let Some(wire) = reg.wires.get(inner_wire_key) {
            let uv_points = collect_wire_uv_polygon(wire, face_key, reg);
            if uv_points.len() >= 3 {
                let area = signed_area_2d(&uv_points);
                let outer_positive = face.same_sense;
                if (area > 0.0) == outer_positive {
                    warnings.push(format!(
                        "face {:?}: inner wire orientation should be opposite of outer (signed_area={:.6})",
                        face_key, area
                    ));
                }
            }
        }
    }

    warnings
}

fn collect_wire_uv_polygon(
    wire: &crate::step::brep::topo::BRepWire,
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Vec<(f32, f32)> {
    let mut points = Vec::new();
    for &(ek, _) in &wire.edges {
        if let Some(edge) = reg.edges.get(ek) {
            if let Some(pc) = edge.pcurves.get(&face_key) {
                let uv = pc.d0(0.0);
                points.push((uv.x, uv.y));
            }
        }
    }
    if points.len() >= 2 {
        if let Some(last_edge) = wire.edges.last().and_then(|&(ek, _)| {
            reg.edges.get(ek).and_then(|e| e.pcurves.get(&face_key)).map(|pc| {
                let uv = pc.d0(1.0);
                (uv.x, uv.y)
            })
        }) {
            points.push(last_edge);
        }
    }
    points
}

fn signed_area_2d(pts: &[(f32, f32)]) -> f32 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].0 * pts[j].1;
        area -= pts[j].0 * pts[i].1;
    }
    area * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::topo::{BRepEdge, BRepFace, BRepShell, BRepWire, Orientation};
    use rc3d_core::math::Vec3;

    fn closed_cube_shell(reg: &mut BRepRegistry) -> ShellKey {
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
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
        });
        let edges_data = [
            (Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), (0.0, 0.0), (10.0, 0.0)),
            (Vec3::new(10.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
            (Vec3::new(10.0, 10.0, 0.0), Vec3::new(0.0, 10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            (Vec3::new(0.0, 10.0, 0.0), Vec3::ZERO, (0.0, 10.0), (0.0, 0.0)),
        ];
        let mut wire_edges = Vec::new();
        for (a, b, u0, u1) in edges_data {
            let v0 = reg.find_or_add_vertex(a, 1e-4);
            let v1 = reg.find_or_add_vertex(b, 1e-4);
            let curve_3d = CurveGeom::Line { origin: a, direction: b - a };
            let pcurve = CurveGeom::Line {
                origin: Vec3::new(u0.0, u0.1, 0.0),
                direction: Vec3::new(u1.0 - u0.0, u1.1 - u0.1, 0.0),
            };
            let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve);
            let edge = reg.edges.get(ek).unwrap();
            let orient = if edge.v_low == v0 {
                Orientation::Forward
            } else {
                Orientation::Reversed
            };
            wire_edges.push((ek, orient));
        }
        let outer_wire = reg.wires.insert(BRepWire { edges: wire_edges });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = outer_wire;
        }
        reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: true,
            step_id: None,
        })
    }

    #[test]
    fn valid_plane_face_has_no_errors() {
        let mut reg = BRepRegistry::new();
        let sk = closed_cube_shell(&mut reg);
        let report = check_shell(sk, &reg);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
    }

    #[test]
    fn open_wire_reports_error() {
        let mut reg = BRepRegistry::new();
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-6,
            seam_edges: vec![],
            color: None,
        });
        let a = Vec3::ZERO;
        let b = Vec3::new(10.0, 0.0, 0.0);
        let c = Vec3::new(10.0, 10.0, 0.0);
        let va = reg.find_or_add_vertex(a, 1e-4);
        let vb = reg.find_or_add_vertex(b, 1e-4);
        let vc = reg.find_or_add_vertex(c, 1e-4);
        let e0 = reg.add_edge_with_pcurve(
            va,
            vb,
            CurveGeom::Line { origin: a, direction: b - a },
            1e-4,
            face_key,
            CurveGeom::Line {
                origin: Vec3::ZERO,
                direction: Vec3::new(10.0, 0.0, 0.0),
            },
        );
        let e1 = reg.add_edge_with_pcurve(
            vb,
            vc,
            CurveGeom::Line { origin: b, direction: c - b },
            1e-4,
            face_key,
            CurveGeom::Line {
                origin: Vec3::new(10.0, 0.0, 0.0),
                direction: Vec3::new(0.0, 10.0, 0.0),
            },
        );
        let outer = reg.wires.insert(BRepWire {
            edges: vec![(e0, Orientation::Forward), (e1, Orientation::Forward)],
        });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = outer;
        }
        let sk = reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let report = check_shell(sk, &reg);
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn zero_area_face_yields_error() {
        let mut reg = BRepRegistry::new();
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
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
        });
        let shell_key = reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let report = check_shell(shell_key, &reg);
        assert!(
            !report.errors.is_empty(),
            "zero-area face (empty wire with no seam edges) should be an error"
        );
    }

    #[test]
    fn non_manifold_edge_yields_warning() {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let curve = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let ek = reg.edges.insert(BRepEdge {
            v_low: v0,
            v_high: v1,
            curve: curve.clone(),
            tolerance: 1e-4,
            pcurves: HashMap::new(),
        });
        // Create 3 faces sharing the same edge
        let mut face_keys = Vec::new();
        for _ in 0..3 {
            let wire = reg.wires.insert(BRepWire {
                edges: vec![(ek, Orientation::Forward)],
            });
            face_keys.push(reg.faces.insert(BRepFace {
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
            }));
        }
        let faces: Vec<_> = face_keys
            .iter()
            .map(|&fk| (fk, Orientation::Forward))
            .collect();
        let shell_key = reg.shells.insert(BRepShell {
            faces,
            closed: false,
            step_id: None,
        });
        let report = check_shell(shell_key, &reg);
        assert!(
            !report.warnings.is_empty(),
            "edge shared by 3 faces should be flagged non-manifold"
        );
    }
}
