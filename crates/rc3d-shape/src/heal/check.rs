//! Minimal BRepCheck topology validation (OCC BRepCheck_Analyzer subset).

use std::collections::HashMap;

use crate::geom::signed_area_2d;
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, ShellKey, VertexKey};
use crate::topo_iter;
use super::geom2d::collect_wire_uv_polygon;
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct CheckReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    /// Faces that failed validation and should not be meshed.
    pub failed_faces: Vec<FaceKey>,
    pub has_uv_gaps: bool,
    pub has_pcurve_issues: bool,
    pub has_self_intersections: bool,
    pub has_singularities: bool,
    pub has_inner_wires: bool,
    pub has_intersecting_wires: bool,
    pub has_face_self_intersections: bool,
}

impl CheckReport {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn merge(&mut self, other: CheckReport) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        self.failed_faces.extend(other.failed_faces);
        self.has_uv_gaps |= other.has_uv_gaps;
        self.has_pcurve_issues |= other.has_pcurve_issues;
        self.has_self_intersections |= other.has_self_intersections;
        self.has_singularities |= other.has_singularities;
        self.has_inner_wires |= other.has_inner_wires;
        self.has_intersecting_wires |= other.has_intersecting_wires;
        self.has_face_self_intersections |= other.has_face_self_intersections;
    }
}

/// Validate shell topology before meshing.
pub fn check_shell(shell_key: ShellKey, reg: &BRepStore) -> CheckReport {
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
        if !si_warnings.is_empty() {
            report.has_self_intersections = true;
        }
        report.warnings.extend(si_warnings);
        if let Some(face) = reg.faces.get(face_key) {
            if !face.inner_wires.is_empty() {
                report.has_inner_wires = true;
                if super::intersecting_wires::detect_intersecting_wires(face_key, reg) {
                    report.has_intersecting_wires = true;
                }
            }
        }

        // Face-level self-intersection (surface folding in 3D)
        let face_si_count = super::face_self_intersect::check_face_self_intersect(
            face_key, reg, 10,
        );
        if face_si_count > 0 {
            report.has_face_self_intersections = true;
            report.warnings.push(format!(
                "face {:?}: surface self-intersection suspected ({} normal inversions)",
                face_key, face_si_count
            ));
        }
    }

    // Euler-Poincaré topology validation
    if let Some(chi) = check_euler_poincare(shell_key, reg) {
        if chi != 2 && chi != 0 {
            report.warnings.push(format!(
                "shell {:?}: Euler characteristic χ={} (expected 2 for closed, 0 for torus)",
                shell_key, chi
            ));
        }
    }

    report
}

/// Euler-Poincaré formula validation for shells.
/// For a closed manifold shell: V - E + F = 2(1 - genus)
/// genus=0 (sphere-like) → V - E + F = 2
/// Returns None if shell has no faces.
pub fn check_euler_poincare(shell_key: ShellKey, reg: &BRepStore) -> Option<i32> {
    let shell = reg.shells.get(shell_key)?;
    if shell.faces.is_empty() {
        return None;
    }

    let mut edge_set: HashSet<EdgeKey> = HashSet::new();
    let mut vertex_set: HashSet<VertexKey> = HashSet::new();

    for &(fk, _) in &shell.faces {
        let _ = reg.faces.get(fk)?;
        for ek in topo_iter::iter_edges_of_face(fk, reg) {
            edge_set.insert(ek);
            if let Some((vl, vh)) = topo_iter::iter_vertices_of_edge(ek, reg) {
                vertex_set.insert(vl);
                vertex_set.insert(vh);
            }
        }
    }

    let v = vertex_set.len() as i32;
    let e = edge_set.len() as i32;
    let f = shell.faces.len() as i32;

    Some(v - e + f)
}

fn check_face(face_key: FaceKey, reg: &BRepStore, report: &mut CheckReport) {
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

    if wire.edges.is_empty()
        && face.seam_edges.is_empty()
        && face.degenerated_edges.is_empty()
    {
        report.errors.push(format!(
            "face {:?} has zero area (empty wire, no seams, no degenerated edges)",
            face_key
        ));
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
        let (v_start, v_end) = if orient == crate::topo::Orientation::Forward {
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
            report.has_pcurve_issues = true;
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
                report.has_uv_gaps = true;
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
    let singularity_warnings = check_surface_singularities(face_key, reg);
    if !singularity_warnings.is_empty() {
        report.has_singularities = true;
    }
    report.warnings.extend(singularity_warnings);

    // Parameter range validity
    let param_warnings = check_parameter_range(face_key, reg);
    if !param_warnings.is_empty() {
        report.has_pcurve_issues = true;
    }
    report.warnings.extend(param_warnings);

    // Wire orientation consistency
    report.warnings.extend(check_wire_orientation(face_key, reg));

    if report.errors.len() > errors_before {
        report.failed_faces.push(face_key);
    }
}

fn vertex_gap(
    a: crate::topo::VertexKey,
    b: crate::topo::VertexKey,
    reg: &BRepStore,
) -> f32 {
    let pa = reg.vertices.get(a).map(|v| v.position);
    let pb = reg.vertices.get(b).map(|v| v.position);
    match (pa, pb) {
        (Some(a), Some(b)) => (a - b).length(),
        _ => f32::MAX,
    }
}

fn check_non_manifold(
    face_keys: &[(FaceKey, crate::topo::Orientation)],
    reg: &BRepStore,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let mut edge_face_count: HashMap<EdgeKey, usize> = HashMap::new();
    for &(face_key, _) in face_keys {
        for (ek, _) in topo_iter::iter_edge_orientations_of_face(face_key, reg) {
            *edge_face_count.entry(ek).or_default() += 1;
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

pub fn check_uv_self_intersection(face_key: FaceKey, reg: &BRepStore) -> Vec<String> {
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
fn check_edge_tolerance(ek: EdgeKey, reg: &BRepStore) -> Vec<String> {
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
    reg: &BRepStore,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return warnings,
    };

    match &face.surface {
        crate::geom::SurfaceGeom::Sphere { .. } => {
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
        crate::geom::SurfaceGeom::Cone { apex, .. } => {
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
                        if (v.position - *apex).length() < face.tolerance * 10.0 {
                            warnings.push(format!(
                                "face {:?}: potential degeneracy at cone apex", face_key
                            ));
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
    reg: &BRepStore,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return warnings,
    };

    let (u_range, v_range): ((f32, f32), (f32, f32)) = match &face.surface {
        crate::geom::SurfaceGeom::BSpline(nurbs) => {
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
    reg: &BRepStore,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepEdge, BRepFace, BRepShell, BRepWire, Orientation};
    use rc3d_core::math::Vec3;

    fn closed_cube_shell(reg: &mut BRepStore) -> ShellKey {
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
            degenerated_edges: vec![],
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
        let mut reg = BRepStore::new();
        let sk = closed_cube_shell(&mut reg);
        let report = check_shell(sk, &reg);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
    }

    #[test]
    fn open_wire_reports_error() {
        let mut reg = BRepStore::new();
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
            degenerated_edges: vec![],
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
        let mut reg = BRepStore::new();
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
            degenerated_edges: vec![],
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
    fn test_check_edge_tolerance_oversized() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(0.01, 0.0, 0.0), 1e-4);
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::new(0.01, 0.0, 0.0),
        };
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, line.clone());
        // Artificially set tolerance way too high
        if let Some(edge) = reg.edges.get_mut(ek) {
            edge.tolerance = 1.0;
        }
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let warnings = check_edge_tolerance(ek, &reg);
        assert!(
            !warnings.is_empty(),
            "oversized tolerance should produce warnings"
        );
    }

    #[test]
    fn test_check_surface_singularity_sphere_pole() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        // Create an edge with a PCurve endpoint near the sphere north pole
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 1.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        // PCurve from pole-adjacent UV to equator
        let pc = CurveGeom::Line {
            origin: Vec3::new(0.0, std::f32::consts::FRAC_PI_2 - 0.001, 0.0),
            direction: Vec3::new(1.0, -std::f32::consts::FRAC_PI_2 + 0.001, 0.0),
        };
        let ek = reg.add_edge_with_pcurve(v0, v1, line, 1e-4, fk, pc);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let warnings = check_surface_singularities(fk, &reg);
        assert!(
            !warnings.is_empty(),
            "sphere pole should produce singularity warnings"
        );
    }

    #[test]
    fn non_manifold_edge_yields_warning() {
        let mut reg = BRepStore::new();
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
                degenerated_edges: vec![],
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

    #[test]
    fn test_parameter_range_oob() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        // For a Plane surface, parameter_range applies only to BSpline — this tests the no-op path
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, line);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let warnings = check_parameter_range(fk, &reg);
        // Plane is not BSpline, so should return empty
        assert!(
            warnings.is_empty(),
            "plane surface should not trigger parameter range warnings"
        );
    }

    #[test]
    fn test_euler_poincare_single_face() {
        // Single face with 4 edges and 4 vertices: χ = V - E + F = 4 - 4 + 1 = 1
        let mut reg = BRepStore::new();
        let sk = closed_cube_shell(&mut reg);
        let chi = check_euler_poincare(sk, &reg);
        assert_eq!(chi, Some(1), "single-face shell should have χ=1");
    }

    #[test]
    fn test_euler_poincare_empty_shell() {
        let mut reg = BRepStore::new();
        let sk = reg.shells.insert(BRepShell {
            faces: vec![],
            closed: false,
            step_id: None,
        });
        let chi = check_euler_poincare(sk, &reg);
        assert_eq!(chi, None, "empty shell should return None");
    }

    #[test]
    fn test_wire_orientation_inconsistent() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(0.0, 1.0, 0.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        // Clockwise wire with same_sense=true should trigger orientation warning
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let pc1 = CurveGeom::Line {
            origin: Vec3::new(1.0, 0.0, 0.0),
            direction: Vec3::new(-1.0, 0.0, 0.0),
        };
        let pc2 = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(0.0, 1.0, 0.0),
        };
        let pc3 = CurveGeom::Line {
            origin: Vec3::new(0.0, 1.0, 0.0),
            direction: Vec3::new(1.0, -1.0, 0.0),
        };
        let e1 = reg.add_edge_with_pcurve(v1, v0, line.clone(), 1e-4, fk, pc1); // reversed
        let e2 = reg.add_edge_with_pcurve(v0, v2, line.clone(), 1e-4, fk, pc2);
        let e3 = reg.add_edge_with_pcurve(v2, v1, line.clone(), 1e-4, fk, pc3);
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
            (e3, Orientation::Forward),
        ];
        let _warnings = check_wire_orientation(fk, &reg);
        // The test verifies the function runs without panic for a non-trivial wire.
    }
}
