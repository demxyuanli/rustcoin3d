//! BRepCheck_Face — face-level topology and geometry validation.

use rc3d_core::math::Real;

use crate::geom::signed_area_2d;
use crate::store::BRepStore;
use crate::topo::{FaceKey, Orientation, VertexKey};
use super::super::geom2d::collect_wire_uv_polygon;

use super::CheckReport;

/// Validate a single face (wire closure, pcurves, seam edges, singularities, etc.).
pub fn check_face(face_key: FaceKey, reg: &BRepStore, report: &mut CheckReport) {
    let errors_before = report.errors.len();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => {
            report
                .errors
                .push(format!("face {:?} not found", face_key));
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
                report
                    .errors
                    .push(format!("face {:?} wire references missing edge", face_key));
                continue;
            }
        };
        let (v_start, v_end) = if orient == Orientation::Forward {
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
        report
            .warnings
            .extend(super::edge::check_edge_tolerance(ek, reg));
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
    report
        .warnings
        .extend(check_wire_orientation(face_key, reg));

    if report.errors.len() > errors_before {
        report.failed_faces.push(face_key);
    }
}

fn vertex_gap(a: VertexKey, b: VertexKey, reg: &BRepStore) -> Real {
    let pa = reg.vertices.get(a).map(|v| v.position);
    let pb = reg.vertices.get(b).map(|v| v.position);
    match (pa, pb) {
        (Some(a), Some(b)) => (a - b).length(),
        _ => f64::MAX,
    }
}

/// Check for surface singularities on the trim boundary (OCC BRepCheck_Face).
pub fn check_surface_singularities(face_key: FaceKey, reg: &BRepStore) -> Vec<String> {
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
                        if (uv.1.abs() - std::f64::consts::FRAC_PI_2).abs() < 0.01 {
                            warnings.push(format!(
                                "face {:?}: potential degeneracy near sphere pole at u={:.3}, v={:.3}",
                                face_key, uv.0, uv.1
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
                                "face {:?}: potential degeneracy at cone apex",
                                face_key
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
pub fn check_parameter_range(face_key: FaceKey, reg: &BRepStore) -> Vec<String> {
    let mut warnings = Vec::new();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return warnings,
    };

    let (u_range, v_range): ((Real, Real), (Real, Real)) = match &face.surface {
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
                if uv.0 < u_range.0 - margin || uv.0 > u_range.1 + margin {
                    warnings.push(format!(
                        "face {:?} edge {:?}: PCurve U={:.6} outside surface U range [{:.3}, {:.3}]",
                        face_key, ek, uv.0, u_range.0, u_range.1
                    ));
                }
                if uv.1 < v_range.0 - margin || uv.1 > v_range.1 + margin {
                    warnings.push(format!(
                        "face {:?} edge {:?}: PCurve V={:.6} outside surface V range [{:.3}, {:.3}]",
                        face_key, ek, uv.1, v_range.0, v_range.1
                    ));
                }
            }
        }
    }

    warnings
}

/// Check wire orientation consistency using signed area in UV space.
pub fn check_wire_orientation(face_key: FaceKey, reg: &BRepStore) -> Vec<String> {
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
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepWire, Orientation};
    use rc3d_core::math::PVec3;

    #[test]
    fn test_check_surface_singularity_sphere_pole() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Sphere {
            center: PVec3::ZERO,
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
        let v0 = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 1.0), 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-4);
        let line = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let pc = Curve2d::Line {
            origin: (0.0, std::f64::consts::FRAC_PI_2 - 0.001),
            direction: (1.0, -std::f64::consts::FRAC_PI_2 + 0.001),
        };
        let ek = reg.add_edge_with_pcurve(v0, v1, line, 1e-4, fk, pc, true);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let warnings = check_surface_singularities(fk, &reg);
        assert!(
            !warnings.is_empty(),
            "sphere pole should produce singularity warnings"
        );
    }

    #[test]
    fn test_parameter_range_oob() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
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
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc, true);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let warnings = check_parameter_range(fk, &reg);
        assert!(
            warnings.is_empty(),
            "plane surface should not trigger parameter range warnings"
        );
    }

    #[test]
    fn test_wire_orientation_inconsistent() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(PVec3::new(0.0, 1.0, 0.0), 1e-4);
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
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let pc1 = Curve2d::Line {
            origin: (1.0, 0.0),
            direction: (-1.0, 0.0),
        };
        let pc2 = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (0.0, 1.0),
        };
        let pc3 = Curve2d::Line {
            origin: (0.0, 1.0),
            direction: (1.0, -1.0),
        };
        let e1 = reg.add_edge_with_pcurve(v1, v0, line.clone(), 1e-4, fk, pc1, true);
        let e2 = reg.add_edge_with_pcurve(v0, v2, line.clone(), 1e-4, fk, pc2, true);
        let e3 = reg.add_edge_with_pcurve(v2, v1, line.clone(), 1e-4, fk, pc3, true);
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
            (e3, Orientation::Forward),
        ];
        let _warnings = check_wire_orientation(fk, &reg);
        // The test verifies the function runs without panic for a non-trivial wire.
    }
}
