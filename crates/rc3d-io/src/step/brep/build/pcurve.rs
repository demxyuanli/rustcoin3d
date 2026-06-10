use super::*;
use super::curve::{build_bspline_2d, parse_trim_bound, resolve_cartesian_2d};

// ── PCURVE resolution ─────────────────────────────────────────────

/// Polyline sample count for synthetic / fallback PCurves (validate uses 8 samples).
const PCURVE_POLYLINE_SAMPLES: u32 = 16;

/// Resolve PCURVE for an edge: STEP first (OCCT), validated against the 3D curve;
/// fall back to synthetic projection when STEP data is missing or inconsistent.
pub fn resolve_edge_pcurve(
    curve: &CurveGeom,
    surface: &SurfaceGeom,
    edge_curve_id: u64,
    surface_id: Option<u64>,
    entities: &EntityIndex,
    tol: f32,
) -> CurveGeom {
    let step_pcs = surface_id
        .map(|sid| collect_pcurves_for_face(edge_curve_id, sid, entities))
        .unwrap_or_default();

    let match_tol = pcurve_match_tol(curve, tol);

    for (idx, pc) in step_pcs.iter().enumerate() {
        if validate_pcurve_on_surface(curve, pc, surface, match_tol, false) {
            log::trace!("[BRep] edge #{edge_curve_id} pcurve step_ok idx={idx}");
            return pc.clone();
        }
        if validate_pcurve_on_surface(curve, pc, surface, match_tol, true) {
            log::trace!("[BRep] edge #{edge_curve_id} pcurve step_reversed idx={idx}");
            return reverse_pcurve(pc);
        }
    }
    if !step_pcs.is_empty() {
        log::debug!(
            "[BRep] STEP PCURVE rejected for edge #{}, trying synthetic",
            edge_curve_id
        );
        log::trace!("[BRep] edge #{edge_curve_id} pcurve step_rejected");
    }

    if let Some(syn) = build_synthetic_pcurve(curve, surface, match_tol) {
        if validate_pcurve_on_surface(curve, &syn, surface, match_tol, false) {
            log::trace!("[BRep] edge #{edge_curve_id} pcurve synthetic_ok");
            return syn;
        }
    }

    let fb = build_parametric_fallback_pcurve(curve, surface, match_tol);
    if validate_pcurve_on_surface(curve, &fb, surface, match_tol, false) {
        log::trace!("[BRep] edge #{edge_curve_id} pcurve fallback_ok");
        return fb;
    }

    if let Some(pc) = step_pcs.first() {
        if !pcurve_uv_is_degenerate(pc) {
            log::trace!("[BRep] edge #{edge_curve_id} pcurve step_force");
            return pc.clone();
        }
    }
    log::trace!(
        "[BRep] edge #{edge_curve_id} pcurve fallback_force degenerate={}",
        pcurve_uv_is_degenerate(&fb)
    );
    fb
}

fn pcurve_match_tol(curve: &CurveGeom, tol: f32) -> f32 {
    let edge_len = (curve.d0(0.0) - curve.d0(1.0)).length();
    (tol.max(1e-4) * 10.0).max(edge_len * 0.05).max(1e-3)
}

/// Sample the 3D curve and PCurve-on-surface; reject if they diverge beyond tolerance.
fn validate_pcurve_on_surface(
    curve: &CurveGeom,
    pcurve: &CurveGeom,
    surface: &SurfaceGeom,
    match_tol: f32,
    reversed: bool,
) -> bool {
    const SAMPLES: usize = 8;
    for i in 0..=SAMPLES {
        let t = i as f32 / SAMPLES as f32;
        let p3 = curve.d0(t);
        let t_pc = if reversed { 1.0 - t } else { t };
        let uv = pcurve.d0(t_pc);
        if !pcurve_uv_matches_3d(surface, p3, uv, match_tol) {
            return false;
        }
    }
    true
}

fn pcurve_uv_native_candidates(surface: &SurfaceGeom, u: f32, v: f32) -> Vec<(f32, f32)> {
    let mut out = vec![(u, v)];
    if matches!(surface, SurfaceGeom::Revolution { .. }) {
        const TAU: f32 = std::f32::consts::TAU;
        if u <= 1.0 + 1e-4 {
            out.push((u * TAU, v));
        }
        if u >= TAU * 0.25 {
            let un = u / TAU;
            if (un - u).abs() > 1e-6 {
                out.push((un, v));
            }
        }
    }
    out
}

fn pcurve_uv_matches_3d(
    surface: &SurfaceGeom,
    p3: Vec3,
    uv: Vec3,
    match_tol: f32,
) -> bool {
    for (u, v) in pcurve_uv_native_candidates(surface, uv.x, uv.y) {
        if (p3 - surface.d0_native(u, v)).length() <= match_tol {
            return true;
        }
        if let Some(period_u) = surface.native_u_period() {
            for shift in [-1.0f32, 1.0] {
                if (p3 - surface.d0_native(u + shift * period_u, v)).length() <= match_tol {
                    return true;
                }
            }
        }
        if let Some(period_v) = surface.native_v_period() {
            for shift in [-1.0f32, 1.0] {
                if (p3 - surface.d0_native(u, v + shift * period_v)).length() <= match_tol {
                    return true;
                }
            }
        }
    }
    false
}

/// Map a 3D edge sample to native surface UV (Revolution uses analytic inverse).
fn sample_3d_to_native_uv(surface: &SurfaceGeom, p3: Vec3, inv_tol: f32) -> (f32, f32) {
    if let SurfaceGeom::Revolution { .. } = surface {
        if let Some(uv) = surface.revolution_native_uv_at(p3) {
            return uv;
        }
    }
    surface
        .inverse_native_uv_build(p3, inv_tol)
        .or_else(|| surface.project(p3))
        .unwrap_or((0.0, 0.0))
}

fn reverse_pcurve(pcurve: &CurveGeom) -> CurveGeom {
    let n = PCURVE_POLYLINE_SAMPLES;
    let mut points = Vec::with_capacity(n as usize + 1);
    for i in 0..=n {
        let t = 1.0 - i as f32 / n as f32;
        points.push(pcurve.d0(t));
    }
    CurveGeom::Polyline { points }
}

fn pcurve_uv_is_degenerate(pcurve: &CurveGeom) -> bool {
    const SAMPLES: usize = 8;
    let p0 = pcurve.d0(0.0);
    pcurve.d0(1.0);
    for i in 1..=SAMPLES {
        let t = i as f32 / SAMPLES as f32;
        if (pcurve.d0(t) - p0).length_squared() > 1e-12 {
            return false;
        }
    }
    true
}

/// Last-resort UV polyline: inverse map each 3D sample into native surface UV.
fn build_parametric_fallback_pcurve(
    curve: &CurveGeom,
    surface: &SurfaceGeom,
    match_tol: f32,
) -> CurveGeom {
    let inv_tol = (match_tol * 5.0).max(0.5);
    let n = PCURVE_POLYLINE_SAMPLES;
    let mut uv_points = Vec::with_capacity(n as usize + 1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let p3 = curve.d0(t);
        let (u, v) = sample_3d_to_native_uv(surface, p3, inv_tol);
        uv_points.push(Vec3::new(u, v, 0.0));
    }
    CurveGeom::Polyline { points: uv_points }
}

/// Preferred PCURVE list index from SURFACE_CURVE / SEAM_CURVE master_rep (.PCURVE_S1. → 0).
fn pcurve_master_list_index(sc_rec: &EntityRecord) -> usize {
    match sc_rec.params.nth_param(3) {
        Some(StepValue::Enum(s)) => {
            if s.contains("PCURVE_S2") {
                1
            } else if s.contains("PCURVE_S3") {
                2
            } else {
                0
            }
        }
        _ => 0,
    }
}

fn surface_curve_record(
    edge_curve_id: u64,
    entities: &EntityIndex,
) -> Option<&EntityRecord> {
    let edge_rec = entities.get(&edge_curve_id)?;
    let sc_id = if edge_rec.name == "EDGE_CURVE" {
        geom::nth_ref(&edge_rec.params, 3)?
    } else {
        edge_curve_id
    };
    let sc_rec = entities.get(&sc_id)?;
    if sc_rec.name != "SURFACE_CURVE"
        && sc_rec.name != "SEAM_CURVE"
        && sc_rec.name != "INTERSECTION_CURVE"
    {
        return None;
    }
    Some(sc_rec)
}

/// All PCurves on `surface_id`, master_rep first (OCC SEAM_CURVE u=0 / u=2pi sheets).
fn collect_pcurves_for_face(
    edge_curve_id: u64,
    surface_id: u64,
    entities: &EntityIndex,
) -> Vec<CurveGeom> {
    let sc_rec = match surface_curve_record(edge_curve_id, entities) {
        Some(r) => r,
        None => return Vec::new(),
    };
    let pcurve_list = match sc_rec.params.nth_param(2).and_then(|v| v.as_list()) {
        Some(list) => list,
        None => {
            if let Some(pid) = geom::nth_ref(&sc_rec.params, 2) {
                if let Some(c) = resolve_pcurve_on_surface(pid, surface_id, entities) {
                    return vec![c];
                }
            }
            return Vec::new();
        }
    };

    let mut matched = Vec::new();
    for item in pcurve_list {
        let Some(pcurve_id) = item.as_ref_id() else {
            continue;
        };
        if let Some(c) = resolve_pcurve_on_surface(pcurve_id, surface_id, entities) {
            matched.push(c);
        }
    }
    if matched.is_empty() {
        return matched;
    }
    let master = pcurve_master_list_index(sc_rec);
    if master < matched.len() && master > 0 {
        let preferred = matched.remove(master);
        matched.insert(0, preferred);
    }
    matched
}

fn resolve_pcurve_on_surface(
    pcurve_id: u64,
    surface_id: u64,
    entities: &EntityIndex,
) -> Option<CurveGeom> {
    let prec = entities.get(&pcurve_id)?;
    if prec.name != "PCURVE" && prec.name != "DEFINITIONAL_REPRESENTATION" {
        return None;
    }
    let basis_surface = geom::nth_ref(&prec.params, 1)?;
    if basis_surface != surface_id {
        return None;
    }
    let curve_ref = geom::nth_ref(&prec.params, 2)?;
    build_2d_curve(curve_ref, entities)
}

/// Build a 2D CurveGeom from a STEP curve entity in UV space.
pub fn build_2d_curve(curve_id: u64, entities: &EntityIndex) -> Option<CurveGeom> {
    let record = entities.get(&curve_id)?;
    match record.name.as_str() {
        "LINE" => {
            let pnt_id = geom::nth_ref(&record.params, 1)?;
            let dir_id = geom::nth_ref(&record.params, 2)?;
            let pnt = resolve_cartesian_2d(pnt_id, entities)?;
            let dir = resolve_vector_2d(dir_id, entities)?;
            Some(CurveGeom::Line {
                origin: Vec3::new(pnt.0, pnt.1, 0.0),
                direction: Vec3::new(dir.0, dir.1, 0.0),
            })
        }
        "CIRCLE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let center = resolve_placement_2d(placement_id, entities)?;
            Some(CurveGeom::circle(Vec3::new(center.0, center.1, 0.0), Vec3::Z, radius))
        }
        "ELLIPSE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let semi_major = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let semi_minor = geom::nth_real(&record.params, 3).unwrap_or(0.5) as f32;
            let center = resolve_placement_2d(placement_id, entities)?;
            Some(CurveGeom::ellipse(Vec3::new(center.0, center.1, 0.0), Vec3::Z, semi_major, semi_minor))
        }
        "B_SPLINE_CURVE" | "B_SPLINE_CURVE_WITH_KNOTS" | "RATIONAL_B_SPLINE_CURVE" => {
            build_bspline_2d(record, entities)
        }
        "DEFINITIONAL_REPRESENTATION" => {
            let items = record.params.nth_param(1)?.as_list()?;
            let first_id = items.first()?.as_ref_id()?;
            build_2d_curve(first_id, entities)
        }
        "TRIMMED_CURVE" => {
            let basis_id = geom::nth_ref(&record.params, 1)?;
            let basis = build_2d_curve(basis_id, entities)?;
            let t_min = parse_trim_bound(&record.params, 2).unwrap_or(0.0);
            let t_max = parse_trim_bound(&record.params, 3).unwrap_or(1.0);
            Some(CurveGeom::Trimmed {
                basis: Box::new(basis),
                t_min: t_min.min(t_max),
                t_max: t_min.max(t_max),
            })
        }
        "POLYLINE" => {
            let pt_ids = geom::nth_list_refs(&record.params, 1).unwrap_or_default();
            let points: Vec<Vec3> = pt_ids
                .iter()
                .filter_map(|&id| {
                    resolve_cartesian_2d(id, entities).map(|(u, v)| Vec3::new(u, v, 0.0))
                })
                .collect();
            if points.len() < 2 {
                None
            } else {
                Some(CurveGeom::Polyline { points })
            }
        }
        "COMPOSITE_CURVE" => {
            let seg_ids = geom::nth_list_refs(&record.params, 1).unwrap_or_default();
            let segments: Vec<(CurveGeom, bool)> = seg_ids
                .iter()
                .filter_map(|&seg_id| {
                    let seg_rec = entities.get(&seg_id)?;
                    let parent_id = geom::nth_ref(&seg_rec.params, 3)?;
                    let same_sense = match seg_rec.params.nth_param(2) {
                        Some(StepValue::Enum(s)) => s == ".T.",
                        _ => true,
                    };
                    build_2d_curve(parent_id, entities).map(|c| (c, same_sense))
                })
                .collect();
            if segments.is_empty() {
                None
            } else {
                Some(CurveGeom::Composite { segments, cached_lengths: None })
            }
        }
        _ => None,
    }
}

// ── 2D point/vector helpers ───────────────────────────────────────

/// Resolve a DIRECTION or VECTOR to a (du, dv) pair.
fn resolve_vector_2d(vec_id: u64, entities: &EntityIndex) -> Option<(f32, f32)> {
    let record = entities.get(&vec_id)?;
    if record.name != "DIRECTION" && record.name != "VECTOR" { return None; }
    let coords = nth_list_f64(&record.params, 1)?;
    if coords.len() < 2 { return None; }
    Some((coords[0] as f32, coords[1] as f32))
}

/// Resolve an AXIS2_PLACEMENT_2D to its origin (u, v).
fn resolve_placement_2d(place_id: u64, entities: &EntityIndex) -> Option<(f32, f32)> {
    let record = entities.get(&place_id)?;
    if record.name != "AXIS2_PLACEMENT_2D" && record.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let pt_id = geom::nth_ref(&record.params, 1)?;
    resolve_cartesian_2d(pt_id, entities)
}

/// Extract a list of f64 values from a param at the given index.
fn nth_list_f64(params: &StepValue, index: usize) -> Option<Vec<f64>> {
    params.nth_param(index)?.as_list()
        .map(|list| list.iter().filter_map(|v| v.as_real()).collect())
}

/// Build a synthetic PCURVE by sampling the 3D curve and projecting
/// each point onto the surface's UV domain.
fn build_synthetic_pcurve(
    curve: &CurveGeom,
    surface: &SurfaceGeom,
    match_tol: f32,
) -> Option<CurveGeom> {
    let n = PCURVE_POLYLINE_SAMPLES;
    let inv_tol = (match_tol * 5.0).max(0.5);

    // Revolution isoparameter lines: per-sample UV can coincide; interpolate between endpoints.
    if let SurfaceGeom::Revolution { .. } = surface {
        let p0 = curve.d0(0.0);
        let p1 = curve.d0(1.0);
        if let (Some(uv0), Some(uv1)) = (
            surface.revolution_native_uv_at(p0),
            surface.revolution_native_uv_at(p1),
        ) {
            let du = uv1.0 - uv0.0;
            let dv = uv1.1 - uv0.1;
            if du * du + dv * dv > 1e-12 {
                let mut uv_points = Vec::with_capacity(n as usize + 1);
                for i in 0..=n {
                    let t = i as f32 / n as f32;
                    let u = uv0.0 + du * t;
                    let v = uv0.1 + dv * t;
                    uv_points.push(Vec3::new(u, v, 0.0));
                }
                return Some(CurveGeom::Polyline { points: uv_points });
            }
            if let (Some(u0), Some(u1)) = (
                surface.revolution_generatrix_u_at(p0),
                surface.revolution_generatrix_u_at(p1),
            ) {
                let du_g = u1 - u0;
                if du_g.abs() > 1e-6 {
                    let v0 = uv0.1;
                    let v1 = uv1.1;
                    let mut uv_points = Vec::with_capacity(n as usize + 1);
                    for i in 0..=n {
                        let t = i as f32 / n as f32;
                        let p3 = curve.d0(t);
                        let u = u0 + du_g * t;
                        let v = surface
                            .revolution_native_uv_at(p3)
                            .map(|(_, v)| v)
                            .unwrap_or(v0 + (v1 - v0) * t);
                        uv_points.push(Vec3::new(u, v, 0.0));
                    }
                    return Some(CurveGeom::Polyline { points: uv_points });
                }
            }
        }
        let mut uv_points = Vec::with_capacity(n as usize + 1);
        let mut all_same = true;
        let mut first: Option<Vec3> = None;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let p3 = curve.d0(t);
            let u = surface.revolution_generatrix_u_at(p3).unwrap_or(t);
            let v = surface
                .revolution_native_uv_at(p3)
                .map(|(_, v)| v)
                .unwrap_or(0.0);
            let uv = Vec3::new(u, v, 0.0);
            if let Some(f) = first {
                if (uv - f).length_squared() > 1e-12 {
                    all_same = false;
                }
            } else {
                first = Some(uv);
            }
            uv_points.push(uv);
        }
        if !all_same && uv_points.len() >= 2 {
            return Some(CurveGeom::Polyline { points: uv_points });
        }
        let edge_len = (p1 - p0).length();
        if edge_len > 1e-6 {
            let v0 = surface
                .revolution_native_uv_at(p0)
                .map(|(_, v)| v)
                .unwrap_or(0.0);
            let v1 = surface
                .revolution_native_uv_at(p1)
                .map(|(_, v)| v)
                .unwrap_or(v0);
            let mut uv_points = Vec::with_capacity(n as usize + 1);
            for i in 0..=n {
                let t = i as f32 / n as f32;
                uv_points.push(Vec3::new(t, v0 + (v1 - v0) * t, 0.0));
            }
            return Some(CurveGeom::Polyline { points: uv_points });
        }
    }

    let mut uv_points = Vec::with_capacity(n as usize + 1);
    let mut map_failures = 0usize;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let p3 = curve.d0(t);
        let mapped = surface.revolution_native_uv_at(p3)
            .or_else(|| surface.inverse_native_uv_build(p3, inv_tol))
            .or_else(|| surface.project(p3));
        if mapped.is_none() {
            map_failures += 1;
        }
        let (u, v) = sample_3d_to_native_uv(surface, p3, inv_tol);
        uv_points.push(Vec3::new(u, v, 0.0));
    }
    if uv_points.len() < 2 {
        return None;
    }
    let first = uv_points[0];
    if uv_points.iter().all(|p| (p - first).length_squared() < 1e-12) {
        log::debug!(
            "[BRep] synthetic_pcurve: all {} points identical at ({:.3},{:.3}), 3D curve d0(0)={:?}, d0(0.5)={:?}",
            uv_points.len(),
            first.x,
            first.y,
            curve.d0(0.0),
            curve.d0(0.5)
        );
        return None;
    }
    if map_failures > 0 {
        log::debug!(
            "[BRep] synthetic_pcurve: {} inverse UV failures out of {}",
            map_failures,
            n + 1
        );
        return None;
    }
    Some(CurveGeom::Polyline { points: uv_points })
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::parser;

    fn make_entities(data_section: &str) -> EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data_section
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_build_brep_simple_box_shell() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#5 = DIRECTION('', (0.0, 0.0, 1.0));
#6 = AXIS2_PLACEMENT_3D('', #1, #5, #2);
#7 = PLANE('', #6);
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = ADVANCED_FACE('', (#15), #7, .T.);
#17 = SHELL('', (#16));
#20 = LINE('', #1, #2);\
",
        );

        let result = build_brep(&entities).unwrap();
        assert!(!result.root_solids.is_empty(), "should produce at least one solid");
        assert_eq!(result.registry.vertices.len(), 4);
        assert_eq!(result.registry.edges.len(), 4);

        // Verify edges have PCURVEs for the face
        let solid = &result.registry.solids[result.root_solids[0]];
        let shell = &result.registry.shells[solid.outer_shell];
        assert!(!shell.faces.is_empty());
        let (face_key, _) = shell.faces[0];
        let face = &result.registry.faces[face_key];
        // The outer wire should have 4 edges
        let wire = &result.registry.wires[face.outer_wire];
        assert_eq!(wire.edges.len(), 4, "face should have 4 edges");
    }

    #[test]
    fn test_build_curve_line() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#10 = LINE('', #1, #2);\
",
        );
        let curve = build_curve(10, &entities).unwrap();
        match curve {
            CurveGeom::Line { origin, direction } => {
                assert!((origin - Vec3::ZERO).length() < 1e-6);
                assert!((direction - Vec3::X).length() < 1e-6);
            }
            _ => panic!("expected Line"),
        }
    }

    #[test]
    fn test_build_curve_circle() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#3 = DIRECTION('', (0.0, 0.0, 1.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #3, #2);
#10 = CIRCLE('', #4, 5.0);\
",
        );
        let curve = build_curve(10, &entities).unwrap();
        match curve {
            CurveGeom::Circle { center, axis, radius, .. } => {
                assert!((center - Vec3::ZERO).length() < 1e-6);
                assert!(radius - 5.0 < 1e-4);
                assert!((axis - Vec3::Z).length() < 1e-6);
            }
            _ => panic!("expected Circle"),
        }
    }

    #[test]
    fn test_build_curve_trimmed() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#10 = LINE('', #1, #2);
#20 = TRIMMED_CURVE('', #10, (0.2), (0.8), .T., .PARAMETER.);\
",
        );
        let curve = build_curve(20, &entities).unwrap();
        match curve {
            CurveGeom::Trimmed { ref basis, t_min, t_max } => {
                assert!((t_min - 0.2).abs() < 1e-4);
                assert!((t_max - 0.8).abs() < 1e-4);
                match basis.as_ref() {
                    CurveGeom::Line { .. } => {}
                    _ => panic!("expected Line basis"),
                }
            }
            _ => panic!("expected Trimmed"),
        }
    }

    #[test]
    fn test_build_2d_curve_trimmed_line_uv() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0));
#2 = DIRECTION('', (1.0, 0.0));
#10 = LINE('', #1, #2);
#20 = TRIMMED_CURVE('', #10, (0.25), (0.75), .T., .PARAMETER.);\
",
        );
        let pcurve = build_2d_curve(20, &entities).expect("2d trimmed pcurve");
        let p0 = pcurve.d0(0.0);
        let p1 = pcurve.d0(1.0);
        assert!((p0.x - 0.25).abs() < 1e-4);
        assert!(p0.y.abs() < 1e-4);
        assert!((p1.x - 0.75).abs() < 1e-4);
        assert!(p1.y.abs() < 1e-4);
    }

    #[test]
    fn test_build_curve_polyline() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (5.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (5.0, 5.0, 0.0));
#10 = POLYLINE('', (#1, #2, #3));\
",
        );
        let curve = build_curve(10, &entities).unwrap();
        match curve {
            CurveGeom::Polyline { ref points } => {
                assert_eq!(points.len(), 3);
            }
            _ => panic!("expected Polyline"),
        }
    }

    #[test]
    fn test_build_surface_plane() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = DIRECTION('', (0.0, 0.0, 1.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #3, #2);
#10 = PLANE('', #4);\
",
        );
        let surface = build_surface(10, &entities).unwrap();
        match surface {
            SurfaceGeom::Plane { origin, normal, .. } => {
                assert!((origin - Vec3::ZERO).length() < 1e-6);
                assert!((normal - Vec3::Z).length() < 1e-6);
            }
            _ => panic!("expected Plane"),
        }
    }

    #[test]
    fn test_build_surface_cylinder() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#3 = DIRECTION('', (0.0, 0.0, 1.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #3, #2);
#10 = CYLINDRICAL_SURFACE('', #4, 2.5);\
",
        );
        let surface = build_surface(10, &entities).unwrap();
        match surface {
            SurfaceGeom::Cylinder { origin, axis, radius, .. } => {
                assert!((origin - Vec3::ZERO).length() < 1e-6);
                assert!((axis - Vec3::Z).length() < 1e-6);
                assert!((radius - 2.5).abs() < 1e-4);
            }
            _ => panic!("expected Cylinder"),
        }
    }

    #[test]
    fn test_build_surface_cone() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#3 = DIRECTION('', (0.0, 0.0, 1.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #3, #2);
#10 = CONICAL_SURFACE('', #4, 1.0, 0.5);\
",
        );
        let surface = build_surface(10, &entities).unwrap();
        match surface {
            SurfaceGeom::Cone { apex, axis, radius_at_apex, semi_angle, .. } => {
                assert!((apex - Vec3::ZERO).length() < 1e-6);
                assert!((radius_at_apex - 1.0).abs() < 1e-4);
                assert!((semi_angle - 0.5).abs() < 1e-4);
                assert!((axis - Vec3::Z).length() < 1e-6);
            }
            _ => panic!("expected Cone"),
        }
    }

    #[test]
    fn test_build_surface_sphere() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#3 = DIRECTION('', (0.0, 0.0, 1.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #3, #2);
#10 = SPHERICAL_SURFACE('', #4, 3.0);\
",
        );
        let surface = build_surface(10, &entities).unwrap();
        match surface {
            SurfaceGeom::Sphere { center, radius } => {
                assert!((center - Vec3::ZERO).length() < 1e-6);
                assert!((radius - 3.0).abs() < 1e-4);
            }
            _ => panic!("expected Sphere"),
        }
    }

    #[test]
    fn test_build_surface_torus() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#3 = DIRECTION('', (0.0, 0.0, 1.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #3, #2);
#10 = TOROIDAL_SURFACE('', #4, 5.0, 1.0);\
",
        );
        let surface = build_surface(10, &entities).unwrap();
        match surface {
            SurfaceGeom::Torus { center, axis, major_r, minor_r, .. } => {
                assert!((center - Vec3::ZERO).length() < 1e-6);
                assert!((major_r - 5.0).abs() < 1e-4);
                assert!((minor_r - 1.0).abs() < 1e-4);
                assert!((axis - Vec3::Z).length() < 1e-6);
            }
            _ => panic!("expected Torus"),
        }
    }

    #[test]
    fn test_build_surface_extrusion() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = DIRECTION('', (0.0, 0.0, 1.0));
#10 = LINE('', #1, #2);
#20 = SURFACE_OF_LINEAR_EXTRUSION('', #10, #3);\
",
        );
        let surface = build_surface(20, &entities).unwrap();
        match surface {
            SurfaceGeom::Extrusion { ref generatrix, direction } => {
                match generatrix.as_ref() {
                    CurveGeom::Line { .. } => {}
                    _ => panic!("expected Line generatrix"),
                }
                assert!((direction - Vec3::Z).length() < 1e-6);
            }
            _ => panic!("expected Extrusion"),
        }
    }

    #[test]
    fn test_build_surface_revolution_axis1() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (5.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (5.0, 0.0, 10.0));
#4 = DIRECTION('', (0.0, 0.0, 1.0));
#5 = AXIS1_PLACEMENT('', #1, #4);
#10 = B_SPLINE_CURVE_WITH_KNOTS('', 1, (#2, #3), .UNSPECIFIED., .F., .F., (2, 2), (0., 1.), .PIECEWISE_BEZIER_KNOTS.);
#20 = SURFACE_OF_REVOLUTION('', #10, #5);\
",
        );
        let surface = build_surface(20, &entities).unwrap();
        match &surface {
            SurfaceGeom::Revolution {
                generatrix,
                axis_origin,
                axis_dir,
            } => {
                assert!(matches!(generatrix.as_ref(), CurveGeom::BSpline { .. }));
                assert!((axis_origin.x).abs() < 1e-4);
                assert!((axis_dir.z - 1.0).abs() < 1e-4);
                // Native (u,v): u = generatrix parameter, v = axis angle in radians.
                let p0 = surface.d0_native(0.0, 0.0);
                assert!((p0 - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-3);
                let p90 = surface.d0_native(0.0, std::f32::consts::FRAC_PI_2);
                assert!((p90 - Vec3::new(0.0, 5.0, 0.0)).length() < 1e-2);
            }
            _ => panic!("expected Revolution"),
        }
    }

    #[test]
    fn test_build_surface_revolution() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (3.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (3.0, 0.0, 1.0));
#3 = DIRECTION('', (0.0, 0.0, 1.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #3, #2);
#10 = CIRCLE('', #4, 1.0);
#20 = SURFACE_OF_REVOLUTION('', #10, #4);\
",
        );
        let surface = build_surface(20, &entities).unwrap();
        match &surface {
            SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } => {
                match generatrix.as_ref() {
                    CurveGeom::Circle { .. } => {}
                    _ => panic!("expected Circle generatrix"),
                }
                // axis_origin should be (3,0,0) from the placement
                assert!((axis_origin.x - 3.0).abs() < 1e-4);
                assert!((axis_dir.z - 1.0).abs() < 1e-4);
            }
            _ => panic!("expected Revolution, got {:?}", std::mem::discriminant(&surface)),
        }
    }

    #[test]
    fn test_build_surface_offset() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = DIRECTION('', (0.0, 0.0, 1.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #3, #2);
#10 = PLANE('', #4);
#20 = OFFSET_SURFACE('', #10, 5.0);\
",
        );
        let surface = build_surface(20, &entities).unwrap();
        match surface {
            SurfaceGeom::Offset { ref basis, distance } => {
                assert!((distance - 5.0).abs() < 1e-4);
                match basis.as_ref() {
                    SurfaceGeom::Plane { .. } => {}
                    _ => panic!("expected Plane basis"),
                }
            }
            _ => panic!("expected Offset"),
        }
    }

    #[test]
    fn test_bspline_curve_building() {
        let entities = make_entities(
            "\
#10 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#11 = CARTESIAN_POINT('', (1.0, 1.0, 0.0));
#12 = CARTESIAN_POINT('', (2.0, 0.0, 0.0));
#13 = CARTESIAN_POINT('', (3.0, 1.0, 0.0));
#20 = B_SPLINE_CURVE_WITH_KNOTS('', 3, (#10, #11, #12, #13),
 .UNSPECIFIED., .F., .F.,
 (4, 4),
 (0.0, 1.0),
 .UNSPECIFIED.);\
",
        );
        let curve = build_curve(20, &entities).unwrap();
        match curve {
            CurveGeom::BSpline { degree, ref control_points, ref knots, weights } => {
                assert_eq!(degree, 3);
                assert_eq!(control_points.len(), 4);
                assert_eq!(knots.len(), 8); // 4 CPs + degree 3 + 1 = 8
                assert!(weights.is_none());
            }
            _ => panic!("expected BSpline"),
        }
    }

    #[test]
    fn test_polyline_returns_none_for_1_point() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#10 = POLYLINE('', (#1));\
",
        );
        assert!(build_curve(10, &entities).is_none());
    }

    #[test]
    fn test_no_geometry_error() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));\
",
        );
        let result = build_brep(&entities);
        assert!(result.is_err());
        match result {
            Err(StepError::NoGeometry) => {}
            other => panic!("expected NoGeometry, got {:?}", other),
        }
    }

    #[test]
    fn test_build_brep_with_void_shells() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#5 = CARTESIAN_POINT('', (2.0, 2.0, 0.0));
#6 = CARTESIAN_POINT('', (8.0, 2.0, 0.0));
#7 = CARTESIAN_POINT('', (8.0, 8.0, 0.0));
#8 = CARTESIAN_POINT('', (2.0, 8.0, 0.0));
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = FACE_SURFACE('', (#15));
#17 = CLOSED_SHELL('', (#16));
#30 = EDGE_CURVE('', #5, #6, #20, .T.);
#31 = EDGE_CURVE('', #6, #7, #20, .T.);
#32 = EDGE_CURVE('', #7, #8, #20, .T.);
#33 = EDGE_CURVE('', #8, #5, #20, .T.);
#34 = EDGE_LOOP('', (#30, #31, #32, #33));
#35 = FACE_OUTER_BOUND('', #34, .T.);
#36 = FACE_SURFACE('', (#35));
#37 = CLOSED_SHELL('', (#36));
#18 = BREP_WITH_VOIDS('', #17, (#37));
#20 = LINE('', #1, #2);\
",
        );
        let result = build_brep(&entities).expect("build with void");
        assert_eq!(result.build_report.void_shell_count, 1);
        let solid = result
            .registry
            .solids
            .get(result.root_solids[0])
            .expect("solid");
        assert_eq!(solid.void_shells.len(), 1);
    }

    #[test]
    fn test_strict_skips_missing_surface() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = FACE_SURFACE('', (#15), #999);
#17 = CLOSED_SHELL('', (#16));
#18 = MANIFOLD_SOLID_BREP('', #17);
#20 = LINE('', #1, #2);\
",
        );
        let strict = BRepBuildOptions {
            allow_geometry_fallback: false,
            strict_voids: false,
        };
        let result = build_brep_with_options(&entities, &strict);
        assert!(result.is_err() || result.unwrap().build_report.skipped_faces >= 1);
    }

    #[test]
    fn test_preview_fallback_missing_surface() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = FACE_SURFACE('', (#15), #999);
#17 = CLOSED_SHELL('', (#16));
#18 = MANIFOLD_SOLID_BREP('', #17);
#20 = LINE('', #1, #2);\
",
        );
        let preview = BRepBuildOptions {
            allow_geometry_fallback: true,
            strict_voids: false,
        };
        let result = build_brep_with_options(&entities, &preview).expect("preview fallback");
        assert_eq!(result.build_report.skipped_faces, 0);
        assert!(!result.root_solids.is_empty());
    }
}
