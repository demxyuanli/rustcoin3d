//! Parametric STEP writer for BRepStore topology.
//!
//! Produces ISO 10303-21 text with exact parametric geometry entities
//! (BSpline surfaces, elementary surfaces, curves) from a BRepStore.

use std::collections::HashMap;

use rc3d_core::math::{Real, PVec3};
use rc3d_shape::geom::CurveGeom;
use rc3d_shape::geom::SurfaceGeom;
use rc3d_shape::store::BRepStore;
use rc3d_shape::topo::*;

// ── Formatting helpers ──────────────────────────────────────────────

/// Format a double with 12 significant digits for STEP parametric output.
fn fmt_real(v: Real) -> String {
    if v == 0.0 {
        "0.".to_string()
    } else if v.abs() < 1e-15 {
        "0.".to_string()
    } else {
        // Use scientific notation with 12 significant digits
        let s = format!("{:.12E}", v);
        // Normalize: lowercase 'e', strip leading '+' in exponent
        s.replace('E', "E").replace("E+", "E").replace("E-0", "E-")
    }
}

/// Format a 3D point as a STEP tuple: (x, y, z)
fn fmt_point(pt: PVec3) -> String {
    format!("({}, {}, {})", fmt_real(pt.x), fmt_real(pt.y), fmt_real(pt.z))
}

/// Format a reference to a STEP entity.
fn fmt_ref(id: u64) -> String {
    format!("#{}", id)
}

// ── Entity ID tracker ───────────────────────────────────────────────

struct IdGen {
    next: u64,
}

impl IdGen {
    fn new() -> Self {
        Self { next: 0 }
    }

    fn next(&mut self) -> u64 {
        self.next += 1;
        self.next
    }
}

// ── Geometry entity emitters ────────────────────────────────────────

/// Emit a CARTESIAN_POINT entity. Returns assigned ID.
fn emit_point(out: &mut String, id: &mut IdGen, pt: PVec3) -> u64 {
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = CARTESIAN_POINT('', {});\n",
        assigned,
        fmt_point(pt)
    ));
    assigned
}

/// Emit a DIRECTION entity. Returns assigned ID.
fn emit_direction(out: &mut String, id: &mut IdGen, dir: PVec3) -> u64 {
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = DIRECTION('', {});\n",
        assigned,
        fmt_point(dir)
    ));
    assigned
}

/// Emit an AXIS2_PLACEMENT_3D entity (point + axis + ref_dir).
/// Sub-entities (point, directions) are emitted inline.
fn emit_axis2(
    out: &mut String,
    id: &mut IdGen,
    origin: PVec3,
    axis: PVec3,
    ref_dir: PVec3,
) -> u64 {
    let pt_id = emit_point(out, id, origin);
    let axis_id = emit_direction(out, id, axis);
    let ref_id = emit_direction(out, id, ref_dir);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = AXIS2_PLACEMENT_3D('', #{}, #{}, #{});\n",
        assigned, pt_id, axis_id, ref_id
    ));
    assigned
}

/// Emit a LINE curve. Returns the LINE entity ID.
fn emit_line(out: &mut String, id: &mut IdGen, origin: PVec3, direction: PVec3) -> u64 {
    let pt_id = emit_point(out, id, origin);
    let dir_id = emit_direction(out, id, direction);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = LINE('', #{}, #{});\n",
        assigned, pt_id, dir_id
    ));
    assigned
}

/// Emit a CIRCLE curve. Returns the CIRCLE entity ID.
fn emit_circle(
    out: &mut String,
    id: &mut IdGen,
    center: PVec3,
    axis: PVec3,
    radius: Real,
    x_dir: PVec3,
) -> u64 {
    let placement_id = emit_axis2(out, id, center, axis, x_dir);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = CIRCLE('', #{}, {});\n",
        assigned, placement_id, fmt_real(radius)
    ));
    assigned
}

/// Emit an ELLIPSE curve. Returns the ELLIPSE entity ID.
fn emit_ellipse(
    out: &mut String,
    id: &mut IdGen,
    center: PVec3,
    axis: PVec3,
    semi_major: Real,
    semi_minor: Real,
    x_dir: PVec3,
) -> u64 {
    let placement_id = emit_axis2(out, id, center, axis, x_dir);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = ELLIPSE('', #{}, {}, {});\n",
        assigned, placement_id, fmt_real(semi_major), fmt_real(semi_minor)
    ));
    assigned
}

/// Emit a B_SPLINE_CURVE_WITH_KNOTS entity (non-rational).
/// For curves with weights, use RATIONAL_B_SPLINE_CURVE instead (stub).
fn emit_bspline_curve(
    out: &mut String,
    id: &mut IdGen,
    degree: usize,
    control_points: &[PVec3],
    knots: &[Real],
    weights: Option<&[Real]>,
) -> u64 {
    // Emit control points as CARTESIAN_POINTs
    let cp_ids: Vec<u64> = control_points
        .iter()
        .map(|&cp| emit_point(out, id, cp))
        .collect();

    // Build knot multiplicities from the knot vector.
    // Group consecutive equal knots and count occurrences.
    let mut unique_knots: Vec<Real> = Vec::new();
    let mut multiplicities: Vec<usize> = Vec::new();
    for &k in knots {
        if let Some(last) = unique_knots.last_mut() {
            if (k - *last).abs() < 1e-12 {
                *multiplicities.last_mut().unwrap() += 1;
                continue;
            }
        }
        unique_knots.push(k);
        multiplicities.push(1);
    }

    let cp_list: Vec<String> = cp_ids.iter().map(|&i| fmt_ref(i)).collect();
    let mult_list: Vec<String> = multiplicities.iter().map(|m| m.to_string()).collect();
    let knot_list: Vec<String> = unique_knots.iter().map(|&k| fmt_real(k)).collect();

    let has_weights = weights.is_some();
    let entity_name = if has_weights {
        "RATIONAL_B_SPLINE_CURVE"
    } else {
        "B_SPLINE_CURVE_WITH_KNOTS"
    };

    let assigned = id.next();

    let line = if has_weights {
        let w = weights.unwrap();
        let w_list: Vec<String> = w.iter().map(|&x| fmt_real(x)).collect();
        format!(
            "#{} = {}({}, {}, ({}), .UNSPECIFIED., .F., .F., ({}), ({}), .UNSPECIFIED., ({}));\n",
            assigned,
            entity_name,
            degree,
            0, // curve_form integer enum
            cp_list.join(", "),
            mult_list.join(", "),
            knot_list.join(", "),
            w_list.join(", ")
        )
    } else {
        format!(
            "#{} = {}({}, {}, ({}), .UNSPECIFIED., .F., .F., ({}), ({}), .UNSPECIFIED.);\n",
            assigned,
            entity_name,
            degree,
            0, // curve_form
            cp_list.join(", "),
            mult_list.join(", "),
            knot_list.join(", ")
        )
    };

    out.push_str(&line);
    assigned
}

// ── Surface entity emitters ─────────────────────────────────────────

/// Emit a PLANE surface entity. Returns the PLANE entity ID.
fn emit_plane(
    out: &mut String,
    id: &mut IdGen,
    origin: PVec3,
    normal: PVec3,
    u_dir: PVec3,
) -> u64 {
    let placement_id = emit_axis2(out, id, origin, normal, u_dir);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = PLANE('', #{});\n",
        assigned, placement_id
    ));
    assigned
}

/// Emit a CYLINDRICAL_SURFACE entity.
fn emit_cylindrical_surface(
    out: &mut String,
    id: &mut IdGen,
    origin: PVec3,
    axis: PVec3,
    radius: Real,
    x_dir: PVec3,
) -> u64 {
    let placement_id = emit_axis2(out, id, origin, axis, x_dir);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = CYLINDRICAL_SURFACE('', #{}, {});\n",
        assigned, placement_id, fmt_real(radius)
    ));
    assigned
}

/// Emit a CONICAL_SURFACE entity.
fn emit_conical_surface(
    out: &mut String,
    id: &mut IdGen,
    apex: PVec3,
    axis: PVec3,
    semi_angle: Real,
    radius_at_apex: Real,
    x_dir: PVec3,
) -> u64 {
    let placement_id = emit_axis2(out, id, apex, axis, x_dir);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = CONICAL_SURFACE('', #{}, {}, {});\n",
        assigned, placement_id, fmt_real(radius_at_apex), fmt_real(semi_angle)
    ));
    assigned
}

/// Emit a SPHERICAL_SURFACE entity.
fn emit_spherical_surface(
    out: &mut String,
    id: &mut IdGen,
    center: PVec3,
    radius: Real,
) -> u64 {
    // Sphere axis is arbitrary; use +Z as axis, +X as ref_dir
    let placement_id = emit_axis2(
        out, id, center,
        PVec3::Z,
        PVec3::X,
    );
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = SPHERICAL_SURFACE('', #{}, {});\n",
        assigned, placement_id, fmt_real(radius)
    ));
    assigned
}

/// Emit a TOROIDAL_SURFACE entity.
fn emit_toroidal_surface(
    out: &mut String,
    id: &mut IdGen,
    center: PVec3,
    axis: PVec3,
    major_r: Real,
    minor_r: Real,
    x_dir: PVec3,
) -> u64 {
    let placement_id = emit_axis2(out, id, center, axis, x_dir);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = TOROIDAL_SURFACE('', #{}, {}, {});\n",
        assigned, placement_id, fmt_real(major_r), fmt_real(minor_r)
    ));
    assigned
}

/// Emit a SURFACE_OF_LINEAR_EXTRUSION entity (stub — uses LINE fallback).
fn emit_extrusion_surface(
    out: &mut String,
    id: &mut IdGen,
    generatrix: &CurveGeom,
    direction: PVec3,
) -> u64 {
    // Emit the generatrix curve inline first, then the extrusion surface.
    // For now only handle LINE and CIRCLE as generatrix curves.
    let curve_id = emit_curve_geom(out, id, generatrix);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = SURFACE_OF_LINEAR_EXTRUSION('', #{}, {});\n",
        assigned,
        curve_id,
        fmt_point(direction)
    ));
    assigned
}

/// Emit a SURFACE_OF_REVOLUTION entity.
fn emit_revolution_surface(
    out: &mut String,
    id: &mut IdGen,
    generatrix: &CurveGeom,
    axis_origin: PVec3,
    axis_dir: PVec3,
) -> u64 {
    let curve_id = emit_curve_geom(out, id, generatrix);
    let placement_id = emit_axis2(out, id, axis_origin, axis_dir, PVec3::X);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = SURFACE_OF_REVOLUTION('', #{}, #{});\n",
        assigned, curve_id, placement_id
    ));
    assigned
}

/// Emit an OFFSET_SURFACE entity (stub).
fn emit_offset_surface(
    out: &mut String,
    id: &mut IdGen,
    basis: &SurfaceGeom,
    distance: Real,
) -> u64 {
    let basis_id = emit_surface_geom(out, id, basis);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = OFFSET_SURFACE('', #{}, {});\n",
        assigned, basis_id, fmt_real(distance)
    ));
    assigned
}

/// Emit a B_SPLINE_SURFACE_WITH_KNOTS entity (stub — non-rational).
fn emit_bspline_surface(
    out: &mut String,
    id: &mut IdGen,
    nurbs: &rc3d_shape::nurbs::NurbsSurface,
) -> u64 {
    // Emit control points as CARTESIAN_POINTs (row-major)
    let mut cp_refs: Vec<String> = Vec::new();
    for row in &nurbs.control_points {
        let row_refs: Vec<String> = row
            .iter()
            .map(|&cp| fmt_ref(emit_point(out, id, cp)))
            .collect();
        cp_refs.push(format!("({})", row_refs.join(", ")));
    }

    // U knot multiplicities
    let u_counts: Vec<usize> = knot_multiplicities(&nurbs.knots_u);
    let v_counts: Vec<usize> = knot_multiplicities(&nurbs.knots_v);
    let u_unique: Vec<Real> = unique_knots(&nurbs.knots_u);
    let v_unique: Vec<Real> = unique_knots(&nurbs.knots_v);

    let u_mult: Vec<String> = u_counts.iter().map(|m| m.to_string()).collect();
    let v_mult: Vec<String> = v_counts.iter().map(|m| m.to_string()).collect();
    let u_knots: Vec<String> = u_unique.iter().map(|&k| fmt_real(k)).collect();
    let v_knots: Vec<String> = v_unique.iter().map(|&k| fmt_real(k)).collect();

    let has_weights = nurbs.weights.iter().any(|row| row.iter().any(|&w| (w - 1.0).abs() > 1e-9));
    let entity_name = if has_weights {
        "RATIONAL_B_SPLINE_SURFACE"
    } else {
        "B_SPLINE_SURFACE_WITH_KNOTS"
    };

    let assigned = id.next();

    let line = if has_weights {
        let mut w_refs: Vec<String> = Vec::new();
        for row in &nurbs.weights {
            let row_refs: Vec<String> = row.iter().map(|&w| fmt_real(w)).collect();
            w_refs.push(format!("({})", row_refs.join(", ")));
        }
        format!(
            "#{} = {}({}, {}, ({}), .UNSPECIFIED., .F., .F., .F., ({}), ({}), ({}), ({}), .UNSPECIFIED., ({}));\n",
            assigned,
            entity_name,
            nurbs.degree_u, nurbs.degree_v,
            cp_refs.join(", "),
            u_mult.join(", "), v_mult.join(", "),
            u_knots.join(", "), v_knots.join(", "),
            w_refs.join(", ")
        )
    } else {
        format!(
            "#{} = {}({}, {}, ({}), .UNSPECIFIED., .F., .F., .F., ({}), ({}), ({}), ({}), .UNSPECIFIED.);\n",
            assigned,
            entity_name,
            nurbs.degree_u, nurbs.degree_v,
            cp_refs.join(", "),
            u_mult.join(", "), v_mult.join(", "),
            u_knots.join(", "), v_knots.join(", ")
        )
    };

    out.push_str(&line);
    assigned
}

/// Compute knot multiplicities: count consecutive equal knots.
fn knot_multiplicities(knots: &[Real]) -> Vec<usize> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < knots.len() {
        let mut count = 1;
        while i + count < knots.len() && (knots[i + count] - knots[i]).abs() < 1e-12 {
            count += 1;
        }
        result.push(count);
        i += count;
    }
    result
}

/// Extract unique knot values (first occurrence of each group).
fn unique_knots(knots: &[Real]) -> Vec<Real> {
    let mut result: Vec<Real> = Vec::new();
    for &k in knots {
        if result.is_empty() || (k - (*result.last().unwrap() as Real)).abs() > 1e-12 {
            result.push(k);
        }
    }
    result
}

// ── Curve geometry dispatcher ───────────────────────────────────────

/// Emit the STEP entity for a CurveGeom. Returns the curve entity ID.
fn emit_curve_geom(out: &mut String, id: &mut IdGen, curve: &CurveGeom) -> u64 {
    match curve {
        CurveGeom::Line { origin, direction } => {
            emit_line(out, id, *origin, *direction)
        }
        CurveGeom::Circle { center, axis, radius, x_dir, .. } => {
            emit_circle(out, id, *center, *axis, *radius, *x_dir)
        }
        CurveGeom::Ellipse { center, axis, semi_major, semi_minor, x_dir, .. } => {
            emit_ellipse(out, id, *center, *axis, *semi_major, *semi_minor, *x_dir)
        }
        CurveGeom::BSpline { degree, control_points, knots, weights } => {
            emit_bspline_curve(out, id, *degree, control_points, knots, weights.as_deref())
        }
        CurveGeom::BezierCurve { degree, control_points, weights } => {
            // Bezier is a special case of BSpline with clamped knot vector [0^n, 1^n]
            let n = control_points.len();
            let knots: Vec<Real> = std::iter::repeat(0.0).take(*degree + 1)
                .chain((1..=n.saturating_sub(*degree)).map(|i| i as Real / (n - *degree) as Real))
                .chain(std::iter::repeat(1.0).take(*degree + 1))
                .collect();
            emit_bspline_curve(out, id, *degree, control_points, &knots, weights.as_deref())
        }
        // Trimmed: unwrap and emit the basis curve (the trim is encoded
        // in the EDGE_CURVE's vertex endpoints, not in the geometry entity).
        CurveGeom::Trimmed { basis, .. } => {
            emit_curve_geom(out, id, basis)
        }
        // Stub: unsupported curve types fall back to a polyline via BSpline degree 1
        _ => {
            let n = 16;
            let cps: Vec<PVec3> = (0..=n).map(|i| {
                let t = i as Real / n as Real;
                curve.d0(t)
            }).collect();
            let knots: Vec<Real> = (0..=n).map(|i| i as Real).collect();
            emit_bspline_curve(out, id, 1, &cps, &knots, None)
        }
    }
}

// ── Surface geometry dispatcher ─────────────────────────────────────

/// Emit the STEP entity for a SurfaceGeom. Returns the surface entity ID.
fn emit_surface_geom(out: &mut String, id: &mut IdGen, surface: &SurfaceGeom) -> u64 {
    match surface {
        SurfaceGeom::Plane { origin, normal, u_dir } => {
            emit_plane(out, id, *origin, *normal, *u_dir)
        }
        SurfaceGeom::Cylinder { origin, axis, radius, x_dir, .. } => {
            emit_cylindrical_surface(out, id, *origin, *axis, *radius, *x_dir)
        }
        SurfaceGeom::Cone { apex, axis, semi_angle, radius_at_apex, x_dir, .. } => {
            emit_conical_surface(out, id, *apex, *axis, *semi_angle, *radius_at_apex, *x_dir)
        }
        SurfaceGeom::Sphere { center, radius } => {
            emit_spherical_surface(out, id, *center, *radius)
        }
        SurfaceGeom::Torus { center, axis, major_r, minor_r, x_dir, .. } => {
            emit_toroidal_surface(out, id, *center, *axis, *major_r, *minor_r, *x_dir)
        }
        SurfaceGeom::BSpline(nurbs) => {
            emit_bspline_surface(out, id, nurbs)
        }
        SurfaceGeom::Extrusion { generatrix, direction } => {
            emit_extrusion_surface(out, id, generatrix, *direction)
        }
        SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } => {
            emit_revolution_surface(out, id, generatrix, *axis_origin, *axis_dir)
        }
        SurfaceGeom::Offset { basis, distance } => {
            emit_offset_surface(out, id, basis, *distance)
        }
    }
}

// ── Topology entity emitters ────────────────────────────────────────

/// Emit EDGE_CURVE for an edge. Returns the EDGE_CURVE entity ID.
fn emit_edge_curve(
    out: &mut String,
    id: &mut IdGen,
    edge: &BRepEdge,
    vertex_map: &HashMap<VertexKey, u64>,
    curve_id: u64,
) -> u64 {
    let v_start = vertex_map.get(&edge.v_low).copied().unwrap_or(0);
    let v_end = vertex_map.get(&edge.v_high).copied().unwrap_or(0);
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = EDGE_CURVE('', #{}, #{}, #{}, .T.);\n",
        assigned, v_start, v_end, curve_id
    ));
    assigned
}

/// Emit an ORIENTED_EDGE entity. Returns the ORIENTED_EDGE entity ID.
fn emit_oriented_edge(
    out: &mut String,
    id: &mut IdGen,
    edge_curve_id: u64,
    orientation: Orientation,
) -> u64 {
    let orient_flag = match orientation {
        Orientation::Forward => ".T.",
        Orientation::Reversed => ".F.",
        Orientation::Internal | Orientation::External => ".T.",
    };
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = ORIENTED_EDGE('', *, *, #{}, {});\n",
        assigned, edge_curve_id, orient_flag
    ));
    assigned
}

/// Emit an EDGE_LOOP entity (list of ORIENTED_EDGE refs).
fn emit_edge_loop(
    out: &mut String,
    id: &mut IdGen,
    oriented_edge_ids: &[u64],
) -> u64 {
    let refs: Vec<String> = oriented_edge_ids.iter().map(|&e| fmt_ref(e)).collect();
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = EDGE_LOOP('', ({}));\n",
        assigned, refs.join(", ")
    ));
    assigned
}

/// Emit a FACE_OUTER_BOUND entity.
fn emit_face_outer_bound(
    out: &mut String,
    id: &mut IdGen,
    loop_id: u64,
    orient: bool,
) -> u64 {
    let o = if orient { ".T." } else { ".F." };
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = FACE_OUTER_BOUND('', #{}, {});\n",
        assigned, loop_id, o
    ));
    assigned
}

/// Emit a FACE_BOUND entity (for inner loops/holes).
fn emit_face_bound(
    out: &mut String,
    id: &mut IdGen,
    loop_id: u64,
    orient: bool,
) -> u64 {
    let o = if orient { ".T." } else { ".F." };
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = FACE_BOUND('', #{}, {});\n",
        assigned, loop_id, o
    ));
    assigned
}

/// Emit an ADVANCED_FACE entity.
fn emit_advanced_face(
    out: &mut String,
    id: &mut IdGen,
    bound_ids: &[u64], // FACE_OUTER_BOUND + FACE_BOUNDs
    surface_id: u64,
    same_sense: bool,
) -> u64 {
    let refs: Vec<String> = bound_ids.iter().map(|&b| fmt_ref(b)).collect();
    let ss = if same_sense { ".T." } else { ".F." };
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = ADVANCED_FACE('', ({}), #{}, {});\n",
        assigned, refs.join(", "), surface_id, ss
    ));
    assigned
}

/// Emit a CLOSED_SHELL entity.
fn emit_closed_shell(out: &mut String, id: &mut IdGen, face_ids: &[u64]) -> u64 {
    let refs: Vec<String> = face_ids.iter().map(|&f| fmt_ref(f)).collect();
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = CLOSED_SHELL('', ({}));\n",
        assigned, refs.join(", ")
    ));
    assigned
}

/// Emit an OPEN_SHELL entity.
fn emit_open_shell(out: &mut String, id: &mut IdGen, face_ids: &[u64]) -> u64 {
    let refs: Vec<String> = face_ids.iter().map(|&f| fmt_ref(f)).collect();
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = OPEN_SHELL('', ({}));\n",
        assigned, refs.join(", ")
    ));
    assigned
}

/// Emit a MANIFOLD_SOLID_BREP entity.
fn emit_manifold_solid_brep(out: &mut String, id: &mut IdGen, shell_id: u64) -> u64 {
    let assigned = id.next();
    out.push_str(&format!(
        "#{} = MANIFOLD_SOLID_BREP('', #{});\n",
        assigned, shell_id
    ));
    assigned
}

// ── Main writer ─────────────────────────────────────────────────────

/// Write a BRepStore as ISO 10303-21 parametric STEP text (AP242 schema).
pub fn write_step_parametric(store: &BRepStore) -> String {
    let mut out = String::with_capacity(store.vertices.len() * 512);
    let mut id = IdGen::new();

    // ── Header ──────────────────────────────────────────────────
    out.push_str("ISO-10303-21;\n");
    out.push_str("HEADER;\n");
    out.push_str("FILE_DESCRIPTION(('rustcoin3d parametric export'), '2;1');\n");
    {
        // ISO 8601-like date
        use std::time::SystemTime;
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| {
                let secs = d.as_secs();
                let days = secs / 86400;
                let mut y = 1970i64;
                let mut remaining = days as i64;
                loop {
                    let yd = if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                        366
                    } else {
                        365
                    };
                    if remaining < yd {
                        break;
                    }
                    remaining -= yd;
                    y += 1;
                }
                let md = if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                    [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                } else {
                    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                };
                let mut m = 1usize;
                for &days_in_month in &md {
                    if remaining < days_in_month as i64 {
                        break;
                    }
                    remaining -= days_in_month as i64;
                    m += 1;
                }
                format!(
                    "{:04}-{:02}-{:02}T00:00:00",
                    y,
                    m,
                    remaining + 1
                )
            })
            .unwrap_or_else(|_| "2024-01-01T00:00:00".to_string());
        out.push_str(&format!(
            "FILE_NAME('export', '{}', ('rustcoin3d'), (''), '', '');\n",
            now
        ));
    }
    out.push_str("FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF'));\n");
    out.push_str("ENDSEC;\n");
    out.push_str("DATA;\n");

    // ── Pass 1: Vertex → CARTESIAN_POINT ────────────────────────
    let mut vertex_map: HashMap<VertexKey, u64> = HashMap::new();
    for (vk, v) in store.vertices.iter() {
        let pt_id = emit_point(&mut out, &mut id, v.position);
        vertex_map.insert(vk, pt_id);
    }

    // ── Pass 2: Edge → Curve geometry + EDGE_CURVE ──────────────
    let mut edge_curve_map: HashMap<EdgeKey, u64> = HashMap::new(); // EdgeKey → curve geometry ID
    let mut edge_curve_entity_map: HashMap<EdgeKey, u64> = HashMap::new(); // EdgeKey → EDGE_CURVE entity ID

    for (ek, edge) in store.edges.iter() {
        let curve_id = emit_curve_geom(&mut out, &mut id, &edge.curve);
        edge_curve_map.insert(ek, curve_id);
    }

    // Pass 2b: Emit EDGE_CURVE for each edge (after curves are emitted)
    for (ek, edge) in store.edges.iter() {
        let curve_id = edge_curve_map[&ek];
        let ec_id = emit_edge_curve(&mut out, &mut id, edge, &vertex_map, curve_id);
        edge_curve_entity_map.insert(ek, ec_id);
    }

    // ── Pass 3: Surface geometry for each face ──────────────────
    let mut face_surface_map: HashMap<FaceKey, u64> = HashMap::new();

    for (fk, face) in store.faces.iter() {
        let surface_id = emit_surface_geom(&mut out, &mut id, &face.surface);
        face_surface_map.insert(fk, surface_id);
    }

    // ── Pass 4: Wires → ORIENTED_EDGE + EDGE_LOOP ──────────────
    // Emit oriented edges per wire, then the loop
    let mut wire_loop_map: HashMap<WireKey, u64> = HashMap::new();

    for (wk, wire) in store.wires.iter() {
        let mut oe_ids: Vec<u64> = Vec::new();
        for &(ek, orientation) in &wire.edges {
            if let Some(&ec_id) = edge_curve_entity_map.get(&ek) {
                let oe_id = emit_oriented_edge(&mut out, &mut id, ec_id, orientation);
                oe_ids.push(oe_id);
            }
        }
        if !oe_ids.is_empty() {
            let loop_id = emit_edge_loop(&mut out, &mut id, &oe_ids);
            wire_loop_map.insert(wk, loop_id);
        }
    }

    // ── Pass 5: Faces → FACE_OUTER_BOUND + ADVANCED_FACE ───────
    let mut advanced_face_map: HashMap<FaceKey, u64> = HashMap::new();

    for (fk, face) in store.faces.iter() {
        let mut bound_ids: Vec<u64> = Vec::new();

        // Outer wire → FACE_OUTER_BOUND
        if let Some(&loop_id) = wire_loop_map.get(&face.outer_wire) {
            let fob_id = emit_face_outer_bound(&mut out, &mut id, loop_id, true);
            bound_ids.push(fob_id);
        }

        // Inner wires → FACE_BOUND
        for &iw in &face.inner_wires {
            if let Some(&loop_id) = wire_loop_map.get(&iw) {
                let fb_id = emit_face_bound(&mut out, &mut id, loop_id, true);
                bound_ids.push(fb_id);
            }
        }

        if !bound_ids.is_empty() {
            if let Some(&surface_id) = face_surface_map.get(&fk) {
                let af_id = emit_advanced_face(
                    &mut out,
                    &mut id,
                    &bound_ids,
                    surface_id,
                    face.same_sense,
                );
                advanced_face_map.insert(fk, af_id);
            }
        }
    }

    // ── Pass 6: Shells → CLOSED_SHELL / OPEN_SHELL ─────────────
    let mut shell_map: HashMap<ShellKey, u64> = HashMap::new();

    for (sk, shell) in store.shells.iter() {
        let face_ids: Vec<u64> = shell
            .faces
            .iter()
            .filter_map(|&(fk, _orient)| advanced_face_map.get(&fk).copied())
            .collect();
        if !face_ids.is_empty() {
            let sh_id = if shell.closed {
                emit_closed_shell(&mut out, &mut id, &face_ids)
            } else {
                emit_open_shell(&mut out, &mut id, &face_ids)
            };
            shell_map.insert(sk, sh_id);
        }
    }

    // ── Pass 7: Solids → MANIFOLD_SOLID_BREP / BREP_WITH_VOIDS ─
    for (_solid_key, solid) in store.solids.iter() {
        if let Some(&outer_id) = shell_map.get(&solid.outer_shell) {
            if solid.void_shells.is_empty() {
                emit_manifold_solid_brep(&mut out, &mut id, outer_id);
            } else {
                // BREP_WITH_VOIDS not yet implemented — emit MANIFOLD_SOLID_BREP for outer only
                emit_manifold_solid_brep(&mut out, &mut id, outer_id);
            }
        }
    }

    // ── Footer ──────────────────────────────────────────────────
    out.push_str("ENDSEC;\n");
    out.push_str("END-ISO-10303-21;\n");

    out
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::PVec3;
    use rc3d_shape::geom::{CurveGeom, SurfaceGeom};

    /// Build a simple unit cube BRepStore (0,0,0) to (1,1,1).
    fn build_unit_cube() -> BRepStore {
        let tol = 1e-6;
        let mut store = BRepStore::with_tolerance(
            rc3d_shape::tolerance::ToleranceContext::default(),
        );

        // 8 vertices
        let v000 = store.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.0), tol);
        let v100 = store.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), tol);
        let v110 = store.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), tol);
        let v010 = store.find_or_add_vertex(PVec3::new(0.0, 1.0, 0.0), tol);
        let v001 = store.find_or_add_vertex(PVec3::new(0.0, 0.0, 1.0), tol);
        let v101 = store.find_or_add_vertex(PVec3::new(1.0, 0.0, 1.0), tol);
        let v111 = store.find_or_add_vertex(PVec3::new(1.0, 1.0, 1.0), tol);
        let v011 = store.find_or_add_vertex(PVec3::new(0.0, 1.0, 1.0), tol);

        /// Helper: create a rectangular face
        struct FaceDef {
            surface: SurfaceGeom,
            corners: [PVec3; 4],  // 4 corners in CCW order
            vertices: [VertexKey; 4],
        }

        let faces = vec![
            // Bottom face z=0, normal -Z (looking from below)
            FaceDef {
                surface: SurfaceGeom::Plane {
                    origin: PVec3::ZERO,
                    normal: -PVec3::Z,
                    u_dir: PVec3::X,
                },
                corners: [
                    PVec3::new(0.0, 0.0, 0.0),
                    PVec3::new(1.0, 0.0, 0.0),
                    PVec3::new(1.0, 1.0, 0.0),
                    PVec3::new(0.0, 1.0, 0.0),
                ],
                vertices: [v000, v100, v110, v010],
            },
            // Top face z=1, normal +Z
            FaceDef {
                surface: SurfaceGeom::Plane {
                    origin: PVec3::new(0.0, 0.0, 1.0),
                    normal: PVec3::Z,
                    u_dir: PVec3::X,
                },
                corners: [
                    PVec3::new(0.0, 0.0, 1.0),
                    PVec3::new(0.0, 1.0, 1.0),
                    PVec3::new(1.0, 1.0, 1.0),
                    PVec3::new(1.0, 0.0, 1.0),
                ],
                vertices: [v001, v011, v111, v101],
            },
            // Front face y=0, normal -Y
            FaceDef {
                surface: SurfaceGeom::Plane {
                    origin: PVec3::ZERO,
                    normal: -PVec3::Y,
                    u_dir: PVec3::X,
                },
                corners: [
                    PVec3::new(0.0, 0.0, 0.0),
                    PVec3::new(0.0, 0.0, 1.0),
                    PVec3::new(1.0, 0.0, 1.0),
                    PVec3::new(1.0, 0.0, 0.0),
                ],
                vertices: [v000, v001, v101, v100],
            },
            // Back face y=1, normal +Y
            FaceDef {
                surface: SurfaceGeom::Plane {
                    origin: PVec3::new(0.0, 1.0, 0.0),
                    normal: PVec3::Y,
                    u_dir: PVec3::X,
                },
                corners: [
                    PVec3::new(0.0, 1.0, 0.0),
                    PVec3::new(1.0, 1.0, 0.0),
                    PVec3::new(1.0, 1.0, 1.0),
                    PVec3::new(0.0, 1.0, 1.0),
                ],
                vertices: [v010, v110, v111, v011],
            },
            // Left face x=0, normal -X
            FaceDef {
                surface: SurfaceGeom::Plane {
                    origin: PVec3::ZERO,
                    normal: -PVec3::X,
                    u_dir: PVec3::Y,
                },
                corners: [
                    PVec3::new(0.0, 0.0, 0.0),
                    PVec3::new(0.0, 1.0, 0.0),
                    PVec3::new(0.0, 1.0, 1.0),
                    PVec3::new(0.0, 0.0, 1.0),
                ],
                vertices: [v000, v010, v011, v001],
            },
            // Right face x=1, normal +X
            FaceDef {
                surface: SurfaceGeom::Plane {
                    origin: PVec3::new(1.0, 0.0, 0.0),
                    normal: PVec3::X,
                    u_dir: PVec3::Y,
                },
                corners: [
                    PVec3::new(1.0, 0.0, 0.0),
                    PVec3::new(1.0, 0.0, 1.0),
                    PVec3::new(1.0, 1.0, 1.0),
                    PVec3::new(1.0, 1.0, 0.0),
                ],
                vertices: [v100, v101, v111, v110],
            },
        ];

        let mut face_keys: Vec<FaceKey> = Vec::new();

        for face_def in &faces {
            let fk = store.add_face(face_def.surface.clone(), tol);

            // Create 4 edges for this face
            let mut edge_keys: Vec<(EdgeKey, Orientation)> = Vec::new();
            for i in 0..4 {
                let j = (i + 1) % 4;
                let p0 = face_def.corners[i];
                let p1 = face_def.corners[j];
                let v0 = face_def.vertices[i];
                let v1 = face_def.vertices[j];

                let dir = p1 - p0;
                let curve = CurveGeom::Line {
                    origin: p0,
                    direction: dir,
                };

                // Simple UV pcurve: map along face UV from corners
                let pc = rc3d_shape::geom::curve2d::Curve2d::Line {
                    origin: (i as f64 * 0.25, 0.0),
                    direction: (0.25, 0.0),
                };

                let ek = store.add_edge_with_pcurve(
                    v0, v1, curve.clone(), tol, fk, pc, true,
                );
                // Determine orientation: Forward if vertex order matches the edge's v_low→v_high
                let edge = store.edges.get(ek).unwrap();
                let orient = if edge.v_low == v0 && edge.v_high == v1 {
                    Orientation::Forward
                } else {
                    Orientation::Reversed
                };
                edge_keys.push((ek, orient));
            }

            // Replace the outer wire with our edge edges
            let wire_key = store.wires.insert(BRepWire {
                edges: edge_keys,
            });
            if let Some(face) = store.faces.get_mut(fk) {
                face.outer_wire = wire_key;
            }
            face_keys.push(fk);
        }

        // Build shell
        let shell_faces: Vec<(FaceKey, Orientation)> = face_keys
            .into_iter()
            .map(|fk| (fk, Orientation::Forward))
            .collect();
        let shell_key = store.shells.insert(BRepShell {
            faces: shell_faces,
            closed: true,
            step_id: None,
        });

        // Build solid
        store.solids.insert(BRepSolid {
            outer_shell: shell_key,
            void_shells: vec![],
        });

        store
    }

    #[test]
    fn test_write_empty_store() {
        let store = BRepStore::new();
        let output = write_step_parametric(&store);
        assert!(output.contains("ISO-10303-21;"));
        assert!(output.contains("HEADER;"));
        assert!(output.contains("DATA;"));
        assert!(output.contains("ENDSEC;"));
        assert!(output.contains("END-ISO-10303-21;"));
        // No vertices → no entities in DATA section
    }

    #[test]
    fn test_write_cube_roundtrip() {
        let store = build_unit_cube();
        let output = write_step_parametric(&store);

        // Should contain expected entity types
        assert!(output.contains("CARTESIAN_POINT"), "should have points");
        assert!(output.contains("DIRECTION"), "should have directions");
        assert!(output.contains("AXIS2_PLACEMENT_3D"), "should have placements");
        assert!(output.contains("PLANE"), "should have planes");
        assert!(output.contains("LINE"), "should have lines");
        assert!(output.contains("EDGE_CURVE"), "should have edge curves");
        assert!(output.contains("ORIENTED_EDGE"), "should have oriented edges");
        assert!(output.contains("EDGE_LOOP"), "should have edge loops");
        assert!(output.contains("FACE_OUTER_BOUND"), "should have face bounds");
        assert!(output.contains("ADVANCED_FACE"), "should have advanced faces");
        assert!(output.contains("CLOSED_SHELL"), "should have closed shell");
        assert!(output.contains("MANIFOLD_SOLID_BREP"), "should have solid");

        // Parse back and verify
        let parse_result = crate::step::parser::parse_exchange(&output);
        assert!(parse_result.is_ok(), "roundtrip parse should succeed: {:?}", parse_result.err());

        let parsed = parse_result.unwrap();
        // Should have a reasonable number of entities
        assert!(parsed.entities.len() >= 40, "expected >= 40 entities, got {}", parsed.entities.len());

        // Verify key entity types are present in the parsed output
        let has_type = |name: &str| -> bool {
            parsed.entities.iter().any(|(_, r)| r.name == name)
        };
        assert!(has_type("CARTESIAN_POINT"));
        assert!(has_type("DIRECTION"));
        assert!(has_type("PLANE"));
        assert!(has_type("MANIFOLD_SOLID_BREP"));
    }

    #[test]
    fn test_fmt_real() {
        // Zero
        assert!(fmt_real(0.0).starts_with("0."));
        // Simple values
        let s = fmt_real(1.5);
        assert!(s.contains("1.5") || s.contains("1.500"), "got {}", s);
        let s = fmt_real(3.14159265358979);
        assert!(!s.is_empty());
        // Negative
        let s = fmt_real(-2.5);
        assert!(s.starts_with('-'), "got {}", s);
    }

    #[test]
    fn test_fmt_point() {
        let pt = PVec3::new(1.0, 2.0, 3.0);
        let s = fmt_point(pt);
        assert!(s.starts_with('('));
        assert!(s.ends_with(')'));
        assert!(s.contains("1."));
    }

    #[test]
    fn test_circle_surface_roundtrip() {
        // Test exporting a cylindrical surface with a circle edge.
        // Vertices MUST be distinct points on the circle (both in XY plane)
        // so that normalize_edge_curve_to_vertices wraps the Circle in a Trimmed
        // with a non-zero arc — the curve stored in the edge will be Trimmed.
        let mut store = BRepStore::new();
        let tol = 1e-6;

        // Two distinct points on the unit circle in the XY plane
        let v0 = store.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), tol); // angle 0
        let v1 = store.find_or_add_vertex(PVec3::new(0.0, 1.0, 0.0), tol); // angle pi/2

        let surface = SurfaceGeom::Cylinder {
            origin: PVec3::ZERO,
            axis: PVec3::Z,
            radius: 1.0,
            x_dir: PVec3::X,
            y_dir: PVec3::Y,
        };
        let fk = store.add_face(surface, tol);

        // Add a circle edge — normalize_edge_curve_to_vertices will wrap
        // it in CurveGeom::Trimmed { basis: Circle, t_min, t_max }.
        let curve = CurveGeom::Circle {
            center: PVec3::ZERO,
            axis: PVec3::Z,
            radius: 1.0,
            x_dir: PVec3::X,
            y_dir: PVec3::Y,
        };
        let pc = rc3d_shape::geom::curve2d::Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let ek = store.add_edge_with_pcurve(v0, v1, curve.clone(), tol, fk, pc, true);
        let edge = store.edges.get(ek).unwrap();

        // Build wire
        let edge_orient = if edge.v_low == v0 { Orientation::Forward } else { Orientation::Reversed };
        let wk = store.wires.insert(BRepWire {
            edges: vec![(ek, edge_orient)],
        });
        if let Some(face) = store.faces.get_mut(fk) {
            face.outer_wire = wk;
        }

        let sk = store.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        store.solids.insert(BRepSolid {
            outer_shell: sk,
            void_shells: vec![],
        });

        let output = write_step_parametric(&store);
        assert!(output.contains("CIRCLE"));
        assert!(output.contains("CYLINDRICAL_SURFACE"));
        assert!(output.contains("OPEN_SHELL"));

        // Verify parseable
        let parsed = crate::step::parser::parse_exchange(&output);
        assert!(parsed.is_ok(), "parse should succeed: {:?}", parsed.err());
    }
}
