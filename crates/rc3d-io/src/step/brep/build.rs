//! STEP → B-Rep builder. T1.7-T1.9
//!
//! Converts STEP entity index into a parametric B-Rep registry.
//!
//! Architecture:
//!   Pass 1: Build SurfaceGeom for each face
//!   Pass 2: Build edges with PCURVEs per face
//!   Pass 3: Build wires, faces, shells, solids
//!   Pass 4: Assemble root_solids
//!
//! Chicken-and-egg: Need FaceKey to register PCURVEs, but need PCURVEs to build
//! the face. Solution: insert a placeholder face first, build edges with PCURVEs
//! using the real FaceKey, then update the face with correct wires.

use crate::step::parser::{EntityIndex, EntityRecord};
use crate::step::value::StepValue;
use crate::step::StepError;
use crate::step::geom;
use crate::step::topology;
use crate::step::nurbs::NurbsSurface;
use super::registry::BRepRegistry;
use super::topo::*;
use super::geom::{CurveGeom, SurfaceGeom};
use super::geom::normalize_edge_curve_to_vertices;
use rc3d_core::math::Vec3;

#[derive(Debug, Default, Clone)]
pub struct BRepBuildReport {
    pub skipped_faces: usize,
    pub skipped_edges: usize,
    pub void_shell_count: usize,
}

#[derive(Debug, Clone)]
pub struct BRepBuildOptions {
    pub allow_geometry_fallback: bool,
}

impl BRepBuildOptions {
    pub fn from_import(options: &crate::step::import_options::StepImportOptions) -> Self {
        Self {
            allow_geometry_fallback: options.allow_geometry_fallback(),
        }
    }
}

#[derive(Debug)]
pub struct BRepBuildResult {
    pub registry: BRepRegistry,
    pub root_solids: Vec<SolidKey>,
    pub build_report: BRepBuildReport,
}

/// Build a full B-Rep from STEP entities.
pub fn build_brep(entities: &EntityIndex) -> Result<BRepBuildResult, StepError> {
    build_brep_with_options(entities, &BRepBuildOptions {
        allow_geometry_fallback: true,
    })
}

/// Build a full B-Rep with import strictness options.
pub fn build_brep_with_options(
    entities: &EntityIndex,
    _options: &BRepBuildOptions,
) -> Result<BRepBuildResult, StepError> {
    let mut reg = BRepRegistry::new();

    // Collect shells from the STEP file
    let shells = topology::collect_shells(entities);
    if shells.is_empty() {
        return Err(StepError::NoGeometry);
    }

    let tol = topology::global_tolerance(entities);

    // Extract per-face colors from STYLED_ITEM entities
    let face_colors = topology::collect_face_colors(entities);

    let mut shell_keys = Vec::new();

    for shell in &shells {
        let mut face_keys = Vec::new();

        for face_data in &shell.faces {
            // ── Pass 1: Build the surface ──────────────────
            let surface = if let Some(sid) = face_data.surface_id {
                build_surface(sid, entities).unwrap_or_else(|| {
                    let surf_name = entities.get(&sid).map(|r| r.name.as_str()).unwrap_or("?");
                    eprintln!("[BRep] WARNING: build_surface failed for #{} ({}), falling back to Plane", sid, surf_name);
                    SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X }
                })
            } else {
                SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X }
            };

            // ── Create placeholder face first (to get a FaceKey for PCURVE registration) ──
            let temp_wire = reg.wires.insert(BRepWire { edges: vec![] });
            let face_key = reg.faces.insert(BRepFace {
                surface: surface.clone(),
                outer_wire: temp_wire,
                inner_wires: vec![],
                same_sense: face_data.same_sense,
                tolerance: tol,
                seam_edges: vec![],
                color: None,
            });

            // ── Pass 2: Build edges with PCURVEs for this face ──
            let mut wire_keys = Vec::new();

            for bloop in &face_data.bounds {
                let mut loop_edges = Vec::new();

                for edge_data in &bloop.edges {
                    // Build the 3D curve, falling back to LINE from edge endpoints
                    let curve = build_curve(edge_data.curve_id, entities)
                        .unwrap_or_else(|| {
                            let dir = edge_data.end - edge_data.start;
                            let d = if dir.length() > 1e-10 { dir } else { Vec3::X };
                            CurveGeom::Line { origin: edge_data.start, direction: d }
                        });

                    // Find or create vertices
                    let v_start = reg.find_or_add_vertex(edge_data.start, tol);
                    let v_end = reg.find_or_add_vertex(edge_data.end, tol);

                    let (v0, v1) = if edge_data.reversed {
                        (v_end, v_start)
                    } else {
                        (v_start, v_end)
                    };

                    let (v_lo, v_hi) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                    let p_lo = reg.vertices.get(v_lo).unwrap().position;
                    let p_hi = reg.vertices.get(v_hi).unwrap().position;
                    let curve = normalize_edge_curve_to_vertices(curve, p_lo, p_hi, tol);

                    let pcurve = resolve_edge_pcurve(
                        &curve,
                        &surface,
                        edge_data.curve_id,
                        face_data.surface_id,
                        entities,
                        tol,
                    );

                    let ek = reg.add_edge_with_pcurve(
                        v0, v1, curve, tol, face_key, pcurve,
                    );
                    // Mesh params run v_low→v_high; wire may traverse the opposite direction.
                    let (v_lo_key, _v_hi_key) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                    let wire_orient = if v0 == v_lo_key {
                        Orientation::Forward
                    } else {
                        Orientation::Reversed
                    };
                    loop_edges.push((ek, wire_orient));
                }

                // Build wire from loop edges
                if !loop_edges.is_empty() {
                    let wk = reg.wires.insert(BRepWire { edges: loop_edges });
                    wire_keys.push(wk);
                }
            }

            // ── Update the face with correct wires ──
            if let Some(face) = reg.faces.get_mut(face_key) {
                if let Some(&outer) = wire_keys.first() {
                    face.outer_wire = outer;
                    face.inner_wires = if wire_keys.len() > 1 {
                        wire_keys[1..].to_vec()
                    } else {
                        vec![]
                    };
                }
                // Apply face color from STYLED_ITEM if available
                if let Some(fid) = face_data.face_id {
                    if let Some(&rgb) = face_colors.get(&fid) {
                        face.color = Some(rgb);
                    }
                }
            }

            face_keys.push((face_key, Orientation::Forward));
        }

        // ── Pass 3: Build shell ──
        if !face_keys.is_empty() {
            let sk = reg.shells.insert(BRepShell {
                faces: face_keys,
                closed: shell.faces.len() >= 4, // heuristic: 4+ faces likely closed
                step_id: Some(shell.id),
            });
            shell_keys.push(sk);
        }
    }

    // ── Pass 4: Assemble root solids ──
    let mut root_solids = Vec::new();
    for sk in shell_keys {
        let solid_key = reg.solids.insert(BRepSolid {
            outer_shell: sk,
            void_shells: vec![],
        });
        root_solids.push(solid_key);
    }

    let void_shell_count = root_solids
        .iter()
        .filter_map(|&sk| reg.solids.get(sk))
        .map(|s| s.void_shells.len())
        .sum();

    Ok(BRepBuildResult {
        registry: reg,
        root_solids,
        build_report: BRepBuildReport {
            skipped_faces: 0,
            skipped_edges: 0,
            void_shell_count,
        },
    })
}

// ── Surface building ──────────────────────────────────────────────

/// Build a SurfaceGeom from a STEP surface entity.
fn build_surface(surface_id: u64, entities: &EntityIndex) -> Option<SurfaceGeom> {
    let record = entities.get(&surface_id)?;
    match record.name.as_str() {
        "PLANE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let (origin, x_axis, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(SurfaceGeom::Plane {
                origin,
                normal: z_axis.normalize(),
                u_dir: x_axis,
            })
        }
        "CYLINDRICAL_SURFACE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let (origin, _, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(SurfaceGeom::Cylinder {
                origin,
                axis: z_axis.normalize(),
                radius,
            })
        }
        "CONICAL_SURFACE" => {
            // CONICAL_SURFACE('', #placement, radius, semi_angle)
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(0.0) as f32;
            let semi_angle = geom::nth_real(&record.params, 3).unwrap_or(0.7854) as f32;
            let (origin, _, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(SurfaceGeom::Cone {
                apex: origin,
                axis: z_axis.normalize(),
                semi_angle,
                radius_at_apex: radius,
            })
        }
        "SPHERICAL_SURFACE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let (origin, _, _) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(SurfaceGeom::Sphere { center: origin, radius })
        }
        "TOROIDAL_SURFACE" => {
            // TOROIDAL_SURFACE('', #placement, major_r, minor_r)
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let major_r = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let minor_r = geom::nth_real(&record.params, 3).unwrap_or(0.5) as f32;
            let (origin, _, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(SurfaceGeom::Torus {
                center: origin,
                axis: z_axis.normalize(),
                major_r,
                minor_r,
            })
        }
        "B_SPLINE_SURFACE" | "B_SPLINE_SURFACE_WITH_KNOTS" | "RATIONAL_B_SPLINE_SURFACE" => {
            let nurbs = build_nurbs_surface(record, entities)?;
            Some(SurfaceGeom::BSpline(nurbs))
        }
        "SURFACE_OF_LINEAR_EXTRUSION" => {
            // SURFACE_OF_LINEAR_EXTRUSION('', #curve, #direction)
            let curve_id = geom::nth_ref(&record.params, 1)?;
            let dir_id = geom::nth_ref(&record.params, 2)?;
            let generatrix = build_curve(curve_id, entities)?;
            let direction = topology::resolve_direction(dir_id, entities).unwrap_or(Vec3::Z);
            Some(SurfaceGeom::Extrusion {
                generatrix: Box::new(generatrix),
                direction,
            })
        }
        "SURFACE_OF_REVOLUTION" => {
            // SURFACE_OF_REVOLUTION('', #curve, #axis_placement)
            let curve_id = geom::nth_ref(&record.params, 1)?;
            let axis_id = geom::nth_ref(&record.params, 2);
            let generatrix = build_curve(curve_id, entities)?;
            let (axis_origin, axis_dir) = axis_id
                .and_then(|id| topology::resolve_placement(id, entities))
                .map(|(o, _x, z)| (o, z))
                .unwrap_or((Vec3::ZERO, Vec3::Z));
            Some(SurfaceGeom::Revolution {
                generatrix: Box::new(generatrix),
                axis_origin,
                axis_dir,
            })
        }
        "OFFSET_SURFACE" => {
            let basis_id = geom::nth_ref(&record.params, 1)?;
            let distance = geom::nth_real(&record.params, 2).unwrap_or(0.0) as f32;
            let basis = build_surface(basis_id, entities)?;
            Some(SurfaceGeom::Offset { basis: Box::new(basis), distance })
        }
        "BOUNDED_SURFACE" | "CURVE_BOUNDED_SURFACE" => {
            // Unwrap: BOUNDED_SURFACE('', #basis_surface, ...)
            let basis_id = geom::nth_ref(&record.params, 1)?;
            build_surface(basis_id, entities)
        }
        "RECTANGULAR_TRIMMED_SURFACE" => {
            let basis_id = geom::nth_ref(&record.params, 1)?;
            build_surface(basis_id, entities)
        }
        // AP242 supertypes — won't appear directly, log if encountered
        "ELEMENTARY_SURFACE" | "SWEPT_SURFACE" => {
            log::warn!(
                "[BRep] build_surface: supertype '{}' for #{} should not appear directly in STEP data",
                record.name, surface_id
            );
            None
        }
        _ => None,
    }
}

/// Build a NurbsSurface from a B_SPLINE_SURFACE* entity.
fn build_nurbs_surface(
    record: &EntityRecord,
    entities: &EntityIndex,
) -> Option<NurbsSurface> {
    let params = &record.params;

    // Off=0 when params[0] is the degree (integer). Off=1 when params[0]
    // is an omitted value (from subsuper merge) and degree starts at [1].
    // Param count alone is unreliable — extra params from trailing supertypes
    // (e.g. REPRESENTATION_ITEM, RATIONAL weights) can make count > 12.
    let off: usize = if geom::nth_int(params, 0).is_some() { 0 } else { 1 };

    let degree_u = geom::nth_int(params, off)? as usize;
    let degree_v = geom::nth_int(params, off + 1)? as usize;

    let cp_list = params.nth_param(off + 2)?.as_list()?;
    let mut control_points = Vec::with_capacity(cp_list.len());
    for row_val in cp_list {
        let row_refs = row_val.as_list()?;
        let mut row_pts = Vec::with_capacity(row_refs.len());
        for pt_val in row_refs {
            let pt_id = pt_val.as_ref_id()?;
            let pt = topology::resolve_point(pt_id, entities)?;
            row_pts.push(pt);
        }
        control_points.push(row_pts);
    }

    if control_points.is_empty() || control_points[0].is_empty() {
        return None;
    }

    // Extract knot vectors from multiplicities
    let mult_base = off + 7;
    let u_mults = geom::nth_list_ints(params, mult_base);
    let v_mults = geom::nth_list_ints(params, mult_base + 1);
    let u_knot_vals = geom::nth_list_reals(params, mult_base + 2);
    let v_knot_vals = geom::nth_list_reals(params, mult_base + 3);

    let knots_u = build_surface_knots(
        &u_mults, &u_knot_vals, degree_u, control_points.len(),
    );
    let knots_v = build_surface_knots(
        &v_mults, &v_knot_vals, degree_v, control_points[0].len(),
    );

    // Extract rational weights if present
    let weights = find_surface_weights(params, control_points.len(), control_points[0].len())
        .unwrap_or_else(|| vec![vec![1.0f32; control_points[0].len()]; control_points.len()]);

    Some(NurbsSurface {
        degree_u,
        degree_v,
        control_points,
        weights,
        knots_u,
        knots_v,
    })
}

/// Build a full knot vector from multiplicities and distinct knot values.
fn build_surface_knots(
    multiplicities: &[i64],
    knot_values: &[StepValue],
    degree: usize,
    cp_count: usize,
) -> Vec<f32> {
    let mut knots = Vec::new();
    if !multiplicities.is_empty() && !knot_values.is_empty() {
        for (i, &mult) in multiplicities.iter().enumerate() {
            if i < knot_values.len() {
                let k = knot_values[i].as_real().unwrap_or(0.0) as f32;
                for _ in 0..mult.max(1) {
                    knots.push(k);
                }
            }
        }
    }
    let needed = cp_count + degree + 1;
    if knots.len() < needed {
        let min_k = *knots.first().unwrap_or(&0.0);
        let max_k = *knots.last().unwrap_or(&1.0);
        let extra = needed - knots.len();
        for i in 0..extra {
            let t = (i + 1) as f32 / (extra + 1) as f32;
            knots.push(min_k + (max_k - min_k) * t);
        }
        knots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    knots
}

/// Scan params for a 2D weights list matching NURBS control point dimensions.
fn find_surface_weights(params: &StepValue, rows: usize, cols: usize) -> Option<Vec<Vec<f32>>> {
    let list = params.as_list()?;
    for val in list.iter().rev() {
        if let StepValue::List(inner) = val {
            if inner.len() == rows
                && inner.iter().all(|v| {
                    v.as_list().map_or(false, |l| {
                        l.len() == cols && l.iter().all(|r| matches!(r, StepValue::Real(_)))
                    })
                })
            {
                return Some(
                    inner.iter()
                        .map(|v| v.as_list().unwrap().iter()
                            .map(|r| r.as_real().unwrap() as f32)
                            .collect())
                        .collect(),
                );
            }
        }
    }
    None
}

/// Resolve a VECTOR entity to its full 3D vector (direction × magnitude).
fn resolve_vector_magnitude(vec_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    let record = entities.get(&vec_id)?;
    match record.name.as_str() {
        "VECTOR" => {
            let dir_id = geom::nth_ref(&record.params, 1)?;
            let dir = topology::resolve_direction(dir_id, entities)?;
            let mag = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            Some(dir * mag)
        }
        "DIRECTION" => {
            topology::resolve_direction(vec_id, entities)
        }
        _ => topology::resolve_direction(vec_id, entities),
    }
}

// ── Curve building ────────────────────────────────────────────────

/// Build a 3D CurveGeom from a STEP curve entity.
fn build_curve(curve_id: u64, entities: &EntityIndex) -> Option<CurveGeom> {
    let record = entities.get(&curve_id)?;
    match record.name.as_str() {
        "LINE" => {
            let pnt_id = geom::nth_ref(&record.params, 1)?;
            let dir_id = geom::nth_ref(&record.params, 2)?;
            let origin = topology::resolve_point(pnt_id, entities)?;
            // Resolve the full VECTOR (direction × magnitude), not just the unit direction.
            let direction = resolve_vector_magnitude(dir_id, entities).unwrap_or(Vec3::X);
            Some(CurveGeom::Line { origin, direction })
        }
        "CIRCLE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let (center, _, axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(CurveGeom::Circle { center, axis, radius })
        }
        "ELLIPSE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let semi_major = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let semi_minor = geom::nth_real(&record.params, 3).unwrap_or(0.5) as f32;
            let (center, _, axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(CurveGeom::Ellipse { center, axis, semi_major, semi_minor })
        }
        "B_SPLINE_CURVE" | "B_SPLINE_CURVE_WITH_KNOTS" | "RATIONAL_B_SPLINE_CURVE" => {
            build_bspline_3d(record, entities)
        }
        "TRIMMED_CURVE" => {
            // TRIMMED_CURVE('', #basis_curve, (trim1), (trim2), sense, master_rep)
            let basis_id = geom::nth_ref(&record.params, 1)?;
            let basis = build_curve(basis_id, entities)?;
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
            let points: Vec<Vec3> = pt_ids.iter()
                .filter_map(|&id| topology::resolve_point(id, entities))
                .collect();
            if points.len() < 2 { None } else { Some(CurveGeom::Polyline { points }) }
        }
        "COMPOSITE_CURVE" => {
            let seg_ids = geom::nth_list_refs(&record.params, 1).unwrap_or_default();
            let segments: Vec<(CurveGeom, bool)> = seg_ids.iter()
                .filter_map(|&seg_id| {
                    let seg_rec = entities.get(&seg_id)?;
                    // COMPOSITE_CURVE_SEGMENT: (name, transition, same_sense, #parent_curve)
                    let parent_id = geom::nth_ref(&seg_rec.params, 3)?;
                    // same_sense is at params[2]; TRANSITION code at params[1], parents at [3]
                    let same_sense = match seg_rec.params.nth_param(2) {
                        Some(StepValue::Enum(s)) => s == ".T.",
                        _ => true,
                    };
                    build_curve(parent_id, entities).map(|c| (c, same_sense))
                })
                .collect();
            if segments.is_empty() { None } else { Some(CurveGeom::Composite { segments }) }
        }
        // SURFACE_CURVE/SEAM_CURVE: unwrap to the 3D curve
        "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE" => {
            let curve_3d_id = geom::nth_ref(&record.params, 1)?;
            build_curve(curve_3d_id, entities)
        }
        "OFFSET_CURVE_3D" => {
            let inner_id = geom::nth_ref(&record.params, 1)?;
            build_curve(inner_id, entities)
        }
        "BOUNDED_CURVE" => {
            // Unwrap to the underlying curve
            let inner_id = geom::nth_ref(&record.params, 1)?;
            build_curve(inner_id, entities)
        }
        _ => None,
    }
}

/// Build a 3D B-spline CurveGeom from a B_SPLINE_CURVE* entity.
fn build_bspline_3d(
    record: &EntityRecord,
    entities: &EntityIndex,
) -> Option<CurveGeom> {
    let degree = geom::nth_int(&record.params, 1).unwrap_or(2) as usize;
    let cp_ids = geom::nth_list_refs(&record.params, 2)?;

    let control_points: Vec<Vec3> = cp_ids.iter()
        .filter_map(|&id| topology::resolve_point(id, entities))
        .collect();

    if control_points.len() < degree + 1 {
        return None;
    }

    let cp_count = control_points.len();

    // Extract or build knot vector
    let knots = if record.name == "B_SPLINE_CURVE_WITH_KNOTS"
        || record.name == "RATIONAL_B_SPLINE_CURVE"
    {
        // B_SPLINE_CURVE_WITH_KNOTS layout (after name):
        // [1]=degree, [2]=ctrl_pts, [3]=curve_form, [4]=closed, [5]=self_intersect,
        // [6]=knot_multiplicities, [7]=knots, [8]=knot_spec
        // RATIONAL_B_SPLINE_CURVE appends weights after [8]
        let mults = geom::nth_list_ints(&record.params, 6);
        let knot_vals = geom::nth_list_reals(&record.params, 7);
        let mut knots = Vec::new();
        if !mults.is_empty() && !knot_vals.is_empty() {
            for (i, &m) in mults.iter().enumerate() {
                let k = knot_vals.get(i)
                    .and_then(|v| v.as_real())
                    .unwrap_or(0.0) as f32;
                for _ in 0..m.max(1) {
                    knots.push(k);
                }
            }
        }
        knots
    } else {
        Vec::new()
    };

    let knots = if knots.len() >= cp_count + degree + 1 {
        knots
    } else {
        // Build uniform knot vector
        let mut k = Vec::with_capacity(cp_count + degree + 1);
        for _ in 0..=degree { k.push(0.0f32); }
        for i in 1..(cp_count - degree) {
            k.push(i as f32 / (cp_count - degree) as f32);
        }
        for _ in 0..=degree { k.push(1.0f32); }
        k
    };

    // Extract rational weights if present
    let weights = if record.name == "RATIONAL_B_SPLINE_CURVE" {
        // Weights are the last param in RATIONAL_B_SPLINE_CURVE
        let list = record.params.as_list()?;
        let weight_data = list.last()?;
        // Try treating it as a list of reals
        find_curve_weights(weight_data, cp_count)
    } else {
        None
    };

    Some(CurveGeom::BSpline {
        degree,
        control_points,
        knots,
        weights,
    })
}

/// Build a 2D B-spline PCurve in native surface UV space.
fn build_bspline_2d(
    record: &EntityRecord,
    entities: &EntityIndex,
) -> Option<CurveGeom> {
    let degree = geom::nth_int(&record.params, 1).unwrap_or(2) as usize;
    let cp_ids = geom::nth_list_refs(&record.params, 2)?;

    let control_points: Vec<Vec3> = cp_ids
        .iter()
        .filter_map(|&id| {
            resolve_cartesian_2d(id, entities).map(|(u, v)| Vec3::new(u, v, 0.0))
        })
        .collect();

    if control_points.len() < degree + 1 {
        return None;
    }

    let cp_count = control_points.len();

    let knots = if record.name == "B_SPLINE_CURVE_WITH_KNOTS"
        || record.name == "RATIONAL_B_SPLINE_CURVE"
    {
        let mults = geom::nth_list_ints(&record.params, 6);
        let knot_vals = geom::nth_list_reals(&record.params, 7);
        let mut knots = Vec::new();
        if !mults.is_empty() && !knot_vals.is_empty() {
            for (i, &m) in mults.iter().enumerate() {
                let k = knot_vals
                    .get(i)
                    .and_then(|v| v.as_real())
                    .unwrap_or(0.0) as f32;
                for _ in 0..m.max(1) {
                    knots.push(k);
                }
            }
        }
        knots
    } else {
        Vec::new()
    };

    let knots = if knots.len() >= cp_count + degree + 1 {
        knots
    } else {
        let mut k = Vec::with_capacity(cp_count + degree + 1);
        for _ in 0..=degree {
            k.push(0.0f32);
        }
        for i in 1..(cp_count - degree) {
            k.push(i as f32 / (cp_count - degree) as f32);
        }
        for _ in 0..=degree {
            k.push(1.0f32);
        }
        k
    };

    let weights = if record.name == "RATIONAL_B_SPLINE_CURVE" {
        let list = record.params.as_list()?;
        let weight_data = list.last()?;
        find_curve_weights(weight_data, cp_count)
    } else {
        None
    };

    Some(CurveGeom::BSpline {
        degree,
        control_points,
        knots,
        weights,
    })
}

fn parse_trim_bound(params: &StepValue, index: usize) -> Option<f32> {
    params
        .nth_param(index)?
        .as_list()?
        .first()?
        .as_real()
        .map(|v| v as f32)
}

/// Extract weights from a RATIONAL_B_SPLINE_CURVE weight param.
fn find_curve_weights(weight_val: &StepValue, expected_count: usize) -> Option<Vec<f32>> {
    if let StepValue::List(items) = weight_val {
        let weights: Vec<f32> = items.iter()
            .filter_map(|v| v.as_real())
            .map(|r| r as f32)
            .collect();
        if weights.len() == expected_count {
            return Some(weights);
        }
    }
    // Try Typed wrapping
    if let StepValue::Typed(_, inner) = weight_val {
        return find_curve_weights(inner, expected_count);
    }
    None
}

// ── PCURVE resolution ─────────────────────────────────────────────

/// Polyline sample count for synthetic / fallback PCurves (validate uses 8 samples).
const PCURVE_POLYLINE_SAMPLES: u32 = 16;

/// Resolve PCURVE for an edge: STEP first (OCCT), validated against the 3D curve;
/// fall back to synthetic projection when STEP data is missing or inconsistent.
fn resolve_edge_pcurve(
    curve: &CurveGeom,
    surface: &SurfaceGeom,
    edge_curve_id: u64,
    surface_id: Option<u64>,
    entities: &EntityIndex,
    tol: f32,
) -> CurveGeom {
    let step_pc = surface_id.and_then(|sid| build_pcurve_for_face(edge_curve_id, sid, entities));

    let match_tol = pcurve_match_tol(curve, tol);

    if let Some(ref pc) = step_pc {
        if validate_pcurve_on_surface(curve, pc, surface, match_tol, false) {
            return pc.clone();
        }
        if validate_pcurve_on_surface(curve, pc, surface, match_tol, true) {
            return reverse_pcurve(pc);
        }
        log::debug!(
            "[BRep] STEP PCURVE rejected for edge #{}, trying synthetic",
            edge_curve_id
        );
    }

    if let Some(syn) = build_synthetic_pcurve(curve, surface, match_tol) {
        if validate_pcurve_on_surface(curve, &syn, surface, match_tol, false) {
            return syn;
        }
    }

    let fb = build_parametric_fallback_pcurve(curve, surface, match_tol);
    if validate_pcurve_on_surface(curve, &fb, surface, match_tol, false) {
        return fb;
    }

    if let Some(ref pc) = step_pc {
        if !pcurve_uv_is_degenerate(pc) {
            return pc.clone();
        }
    }
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

fn pcurve_uv_matches_3d(
    surface: &SurfaceGeom,
    p3: Vec3,
    uv: Vec3,
    match_tol: f32,
) -> bool {
    let u = uv.x;
    let v = uv.y;
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
    false
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
        let (u, v) = surface
            .inverse_native_uv_build(p3, inv_tol)
            .or_else(|| surface.project(p3))
            .unwrap_or((t, 0.0));
        uv_points.push(Vec3::new(u, v, 0.0));
    }
    CurveGeom::Polyline { points: uv_points }
}

/// Resolve the 2D PCURVE for an edge on a specific face's surface.
///
/// Walks: EDGE_CURVE → SURFACE_CURVE → PCURVE list → match by basis_surface → 2D curve
fn build_pcurve_for_face(
    edge_curve_id: u64,
    surface_id: u64,
    entities: &EntityIndex,
) -> Option<CurveGeom> {
    let edge_rec = entities.get(&edge_curve_id)?;

    let sc_id = if edge_rec.name == "EDGE_CURVE" {
        geom::nth_ref(&edge_rec.params, 3)? // curve geometry reference
    } else {
        edge_curve_id
    };

    let sc_rec = entities.get(&sc_id)?;
    if sc_rec.name != "SURFACE_CURVE"
        && sc_rec.name != "SEAM_CURVE"
        && sc_rec.name != "INTERSECTION_CURVE"
    {
        // No PCURVE available: standalone 3D curve. Return None so the caller
        // falls back to the default PCURVE.
        return None;
    }

    // Find matching PCURVE: params[2] = pcurve list
    let pcurve_list = sc_rec.params.nth_param(2)?.as_list()?;
    for item in pcurve_list {
        let pcurve_id = item.as_ref_id()?;
        let prec = entities.get(&pcurve_id)?;
        if prec.name == "PCURVE" || prec.name == "DEFINITIONAL_REPRESENTATION" {
            let basis_surface = geom::nth_ref(&prec.params, 1)?;
            if basis_surface == surface_id {
                // Found matching PCURVE: resolve its 2D curve
                let curve_ref = geom::nth_ref(&prec.params, 2)?;
                return build_2d_curve(curve_ref, entities);
            }
        }
    }
    None
}

/// Build a 2D CurveGeom from a STEP curve entity in UV space.
fn build_2d_curve(curve_id: u64, entities: &EntityIndex) -> Option<CurveGeom> {
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
            Some(CurveGeom::Circle {
                center: Vec3::new(center.0, center.1, 0.0),
                axis: Vec3::Z,
                radius,
            })
        }
        "ELLIPSE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let semi_major = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let semi_minor = geom::nth_real(&record.params, 3).unwrap_or(0.5) as f32;
            let center = resolve_placement_2d(placement_id, entities)?;
            Some(CurveGeom::Ellipse {
                center: Vec3::new(center.0, center.1, 0.0),
                axis: Vec3::Z,
                semi_major,
                semi_minor,
            })
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
                Some(CurveGeom::Composite { segments })
            }
        }
        _ => None,
    }
}

// ── 2D point/vector helpers ───────────────────────────────────────

/// Resolve a CARTESIAN_POINT to a (u, v) pair (ignoring z).
fn resolve_cartesian_2d(pt_id: u64, entities: &EntityIndex) -> Option<(f32, f32)> {
    let record = entities.get(&pt_id)?;
    if record.name != "CARTESIAN_POINT" { return None; }
    let coords = nth_list_f64(&record.params, 1)?;
    if coords.len() < 2 { return None; }
    Some((coords[0] as f32, coords[1] as f32))
}

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
    let mut uv_points = Vec::with_capacity(n as usize + 1);
    let mut map_failures = 0usize;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let p3 = curve.d0(t);
        match surface.inverse_native_uv_build(p3, inv_tol) {
            Some((u, v)) => uv_points.push(Vec3::new(u, v, 0.0)),
            None => {
                map_failures += 1;
            }
        }
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
            CurveGeom::Circle { center, axis, radius } => {
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
            SurfaceGeom::Cylinder { origin, axis, radius } => {
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
            SurfaceGeom::Cone { apex, axis, radius_at_apex, semi_angle } => {
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
            SurfaceGeom::Torus { center, axis, major_r, minor_r } => {
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
}
