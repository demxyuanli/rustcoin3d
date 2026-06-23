use super::*;
use super::pcurve::resolve_edge_pcurve;
pub(crate) fn build_shell_from_step(shell: &topology::StepShell, ctx: &mut ShellBuildCtx<'_>) -> Option<ShellKey> {
    let mut face_keys = Vec::new();

    for face_data in &shell.faces {
        let (surface, trim_range) = resolve_face_surface(
            face_data,
            ctx.entities,
            ctx.options,
            ctx.skipped_faces,
            ctx.geometry_fallback_count,
        )?;
        let surface_id = face_data.surface_id;

        let temp_wire = ctx.reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = ctx.reg.faces.insert(BRepFace {
            surface: surface.clone(),
            outer_wire: temp_wire,
            inner_wires: vec![],
            same_sense: face_data.same_sense,
            tolerance: ctx.tol,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let mut wire_keys: Vec<(WireKey, bool)> = Vec::new();

        for bloop in &face_data.bounds {
            if let Some(anchor) = bloop.vertex_loop_point {
                let wk = if matches!(&surface, SurfaceGeom::Sphere { .. }) {
                    build_sphere_pole_wire(anchor, &surface, face_key, ctx)
                } else {
                    build_vertex_loop_wire(anchor, &surface, face_key, ctx)
                };
                wire_keys.push((wk, true));
                continue;
            }

            let mut loop_edges = Vec::new();

            for edge_data in &bloop.edges {
                let curve = match resolve_edge_curve(
                    edge_data,
                    ctx.entities,
                    ctx.options,
                    ctx.skipped_edges,
                    ctx.geometry_fallback_count,
                ) {
                    Some(c) => c,
                    None => {
                        log::warn!("[BRep] shell {:?}: skipping edge (unresolved curve)", shell.id);
                        continue;
                    }
                };

                let v_start = ctx.reg.find_or_add_vertex(edge_data.start, ctx.tol);
                let v_end = ctx.reg.find_or_add_vertex(edge_data.end, ctx.tol);

                let (v0, v1) = if edge_data.reversed {
                    (v_end, v_start)
                } else {
                    (v_start, v_end)
                };

                let (v_lo, v_hi) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                let p_lo = ctx.reg.vertices.get(v_lo).unwrap().position;
                let p_hi = ctx.reg.vertices.get(v_hi).unwrap().position;
                let curve = normalize_edge_curve_to_vertices(curve, p_lo, p_hi, ctx.tol);

                let pcurve = resolve_edge_pcurve(
                    &curve,
                    &surface,
                    edge_data.curve_id,
                    surface_id,
                    ctx.entities,
                    ctx.tol,
                );

                let ek = ctx.reg.add_edge_with_pcurve(v0, v1, curve, ctx.tol, face_key,
                    rc3d_shape::geom::curve2d::Curve2d::from_pcurve_3d(&pcurve), true);
                let (v_lo_key, _v_hi_key) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                let wire_orient = if v0 == v_lo_key {
                    Orientation::Forward
                } else {
                    Orientation::Reversed
                };
                loop_edges.push((ek, wire_orient));
            }

            if !loop_edges.is_empty() {
                let wk = ctx.reg.wires.insert(BRepWire { edges: loop_edges });
                wire_keys.push((wk, bloop.bound_forward));
            }
        }

        if wire_keys.is_empty() {
            *ctx.skipped_faces += 1;
            continue;
        }

        // Sort: FACE_OUTER_BOUND (bound_forward==true) first, FACE_BOUND inner loops after
        wire_keys.sort_by(|a, b| b.1.cmp(&a.1));
        let sorted_wires: Vec<WireKey> = wire_keys.into_iter().map(|(wk, _)| wk).collect();

        if let Some(face) = ctx.reg.faces.get_mut(face_key) {
            if let Some(&outer) = sorted_wires.first() {
                face.outer_wire = outer;
                face.inner_wires = if sorted_wires.len() > 1 {
                    sorted_wires[1..].to_vec()
                } else {
                    vec![]
                };
            }
            if let Some(fid) = face_data.face_id {
                if let Some(&rgb) = ctx.face_colors.get(&fid) {
                    face.color = Some(rgb);
                }
            }
        }

        // Store RECTANGULAR_TRIMMED_SURFACE trim range for mesh-time UV clamping
        if let Some(trim) = trim_range {
            ctx.reg.trim_ranges.insert(face_key, trim);
        }

        // Face orientation in shell comes from STEP CLOSED_SHELL ORIENTED_FACE flag.
        // oriented_forward=true: face is used with its natural orientation in the solid.
        // oriented_forward=false: face is reversed (flipped) relative to its natural orientation.
        // The natural orientation already incorporates face.same_sense (face vs surface normal).
        let face_orient = if face_data.oriented_forward {
            *ctx.oriented_forward_faces += 1;
            Orientation::Forward
        } else {
            *ctx.oriented_reversed_faces += 1;
            Orientation::Reversed
        };
        face_keys.push((face_key, face_orient));
    }

    if face_keys.is_empty() {
        return None;
    }

    Some(ctx.reg.shells.insert(BRepShell {
        faces: face_keys,
        closed: shell.closed,
        step_id: Some(shell.id),
    }))
}

/// Build a complete single-face sphere topology from a VERTEX_LOOP at one pole.
///
/// A sphere parameterized as U=[0,2π] (longitude), V=[0,π] (latitude) needs:
/// - A seam edge at U=0 connecting the two poles (half-circle meridian)
/// - Degenerated edges at both poles
/// - A wire: [seam_forward, degenerated_north, seam_reversed, degenerated_south]
///
/// The VERTEX_LOOP gives us one pole (typically south, V=π).
/// We create the opposite pole and build the full cyclic boundary.
pub(crate) fn build_sphere_pole_wire(
    pole_anchor: PVec3,
    surface: &SurfaceGeom,
    face_key: FaceKey,
    ctx: &mut ShellBuildCtx<'_>,
) -> WireKey {
    let tol = ctx.tol;
    let (pole_given, pole_opposite) = match surface {
        SurfaceGeom::Sphere { center, radius } => {
            // The pole_anchor is one pole. The opposite pole is center + (center - anchor).
            let pole_dir = (pole_anchor - center).normalize();
            let given = center + pole_dir * *radius;
            let opposite = center - pole_dir * *radius;
            (given, opposite)
        }
        _ => {
            return build_vertex_loop_wire(pole_anchor, surface, face_key, ctx);
        }
    };

    // Create vertices at both poles
    let v_pole_given = ctx.reg.find_or_add_vertex(pole_given, tol);
    let v_pole_opposite = ctx.reg.find_or_add_vertex(pole_opposite, tol);

    // Create the seam edge: a half-circle from given pole to opposite pole.
    // This is the meridian at U=0 (or U=2π).
    let (x_dir, y_dir) = rc3d_shape::geom::curve_eval::build_ortho_axes(
        (pole_opposite - pole_given).normalize()
    );
    let seam_center = (pole_given + pole_opposite) * 0.5;
    let seam_curve = CurveGeom::Circle {
        center: seam_center,
        axis: y_dir,
        radius: (pole_opposite - pole_given).length() * 0.5,
        x_dir,
        y_dir: y_dir.cross(x_dir).normalize(),
    };
    // The seam is a trimmed half-circle arc from pole_given to pole_opposite
    let seam_trim = CurveGeom::Trimmed {
        basis: Box::new(seam_curve),
        t_min: 0.0,
        t_max: std::f64::consts::PI,
    };

    // Build a synthetic 2D pcurve for the seam edge on the sphere face
    let pcurve_2d = build_sphere_seam_pcurve(surface, pole_given, pole_opposite, tol);

    let ek_seam = ctx.reg.add_seam_edge(
        v_pole_given, v_pole_opposite, seam_trim, tol, face_key,
        pcurve_2d, true,
    );

    // Create degenerated edges at each pole
    let uv_given = surface.project(pole_given)
        .or_else(|| surface.inverse_native_uv(pole_given, tol.max(1e-3)))
        .unwrap_or((0.0, std::f64::consts::PI)); // south pole V=π
    let uv_opposite = surface.project(pole_opposite)
        .or_else(|| surface.inverse_native_uv(pole_opposite, tol.max(1e-3)))
        .unwrap_or((0.0, 0.0)); // north pole V=0

    let ek_degen_given = add_degenerated_edge_at_pole(
        v_pole_given, uv_given, uv_given, pole_given, tol, face_key, ctx.reg,
    );
    let ek_degen_opposite = add_degenerated_edge_at_pole(
        v_pole_opposite, uv_opposite, uv_opposite, pole_opposite, tol, face_key, ctx.reg,
    );

    // Register degenerated edges on the face
    if let Some(face) = ctx.reg.faces.get_mut(face_key) {
        if !face.degenerated_edges.contains(&ek_degen_given) {
            face.degenerated_edges.push(ek_degen_given);
        }
        if !face.degenerated_edges.contains(&ek_degen_opposite) {
            face.degenerated_edges.push(ek_degen_opposite);
        }
        if !face.seam_edges.contains(&ek_seam) {
            face.seam_edges.push(ek_seam);
        }
    }

    // Wire: [seam_forward, degen_opposite, seam_reversed, degen_given]
    ctx.reg.wires.insert(BRepWire {
        edges: vec![
            (ek_seam, Orientation::Forward),
            (ek_degen_opposite, Orientation::Forward),
            (ek_seam, Orientation::Reversed),
            (ek_degen_given, Orientation::Forward),
        ],
    })
}

/// Build a synthetic 2D Line pcurve for the sphere seam edge.
/// The seam follows the U=0 isoparametric line from V=π (south) to V=0 (north).
fn build_sphere_seam_pcurve(
    _surface: &SurfaceGeom,
    pole_given: PVec3,
    pole_opposite: PVec3,
    _tol: Real,
) -> Curve2d {
    // The seam at U=0: V goes from π (south) to 0 (north)
    // Determine which pole is which by looking at the axis
    let _chord = pole_opposite - pole_given;
    // Use a 2D Line pcurve: (U=0, V) from V=max to V=min
    Curve2d::Line {
        origin: (0.0, std::f64::consts::PI),  // start at south pole V=π
        direction: (0.0, -std::f64::consts::PI), // go to north pole V=0
    }
}

/// OCC StepToTopoDS_TranslateVertexLoop: wire with one degenerated edge at the loop vertex.
pub(crate) fn build_vertex_loop_wire(
    anchor: PVec3,
    surface: &SurfaceGeom,
    face_key: FaceKey,
    ctx: &mut ShellBuildCtx<'_>,
) -> WireKey {
    let vk = ctx.reg.find_or_add_vertex(anchor, ctx.tol);
    let inv_tol = ctx.tol.max(1e-3);
    let uv = surface
        .project(anchor)
        .or_else(|| surface.inverse_native_uv(anchor, inv_tol))
        .unwrap_or((0.0, 0.0));
    let ek = add_degenerated_edge_at_pole(vk, uv, uv, anchor, ctx.tol, face_key, ctx.reg);
    if let Some(face) = ctx.reg.faces.get_mut(face_key) {
        if !face.degenerated_edges.contains(&ek) {
            face.degenerated_edges.push(ek);
        }
    }
    ctx.reg.wires.insert(BRepWire {
        edges: vec![(ek, Orientation::Forward)],
    })
}
