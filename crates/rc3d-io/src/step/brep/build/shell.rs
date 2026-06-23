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
/// OCC reference format for a sphere of radius 5 with axis Z:
/// - 2 vertices: south pole (0,0,-5), north pole (0,0,5)
/// - 1 curve: great circle in XZ plane, center=(0,0,0), axis=(0,1,0), radius=5
/// - Seam edge at U=0: uses great circle trimmed t=[-π/2, π/2]
/// - 2 degenerated edges (flags 0101100): zero-length 3D curves at poles
/// - Wire: [seam_forward, degen_south, seam_reversed, degen_north]
pub(crate) fn build_sphere_pole_wire(
    pole_anchor: PVec3,
    surface: &SurfaceGeom,
    face_key: FaceKey,
    ctx: &mut ShellBuildCtx<'_>,
) -> WireKey {
    let tol = ctx.tol;
    let (center, radius) = match surface {
        SurfaceGeom::Sphere { center, radius } => (*center, *radius),
        _ => return build_vertex_loop_wire(pole_anchor, surface, face_key, ctx),
    };

    // The pole axis direction (from center toward the given pole).
    let pole_dir = (pole_anchor - center).normalize();
    let south = center - pole_dir * radius;
    let north = center + pole_dir * radius;

    // Create both pole vertices.
    let v_south = ctx.reg.find_or_add_vertex(south, tol);
    let v_north = ctx.reg.find_or_add_vertex(north, tol);

    // Great circle passing through both poles — centered at sphere center,
    // axis perpendicular to the pole axis (so the circle lies in a plane
    // containing the poles).
    let (gc_x, gc_y) = rc3d_shape::geom::curve_eval::build_ortho_axes(pole_dir);
    let great_circle = CurveGeom::Circle {
        center,
        axis: gc_x.cross(gc_y).normalize(),  // = pole_dir
        radius,
        x_dir: gc_x,
        y_dir: gc_y,
    };
    // Seam: half-circle arc from south pole to north pole.
    // On the great circle: south pole at t = -π/2, north pole at t = +π/2.
    let seam_curve = CurveGeom::Trimmed {
        basis: Box::new(great_circle),
        t_min: -std::f64::consts::FRAC_PI_2,
        t_max: std::f64::consts::FRAC_PI_2,
    };

    // Seam pcurve: at U=0, V runs from V_south to V_north.
    let seam_pc = Curve2d::Line {
        origin: (0.0, -std::f64::consts::FRAC_PI_2),
        direction: (0.0, std::f64::consts::PI),
    };

    let ek_seam = ctx.reg.add_seam_edge(
        v_south, v_north, seam_curve, tol, face_key, seam_pc, true,
    );

    // Degenerated edges at each pole: 3D curve is a zero-length line
    // (the CAD kernel ignores it due to the degeneracy flag).
    // 2D pcurve must span the U range [0, 2π] at the pole's V.
    let degen_south = add_degenerated_edge_at_pole(
        v_south,
        (0.0, -std::f64::consts::FRAC_PI_2),
        (std::f64::consts::TAU, -std::f64::consts::FRAC_PI_2),
        south, tol, face_key, ctx.reg,
    );
    let degen_north = add_degenerated_edge_at_pole(
        v_north,
        (std::f64::consts::TAU, std::f64::consts::FRAC_PI_2),
        (0.0, std::f64::consts::FRAC_PI_2),
        north, tol, face_key, ctx.reg,
    );

    // Register on face
    if let Some(face) = ctx.reg.faces.get_mut(face_key) {
        if !face.seam_edges.contains(&ek_seam) { face.seam_edges.push(ek_seam); }
        if !face.degenerated_edges.contains(&degen_south) { face.degenerated_edges.push(degen_south); }
        if !face.degenerated_edges.contains(&degen_north) { face.degenerated_edges.push(degen_north); }
    }

    // Wire: [seam_forward, degen_south, seam_reversed, degen_north]
    ctx.reg.wires.insert(BRepWire {
        edges: vec![
            (ek_seam, Orientation::Forward),
            (degen_south, Orientation::Forward),
            (ek_seam, Orientation::Reversed),
            (degen_north, Orientation::Forward),
        ],
    })
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
