use super::*;
use super::pcurve::resolve_edge_pcurve;
pub(crate) fn build_shell_from_step(shell: &topology::StepShell, ctx: &mut ShellBuildCtx<'_>) -> Option<ShellKey> {
    let mut face_keys = Vec::new();

    for face_data in &shell.faces {
        let surface = resolve_face_surface(
            face_data,
            ctx.entities,
            ctx.options,
            ctx.skipped_faces,
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

        let mut wire_keys = Vec::new();

        for bloop in &face_data.bounds {
            if let Some(anchor) = bloop.vertex_loop_point {
                let wk = build_vertex_loop_wire(anchor, &surface, face_key, ctx);
                wire_keys.push(wk);
                continue;
            }

            let mut loop_edges = Vec::new();

            for edge_data in &bloop.edges {
                let curve = match resolve_edge_curve(
                    edge_data,
                    ctx.entities,
                    ctx.options,
                    ctx.skipped_edges,
                ) {
                    Some(c) => c,
                    None => continue,
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
                    rc3d_shape::geom::curve2d::Curve2d::from_pcurve_3d(&pcurve));
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
                wire_keys.push(wk);
            }
        }

        if wire_keys.is_empty() {
            *ctx.skipped_faces += 1;
            continue;
        }

        if let Some(face) = ctx.reg.faces.get_mut(face_key) {
            if let Some(&outer) = wire_keys.first() {
                face.outer_wire = outer;
                face.inner_wires = if wire_keys.len() > 1 {
                    wire_keys[1..].to_vec()
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
        closed: shell.faces.len() >= 4,
        step_id: Some(shell.id),
    }))
}

/// OCC StepToTopoDS_TranslateVertexLoop: wire with one degenerated edge at the loop vertex.
pub(crate) fn build_vertex_loop_wire(
    anchor: Vec3,
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
