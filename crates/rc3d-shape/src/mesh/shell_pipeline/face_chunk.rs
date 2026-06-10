use std::collections::HashMap;

use crate::geom::SurfaceGeom;
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, Orientation};
use crate::mesh::algo_factory::FaceMeshAlgo;
use crate::mesh::boundary::SharedBoundaryPool;
use crate::mesh::config::{count_valid_tris_in_range, min_adequate_trim_tris, BRepMeshConfig};
use crate::mesh::diagnostic::format_wire_loop_lines;
use crate::mesh::edge_disc::EdgePolygon;
use crate::mesh::face_dispatch::{algo_from_plan, plan_face_mesh, FaceMeshPlan};
use crate::mesh::face_fill::{
    face_boundary_is_mixed, fill_trimmed, mesh_plane_fan_3d, prefers_native_uv_ruled,
    surface_fill_3d, measure_face_chord_error, surface_is_planar_trim,
    surface_is_revolution_like, FaceMeshRange,
};
use crate::mesh::face_uv::{
    cylinder_loop_needs_uv_rebuild, loops_from_boundary_indices, loops_native_surface_uv,
    repair_cylinder_uv_loops, revolution_boundary_v_collapsed, revolution_loops_from_wire_edges,
    uv_loop_is_degenerate, UvSource,
};
use crate::mesh::fallback_policy::{
    allows_parametric_grid_fallback, allows_trimmed_uv_grid, revolution_fallback_uv_bounds,
    uses_closed_parametric_mesh,
};
use crate::mesh::grid::{mesh_closed_surface, mesh_parametric_grid, mesh_trimmed_uv_grid, mesh_uv_bbox_grid};
use crate::mesh::report::{FaceMeshStats, ShellMeshReport};
use crate::mesh::ruled::{try_ruled_two_wire_mesh, RuledMeshBuffers};
use super::types::{mesh_face_view, ChunkOutput, FaceLoopData};

pub(crate) fn mesh_faces_in_chunk(
    chunk: &[FaceLoopData],
    reg: &BRepStore,
    scaled_config: &BRepMeshConfig,
    shell_diag: f32,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &HashMap<(FaceKey, EdgeKey, usize), usize>,
    face_orient: &HashMap<FaceKey, bool>,
    collect_diag: bool,
    shared_boundary: &SharedBoundaryPool,
) -> ChunkOutput {
    let shared = Some(shared_boundary);
    let mut global_vertices =
        Vec::with_capacity(shared_boundary.count.saturating_add(chunk.len().saturating_mul(32)));
    global_vertices.extend_from_slice(&shared_boundary.vertices);
    let mut global_normals =
        Vec::with_capacity(shared_boundary.count.saturating_add(chunk.len().saturating_mul(32)));
    global_normals.extend_from_slice(&shared_boundary.normals);
    let mut pos_to_idx = shared_boundary.spawn_interior_index();
    let mut all_indices: Vec<i32> = Vec::new();
    let mut face_ranges: Vec<FaceMeshRange> = Vec::new();
    let mut report = ShellMeshReport {
        face_count: chunk.len(),
        shell_diag,
        ..Default::default()
    };
    let mut wire_diag: Vec<String> = Vec::new();
    for fld in chunk {
        let info_face_key = fld.face_key;
        let info_wire_edges = &fld.wire_edges;
        let face_raw = match reg.faces.get(info_face_key) {
            Some(f) => f,
            None => continue,
        };
        let eff_sense = face_orient
            .get(&info_face_key)
            .copied()
            .unwrap_or(face_raw.same_sense);
        let mesh_face_owned = mesh_face_view(face_raw, eff_sense);
        let face = mesh_face_owned.as_ref();

        // Degenerate face guard: a single-edge wire cannot form a closed boundary loop.
        // Two-edge revolution/sphere pole patches are valid (ruled strip / trimmed CDT).
        // Closed parametric meshes (sphere, torus, revolution pole) may have 0-1 edges.
        if info_wire_edges.len() == 1 && !uses_closed_parametric_mesh(reg, face) {
            eprintln!("[mesh face] {:?} SKIP (degenerate: {} edges)", info_face_key, info_wire_edges.len());
            continue;
        }

        if info_wire_edges.is_empty() || uses_closed_parametric_mesh(reg, face) {
            let _tf = std::time::Instant::now();
            let tris_before = all_indices.len() / 4;
            mesh_closed_surface(
        face,
        &scaled_config.face,
        &mut global_vertices,
        &mut global_normals,
                &mut pos_to_idx,
                &mut all_indices,
                shared,
            );
            let tri_count = all_indices.len() / 4 - tris_before;
            let t = _tf.elapsed().as_secs_f32();
            if t > 0.1 { log::debug!("[mesh] closed_surface {:?}: {:.2}s ({} tris)", info_face_key, t, tri_count); }
            report.faces.push(FaceMeshStats {
        face_key: info_face_key,
        tri_count,
        first_tri: tris_before,
        uv_source: UvSource::SurfaceFill,
        max_chord_error: 0.0,
        cdt_constraint_failures: 0,
        grid_fallback: false,
            });
            if tri_count > 0 {
        report.meshed_faces += 1;
            }
            continue;
        }

        let loops = match &fld.loops {
            Some(l) => l.clone(),
            None => {
        // Closed parametric surface — mesh directly
        continue;
            }
        };

        if collect_diag && !info_wire_edges.is_empty() {
            let we_orient: Vec<_> = info_wire_edges.iter().map(|&(ek, ref pis)| (ek, Orientation::Forward, pis.clone())).collect();
            wire_diag.extend(format_wire_loop_lines(
        info_face_key,
        &loops,
        &we_orient,
        edge_boundary_idx,
        &global_vertices,
            ));
        }

        let wire_empty = info_wire_edges.is_empty();
        let mesh_plan = plan_face_mesh(face, &loops, wire_empty, false);
        log::trace!("[BRep mesh] face {:?} plan {:?}", info_face_key, mesh_plan);
        if matches!(mesh_plan, FaceMeshPlan::Failed) {
            continue;
        }
        let prefer_native_cdt = matches!(
            &face.surface,
            SurfaceGeom::Revolution { .. }
        | SurfaceGeom::BSpline(_)
        | SurfaceGeom::Offset { .. }
        | SurfaceGeom::Cylinder { .. }
        | SurfaceGeom::Cone { .. }
        | SurfaceGeom::Torus { .. }
        );
        let mut algo = algo_from_plan(mesh_plan).unwrap_or(FaceMeshAlgo::SurfaceFill3d);
        // Respect FallbackAllowlist: skip face when surface_fill_3d is disallowed
        if algo == FaceMeshAlgo::SurfaceFill3d && !scaled_config.fallback.surface_fill_3d {
            log::debug!("[BRep mesh] face {:?}: surface_fill_3d disallowed by FallbackAllowlist, skip", info_face_key);
            continue;
        }
        if info_wire_edges.len() == 2 && uv_loop_is_degenerate(&loops) {
            algo = FaceMeshAlgo::TrimmedCdt;
        }
        if prefer_native_cdt && loops.is_fillable() {
            algo = FaceMeshAlgo::TrimmedCdt;
        }
        if matches!(face.surface, SurfaceGeom::Plane { .. }) && loops.is_fillable() {
            algo = FaceMeshAlgo::TrimmedCdt;
        }
        let mut used_surface_fill = algo == FaceMeshAlgo::SurfaceFill3d;
        let mut used_parametric_grid = false;
        let mixed_boundary = algo == FaceMeshAlgo::TrimmedCdt
            && !prefer_native_cdt
            && !matches!(face.surface, SurfaceGeom::Plane { .. })
            && face_boundary_is_mixed(&loops, &global_vertices);

        let mut boundary_ordered = Vec::new();
        for &(ek, ref pis) in info_wire_edges {
            for &pi in pis {
        if let Some(gi) = edge_boundary_idx.get(&(info_face_key, ek, pi)).copied() {
            if boundary_ordered.last() != Some(&gi) {
                boundary_ordered.push(gi);
            }
        }
            }
        }
        if boundary_ordered.len() >= 2 && boundary_ordered.first() == boundary_ordered.last() {
            boundary_ordered.pop();
        }

        let wire_len = info_wire_edges.len();
        let multi_wire_rev = wire_len >= 3
            && matches!(face.surface, SurfaceGeom::Revolution { .. })
            && boundary_ordered.len() >= 3;

        // Plane two-wire patches: ruled strip when PCURVE loop is not fillable.
        if wire_len == 2
            && matches!(face.surface, SurfaceGeom::Plane { .. })
            && !loops.is_fillable()
        {
            let ruled = try_ruled_two_wire_mesh(
        info_face_key,
        face,
        info_wire_edges,
        edge_polygons,
        edge_boundary_idx,
        RuledMeshBuffers {
            global_vertices: &mut global_vertices,
            global_normals: &mut global_normals,
            all_indices: &mut all_indices,
                    pos_to_idx: &mut pos_to_idx,
                    shared_boundary: shared,
                },
        &scaled_config.face,
            );
            if ruled.tri_count > 0 {
        let tri_n = ruled.tri_count;
        report.faces.push(FaceMeshStats {
            face_key: info_face_key,
            tri_count: tri_n,
            first_tri: ruled.first_tri,
            uv_source: UvSource::Synthetic,
            max_chord_error: ruled.max_chord_error,
            cdt_constraint_failures: 0,
            grid_fallback: false,
        });
        face_ranges.push(ruled);
        report.meshed_faces += 1;
        log::debug!(
            "[BRep mesh] face {:?}: plane ruled two-wire strip ({} tris)",
            info_face_key,
            tri_n
        );
        continue;
            }
        }

        let tris_before_face = all_indices.len() / 4;
        let chord_reject = (scaled_config.face.deflection_interior * 10.0).max(0.05);
        let min_adequate = min_adequate_trim_tris(&face.surface, wire_len);

        if wire_len == 2 && prefers_native_uv_ruled(&face.surface, info_wire_edges) {
            let ruled = try_ruled_two_wire_mesh(
        info_face_key,
        face,
        info_wire_edges,
        edge_polygons,
        edge_boundary_idx,
        RuledMeshBuffers {
            global_vertices: &mut global_vertices,
            global_normals: &mut global_normals,
            all_indices: &mut all_indices,
                    pos_to_idx: &mut pos_to_idx,
                    shared_boundary: shared,
                },
        &scaled_config.face,
            );
            let valid = count_valid_tris_in_range(
        &all_indices,
        &global_vertices,
        ruled.first_tri,
        ruled.tri_count,
            );
            let offset_min_tris = if matches!(face.surface, SurfaceGeom::Offset { .. }) {
        min_adequate.saturating_mul(8)
            } else {
        0
            };
            if valid >= min_adequate
        && (offset_min_tris == 0 || valid >= offset_min_tris)
            {
        let mut max_chord = ruled.max_chord_error;
        if matches!(face.surface, SurfaceGeom::Offset { .. })
            && !scaled_config.fast_export
        {
            let ruled_range = FaceMeshRange {
                face_key: info_face_key,
                first_tri: ruled.first_tri,
                tri_count: valid,
                boundary_global: ruled.boundary_global.clone(),
                max_chord_error: 0.0,
                cdt_constraint_failures: 0,
            };
            max_chord = measure_face_chord_error(
                face,
                &global_vertices,
                &all_indices,
                &ruled_range,
            );
        }
        report.faces.push(FaceMeshStats {
            face_key: info_face_key,
            tri_count: valid,
            first_tri: ruled.first_tri,
            uv_source: UvSource::Synthetic,
            max_chord_error: max_chord,
            cdt_constraint_failures: 0,
            grid_fallback: false,
        });
        face_ranges.push(FaceMeshRange {
            face_key: info_face_key,
            first_tri: ruled.first_tri,
            tri_count: valid,
            boundary_global: ruled.boundary_global,
            max_chord_error: max_chord,
            cdt_constraint_failures: 0,
        });
        report.meshed_faces += 1;
        log::debug!(
            "[BRep mesh] face {:?}: revolution-like ruled two-wire ({} tris, chord {:.4})",
            info_face_key,
            valid,
            max_chord
        );
        continue;
            }
            if valid >= min_adequate {
        log::debug!(
            "[BRep mesh] face {:?}: early ruled low tris ({}/{}), fall through",
            info_face_key,
            valid,
            offset_min_tris
        );
            }
            all_indices.truncate(tris_before_face * 4);
        }

        // Revolution/Offset faces: rebuild UV loops for CDT (avoid degenerate PCURVE / collapsed V).
        let mut mesh_loops = loops.clone();
        if multi_wire_rev {
            if let Some(s) = revolution_loops_from_wire_edges(
        info_face_key,
        face,
        info_wire_edges,
        edge_polygons,
        edge_boundary_idx,
        &global_vertices,
            ) {
        if !uv_loop_is_degenerate(&s) {
            mesh_loops = s;
            algo = FaceMeshAlgo::TrimmedCdt;
        }
            }
        } else if wire_len == 2 && matches!(face.surface, SurfaceGeom::Revolution { .. }) {
            if let Some(s) = revolution_loops_from_wire_edges(
        info_face_key,
        face,
        info_wire_edges,
        edge_polygons,
        edge_boundary_idx,
        &global_vertices,
            ) {
        if !uv_loop_is_degenerate(&s) {
            mesh_loops = s;
            algo = FaceMeshAlgo::TrimmedCdt;
        }
            }
        } else if matches!(face.surface, SurfaceGeom::Revolution { .. })
            && uv_loop_is_degenerate(&mesh_loops)
            && boundary_ordered.len() >= 3
        {
            if let Some(s) = revolution_loops_from_wire_edges(
        info_face_key,
        face,
        info_wire_edges,
        edge_polygons,
        edge_boundary_idx,
        &global_vertices,
            ) {
        if !uv_loop_is_degenerate(&s) {
            mesh_loops = s;
            algo = FaceMeshAlgo::TrimmedCdt;
        }
            }
        }
        if wire_len == 2
            && matches!(face.surface, SurfaceGeom::Offset { .. })
            && boundary_ordered.len() >= 3
        {
            if let Some(s) = loops_from_boundary_indices(&boundary_ordered, face, &global_vertices) {
        let native = loops_native_surface_uv(&s, face, &global_vertices);
        if !uv_loop_is_degenerate(&native) {
            mesh_loops = native;
            algo = FaceMeshAlgo::TrimmedCdt;
        }
            }
        }
        if matches!(face.surface, SurfaceGeom::Revolution { .. } | SurfaceGeom::Offset { .. })
            && uv_loop_is_degenerate(&mesh_loops)
            && boundary_ordered.len() >= 3
        {
            let syn = loops_from_boundary_indices(&boundary_ordered, face, &global_vertices);
            if let Some(s) = syn {
        if !uv_loop_is_degenerate(&s) {
            mesh_loops = s;
            algo = FaceMeshAlgo::TrimmedCdt;
        }
            }
        }
        if matches!(face.surface, SurfaceGeom::Cylinder { .. })
            && cylinder_loop_needs_uv_rebuild(&mesh_loops, &face.surface)
            && boundary_ordered.len() >= 3
        {
            if let Some(s) = loops_from_boundary_indices(&boundary_ordered, face, &global_vertices)
            {
        let native = repair_cylinder_uv_loops(&s, face, &global_vertices);
        if !uv_loop_is_degenerate(&native) {
            mesh_loops = native;
            algo = FaceMeshAlgo::TrimmedCdt;
            used_surface_fill = false;
        }
            }
        } else if matches!(face.surface, SurfaceGeom::BSpline(_))
            && uv_loop_is_degenerate(&mesh_loops)
            && boundary_ordered.len() >= 3
        {
            if let Some(s) = loops_from_boundary_indices(&boundary_ordered, face, &global_vertices)
            {
        let native = loops_native_surface_uv(&s, face, &global_vertices);
        if !uv_loop_is_degenerate(&native) {
            mesh_loops = native;
            algo = FaceMeshAlgo::TrimmedCdt;
            used_surface_fill = false;
        }
            }
        }
        if !uv_loop_is_degenerate(&mesh_loops) {
            algo = FaceMeshAlgo::TrimmedCdt;
        }

        if wire_len == 1
            && matches!(face.surface, SurfaceGeom::Plane { .. })
            && boundary_ordered.len() >= 3
        {
            let fan = mesh_plane_fan_3d(
        info_face_key,
        &boundary_ordered,
        face,
        &mut global_vertices,
        &mut global_normals,
        &mut all_indices,
        &mut pos_to_idx,
                &scaled_config.face,
                shared,
            );
            if fan.tri_count > 0 {
        let tri_n = fan.tri_count;
        report.faces.push(FaceMeshStats {
            face_key: info_face_key,
            tri_count: tri_n,
            first_tri: fan.first_tri,
            uv_source: loops.uv_source,
            max_chord_error: fan.max_chord_error,
            cdt_constraint_failures: 0,
            grid_fallback: false,
        });
        face_ranges.push(fan);
        report.meshed_faces += 1;
        log::debug!(
            "[BRep mesh] face {:?}: plane single-wire center fan ({} tris)",
            info_face_key,
            tri_n
        );
        continue;
            }
        }

        let fill_loops = if !uv_loop_is_degenerate(&mesh_loops) {
            &mesh_loops
        } else {
            &loops
        };

        let inner_boundaries: Vec<Vec<usize>> = fill_loops.inners.iter()
            .map(|inner| inner.boundary.iter().map(|v| v.global_idx).collect())
            .collect();

        let mut range = match algo {
            FaceMeshAlgo::SurfaceFill3d | FaceMeshAlgo::ClosedParametric if mixed_boundary => {
        surface_fill_3d(
            info_face_key,
            &boundary_ordered,
            &inner_boundaries,
            face,
            &mut global_vertices,
            &mut global_normals,
            &mut all_indices,
            &mut pos_to_idx,
            &scaled_config.face,
        shared,
        )
            }
            FaceMeshAlgo::SurfaceFill3d => surface_fill_3d(
        info_face_key,
        &boundary_ordered,
        &inner_boundaries,
        face,
        &mut global_vertices,
        &mut global_normals,
        &mut all_indices,
        &mut pos_to_idx,
        &scaled_config.face,
        shared,
            ),
            FaceMeshAlgo::TrimmedCdt if mixed_boundary => surface_fill_3d(
        info_face_key,
        &boundary_ordered,
        &inner_boundaries,
        face,
        &mut global_vertices,
        &mut global_normals,
        &mut all_indices,
        &mut pos_to_idx,
        &scaled_config.face,
        shared,
            ),
            FaceMeshAlgo::TrimmedCdt | FaceMeshAlgo::ClosedParametric => fill_trimmed(
        info_face_key,
        fill_loops,
        face,
        reg,
        &mut global_vertices,
        &mut global_normals,
        &mut all_indices,
        &mut pos_to_idx,
        wire_len,
                &scaled_config.face,
                shared,
            ),
        };

        range.face_key = info_face_key;

        if range.cdt_constraint_failures > 0 {
            report.cdt_constraint_failure_count += range.cdt_constraint_failures;
            if range.tri_count > 0 {
        used_parametric_grid = true;
        report.grid_fallback_count += 1;
            }
        }

        if used_surface_fill
            && range.tri_count > 0
            && range.max_chord_error > chord_reject
            && !scaled_config.fast_export
        {
            log::debug!(
        "[BRep mesh] face {:?}: reject surface fill (chord {:.4} > {:.4}), retry trimmed CDT",
        info_face_key,
        range.max_chord_error,
        chord_reject
            );
            all_indices.truncate(tris_before_face * 4);
            range = fill_trimmed(
        info_face_key,
        fill_loops,
        face,
        reg,
        &mut global_vertices,
        &mut global_normals,
        &mut all_indices,
        &mut pos_to_idx,
        wire_len,
                &scaled_config.face,
                shared,
            );
            range.face_key = info_face_key;
            used_surface_fill = false;
        }

        if used_surface_fill {
            report.grid_fallback_count += 1;
        }

        // Chord-error rejection for TrimmedCdt: retry with tighter deflection
        // when interior grid points fail to capture surface curvature.
        // Only retry when the face is clearly under-tessellated (few tris AND high chord).
        // Faces with adequate triangle count skip this expensive retry.
        let retry_low_tri_count = range.tri_count < min_adequate.saturating_mul(2);
        if !used_surface_fill
            && !used_parametric_grid
            && range.tri_count > 0
            && range.max_chord_error > chord_reject
            && retry_low_tri_count
            && scaled_config.face.deflection_interior > 1e-6
            && !scaled_config.fast_export
        {
            let tighter = (scaled_config.face.deflection_interior * 0.25)
        .max(1e-5);
            log::debug!(
        "[BRep mesh] face {:?}: TrimmedCdt chord {:.4} > {:.4}, retry with deflection {:.6}",
        info_face_key, range.max_chord_error, chord_reject, tighter,
            );
            let mut retry_cfg = scaled_config.face.clone();
            retry_cfg.deflection_interior = tighter;
            all_indices.truncate(tris_before_face * 4);
            range = fill_trimmed(
        info_face_key,
        fill_loops,
        face,
        reg,
        &mut global_vertices,
        &mut global_normals,
        &mut all_indices,
        &mut pos_to_idx,
        wire_len,
                &retry_cfg,
                shared,
            );
            range.face_key = info_face_key;
            if range.tri_count == 0 {
        // Still empty — fall back to parametric grid
        log::debug!(
            "[BRep mesh] face {:?}: trimmed CDT empty after retry, using trimmed UV grid",
            info_face_key,
        );
        all_indices.truncate(tris_before_face * 4);
        let uv_bounds = loops.outer.boundary.iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(u0, u1, v0, v1), v| {
                (u0.min(v.uv.0), u1.max(v.uv.0), v0.min(v.uv.1), v1.max(v.uv.1))
            },
        );
        mesh_trimmed_uv_grid(
            face,
            fill_loops,
            uv_bounds,
            Some(&retry_cfg),
            None,
            &mut global_vertices,
            &mut global_normals,
            &mut pos_to_idx,
                    &mut all_indices,
                    shared,
                );
                range = FaceMeshRange {
                    face_key: info_face_key,
                    first_tri: tris_before_face,
                    tri_count: all_indices.len() / 4 - tris_before_face,
                    boundary_global: range.boundary_global,
                    max_chord_error: 0.0,
                    cdt_constraint_failures: 0,
                };
                used_parametric_grid = true;
                report.grid_fallback_count += 1;
            }
        }

        if range.tri_count == 0 && !used_surface_fill && !mixed_boundary {
            range = surface_fill_3d(
        info_face_key,
        &boundary_ordered,
        &inner_boundaries,
        face,
        &mut global_vertices,
        &mut global_normals,
        &mut all_indices,
        &mut pos_to_idx,
        &scaled_config.face,
        shared,
            );
            if range.tri_count > 0 {
        report.grid_fallback_count += 1;
            }
        }

        let keep_trimmed = range.tri_count > 0
            && range.max_chord_error <= chord_reject
            && !used_surface_fill
            && !mixed_boundary;

        let grid_loops = if matches!(face.surface, SurfaceGeom::Revolution { .. })
            || prefer_native_cdt
            || matches!(face.surface, SurfaceGeom::Plane { .. })
            || matches!(face.surface, SurfaceGeom::BSpline(_))
            || (matches!(face.surface, SurfaceGeom::Cylinder { .. })
        && cylinder_loop_needs_uv_rebuild(&mesh_loops, &face.surface))
        {
            if matches!(face.surface, SurfaceGeom::Cylinder { .. }) {
        repair_cylinder_uv_loops(&mesh_loops, face, &global_vertices)
            } else {
        loops_native_surface_uv(&mesh_loops, face, &global_vertices)
            }
        } else {
            mesh_loops.clone()
        };

        let diag_shape = std::env::var("SHAPE_FACE_DIAG").is_ok();
        if diag_shape {
            let wire_len = reg
        .wires
        .get(face.outer_wire)
        .map(|w| w.edges.len())
        .unwrap_or(0);
            let v_min = grid_loops
        .outer
        .boundary
        .iter()
        .map(|v| v.uv.1)
        .fold(f32::INFINITY, f32::min);
            let v_max = grid_loops
        .outer
        .boundary
        .iter()
        .map(|v| v.uv.1)
        .fold(f32::MIN, f32::max);
            eprintln!(
        "[mesh diag] {:?} pre-grid tris={} wire={} outer_uv={} v=[{:.4},{:.4}] native={:?} rv={:?} allow={}",
        info_face_key,
        range.tri_count,
        wire_len,
        grid_loops.outer.boundary.len(),
        v_min,
        v_max,
        grid_loops.native_uv_bounds(),
        grid_loops.revolution_v_bounds_from_3d(face, &global_vertices),
        allows_trimmed_uv_grid(reg, face, &grid_loops, &global_vertices),
            );
        }

        if !scaled_config.fast_export
            && wire_len == 2
            && matches!(
        face.surface,
        SurfaceGeom::BSpline(_) | SurfaceGeom::Offset { .. }
            )
            && range.tri_count > 0
        {
            range.max_chord_error = measure_face_chord_error(
        face,
        &global_vertices,
        &all_indices,
        &range,
            );
        }

        let deflection_goal = scaled_config.face.deflection_interior * 2.0;
        let offset_low_tris = wire_len == 2
            && matches!(face.surface, SurfaceGeom::Offset { .. })
            && !surface_is_planar_trim(&face.surface)
            && range.tri_count < min_adequate.saturating_mul(8);
        let freeform_chord_bad = wire_len == 2
            && matches!(
        face.surface,
        SurfaceGeom::BSpline(_) | SurfaceGeom::Offset { .. }
            )
            && range.tri_count > 0
            && range.max_chord_error > deflection_goal;

        if wire_len == 2
            && (range.tri_count < min_adequate
        || used_surface_fill
        || freeform_chord_bad
        || offset_low_tris)
        {
            let try_surface_fill = boundary_ordered.len() >= 3
        && range.tri_count < min_adequate
        && (matches!(face.surface, SurfaceGeom::Offset { .. })
            || (matches!(face.surface, SurfaceGeom::Revolution { .. })
                && revolution_boundary_v_collapsed(&mesh_loops)));

            if try_surface_fill {
        if range.tri_count > 0 {
            all_indices.truncate(range.first_tri * 4);
        }
        let sf = surface_fill_3d(
            info_face_key,
            &boundary_ordered,
            &inner_boundaries,
            face,
            &mut global_vertices,
            &mut global_normals,
            &mut all_indices,
            &mut pos_to_idx,
            &scaled_config.face,
        shared,
        );
        if sf.tri_count >= min_adequate {
            range = sf;
            range.face_key = info_face_key;
            used_surface_fill = true;
        }
            }

            let skip_ruled = range.tri_count >= min_adequate
        && (!used_surface_fill || range.max_chord_error <= chord_reject)
        && !freeform_chord_bad
        && !offset_low_tris;

            if skip_ruled {
        // adequate trimmed / surface fill
            } else if matches!(face.surface, SurfaceGeom::Offset { .. })
        && boundary_ordered.len() >= 3
            {
        if range.tri_count > 0 {
            all_indices.truncate(range.first_tri * 4);
        }
        let sf = surface_fill_3d(
            info_face_key,
            &boundary_ordered,
            &inner_boundaries,
            face,
            &mut global_vertices,
            &mut global_normals,
            &mut all_indices,
            &mut pos_to_idx,
            &scaled_config.face,
        shared,
        );
        if sf.tri_count > 0 {
            range = sf;
            range.face_key = info_face_key;
            used_surface_fill = true;
        }
            } else {
            let saved_first = range.first_tri;
            let saved_count = range.tri_count;
            if range.tri_count > 0 {
        all_indices.truncate(range.first_tri * 4);
            }
            range = try_ruled_two_wire_mesh(
        info_face_key,
        face,
        info_wire_edges,
        edge_polygons,
        edge_boundary_idx,
        RuledMeshBuffers {
            global_vertices: &mut global_vertices,
            global_normals: &mut global_normals,
            all_indices: &mut all_indices,
                    pos_to_idx: &mut pos_to_idx,
                    shared_boundary: shared,
                },
        &scaled_config.face,
            );
            let valid = count_valid_tris_in_range(
        &all_indices,
        &global_vertices,
        range.first_tri,
        range.tri_count,
            );
            let valid_ratio = if range.tri_count > 0 {
        valid as f32 / range.tri_count as f32
            } else {
        0.0
            };
            let ruled_tri_count = range.tri_count;
            let mut ruled_ok = if matches!(face.surface, SurfaceGeom::Plane { .. }) {
        range.tri_count > 0
            } else {
        valid >= min_adequate.min(8) && valid_ratio >= 0.25
            };
            if ruled_ok
        && !matches!(face.surface, SurfaceGeom::Plane { .. })
        && !matches!(face.surface, SurfaceGeom::Offset { .. })
        && !scaled_config.fast_export
            {
        range.max_chord_error = measure_face_chord_error(
            face,
            &global_vertices,
            &all_indices,
            &range,
        );
        if range.max_chord_error > chord_reject {
            ruled_ok = false;
        }
            }
            if ruled_ok {
        range.tri_count = valid;
        used_surface_fill = false;
            } else {
        all_indices.truncate(saved_first * 4);
        range.first_tri = saved_first;
        range.tri_count = saved_count;
        log::debug!(
            "[BRep mesh] face {:?}: rejected ruled strip (valid={}/{} ratio={:.2})",
            info_face_key,
            valid,
            ruled_tri_count,
            valid_ratio
        );
            }
            }
        }

        let grid_valid = count_valid_tris_in_range(
            &all_indices,
            &global_vertices,
            range.first_tri,
            range.tri_count,
        );
        if (range.tri_count == 0
            || grid_valid < min_adequate
            || range.max_chord_error > chord_reject)
            && !multi_wire_rev
            && !surface_is_planar_trim(&face.surface)
            && allows_trimmed_uv_grid(reg, face, &grid_loops, &global_vertices)
            && !(scaled_config.fast_export && range.tri_count >= min_adequate)
        {
            let mut uv_bounds = grid_loops
        .revolution_native_uv_bounds_from_boundary(face, &global_vertices)
        .or_else(|| revolution_fallback_uv_bounds(&grid_loops, face, &global_vertices))
        .or_else(|| grid_loops.native_uv_bounds())
        .or_else(|| grid_loops.uv_bounds_from_projection(face, &global_vertices));
            if let (Some((u0, u1, v0, v1)), Some((rv0, rv1))) = (
        uv_bounds,
        grid_loops.revolution_v_bounds_from_3d(face, &global_vertices),
            ) {
        if surface_is_revolution_like(&face.surface) && (v1 - v0).abs() < 1e-5 {
            uv_bounds = Some((u0, u1, rv0, rv1));
        }
            }
            if let Some(uv_bounds) = uv_bounds {
        let saved_first = range.first_tri;
        let saved_count = range.tri_count;
        let saved_chord = range.max_chord_error;
        let append_start_tri = all_indices.len() / 4;
        mesh_trimmed_uv_grid(
            face,
            &grid_loops,
            uv_bounds,
            Some(&scaled_config.face),
            None,
            &mut global_vertices,
            &mut global_normals,
            &mut pos_to_idx,
            &mut all_indices,
            shared,
        );
        let mut new_tris = all_indices.len() / 4 - append_start_tri;
        if new_tris == 0 {
            let wire_len = reg
                .wires
                .get(face.outer_wire)
                .map(|w| w.edges.len())
                .unwrap_or(0);
            if wire_len == 2 && matches!(face.surface, SurfaceGeom::Revolution { .. }) {
                mesh_uv_bbox_grid(
                    face,
                    uv_bounds,
                    Some(&scaled_config.face),
                    &mut global_vertices,
                    &mut global_normals,
                    &mut pos_to_idx,
                    &mut all_indices,
                );
                new_tris = all_indices.len() / 4 - append_start_tri;
            }
        }
        if new_tris > 0 {
            let trial_range = FaceMeshRange {
                face_key: info_face_key,
                first_tri: append_start_tri,
                tri_count: new_tris,
                boundary_global: range.boundary_global.clone(),
                max_chord_error: 0.0,
                cdt_constraint_failures: 0,
            };
            let grid_chord = measure_face_chord_error(
                face,
                &global_vertices,
                &all_indices,
                &trial_range,
            );
            let accept_grid = (grid_chord <= chord_reject
                || (saved_count > 0
                    && grid_chord < saved_chord
                    && grid_chord <= chord_reject * 2.0))
                && (saved_count == 0 || new_tris >= saved_count / 4);
            if accept_grid {
                if saved_count > 0 {
                    all_indices.drain(saved_first * 4..append_start_tri * 4);
                }
                range.first_tri = saved_first;
                range.tri_count = all_indices.len() / 4 - saved_first;
                range.max_chord_error = grid_chord;
                used_parametric_grid = true;
                report.grid_fallback_count += 1;
                log::debug!(
                    "[BRep mesh] face {:?}: trimmed UV grid ({} tris, chord {:.4})",
                    info_face_key,
                    range.tri_count,
                    grid_chord
                );
            } else {
                all_indices.truncate(append_start_tri * 4);
                range.first_tri = saved_first;
                range.tri_count = saved_count;
                range.max_chord_error = saved_chord;
                log::debug!(
                    "[BRep mesh] face {:?}: reject trimmed UV grid (chord {:.4} > {:.4})",
                    info_face_key,
                    grid_chord,
                    chord_reject
                );
            }
        }
            }
        }

        if range.tri_count == 0
            && !keep_trimmed
            && !surface_is_planar_trim(&face.surface)
            && allows_parametric_grid_fallback(reg, face, &loops)
        {
            if let Some(uv_bounds) = loops.native_uv_bounds() {
        let tris_before = all_indices.len() / 4;
        mesh_parametric_grid(
            face,
            Some(uv_bounds),
            Some(&scaled_config.face),
            &mut global_vertices,
            &mut global_normals,
            &mut pos_to_idx,
            &mut all_indices,
            shared,
        );
        let grid_tris = all_indices.len() / 4 - tris_before;
        if grid_tris > 0 {
            range.tri_count = grid_tris;
            range.first_tri = tris_before;
            used_parametric_grid = true;
            report.grid_fallback_count += 1;
            log::debug!(
                "[BRep mesh] face {:?}: UV-clipped grid fallback ({} tris)",
                info_face_key,
                grid_tris
            );
        }
            }
        }

        if range.tri_count == 0 && boundary_ordered.len() >= 3 {
            let sf = surface_fill_3d(
        info_face_key,
        &boundary_ordered,
        &inner_boundaries,
        face,
        &mut global_vertices,
        &mut global_normals,
        &mut all_indices,
        &mut pos_to_idx,
        &scaled_config.face,
        shared,
            );
            if sf.tri_count > 0 {
        range = sf;
        range.face_key = info_face_key;
        used_surface_fill = true;
        report.grid_fallback_count += 1;
            }
        }

        if range.tri_count == 0 && wire_len == 2 {
            let ruled = try_ruled_two_wire_mesh(
        info_face_key,
        face,
        info_wire_edges,
        edge_polygons,
        edge_boundary_idx,
        RuledMeshBuffers {
            global_vertices: &mut global_vertices,
            global_normals: &mut global_normals,
            all_indices: &mut all_indices,
                    pos_to_idx: &mut pos_to_idx,
                    shared_boundary: shared,
                },
        &scaled_config.face,
            );
            let valid = count_valid_tris_in_range(
        &all_indices,
        &global_vertices,
        ruled.first_tri,
        ruled.tri_count,
            );
            if valid > 0 {
        range = ruled;
        range.tri_count = valid;
        range.face_key = info_face_key;
        if !scaled_config.fast_export {
            range.max_chord_error = measure_face_chord_error(
                face,
                &global_vertices,
                &all_indices,
                &range,
            );
        }
        log::debug!(
            "[BRep mesh] face {:?}: last-resort ruled two-wire ({} tris)",
            info_face_key,
            valid
        );
            }
        }

        report.faces.push(FaceMeshStats {
            face_key: info_face_key,
            tri_count: range.tri_count,
            first_tri: range.first_tri,
            uv_source: loops.uv_source,
            max_chord_error: range.max_chord_error,
            cdt_constraint_failures: range.cdt_constraint_failures,
            grid_fallback: used_surface_fill || used_parametric_grid,
        });

        if range.tri_count > 0 {
            log::debug!(
        "[BRep mesh] face {:?}: {} tris uv={:?} max_chord={:.6}",
        info_face_key,
        range.tri_count,
        loops.uv_source,
        range.max_chord_error,
            );
            face_ranges.push(range);
            report.meshed_faces += 1;
        }
    }

    report.total_tris = all_indices.len() / 4;

    ChunkOutput {
        vertices: global_vertices,
        normals: global_normals,
        indices: all_indices,
        face_ranges,
        report,
        wire_diag,
    }
}
