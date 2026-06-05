use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;
use crate::topo::{BRepFace, ShellKey, EdgeKey, FaceKey, Orientation, VertexKey};
use crate::store::BRepStore;
use crate::geom::SurfaceGeom;
use crate::mesh_result::MeshResult;
use super::boundary::register_boundary_point;
use super::config::{count_valid_tris_in_range, min_adequate_trim_tris, BRepMeshConfig};
use super::edge_disc::{discretize_edge, EdgePolygon};
use super::face_fill::{
    face_boundary_is_mixed, FaceMeshRange, fill_trimmed, mesh_plane_fan_3d, prefers_native_uv_ruled,
    surface_fill_3d, measure_face_chord_error, surface_is_revolution_like,
};
use super::face_uv::{
    collect_face_loops, cylinder_loop_needs_uv_rebuild, loops_from_boundary_indices,
    loops_native_surface_uv, repair_cylinder_uv_loops, revolution_boundary_v_collapsed,
    revolution_loops_from_wire_edges, uv_loop_is_degenerate, FaceUvLoops, UvSource,
};
use super::fallback_policy::{
    allows_parametric_grid_fallback, allows_trimmed_uv_grid, revolution_fallback_uv_bounds,
    uses_closed_parametric_mesh,
};
use super::grid::{mesh_closed_surface, mesh_parametric_grid, mesh_trimmed_uv_grid, mesh_uv_bbox_grid};
use super::post_process::{compact_mesh_vertices, cull_degenerate_tris, recompute_normals_from_tris};
use super::edge_pool::build_face_boundary_pool;
use super::ruled::{try_ruled_two_wire_mesh, RuledMeshBuffers};
use super::refiner::{merge_refined_face, refine_mesh_interior, extract_face_mesh_with_map};
use super::optimize::optimize_mesh;
use super::same_param::apply_same_parameter;
use super::algo_factory::FaceMeshAlgo;
use super::diagnostic::{diag_enabled, format_wire_loop_lines, log_mesh_coordinates_if_requested};
use super::report::{apply_relative_deflection, shell_bbox_diagonal, FaceMeshStats, ShellMeshReport};
use super::face_dispatch::{algo_from_plan, plan_face_mesh, FaceMeshPlan};
use super::shell_mesh::ShellMeshOutput;
use rayon::prelude::*;

/// Shell orientation XOR face.same_sense for mesh normals and winding.
fn mesh_face_view(face: &BRepFace, effective_same_sense: bool) -> Cow<'_, BRepFace> {
    if effective_same_sense == face.same_sense {
        Cow::Borrowed(face)
    } else {
        let mut f = face.clone();
        f.same_sense = effective_same_sense;
        Cow::Owned(f)
    }
}

// ── Shared data types for the mesh pipeline ─────────────────────────────

struct FaceWireInfo {
    face_key: FaceKey,
    wire_edges: Vec<(EdgeKey, Vec<usize>)>,
}

struct FaceLoopData {
    face_key: FaceKey,
    wire_edges: Vec<(EdgeKey, Vec<usize>)>,
    loops: Option<FaceUvLoops>,
}

struct ChunkOutput {
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
    indices: Vec<i32>,
    face_ranges: Vec<FaceMeshRange>,
    report: ShellMeshReport,
    wire_diag: Vec<String>,
}

struct BoundaryPoolResult {
    boundary_vertices: Vec<Vec3>,
    boundary_normals: Vec<Vec3>,
    boundary_pos_to_idx: HashMap<[u32; 3], usize>,
    boundary_vertex_count: usize,
    edge_boundary_idx: HashMap<(FaceKey, EdgeKey, usize), usize>,
    face_infos: Vec<FaceWireInfo>,
    face_orient: HashMap<FaceKey, bool>,
    heal_skipped_faces: Vec<FaceKey>,
    global_vertices: Vec<Vec3>,
}

// ── Phase 1: config scaling, shell lookup, edge discretization ──────────

/// Returns `None` when the shell key does not exist in the store.
fn init_shell_mesh(
    shell_key: ShellKey,
    reg: &BRepStore,
    config: &BRepMeshConfig,
) -> Option<(BRepMeshConfig, f32, ShellMeshReport, HashMap<EdgeKey, EdgePolygon>)> {
    let shell_diag = shell_bbox_diagonal(shell_key, reg);
    let mut scaled_config = config.clone();
    apply_relative_deflection(&mut scaled_config, shell_diag);
    if scaled_config.relative_deflection > 0.0 && shell_diag > 0.0 {
        scaled_config.face.shell_min_size =
            shell_diag * scaled_config.face.min_size_relative;
    }

    let shell = reg.shells.get(shell_key)?;

    let report = ShellMeshReport {
        face_count: shell.faces.len(),
        shell_diag,
        ..Default::default()
    };

    // Phase 1: edge discretization — only for edges belonging to this shell
    let _te = std::time::Instant::now();
    let shell_edges = crate::topo_iter::iter_edges_of_shell(shell_key, reg);
    log::debug!("[mesh] shell {:?}: {} edges to discretize", shell_key, shell_edges.len());
    let mut edge_polygons: HashMap<EdgeKey, EdgePolygon> = HashMap::with_capacity(shell_edges.len());
    let edge_cfg = &scaled_config.edge;
    for ek in &shell_edges {
        edge_polygons.insert(*ek, discretize_edge(*ek, reg, edge_cfg));
    }
    log::debug!("[mesh] edge discretization: {:.1}s for {} edges", _te.elapsed().as_secs_f32(), shell_edges.len());

    // SameParameter snap (Phase 1b)
    if scaled_config.same_parameter_tol > 0.0 {
        apply_same_parameter(
            &mut edge_polygons,
            reg,
            scaled_config.same_parameter_tol,
        );
    }

    Some((scaled_config, shell_diag, report, edge_polygons))
}

// ── Phase 2: shared boundary vertex pool ────────────────────────────────

fn build_shell_boundary_pool(
    shell_key: ShellKey,
    reg: &BRepStore,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    skip_face_keys: &[FaceKey],
) -> BoundaryPoolResult {
    let mut global_vertices: Vec<Vec3> = Vec::new();
    let mut global_normals: Vec<Vec3> = Vec::new();
    let mut pos_to_idx: HashMap<[u32; 3], usize> = HashMap::new();
    let mut vertex_mesh_idx: HashMap<VertexKey, usize> = HashMap::new();

    for (vk, v) in reg.vertices.iter() {
        let idx = register_boundary_point(
            v.position,
            &mut global_vertices,
            &mut global_normals,
            &mut pos_to_idx,
        );
        vertex_mesh_idx.insert(vk, idx);
    }

    let total_edge_pts: usize = edge_polygons.values().map(|p| p.params_3d.len()).sum();
    let edge_boundary_idx = build_face_boundary_pool(
        reg,
        shell_key,
        edge_polygons,
        &mut global_vertices,
        &mut global_normals,
        &mut pos_to_idx,
        &vertex_mesh_idx,
    );
    log::debug!(
        "[BRep mesh] {} edge polygons, {} total edge pts -> {} unique boundary verts",
        edge_polygons.len(),
        total_edge_pts,
        global_vertices.len()
    );

    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => {
            return BoundaryPoolResult {
                boundary_vertices: global_vertices.clone(),
                boundary_normals: global_normals.clone(),
                boundary_pos_to_idx: pos_to_idx.clone(),
                boundary_vertex_count: global_vertices.len(),
                edge_boundary_idx,
                face_infos: Vec::new(),
                face_orient: HashMap::new(),
                heal_skipped_faces: Vec::new(),
                global_vertices,
            };
        }
    };

    let mut face_infos: Vec<FaceWireInfo> = Vec::new();
    let mut heal_skipped_faces: Vec<FaceKey> = Vec::new();

    // Fix: build per-face effective sense from shell orientation.
    // Previously _orient was discarded, causing broken normals on reversed faces.
    let face_orient: HashMap<FaceKey, bool> = shell.faces.iter()
        .filter_map(|&(fk, orient)| {
            reg.faces.get(fk).map(|f| (fk, super::orient::effective_sense_raw(f.same_sense, orient)))
        })
        .collect();

    for &(face_key, _orient) in &shell.faces {
        if skip_face_keys.contains(&face_key) {
            heal_skipped_faces.push(face_key);
            continue;
        }
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        let wire = match reg.wires.get(face.outer_wire) {
            Some(w) => w,
            None => continue,
        };

        let mut wire_edges = Vec::new();
        for &(ek, orient) in &wire.edges {
            if let Some(poly) = edge_polygons.get(&ek) {
                let n_pts = poly.params_3d.len();
                if n_pts == 0 {
                    continue;
                }
                let mut indices: Vec<usize> = (0..n_pts).collect();
                if orient == Orientation::Reversed {
                    indices.reverse();
                }
                wire_edges.push((ek, indices));
            }
        }

        if !wire_edges.is_empty() || wire.edges.is_empty() {
            face_infos.push(FaceWireInfo {
                face_key,
                wire_edges,
            });
        }
    }

    // Save boundary vertex pool as immutable baseline for parallel face meshing.
    // Each parallel chunk clones this to get its own independent mutable state.
    let boundary_vertices: Vec<Vec3> = global_vertices.clone();
    let boundary_normals: Vec<Vec3> = global_normals.clone();
    let boundary_pos_to_idx: HashMap<[u32; 3], usize> = pos_to_idx.clone();
    let boundary_vertex_count = boundary_vertices.len();

    BoundaryPoolResult {
        boundary_vertices,
        boundary_normals,
        boundary_pos_to_idx,
        boundary_vertex_count,
        edge_boundary_idx,
        face_infos,
        face_orient,
        heal_skipped_faces,
        global_vertices,
    }
}

// ── Phase 2a: pre-compute UV loops in parallel (read-only) ──────────────

fn precompute_face_uv_loops(
    face_infos: &[FaceWireInfo],
    reg: &BRepStore,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &HashMap<(FaceKey, EdgeKey, usize), usize>,
    global_vertices: &[Vec3],
) -> Vec<FaceLoopData> {
    face_infos
        .par_iter()
        .filter_map(|finfo| {
            let face = reg.faces.get(finfo.face_key)?;
            if finfo.wire_edges.is_empty() {
                // Closed parametric surface — no loops needed
                return Some(FaceLoopData {
                    face_key: finfo.face_key,
                    wire_edges: finfo.wire_edges.clone(),
                    loops: None,
                });
            }
            let loops = collect_face_loops(
                finfo.face_key, face, reg, edge_polygons,
                edge_boundary_idx, global_vertices,
            );
            Some(FaceLoopData {
                face_key: finfo.face_key,
                wire_edges: finfo.wire_edges.clone(),
                loops: Some(loops),
            })
        })
        .collect()
}

// ── Phase 2c: merge chunk outputs into global state ─────────────────────

fn merge_chunk_outputs(
    chunk_outputs: Vec<ChunkOutput>,
    boundary_vertex_count: usize,
    collect_diag: bool,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<i32>, Vec<FaceMeshRange>, ShellMeshReport, Vec<String>) {
    if chunk_outputs.is_empty() {
        return (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            ShellMeshReport::default(),
            Vec::new(),
        );
    }

    let base = &chunk_outputs[0];
    let mut global_vertices = base.vertices.clone();
    let mut global_normals = base.normals.clone();
    let mut all_indices = base.indices.clone();
    let mut face_ranges = base.face_ranges.clone();
    let mut report = base.report.clone();
    let mut wire_diag = base.wire_diag.clone();

    for chunk in &chunk_outputs[1..] {
        let interior_offset = global_vertices.len() as i32;
        let chunk_interior_start = boundary_vertex_count;

        // Append interior vertices (skip boundary vertices already in base)
        global_vertices.extend_from_slice(&chunk.vertices[chunk_interior_start..]);
        global_normals.extend_from_slice(&chunk.normals[chunk_interior_start..]);

        // Remap indices: boundary (< boundary_vertex_count) stay unchanged;
        // interior (>= boundary_vertex_count) get shifted by offset.
        let index_base = all_indices.len() / 4;
        for &idx in &chunk.indices {
            if idx < 0 {
                all_indices.push(idx); // terminator
            } else if (idx as usize) < boundary_vertex_count {
                all_indices.push(idx); // boundary — shared across all chunks
            } else {
                all_indices.push(idx + interior_offset - boundary_vertex_count as i32);
            }
        }

        // Remap face_ranges: first_tri is relative to chunk's local index buffer
        for mut fr in chunk.face_ranges.clone() {
            fr.first_tri += index_base;
            face_ranges.push(fr);
        }

        // Merge report stats
        report.face_count += chunk.report.face_count;
        report.meshed_faces += chunk.report.meshed_faces;
        report.grid_fallback_count += chunk.report.grid_fallback_count;
        report.total_tris += chunk.report.total_tris;
        report.max_equiv_edge_weld_gap = report.max_equiv_edge_weld_gap
            .max(chunk.report.max_equiv_edge_weld_gap);
        report.faces.extend_from_slice(&chunk.report.faces);

        if collect_diag {
            wire_diag.extend_from_slice(&chunk.wire_diag);
        }
    }

    (
        global_vertices,
        global_normals,
        all_indices,
        face_ranges,
        report,
        wire_diag,
    )
}

// ── Post-processing: refine, cull, compact, weld, optimize ──────────────

fn finalize_shell_mesh(
    shell_key: ShellKey,
    reg: &BRepStore,
    face_ranges: &[FaceMeshRange],
    mut all_indices: Vec<i32>,
    mut global_vertices: Vec<Vec3>,
    mut global_normals: Vec<Vec3>,
    mut report: ShellMeshReport,
    heal_skipped_faces: &[FaceKey],
    scaled_config: &BRepMeshConfig,
    shell_diag: f32,
    face_infos_len: usize,
    boundary_vertex_count: usize,
    wire_diag: &[String],
    collect_diag: bool,
) -> ShellMeshOutput {
    for face_key in heal_skipped_faces {
        report.faces.push(FaceMeshStats {
            face_key: *face_key,
            tri_count: 0,
            first_tri: 0,
            uv_source: UvSource::SurfaceFill,
            max_chord_error: 0.0,
            grid_fallback: false,
        });
        log::debug!("[BRep mesh] face {:?} skipped by heal (no grid fallback)", face_key);
    }

    if scaled_config.refine.enable_post_refine && scaled_config.refine.max_iterations > 0 {
        let mut tri_offset: isize = 0;
        for range in face_ranges {
            let face = match reg.faces.get(range.face_key) {
                Some(f) => f,
                None => continue,
            };
            // Analytic surfaces produce exact meshes; post-refine Steiner splits
            // would waste time and potentially break watertight edges.
            if surface_chord_error_is_trivial(&face.surface) {
                continue;
            }
            let adj_start_raw = (range.first_tri as isize + tri_offset) * 4;
            if adj_start_raw < 0 { tri_offset -= adj_start_raw / 4; continue; }
            let adj_start = adj_start_raw as usize;
            let adj_end = adj_start + range.tri_count * 4;
            let (local_mesh, local_to_global) = extract_face_mesh_with_map(
                &global_vertices,
                &global_normals,
                &all_indices,
                adj_start,
                adj_end,
            );
            let local_boundary: std::collections::HashSet<usize> = local_to_global
                .iter()
                .filter(|(_, &g)| range.boundary_global.contains(&g))
                .map(|(&l, _)| l)
                .collect();
            let refined = refine_mesh_interior(
                &local_mesh,
                face,
                &local_boundary,
                &scaled_config.refine,
            );
            let new_tri_count = merge_refined_face(
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                adj_start / 4,
                adj_end / 4,
                &refined,
                &local_to_global,
            );
            tri_offset += new_tri_count as isize - range.tri_count as isize;
        }
    }

    report.total_tris = all_indices.len() / 4;
    if !scaled_config.fast_export {
        for stats in &mut report.faces {
            if stats.grid_fallback || stats.tri_count == 0 {
                continue;
            }
            let Some(range) = face_ranges.iter().find(|r| r.face_key == stats.face_key) else {
                continue;
            };
            let Some(face) = reg.faces.get(stats.face_key) else {
                continue;
            };
            // Skip chord-error measurement for analytic surfaces — their meshes
            // are exact (within floating-point precision). Only BSpline and Offset
            // surfaces benefit from chord verification.
            if surface_chord_error_is_trivial(&face.surface) {
                continue;
            }
            stats.max_chord_error =
                measure_face_chord_error(face, &global_vertices, &all_indices, range);
        }
    }
    report.log_summary(shell_key);

    log::debug!(
        "[BRep mesh] shell {:?}: {} boundary verts, {} faces, {} tris total",
        shell_key,
        global_vertices.len(),
        face_infos_len,
        report.total_tris
    );
    let mut mesh = MeshResult {
        vertices: global_vertices,
        indices: all_indices,
        normals: global_normals,
    };
    let culled = { let _c = cull_degenerate_tris(&mut mesh.indices, &mesh.vertices); log::info!("[BRep mesh] culled {} degenerate tris, {} remain", _c, mesh.indices.len()/4); _c };
    if culled > 0 {
        log::debug!("[BRep mesh] culled {culled} degenerate triangle(s)");
        report.total_tris = mesh.indices.len() / 4;
    }
    compact_mesh_vertices(&mut mesh);
    report.total_tris = mesh.indices.len() / 4;
    let weld_tol = scaled_config
        .weld_tolerance
        .max(shell_diag * 1e-5)
        .max(1e-6);

    // Build protected set: boundary vertices (indices < boundary_vertex_count)
    // that are referenced by emitted triangles. These are discretized edge
    // points shared between adjacent faces — they must not move during welding
    // to preserve cross-face watertightness.
    // OCC: BRepMesh_FastDiscret boundary vertex locking.
    let mut protected: HashSet<usize> = HashSet::new();
    for chunk in mesh.indices.chunks(4) {
        if chunk.len() < 3 { continue; }
        for &idx in &chunk[0..3] {
            if idx >= 0 {
                let i = idx as usize;
                if i < boundary_vertex_count {
                    protected.insert(i);
                }
            }
        }
    }
    // Also protect all B-Rep canonical vertex positions (topology anchors)
    for i in 0..boundary_vertex_count.min(mesh.vertices.len()) {
        protected.insert(i);
    }

    let welded = mesh.weld_vertices_protected(weld_tol, &protected);
    log::debug!("[BRep mesh] welded {} interior vertices ({} boundary protected)",
        welded, protected.len());
    recompute_normals_from_tris(&mesh.vertices, &mesh.indices, &mut mesh.normals);
    optimize_mesh(&mut mesh, &scaled_config.optimize);
    if collect_diag {
        log_mesh_coordinates_if_requested(
            shell_key,
            reg,
            &mesh.vertices,
            &mesh.indices,
            face_ranges,
            &report,
            wire_diag,
        );
    }
    ShellMeshOutput { mesh, report }
}

// ── Main orchestration ──────────────────────────────────────────────────

pub(crate) fn mesh_brep_shell_with_report_impl(
    shell_key: ShellKey,
    reg: &BRepStore,
    config: &BRepMeshConfig,
    skip_face_keys: &[FaceKey],
) -> ShellMeshOutput {
    // Phase 1: config scaling, shell lookup, edge discretization, same parameter
    let Some((scaled_config, shell_diag, _report, edge_polygons)) =
        init_shell_mesh(shell_key, reg, config)
    else {
        return ShellMeshOutput {
            mesh: MeshResult::default(),
            report: ShellMeshReport {
                shell_diag: shell_bbox_diagonal(shell_key, reg),
                ..Default::default()
            },
        };
    };

    // Phase 2: shared boundary vertex pool (B-Rep vertices own canonical indices)
    let BoundaryPoolResult {
        boundary_vertices,
        boundary_normals,
        boundary_pos_to_idx,
        boundary_vertex_count,
        edge_boundary_idx,
        face_infos,
        face_orient,
        heal_skipped_faces,
        global_vertices,
    } = build_shell_boundary_pool(
        shell_key,
        reg,
        &edge_polygons,
        skip_face_keys,
    );

    // Phase 2a: Pre-compute UV loops in parallel (read-only, rayon par_iter)
    let face_loop_data = precompute_face_uv_loops(
        &face_infos,
        reg,
        &edge_polygons,
        &edge_boundary_idx,
        &global_vertices,
    );

    let collect_diag = diag_enabled();

    // Phase 2b: Mesh faces in parallel. Each chunk clones the boundary vertex
    let _tp2 = std::time::Instant::now();
    // Phase 2b: Mesh faces in parallel. Each chunk clones the boundary vertex
    // pool and processes its faces independently — no cross-face synchronization.
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = (face_loop_data.len() + num_threads - 1) / num_threads;

    let chunk_outputs: Vec<ChunkOutput> = if face_loop_data.is_empty() {
        Vec::new()
    } else {
        face_loop_data
            .par_chunks(chunk_size)
            .map(|chunk: &[FaceLoopData]| {
                // Shadow mutable state with per-chunk local copies.
                // The loop body below references these names — they resolve to
                // the local copies, not the outer variables.
                let mut global_vertices = boundary_vertices.clone();
                let mut global_normals = boundary_normals.clone();
                let mut pos_to_idx = boundary_pos_to_idx.clone();
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
                &edge_boundary_idx,
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
                &edge_polygons,
                &edge_boundary_idx,
                RuledMeshBuffers {
                    global_vertices: &mut global_vertices,
                    global_normals: &mut global_normals,
                    all_indices: &mut all_indices,
                    pos_to_idx: &mut pos_to_idx,
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
                &edge_polygons,
                &edge_boundary_idx,
                RuledMeshBuffers {
                    global_vertices: &mut global_vertices,
                    global_normals: &mut global_normals,
                    all_indices: &mut all_indices,
                    pos_to_idx: &mut pos_to_idx,
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
                    grid_fallback: false,
                });
                face_ranges.push(FaceMeshRange {
                    face_key: info_face_key,
                    first_tri: ruled.first_tri,
                    tri_count: valid,
                    boundary_global: ruled.boundary_global,
                    max_chord_error: max_chord,
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
                &edge_polygons,
                &edge_boundary_idx,
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
                &edge_polygons,
                &edge_boundary_idx,
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
                &edge_polygons,
                &edge_boundary_idx,
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
            );
            if fan.tri_count > 0 {
                let tri_n = fan.tri_count;
                report.faces.push(FaceMeshStats {
                    face_key: info_face_key,
                    tri_count: tri_n,
                    first_tri: fan.first_tri,
                    uv_source: loops.uv_source,
                    max_chord_error: fan.max_chord_error,
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

        let mut range = match algo {
            FaceMeshAlgo::SurfaceFill3d | FaceMeshAlgo::ClosedParametric if mixed_boundary => {
                surface_fill_3d(
                    info_face_key,
                    &boundary_ordered,
                    face,
                    &mut global_vertices,
                    &mut global_normals,
                    &mut all_indices,
                    &mut pos_to_idx,
                    &scaled_config.face,
                )
            }
            FaceMeshAlgo::SurfaceFill3d => surface_fill_3d(
                info_face_key,
                &boundary_ordered,
                face,
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                &mut pos_to_idx,
                &scaled_config.face,
            ),
            FaceMeshAlgo::TrimmedCdt if mixed_boundary => surface_fill_3d(
                info_face_key,
                &boundary_ordered,
                face,
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                &mut pos_to_idx,
                &scaled_config.face,
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
            ),
        };

        range.face_key = info_face_key;

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
                );
                range = FaceMeshRange {
                    face_key: info_face_key,
                    first_tri: tris_before_face,
                    tri_count: all_indices.len() / 4 - tris_before_face,
                    boundary_global: range.boundary_global,
                    max_chord_error: 0.0,
                };
                used_parametric_grid = true;
                report.grid_fallback_count += 1;
            }
        }

        if range.tri_count == 0 && !used_surface_fill && !mixed_boundary {
            range = surface_fill_3d(
                info_face_key,
                &boundary_ordered,
                face,
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                &mut pos_to_idx,
                &scaled_config.face,
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
                    face,
                    &mut global_vertices,
                    &mut global_normals,
                    &mut all_indices,
                    &mut pos_to_idx,
                    &scaled_config.face,
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
                    face,
                    &mut global_vertices,
                    &mut global_normals,
                    &mut all_indices,
                    &mut pos_to_idx,
                    &scaled_config.face,
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
                &info_wire_edges,
                &edge_polygons,
                &edge_boundary_idx,
                RuledMeshBuffers {
                    global_vertices: &mut global_vertices,
                    global_normals: &mut global_normals,
                    all_indices: &mut all_indices,
                    pos_to_idx: &mut pos_to_idx,
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
            && !matches!(face.surface, SurfaceGeom::Plane { .. })
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
            && !matches!(face.surface, SurfaceGeom::Plane { .. })
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
                face,
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                &mut pos_to_idx,
                &scaled_config.face,
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
                &edge_polygons,
                &edge_boundary_idx,
                RuledMeshBuffers {
                    global_vertices: &mut global_vertices,
                    global_normals: &mut global_normals,
                    all_indices: &mut all_indices,
                    pos_to_idx: &mut pos_to_idx,
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
                } // end per-face loop (within par_chunks closure)

                report.total_tris = all_indices.len() / 4;

                ChunkOutput {
                    vertices: global_vertices,
                    normals: global_normals,
                    indices: all_indices,
                    face_ranges,
                    report,
                    wire_diag,
                }
            })
            .collect()
    };

    log::info!("[mesh] face meshing ({} faces): {:.1}s", face_loop_data.len(), _tp2.elapsed().as_secs_f32());

    // Phase 2c: Merge chunk outputs into global state.
    // Chunk 0 is the base; interior vertices from subsequent chunks are remapped.
    let (global_vertices, global_normals, all_indices, face_ranges, report, wire_diag) =
        merge_chunk_outputs(chunk_outputs, boundary_vertex_count, collect_diag);

    let face_infos_len = face_infos.len();

    // Post-processing: heal skipped faces, refine, cull, compact, weld, optimize
    finalize_shell_mesh(
        shell_key,
        reg,
        &face_ranges,
        all_indices,
        global_vertices,
        global_normals,
        report,
        &heal_skipped_faces,
        &scaled_config,
        shell_diag,
        face_infos_len,
        boundary_vertex_count,
        &wire_diag,
        collect_diag,
    )
}

/// Analytic surfaces produce exact meshes — chord error is always negligible.
/// Only freeform surfaces (BSpline, Offset) need post-mesh chord verification.
fn surface_chord_error_is_trivial(surface: &SurfaceGeom) -> bool {
    matches!(
        surface,
        SurfaceGeom::Plane { .. }
            | SurfaceGeom::Cylinder { .. }
            | SurfaceGeom::Cone { .. }
            | SurfaceGeom::Sphere { .. }
            | SurfaceGeom::Torus { .. }
            | SurfaceGeom::Revolution { .. }
            | SurfaceGeom::Extrusion { .. }
    )
}
