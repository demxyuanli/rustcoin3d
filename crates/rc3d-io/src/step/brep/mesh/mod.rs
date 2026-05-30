pub mod edge_disc;
pub mod refiner;
pub mod optimize;
pub mod face_uv;
pub mod face_fill;
pub mod face_cdt;
pub mod same_param;
pub mod report;
pub mod t4_quality;
pub mod diagnostic;
pub mod algo_factory;
pub mod void_subtract;
pub mod face_dispatch;
pub mod shell_mesh;

pub use face_dispatch::{algo_from_plan, FaceMeshPlan, SurfaceFillReason, plan_face_mesh};
pub use shell_mesh::{mesh_brep_shell, mesh_brep_shell_with_report, ShellMeshOutput};

use std::collections::{HashMap, HashSet};
use rc3d_core::math::Vec3;
use rc3d_core::utils::hash::f32x3_quantized_bits;
use super::topo::{ShellKey, EdgeKey, FaceKey, Orientation, VertexKey};
use super::registry::BRepRegistry;
use super::geom::SurfaceGeom;
use crate::step::mesh_result::MeshResult;
use edge_disc::{EdgeDiscConfig, EdgePolygon, discretize_all_edges};
use face_fill::{
    face_boundary_is_mixed, fix_tri_winding, FaceFillConfig, FaceMeshRange, fill_trimmed,
    mesh_plane_fan_3d, mesh_plane_fan_wire_polygons, mesh_revolution_native_grid,
    mesh_ruled_two_wire_edges, mesh_ruled_wire_polygons_3d, surface_fill_3d,
    measure_face_chord_error, surface_is_revolution_like, prefers_native_uv_ruled,
};
use super::geom::SurfaceParamRange;
use face_uv::{
    collect_face_loops, loops_from_boundary_indices, loops_from_wire_edges,
    loops_native_surface_uv, point_in_trim, repair_plane_loop_uv,
    revolution_boundary_v_collapsed, revolution_loops_from_wire_edges, signed_area_2d,
    unwrap_periodic_uv_loops, uv_loop_is_degenerate,
    FaceUvLoops, UvSource,
};
use refiner::{merge_refined_face, refine_mesh_interior, extract_face_mesh_with_map, RefineConfig};
use optimize::{OptimizeConfig, optimize_mesh};
use same_param::apply_same_parameter;
use algo_factory::FaceMeshAlgo;
use diagnostic::{diag_enabled, format_wire_loop_lines, log_mesh_coordinates_if_requested};
use report::{
    apply_relative_deflection, shell_bbox_diagonal, FaceMeshStats, ShellMeshReport,
};

/// UV grid resolution fallback when deflection config is unavailable.
pub const MESH_CLOSED_SURFACE_SEGS: u32 = 64;

fn min_adequate_trim_tris(surface: &SurfaceGeom, _wire_edge_count: usize) -> usize {
    match surface {
        SurfaceGeom::Revolution { .. }
        | SurfaceGeom::BSpline(_)
        | SurfaceGeom::Offset { .. } => 64,
        _ => 8,
    }
}

/// Count triangles with non-zero area in a face index range (before global cull).
fn count_valid_tris_in_range(
    indices: &[i32],
    vertices: &[Vec3],
    first_tri: usize,
    tri_count: usize,
) -> usize {
    let mut valid = 0usize;
    for ti in first_tri..first_tri.saturating_add(tri_count) {
        let base = ti * 4;
        if base + 3 >= indices.len() {
            break;
        }
        if indices[base + 3] != -1 {
            continue;
        }
        let (i0, i1, i2) = (
            indices[base] as usize,
            indices[base + 1] as usize,
            indices[base + 2] as usize,
        );
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        let area = (vertices[i0] - vertices[i1])
            .cross(vertices[i0] - vertices[i2])
            .length();
        if area > 1e-12 {
            valid += 1;
        }
    }
    valid
}

/// Ruled quad strip between exactly two boundary wires (no untrimmed native UV sheet).
#[allow(clippy::too_many_arguments)]
fn try_ruled_two_wire_mesh(
    face_key: FaceKey,
    face: &super::topo::BRepFace,
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &HashMap<(EdgeKey, usize), usize>,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    fill_config: &FaceFillConfig,
) -> FaceMeshRange {
    let first_tri = all_indices.len() / 4;
    if wire_edges.len() != 2 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
        };
    }

    let (mut segs_u, mut segs_v) = ruled_strip_uv_segs(wire_edges, edge_polygons, fill_config);
    let base = parametric_grid_segs(face, fill_config);
    segs_u = segs_u.max(base).min(64);
    segs_v = segs_v.max(2).min(32);

    if prefers_native_uv_ruled(&face.surface, wire_edges) {
        return mesh_ruled_two_wire_edges(
            face_key,
            face,
            wire_edges,
            edge_polygons,
            edge_boundary_idx,
            global_vertices,
            global_normals,
            all_indices,
            pos_to_idx,
            segs_u,
            segs_v,
        );
    }

    let we_orient: Vec<_> = wire_edges
        .iter()
        .map(|&(ek, ref pis)| (ek, Orientation::Forward, pis.clone()))
        .collect();

    let ruled = mesh_ruled_wire_polygons_3d(
        face_key,
        face,
        &we_orient,
        edge_polygons,
        global_vertices,
        global_normals,
        all_indices,
        segs_u,
        segs_v,
    );
    if ruled.tri_count > 0 {
        ruled
    } else {
        mesh_ruled_two_wire_edges(
            face_key,
            face,
            wire_edges,
            edge_polygons,
            edge_boundary_idx,
            global_vertices,
            global_normals,
            all_indices,
            pos_to_idx,
            segs_u,
            segs_v,
        )
    }
}

/// Register a boundary point in the shared pool (quantized dedup; topology vertices seeded first).
fn register_boundary_point(
    pt: Vec3,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
) -> usize {
    let hash = f32x3_quantized_bits([pt.x, pt.y, pt.z]);
    *pos_to_idx.entry(hash).or_insert_with(|| {
        let i = global_vertices.len();
        global_vertices.push(pt);
        global_normals.push(Vec3::ZERO);
        i
    })
}

fn enrich_loops_from_wire_edges(
    face_key: FaceKey,
    face: &super::topo::BRepFace,
    _loops: &FaceUvLoops,
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &HashMap<(EdgeKey, usize), usize>,
    global_vertices: &[Vec3],
) -> Option<FaceUvLoops> {
    if wire_edges.is_empty() { return None; }
    let inv_tol = face.tolerance.max(1e-3);
    let mut enriched = loops_from_wire_edges(face_key, &face.surface, inv_tol, wire_edges, edge_polygons, edge_boundary_idx, global_vertices)?;
    unwrap_periodic_uv_loops(&mut enriched, &face.surface);
    if matches!(face.surface, SurfaceGeom::Plane { .. }) {
        repair_plane_loop_uv(&mut enriched.outer, face, global_vertices);
        if !uv_loop_is_degenerate(&enriched) { return Some(enriched); }
    } else if enriched.uv_source == UvSource::Pcurve && !uv_loop_is_degenerate(&enriched) {
        return Some(enriched);
    }
    let native = loops_native_surface_uv(&enriched, face, global_vertices);
    if !uv_loop_is_degenerate(&native) { Some(native) }
    else if !uv_loop_is_degenerate(&enriched) { Some(enriched) }
    else { None }
}

fn revolution_prefers_ruled_first(
    face: &super::topo::BRepFace, mesh_loops: &FaceUvLoops, wire_edges: &[(EdgeKey, Vec<usize>)],
) -> bool {
    if wire_edges.len() != 2 { return false; }
    if !matches!(face.surface, SurfaceGeom::Revolution { .. } | SurfaceGeom::Offset { .. }) { return false; }
    if uv_loop_is_degenerate(mesh_loops) { return true; }
    let outer_uv: Vec<(f32, f32)> = mesh_loops.outer.boundary.iter().map(|v| (v.uv.0, v.uv.1)).collect();
    signed_area_2d(&outer_uv).abs() <= 1e-6
}

/// True when UV bounds cover most of the native period (untrimmed analytic sheet).
fn uv_bounds_span_untrimmed_period(
    surface: &SurfaceGeom,
    bounds: (f32, f32, f32, f32),
) -> bool {
    let pr = surface.param_range();
    let du = (bounds.1 - bounds.0) / (pr.u_max - pr.u_min).max(1e-6);
    let dv = (bounds.3 - bounds.2) / (pr.v_max - pr.v_min).max(1e-6);
    du > 0.85 && dv > 0.85
}

/// Revolution trim fallback: use native U from loop UV and V from axis-angle sampling when boundary V is degenerate.
fn revolution_fallback_uv_bounds(
    loops: &face_uv::FaceUvLoops,
    face: &super::topo::BRepFace,
    global_vertices: &[Vec3],
) -> Option<(f32, f32, f32, f32)> {
    let (u0, u1, v0, v1) = loops.native_uv_bounds()?;
    if (u1 - u0).abs() < 1e-6 {
        return None;
    }
    if (v1 - v0).abs() >= 1e-5 {
        return Some((u0, u1, v0, v1));
    }
    let (rv0, rv1) = loops.revolution_v_bounds_from_3d(face, global_vertices)?;
    Some((u0, u1, rv0, rv1))
}

fn allows_trimmed_uv_grid(
    reg: &BRepRegistry,
    face: &super::topo::BRepFace,
    loops: &FaceUvLoops,
    global_vertices: &[Vec3],
) -> bool {
    if loops.native_uv_bounds().is_none()
        && loops.uv_bounds_from_projection(face, global_vertices).is_none()
    {
        return false;
    }
    let wire_len = reg
        .wires
        .get(face.outer_wire)
        .map(|w| w.edges.len())
        .unwrap_or(0);
    if wire_len == 0 {
        return false;
    }
    matches!(
        &face.surface,
        SurfaceGeom::Revolution { .. }
            | SurfaceGeom::Cone { .. }
            | SurfaceGeom::Cylinder { .. }
            | SurfaceGeom::Sphere { .. }
            | SurfaceGeom::Torus { .. }
            | SurfaceGeom::Extrusion { .. }
            | SurfaceGeom::Offset { .. }
            | SurfaceGeom::BSpline(_)
            | SurfaceGeom::Plane { .. }
    )
}

fn allows_parametric_grid_fallback(
    reg: &BRepRegistry,
    face: &super::topo::BRepFace,
    loops: &face_uv::FaceUvLoops,
) -> bool {
    let Some(bounds) = loops.native_uv_bounds() else {
        return false;
    };
    let wire_len = reg
        .wires
        .get(face.outer_wire)
        .map(|w| w.edges.len())
        .unwrap_or(0);
    match &face.surface {
        SurfaceGeom::Revolution { .. }
        | SurfaceGeom::Cone { .. }
        | SurfaceGeom::Cylinder { .. }
        | SurfaceGeom::Offset { .. } => {
            // Few edges = trimmed patch; never paint the full native period (causes cone/sheet artifacts).
            if wire_len < 3 {
                return false;
            }
            !uv_bounds_span_untrimmed_period(&face.surface, bounds)
        }
        _ => true,
    }
}

/// VERTEX_LOOP on a closed analytic surface: wire is only pole degeneracy (+ optional seam),
/// not a trim boundary. Mesh the full native parameter rectangle (OCC closed-face path).
fn uses_closed_parametric_mesh(reg: &BRepRegistry, face: &super::topo::BRepFace) -> bool {
    if !face.inner_wires.is_empty() {
        return false;
    }
    if !matches!(
        &face.surface,
        SurfaceGeom::Sphere { .. } | SurfaceGeom::Torus { .. }
    ) {
        return false;
    }
    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => return false,
    };
    if wire.edges.is_empty() {
        return true;
    }
    if face.degenerated_edges.iter().any(|&dek| {
        wire.edges.iter().any(|&(ek, _)| ek == dek)
    }) {
        return true;
    }
    wire.edges.iter().all(|&(ek, _)| {
        let Some(e) = reg.edges.get(ek) else {
            return false;
        };
        e.v_low == e.v_high || face.seam_edges.contains(&ek)
    })
}

#[derive(Debug, Clone)]
pub struct BRepMeshConfig {
    pub edge: EdgeDiscConfig,
    pub face: FaceFillConfig,
    pub refine: RefineConfig,
    pub optimize: OptimizeConfig,
    /// When > 0, deflection = shell_bbox_diagonal * relative_deflection (OCC Relative).
    pub relative_deflection: f32,
    pub same_parameter_tol: f32,
}

impl Default for BRepMeshConfig {
    fn default() -> Self {
        Self {
            edge: EdgeDiscConfig::default(),
            face: FaceFillConfig::default(),
            refine: RefineConfig::default(),
            optimize: OptimizeConfig::default(),
            relative_deflection: 0.0,
            same_parameter_tol: 1e-4,
        }
    }
}

// Shell-level meshing lives in shell_mesh.rs (re-exported above).

fn mesh_brep_shell_with_report_impl(
    shell_key: ShellKey,
    reg: &BRepRegistry,
    config: &BRepMeshConfig,
    skip_face_keys: &[FaceKey],
) -> ShellMeshOutput {
    let shell_diag = shell_bbox_diagonal(shell_key, reg);
    let mut scaled_config = config.clone();
    apply_relative_deflection(&mut scaled_config, shell_diag);
    if scaled_config.relative_deflection > 0.0 && shell_diag > 0.0 {
        scaled_config.face.shell_min_size =
            shell_diag * scaled_config.face.min_size_relative;
    }

    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => {
            return ShellMeshOutput {
                mesh: MeshResult::default(),
                report: ShellMeshReport {
                    shell_diag,
                    ..Default::default()
                },
            };
        }
    };

    let mut report = ShellMeshReport {
        face_count: shell.faces.len(),
        shell_diag,
        ..Default::default()
    };

    // Phase 1: edge discretization
    let mut edge_polygons: HashMap<EdgeKey, EdgePolygon> =
        discretize_all_edges(reg, &scaled_config.edge);

    // SameParameter snap (Phase 1b)
    if scaled_config.same_parameter_tol > 0.0 {
        apply_same_parameter(
            &mut edge_polygons,
            reg,
            scaled_config.same_parameter_tol,
        );
    }

    // Phase 2: shared boundary vertex pool (B-Rep vertices own canonical indices)
    let mut global_vertices: Vec<Vec3> = Vec::new();
    let mut global_normals: Vec<Vec3> = Vec::new();
    let mut pos_to_idx: HashMap<[u32; 3], usize> = HashMap::new();
    let mut vertex_mesh_idx: HashMap<VertexKey, usize> = HashMap::new();
    let mut edge_boundary_idx: HashMap<(EdgeKey, usize), usize> = HashMap::new();

    for (vk, v) in reg.vertices.iter() {
        let idx = register_boundary_point(
            v.position,
            &mut global_vertices,
            &mut global_normals,
            &mut pos_to_idx,
        );
        vertex_mesh_idx.insert(vk, idx);
    }

    let mut total_edge_pts = 0usize;
    for (&ek, poly) in &edge_polygons {
        total_edge_pts += poly.params_3d.len();
        let edge = reg.edges.get(ek);
        let n = poly.params_3d.len();
        for (pi, &(_t, pt)) in poly.params_3d.iter().enumerate() {
            let idx = if let Some(e) = edge {
                if e.v_low != e.v_high {
                    if pi == 0 {
                        vertex_mesh_idx.get(&e.v_low).copied()
                    } else if pi + 1 == n {
                        vertex_mesh_idx.get(&e.v_high).copied()
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            let idx = idx.unwrap_or_else(|| {
                register_boundary_point(
                    pt,
                    &mut global_vertices,
                    &mut global_normals,
                    &mut pos_to_idx,
                )
            });
            edge_boundary_idx.insert((ek, pi), idx);
        }
    }
    log::debug!(
        "[BRep mesh] {} edge polygons, {} total edge pts -> {} unique boundary verts",
        edge_polygons.len(),
        total_edge_pts,
        global_vertices.len()
    );

    struct FaceWireInfo {
        face_key: FaceKey,
        wire_edges: Vec<(EdgeKey, Vec<usize>)>,
    }

    let mut face_infos: Vec<FaceWireInfo> = Vec::new();
    let mut heal_skipped_faces: Vec<FaceKey> = Vec::new();

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

    let mut all_indices: Vec<i32> = Vec::new();
    let mut face_ranges: Vec<FaceMeshRange> = Vec::new();
    let collect_diag = diag_enabled();
    let mut wire_diag: Vec<String> = Vec::new();

    for info in &face_infos {
        let face = match reg.faces.get(info.face_key) {
            Some(f) => f,
            None => continue,
        };

        if info.wire_edges.is_empty() || uses_closed_parametric_mesh(reg, face) {
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
            report.faces.push(FaceMeshStats {
                face_key: info.face_key,
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

        let loops = collect_face_loops(
            info.face_key,
            face,
            reg,
            &edge_polygons,
            &edge_boundary_idx,
            &global_vertices,
        );

        if collect_diag && !info.wire_edges.is_empty() {
            let we_orient: Vec<_> = info.wire_edges.iter().map(|&(ek, ref pis)| (ek, Orientation::Forward, pis.clone())).collect();
            wire_diag.extend(format_wire_loop_lines(
                info.face_key,
                &loops,
                &we_orient,
                &edge_boundary_idx,
                &global_vertices,
            ));
        }

        let wire_empty = info.wire_edges.is_empty();
        let mesh_plan = plan_face_mesh(face, &loops, wire_empty, false);
        log::trace!("[BRep mesh] face {:?} plan {:?}", info.face_key, mesh_plan);
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
        );
        let mut algo = algo_from_plan(mesh_plan).unwrap_or(FaceMeshAlgo::SurfaceFill3d);
        if info.wire_edges.len() == 2 && uv_loop_is_degenerate(&loops) {
            algo = FaceMeshAlgo::TrimmedCdt;
        }
        if prefer_native_cdt && loops.is_fillable() {
            algo = FaceMeshAlgo::TrimmedCdt;
        }
        if matches!(face.surface, SurfaceGeom::Plane { .. }) && loops.is_fillable() {
            algo = FaceMeshAlgo::TrimmedCdt;
        }
        let mut used_surface_fill = algo == FaceMeshAlgo::SurfaceFill3d;
        let mixed_boundary = algo == FaceMeshAlgo::TrimmedCdt
            && !prefer_native_cdt
            && !matches!(face.surface, SurfaceGeom::Plane { .. })
            && face_boundary_is_mixed(&loops, &global_vertices);

        let mut boundary_ordered = Vec::new();
        for &(ek, ref pis) in &info.wire_edges {
            for &pi in pis {
                if let Some(gi) = edge_boundary_idx.get(&(ek, pi)).copied() {
                    if boundary_ordered.last() != Some(&gi) {
                        boundary_ordered.push(gi);
                    }
                }
            }
        }
        if boundary_ordered.len() >= 2 && boundary_ordered.first() == boundary_ordered.last() {
            boundary_ordered.pop();
        }

        let wire_len = info.wire_edges.len();
        let multi_wire_rev = wire_len >= 3
            && matches!(face.surface, SurfaceGeom::Revolution { .. })
            && boundary_ordered.len() >= 3;

        // Plane two-wire patches: ruled strip when PCURVE loop is not fillable.
        if wire_len == 2
            && matches!(face.surface, SurfaceGeom::Plane { .. })
            && !loops.is_fillable()
        {
            let ruled = try_ruled_two_wire_mesh(
                info.face_key,
                face,
                &info.wire_edges,
                &edge_polygons,
                &edge_boundary_idx,
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                &mut pos_to_idx,
                &scaled_config.face,
            );
            if ruled.tri_count > 0 {
                let tri_n = ruled.tri_count;
                report.faces.push(FaceMeshStats {
                    face_key: info.face_key,
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
                    info.face_key,
                    tri_n
                );
                continue;
            }
        }

        let tris_before_face = all_indices.len() / 4;
        let chord_reject = (scaled_config.face.deflection_interior * 10.0).max(0.05);
        let min_adequate = min_adequate_trim_tris(&face.surface, wire_len);

        if wire_len == 2 && prefers_native_uv_ruled(&face.surface, &info.wire_edges) {
            let ruled = try_ruled_two_wire_mesh(
                info.face_key,
                face,
                &info.wire_edges,
                &edge_polygons,
                &edge_boundary_idx,
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                &mut pos_to_idx,
                &scaled_config.face,
            );
            let valid = count_valid_tris_in_range(
                &all_indices,
                &global_vertices,
                ruled.first_tri,
                ruled.tri_count,
            );
            if valid >= min_adequate {
                report.faces.push(FaceMeshStats {
                    face_key: info.face_key,
                    tri_count: valid,
                    first_tri: ruled.first_tri,
                    uv_source: UvSource::Synthetic,
                    max_chord_error: ruled.max_chord_error,
                    grid_fallback: false,
                });
                face_ranges.push(FaceMeshRange {
                    face_key: info.face_key,
                    first_tri: ruled.first_tri,
                    tri_count: valid,
                    boundary_global: ruled.boundary_global,
                    max_chord_error: ruled.max_chord_error,
                });
                report.meshed_faces += 1;
                log::debug!(
                    "[BRep mesh] face {:?}: revolution-like ruled two-wire ({} tris)",
                    info.face_key,
                    valid
                );
                continue;
            }
            all_indices.truncate(tris_before_face * 4);
        }

        // Revolution/Offset faces: rebuild UV loops for CDT (avoid degenerate PCURVE / collapsed V).
        let mut mesh_loops = loops.clone();
        if multi_wire_rev {
            if let Some(s) = revolution_loops_from_wire_edges(
                info.face_key,
                face,
                &info.wire_edges,
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
                info.face_key,
                face,
                &info.wire_edges,
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
                info.face_key,
                face,
                &info.wire_edges,
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
        if !uv_loop_is_degenerate(&mesh_loops) {
            algo = FaceMeshAlgo::TrimmedCdt;
        }

        if wire_len == 1
            && matches!(face.surface, SurfaceGeom::Plane { .. })
            && boundary_ordered.len() >= 3
        {
            let fan = mesh_plane_fan_3d(
                info.face_key,
                &boundary_ordered,
                face,
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                &mut pos_to_idx,
            );
            if fan.tri_count > 0 {
                let tri_n = fan.tri_count;
                report.faces.push(FaceMeshStats {
                    face_key: info.face_key,
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
                    info.face_key,
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
                    info.face_key,
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
                info.face_key,
                &boundary_ordered,
                face,
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                &mut pos_to_idx,
                &scaled_config.face,
            ),
            FaceMeshAlgo::TrimmedCdt if mixed_boundary => surface_fill_3d(
                info.face_key,
                &boundary_ordered,
                face,
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                &mut pos_to_idx,
                &scaled_config.face,
            ),
            FaceMeshAlgo::TrimmedCdt | FaceMeshAlgo::ClosedParametric => fill_trimmed(
                info.face_key,
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

        range.face_key = info.face_key;

        if used_surface_fill
            && range.tri_count > 0
            && range.max_chord_error > chord_reject
        {
            log::debug!(
                "[BRep mesh] face {:?}: reject surface fill (chord {:.4} > {:.4}), retry trimmed CDT",
                info.face_key,
                range.max_chord_error,
                chord_reject
            );
            all_indices.truncate(tris_before_face * 4);
            range = fill_trimmed(
                info.face_key,
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
            range.face_key = info.face_key;
            used_surface_fill = false;
        }

        if used_surface_fill {
            report.grid_fallback_count += 1;
        }

        if range.tri_count == 0 && !used_surface_fill && !mixed_boundary {
            range = surface_fill_3d(
                info.face_key,
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

        let mut used_parametric_grid = false;
        let keep_trimmed = range.tri_count > 0
            && range.max_chord_error <= chord_reject
            && !used_surface_fill
            && !mixed_boundary;

        let grid_loops = if matches!(face.surface, SurfaceGeom::Revolution { .. })
            || prefer_native_cdt
            || matches!(face.surface, SurfaceGeom::Plane { .. })
        {
            loops_native_surface_uv(&mesh_loops, face, &global_vertices)
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
                info.face_key,
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

        if wire_len == 2 && (range.tri_count < min_adequate || used_surface_fill) {
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
                    info.face_key,
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
                    range.face_key = info.face_key;
                    used_surface_fill = true;
                }
            }

            let skip_ruled = range.tri_count >= min_adequate
                && (!used_surface_fill || range.max_chord_error <= chord_reject);

            if skip_ruled {
                // adequate trimmed / surface fill
            } else if matches!(face.surface, SurfaceGeom::Offset { .. })
                && boundary_ordered.len() >= 3
            {
                if range.tri_count > 0 {
                    all_indices.truncate(range.first_tri * 4);
                }
                let sf = surface_fill_3d(
                    info.face_key,
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
                    range.face_key = info.face_key;
                    used_surface_fill = true;
                }
            } else {
            let saved_first = range.first_tri;
            let saved_count = range.tri_count;
            if range.tri_count > 0 {
                all_indices.truncate(range.first_tri * 4);
            }
            range = try_ruled_two_wire_mesh(
                info.face_key,
                face,
                &info.wire_edges,
                &edge_polygons,
                &edge_boundary_idx,
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                &mut pos_to_idx,
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
            if ruled_ok && !matches!(face.surface, SurfaceGeom::Plane { .. }) {
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
                    info.face_key,
                    valid,
                    ruled_tri_count,
                    valid_ratio
                );
            }
            }
        }

        if (range.tri_count < min_adequate)
            && !multi_wire_rev
            && !matches!(face.surface, SurfaceGeom::Plane { .. })
            && allows_trimmed_uv_grid(reg, face, &grid_loops, &global_vertices)
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
                        face_key: info.face_key,
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
                        || (saved_count > 0 && grid_chord <= saved_chord.max(chord_reject) * 1.5))
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
                            info.face_key,
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
                            info.face_key,
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
                        info.face_key,
                        grid_tris
                    );
                }
            }
        }

        if range.tri_count == 0 && boundary_ordered.len() >= 3 {
            let sf = surface_fill_3d(
                info.face_key,
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
                range.face_key = info.face_key;
                used_surface_fill = true;
                report.grid_fallback_count += 1;
            }
        }

        report.faces.push(FaceMeshStats {
            face_key: info.face_key,
            tri_count: range.tri_count,
            first_tri: range.first_tri,
            uv_source: loops.uv_source,
            max_chord_error: range.max_chord_error,
            grid_fallback: used_surface_fill || used_parametric_grid,
        });

        if range.tri_count > 0 {
            log::debug!(
                "[BRep mesh] face {:?}: {} tris uv={:?} max_chord={:.6}",
                info.face_key,
                range.tri_count,
                loops.uv_source,
                range.max_chord_error,
            );
            face_ranges.push(range);
            report.meshed_faces += 1;
        }
    }

    for face_key in heal_skipped_faces {
        report.faces.push(FaceMeshStats {
            face_key,
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
        for range in &face_ranges {
            let face = match reg.faces.get(range.face_key) {
                Some(f) => f,
                None => continue,
            };
            // Planes are exact; post-refine Steiner splits break watertight quads (Cube.step).
            if matches!(face.surface, SurfaceGeom::Plane { .. }) {
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
        stats.max_chord_error =
            measure_face_chord_error(face, &global_vertices, &all_indices, range);
    }
    report.log_summary(shell_key);

    log::debug!(
        "[BRep mesh] shell {:?}: {} boundary verts, {} faces, {} tris total",
        shell_key,
        global_vertices.len(),
        face_infos.len(),
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
    recompute_normals_from_tris(&mesh.vertices, &mesh.indices, &mut mesh.normals);
    optimize_mesh(&mut mesh, &scaled_config.optimize);
    if collect_diag {
        log_mesh_coordinates_if_requested(
            shell_key,
            reg,
            &mesh.vertices,
            &mesh.indices,
            &face_ranges,
            &report,
            &wire_diag,
        );
    }
    ShellMeshOutput { mesh, report }
}

// shell_mesh.rs wraps the implementation above.

/// Area-weighted vertex normals from final triangle winding (fixes ruled/off-surface fills).
fn recompute_normals_from_tris(vertices: &[Vec3], indices: &[i32], normals: &mut Vec<Vec3>) {
    if normals.len() != vertices.len() {
        normals.resize(vertices.len(), Vec3::ZERO);
    }
    for n in normals.iter_mut() {
        *n = Vec3::ZERO;
    }
    for chunk in indices.chunks(4) {
        if chunk.len() < 4 || chunk[3] != -1 {
            continue;
        }
        let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        let n = (vertices[i1] - vertices[i0]).cross(vertices[i2] - vertices[i0]);
        if n.length_squared() < 1e-20 {
            continue;
        }
        normals[i0] += n;
        normals[i1] += n;
        normals[i2] += n;
    }
    for n in normals.iter_mut() {
        let len = n.length();
        if len > 1e-10 {
            *n = *n * (1.0 / len);
        } else {
            *n = Vec3::Y;
        }
    }
}

fn cull_degenerate_tris(indices: &mut Vec<i32>, vertices: &[Vec3]) -> usize {
    let mut out = Vec::with_capacity(indices.len());
    let mut removed = 0usize;
    for chunk in indices.chunks(4) {
        if chunk.len() < 4 || chunk[3] != -1 {
            out.extend_from_slice(chunk);
            continue;
        }
        let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            out.extend_from_slice(chunk);
            continue;
        }
        let area = (vertices[i0] - vertices[i1]).cross(vertices[i0] - vertices[i2]).length();
        if area > 1e-12 {
            out.extend_from_slice(chunk);
        } else {
            removed += 1;
        }
    }
    *indices = out;
    removed
}

fn sample_ruled_polyline(curve: &[Vec3], t: f32) -> Vec3 {
    if curve.is_empty() {
        return Vec3::ZERO;
    }
    if curve.len() == 1 {
        return curve[0];
    }
    let f = t * (curve.len() - 1) as f32;
    let i = f.floor() as usize;
    let j = (i + 1).min(curve.len() - 1);
    let u = f - i as f32;
    curve[i].lerp(curve[j], u)
}

/// Grid resolution for a two-wire ruled strip: u along edges, v across strip width.
fn ruled_strip_uv_segs(
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    config: &FaceFillConfig,
) -> (u32, u32) {
    let defl = config.deflection_interior.max(1e-6);
    let mut max_arc = 0.0f32;
    let mut curves: [Vec<Vec3>; 2] = [Vec::new(), Vec::new()];
    for (side, &(ek, ref pis)) in wire_edges.iter().enumerate() {
        let Some(poly) = edge_polygons.get(&ek) else {
            continue;
        };
        let mut chain_wire = 0.0f32;
        let mut prev: Option<Vec3> = None;
        for &pi in pis {
            let Some(&(_, pt)) = poly.params_3d.get(pi) else {
                continue;
            };
            if let Some(p) = prev {
                chain_wire += (pt - p).length();
            }
            prev = Some(pt);
            if curves[side]
                .last()
                .map(|q| (*q - pt).length_squared() > 1e-14)
                .unwrap_or(true)
            {
                curves[side].push(pt);
            }
        }
        let mut chain_full = 0.0f32;
        for w in poly.params_3d.windows(2) {
            chain_full += (w[1].1 - w[0].1).length();
        }
        if curves[side].len() < 2 && poly.params_3d.len() >= 2 {
            curves[side].clear();
            for &(_, pt) in &poly.params_3d {
                if curves[side]
                    .last()
                    .map(|q| (*q - pt).length_squared() > 1e-14)
                    .unwrap_or(true)
                {
                    curves[side].push(pt);
                }
            }
        }
        max_arc = max_arc.max(chain_wire.max(chain_full));
    }
    let segs_u = ((max_arc / (defl * 0.85)).ceil() as u32).clamp(8, 96);
    let mut max_width = 0.0f32;
    if curves[0].len() >= 2 && curves[1].len() >= 2 {
        for i in 0..=16 {
            let t = i as f32 / 16.0;
            let p0 = sample_ruled_polyline(&curves[0], t);
            let p1 = sample_ruled_polyline(&curves[1], t);
            max_width = max_width.max((p1 - p0).length());
        }
    }
    let segs_v = ((max_width / (defl * 0.85)).ceil() as u32).clamp(4, 48);
    (segs_u, segs_v)
}

/// Grid resolution from DeflectionInterior (OCC BRepMesh-style, not a fixed 64x64).
fn parametric_grid_segs(face: &super::topo::BRepFace, config: &FaceFillConfig) -> u32 {
    let defl = config.deflection_interior.max(1e-6);
    match &face.surface {
        SurfaceGeom::Plane { .. } => 2,
        SurfaceGeom::Sphere { radius, .. } => {
            let circ = 2.0 * std::f32::consts::PI * radius;
            ((circ / defl).ceil() as u32).clamp(8, 128)
        }
        SurfaceGeom::Torus { major_r, minor_r, .. } => {
            let circ = 2.0 * std::f32::consts::PI * (major_r + minor_r);
            ((circ / defl).ceil() as u32).clamp(8, 128)
        }
        SurfaceGeom::Cylinder { radius, .. } => {
            let circ = 2.0 * std::f32::consts::PI * radius;
            ((circ / defl).ceil() as u32).clamp(8, 128)
        }
        SurfaceGeom::Cone { radius_at_apex, .. } => {
            let circ = 2.0 * std::f32::consts::PI * radius_at_apex.max(1e-6);
            ((circ / defl).ceil() as u32).clamp(8, 128)
        }
        SurfaceGeom::Revolution { generatrix, .. } => {
            let max_r = generatrix_max_radius(generatrix);
            let circ = 2.0 * std::f32::consts::PI * max_r;
            ((circ / defl).ceil() as u32).clamp(8, 64)
        }
        SurfaceGeom::BSpline(_) | SurfaceGeom::Offset { .. } => 24,
        _ => 24,
    }
}

/// Estimate max radius of a revolution generatrix curve.
fn generatrix_max_radius(curve: &super::geom::CurveGeom) -> f32 {
    let n = 16;
    let mut max_r2 = 0.0f32;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let p = curve.d0(t);
        max_r2 = max_r2.max(p.x * p.x + p.y * p.y);
    }
    max_r2.sqrt().max(1.0)
}

/// Parametric grid over the boundary UV bounding box (no trim test; for 2-edge revolution patches).
fn mesh_uv_bbox_grid(
    face: &super::topo::BRepFace,
    uv_bounds: (f32, f32, f32, f32),
    fill_config: Option<&FaceFillConfig>,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    all_indices: &mut Vec<i32>,
) {
    mesh_parametric_grid(
        face,
        Some(uv_bounds),
        fill_config,
        global_vertices,
        global_normals,
        pos_to_idx,
        all_indices,
    );
}

/// Parametric grid clipped to the face trim (outer/holes in UV).

// -- Sutherland-Hodgman polygon clipping helpers for partial trim cells --

/// Check if a 2D polygon is convex by verifying all cross products have the same sign.
fn is_polygon_convex(poly: &[(f32, f32)]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let n = poly.len();
    let mut positive = false;
    let mut negative = false;
    for i in 0..n {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % n];
        let p2 = poly[(i + 2) % n];
        let cross = (p1.0 - p0.0) * (p2.1 - p1.1) - (p1.1 - p0.1) * (p2.0 - p1.0);
        if cross > 1e-8 {
            positive = true;
        } else if cross < -1e-8 {
            negative = true;
        }
        if positive && negative {
            return false;
        }
    }
    true
}

/// Find the intersection point of two 2D line segments (p1->p2) and (p3->p4).
/// Returns the intersection point if the segments are not parallel and intersect.
fn line_segment_intersection(
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    p4: (f32, f32),
) -> Option<(f32, f32)> {
    let dx1 = p2.0 - p1.0;
    let dy1 = p2.1 - p1.1;
    let dx2 = p4.0 - p3.0;
    let dy2 = p4.1 - p3.1;
    let denom = dx1 * dy2 - dy1 * dx2;
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = ((p3.0 - p1.0) * dy2 - (p3.1 - p1.1) * dx2) / denom;
    let s = ((p3.0 - p1.0) * dy1 - (p3.1 - p1.1) * dx1) / denom;
    if t >= -1e-6 && t <= 1.0 + 1e-6 && s >= -1e-6 && s <= 1.0 + 1e-6 {
        Some((p1.0 + t * dx1, p1.1 + t * dy1))
    } else {
        None
    }
}

/// Clip a convex polygon (subject) against each edge of a clip polygon using
/// the Sutherland-Hodgman algorithm. Returns the resulting polygon vertices.
fn sutherland_hodgman_clip(
    subject: &[(f32, f32)],
    clip: &[(f32, f32)],
) -> Vec<(f32, f32)> {
    if subject.is_empty() || clip.len() < 3 {
        return subject.to_vec();
    }
    let mut output = subject.to_vec();
    let n = clip.len();
    for i in 0..n {
        if output.is_empty() {
            return Vec::new();
        }
        let a = clip[i];
        let b = clip[(i + 1) % n];
        let edge_dx = b.0 - a.0;
        let edge_dy = b.1 - a.1;
        let input = std::mem::take(&mut output);
        let m = input.len();
        for j in 0..m {
            let current = input[j];
            let prev = input[(j + m - 1) % m];
            let cur_inside = edge_dx * (current.1 - a.1) - edge_dy * (current.0 - a.0) >= -1e-8;
            let prev_inside = edge_dx * (prev.1 - a.1) - edge_dy * (prev.0 - a.0) >= -1e-8;
            match (prev_inside, cur_inside) {
                (true, true) => { output.push(current); }
                (true, false) => {
                    if let Some(pt) = line_segment_intersection(prev, current, a, b) {
                        output.push(pt);
                    }
                }
                (false, true) => {
                    if let Some(pt) = line_segment_intersection(prev, current, a, b) {
                        output.push(pt);
                    }
                    output.push(current);
                }
                (false, false) => {}
            }
        }
    }
    output
}

fn mesh_trimmed_uv_grid(
    face: &super::topo::BRepFace,
    loops: &FaceUvLoops,
    uv_bounds: (f32, f32, f32, f32),
    fill_config: Option<&FaceFillConfig>,
    grid_segs: Option<u32>,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    all_indices: &mut Vec<i32>,
) {
    let outer_uv: Vec<(f32, f32)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
    if outer_uv.len() < 3 {
        return;
    }
    let holes: Vec<Vec<(f32, f32)>> = loops
        .inners
        .iter()
        .map(|l| l.boundary.iter().map(|v| v.uv).collect())
        .collect();

    let segs = grid_segs
        .unwrap_or_else(|| {
            fill_config
                .map(|c| parametric_grid_segs(face, c))
                .unwrap_or(16)
        })
        .clamp(4, 64);
    let (u_min, u_max, v_min, v_max) = uv_bounds;

    for iu in 0..segs {
        for iv in 0..segs {
            let u0 = u_min + (u_max - u_min) * iu as f32 / segs as f32;
            let u1 = u_min + (u_max - u_min) * (iu + 1) as f32 / segs as f32;
            let v0 = v_min + (v_max - v_min) * iv as f32 / segs as f32;
            let v1 = v_min + (v_max - v_min) * (iv + 1) as f32 / segs as f32;
            let corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)];
            let n_inside = corners
                .iter()
                .filter(|&&(u, v)| point_in_trim(u, v, &outer_uv, &holes))
                .count();

            if n_inside == 0 {
                continue;
            }

            if n_inside == 4 {
                // Fully inside: emit quad as two triangles
                let mut idx = [0i32; 4];
                for (k, &(u, v)) in corners.iter().enumerate() {
                    let pt = face.surface.d0_native(u, v);
                    let mut n = face.surface.normal_native(u, v);
                    if !face.same_sense {
                        n = -n;
                    }
                    let hash = f32x3_quantized_bits([pt.x, pt.y, pt.z]);
                    let gi = *pos_to_idx.entry(hash).or_insert_with(|| {
                        let i = global_vertices.len();
                        global_vertices.push(pt);
                        global_normals.push(n);
                        i
                    });
                    idx[k] = gi as i32;
                }
                for (mut i0, mut i1, mut i2) in [(idx[0], idx[1], idx[2]), (idx[0], idx[2], idx[3])] {
                    if i0 == i1 || i1 == i2 || i2 == i0 {
                        continue;
                    }
                    fix_tri_winding(
                        &mut i0,
                        &mut i1,
                        &mut i2,
                        global_vertices,
                        &face.surface,
                        face.same_sense,
                    );
                    all_indices.extend_from_slice(&[i0, i1, i2, -1]);
                }
            } else {
                // Partially inside: clip cell quad against outer trim boundary.
                // NOTE: Sutherland-Hodgman requires a convex clip polygon. For non-convex
                // UV boundaries, fall back to using only the corners that are inside the trim.
                let cell_poly: Vec<(f32, f32)> = corners.to_vec();
                let clipped = if is_polygon_convex(&outer_uv) {
                    sutherland_hodgman_clip(&cell_poly, &outer_uv)
                } else {
                    // Non-convex fallback: keep only corners that pass point_in_trim
                    corners
                        .iter()
                        .copied()
                        .filter(|&(u, v)| point_in_trim(u, v, &outer_uv, &holes))
                        .collect()
                };
                if clipped.len() < 3 {
                    continue;
                }
                // Filter out points that fall inside a hole
                let clipped: Vec<(f32, f32)> = clipped
                    .into_iter()
                    .filter(|&(u, v)| point_in_trim(u, v, &outer_uv, &holes))
                    .collect();
                if clipped.len() < 3 {
                    continue;
                }
                // Fan triangulation from first vertex
                let mut tri_idx = Vec::new();
                for &(u, v) in &clipped {
                    let pt = face.surface.d0_native(u, v);
                    let mut n = face.surface.normal_native(u, v);
                    if !face.same_sense {
                        n = -n;
                    }
                    let hash = f32x3_quantized_bits([pt.x, pt.y, pt.z]);
                    let gi = *pos_to_idx.entry(hash).or_insert_with(|| {
                        let i = global_vertices.len();
                        global_vertices.push(pt);
                        global_normals.push(n);
                        i
                    });
                    tri_idx.push(gi as i32);
                }
                for k in 1..tri_idx.len().saturating_sub(1) {
                    let mut i0 = tri_idx[0];
                    let mut i1 = tri_idx[k];
                    let mut i2 = tri_idx[k + 1];
                    if i0 == i1 || i1 == i2 || i2 == i0 {
                        continue;
                    }
                    fix_tri_winding(
                        &mut i0,
                        &mut i1,
                        &mut i2,
                        global_vertices,
                        &face.surface,
                        face.same_sense,
                    );
                    all_indices.extend_from_slice(&[i0, i1, i2, -1]);
                }
            }
        }
    }
}

/// UV parametric grid tessellation for any surface type (last-resort / closed faces).
fn mesh_parametric_grid(
    face: &super::topo::BRepFace,
    uv_bounds: Option<(f32, f32, f32, f32)>,
    fill_config: Option<&FaceFillConfig>,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    all_indices: &mut Vec<i32>,
) {
    let segs = fill_config
        .map(|c| parametric_grid_segs(face, c))
        .unwrap_or_else(|| match &face.surface {
            SurfaceGeom::Plane { .. } => 2,
            _ => MESH_CLOSED_SURFACE_SEGS,
        });
    let pr = if let Some((u_min, u_max, v_min, v_max)) = uv_bounds {
        SurfaceParamRange {
            u_min,
            u_max,
            v_min,
            v_max,
        }
    } else {
        face.surface.param_range()
    };

    for iu in 0..segs {
        for iv in 0..segs {
            let u0 = pr.u_min + (pr.u_max - pr.u_min) * iu as f32 / segs as f32;
            let u1 = pr.u_min + (pr.u_max - pr.u_min) * (iu + 1) as f32 / segs as f32;
            let v0 = pr.v_min + (pr.v_max - pr.v_min) * iv as f32 / segs as f32;
            let v1 = pr.v_min + (pr.v_max - pr.v_min) * (iv + 1) as f32 / segs as f32;
            let corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)];
            let mut idx = [0i32; 4];
            for (k, &(u, v)) in corners.iter().enumerate() {
                let pt = face.surface.d0_native(u, v);
                let mut n = face.surface.normal_native(u, v);
                if !face.same_sense {
                    n = -n;
                }
                let hash = f32x3_quantized_bits([pt.x, pt.y, pt.z]);
                let gi = *pos_to_idx.entry(hash).or_insert_with(|| {
                    let i = global_vertices.len();
                    global_vertices.push(pt);
                    global_normals.push(n);
                    i
                });
                idx[k] = gi as i32;
            }
            for (mut i0, mut i1, mut i2) in [(idx[0], idx[1], idx[2]), (idx[0], idx[2], idx[3])] {
                if i0 == i1 || i1 == i2 || i2 == i0 {
                    continue;
                }
                fix_tri_winding(
                    &mut i0,
                    &mut i1,
                    &mut i2,
                    global_vertices,
                    &face.surface,
                    face.same_sense,
                );
                all_indices.extend_from_slice(&[i0, i1, i2, -1]);
                let p0 = global_vertices[i0 as usize];
                let p1 = global_vertices[i1 as usize];
                let p2 = global_vertices[i2 as usize];
                let tri_n = (p1 - p0).cross(p2 - p0);
                if tri_n.length() > 1e-10 {
                    let n = tri_n.normalize();
                    global_normals[i0 as usize] = global_normals[i0 as usize] + n;
                    global_normals[i1 as usize] = global_normals[i1 as usize] + n;
                    global_normals[i2 as usize] = global_normals[i2 as usize] + n;
                }
            }
        }
    }
}

/// Tessellate a closed analytic surface face (VERTEX_LOOP fallback when CDT cannot run).
fn mesh_closed_surface(
    face: &super::topo::BRepFace,
    fill_config: &FaceFillConfig,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    all_indices: &mut Vec<i32>,
) {
    mesh_parametric_grid(
        face,
        None,
        Some(fill_config),
        global_vertices,
        global_normals,
        pos_to_idx,
        all_indices,
    );
}

#[cfg(test)]
mod mesh_integration {
    use super::*;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::registry::BRepRegistry;
    use crate::step::brep::topo::{BRepFace, BRepShell, BRepWire, BRepSolid, Orientation};

    fn build_plane_square_shell() -> (BRepRegistry, ShellKey) {
        let mut reg = BRepRegistry::new();
        let face_key = {
            let wire = reg.wires.insert(BRepWire { edges: vec![] });
            reg.faces.insert(BRepFace {
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
            })
        };
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
            wire_edges.push((ek, Orientation::Forward));
        }
        let outer_wire = reg.wires.insert(BRepWire { edges: wire_edges });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = outer_wire;
        }
        let shell_key = reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let _ = reg.solids.insert(BRepSolid {
            outer_shell: shell_key,
            void_shells: vec![],
        });
        (reg, shell_key)
    }

    #[test]
    fn mesh_brep_shell_plane_face_has_interior_tris() {
        let (reg, shell_key) = build_plane_square_shell();
        let out = mesh_brep_shell_with_report(shell_key, &reg, &BRepMeshConfig::default(), &[]);
        let tri_count = out.mesh.indices.len() / 4;
        assert!(tri_count >= 2, "expected interior fill, got {tri_count} tris");
        assert!(!out.mesh.vertices.is_empty());
        assert!(out.report.meshed_faces >= 1);
    }
}
