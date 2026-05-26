pub mod edge_disc;
pub mod refiner;
pub mod optimize;
pub mod face_uv;
pub mod face_fill;
pub mod face_cdt;
pub mod same_param;
pub mod report;
pub mod t4_quality;

use std::collections::HashMap;
use rc3d_core::math::Vec3;
use rc3d_core::utils::hash::f32x3_quantized_bits;
use super::topo::{ShellKey, EdgeKey, FaceKey, Orientation};
use super::registry::BRepRegistry;
use super::geom::SurfaceGeom;
use crate::step::mesh_result::MeshResult;
use edge_disc::{EdgeDiscConfig, EdgePolygon, discretize_all_edges};
use face_fill::{FaceFillConfig, FaceMeshRange, fill_trimmed, surface_fill_3d, measure_face_chord_error};
use face_uv::{collect_face_loops, UvSource};
use refiner::{merge_refined_face, refine_mesh_interior, extract_face_mesh_with_map, RefineConfig};
use optimize::{OptimizeConfig, optimize_mesh};
use same_param::apply_same_parameter;
use report::{
    apply_relative_deflection, shell_bbox_diagonal, FaceMeshStats, ShellMeshReport,
};

/// UV grid resolution for closed analytic surfaces (sphere, torus).
pub const MESH_CLOSED_SURFACE_SEGS: u32 = 64;

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
            relative_deflection: 0.005,
            same_parameter_tol: 1e-4,
        }
    }
}

#[derive(Debug)]
pub struct ShellMeshOutput {
    pub mesh: MeshResult,
    pub report: ShellMeshReport,
}

/// Mesh a B-Rep shell (OCC BRepMesh_IncrementalMesh equivalent).
pub fn mesh_brep_shell(
    shell_key: ShellKey,
    reg: &BRepRegistry,
    config: &BRepMeshConfig,
    skip_face_keys: &[FaceKey],
) -> MeshResult {
    mesh_brep_shell_with_report(shell_key, reg, config, skip_face_keys).mesh
}

pub fn mesh_brep_shell_with_report(
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

    // Phase 2: shared boundary vertex pool
    let mut global_vertices: Vec<Vec3> = Vec::new();
    let mut global_normals: Vec<Vec3> = Vec::new();
    let mut pos_to_idx: HashMap<[u32; 3], usize> = HashMap::new();
    let mut edge_boundary_idx: HashMap<(EdgeKey, usize), usize> = HashMap::new();

    let mut total_edge_pts = 0usize;
    for (&ek, poly) in &edge_polygons {
        total_edge_pts += poly.params_3d.len();
        for (pi, &(_t, pt)) in poly.params_3d.iter().enumerate() {
            let hash = f32x3_quantized_bits([pt.x, pt.y, pt.z]);
            let idx = *pos_to_idx.entry(hash).or_insert_with(|| {
                let i = global_vertices.len();
                global_vertices.push(pt);
                global_normals.push(Vec3::ZERO);
                i
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

    for &(face_key, _orient) in &shell.faces {
        if skip_face_keys.contains(&face_key) {
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

    for info in &face_infos {
        let face = match reg.faces.get(info.face_key) {
            Some(f) => f,
            None => continue,
        };

        if info.wire_edges.is_empty() {
            let tris_before = all_indices.len() / 4;
            mesh_closed_surface(
                face,
                &mut global_vertices,
                &mut global_normals,
                &mut pos_to_idx,
                &mut all_indices,
            );
            let tri_count = all_indices.len() / 4 - tris_before;
            report.faces.push(FaceMeshStats {
                face_key: info.face_key,
                tri_count,
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

        let used_surface_fill = !loops.is_fillable() || loops.uv_source == UvSource::SurfaceFill;
        let mut range = if !used_surface_fill {
            fill_trimmed(
                info.face_key,
                &loops,
                face,
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                &mut pos_to_idx,
                &scaled_config.face,
            )
        } else {
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
            if boundary_ordered.len() >= 2
                && boundary_ordered.first() == boundary_ordered.last()
            {
                boundary_ordered.pop();
            }
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
        };

        range.face_key = info.face_key;
        if used_surface_fill {
            report.grid_fallback_count += 1;
        }
        report.faces.push(FaceMeshStats {
            face_key: info.face_key,
            tri_count: range.tri_count,
            uv_source: loops.uv_source,
            max_chord_error: range.max_chord_error,
            grid_fallback: used_surface_fill,
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

    let deg_after_fill = cull_degenerate_tris(&mut all_indices, &global_vertices);
    if deg_after_fill > 0 {
        log::debug!("[BRep mesh] removed {} degenerate tris after fill", deg_after_fill);
    }

    if scaled_config.refine.enable_post_refine && scaled_config.refine.max_iterations > 0 {
        for range in &face_ranges {
            let face = match reg.faces.get(range.face_key) {
                Some(f) => f,
                None => continue,
            };
            let (local_mesh, local_to_global) = extract_face_mesh_with_map(
                &global_vertices,
                &global_normals,
                &all_indices,
                range,
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
            merge_refined_face(
                &mut global_vertices,
                &mut global_normals,
                &mut all_indices,
                range,
                &refined,
                &local_to_global,
            );
        }
    }

    let deg_after_refine = cull_degenerate_tris(&mut all_indices, &global_vertices);
    if deg_after_refine > 0 {
        log::debug!("[BRep mesh] removed {} degenerate tris after refine", deg_after_refine);
    }

    for n in &mut global_normals {
        let len = n.length();
        if len > 1e-10 {
            *n = *n * (1.0 / len);
        } else {
            *n = Vec3::Z;
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
    optimize_mesh(&mut mesh, &scaled_config.optimize);
    ShellMeshOutput { mesh, report }
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

/// Tessellate a closed analytic surface face (VERTEX_LOOP, e.g. full sphere).
fn mesh_closed_surface(
    face: &super::topo::BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    all_indices: &mut Vec<i32>,
) {
    const SEGS: u32 = MESH_CLOSED_SURFACE_SEGS;
    let pr = face.surface.param_range();

    let insert_uv = |u: f32, v: f32,
                     global_vertices: &mut Vec<Vec3>,
                     global_normals: &mut Vec<Vec3>,
                     pos_to_idx: &mut HashMap<[u32; 3], usize>,
                     face: &super::topo::BRepFace|
     -> i32 {
        let pt = face.surface.d0_native(u, v);
        let mut n = face.surface.normal_native(u, v);
        if !face.same_sense {
            n = -n;
        }
        let hash = f32x3_quantized_bits([pt.x, pt.y, pt.z]);
        let idx = *pos_to_idx.entry(hash).or_insert_with(|| {
            let i = global_vertices.len();
            global_vertices.push(pt);
            global_normals.push(n);
            i
        });
        idx as i32
    };

    let emit_tri = |i0: i32, i1: i32, i2: i32,
                    global_vertices: &[Vec3],
                    global_normals: &mut [Vec3],
                    all_indices: &mut Vec<i32>| {
        if i0 == i1 || i1 == i2 || i2 == i0 {
            return;
        }
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
    };

    match &face.surface {
        SurfaceGeom::Sphere { .. }
        | SurfaceGeom::Torus { .. }
        | SurfaceGeom::Cylinder { .. }
        | SurfaceGeom::Cone { .. } => {
            for iu in 0..SEGS {
                for iv in 0..SEGS {
                    let u0 = pr.u_min + (pr.u_max - pr.u_min) * iu as f32 / SEGS as f32;
                    let u1 = pr.u_min + (pr.u_max - pr.u_min) * (iu + 1) as f32 / SEGS as f32;
                    let v0 = pr.v_min + (pr.v_max - pr.v_min) * iv as f32 / SEGS as f32;
                    let v1 = pr.v_min + (pr.v_max - pr.v_min) * (iv + 1) as f32 / SEGS as f32;
                    let a = insert_uv(u0, v0, global_vertices, global_normals, pos_to_idx, face);
                    let b = insert_uv(u1, v0, global_vertices, global_normals, pos_to_idx, face);
                    let c = insert_uv(u1, v1, global_vertices, global_normals, pos_to_idx, face);
                    let d = insert_uv(u0, v1, global_vertices, global_normals, pos_to_idx, face);
                    emit_tri(a, b, c, global_vertices, global_normals, all_indices);
                    emit_tri(a, c, d, global_vertices, global_normals, all_indices);
                }
            }
        }
        _ => {
            eprintln!(
                "[BRep mesh] closed surface {:?} not supported, skipping",
                std::mem::discriminant(&face.surface)
            );
        }
    }
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
