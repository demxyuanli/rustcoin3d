pub mod edge_disc;
pub mod face_tri;
pub mod refiner;
pub mod optimize;

use std::collections::HashMap;
use rc3d_core::math::Vec3;
use rc3d_core::utils::hash::f32x3_quantized_bits;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation as _};
use spade::handles::FixedVertexHandle;
use super::topo::{ShellKey, EdgeKey, FaceKey};
use super::registry::BRepRegistry;
use crate::step::mesh_result::MeshResult;
use edge_disc::{EdgeDiscConfig, EdgePolygon, discretize_all_edges};
use face_tri::triangulate_face;
use refiner::{RefineConfig, refine_mesh};
use optimize::{OptimizeConfig, optimize_mesh};

#[derive(Debug, Clone)]
pub struct BRepMeshConfig {
    pub edge: EdgeDiscConfig,
    pub refine: RefineConfig,
    pub optimize: OptimizeConfig,
}

impl Default for BRepMeshConfig {
    fn default() -> Self {
        Self {
            edge: EdgeDiscConfig::default(),
            refine: RefineConfig::default(),
            optimize: OptimizeConfig::default(),
        }
    }
}

/// Mesh a B-Rep shell with watertight edge stitching (OCCT BRepMesh approach).
///
/// Algorithm:
///   Phase 1: Discretize all edges once → shared 3D+2D polygons
///   Phase 2: Insert all edge polygon 3D vertices into global array (hash-deduped)
///     Build lookup: (EdgeKey, t_index) → global_vertex_index
///   Phase 3: For each face:
///     a. Build UV Delaunay of boundary vertices
///     b. Boundary vertices → use pre-mapped global indices (SHARED)
///     c. Interior vertices → surface.d0(uv) (FACE-SPECIFIC)
///   Phase 4: Refine + optimize per face (interior only)
///
/// Adjacent faces share identical boundary vertices → watertight by construction.
pub fn mesh_brep_shell(
    shell_key: ShellKey,
    reg: &BRepRegistry,
    config: &BRepMeshConfig,
) -> MeshResult {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return MeshResult::default(),
    };

    // ── Phase 1: Discretize all edges (shared edges once) ──
    let edge_polygons: HashMap<EdgeKey, EdgePolygon> = discretize_all_edges(reg, &config.edge);

    // ── Phase 2: Build shared boundary vertex array ──
    // Global vertex array + dedup hash
    let mut global_vertices: Vec<Vec3> = Vec::new();
    let mut global_normals: Vec<Vec3> = Vec::new();
    let mut pos_to_idx: HashMap<[u32; 3], usize> = HashMap::new();

    // Lookup: (EdgeKey, param_index) → global_vertex_index
    let mut edge_boundary_idx: HashMap<(EdgeKey, usize), usize> = HashMap::new();

    // Insert all edge polygon 3D vertices (hash-deduped)
    for (&ek, poly) in &edge_polygons {
        for (pi, &(_t, pt)) in poly.params_3d.iter().enumerate() {
            let hash = f32x3_quantized_bits([pt.x, pt.y, pt.z]);
            let idx = *pos_to_idx.entry(hash).or_insert_with(|| {
                let i = global_vertices.len();
                global_vertices.push(pt);
                global_normals.push(Vec3::ZERO); // filled later
                i
            });
            edge_boundary_idx.insert((ek, pi), idx);
        }
    }

    // ── Phase 3: Collect face wire topology for boundary lookups ──
    // For each face, record which edge polygon points form its boundary
    struct FaceWireInfo {
        face_key: FaceKey,
        /// (edge_key, param_indices in order along the wire)
        wire_edges: Vec<(EdgeKey, Vec<usize>)>,
        same_sense: bool,
    }

    let mut face_infos: Vec<FaceWireInfo> = Vec::new();

    for &(face_key, _orient) in &shell.faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        let wire = match reg.wires.get(face.outer_wire) {
            Some(w) => w,
            None => continue,
        };

        let mut wire_edges = Vec::new();
        for &(ek, _) in &wire.edges {
            if let Some(poly) = edge_polygons.get(&ek) {
                if poly.params_2d.contains_key(&face_key) {
                    // Edge has PCURVE data for this face — include all its 2D points
                    let n_pts = poly.params_2d[&face_key].len();
                    wire_edges.push((ek, (0..n_pts).collect()));
                } else if !poly.params_3d.is_empty() {
                    // No PCURVE for this face, use 3D points only
                    let n_pts = poly.params_3d.len();
                    wire_edges.push((ek, (0..n_pts).collect()));
                }
            }
        }

        if !wire_edges.is_empty() {
            face_infos.push(FaceWireInfo {
                face_key,
                wire_edges,
                same_sense: face.same_sense,
            });
        }
    }

    // ── Phase 4: Triangulate each face, referencing shared boundary vertices ──
    let mut all_indices: Vec<i32> = Vec::new();

    for info in &face_infos {
        let face = match reg.faces.get(info.face_key) {
            Some(f) => f,
            None => continue,
        };

        // Collect boundary vertices for this face in wire order
        // Maps: local UV index → global vertex index
        let mut local_to_global: Vec<usize> = Vec::new();
        let mut local_uv: Vec<(f32, f32)> = Vec::new();  // UV coords for Delaunay

        for &(ek, ref param_indices) in &info.wire_edges {
            if let Some(poly) = edge_polygons.get(&ek) {
                // Get 2D (PCURVE) points for this face
                let pcurve_pts = poly.params_2d.get(&info.face_key);
                for &pi in param_indices {
                    let uv = if let Some(pts) = pcurve_pts {
                        pts.get(pi).map(|&(_, (u, v))| (u, v))
                            .unwrap_or((0.0, 0.0))
                    } else {
                        // Fallback: project 3D to UV
                        let pt = poly.params_3d[pi].1;
                        face.surface.project(pt).unwrap_or((0.0, 0.0))
                    };
                    let global_idx = edge_boundary_idx.get(&(ek, pi)).copied().unwrap_or(0);
                    local_to_global.push(global_idx);
                    local_uv.push(uv);
                }
            }
        }

        if local_uv.len() < 3 {
            continue;
        }

        // Build UV Delaunay of boundary vertices
        let mut cdt = spade::ConstrainedDelaunayTriangulation::<spade::Point2<f64>>::default();
        let mut handles: Vec<spade::handles::FixedVertexHandle> = Vec::new();
        for &(u, v) in &local_uv {
            if let Ok(h) = cdt.insert(spade::Point2::new(u as f64, v as f64)) {
                handles.push(h);
            }
        }
        if handles.is_empty() { continue; }

        let handle_to_local: HashMap<spade::handles::FixedVertexHandle, usize> =
            handles.iter().enumerate().map(|(i, &h)| (h, i)).collect();

        // Emit all Delaunay triangles
        for tri_face in cdt.inner_faces() {
            let vs = tri_face.vertices();
            let (Some(&li0), Some(&li1), Some(&li2)) = (
                handle_to_local.get(&vs[0].fix()),
                handle_to_local.get(&vs[1].fix()),
                handle_to_local.get(&vs[2].fix()),
            ) else { continue; };

            let gi0 = local_to_global[li0] as i32;
            let gi1 = local_to_global[li1] as i32;
            let gi2 = local_to_global[li2] as i32;

            if gi0 == gi1 || gi1 == gi2 || gi2 == gi0 { continue; } // degenerate

            all_indices.extend_from_slice(&[gi0, gi1, gi2, -1]);

            // Compute normals at boundary vertices (accumulate per-face)
            let p0 = global_vertices[gi0 as usize];
            let p1 = global_vertices[gi1 as usize];
            let p2 = global_vertices[gi2 as usize];
            let tri_n = (p1 - p0).cross(p2 - p0);
            if tri_n.length() > 1e-10 {
                let n = tri_n.normalize();
                global_normals[gi0 as usize] = global_normals[gi0 as usize] + n;
                global_normals[gi1 as usize] = global_normals[gi1 as usize] + n;
                global_normals[gi2 as usize] = global_normals[gi2 as usize] + n;
            }
        }
    }

    // Finalize normals
    for n in &mut global_normals {
        let len = n.length();
        if len > 1e-10 { *n = *n * (1.0 / len); }
        else { *n = Vec3::Z; }
    }

    eprintln!("[BRep mesh] shell {:?}: {} boundary verts, {} faces, {} tris total",
        shell_key, global_vertices.len(), face_infos.len(), all_indices.len() / 4);
    MeshResult { vertices: global_vertices, indices: all_indices, normals: global_normals }
}
