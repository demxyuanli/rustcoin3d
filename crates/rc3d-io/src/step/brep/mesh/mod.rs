pub mod edge_disc;
pub mod face_tri;
pub mod refiner;
pub mod optimize;

use std::collections::HashMap;
use super::topo::{ShellKey, EdgeKey};
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

/// Mesh a single B-Rep shell: discretize → triangulate → refine → optimize → merge.
pub fn mesh_brep_shell(
    shell_key: ShellKey,
    reg: &BRepRegistry,
    config: &BRepMeshConfig,
) -> MeshResult {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return MeshResult::default(),
    };

    // Phase 1: Discretize all edges (shared edges once)
    let edge_polygons: HashMap<EdgeKey, EdgePolygon> = discretize_all_edges(reg, &config.edge);

    // Phase 2-4: Triangulate each face, then refine, then optimize
    let mut all_vertices = Vec::new();
    let mut all_normals = Vec::new();
    let mut all_indices = Vec::new();
    let mut vertex_offset = 0i32;

    for &(face_key, _orient) in &shell.faces {
        // Phase 2: CDT triangulation
        if let Some(face_mesh) = triangulate_face(face_key, reg, &edge_polygons) {
            let face = match reg.faces.get(face_key) {
                Some(f) => f,
                None => continue,
            };

            // Phase 3: Refinement
            let face_mesh = refine_mesh(
                &face_mesh, &face.surface, face.same_sense, &config.refine,
            );

            // Phase 4: Optimization
            let mut face_mesh = face_mesh;
            optimize_mesh(&mut face_mesh, &config.optimize);

            // Diagnostic: report face mesh stats
            let face = reg.faces.get(face_key).unwrap();
            let edge_count: usize = reg.wires.get(face.outer_wire)
                .map(|w| w.edges.len()).unwrap_or(0);
            let tri_count = face_mesh.indices.len() / 4;
            let vs = &face_mesh.vertices;
            let (min_x, max_x, min_y, max_y, min_z, max_z) = if vs.is_empty() {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            } else {
                let (mut mnx, mut mxx, mut mny, mut mxy, mut mnz, mut mxz) =
                    (f32::MAX, f32::MIN, f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                for v in vs { mnx=mnx.min(v.x); mxx=mxx.max(v.x); mny=mny.min(v.y); mxy=mxy.max(v.y); mnz=mnz.min(v.z); mxz=mxz.max(v.z); }
                (mnx, mxx, mny, mxy, mnz, mxz)
            };
            eprintln!("[BRep mesh] face {:?}: {} edges → {} tris, {} verts, bbox=[{:.2}..{:.2}, {:.2}..{:.2}, {:.2}..{:.2}]",
                face_key, edge_count, tri_count, vs.len(), min_x, max_x, min_y, max_y, min_z, max_z);

            // Merge into global mesh (with vertex offset)
            all_vertices.extend(face_mesh.vertices);
            all_normals.extend(face_mesh.normals);
            for chunk in face_mesh.indices.chunks(4) {
                if chunk.len() == 4 && chunk[3] == -1 {
                    all_indices.push(chunk[0] + vertex_offset);
                    all_indices.push(chunk[1] + vertex_offset);
                    all_indices.push(chunk[2] + vertex_offset);
                    all_indices.push(-1);
                }
            }
            vertex_offset = all_vertices.len() as i32;
        }
    }

    MeshResult { vertices: all_vertices, indices: all_indices, normals: all_normals }
}
