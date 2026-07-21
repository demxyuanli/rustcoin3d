//! Per-frame GPU resource lifecycle: uniform pool reset, mesh pool, mesh upload.

use std::sync::Arc;

use crate::gpu_resource::MeshId;
use crate::gpu_skinning::GpuSkinningPass;
use crate::render_action::DrawCall;

impl super::Renderer {
    /// Reset uniform pools and mesh-pool upload counters at frame start.
    pub(crate) fn begin_frame_gpu_resources(&mut self) {
        self.gpu.phong_pool.reset();
        self.gpu.shadow_pool.reset();
        self.gpu.flat_pool.reset();
        self.gpu.line_pool.reset();
        self.gpu.section_cap_pool.reset();
        self.gpu.outline_pool.reset();
        if let Some(ref mut pool) = self.gpu.assets.mesh_pool {
            pool.begin_frame();
        }
    }

    /// Upload / touch GPU meshes for the visible draw-call set.
    pub(crate) fn upload_visible_meshes(
        &mut self,
        visible: &[&DrawCall],
    ) -> Vec<Option<MeshId>> {
        let mesh_upload_start = std::time::Instant::now();
        let mut mesh_handles: Vec<Option<crate::gpu_resource::MeshId>> = Vec::with_capacity(visible.len());
        for dc in visible {
            let handle = if dc.vertices.is_empty() {
                if let Some(md) = dc.meshlet_data.as_ref() {
                    let ptr = Arc::as_ptr(md) as u64;
                    if let Some(mesh_id) = self.gpu.assets.mesh_touch(&ptr, self.frame.frame_counter) {
                        Some(mesh_id)
                    } else {
                        let verts: Vec<crate::vertex::Vertex> =
                            md.vertices
                                .iter()
                                .map(|mv| crate::vertex::Vertex {
                                    position: mv.position,
                                    normal: mv.normal,
                                    texcoord: mv.texcoord,
                                    tangent: mv.tangent,
                                })
                                .collect();
                        let mesh_id = self.gpu.gpu_meshes.upload_mesh(
                            &self.device,
                            &verts,
                            Some(&md.indices),
                            &dc.edge_positions,
                            &dc.wireframe_edge_positions,
                        );
                        self.gpu.assets.mesh_insert(ptr, mesh_id, self.frame.frame_counter, Some(&mut self.gpu.gpu_meshes));
                        Some(mesh_id)
                    }
                } else {
                    None
                }
            } else {
                let base_hash = dc.mesh_hash.unwrap_or_else(|| {
                    let p = Arc::as_ptr(&dc.vertices) as u64;
                    let i = dc.indices.as_ref().map_or(0u64, |a| Arc::as_ptr(a) as u64);
                    p ^ i
                });
                let hash = if let Some(ref skin) = dc.skinning {
                    base_hash ^ (Arc::as_ptr(skin) as u64)
                } else {
                    base_hash
                };
                if let Some(mesh_id) = self.gpu.assets.mesh_touch(&hash, self.frame.frame_counter) {
                    Some(mesh_id)
                } else {
                    if let Some(ref skin) = dc.skinning {
                        let verts = &*dc.vertices;
                        let skin_slice = skin.skin_data.as_slice();
                        if verts.len() != skin_slice.len() {
                            log::warn!(
                                "skinning data len {} != mesh verts {}; using rigid mesh",
                                skin_slice.len(),
                                verts.len()
                            );
                            let mesh_id = self.gpu.gpu_meshes.upload_mesh(
                                &self.device,
                                &dc.vertices,
                                dc.indices.as_ref().map(|a| a.as_slice()),
                                &dc.edge_positions,
                                &dc.wireframe_edge_positions,
                            );
                            self.gpu.assets.mesh_insert(hash, mesh_id, self.frame.frame_counter, Some(&mut self.gpu.gpu_meshes));
                            Some(mesh_id)
                        } else {
                            let pass = self
                                .gpu.gpu_skinning_pass
                                .get_or_insert_with(|| GpuSkinningPass::new(&self.device));
                            let positions: Vec<[f32; 3]> = verts.iter().map(|v| v.position).collect();
                            let normals: Vec<[f32; 3]> = verts.iter().map(|v| v.normal).collect();
                            let texcoords: Vec<[f32; 2]> = verts.iter().map(|v| v.texcoord).collect();
                            let tangents: Vec<[f32; 4]> = verts.iter().map(|v| v.tangent).collect();
                            let max_bones = skin.skeleton.joint_count().max(1) as u32;
                            let resources = pass.create_skinned_mesh(
                                &self.device,
                                &positions,
                                &normals,
                                &texcoords,
                                &tangents,
                                skin_slice,
                                max_bones,
                            );
                            let dst_vb = resources.dst_vertex_buffer.clone();
                            let mesh_id = self.gpu.gpu_meshes.insert_skinned_mesh(
                                &self.device,
                                dst_vb,
                                verts.len() as u32,
                                dc.indices.as_ref().map(|a| a.as_slice()),
                                &dc.edge_positions,
                                &dc.wireframe_edge_positions,
                            );
                            self.gpu.skinned_mesh_resources.insert(mesh_id, resources);
                            self.gpu.assets.mesh_insert(hash, mesh_id, self.frame.frame_counter, Some(&mut self.gpu.gpu_meshes));
                            Some(mesh_id)
                        }
                    } else {
                        let mesh_id = self.gpu.gpu_meshes.upload_mesh(
                            &self.device,
                            &dc.vertices,
                            dc.indices.as_ref().map(|a| a.as_slice()),
                            &dc.edge_positions,
                            &dc.wireframe_edge_positions,
                        );
                        self.gpu.assets.mesh_insert(hash, mesh_id, self.frame.frame_counter, Some(&mut self.gpu.gpu_meshes));
                        Some(mesh_id)
                    }
                }
            };
            mesh_handles.push(handle);
        }
        self.cpu_span.record("mesh_upload", mesh_upload_start.elapsed().as_secs_f64() * 1000.0);

        mesh_handles
    }
}
