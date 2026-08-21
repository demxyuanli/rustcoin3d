use wgpu::util::DeviceExt;

use crate::gpu_resource::{EdgeLineKind, GpuMesh, GpuUniformPool};
use crate::render_action::DrawCall;
use crate::vertex::LineVertex;
use super::types::{BatchAnalysis, FrameDiagnostics, MemoryBudget, NodeTypeDrawStat};
use super::Renderer;

impl Renderer {
    pub(crate) fn get_mesh(&self, mesh_id: crate::gpu_resource::MeshId) -> Option<&GpuMesh> {
        self.gpu.gpu_meshes.get(mesh_id)
    }

    pub(crate) fn draw_mesh_batched(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        mesh_id: crate::gpu_resource::MeshId,
        last_bound: &mut Option<crate::gpu_resource::MeshId>,
    ) {
        let Some(mesh) = self.get_mesh(mesh_id) else { return };
        if last_bound.map_or(true, |id| id != mesh_id) {
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            if let Some(index_buffer) = &mesh.index_buffer {
                pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            }
            *last_bound = Some(mesh_id);
        }
        if mesh.index_buffer.is_some() {
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        } else {
            pass.draw(0..mesh.vertex_count, 0..1);
        }
    }

    /// Instanced draw for solid pass batching. Each instance reads distinct
    /// model/mvp/data from the SSBO via `@builtin(instance_index)`.
    /// `first_instance` is the global offset into the instance SSBO so each
    /// subgroup writes to a non-overlapping region.
    pub(crate) fn draw_mesh_instanced(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        mesh_id: crate::gpu_resource::MeshId,
        first_instance: u32,
        instance_count: u32,
        last_bound: &mut Option<crate::gpu_resource::MeshId>,
    ) {
        let Some(mesh) = self.get_mesh(mesh_id) else { return };
        if last_bound.map_or(true, |id| id != mesh_id) {
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            if let Some(index_buffer) = &mesh.index_buffer {
                pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            }
            *last_bound = Some(mesh_id);
        }
        if mesh.index_buffer.is_some() {
            pass.draw_indexed(0..mesh.index_count, 0, first_instance..first_instance + instance_count);
        } else {
            pass.draw(0..mesh.vertex_count, first_instance..first_instance + instance_count);
        }
    }

    /// Multi-draw indirect for solid pass batching. Issues N draws in a single
    /// GPU command, sharing vertex/index buffer bindings.
    /// All draws in the batch must use the same mesh.
    pub(crate) fn draw_mesh_multi_indirect(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        mesh_id: crate::gpu_resource::MeshId,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        draw_count: u32,
        last_bound: &mut Option<crate::gpu_resource::MeshId>,
    ) {
        let Some(mesh) = self.get_mesh(mesh_id) else { return };
        if last_bound.map_or(true, |id| id != mesh_id) {
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            if let Some(index_buffer) = &mesh.index_buffer {
                pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            }
            *last_bound = Some(mesh_id);
        }
        if mesh.index_buffer.is_some() {
            pass.multi_draw_indexed_indirect(indirect_buffer, indirect_offset, draw_count);
        } else {
            // Fall back to per-draw for non-indexed meshes (rare for triangle geometry).
            pass.draw(0..mesh.vertex_count, 0..draw_count);
        }
    }

    pub(crate) fn draw_edges_batched(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        mesh_id: crate::gpu_resource::MeshId,
        last_bound: &mut Option<crate::gpu_resource::MeshId>,
        kind: EdgeLineKind,
    ) -> bool {
        let Some(mesh) = self.get_mesh(mesh_id) else { return false };
        let (buf, count) = match kind {
            EdgeLineKind::Feature => (&mesh.edge_vertex_buffer, mesh.edge_vertex_count),
            EdgeLineKind::WireframeFull => (
                &mesh.wireframe_edge_vertex_buffer,
                mesh.wireframe_edge_vertex_count,
            ),
        };
        let Some(edge_buffer) = buf else { return false };
        if count == 0 {
            return false;
        }
        if last_bound.map_or(true, |id| id != mesh_id) {
            pass.set_vertex_buffer(0, edge_buffer.slice(..));
            *last_bound = Some(mesh_id);
        }
        pass.draw(0..count, 0..1);
        true
    }

    /// Draw expanded edge geometry for anti-aliased line rendering.
    pub(crate) fn draw_expanded_edges_batched(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        mesh_id: crate::gpu_resource::MeshId,
        last_bound: &mut Option<crate::gpu_resource::MeshId>,
    ) -> bool {
        let Some(mesh) = self.get_mesh(mesh_id) else { return false };
        let Some(ref buf) = mesh.edge_expanded_buffer else { return false };
        if mesh.edge_expanded_count == 0 {
            return false;
        }
        if last_bound.map_or(true, |id| id != mesh_id) {
            pass.set_vertex_buffer(0, buf.slice(..));
            *last_bound = Some(mesh_id);
        }
        pass.draw(0..mesh.edge_expanded_count, 0..1);
        true
    }

    pub(crate) fn bind_and_draw_edges(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        dc: &DrawCall,
        handle: Option<crate::gpu_resource::MeshId>,
        kind: EdgeLineKind,
    ) {
        if let Some(mesh_id) = handle {
            if let Some(mesh) = self.get_mesh(mesh_id) {
                let (buf, count) = match kind {
                    EdgeLineKind::Feature => (&mesh.edge_vertex_buffer, mesh.edge_vertex_count),
                    EdgeLineKind::WireframeFull => (
                        &mesh.wireframe_edge_vertex_buffer,
                        mesh.wireframe_edge_vertex_count,
                    ),
                };
                if let Some(edge_buffer) = buf {
                    if count > 0 {
                        pass.set_vertex_buffer(0, edge_buffer.slice(..));
                        pass.draw(0..count, 0..1);
                        return;
                    }
                }
            }
        }
        let cpu_edges = match kind {
            EdgeLineKind::Feature => &dc.edge_positions,
            EdgeLineKind::WireframeFull => &dc.wireframe_edge_positions,
        };
        if !cpu_edges.is_empty() {
            let line_verts: Vec<LineVertex> = cpu_edges
                .iter()
                .map(|&p| LineVertex { position: p })
                .collect();
            let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Edge VB"),
                contents: bytemuck::cast_slice(&line_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.draw(0..line_verts.len() as u32, 0..1);
        }
    }

    pub(crate) fn prune_mesh_cache(&mut self) {
        self.gpu.assets.prune_stale_meshes(
            self.frame.frame_counter,
            &mut self.gpu.gpu_meshes,
            Some(&mut self.gpu.skinned_mesh_resources),
        );
    }

    pub(crate) fn build_frame_diagnostics(
        &self,
        visible: &[&DrawCall],
        mesh_handles: &[Option<crate::gpu_resource::MeshId>],
        solid_order: &[usize],
        transparent_order: &[usize],
        gpu_pass: Option<[f64; 4]>,
    ) -> FrameDiagnostics {
        let mut per_type: std::collections::HashMap<String, (usize, u64)> =
            std::collections::HashMap::new();
        for dc in visible {
            let tris = if let Some(md) = dc.meshlet_data.as_ref() {
                md.total_triangles as u64
            } else if let Some(indices) = dc.indices.as_ref() {
                (indices.len() / 3) as u64
            } else {
                (dc.vertices.len() / 3) as u64
            };
            let entry = per_type
                .entry(dc.node_type_label.as_ref().to_string())
                .or_insert((0, 0));
            entry.0 += 1;
            entry.1 += tris;
        }
        let mut node_type_stats: Vec<NodeTypeDrawStat> = per_type
            .into_iter()
            .map(|(k, (draws, tris))| NodeTypeDrawStat {
                node_type: k,
                draw_calls: draws,
                triangles: tris,
            })
            .collect();
        rc3d_core::utils::sort::sort_by_key_count_desc(&mut node_type_stats, |s| s.draw_calls);
        node_type_stats.truncate(12);

        let mut estimated_batch_count = 0usize;
        let mut last_mesh: Option<crate::gpu_resource::MeshId> = None;
        for &idx in solid_order {
            if let Some(mesh) = mesh_handles[idx] {
                if last_mesh != Some(mesh) {
                    estimated_batch_count += 1;
                    last_mesh = Some(mesh);
                }
            }
        }
        let mut missed_reasons: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let opaque_candidates = solid_order.len();
        let merge_loss = opaque_candidates.saturating_sub(estimated_batch_count);
        if merge_loss > 0 {
            missed_reasons.insert("state_or_material_change".to_string(), merge_loss);
        }
        if !transparent_order.is_empty() {
            missed_reasons.insert(
                "transparent_order_constraint".to_string(),
                transparent_order.len(),
            );
        }
        let meshlet_count = visible
            .iter()
            .filter(|dc| dc.meshlet_data.is_some())
            .count();
        if meshlet_count > 0 {
            missed_reasons.insert("meshlet_path_split".to_string(), meshlet_count);
        }

        let gpu_mesh_bytes = self.estimate_gpu_mesh_bytes(visible);
        let gpu_uniform_bytes = self.estimate_gpu_uniform_pool_bytes();
        let cpu_draw_bytes = visible
            .iter()
            .map(|dc| {
                (dc.vertices.len() * std::mem::size_of::<crate::vertex::Vertex>())
                    + dc.indices
                        .as_ref()
                        .map(|i| i.len() * std::mem::size_of::<u32>())
                        .unwrap_or(0)
                    + (dc.edge_positions.len() + dc.wireframe_edge_positions.len())
                        * std::mem::size_of::<[f32; 3]>()
            })
            .sum::<usize>() as u64;

        let mut gpu_pass_timings_us = Vec::new();
        if let Some([shadow, solid, post, total]) = gpu_pass {
            gpu_pass_timings_us.push(("shadow".to_string(), shadow));
            gpu_pass_timings_us.push(("solid".to_string(), solid));
            gpu_pass_timings_us.push(("post".to_string(), post));
            gpu_pass_timings_us.push(("total".to_string(), total));
        }

        let mut missed_vec: Vec<(String, usize)> = missed_reasons.into_iter().collect();
        rc3d_core::utils::sort::sort_by_count_desc(&mut missed_vec);

        FrameDiagnostics {
            frame_index: self.frame.frame_counter,
            gpu_pass_timings_us,
            cpu_memory: vec![
                MemoryBudget {
                    label: "draw_call_payload".to_string(),
                    used_bytes: cpu_draw_bytes,
                    budget_bytes: 512 * 1024 * 1024,
                },
                MemoryBudget {
                    label: "material_library".to_string(),
                    used_bytes: (self.gpu.materials.len() as u64) * 512,
                    budget_bytes: 64 * 1024 * 1024,
                },
            ],
            gpu_memory: vec![
                MemoryBudget {
                    label: "mesh_buffers_estimate".to_string(),
                    used_bytes: gpu_mesh_bytes,
                    budget_bytes: 2 * 1024 * 1024 * 1024,
                },
                MemoryBudget {
                    label: "uniform_pools_estimate".to_string(),
                    used_bytes: gpu_uniform_bytes,
                    budget_bytes: 256 * 1024 * 1024,
                },
            ],
            node_type_stats,
            batch_analysis: BatchAnalysis {
                candidate_draw_calls: visible.len(),
                submitted_draw_calls: opaque_candidates + transparent_order.len(),
                estimated_batch_count,
                missed_reasons: missed_vec,
            },
            timestamp_supported: self.gpu.timing_supported,
        }
    }

    fn estimate_gpu_uniform_pool_bytes(&self) -> u64 {
        fn pool_bytes(pool: &GpuUniformPool) -> u64 {
            pool.stride() * pool.capacity() as u64
        }
        pool_bytes(&self.gpu.phong_pool)
            + pool_bytes(&self.gpu.shadow_pool)
            + pool_bytes(&self.gpu.flat_pool)
            + pool_bytes(&self.gpu.section_cap_pool)
    }

    fn estimate_gpu_mesh_bytes(&self, visible: &[&DrawCall]) -> u64 {
        visible
            .iter()
            .map(|dc| {
                (dc.vertices.len() * std::mem::size_of::<crate::vertex::Vertex>()
                    + dc.indices
                        .as_ref()
                        .map(|v| v.len() * std::mem::size_of::<u32>())
                        .unwrap_or(0)
                    + (dc.edge_positions.len() + dc.wireframe_edge_positions.len())
                        * std::mem::size_of::<[f32; 3]>()) as u64
            })
            .sum()
    }

    pub(crate) fn ensure_ldr_shade_target(&mut self) {
        let w = self.config.width.max(1);
        let h = self.config.height.max(1);
        let fmt = self.config.format;
        let need_new = self.gpu.ldr_shade_tex.as_ref().map_or(true, |t| {
            let s = t.size();
            s.width != w || s.height != h || t.format() != fmt
        });
        if need_new {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("LDR shade (FXAA source)"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.gpu.ldr_shade_tex = Some(texture);
            self.gpu.ldr_shade_view = Some(view);
        }
    }
}
