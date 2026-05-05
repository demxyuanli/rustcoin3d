use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::gpu_resource::{EdgeLineKind, GpuMesh, GpuUniformPool};
use crate::render_action::DrawCall;
use crate::vertex::LineVertex;
use super::renderer_types::{BatchAnalysis, FrameDiagnostics, MemoryBudget, NodeTypeDrawStat};
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

    pub(crate) fn write_gpu_timestamp(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ref qs) = self.gpu.gpu_query_set {
            if self.gpu.gpu_query_slots < 16 {
                encoder.write_timestamp(qs, self.gpu.gpu_query_slots);
                self.gpu.gpu_query_slots += 1;
            }
        }
    }

    pub(crate) fn resolve_gpu_timestamps(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(ref qs), Some(ref qb)) = (&self.gpu.gpu_query_set, &self.gpu.gpu_query_buffer) {
            let written = self.gpu.gpu_query_slots.min(16);
            if written > 0 {
                encoder.resolve_query_set(qs, 0..written, qb, 0);
            }
        }
    }

    pub(crate) fn read_gpu_timestamps(&mut self) -> Option<[f64; 4]> {
        let qb = self.gpu.gpu_query_buffer.as_ref()?;
        let written = (self.gpu.gpu_query_slots.min(16)) as usize;
        if written < 8 {
            self.gpu.gpu_query_slots = 0;
            return None;
        }
        let size = (written as u64) * 8;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU query staging readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GPU query copy encoder"),
            });
        encoder.copy_buffer_to_buffer(qb, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.is_ok());
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        let mapped = rx.recv().ok().unwrap_or(false);
        if !mapped {
            self.gpu.gpu_query_slots = 0;
            return None;
        }
        let data = slice.get_mapped_range();
        let ticks: Vec<u64> = bytemuck::cast_slice::<u8, u64>(&data).to_vec();
        drop(data);
        staging.unmap();
        self.gpu.gpu_query_slots = 0;
        if ticks.len() < 8 {
            return None;
        }
        let to_us = |a: u64, b: u64| -> f64 {
            let ns = (b.saturating_sub(a) as f64) * self.gpu.gpu_query_period as f64;
            ns / 1000.0
        };
        let shadow = to_us(ticks[0], ticks[1]);
        let solid = to_us(ticks[2], ticks[3]);
        let post = to_us(ticks[4], ticks[5]);
        let total = to_us(ticks[6], ticks[7]);
        Some([shadow, solid, post, total])
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
        node_type_stats.sort_by(|a, b| b.draw_calls.cmp(&a.draw_calls));
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
        missed_vec.sort_by(|a, b| b.1.cmp(&a.1));

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
            (pool.stride() * pool.capacity() as u64) as u64
        }
        pool_bytes(&self.gpu.phong_pool)
            + pool_bytes(&self.gpu.shadow_pool)
            + pool_bytes(&self.gpu.flat_pool)
            + pool_bytes(&self.gpu.outline_pool)
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
