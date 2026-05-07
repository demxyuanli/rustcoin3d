//! Advanced rendering effects: Decal, Volume, PointCloud.
//! Each effect is lazy-initialized when first needed.

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use wgpu::util::DeviceExt;
use rc3d_scene::{NodeData, SceneGraph};

#[derive(Clone, Debug, Default)]
pub struct EffectCommands {
    pub decals: Vec<DecalDrawCommand>,
    pub volumes: Vec<VolumeDrawCommand>,
    pub point_clouds: Vec<PointCloudDrawCommand>,
}

impl EffectCommands {
    pub fn is_empty(&self) -> bool {
        self.decals.is_empty() && self.volumes.is_empty() && self.point_clouds.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct DecalDrawCommand {
    pub model_matrix: Mat4,
    pub position: Vec3,
    pub direction: Vec3,
    pub size: [f32; 2],
    pub texture_path: String,
    pub color: [f32; 4],
    pub opacity: f32,
    pub is_overlay: bool,
}

#[derive(Clone, Debug)]
pub struct VolumeDrawCommand {
    pub model_matrix: Mat4,
    pub dimensions: [u32; 3],
    pub texture_path: String,
    pub density_scale: f32,
    pub color_map: [[f32; 4]; 4],
    pub is_overlay: bool,
}

#[derive(Clone, Debug)]
pub struct PointCloudDrawCommand {
    pub model_matrix: Mat4,
    pub file_path: String,
    pub max_visible_points: u32,
    pub point_size: f32,
    pub color: [f32; 4],
    pub is_overlay: bool,
}

pub fn collect_effect_nodes(graph: &SceneGraph) -> EffectCommands {
    let mut commands = EffectCommands::default();
    for &root in graph.roots() {
        collect_effect_recursive(graph, root, Mat4::IDENTITY, false, &mut commands);
    }
    commands
}

fn collect_effect_recursive(
    graph: &SceneGraph,
    node: NodeId,
    model_matrix: Mat4,
    inside_annotation: bool,
    commands: &mut EffectCommands,
) {
    let Some(entry) = graph.get(node) else { return };
    match &entry.data {
        NodeData::Decal(decal) => {
            commands.decals.push(DecalDrawCommand {
                model_matrix,
                position: decal.position,
                direction: decal.direction,
                size: decal.size,
                texture_path: decal.texture_path.clone(),
                color: decal.color,
                opacity: decal.opacity,
                is_overlay: inside_annotation,
            });
            for &child in &entry.children {
                collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
            }
        }
        NodeData::Volume(volume) => {
            commands.volumes.push(VolumeDrawCommand {
                model_matrix,
                dimensions: volume.dimensions,
                texture_path: volume.texture_path.clone(),
                density_scale: volume.density_scale,
                color_map: volume.color_map,
                is_overlay: inside_annotation,
            });
            for &child in &entry.children {
                collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
            }
        }
        NodeData::PointCloud(point_cloud) => {
            commands.point_clouds.push(PointCloudDrawCommand {
                model_matrix,
                file_path: point_cloud.file_path.clone(),
                max_visible_points: point_cloud.max_visible_points,
                point_size: point_cloud.point_size,
                color: point_cloud.color,
                is_overlay: inside_annotation,
            });
            for &child in &entry.children {
                collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
            }
        }
        NodeData::Transform(transform) => {
            let next_model = model_matrix * transform.to_matrix();
            for &child in &entry.children {
                collect_effect_recursive(graph, child, next_model, inside_annotation, commands);
            }
        }
        NodeData::ResetTransform(_) => {
            for &child in &entry.children {
                collect_effect_recursive(graph, child, Mat4::IDENTITY, inside_annotation, commands);
            }
        }
        NodeData::Annotation(_) => {
            for &child in &entry.children {
                collect_effect_recursive(graph, child, model_matrix, true, commands);
            }
        }
        NodeData::Switch(sw) => match sw.which_child {
            -2 => {}
            -1 => {
                for &child in &sw.children {
                    collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
                }
            }
            idx if idx >= 0 => {
                if let Some(&child) = sw.children.get(idx as usize) {
                    collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
                }
            }
            _ => {}
        },
        NodeData::MultipleCopy(mc) => {
            for &copy_matrix in &mc.copies {
                let next_model = model_matrix * copy_matrix;
                for &child in &mc.children {
                    collect_effect_recursive(graph, child, next_model, inside_annotation, commands);
                }
            }
        }
        NodeData::Lod(lod) => {
            let level = lod.current_level.min(lod.levels.len().saturating_sub(1));
            if let Some(level_data) = lod.levels.get(level) {
                for &child in &level_data.children {
                    collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
                }
            }
        }
        _ => {
            for &child in &entry.children {
                collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
            }
        }
    }
}

/// Decal projection pipeline resources.
pub struct DecalPass {
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
    pub params_buf: wgpu::Buffer,
}

impl DecalPass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Decal Project"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/decal_project.wgsl").into(),
            ),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Decal BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: false } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Decal PPL"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Decal"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32FloatStencil8, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::LessEqual, stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
            multiview: None, cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Decal Sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToBorder, address_mode_v: wgpu::AddressMode::ClampToBorder,
            border_color: Some(wgpu::SamplerBorderColor::TransparentBlack), ..Default::default()
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Decal Params"), size: 256, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        Self { bgl, pipeline, sampler, params_buf }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        shade_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        commands: &[DecalDrawCommand],
        _viewport_w: u32,
        _viewport_h: u32,
    ) {
        if commands.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Decal Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: shade_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);

        let placeholder_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("decal placeholder"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let placeholder_view = placeholder_tex.create_view(&wgpu::TextureViewDescriptor::default());

        for cmd in commands {
            if cmd.texture_path.is_empty() {
                continue;
            }
            // Pack uniform data: model(16) + position(4) + direction(4) + size(2) + color(4) + opacity(1) + pad(1)
            let mut uniform_data = Vec::<f32>::with_capacity(32);
            let m = cmd.model_matrix.to_cols_array_2d();
            for row in &m { uniform_data.extend_from_slice(row); }
            uniform_data.extend_from_slice(&[cmd.position.x, cmd.position.y, cmd.position.z, 1.0]);
            uniform_data.extend_from_slice(&[cmd.direction.x, cmd.direction.y, cmd.direction.z, 0.0]);
            uniform_data.extend_from_slice(&[cmd.size[0], cmd.size[1]]);
            uniform_data.extend_from_slice(&[cmd.color[0], cmd.color[1], cmd.color[2], cmd.color[3]]);
            uniform_data.extend_from_slice(&[cmd.opacity, 0.0]);
            queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&uniform_data));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Decal BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&placeholder_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding { buffer: &self.params_buf, offset: 0, size: None }
                    ) },
                ],
            });
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..6, 0..1);
        }
    }
}

/// Volume ray-march pipeline resources.
pub struct VolumePass {
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
    pub params_buf: wgpu::Buffer,
}

impl VolumePass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volume Raymarch"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/volume_raymarch.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Volume BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: false } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volume PPL"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Volume Raymarch"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32FloatStencil8, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::LessEqual, stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() }, multiview: None, cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Volume Sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToBorder, address_mode_v: wgpu::AddressMode::ClampToBorder,
            address_mode_w: wgpu::AddressMode::ClampToBorder, border_color: Some(wgpu::SamplerBorderColor::TransparentBlack), ..Default::default()
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Volume Params"), size: 64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        Self { bgl, pipeline, sampler, params_buf }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        shade_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        commands: &[VolumeDrawCommand],
        _viewport_w: u32,
        _viewport_h: u32,
    ) {
        if commands.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Volume Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: shade_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);

        let placeholder_vol = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume placeholder"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let placeholder_view = placeholder_vol.create_view(&wgpu::TextureViewDescriptor::default());

        for cmd in commands {
            if cmd.texture_path.is_empty() {
                continue;
            }
            let mut uniform_data = Vec::<f32>::with_capacity(80);
            let m = cmd.model_matrix.to_cols_array_2d();
            for row in &m { uniform_data.extend_from_slice(row); }
            uniform_data.extend_from_slice(&[cmd.dimensions[0] as f32, cmd.dimensions[1] as f32, cmd.dimensions[2] as f32, 0.0]);
            uniform_data.push(cmd.density_scale);
            for cm in &cmd.color_map { uniform_data.extend_from_slice(cm); }
            for _ in 0..3 { uniform_data.push(0.0); }
            queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&uniform_data));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Volume BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&placeholder_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding { buffer: &self.params_buf, offset: 0, size: None }
                    ) },
                ],
            });
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..6, 0..1);
        }
    }
}

/// Point cloud rendering pipeline resources.
pub struct PointCloudPass {
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
}

impl PointCloudPass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Point Cloud"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/point_cloud.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PointCloud BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PointCloud PPL"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PointCloud"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::PointList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32FloatStencil8, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::LessEqual, stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
            multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
            multiview: None, cache: None,
        });
        Self { bgl, pipeline }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        shade_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        commands: &[PointCloudDrawCommand],
        projection: glam::Mat4,
        _inv_projection: glam::Mat4,
        _viewport_w: u32,
        _viewport_h: u32,
    ) {
        if commands.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("PointCloud Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: shade_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);

        for cmd in commands {
            if cmd.file_path.is_empty() || cmd.max_visible_points == 0 {
                continue;
            }
            let mvp = projection * cmd.model_matrix;
            let mut uniform_data = Vec::<f32>::with_capacity(40);
            let m = mvp.to_cols_array_2d();
            for row in &m { uniform_data.extend_from_slice(row); }
            let model_m = cmd.model_matrix.to_cols_array_2d();
            for row in &model_m { uniform_data.extend_from_slice(row); }
            uniform_data.extend_from_slice(&[cmd.point_size, 0.0, 0.0, 0.0]);
            uniform_data.extend_from_slice(&[cmd.color[0], cmd.color[1], cmd.color[2], cmd.color[3]]);
            let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PC uniforms"),
                contents: bytemuck::cast_slice(&uniform_data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            // Placeholder point buffer: fill with zero (real impl loads from cmd.file_path)
            let point_buf_size = (cmd.max_visible_points as u64).min(16384) * 16; // Point: xyz rgba (16B)
            let point_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PC points"),
                size: point_buf_size.max(64),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PointCloud BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: point_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding { buffer: &ub, offset: 0, size: None }
                    ) },
                ],
            });
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..(cmd.max_visible_points.min(16384)), 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::Vec3;
    use rc3d_scene::{
        DecalNode, GroupNode, NodeData, PointCloudNode, SceneGraph, TransformNode, VolumeNode,
    };

    #[test]
    fn collect_effect_nodes_preserves_decal_volume_and_point_cloud_fields() {
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Group(GroupNode));
        graph.add_child(
            root,
            NodeData::Decal(DecalNode {
                position: Vec3::new(1.0, 2.0, 3.0),
                direction: Vec3::NEG_Y,
                size: [4.0, 5.0],
                texture_path: "decal.png".to_string(),
                color: [1.0, 0.5, 0.25, 0.75],
                opacity: 0.8,
            }),
        );
        graph.add_child(
            root,
            NodeData::Volume(VolumeNode {
                dimensions: [16, 32, 64],
                texture_path: "volume.raw".to_string(),
                density_scale: 2.5,
                color_map: [[0.1, 0.2, 0.3, 0.4]; 4],
            }),
        );
        graph.add_child(
            root,
            NodeData::PointCloud(PointCloudNode {
                file_path: "points.bin".to_string(),
                max_visible_points: 1234,
                point_size: 3.0,
                color: [0.2, 0.4, 0.6, 1.0],
            }),
        );

        let commands = collect_effect_nodes(&graph);

        assert_eq!(commands.decals.len(), 1);
        assert_eq!(commands.decals[0].texture_path, "decal.png");
        assert_eq!(commands.decals[0].position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(commands.decals[0].size, [4.0, 5.0]);
        assert_eq!(commands.decals[0].opacity, 0.8);
        assert_eq!(commands.volumes.len(), 1);
        assert_eq!(commands.volumes[0].texture_path, "volume.raw");
        assert_eq!(commands.volumes[0].dimensions, [16, 32, 64]);
        assert_eq!(commands.volumes[0].density_scale, 2.5);
        assert_eq!(commands.point_clouds.len(), 1);
        assert_eq!(commands.point_clouds[0].file_path, "points.bin");
        assert_eq!(commands.point_clouds[0].max_visible_points, 1234);
        assert_eq!(commands.point_clouds[0].point_size, 3.0);
    }

    #[test]
    fn collect_effect_nodes_recurses_through_effect_children_and_transforms() {
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Group(GroupNode));
        let transform = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(10.0, 0.0, 0.0))),
        );
        let decal = graph.add_child(
            transform,
            NodeData::Decal(DecalNode {
                texture_path: "outer.png".to_string(),
                ..Default::default()
            }),
        );
        graph.add_child(
            decal,
            NodeData::PointCloud(PointCloudNode {
                file_path: "nested.bin".to_string(),
                ..Default::default()
            }),
        );

        let commands = collect_effect_nodes(&graph);

        assert_eq!(commands.decals.len(), 1);
        assert_eq!(commands.point_clouds.len(), 1);
        assert_eq!(commands.decals[0].model_matrix.w_axis.x, 10.0);
        assert_eq!(commands.point_clouds[0].model_matrix.w_axis.x, 10.0);
        assert_eq!(commands.point_clouds[0].file_path, "nested.bin");
    }
}
