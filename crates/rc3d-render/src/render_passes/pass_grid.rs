//! Ground-plane reference grid overlay.
//!
//! Renders a world-space grid in the XZ plane (Y=0) with:
//! - Major lines at 1-unit spacing (brighter)
//! - Minor lines at 0.1-unit spacing (dimmer, shorter radius)
//! - X axis in red, Z axis in blue

use crate::vertex::{FlatUniforms, LineVertex};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Grid geometry for one frame, regenerated per-frame to center on camera.
pub struct GridGeometry {
    pub major_lines: Vec<LineVertex>,
    pub minor_lines: Vec<LineVertex>,
    pub axis_x: Vec<LineVertex>,
    pub axis_z: Vec<LineVertex>,
}

impl GridGeometry {
    /// Build grid lines centered on `camera_pos` in the XZ plane.
    pub fn build(camera_pos: Vec3) -> Self {
        let cx = (camera_pos.x / 0.1).round() * 0.1;
        let cz = (camera_pos.z / 0.1).round() * 0.1;
        let major_extent: i32 = 50;
        let minor_extent: i32 = 15;

        let mut major_lines = Vec::new();
        let mut minor_lines = Vec::new();

        for i in -minor_extent..=minor_extent {
            let x = cx + i as f32 * 0.1;
            let z = cz + i as f32 * 0.1;

            // Minor lines along X
            if i >= -minor_extent && i <= minor_extent {
                minor_lines.push(LineVertex {
                    position: [cx - minor_extent as f32 * 0.1, 0.0, z],
                });
                minor_lines.push(LineVertex {
                    position: [cx + minor_extent as f32 * 0.1, 0.0, z],
                });
            }
            // Minor lines along Z
            if i >= -minor_extent && i <= minor_extent {
                minor_lines.push(LineVertex {
                    position: [x, 0.0, cz - minor_extent as f32 * 0.1],
                });
                minor_lines.push(LineVertex {
                    position: [x, 0.0, cz + minor_extent as f32 * 0.1],
                });
            }
        }

        for i in -major_extent..=major_extent {
            let x = cx + i as f32;
            let z = cz + i as f32;
            // Major lines along X
            major_lines.push(LineVertex {
                position: [cx - major_extent as f32, 0.0, z],
            });
            major_lines.push(LineVertex {
                position: [cx + major_extent as f32, 0.0, z],
            });
            // Major lines along Z
            major_lines.push(LineVertex {
                position: [x, 0.0, cz - major_extent as f32],
            });
            major_lines.push(LineVertex {
                position: [x, 0.0, cz + major_extent as f32],
            });
        }

        let axis_len = major_extent as f32;
        let axis_x = vec![
            LineVertex {
                position: [-axis_len, 0.0, 0.0],
            },
            LineVertex {
                position: [axis_len, 0.0, 0.0],
            },
        ];
        let axis_z = vec![
            LineVertex {
                position: [0.0, 0.0, -axis_len],
            },
            LineVertex {
                position: [0.0, 0.0, axis_len],
            },
        ];

        GridGeometry {
            major_lines,
            minor_lines,
            axis_x,
            axis_z,
        }
    }
}

/// Draw the grid using the grid_lines pipeline.
pub fn pass_grid(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    vp: Mat4,
    camera_pos: Vec3,
    depth_reversed_z: bool,
) {
    let geom = GridGeometry::build(camera_pos);
    let mvp = vp.to_cols_array_2d();

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Grid Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
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
        occlusion_query_set: None, multiview_mask: None,
    });
    renderer.apply_scene_viewport(&mut pass);

    let grid_pl = if depth_reversed_z {
        &renderer.gpu.pipelines.grid_lines_reverse
    } else {
        &renderer.gpu.pipelines.grid_lines_forward
    };
    pass.set_pipeline(grid_pl);

    // Minor grid lines: dim, thin
    if !geom.minor_lines.is_empty() {
        let uniforms = FlatUniforms {
            mvp,
            color: [0.25, 0.25, 0.25, 0.35],
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
            draw_lines(&renderer.device, &mut pass, &geom.minor_lines);
        }
    }

    // Major grid lines: medium
    if !geom.major_lines.is_empty() {
        let uniforms = FlatUniforms {
            mvp,
            color: [0.35, 0.35, 0.35, 0.55],
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
            draw_lines(&renderer.device, &mut pass, &geom.major_lines);
        }
    }

    // X axis: red
    {
        let uniforms = FlatUniforms {
            mvp,
            color: [0.9, 0.2, 0.2, 0.8],
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
            draw_lines(&renderer.device, &mut pass, &geom.axis_x);
        }
    }

    // Z axis: blue
    {
        let uniforms = FlatUniforms {
            mvp,
            color: [0.2, 0.3, 0.9, 0.8],
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
            draw_lines(&renderer.device, &mut pass, &geom.axis_z);
        }
    }
}

/// World-space transform gizmo overlay (always on top of the shade target).
pub fn pass_gizmo_lines(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    vp: Mat4,
    depth_reversed_z: bool,
) {
    if renderer.gizmo_line_batches.is_empty() {
        return;
    }
    let batches = renderer.gizmo_line_batches.clone();
    let mvp = vp.to_cols_array_2d();

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Gizmo Overlay"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
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
        occlusion_query_set: None, multiview_mask: None,
    });
    renderer.apply_scene_viewport(&mut pass);

    let grid_pl = if depth_reversed_z {
        &renderer.gpu.pipelines.grid_lines_reverse
    } else {
        &renderer.gpu.pipelines.grid_lines_forward
    };
    pass.set_pipeline(grid_pl);

    for (lines, color) in batches {
        if lines.len() < 2 {
            continue;
        }
        let uniforms = FlatUniforms {
            mvp,
            color,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
            draw_lines(&renderer.device, &mut pass, &lines);
        }
    }
}

fn draw_lines(
    device: &wgpu::Device,
    pass: &mut wgpu::RenderPass<'_>,
    lines: &[LineVertex],
) {
    if lines.len() < 2 {
        return;
    }
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grid vb"),
        contents: bytemuck::cast_slice(lines),
        usage: wgpu::BufferUsages::VERTEX,
    });
    pass.set_vertex_buffer(0, vb.slice(..));
    pass.draw(0..lines.len() as u32, 0..1);
}
