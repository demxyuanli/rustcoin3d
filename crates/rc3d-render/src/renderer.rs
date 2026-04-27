use wgpu::util::DeviceExt;

use crate::frustum::Frustum;
use crate::gpu_resource::{GpuResourceManager, GpuUniformPool};
use crate::pipelines::PipelineSet;
use crate::render_action::DrawCall;
use crate::vertex::{FlatUniforms, OutlineUniforms, SceneUniforms};
use rc3d_core::DisplayMode;

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pipelines: PipelineSet,
    phong_pool: GpuUniformPool,
    flat_pool: GpuUniformPool,
    outline_pool: GpuUniformPool,
    gpu_meshes: GpuResourceManager,
    depth_texture: Option<(wgpu::Texture, wgpu::TextureView)>,
    global_display_mode: DisplayMode,
    clip_planes: Vec<[f32; 4]>,
    wireframe_supported: bool,
}

impl Renderer {
    pub async fn new(window: &winit::window::Window) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = unsafe {
            instance
                .create_surface_unsafe(
                    wgpu::SurfaceTargetUnsafe::from_window(window)
                        .expect("failed to create surface target"),
                )
                .expect("failed to create surface")
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("failed to find adapter");

        let required_features = wgpu::Features::POLYGON_MODE_LINE;
        let features = adapter.features() & required_features;
        let wireframe_supported = features.contains(wgpu::Features::POLYGON_MODE_LINE);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: features,
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let pipelines = PipelineSet::create(&device, config.format);
        let phong_pool = GpuUniformPool::new_phong(&device, 1024);
        let flat_pool = GpuUniformPool::new_flat(&device, 2048);
        let outline_pool = GpuUniformPool::new_outline(&device, 1024);

        let mut renderer = Self {
            device,
            queue,
            surface,
            config,
            pipelines,
            phong_pool,
            flat_pool,
            outline_pool,
            gpu_meshes: GpuResourceManager::new(),
            depth_texture: None,
            global_display_mode: DisplayMode::ShadedWithEdges,
            clip_planes: Vec::new(),
            wireframe_supported,
        };
        renderer.create_depth_texture();
        renderer
    }

    fn create_depth_texture(&mut self) {
        let size = wgpu::Extent3d {
            width: self.config.width,
            height: self.config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = self.device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_texture = Some((texture, view));
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.create_depth_texture();
        }
    }

    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        self.global_display_mode = mode;
    }

    pub fn display_mode(&self) -> DisplayMode {
        self.global_display_mode
    }

    pub fn set_clip_planes(&mut self, planes: Vec<[f32; 4]>) {
        self.clip_planes = planes;
    }

    pub fn clip_planes(&self) -> &[[f32; 4]] {
        &self.clip_planes
    }

    pub fn toggle_clip_plane(&mut self, axis: usize) {
        // axis: 0=X, 1=Y, 2=Z. Toggle clip plane at origin for that axis.
        let normal = match axis {
            0 => [1.0, 0.0, 0.0, 0.0],
            1 => [0.0, 1.0, 0.0, 0.0],
            2 => [0.0, 0.0, 1.0, 0.0],
            _ => return,
        };
        if let Some(pos) = self.clip_planes.iter().position(|p| p[0] == normal[0] && p[1] == normal[1] && p[2] == normal[2]) {
            self.clip_planes.remove(pos);
        } else {
            self.clip_planes.push(normal);
        }
    }

    pub fn render_draw_calls(&mut self, draw_calls: &[DrawCall]) {
        if draw_calls.is_empty() {
            return;
        }

        let output = match self.surface.get_current_texture() {
            Ok(o) => o,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth_view = &self
            .depth_texture
            .as_ref()
            .expect("depth texture missing")
            .1;

        // Frustum culling: compute VP from first draw call
        let first = &draw_calls[0];
        let vp = first.mvp * first.model_matrix.inverse();
        let frustum = Frustum::from_view_projection(vp);
        let visible: Vec<&DrawCall> = draw_calls
            .iter()
            .filter(|dc| dc.aabb.as_ref().map_or(true, |aabb| frustum.intersects_aabb(aabb)))
            .collect();

        self.phong_pool.reset();
        self.flat_pool.reset();
        self.outline_pool.reset();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let bg_color = wgpu::Color { r: 0.08, g: 0.08, b: 0.08, a: 1.0 };

        let mode = self.global_display_mode;

        // Outline pass (inverted hull): draw extruded backfaces before solid
        let run_outline = mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine;
        if run_outline {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Outline Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(bg_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipelines.outline);

            let outline_color = if mode == DisplayMode::HiddenLine {
                [0.5, 0.7, 1.0, 1.0]
            } else {
                [0.0, 0.0, 0.0, 1.0]
            };
            let outline_width = 0.015;

            for dc in &visible {
                if dc.vertices.is_empty() { continue; }
                let uniforms = OutlineUniforms {
                    mvp: dc.mvp.to_cols_array_2d(),
                    outline_width,
                    _pad: [0.0; 3],
                    color: outline_color,
                };
                if let Some((bg, offset)) = self.outline_pool.push_outline(&self.device, &self.queue, &uniforms) {
                    pass.set_bind_group(0, &bg, &[offset]);
                    let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Outline VB"),
                        contents: bytemuck::cast_slice(&dc.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    pass.set_vertex_buffer(0, vb.slice(..));
                    if let Some(ref indices) = dc.indices {
                        let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Outline IB"),
                            contents: bytemuck::cast_slice(indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                    } else {
                        pass.draw(0..dc.vertices.len() as u32, 0..1);
                    }
                }
            }
        }

        // Pass 1: Solid (fills depth + shading)
        if mode == DisplayMode::Shaded || mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine {
            let color_load = if run_outline { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(bg_color) };
            let depth_load = if run_outline { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(1.0) };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Solid Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: color_load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: depth_load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipelines.solid);

            let mut clip_arr = [[0.0f32; 4]; 6];
            for (i, cp) in self.clip_planes.iter().enumerate() {
                if i < 6 { clip_arr[i] = *cp; }
            }
            let clip_count = [self.clip_planes.len().min(6) as f32, 0.0, 0.0, 0.0];

            for dc in &visible {
                if dc.vertices.is_empty() { continue; }
                let obj_color = if mode == DisplayMode::HiddenLine {
                    [0.08, 0.08, 0.08, 1.0]
                } else {
                    [dc.color.x, dc.color.y, dc.color.z, 1.0]
                };
                let uniforms = SceneUniforms {
                    mvp: dc.mvp.to_cols_array_2d(),
                    model: dc.model_matrix.to_cols_array_2d(),
                    camera_pos: [dc.camera_pos.x, dc.camera_pos.y, dc.camera_pos.z, 1.0],
                    light_dir: [dc.light_dir.x, dc.light_dir.y, dc.light_dir.z, 0.0],
                    light_color: [dc.light_color.x, dc.light_color.y, dc.light_color.z, 1.0],
                    object_color: obj_color,
                    clip_planes: clip_arr,
                    clip_count,
                };
                if let Some((bg, offset)) = self.phong_pool.push_scene(&self.device, &self.queue, &uniforms) {
                    pass.set_bind_group(0, &bg, &[offset]);
                    let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("VB"),
                        contents: bytemuck::cast_slice(&dc.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    pass.set_vertex_buffer(0, vb.slice(..));
                    if let Some(ref indices) = dc.indices {
                        let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("IB"),
                            contents: bytemuck::cast_slice(indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                    } else {
                        pass.draw(0..dc.vertices.len() as u32, 0..1);
                    }
                }
            }
        }

        // Wireframe pass (standalone only)
        if self.wireframe_supported && mode == DisplayMode::Wireframe {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Wireframe Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(bg_color), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipelines.wireframe);
            let line_color = if mode == DisplayMode::HiddenLine {
                [0.6, 0.8, 1.0, 1.0] // light blue edges for hidden line
            } else {
                [0.2, 1.0, 0.4, 1.0] // green wireframe
            };

            for dc in &visible {
                if dc.vertices.is_empty() { continue; }
                let uniforms = FlatUniforms {
                    mvp: dc.mvp.to_cols_array_2d(),
                    color: line_color,
                };
                if let Some((bg, offset)) = self.flat_pool.push_flat(&self.device, &self.queue, &uniforms) {
                    pass.set_bind_group(0, &bg, &[offset]);
                    let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("VB"),
                        contents: bytemuck::cast_slice(&dc.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    pass.set_vertex_buffer(0, vb.slice(..));
                    if let Some(ref indices) = dc.indices {
                        let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("IB"),
                            contents: bytemuck::cast_slice(indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                    } else {
                        pass.draw(0..dc.vertices.len() as u32, 0..1);
                    }
                }
            }
        }

        // Edge overlay pass (shaded + edges mode, or hidden-line silhouette, or measurement overlay)
        let edge_worthy = mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine;
        let has_overlay = visible.iter().any(|dc| dc.overlay_color.is_some());
        if edge_worthy || has_overlay {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Edge Overlay Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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

            pass.set_pipeline(&self.pipelines.edge_overlay);

            let default_edge_color = if mode == DisplayMode::HiddenLine {
                [0.6, 0.8, 1.0, 0.8]
            } else {
                [0.0, 0.0, 0.0, 0.5]
            };

            for dc in &visible {
                if dc.edge_positions.is_empty() {
                    continue;
                }
                // Skip regular edges in non-edge modes (only draw overlays)
                if !edge_worthy && dc.overlay_color.is_none() {
                    continue;
                }
                let edge_color = dc.overlay_color.unwrap_or(default_edge_color);
                let uniforms = FlatUniforms {
                    mvp: dc.mvp.to_cols_array_2d(),
                    color: edge_color,
                };
                if let Some((bg, offset)) = self.flat_pool.push_flat(&self.device, &self.queue, &uniforms) {
                    pass.set_bind_group(0, &bg, &[offset]);
                    use crate::vertex::LineVertex;
                    let line_verts: Vec<LineVertex> = dc.edge_positions.iter()
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
        }

        // Selection highlight pass (additive orange overlay + bright edges)
        let has_selection = visible.iter().any(|dc| dc.selected);
        if has_selection {
            // Fill highlight
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Selection Fill Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
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

                pass.set_pipeline(&self.pipelines.selection_fill);

                for dc in &visible {
                    if !dc.selected || dc.vertices.is_empty() {
                        continue;
                    }
                    let uniforms = FlatUniforms {
                        mvp: dc.mvp.to_cols_array_2d(),
                        color: [1.0, 0.6, 0.0, 0.35],
                    };
                    if let Some((bg, offset)) = self.flat_pool.push_flat(&self.device, &self.queue, &uniforms) {
                        pass.set_bind_group(0, &bg, &[offset]);
                        let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Selection VB"),
                            contents: bytemuck::cast_slice(&dc.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                        pass.set_vertex_buffer(0, vb.slice(..));
                        if let Some(ref indices) = dc.indices {
                            let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Selection IB"),
                                contents: bytemuck::cast_slice(indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                        } else {
                            pass.draw(0..dc.vertices.len() as u32, 0..1);
                        }
                    }
                }
            }

            // Edge highlight (bright orange wireframe on selected objects)
            if self.wireframe_supported {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Selection Edge Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
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

                pass.set_pipeline(&self.pipelines.wireframe);
                let sel_color = [1.0, 0.5, 0.0, 1.0];

                for dc in &visible {
                    if !dc.selected || dc.vertices.is_empty() {
                        continue;
                    }
                    let uniforms = FlatUniforms {
                        mvp: dc.mvp.to_cols_array_2d(),
                        color: sel_color,
                    };
                    if let Some((bg, offset)) = self.flat_pool.push_flat(&self.device, &self.queue, &uniforms) {
                        pass.set_bind_group(0, &bg, &[offset]);
                        let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Sel Edge VB"),
                            contents: bytemuck::cast_slice(&dc.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                        pass.set_vertex_buffer(0, vb.slice(..));
                        if let Some(ref indices) = dc.indices {
                            let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Sel Edge IB"),
                                contents: bytemuck::cast_slice(indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                        } else {
                            pass.draw(0..dc.vertices.len() as u32, 0..1);
                        }
                    }
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}
