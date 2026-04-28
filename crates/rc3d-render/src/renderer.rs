use std::collections::HashMap;
use std::sync::Arc;
use slotmap::Key;
use wgpu::util::DeviceExt;

use crate::frustum::Frustum;
use crate::gpu_resource::{GpuMesh, GpuResourceManager, GpuUniformPool};
use crate::hud::HudRenderer;
use crate::pipelines::PipelineSet;
use crate::render_action::DrawCall;
use crate::vertex::{FlatUniforms, LineVertex, OutlineUniforms, SceneUniforms};
use rc3d_core::DisplayMode;

const MESH_CACHE_MAX: usize = 256;
const MESH_CACHE_IDLE_FRAMES: u64 = 120;
const PERFORMANCE_MODE_TRIANGLE_THRESHOLD: u64 = 2_000_000;
const ADAPTIVE_MEDIUM_ENTER_MS: f32 = 26.0;
const ADAPTIVE_LOW_ENTER_MS: f32 = 40.0;
const ADAPTIVE_MEDIUM_EXIT_MS: f32 = 22.0;
const ADAPTIVE_LOW_EXIT_MS: f32 = 33.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AdaptiveQuality {
    High,
    Medium,
    Low,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub visible_triangles: u64,
    pub visible_draw_calls: usize,
    pub culled_draw_calls: usize,
}

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
    mesh_cache: HashMap<u64, (crate::gpu_resource::MeshId, u64)>,
    frame_counter: u64,
    performance_mode_active: bool,
    hud: Option<HudRenderer>,
    hud_enabled: bool,
    adaptive_quality: AdaptiveQuality,
    last_hud_update_frame: u64,
}

impl Renderer {
    fn display_mode_sort_key(mode: DisplayMode) -> u8 {
        match mode {
            DisplayMode::Shaded => 0,
            DisplayMode::ShadedWithEdges => 1,
            DisplayMode::Wireframe => 2,
            DisplayMode::HiddenLine => 3,
        }
    }

    fn color_sort_key(color: [f32; 4]) -> [u32; 4] {
        [
            color[0].to_bits(),
            color[1].to_bits(),
            color[2].to_bits(),
            color[3].to_bits(),
        ]
    }

    fn light_dirs_sort_key(dirs: [[f32; 4]; 4]) -> [[u32; 4]; 4] {
        dirs.map(Self::color_sort_key)
    }

    fn light_colors_sort_key(colors: [[f32; 4]; 4]) -> [[u32; 4]; 4] {
        colors.map(Self::color_sort_key)
    }

    fn light_types_sort_key(types: [[f32; 4]; 4]) -> [[u32; 4]; 4] {
        types.map(Self::color_sort_key)
    }

    fn light_positions_sort_key(positions: [[f32; 4]; 4]) -> [[u32; 4]; 4] {
        positions.map(Self::color_sort_key)
    }

    fn spot_params_sort_key(params: [[f32; 4]; 4]) -> [[u32; 4]; 4] {
        params.map(Self::color_sort_key)
    }

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

        let required_features = wgpu::Features::POLYGON_MODE_LINE
            | wgpu::Features::DEPTH32FLOAT_STENCIL8;
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
            mesh_cache: HashMap::new(),
            frame_counter: 0,
            performance_mode_active: false,
            hud: None,
            hud_enabled: true,
            adaptive_quality: AdaptiveQuality::High,
            last_hud_update_frame: 0,
        };
        renderer.hud = Some(HudRenderer::new(
            &renderer.device,
            &renderer.queue,
            renderer.config.format,
            renderer.config.width,
            renderer.config.height,
        ));
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
            format: wgpu::TextureFormat::Depth32FloatStencil8,
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
            if let Some(hud) = &mut self.hud {
                hud.resize(&self.queue, width, height);
            }
        }
    }

    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        self.global_display_mode = mode;
    }

    pub fn display_mode(&self) -> DisplayMode {
        self.global_display_mode
    }

    pub fn performance_mode_active(&self) -> bool {
        self.performance_mode_active
    }

    pub fn adaptive_quality_name(&self) -> &'static str {
        match self.adaptive_quality {
            AdaptiveQuality::High => "High",
            AdaptiveQuality::Medium => "Medium",
            AdaptiveQuality::Low => "Low",
        }
    }

    pub fn adaptive_is_low(&self) -> bool {
        self.adaptive_quality == AdaptiveQuality::Low
    }

    pub fn report_frame_time_ms(&mut self, frame_time_ms: f32) {
        let previous = self.adaptive_quality;
        self.adaptive_quality = match self.adaptive_quality {
            AdaptiveQuality::High => {
                if frame_time_ms >= ADAPTIVE_LOW_ENTER_MS {
                    AdaptiveQuality::Low
                } else if frame_time_ms >= ADAPTIVE_MEDIUM_ENTER_MS {
                    AdaptiveQuality::Medium
                } else {
                    AdaptiveQuality::High
                }
            }
            AdaptiveQuality::Medium => {
                if frame_time_ms >= ADAPTIVE_LOW_ENTER_MS {
                    AdaptiveQuality::Low
                } else if frame_time_ms <= ADAPTIVE_MEDIUM_EXIT_MS {
                    AdaptiveQuality::High
                } else {
                    AdaptiveQuality::Medium
                }
            }
            AdaptiveQuality::Low => {
                if frame_time_ms <= ADAPTIVE_LOW_EXIT_MS {
                    if frame_time_ms <= ADAPTIVE_MEDIUM_EXIT_MS {
                        AdaptiveQuality::High
                    } else {
                        AdaptiveQuality::Medium
                    }
                } else {
                    AdaptiveQuality::Low
                }
            }
        };

        if previous != self.adaptive_quality {
            log::warn!(
                "Adaptive quality changed: {:?} -> {:?} (frame_ms={:.2})",
                previous,
                self.adaptive_quality,
                frame_time_ms
            );
        }
    }

    pub fn set_clip_planes(&mut self, planes: Vec<[f32; 4]>) {
        self.clip_planes = planes;
    }

    pub fn set_vsync(&mut self, enabled: bool) {
        self.config.present_mode = if enabled {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        self.surface.configure(&self.device, &self.config);
    }

    pub fn clip_planes(&self) -> &[[f32; 4]] {
        &self.clip_planes
    }

    pub fn toggle_clip_plane(&mut self, axis: usize) {
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

    pub fn invalidate_mesh_cache(&mut self) {
        self.mesh_cache.clear();
    }

    pub fn render_draw_calls(&mut self, draw_calls: &[DrawCall]) -> FrameStats {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        if draw_calls.is_empty() {
            return FrameStats::default();
        }

        let output = match self.surface.get_current_texture() {
            Ok(o) => o,
            Err(_) => return FrameStats::default(),
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        if self.depth_texture.is_none() {
            self.create_depth_texture();
        }
        let Some((_, depth_view)) = self.depth_texture.as_ref() else {
            return FrameStats::default();
        };
        let depth_view = depth_view.clone();

        // Frustum culling: compute VP from first draw call
        let first = &draw_calls[0];
        let vp = first.mvp * first.model_matrix.inverse();
        let frustum = Frustum::from_view_projection(vp);
        let visible: Vec<&DrawCall> = draw_calls
            .iter()
            .filter(|dc| dc.aabb.as_ref().map_or(true, |aabb| frustum.intersects_aabb(aabb)))
            .collect();
        let stats = FrameStats {
            visible_triangles: 0,
            visible_draw_calls: visible.len(),
            culled_draw_calls: draw_calls.len().saturating_sub(visible.len()),
        };
        let total_visible_triangles: u64 = visible
            .iter()
            .map(|dc| {
                if let Some(indices) = dc.indices.as_ref() {
                    (indices.len() / 3) as u64
                } else {
                    (dc.vertices.len() / 3) as u64
                }
            })
            .sum();
        let stats = FrameStats {
            visible_triangles: total_visible_triangles,
            ..stats
        };
        let enable_perf_mode = total_visible_triangles > PERFORMANCE_MODE_TRIANGLE_THRESHOLD;
        if enable_perf_mode != self.performance_mode_active {
            self.performance_mode_active = enable_perf_mode;
            if enable_perf_mode {
                log::warn!(
                    "Performance mode enabled: triangle_count={} threshold={}",
                    total_visible_triangles,
                    PERFORMANCE_MODE_TRIANGLE_THRESHOLD
                );
            } else {
                log::info!("Performance mode disabled");
            }
        }

        // Precompute mesh handles once for all passes
        // Use Arc pointer identity as fast-path hash; full data hash as fallback.
        let mut mesh_handles: Vec<Option<crate::gpu_resource::MeshId>> = Vec::with_capacity(visible.len());
        {
            for dc in &visible {
                let handle = if dc.vertices.is_empty() {
                    None
                } else {
                    let hash = dc.mesh_hash.unwrap_or_else(|| {
                        let ptr_key = (
                            Arc::as_ptr(&dc.vertices) as u64,
                            dc.indices.as_ref().map_or(0u64, |a| Arc::as_ptr(a) as u64),
                        );
                        let mut h = twox_hash::XxHash64::with_seed(0);
                        std::hash::Hasher::write_u64(&mut h, ptr_key.0);
                        std::hash::Hasher::write_u64(&mut h, ptr_key.1);
                        std::hash::Hasher::finish(&h)
                    });
                    if let Some((mesh_id, last_used)) = self.mesh_cache.get_mut(&hash) {
                        *last_used = self.frame_counter;
                        Some(*mesh_id)
                    } else {
                        // Hash cached but mesh evicted - need to re-upload
                        let h = hash;
                        if self.mesh_cache.len() >= MESH_CACHE_MAX {
                            self.prune_mesh_cache();
                        }
                        let mesh_id = self.gpu_meshes.upload_mesh(
                            &self.device,
                            &dc.vertices,
                            dc.indices.as_ref().map(|a| a.as_slice()),
                            &dc.edge_positions,
                        );
                        self.mesh_cache.insert(h, (mesh_id, self.frame_counter));
                        Some(mesh_id)
                    }
                };
                mesh_handles.push(handle);
            }
        }
        let mut solid_order: Vec<usize> = (0..visible.len())
            .filter(|&i| !visible[i].vertices.is_empty())
            .collect();
        solid_order.sort_by_key(|&i| {
            let dc = visible[i];
            (
                Self::light_dirs_sort_key(dc.light_dirs),
                Self::light_colors_sort_key(dc.light_colors),
                Self::light_types_sort_key(dc.light_types),
                Self::light_positions_sort_key(dc.light_positions),
                Self::spot_params_sort_key(dc.spot_params),
                dc.light_count,
                Self::display_mode_sort_key(dc.display_mode),
                Self::color_sort_key([dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]),
                Self::color_sort_key([dc.ambient_color.x, dc.ambient_color.y, dc.ambient_color.z, 1.0]),
                Self::color_sort_key([dc.specular_color.x, dc.specular_color.y, dc.specular_color.z, 1.0]),
                dc.shininess.to_bits(),
                mesh_handles[i].map(|m| m.data().as_ffi()).unwrap_or(0),
            )
        });
        let mut edge_order: Vec<usize> = (0..visible.len())
            .filter(|&i| !visible[i].edge_positions.is_empty())
            .collect();
        edge_order.sort_by_key(|&i| {
            let dc = visible[i];
            (
                Self::display_mode_sort_key(dc.display_mode),
                Self::color_sort_key(dc.overlay_color.unwrap_or([0.0, 0.0, 0.0, 0.5])),
                mesh_handles[i].map(|m| m.data().as_ffi()).unwrap_or(0),
            )
        });
        let mut selected_order: Vec<usize> = (0..visible.len())
            .filter(|&i| visible[i].selected && !visible[i].vertices.is_empty())
            .collect();
        selected_order.sort_by_key(|&i| {
            let dc = visible[i];
            (
                Self::display_mode_sort_key(dc.display_mode),
                mesh_handles[i].map(|m| m.data().as_ffi()).unwrap_or(0),
            )
        });

        self.phong_pool.reset();
        self.flat_pool.reset();
        self.outline_pool.reset();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let bg_color = wgpu::Color { r: 0.08, g: 0.08, b: 0.08, a: 1.0 };

        let base_mode = if self.performance_mode_active {
            DisplayMode::Shaded
        } else {
            self.global_display_mode
        };
        let mode = if self.adaptive_quality == AdaptiveQuality::Low && base_mode == DisplayMode::Wireframe {
            DisplayMode::Shaded
        } else {
            base_mode
        };

        // Inverted-hull + stencil: solid writes stencil=1; expanded backfaces (cull front) only
        // pass where stencil==0 (halo) so interior is not overdrawn in outline color.
        let run_outline = !self.performance_mode_active
            && self.adaptive_quality == AdaptiveQuality::High
            && (mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine);

        if mode == DisplayMode::Shaded || mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine {
            let pass_label = if run_outline {
                "Solid+Outline Pass"
            } else {
                "Solid Pass"
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(pass_label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(bg_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0u32),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipelines.solid);
            // Reference value 1: solid Replace writes stencil=1 for lit surface pixels.
            pass.set_stencil_reference(1);

            let mut clip_arr = [[0.0f32; 4]; 6];
            for (i, cp) in self.clip_planes.iter().enumerate() {
                if i < 6 { clip_arr[i] = *cp; }
            }
            let clip_count = [self.clip_planes.len().min(6) as f32, 0.0, 0.0, 0.0];

            let mut last_bound_mesh = None;
            let mut start = 0usize;
            while start < solid_order.len() {
                let head_idx = solid_order[start];
                let head_dc = visible[head_idx];
                let light_key = (
                    Self::light_dirs_sort_key(head_dc.light_dirs),
                    Self::light_colors_sort_key(head_dc.light_colors),
                    Self::light_types_sort_key(head_dc.light_types),
                    Self::light_positions_sort_key(head_dc.light_positions),
                    Self::spot_params_sort_key(head_dc.spot_params),
                    head_dc.light_count,
                    Self::color_sort_key([head_dc.diffuse_color.x, head_dc.diffuse_color.y, head_dc.diffuse_color.z, 1.0]),
                    Self::color_sort_key([head_dc.ambient_color.x, head_dc.ambient_color.y, head_dc.ambient_color.z, 1.0]),
                    Self::color_sort_key([head_dc.specular_color.x, head_dc.specular_color.y, head_dc.specular_color.z, 1.0]),
                    head_dc.shininess.to_bits(),
                );
                let mut end = start + 1;
                while end < solid_order.len() {
                    let idx = solid_order[end];
                    let dc = visible[idx];
                    let key = (
                        Self::light_dirs_sort_key(dc.light_dirs),
                        Self::light_colors_sort_key(dc.light_colors),
                        Self::light_types_sort_key(dc.light_types),
                        Self::light_positions_sort_key(dc.light_positions),
                        Self::spot_params_sort_key(dc.spot_params),
                        dc.light_count,
                        Self::color_sort_key([dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]),
                        Self::color_sort_key([dc.ambient_color.x, dc.ambient_color.y, dc.ambient_color.z, 1.0]),
                        Self::color_sort_key([dc.specular_color.x, dc.specular_color.y, dc.specular_color.z, 1.0]),
                        dc.shininess.to_bits(),
                    );
                    if key != light_key {
                        break;
                    }
                    end += 1;
                }

                for &i in &solid_order[start..end] {
                    let dc = visible[i];
                    let diffuse_color = if mode == DisplayMode::HiddenLine {
                        [0.08, 0.08, 0.08, 1.0]
                    } else {
                        [dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]
                    };
                    let uniforms = SceneUniforms {
                        mvp: dc.mvp.to_cols_array_2d(),
                        model: dc.model_matrix.to_cols_array_2d(),
                        camera_pos: [dc.camera_pos.x, dc.camera_pos.y, dc.camera_pos.z, 1.0],
                        light_dirs: head_dc.light_dirs,
                        light_colors: head_dc.light_colors,
                        light_types: head_dc.light_types,
                        light_positions: head_dc.light_positions,
                        spot_params: head_dc.spot_params,
                        light_count: [head_dc.light_count as f32, 0.0, 0.0, 0.0],
                        diffuse_color,
                        ambient_color: [dc.ambient_color.x, dc.ambient_color.y, dc.ambient_color.z, 1.0],
                        specular_color: [dc.specular_color.x, dc.specular_color.y, dc.specular_color.z, 1.0],
                        shininess: [dc.shininess, 0.0, 0.0, 0.0],
                        clip_planes: clip_arr,
                        clip_count,
                    };
                    if let Some(offset) = self.phong_pool.push_scene(&uniforms) {
                        pass.set_bind_group(0, self.phong_pool.bind_group(), &[offset]);
                        if let Some(mesh_id) = mesh_handles[i] {
                            self.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
                        }
                    }
                }
                start = end;
            }

            if run_outline {
                pass.set_pipeline(&self.pipelines.outline);
                // Equal(0): only halo pixels; interior has stencil=1 and fails the test.
                pass.set_stencil_reference(0);

                let outline_color = if mode == DisplayMode::HiddenLine {
                    [0.5, 0.7, 1.0, 1.0]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                };
                let outline_width = 0.022;

                let mut last_bound_mesh = None;
                for &i in &solid_order {
                    let dc = visible[i];
                    let uniforms = OutlineUniforms {
                        mvp: dc.mvp.to_cols_array_2d(),
                        outline_width,
                        _pad: [0.0; 3],
                        color: outline_color,
                    };
                    if let Some(offset) = self.outline_pool.push_outline(&uniforms) {
                        pass.set_bind_group(0, self.outline_pool.bind_group(), &[offset]);
                        if let Some(mesh_id) = mesh_handles[i] {
                            self.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
                        }
                    }
                }
            }
        }

        // Wireframe pass (standalone only)
        if !self.performance_mode_active && self.wireframe_supported && mode == DisplayMode::Wireframe {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Wireframe Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(bg_color), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0u32),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipelines.wireframe);
            let line_color = [0.2, 1.0, 0.4, 1.0];

            let mut last_bound_mesh = None;
            for &i in &solid_order {
                let dc = visible[i];
                let uniforms = FlatUniforms {
                    mvp: dc.mvp.to_cols_array_2d(),
                    color: line_color,
                };
                if let Some(offset) = self.flat_pool.push_flat(&uniforms) {
                    pass.set_bind_group(0, self.flat_pool.bind_group(), &[offset]);
                    if let Some(mesh_id) = mesh_handles[i] {
                        self.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
                    }
                }
            }
        }

        // Edge overlay pass
        let edge_worthy = !self.performance_mode_active
            && self.adaptive_quality != AdaptiveQuality::Low
            && (mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine);
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
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_stencil_reference(0);
            pass.set_pipeline(&self.pipelines.edge_overlay);

            let default_edge_color = if mode == DisplayMode::HiddenLine {
                [0.6, 0.8, 1.0, 0.8]
            } else {
                [0.0, 0.0, 0.0, 0.5]
            };

            let mut last_bound_edge_mesh = None;
            for &i in &edge_order {
                let dc = visible[i];
                if !edge_worthy && dc.overlay_color.is_none() {
                    continue;
                }
                let edge_color = dc.overlay_color.unwrap_or(default_edge_color);
                let uniforms = FlatUniforms {
                    mvp: dc.mvp.to_cols_array_2d(),
                    color: edge_color,
                };
                if let Some(offset) = self.flat_pool.push_flat(&uniforms) {
                    pass.set_bind_group(0, self.flat_pool.bind_group(), &[offset]);
                    let drawn_from_cache = if let Some(mesh_id) = mesh_handles[i] {
                        self.draw_edges_batched(&mut pass, mesh_id, &mut last_bound_edge_mesh)
                    } else {
                        false
                    };
                    if !drawn_from_cache {
                        self.bind_and_draw_edges(&mut pass, dc, mesh_handles[i]);
                    }
                }
            }
        }

        // Selection highlight pass
        let has_selection = visible.iter().any(|dc| dc.selected);
        if has_selection && !self.performance_mode_active {
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
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                pass.set_stencil_reference(0);
                pass.set_pipeline(&self.pipelines.selection_fill);

                let mut last_bound_mesh = None;
                for &i in &selected_order {
                    let dc = visible[i];
                    let uniforms = FlatUniforms {
                        mvp: dc.mvp.to_cols_array_2d(),
                        color: [1.0, 0.6, 0.0, 0.35],
                    };
                    if let Some(offset) = self.flat_pool.push_flat(&uniforms) {
                        pass.set_bind_group(0, self.flat_pool.bind_group(), &[offset]);
                        if let Some(mesh_id) = mesh_handles[i] {
                            self.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
                        }
                    }
                }
            }

            // Edge highlight (bright orange wireframe on selected objects)
            if self.wireframe_supported && self.adaptive_quality != AdaptiveQuality::Low {
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
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                pass.set_stencil_reference(0);
                pass.set_pipeline(&self.pipelines.wireframe);
                let sel_color = [1.0, 0.5, 0.0, 1.0];

                let mut last_bound_mesh = None;
                for &i in &selected_order {
                    let dc = visible[i];
                    let uniforms = FlatUniforms {
                        mvp: dc.mvp.to_cols_array_2d(),
                        color: sel_color,
                    };
                    if let Some(offset) = self.flat_pool.push_flat(&uniforms) {
                        pass.set_bind_group(0, self.flat_pool.bind_group(), &[offset]);
                        if let Some(mesh_id) = mesh_handles[i] {
                            self.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
                        }
                    }
                }
            }
        }

        self.outline_pool.flush(&self.queue);
        self.phong_pool.flush(&self.queue);
        self.flat_pool.flush(&self.queue);
        if self.hud_enabled {
            if let Some(hud) = self.hud.as_ref() {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("HUD Overlay Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                hud.render(&mut pass);
            }
        }
        self.prune_mesh_cache();
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        stats
    }

    pub fn update_hud(&mut self, fps: f32, frame_time_ms: f32, stats: FrameStats, mode_name: &str) {
        if !self.hud_enabled {
            return;
        }
        let interval = match self.adaptive_quality {
            AdaptiveQuality::High => 1,
            AdaptiveQuality::Medium => 2,
            AdaptiveQuality::Low => 6,
        };
        if self.frame_counter.saturating_sub(self.last_hud_update_frame) < interval {
            return;
        }
        self.last_hud_update_frame = self.frame_counter;
        let quality_name = self.adaptive_quality_name();
        let hud_mode_name = format!("{mode_name} [{quality_name}]");
        if let Some(hud) = &mut self.hud {
            hud.update_text(
                &self.device,
                &self.queue,
                fps,
                frame_time_ms,
                stats,
                &hud_mode_name,
            );
        }
    }

    // -- Mesh cache helpers --

    fn get_mesh(&self, mesh_id: crate::gpu_resource::MeshId) -> Option<&GpuMesh> {
        self.gpu_meshes.get(mesh_id)
    }

    fn draw_mesh_batched(
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

    fn draw_edges_batched(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        mesh_id: crate::gpu_resource::MeshId,
        last_bound: &mut Option<crate::gpu_resource::MeshId>,
    ) -> bool {
        let Some(mesh) = self.get_mesh(mesh_id) else { return false };
        let Some(edge_buffer) = &mesh.edge_vertex_buffer else { return false };
        if last_bound.map_or(true, |id| id != mesh_id) {
            pass.set_vertex_buffer(0, edge_buffer.slice(..));
            *last_bound = Some(mesh_id);
        }
        pass.draw(0..mesh.edge_vertex_count, 0..1);
        true
    }

    fn bind_and_draw_edges(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        dc: &DrawCall,
        handle: Option<crate::gpu_resource::MeshId>,
    ) {
        if let Some(mesh_id) = handle {
            if let Some(mesh) = self.get_mesh(mesh_id) {
                if let Some(edge_buffer) = &mesh.edge_vertex_buffer {
                    pass.set_vertex_buffer(0, edge_buffer.slice(..));
                    pass.draw(0..mesh.edge_vertex_count, 0..1);
                    return;
                }
            }
        }
        // Fallback: edge-only draw calls (no vertices) or missing edge buffer
        if !dc.edge_positions.is_empty() {
            let line_verts: Vec<LineVertex> = dc
                .edge_positions
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

    fn prune_mesh_cache(&mut self) {
        let frame = self.frame_counter;
        let stale_keys: Vec<u64> = self
            .mesh_cache
            .iter()
            .filter(|(_, (_, last_used))| frame.saturating_sub(*last_used) > MESH_CACHE_IDLE_FRAMES)
            .map(|(k, _)| *k)
            .collect();
        for key in stale_keys {
            if let Some((mesh_id, _)) = self.mesh_cache.remove(&key) {
                self.gpu_meshes.remove(mesh_id);
            }
        }
    }
}
