use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use glam::Vec3;
use slotmap::Key;
use wgpu::util::DeviceExt;

use crate::cluster::{ClusterRenderer, ClusterSet};
use crate::frustum::Frustum;
use crate::gpu_resource::{GpuMesh, GpuResourceManager, GpuUniformPool};
use crate::hud::HudRenderer;
use crate::hzb::{HzbBaker, HzbPyramids};
use crate::pipelines::PipelineSet;
use crate::render_action::DrawCall;
use crate::texture_cache::{ibl_from_image_path, TextureCache};
use crate::render_passes::{self, PassContext};
use crate::shadow_map::{aabb_from_scene, directional_light_view_proj, primary_directional_light_dir, union_draw_call_aabbs};
use crate::sort_keys;
use crate::vertex::LineVertex;
use glam::Mat4;
use rc3d_core::DisplayMode;
use rc3d_scene::SceneGraph;

const MESH_CACHE_MAX: usize = 256;
const MESH_CACHE_IDLE_FRAMES: u64 = 120;
const PERFORMANCE_MODE_TRIANGLE_THRESHOLD: u64 = 2_000_000;
const ADAPTIVE_MEDIUM_ENTER_MS: f32 = 26.0;
const ADAPTIVE_LOW_ENTER_MS: f32 = 40.0;
const ADAPTIVE_MEDIUM_EXIT_MS: f32 = 22.0;
const ADAPTIVE_LOW_EXIT_MS: f32 = 33.0;
const CLUSTER_PIPELINE_GENERATION: u32 = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AdaptiveQuality {
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
    pub pipelines: PipelineSet,
    pub phong_pool: GpuUniformPool,
    pub flat_pool: GpuUniformPool,
    pub outline_pool: GpuUniformPool,
    pub gpu_meshes: GpuResourceManager,
    pub depth_texture: Option<(wgpu::Texture, wgpu::TextureView, wgpu::TextureView)>,
    pub global_display_mode: DisplayMode,
    pub clip_planes: Vec<[f32; 4]>,
    pub wireframe_supported: bool,
    pub mesh_cache: HashMap<u64, (crate::gpu_resource::MeshId, u64)>,
    pub frame_counter: u64,
    pub performance_mode_active: bool,
    pub hud: Option<HudRenderer>,
    pub hud_enabled: bool,
    pub(super) adaptive_quality: AdaptiveQuality,
    pub last_hud_update_frame: u64,
    pub outline_width: f32,
    pub outline_color: [f32; 4],
    pub cluster_renderer: Option<ClusterRenderer>,
    pub cluster_cache: HashMap<u64, ClusterSet>,
    pub hzb: Option<HzbPyramids>,
    pub hzb_baker: Option<HzbBaker>,
    pub cluster_pipeline_generation: u32,
    depth_reversed_z_mismatch_warned: bool,
    pub texture_cache: TextureCache,
    pub ibl_diffuse: [f32; 4],
    pub ibl_specular: [f32; 4],
    pub shadow_pool: GpuUniformPool,
    pub shadow_compare_sampler: wgpu::Sampler,
    pub shadow_depth_tex: wgpu::Texture,
    pub shadow_depth_view: wgpu::TextureView,
    pub shadow_bind_group: wgpu::BindGroup,
    pub shadow_map_resolution: u32,
    pub hdr_post_processing: bool,
    pub post_process_bgl: wgpu::BindGroupLayout,
    pub post_process_sampler: wgpu::Sampler,
    /// Tonemap + FXAA: HDR scene -> linear LDR (`post_ldr`).
    pub post_process_pipeline: wgpu::RenderPipeline,
    /// Fullscreen blit: `post_ldr` -> swapchain (display encoding).
    pub blit_ldr_pipeline: wgpu::RenderPipeline,
    pub hdr_scene_tex: Option<wgpu::Texture>,
    pub hdr_scene_view: Option<wgpu::TextureView>,
    pub post_process_bind_group: Option<wgpu::BindGroup>,
    pub post_ldr_tex: Option<wgpu::Texture>,
    pub post_ldr_view: Option<wgpu::TextureView>,
    pub blit_ldr_bind_group: Option<wgpu::BindGroup>,
}

impl Renderer {
    fn create_shadow_map_resources(
        device: &wgpu::Device,
        pipelines: &PipelineSet,
        compare_sampler: &wgpu::Sampler,
        resolution: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
        let resolution = resolution.max(1);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow map"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow resources"),
            layout: &pipelines.shadow_resource_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(compare_sampler),
                },
            ],
        });
        (tex, view, bg)
    }

    pub fn ensure_shadow_map(&mut self, resolution: u32) {
        let resolution = resolution.max(1);
        if resolution == self.shadow_map_resolution {
            return;
        }
        let (tex, view, bg) = Self::create_shadow_map_resources(
            &self.device,
            &self.pipelines,
            &self.shadow_compare_sampler,
            resolution,
        );
        self.shadow_depth_tex = tex;
        self.shadow_depth_view = view;
        self.shadow_bind_group = bg;
        self.shadow_map_resolution = resolution;
    }

    pub fn set_hdr_post_processing(&mut self, enabled: bool) {
        self.hdr_post_processing = enabled;
        if !enabled {
            self.hdr_scene_tex = None;
            self.hdr_scene_view = None;
            self.post_process_bind_group = None;
            self.post_ldr_tex = None;
            self.post_ldr_view = None;
            self.blit_ldr_bind_group = None;
        } else {
            self.ensure_hdr_scene_target();
        }
    }

    pub(super) fn ensure_hdr_scene_target(&mut self) {
        if !self.hdr_post_processing {
            return;
        }
        let w = self.config.width.max(1);
        let h = self.config.height.max(1);
        if let (Some(ht), Some(pt)) = (&self.hdr_scene_tex, &self.post_ldr_tex) {
            let hsz = ht.size();
            let psz = pt.size();
            if hsz.width == w
                && hsz.height == h
                && psz.width == w
                && psz.height == h
                && self.post_process_bind_group.is_some()
                && self.blit_ldr_bind_group.is_some()
            {
                return;
            }
        }
        self.hdr_scene_tex = None;
        self.hdr_scene_view = None;
        self.post_process_bind_group = None;
        self.post_ldr_tex = None;
        self.post_ldr_view = None;
        self.blit_ldr_bind_group = None;
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR scene color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let tv = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post process bind"),
            layout: &self.post_process_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.post_process_sampler),
                },
            ],
        });
        self.hdr_scene_tex = Some(tex);
        self.hdr_scene_view = Some(tv);
        self.post_process_bind_group = Some(bg);

        let post_ldr = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Post LDR color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let post_ldr_v = post_ldr.create_view(&wgpu::TextureViewDescriptor::default());
        let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit post LDR"),
            layout: &self.post_process_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&post_ldr_v),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.post_process_sampler),
                },
            ],
        });
        self.post_ldr_tex = Some(post_ldr);
        self.post_ldr_view = Some(post_ldr_v);
        self.blit_ldr_bind_group = Some(blit_bg);
    }

    fn log_mesh_shader_assessment(adapter: &wgpu::Adapter) {
        let info = adapter.get_info();
        let backend = format!("{:?}", info.backend);
        let driver = info.driver.clone();
        let device = info.name.clone();
        let features = adapter.features();
        let has_indirect = features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        log::warn!(
            "Mesh shader assessment: backend={}, device={}, driver={}, indirect_first_instance={}, recommendation=keep compute-culling+indirect-draw for portability; evaluate mesh shaders only on vendor-locked high-end targets",
            backend,
            device,
            driver,
            has_indirect
        );
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
        Self::log_mesh_shader_assessment(&adapter);

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
        let shadow_pool = GpuUniformPool::new_shadow_pool(&device, &pipelines.shadow_draw_bgl, 1024);
        let flat_pool = GpuUniformPool::new_flat(&device, 2048);
        let outline_pool = GpuUniformPool::new_outline(&device, 1024);
        let texture_cache = TextureCache::new(&device, &queue);
        let shadow_compare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow compare"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::Less),
            ..Default::default()
        });
        let (shadow_depth_tex, shadow_depth_view, shadow_bind_group) =
            Self::create_shadow_map_resources(&device, &pipelines, &shadow_compare_sampler, 1);
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Post tonemap FXAA"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/post_tonemap_fxaa.wgsl").into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit LDR to swapchain"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit_tex.wgsl").into()),
        });
        let post_process_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Post process BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let post_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Post PLL"),
            bind_group_layouts: &[&post_process_bgl],
            push_constant_ranges: &[],
        });
        let post_process_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Post tonemap FXAA"),
            layout: Some(&post_pll),
            vertex: wgpu::VertexState {
                module: &post_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &post_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let blit_ldr_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit post LDR to swapchain"),
            layout: Some(&post_pll),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let post_process_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Post process sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let ibl_path = Path::new("test_data/studio.hdr");
        let (idl, isl) = ibl_from_image_path(ibl_path).unwrap_or((Vec3::splat(0.07), Vec3::splat(0.15)));
        let ibl_diffuse = [idl.x, idl.y, idl.z, 1.0];
        let ibl_specular = [isl.x, isl.y, isl.z, 1.0];

        let mut renderer = Self {
            device,
            queue,
            surface,
            config,
            pipelines,
            phong_pool,
            shadow_pool,
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
            outline_width: 0.022,
            outline_color: [0.0, 0.0, 0.0, 1.0],
            cluster_renderer: None,
            cluster_cache: HashMap::new(),
            hzb: None,
            hzb_baker: None,
            cluster_pipeline_generation: 0,
            depth_reversed_z_mismatch_warned: false,
            texture_cache,
            ibl_diffuse,
            ibl_specular,
            shadow_compare_sampler,
            shadow_depth_tex,
            shadow_depth_view,
            shadow_bind_group,
            shadow_map_resolution: 1,
            hdr_post_processing: false,
            post_process_bgl,
            post_process_sampler,
            post_process_pipeline,
            blit_ldr_pipeline,
            hdr_scene_tex: None,
            hdr_scene_view: None,
            post_process_bind_group: None,
            post_ldr_tex: None,
            post_ldr_view: None,
            blit_ldr_bind_group: None,
        };
        renderer.hud = Some(HudRenderer::new(
            &renderer.device,
            &renderer.queue,
            renderer.config.format,
            renderer.config.width,
            renderer.config.height,
        ));
        renderer.create_depth_texture();
        renderer.hzb = Some(HzbPyramids::new(
            &renderer.device,
            renderer.config.width,
            renderer.config.height,
        ));
        renderer.hzb_baker = Some(HzbBaker::new(&renderer.device));
        renderer
    }

    pub(super) fn create_depth_texture(&mut self) {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let texture = self.device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_only = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Depth Only"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        });
        self.depth_texture = Some((texture, view, depth_only));
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.create_depth_texture();
            self.hzb = Some(HzbPyramids::new(&self.device, width, height));
            self.hzb_baker = Some(HzbBaker::new(&self.device));
            if let Some(hud) = &mut self.hud {
                hud.resize(&self.queue, width, height);
            }
            if self.hdr_post_processing {
                self.ensure_hdr_scene_target();
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
                previous, self.adaptive_quality, frame_time_ms
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
        self.cluster_cache.clear();
    }

    pub fn render_draw_calls(&mut self, draw_calls: &[DrawCall], scene: &SceneGraph) -> FrameStats {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        if draw_calls.is_empty() {
            return FrameStats::default();
        }

        let first = &draw_calls[0];
        let vp = first.mvp * first.model_matrix.inverse();
        let frustum = Frustum::from_view_projection(vp);
        let visible: Vec<&DrawCall> = draw_calls
            .iter()
            .filter(|dc| dc.aabb.as_ref().map_or(true, |aabb| frustum.intersects_aabb(aabb)))
            .collect();

        if let Some(head) = visible.first() {
            let dz = head.depth_reversed_z;
            let inconsistent = visible.iter().any(|dc| dc.depth_reversed_z != dz);
            if inconsistent {
                if !self.depth_reversed_z_mismatch_warned {
                    log::warn!(
                        "visible draw calls disagree on depth_reversed_z; using first visible ({}) for pipelines and depth clears",
                        dz
                    );
                    self.depth_reversed_z_mismatch_warned = true;
                }
            } else {
                self.depth_reversed_z_mismatch_warned = false;
            }
        }

        let total_visible_triangles: u64 = visible
            .iter()
            .map(|dc| {
                if let Some(md) = dc.meshlet_data.as_ref() {
                    md.total_triangles as u64
                } else if let Some(indices) = dc.indices.as_ref() {
                    (indices.len() / 3) as u64
                } else {
                    (dc.vertices.len() / 3) as u64
                }
            })
            .sum();
        let enable_perf_mode = total_visible_triangles > PERFORMANCE_MODE_TRIANGLE_THRESHOLD;
        if enable_perf_mode != self.performance_mode_active {
            self.performance_mode_active = enable_perf_mode;
            if enable_perf_mode {
                log::warn!("Performance mode enabled: triangle_count={}", total_visible_triangles);
            } else {
                log::info!("Performance mode disabled");
            }
        }

        let mut mesh_handles: Vec<Option<crate::gpu_resource::MeshId>> = Vec::with_capacity(visible.len());
        for dc in &visible {
            let handle = if dc.vertices.is_empty() && dc.meshlet_data.is_none() {
                None
            } else if dc.meshlet_data.is_some() {
                // Meshlet path: upload as standard mesh using meshlet-expanded buffers
                let md = dc.meshlet_data.as_ref().unwrap();
                let ptr = Arc::as_ptr(md) as u64;
                if let Some((mesh_id, last_used)) = self.mesh_cache.get_mut(&ptr) {
                    *last_used = self.frame_counter;
                    Some(*mesh_id)
                } else {
                    if self.mesh_cache.len() >= MESH_CACHE_MAX {
                        self.prune_mesh_cache();
                    }
                    // Convert meshlet vertices to standard Vertex format
                    let verts: Vec<crate::vertex::Vertex> = md.vertices.iter().map(|mv| crate::vertex::Vertex {
                        position: mv.position,
                        normal: mv.normal,
                        texcoord: mv.texcoord,
                    }).collect();
                    let mesh_id = self.gpu_meshes.upload_mesh(
                        &self.device,
                        &verts,
                        Some(&md.indices),
                        &[],
                    );
                    self.mesh_cache.insert(ptr, (mesh_id, self.frame_counter));
                    Some(mesh_id)
                }
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
                    if self.mesh_cache.len() >= MESH_CACHE_MAX {
                        self.prune_mesh_cache();
                    }
                    let mesh_id = self.gpu_meshes.upload_mesh(
                        &self.device,
                        &dc.vertices,
                        dc.indices.as_ref().map(|a| a.as_slice()),
                        &dc.edge_positions,
                    );
                    self.mesh_cache.insert(hash, (mesh_id, self.frame_counter));
                    Some(mesh_id)
                }
            };
            mesh_handles.push(handle);
        }

        let mut solid_order: Vec<usize> = (0..visible.len())
            .filter(|&i| !visible[i].vertices.is_empty() || visible[i].meshlet_data.is_some())
            .collect();
        solid_order.sort_by_key(|&i| {
            let dc = visible[i];
            (
                sort_keys::vec4_array_sort_key(dc.light_dirs),
                sort_keys::vec4_array_sort_key(dc.light_colors),
                sort_keys::vec4_array_sort_key(dc.light_types),
                sort_keys::vec4_array_sort_key(dc.light_positions),
                sort_keys::vec4_array_sort_key(dc.spot_params),
                dc.light_count,
                sort_keys::display_mode_sort_key(dc.display_mode),
                sort_keys::color_sort_key([dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]),
                sort_keys::color_sort_key([dc.ambient_color.x, dc.ambient_color.y, dc.ambient_color.z, 1.0]),
                sort_keys::color_sort_key([dc.specular_color.x, dc.specular_color.y, dc.specular_color.z, 1.0]),
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
                sort_keys::display_mode_sort_key(dc.display_mode),
                sort_keys::color_sort_key(dc.overlay_color.unwrap_or([0.0, 0.0, 0.0, 0.5])),
                mesh_handles[i].map(|m| m.data().as_ffi()).unwrap_or(0),
            )
        });
        let mut selected_order: Vec<usize> = (0..visible.len())
            .filter(|&i| visible[i].selected && (!visible[i].vertices.is_empty() || visible[i].meshlet_data.is_some()))
            .collect();
        selected_order.sort_by_key(|&i| {
            let dc = visible[i];
            (
                sort_keys::display_mode_sort_key(dc.display_mode),
                mesh_handles[i].map(|m| m.data().as_ffi()).unwrap_or(0),
            )
        });

        // Collect meshlet visible indices and upload ClusterSets
        let mut meshlet_indices: Vec<usize> = Vec::new();
        for (i, dc) in visible.iter().enumerate() {
            if dc.meshlet_data.is_some() {
                meshlet_indices.push(i);
                let md = dc.meshlet_data.as_ref().unwrap();
                let ptr = Arc::as_ptr(md) as u64;
                if !self.cluster_cache.contains_key(&ptr) {
                    let cs = ClusterSet::from_meshlet_data(&self.device, md);
                    self.cluster_cache.insert(ptr, cs);
                }
            }
        }
        if !meshlet_indices.is_empty() {
            if self.cluster_pipeline_generation != CLUSTER_PIPELINE_GENERATION {
                self.cluster_renderer = None;
                self.cluster_pipeline_generation = CLUSTER_PIPELINE_GENERATION;
            }
            if self.cluster_renderer.is_none() {
                self.cluster_renderer = Some(ClusterRenderer::new(&self.device));
            }
        }

        self.phong_pool.reset();
        self.shadow_pool.reset();
        self.flat_pool.reset();
        self.outline_pool.reset();

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
        let run_outline = !self.performance_mode_active
            && self.adaptive_quality == AdaptiveQuality::High
            && (mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine);

        let solid_wants_shadow = !self.performance_mode_active
            && matches!(
                mode,
                DisplayMode::Shaded | DisplayMode::ShadedWithEdges | DisplayMode::HiddenLine
            );

        let mut light_view_proj = Mat4::IDENTITY;
        let mut shadow_params = [0.0_f32, 0.0004, 0.0, 0.0];
        let mut run_shadow_pass = false;

        if solid_wants_shadow {
            if let Some(dir) = primary_directional_light_dir(scene) {
                let aabb = aabb_from_scene(scene).or_else(|| union_draw_call_aabbs(visible.iter().copied()));
                if let Some(aabb) = aabb {
                    let sm_size = match self.adaptive_quality {
                        AdaptiveQuality::High => 2048,
                        AdaptiveQuality::Medium => 1024,
                        AdaptiveQuality::Low => 512,
                    };
                    self.ensure_shadow_map(sm_size);
                    light_view_proj = directional_light_view_proj(dir, &aabb, 8.0);
                    let inv = 1.0 / sm_size as f32;
                    let (bias, pcf) = match self.adaptive_quality {
                        AdaptiveQuality::High => (0.00015_f32, 2.0_f32),
                        AdaptiveQuality::Medium => (0.00028, 1.0),
                        AdaptiveQuality::Low => (0.00045, 0.0),
                    };
                    shadow_params = [inv, bias, pcf, 1.0];
                    run_shadow_pass = true;
                }
            }
        }

        let ctx = PassContext {
            visible: &visible,
            solid_order: &solid_order,
            edge_order: &edge_order,
            selected_order: &selected_order,
            mesh_handles: &mesh_handles,
            mode,
            run_outline,
            bg_color: wgpu::Color { r: 0.08, g: 0.08, b: 0.08, a: 1.0 },
            performance_mode_active: self.performance_mode_active,
            wireframe_supported: self.wireframe_supported,
            adaptive_quality: self.adaptive_quality,
            outline_width: self.outline_width,
            outline_color: self.outline_color,
            meshlet_indices: &meshlet_indices,
            camera_pos: [first.camera_pos.x, first.camera_pos.y, first.camera_pos.z],
            depth_reversed_z: first.depth_reversed_z,
            light_view_proj,
            shadow_params,
            run_shadow_pass,
        };

        render_passes::execute_passes(self, &ctx, draw_calls, self.frame_counter)
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
            hud.update_text(&self.device, &self.queue, fps, frame_time_ms, stats, &hud_mode_name);
        }
    }

    fn get_mesh(&self, mesh_id: crate::gpu_resource::MeshId) -> Option<&GpuMesh> {
        self.gpu_meshes.get(mesh_id)
    }

    pub(super) fn draw_mesh_batched(
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

    pub(super) fn draw_edges_batched(
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

    pub(super) fn bind_and_draw_edges(
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
        if !dc.edge_positions.is_empty() {
            let line_verts: Vec<LineVertex> = dc.edge_positions.iter().map(|&p| LineVertex { position: p }).collect();
            let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Edge VB"),
                contents: bytemuck::cast_slice(&line_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.draw(0..line_verts.len() as u32, 0..1);
        }
    }

    pub(super) fn prune_mesh_cache(&mut self) {
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
