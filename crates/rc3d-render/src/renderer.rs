use std::path::{Path, PathBuf};

use wgpu::util::DeviceExt;

#[path = "renderer_types.rs"]
mod renderer_types;
#[path = "renderer_helpers.rs"]
mod renderer_helpers;
#[path = "renderer_render.rs"]
mod renderer_render;
#[path = "renderer_skinning.rs"]
mod renderer_skinning;
#[path = "renderer_internals.rs"]
mod renderer_internals;

pub use renderer_types::*;

use crate::adaptive_quality::AdaptiveQuality;
use crate::asset_manager::GpuAssetManager;
use crate::auto_exposure::AutoExposure;
use crate::cluster_lighting::{ClusterLightCuller, ClusterLightResources};
use crate::color_grading::ColorGradingPass;
use crate::dof_pass::DofPass;
use crate::gpu_resource::{GpuResourceManager, GpuUniformPool};
use crate::hud::HudRenderer;
use crate::material_library::MaterialLibrary;
use crate::hzb::{HzbBaker, HzbPyramids};
use crate::motion_blur::MotionBlurPass;
use crate::pipeline_cache::PipelineCacheManager;
use crate::pipelines::PipelineSet;
use crate::post_processor::{self};
use crate::shadow_omni::OmniShadowRenderer;
use crate::shadow_pass::{self};
use crate::shader_permutation::ShaderVariantCache;
use crate::shader_reload::ShaderHotReload;
use crate::vertex::{InstanceData, MAX_INSTANCES};
use crate::viewport::ViewportLayout;
use crate::ssr_pass::SsrPass;
use crate::taa::{TaaJitter, TaaPass};
use crate::texture_cache::TextureCache;
use crate::volumetric_fog::VolumetricFogPass;
use crate::ibl::IblPreset;
use self::renderer_internals::{FrameState, GpuInternals};
use crate::settings::RenderSettings;
use glam::{Mat4, Vec3};
use rc3d_core::DisplayMode;

const PERFORMANCE_MODE_TRIANGLE_THRESHOLD: u64 = 5_000_000;
const CLUSTER_PIPELINE_GENERATION: u32 = 5;

fn resolve_studio_hdr_path() -> PathBuf {
    if let Ok(custom) = std::env::var("RC3D_STUDIO_HDR") {
        let custom_path = PathBuf::from(custom);
        if custom_path.is_file() {
            return custom_path;
        }
    }

    let cwd_candidate = PathBuf::from("test_data").join("studio.hdr");
    if cwd_candidate.is_file() {
        return cwd_candidate;
    }

    let repo_candidate = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("studio.hdr");
    if repo_candidate.is_file() {
        return repo_candidate;
    }

    PathBuf::new()
}

pub struct Renderer {
    // ── One-time config ──
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    wireframe_supported: bool,

    // ── Settings tier ──
    pub(crate) settings: RenderSettings,

    // Feature toggles (kept pub for transition)
    pub enable_taa: bool,
    pub enable_motion_blur: bool,
    pub enable_ssr: bool,
    pub enable_color_grading: bool,
    pub enable_dof: bool,
    pub enable_volumetric_fog: bool,
    pub enable_cluster_lights: bool,
    pub enable_omni_shadows: bool,
    pub enable_ldr_fxaa: bool,
    pub hdr_post_processing: bool,
    pub global_display_mode: DisplayMode,
    pub grid_enabled: bool,
    pub hud_enabled: bool,
    pub outline_width: f32,
    pub outline_color: [f32; 4],
    pub xray_mode: bool,
    pub screen_space_selection_outline: bool,
    pub ibl_preset: IblPreset,

    /// When true, the renderer skips expensive passes (shadows, edges, SSAO)
    /// even below the triangle threshold. Set by the app during camera orbit/pan/zoom.
    pub interaction_active: bool,

    // ── Frame state tier ──
    pub(crate) frame: FrameState,

    // ── Profiler tier ──
    pub gpu_timer: crate::profiler::GpuTimer,
    pub cpu_span: crate::profiler::CpuSpanCollector,

    // ── Render cache tier ──
    pub draw_cache: crate::flat_draw_cache::FlatDrawCache,
    pub texture_table: crate::global_tables::TexturePathTable,
    pub texture_streamer: crate::texture_streaming::TextureStreamer,

    // ── Material bind group cache ──
    /// Cached PBR material bind group, keyed by hash of (albedo, normal, mr, emissive, occlusion) texture IDs.
    pub(crate) last_material_bg_key: u64,
    pub(crate) last_material_bg: Option<wgpu::BindGroup>,

    // ── Light-set table (populated by traversal, consumed by passes) ──
    pub(crate) light_sets: crate::light_set::LightSetTable,

    // ── GPU internals tier ──
    pub(crate) gpu: GpuInternals,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveControl {
    Disabled,
    Locked,
    Dynamic { allow_downgrade: bool },
}

impl Renderer {
    pub fn ensure_decal_pass(&mut self) {
        if self.gpu.decal_pass.is_none() {
            self.gpu.decal_pass = Some(crate::render_passes::pass_effects::DecalPass::new(
                &self.device, self.config.format,
            ));
        }
    }
    pub fn ensure_volume_pass(&mut self) {
        if self.gpu.volume_pass.is_none() {
            self.gpu.volume_pass = Some(crate::render_passes::pass_effects::VolumePass::new(
                &self.device, self.config.format,
            ));
        }
    }
    pub fn ensure_point_cloud_pass(&mut self) {
        if self.gpu.point_cloud_pass.is_none() {
            self.gpu.point_cloud_pass = Some(crate::render_passes::pass_effects::PointCloudPass::new(
                &self.device, self.config.format,
            ));
        }
    }

    /// Set effect commands for this frame (called by app layer after traversal).
    pub fn set_effect_commands(&mut self, cmds: crate::render_passes::pass_effects::EffectCommands) {
        self.frame.effect_commands = cmds;
    }

    fn ensure_csm_shadow(&mut self, resolution: u32, cascade_count: u32) {
        let resolution = resolution.max(1);
        let cascade_count = cascade_count.max(1);
        if let Some(ref csm) = self.gpu.csm_shadow {
            if csm.resolution == resolution && csm.cascade_count == cascade_count {
                return;
            }
        }
        self.gpu.csm_shadow = Some(shadow_pass::create_csm_shadow_resources(
            &self.device,
            &self.gpu.pipelines,
            &self.gpu.shadow_compare_sampler,
            resolution,
            cascade_count,
        ));
    }

    pub fn set_hdr_post_processing(&mut self, enabled: bool) {
        self.hdr_post_processing = enabled;
        if !enabled {
            self.gpu.post_fx = None;
        } else {
            self.ensure_post_fx_targets();
        }
    }

    pub(super) fn ensure_post_fx_targets(&mut self) {
        if !self.hdr_post_processing {
            return;
        }
        let w = self.config.width.max(1);
        let h = self.config.height.max(1);
        if let Some(ref fx) = self.gpu.post_fx {
            if fx.hdr_tex.size().width == w && fx.hdr_tex.size().height == h {
                return;
            }
        }
        // Create a 1x1 black texture for dummy slots
        let (black_tex, black_view) = self.gpu.texture_cache.black_placeholder(&self.device);
        self.gpu.post_fx = Some(post_processor::ensure_post_fx_textures(
            &self.device, &self.gpu.post_fx_pipelines, w, h, &black_tex, &black_view,
        ));
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

    pub fn create_vertex_buffer<T: bytemuck::Pod>(&self, data: &[T], label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    pub fn device_limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn device_clone(&self) -> wgpu::Device {
        self.device.clone()
    }

    pub fn queue_clone(&self) -> wgpu::Queue {
        self.queue.clone()
    }

    pub fn surface_size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn acquire_surface_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }

    pub fn last_diagnostics(&self) -> Option<&FrameDiagnostics> {
        self.frame.last_diagnostics.as_ref()
    }

    pub fn frame_counter(&self) -> u64 {
        self.frame.frame_counter
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

        let requested_features = wgpu::Features::POLYGON_MODE_LINE
            | wgpu::Features::DEPTH32FLOAT_STENCIL8
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::PIPELINE_CACHE;
        let features = adapter.features() & requested_features;
        let wireframe_supported = features.contains(wgpu::Features::POLYGON_MODE_LINE);
        let timing_supported = features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut shader_cache = ShaderVariantCache::new();
        let pipelines = PipelineSet::create(&device, config.format, &mut shader_cache);
        let selection_outline_pipelines =
            crate::selection_outline::SelectionOutlinePipelines::new(&device, &pipelines.flat_bgl);
        let phong_pool = GpuUniformPool::new_phong(&device, 65536);
        let shadow_pool = GpuUniformPool::new_shadow_pool(&device, &pipelines.shadow_draw_bgl, 32768);
        let flat_pool = GpuUniformPool::new_flat(&device, 32768);
        let section_cap_pool = GpuUniformPool::new_section_cap(&device, &pipelines.flat_bgl, 16384);
        let outline_pool = GpuUniformPool::new_outline(&device, 16384);
        let texture_cache = TextureCache::new(&device, &queue);
        let shadow_compare_sampler = shadow_pass::create_shadow_compare_sampler(&device);
        let csm_shadow = Some(shadow_pass::create_csm_shadow_resources(
            &device,
            &pipelines,
            &shadow_compare_sampler,
            1,
            1,
        ));
        let post_fx_pipelines = post_processor::create_post_fx_pipelines(&device, config.format);
        let (ssao_noise_tex, ssao_noise_view) = post_processor::create_ssao_noise(&device, &queue);
        let instance_stride = std::mem::size_of::<InstanceData>() as u64;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance data SSBO"),
            size: instance_stride * MAX_INSTANCES as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // IBL: load envmap + BRDF LUT, compute diffuse/specular constants
        let ibl_path = resolve_studio_hdr_path();
        let ibl_preset = IblPreset::Studio;
        // Create IBL resources (envmap + BRDF LUT textures)
        let ibl_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("IBL sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let ibl_res = crate::ibl::IblResources::new(
            &device, &queue,
            &crate::ibl::create_ibl_bind_group_layout(&device),
            &ibl_sampler, ibl_path.as_path(), ibl_preset,
        );
        let ibl_diffuse = ibl_res.ibl_diffuse;
        let ibl_specular = ibl_res.ibl_specular;

        // Combined IBL + instance bind group (group 3, 7 bindings)
        let morph_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Morph dummy buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let morph_params_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Morph params dummy"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&morph_params_dummy, 0, &[0u8; 16]);

        let ibl_instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL + Instance BG"),
            layout: &pipelines.ibl_instance_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&ibl_res.env_map_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ibl_res.brdf_lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&ibl_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: instance_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: morph_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: morph_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: morph_params_dummy.as_entire_binding() },
            ],
        });

        // Clone before move into struct (wgpu objects are Arc-internally cheap to clone)
        let auto_exposure = AutoExposure::new(&device, config.width, config.height);
        let gpu_timer = crate::profiler::GpuTimer::new(&device, 32, queue.get_timestamp_period());
        let mut renderer = Self {
            device,
            queue,
            surface,
            config,
            wireframe_supported,
            settings: RenderSettings::default(),
            // Feature toggles
            enable_taa: false,
            enable_motion_blur: false,
            enable_ssr: false,
            enable_color_grading: false,
            enable_dof: false,
            enable_volumetric_fog: false,
            enable_cluster_lights: true,
            enable_omni_shadows: true,
            enable_ldr_fxaa: true,
            hdr_post_processing: false,
            global_display_mode: DisplayMode::ShadedWithEdges,
            grid_enabled: false,
            hud_enabled: true,
            outline_width: 0.022,
            outline_color: [1.0, 0.5, 0.0, 1.0],
            xray_mode: false,
            screen_space_selection_outline: true,
            ibl_preset,
            interaction_active: false,
            // Profiler
            gpu_timer,
            cpu_span: crate::profiler::CpuSpanCollector::default(),
            // Render cache
            draw_cache: crate::flat_draw_cache::FlatDrawCache::new(),
            texture_table: crate::global_tables::TexturePathTable::new(),
            texture_streamer: crate::texture_streaming::TextureStreamer::new(),
            light_sets: crate::light_set::LightSetTable::new(),
            last_material_bg_key: 0,
            last_material_bg: None,
            // Frame state
            frame: FrameState {
                markup_vertices: Vec::new(),
                clip_planes: Vec::new(),
                section_cap_tints: Vec::new(),
                scene_vp: Mat4::IDENTITY,
                scene_camera_pos: Vec3::ZERO,
                animation_time_sec: 0.0,
                frame_counter: 0,
                performance_mode_active: false,
                last_diagnostics: None,
                last_hud_update_frame: 0,
                viewport_layout: ViewportLayout::new(),
                frame_stats: FrameStats::default(),
                effect_commands: crate::render_passes::pass_effects::EffectCommands::default(),
                cached_bvh: None,
                has_text_nodes: true,    // optimistic; auto-disabled after 2 empty frames
                has_effect_nodes: true,  // optimistic; auto-disabled after 2 empty frames
                bvh_out: Vec::with_capacity(1024),
                visible_indices: Vec::with_capacity(1024),
                solid_order_buf: Vec::with_capacity(1024),
                edge_order_buf: Vec::with_capacity(256),
                selected_order_buf: Vec::with_capacity(64),
                transparent_order_buf: Vec::with_capacity(128),
                light_hashes_buf: Vec::with_capacity(1024),
                meshlet_indices_buf: Vec::with_capacity(256),
            },
            // GPU internals
            gpu: GpuInternals {
                pipelines,
                pbr_variant_cache: crate::pipelines::PbrVariantCache::new(),
                phong_pool,
                shadow_pool,
                flat_pool,
                section_cap_pool,
                outline_pool,
                gpu_meshes: GpuResourceManager::new(),
                gpu_skinning_pass: None,
                skinned_mesh_resources: std::collections::HashMap::new(),
                depth_texture: None,
                assets: GpuAssetManager::new(),
                materials: MaterialLibrary::new(),
                hud: None,
                adaptive_quality: AdaptiveQuality::High,
                adaptive_frame_time_ema_ms: 16.7,
                adaptive_switch_cooldown_frames: 0,
                cluster_renderer: None,
                hzb: None,
                hzb_baker: None,
                cluster_pipeline_generation: 0,
                depth_reversed_z_mismatch_warned: false,
                texture_cache,
                ibl_diffuse,
                ibl_specular,
                shadow_compare_sampler,
                csm_shadow,
                post_fx_pipelines,
                post_fx: None,
                ssao_noise_tex,
                ssao_noise_view,
                instance_buffer,
                ibl_instance_bind_group,
                timing_supported,
                pipeline_cache: None,
                shader_cache,
                shader_reload: ShaderHotReload::new(),
                auto_exposure,
                taa_pass: None,
                taa_jitter: TaaJitter::new(),
                motion_blur: None,
                ssr_pass: None,
                color_grading: None,
                dof_pass: None,
                volumetric_fog: None,
                cluster_lights: None,
                cluster_light_culler: None,
                omni_shadow: None,
                decal_pass: None,
                volume_pass: None,
                point_cloud_pass: None,
                selection_outline_pipelines: Some(selection_outline_pipelines),
                selection_outline_targets: None,
                ldr_shade_tex: None,
                ldr_shade_view: None,
                transform_buffer: None,
                indirect_args_buffer: None,
                instance_indices_buffer: None,
                gpu_cull_pass: None,
                gpu_cull_bg: None,
                frustum_uniform: None,
                gpu_cull_enabled: false,
                max_gpu_cull_objects: 65536,
            },
        };
        renderer.gpu.hud = Some(HudRenderer::new(
            &renderer.device,
            &renderer.queue,
            renderer.config.format,
            renderer.config.width,
            renderer.config.height,
        ));
        renderer.create_depth_texture();
        renderer.gpu.hzb_baker = Some(HzbBaker::new(&renderer.device));
        let ds_bgl = &renderer.gpu.hzb_baker.as_ref().unwrap().downsample_bgl;
        renderer.gpu.hzb = Some(HzbPyramids::new(
            &renderer.device,
            ds_bgl,
            renderer.config.width,
            renderer.config.height,
        ));

        // ── Phase 2: post-processing passes ──
        renderer.gpu.taa_pass = Some(TaaPass::new(&renderer.device));
        renderer.gpu.motion_blur = Some(MotionBlurPass::new(&renderer.device));
        renderer.gpu.ssr_pass = Some(SsrPass::new(&renderer.device));
        renderer.gpu.color_grading = Some(ColorGradingPass::new(&renderer.device, &renderer.queue));
        renderer.gpu.dof_pass = Some(DofPass::new(&renderer.device));
        renderer.gpu.volumetric_fog = Some(VolumetricFogPass::new(&renderer.device));

        // ── Phase 2: lighting ──
        renderer.gpu.cluster_light_culler = Some(ClusterLightCuller::new(&renderer.device));
        renderer.gpu.cluster_lights = Some(ClusterLightResources::new(&renderer.device));
        renderer.gpu.omni_shadow = Some(OmniShadowRenderer::new(&renderer.device));

        // ── Pipeline cache ──
        let cache_dir = std::path::Path::new("cache");
        renderer.gpu.pipeline_cache = Some(PipelineCacheManager::new(&renderer.device, cache_dir));

        // ── Shader hot-reload: watch shaders directory ──
        let shaders_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
        renderer.gpu.shader_reload.watch_directory(&shaders_dir);

        // ── Material library: set BGL for bind group building ──
        renderer.gpu.materials.set_bind_group_layout(&renderer.gpu.pipelines.pbr_material_bgl);

        // ── Multi-viewport layout ──
        renderer.frame.viewport_layout.rebuild(renderer.config.width, renderer.config.height);

        renderer
    }

    pub(super) fn create_depth_texture_at(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        let size = wgpu::Extent3d {
            width,
            height,
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
        self.gpu.depth_texture = Some((texture, view, depth_only));
    }

    pub(super) fn create_depth_texture(&mut self) {
        self.create_depth_texture_at(self.config.width, self.config.height);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.frame.viewport_layout.rebuild(width, height);
            self.create_depth_texture();
            self.gpu.selection_outline_targets = None;
            self.gpu.ldr_shade_tex = None;
            self.gpu.ldr_shade_view = None;
            self.gpu.hzb_baker = Some(HzbBaker::new(&self.device));
            let ds_bgl = &self.gpu.hzb_baker.as_ref().unwrap().downsample_bgl;
            self.gpu.hzb = Some(HzbPyramids::new(&self.device, ds_bgl, width, height));
            if let Some(hud) = &mut self.gpu.hud {
                hud.resize(&self.queue, width, height);
            }
            if self.hdr_post_processing {
                self.ensure_post_fx_targets();
            }
        }
    }

    /// Replace the light-set table (populated during scene traversal).
    pub fn set_light_sets(&mut self, table: crate::light_set::LightSetTable) {
        self.light_sets = table;
    }

    /// Enable GPU compute culling with buffers sized for max_objects.
    /// Allocates transform_buffer (storage), indirect_args (storage+indirect),
    /// and instance_indices (storage). Call once during setup.
    pub fn enable_gpu_culling(&mut self, max_objects: u64) {
        let stride = std::mem::size_of::<crate::vertex::GpuObjectTransform>() as u64;
        self.gpu.transform_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Cull Transforms"),
            size: stride * max_objects,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.gpu.indirect_args_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Cull Indirect"),
            size: std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64 * 4096,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));
        self.gpu.instance_indices_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Cull Indices"),
            size: 4 * max_objects,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        self.gpu.frustum_uniform = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Frustum Uniform"),
            size: 6 * 16, // 6 planes × vec4<f32>
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let cull_pass = crate::gpu_culling::GpuCullPass::new(&self.device);
        let bg = cull_pass.create_bind_group(
            &self.device,
            self.gpu.transform_buffer.as_ref().unwrap(),
            self.gpu.indirect_args_buffer.as_ref().unwrap(),
            self.gpu.frustum_uniform.as_ref().unwrap(),
            self.gpu.instance_indices_buffer.as_ref().unwrap(),
        );
        self.gpu.gpu_cull_pass = Some(cull_pass);
        self.gpu.gpu_cull_bg = Some(bg);
        self.gpu.gpu_cull_enabled = true;
        self.gpu.max_gpu_cull_objects = max_objects;
    }

    pub fn set_screen_space_selection_outline(&mut self, enabled: bool) {
        self.screen_space_selection_outline = enabled;
    }

    pub fn set_ldr_fxaa(&mut self, enabled: bool) {
        self.enable_ldr_fxaa = enabled;
        self.gpu.ldr_shade_tex = None;
        self.gpu.ldr_shade_view = None;
    }

    /// Batch-apply render settings (HOOPS HPS::RenderingMode style).
    pub fn apply_settings(&mut self, settings: RenderSettings) {
        self.hdr_post_processing = settings.post_effect.hdr;
        self.enable_taa = settings.post_effect.taa;
        self.enable_motion_blur = settings.post_effect.motion_blur;
        self.enable_ssr = settings.post_effect.ssr;
        self.enable_color_grading = settings.post_effect.color_grading;
        self.enable_dof = settings.post_effect.dof;
        self.enable_volumetric_fog = settings.post_effect.volumetric_fog;
        self.enable_ldr_fxaa = settings.post_effect.ldr_fxaa;
        self.enable_cluster_lights = settings.lighting.cluster_lights;
        self.enable_omni_shadows = settings.lighting.omni_shadows;
        self.set_ibl_preset(settings.lighting.ibl_preset);
        self.global_display_mode = settings.display.display_mode;
        self.grid_enabled = settings.display.grid_enabled;
        self.hud_enabled = settings.display.hud_enabled;
        self.outline_width = settings.display.outline_width;
        self.outline_color = settings.display.outline_color;
        self.xray_mode = settings.display.xray_mode;
        self.screen_space_selection_outline = settings.display.screen_space_selection_outline;
        if settings.display.vsync_enabled != self.is_vsync_enabled() {
            self.set_vsync(settings.display.vsync_enabled);
        }
        if !self.hdr_post_processing {
            self.gpu.post_fx = None;
        } else {
            self.ensure_post_fx_targets();
        }
        if !self.enable_ldr_fxaa {
            self.gpu.ldr_shade_tex = None;
            self.gpu.ldr_shade_view = None;
        }
        self.settings = settings;
    }

    /// Read-only snapshot of current settings.
    pub fn settings(&self) -> &RenderSettings {
        &self.settings
    }

    fn is_vsync_enabled(&self) -> bool {
        matches!(self.config.present_mode, wgpu::PresentMode::AutoVsync)
    }

    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        self.global_display_mode = mode;
    }

    /// RGBA for mesh outline pass, wireframe/edge-overlay defaults, and selection edge/bbox lines.
    pub fn set_outline_color(&mut self, rgba: [f32; 4]) {
        self.outline_color = rgba;
    }

    pub fn set_taa(&mut self, enabled: bool) {
        self.enable_taa = enabled;
    }

    pub fn set_motion_blur(&mut self, enabled: bool) {
        self.enable_motion_blur = enabled;
    }

    pub fn set_ssr(&mut self, enabled: bool) {
        self.enable_ssr = enabled;
    }

    pub fn set_color_grading(&mut self, enabled: bool) {
        self.enable_color_grading = enabled;
    }

    pub fn set_dof(&mut self, enabled: bool) {
        self.enable_dof = enabled;
    }

    pub fn set_volumetric_fog(&mut self, enabled: bool) {
        self.enable_volumetric_fog = enabled;
    }

    pub fn set_cluster_lights(&mut self, enabled: bool) {
        self.enable_cluster_lights = enabled;
    }

    pub fn set_omni_shadows(&mut self, enabled: bool) {
        self.enable_omni_shadows = enabled;
    }

    pub fn set_grid_enabled(&mut self, enabled: bool) {
        self.grid_enabled = enabled;
    }

    pub fn set_hud_enabled(&mut self, enabled: bool) {
        self.hud_enabled = enabled;
    }

    pub fn set_materials(&mut self, materials: MaterialLibrary) {
        self.gpu.materials = materials;
    }

    pub fn set_xray_mode(&mut self, enabled: bool) {
        self.xray_mode = enabled;
    }

    pub fn set_outline_width(&mut self, width: f32) {
        self.outline_width = width;
    }

    // ── Internal accessors for render_passes (pub(crate)) ──
    #[allow(dead_code)]
    pub fn viewport_layout(&self) -> &ViewportLayout {
        &self.frame.viewport_layout
    }
    pub fn viewport_layout_mut(&mut self) -> &mut ViewportLayout {
        &mut self.frame.viewport_layout
    }

    /// Update post-processing effect parameters at runtime.
    pub fn set_post_effect_params(&mut self, vignette: f32, chromatic: f32, bloom_str: f32, grain: f32) {
        let params = crate::post_processor::PostEffectParams { vignette, chromatic, bloom_str, grain };
        self.queue.write_buffer(&self.gpu.post_fx_pipelines.post_params_buf, 0, bytemuck::bytes_of(&params));
    }

    pub fn ibl_preset_name(&self) -> &'static str {
        self.ibl_preset.name()
    }

    pub fn cycle_ibl_preset(&mut self) {
        let next = self.ibl_preset.next();
        self.set_ibl_preset(next);
    }

    pub fn set_ibl_preset(&mut self, preset: IblPreset) {
        self.ibl_preset = preset;
        let ibl_path = resolve_studio_hdr_path();
        let ibl_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("IBL sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let ibl_res = crate::ibl::IblResources::new(
            &self.device,
            &self.queue,
            &crate::ibl::create_ibl_bind_group_layout(&self.device),
            &ibl_sampler,
            ibl_path.as_path(),
            preset,
        );
        self.gpu.ibl_diffuse = ibl_res.ibl_diffuse;
        self.gpu.ibl_specular = ibl_res.ibl_specular;
        let morph_dummy = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Morph dummy buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let morph_params_dummy = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Morph params dummy"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&morph_params_dummy, 0, &[0u8; 16]);
        self.gpu.ibl_instance_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL + Instance BG"),
            layout: &self.gpu.pipelines.ibl_instance_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&ibl_res.env_map_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ibl_res.brdf_lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&ibl_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.gpu.instance_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: morph_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: morph_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: morph_params_dummy.as_entire_binding() },
            ],
        });
        log::info!("IBL preset switched to {}", preset.name());
    }

    pub fn display_mode(&self) -> DisplayMode {
        self.global_display_mode
    }

    /// Current scene view-projection matrix (set during last render).
    pub fn frame_vp(&self) -> glam::Mat4 {
        self.frame.scene_vp
    }

    pub fn performance_mode_active(&self) -> bool {
        self.frame.performance_mode_active
    }

    pub fn adaptive_quality_name(&self) -> &'static str {
        self.gpu.adaptive_quality.name()
    }

    pub fn adaptive_is_low(&self) -> bool {
        self.gpu.adaptive_quality.is_low()
    }

    pub fn report_frame_time_ms(&mut self, frame_time_ms: f32, control: AdaptiveControl) {
        if matches!(control, AdaptiveControl::Disabled) {
            self.gpu.adaptive_quality = AdaptiveQuality::High;
            self.gpu.adaptive_switch_cooldown_frames = 0;
            self.gpu.adaptive_frame_time_ema_ms = frame_time_ms.max(0.0);
            return;
        }
        if matches!(control, AdaptiveControl::Locked) {
            return;
        }
        // Smooth short spikes and enforce a brief cooldown after each switch
        // to prevent quality oscillation on borderline frame times.
        let alpha = 0.12_f32;
        self.gpu.adaptive_frame_time_ema_ms =
            self.gpu.adaptive_frame_time_ema_ms + (frame_time_ms - self.gpu.adaptive_frame_time_ema_ms) * alpha;

        if self.gpu.adaptive_switch_cooldown_frames > 0 {
            self.gpu.adaptive_switch_cooldown_frames -= 1;
            return;
        }

        let previous = self.gpu.adaptive_quality;
        let mut next = self.gpu.adaptive_quality.update(self.gpu.adaptive_frame_time_ema_ms);
        if let AdaptiveControl::Dynamic { allow_downgrade: false } = control {
            if next as u8 > previous as u8 {
                next = previous;
            }
        }
        self.gpu.adaptive_quality = next;
        if previous != self.gpu.adaptive_quality {
            self.gpu.adaptive_switch_cooldown_frames = 30;
            log::warn!(
                "Adaptive quality changed: {:?} -> {:?} (frame_ms={:.2}, ema_ms={:.2})",
                previous, self.gpu.adaptive_quality, frame_time_ms, self.gpu.adaptive_frame_time_ema_ms
            );
        }
    }

    pub fn set_clip_planes(&mut self, planes: Vec<[f32; 4]>, mut cap_tints: Vec<Option<[f32; 4]>>) {
        if cap_tints.len() < planes.len() {
            cap_tints.resize(planes.len(), None);
        } else if cap_tints.len() > planes.len() {
            cap_tints.truncate(planes.len());
        }
        self.frame.clip_planes = planes;
        self.frame.section_cap_tints = cap_tints;
    }

    pub fn collect_markup_vertices(&mut self, graph: &rc3d_scene::SceneGraph, root: rc3d_core::NodeId) {
        self.frame.markup_vertices = crate::render_passes::pass_markup::collect_markup_lines(
            graph,
            root,
            self.config.width,
            self.config.height,
        );
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
        &self.frame.clip_planes
    }

    /// Shared-device RGBA8 offscreen target for screenshots or readback (same `Device` / `Queue` as the main surface).
    pub fn offscreen_target_rgba8(&self, width: u32, height: u32) -> crate::offscreen::OffscreenTarget {
        crate::offscreen::OffscreenTarget::new_with_device(&self.device, &self.queue, width, height)
    }

    pub fn toggle_clip_plane(&mut self, axis: usize) {
        let normal = match axis {
            0 => [1.0, 0.0, 0.0, 0.0],
            1 => [0.0, 1.0, 0.0, 0.0],
            2 => [0.0, 0.0, 1.0, 0.0],
            _ => return,
        };
        if let Some(pos) = self.frame.clip_planes.iter().position(|p| p[0] == normal[0] && p[1] == normal[1] && p[2] == normal[2]) {
            self.frame.clip_planes.remove(pos);
            if pos < self.frame.section_cap_tints.len() {
                self.frame.section_cap_tints.remove(pos);
            }
        } else {
            self.frame.clip_planes.push(normal);
            self.frame.section_cap_tints.push(None);
        }
    }

    pub fn invalidate_mesh_cache(&mut self) {
        self.gpu.assets.invalidate_all();
    }
}

