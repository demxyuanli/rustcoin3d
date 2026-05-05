use std::path::{Path, PathBuf};

#[path = "renderer_types.rs"]
mod renderer_types;
#[path = "renderer_helpers.rs"]
mod renderer_helpers;
#[path = "renderer_render.rs"]
mod renderer_render;
#[path = "renderer_skinning.rs"]
mod renderer_skinning;

pub use renderer_types::*;

use crate::adaptive_quality::AdaptiveQuality;
use crate::asset_manager::GpuAssetManager;
use crate::auto_exposure::AutoExposure;
use crate::cluster::ClusterRenderer;
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
use crate::post_processor::{self, PostFxPipelines, PostFxTextures};
use crate::shadow_omni::OmniShadowRenderer;
use crate::shadow_pass::{self, CsmShadowResources};
use crate::shader_permutation::ShaderVariantCache;
use crate::shader_reload::ShaderHotReload;
use crate::vertex::{InstanceData, MAX_INSTANCES};
use crate::viewport::ViewportLayout;
use crate::ssr_pass::SsrPass;
use crate::taa::{TaaJitter, TaaPass};
use crate::texture_cache::TextureCache;
use crate::volumetric_fog::VolumetricFogPass;
use crate::gpu_skinning::{GpuSkinningPass, GpuSkinningResources};
use crate::ibl::IblPreset;
use glam::{Mat4, Vec3};
use rc3d_core::DisplayMode;

const PERFORMANCE_MODE_TRIANGLE_THRESHOLD: u64 = 2_000_000;
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
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub pipelines: PipelineSet,
    pub phong_pool: GpuUniformPool,
    pub flat_pool: GpuUniformPool,
    pub outline_pool: GpuUniformPool,
    pub gpu_meshes: GpuResourceManager,
    pub gpu_skinning_pass: Option<GpuSkinningPass>,
    pub skinned_mesh_resources: std::collections::HashMap<crate::gpu_resource::MeshId, GpuSkinningResources>,
    pub animation_time_sec: f32,
    pub depth_texture: Option<(wgpu::Texture, wgpu::TextureView, wgpu::TextureView)>,
    pub global_display_mode: DisplayMode,
    pub clip_planes: Vec<[f32; 4]>,
    pub wireframe_supported: bool,
    pub assets: GpuAssetManager,
    pub materials: MaterialLibrary,
    pub frame_counter: u64,
    pub performance_mode_active: bool,
    pub hud: Option<HudRenderer>,
    pub hud_enabled: bool,
    pub(super) adaptive_quality: AdaptiveQuality,
    adaptive_frame_time_ema_ms: f32,
    adaptive_switch_cooldown_frames: u8,
    pub last_hud_update_frame: u64,
    pub outline_width: f32,
    pub outline_color: [f32; 4],
    pub cluster_renderer: Option<ClusterRenderer>,
    pub hzb: Option<HzbPyramids>,
    pub hzb_baker: Option<HzbBaker>,
    pub cluster_pipeline_generation: u32,
    depth_reversed_z_mismatch_warned: bool,
    pub texture_cache: TextureCache,
    pub ibl_diffuse: [f32; 4],
    pub ibl_specular: [f32; 4],
    pub ibl_preset: IblPreset,
    pub shadow_pool: GpuUniformPool,
    pub shadow_compare_sampler: wgpu::Sampler,
    pub csm_shadow: Option<CsmShadowResources>,
    pub hdr_post_processing: bool,
    pub post_fx_pipelines: PostFxPipelines,
    pub post_fx: Option<PostFxTextures>,
    pub ssao_noise_tex: wgpu::Texture,
    pub ssao_noise_view: wgpu::TextureView,
    pub instance_buffer: wgpu::Buffer,
    pub ibl_instance_bind_group: wgpu::BindGroup,
    pub timing_supported: bool,
    pub gpu_query_set: Option<wgpu::QuerySet>,
    pub gpu_query_buffer: Option<wgpu::Buffer>,
    pub gpu_query_slots: u32,
    pub gpu_query_period: f32, // timestamp period in ns
    // ── Phase 2 render features ──
    pub pipeline_cache: Option<PipelineCacheManager>,
    pub shader_cache: ShaderVariantCache,
    pub viewport_layout: ViewportLayout,
    pub grid_enabled: bool,
    /// Pre-collected markup line vertices for overlay rendering.
    pub markup_vertices: Vec<crate::vertex::LineVertex>,
    pub(crate) scene_vp: Mat4,
    pub(crate) scene_camera_pos: Vec3,
    pub shader_reload: ShaderHotReload,
    pub auto_exposure: AutoExposure,
    pub taa_pass: Option<TaaPass>,
    pub taa_jitter: TaaJitter,
    pub motion_blur: Option<MotionBlurPass>,
    pub ssr_pass: Option<SsrPass>,
    pub color_grading: Option<ColorGradingPass>,
    pub dof_pass: Option<DofPass>,
    pub volumetric_fog: Option<VolumetricFogPass>,
    pub cluster_lights: Option<ClusterLightResources>,
    pub cluster_light_culler: Option<ClusterLightCuller>,
    pub omni_shadow: Option<OmniShadowRenderer>,
    // Feature toggles
    pub enable_taa: bool,
    pub enable_motion_blur: bool,
    pub enable_ssr: bool,
    pub enable_color_grading: bool,
    pub enable_dof: bool,
    pub enable_volumetric_fog: bool,
    pub enable_cluster_lights: bool,
    pub enable_omni_shadows: bool,
    pub xray_mode: bool,
    pub last_diagnostics: Option<FrameDiagnostics>,
    pub selection_outline_pipelines: Option<crate::selection_outline::SelectionOutlinePipelines>,
    pub selection_outline_targets: Option<crate::selection_outline::SelectionOutlineTargets>,
    /// three.js `OutlinePass`-style screen-space selection outline.
    pub screen_space_selection_outline: bool,
    /// When HDR is off, render 3D to `ldr_shade_*` then FXAA into the swapchain (whole-frame AA).
    pub enable_ldr_fxaa: bool,
    pub(crate) ldr_shade_tex: Option<wgpu::Texture>,
    pub(crate) ldr_shade_view: Option<wgpu::TextureView>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveControl {
    Disabled,
    Locked,
    Dynamic { allow_downgrade: bool },
}

impl Renderer {
    fn ensure_csm_shadow(&mut self, resolution: u32, cascade_count: u32) {
        let resolution = resolution.max(1);
        let cascade_count = cascade_count.max(1);
        if let Some(ref csm) = self.csm_shadow {
            if csm.resolution == resolution && csm.cascade_count == cascade_count {
                return;
            }
        }
        self.csm_shadow = Some(shadow_pass::create_csm_shadow_resources(
            &self.device,
            &self.pipelines,
            &self.shadow_compare_sampler,
            resolution,
            cascade_count,
        ));
    }

    pub fn set_hdr_post_processing(&mut self, enabled: bool) {
        self.hdr_post_processing = enabled;
        if !enabled {
            self.post_fx = None;
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
        if let Some(ref fx) = self.post_fx {
            if fx.hdr_tex.size().width == w && fx.hdr_tex.size().height == h {
                return;
            }
        }
        // Create a 1x1 black texture for dummy slots
        let (black_tex, black_view) = self.texture_cache.black_placeholder(&self.device);
        self.post_fx = Some(post_processor::ensure_post_fx_textures(
            &self.device, &self.post_fx_pipelines, w, h, &black_tex, &black_view,
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

        // GPU timestamp query setup
        let gpu_query_period = queue.get_timestamp_period(); // in nanoseconds
        let (gpu_query_set, gpu_query_buffer) = if timing_supported {
            let query_count = 16u32; // 8 start/end pairs per frame
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("GPU timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: query_count,
            });
            let qb = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU query resolve"),
                size: query_count as u64 * 8, // u64 per timestamp
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            (Some(qs), Some(qb))
        } else {
            (None, None)
        };

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
        let phong_pool = GpuUniformPool::new_phong(&device, 1024);
        let shadow_pool = GpuUniformPool::new_shadow_pool(&device, &pipelines.shadow_draw_bgl, 1024);
        let flat_pool = GpuUniformPool::new_flat(&device, 2048);
        let outline_pool = GpuUniformPool::new_outline(&device, 1024);
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
            gpu_skinning_pass: None,
            skinned_mesh_resources: std::collections::HashMap::new(),
            animation_time_sec: 0.0,
            depth_texture: None,
            global_display_mode: DisplayMode::ShadedWithEdges,
            clip_planes: Vec::new(),
            wireframe_supported,
            assets: GpuAssetManager::new(),
            materials: MaterialLibrary::new(),
            frame_counter: 0,
            performance_mode_active: false,
            hud: None,
            hud_enabled: true,
            adaptive_quality: AdaptiveQuality::High,
            adaptive_frame_time_ema_ms: 16.7,
            adaptive_switch_cooldown_frames: 0,
            last_hud_update_frame: 0,
            outline_width: 0.022,
            outline_color: [1.0, 0.5, 0.0, 1.0],
            cluster_renderer: None,
            hzb: None,
            hzb_baker: None,
            cluster_pipeline_generation: 0,
            depth_reversed_z_mismatch_warned: false,
            texture_cache,
            ibl_diffuse,
            ibl_specular,
            ibl_preset,
            shadow_compare_sampler,
            csm_shadow,
            hdr_post_processing: false,
            post_fx_pipelines,
            post_fx: None,
            ssao_noise_tex,
            ssao_noise_view,
            instance_buffer,
            ibl_instance_bind_group,
            timing_supported,
            gpu_query_set,
            gpu_query_buffer,
            gpu_query_slots: 0,
            gpu_query_period,
            // Phase 2 features (lazy init below)
            pipeline_cache: None,
            shader_cache,
            viewport_layout: ViewportLayout::new(),
            grid_enabled: false,
            markup_vertices: Vec::new(),
            scene_vp: Mat4::IDENTITY,
            scene_camera_pos: Vec3::ZERO,
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
            enable_taa: false,
            enable_motion_blur: false,
            enable_ssr: false,
            enable_color_grading: false,
            enable_dof: false,
            enable_volumetric_fog: false,
            enable_cluster_lights: true,
            enable_omni_shadows: true,
            xray_mode: false,
            selection_outline_pipelines: Some(selection_outline_pipelines),
            selection_outline_targets: None,
            screen_space_selection_outline: true,
            enable_ldr_fxaa: true,
            ldr_shade_tex: None,
            ldr_shade_view: None,
            last_diagnostics: None,
        };
        renderer.hud = Some(HudRenderer::new(
            &renderer.device,
            &renderer.queue,
            renderer.config.format,
            renderer.config.width,
            renderer.config.height,
        ));
        renderer.create_depth_texture();
        renderer.hzb_baker = Some(HzbBaker::new(&renderer.device));
        let ds_bgl = &renderer.hzb_baker.as_ref().unwrap().downsample_bgl;
        renderer.hzb = Some(HzbPyramids::new(
            &renderer.device,
            ds_bgl,
            renderer.config.width,
            renderer.config.height,
        ));

        // ── Phase 2: post-processing passes ──
        renderer.taa_pass = Some(TaaPass::new(&renderer.device));
        renderer.motion_blur = Some(MotionBlurPass::new(&renderer.device));
        renderer.ssr_pass = Some(SsrPass::new(&renderer.device));
        renderer.color_grading = Some(ColorGradingPass::new(&renderer.device, &renderer.queue));
        renderer.dof_pass = Some(DofPass::new(&renderer.device));
        renderer.volumetric_fog = Some(VolumetricFogPass::new(&renderer.device));

        // ── Phase 2: lighting ──
        renderer.cluster_light_culler = Some(ClusterLightCuller::new(&renderer.device));
        renderer.cluster_lights = Some(ClusterLightResources::new(&renderer.device));
        renderer.omni_shadow = Some(OmniShadowRenderer::new(&renderer.device));

        // ── Pipeline cache ──
        let cache_dir = std::path::Path::new("cache");
        renderer.pipeline_cache = Some(PipelineCacheManager::new(&renderer.device, cache_dir));

        // ── Shader hot-reload: watch shaders directory ──
        let shaders_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
        renderer.shader_reload.watch_directory(&shaders_dir);

        // ── Material library: set BGL for bind group building ──
        renderer.materials.set_bind_group_layout(&renderer.pipelines.pbr_material_bgl);

        // ── Multi-viewport layout ──
        renderer.viewport_layout.rebuild(renderer.config.width, renderer.config.height);

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
            self.viewport_layout.rebuild(width, height);
            self.create_depth_texture();
            self.selection_outline_targets = None;
            self.ldr_shade_tex = None;
            self.ldr_shade_view = None;
            self.hzb_baker = Some(HzbBaker::new(&self.device));
            let ds_bgl = &self.hzb_baker.as_ref().unwrap().downsample_bgl;
            self.hzb = Some(HzbPyramids::new(&self.device, ds_bgl, width, height));
            if let Some(hud) = &mut self.hud {
                hud.resize(&self.queue, width, height);
            }
            if self.hdr_post_processing {
                self.ensure_post_fx_targets();
            }
        }
    }

    pub fn set_screen_space_selection_outline(&mut self, enabled: bool) {
        self.screen_space_selection_outline = enabled;
    }

    pub fn set_ldr_fxaa(&mut self, enabled: bool) {
        self.enable_ldr_fxaa = enabled;
        self.ldr_shade_tex = None;
        self.ldr_shade_view = None;
    }

    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        self.global_display_mode = mode;
    }

    /// RGBA for mesh outline pass, wireframe/edge-overlay defaults, and selection edge/bbox lines.
    pub fn set_outline_color(&mut self, rgba: [f32; 4]) {
        self.outline_color = rgba;
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
        self.ibl_diffuse = ibl_res.ibl_diffuse;
        self.ibl_specular = ibl_res.ibl_specular;
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
        self.ibl_instance_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL + Instance BG"),
            layout: &self.pipelines.ibl_instance_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&ibl_res.env_map_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ibl_res.brdf_lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&ibl_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.instance_buffer.as_entire_binding() },
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

    pub fn performance_mode_active(&self) -> bool {
        self.performance_mode_active
    }

    pub fn adaptive_quality_name(&self) -> &'static str {
        self.adaptive_quality.name()
    }

    pub fn adaptive_is_low(&self) -> bool {
        self.adaptive_quality.is_low()
    }

    pub fn report_frame_time_ms(&mut self, frame_time_ms: f32, control: AdaptiveControl) {
        if matches!(control, AdaptiveControl::Disabled) {
            self.adaptive_quality = AdaptiveQuality::High;
            self.adaptive_switch_cooldown_frames = 0;
            self.adaptive_frame_time_ema_ms = frame_time_ms.max(0.0);
            return;
        }
        if matches!(control, AdaptiveControl::Locked) {
            return;
        }
        // Smooth short spikes and enforce a brief cooldown after each switch
        // to prevent quality oscillation on borderline frame times.
        let alpha = 0.12_f32;
        self.adaptive_frame_time_ema_ms =
            self.adaptive_frame_time_ema_ms + (frame_time_ms - self.adaptive_frame_time_ema_ms) * alpha;

        if self.adaptive_switch_cooldown_frames > 0 {
            self.adaptive_switch_cooldown_frames -= 1;
            return;
        }

        let previous = self.adaptive_quality;
        let mut next = self.adaptive_quality.update(self.adaptive_frame_time_ema_ms);
        if let AdaptiveControl::Dynamic { allow_downgrade: false } = control {
            if next as u8 > previous as u8 {
                next = previous;
            }
        }
        self.adaptive_quality = next;
        if previous != self.adaptive_quality {
            self.adaptive_switch_cooldown_frames = 30;
            log::warn!(
                "Adaptive quality changed: {:?} -> {:?} (frame_ms={:.2}, ema_ms={:.2})",
                previous, self.adaptive_quality, frame_time_ms, self.adaptive_frame_time_ema_ms
            );
        }
    }

    pub fn set_clip_planes(&mut self, planes: Vec<[f32; 4]>) {
        self.clip_planes = planes;
    }

    pub fn collect_markup_vertices(&mut self, graph: &rc3d_scene::SceneGraph, root: rc3d_core::NodeId) {
        self.markup_vertices = crate::render_passes::pass_markup::collect_markup_lines(
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
        &self.clip_planes
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
        if let Some(pos) = self.clip_planes.iter().position(|p| p[0] == normal[0] && p[1] == normal[1] && p[2] == normal[2]) {
            self.clip_planes.remove(pos);
        } else {
            self.clip_planes.push(normal);
        }
    }

    pub fn invalidate_mesh_cache(&mut self) {
        self.assets.invalidate_all();
    }
}

