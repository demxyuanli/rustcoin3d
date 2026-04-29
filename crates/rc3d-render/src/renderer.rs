use std::path::Path;
use std::sync::Arc;

use slotmap::Key;
use wgpu::util::DeviceExt;

use crate::adaptive_quality::AdaptiveQuality;
use crate::asset_manager::{GpuAssetManager, MESH_CACHE_MAX};
use crate::auto_exposure::AutoExposure;
use crate::cluster::{ClusterRenderer, ClusterSet};
use crate::cluster_lighting::{ClusterLightCuller, ClusterLightResources};
use crate::color_grading::ColorGradingPass;
use crate::dof_pass::DofPass;
use crate::frustum::Frustum;
use crate::gpu_resource::{GpuMesh, GpuResourceManager, GpuUniformPool};
use crate::hud::HudRenderer;
use crate::material_library::MaterialLibrary;
use crate::hzb::{HzbBaker, HzbPyramids};
use crate::motion_blur::MotionBlurPass;
use crate::pipeline_cache::PipelineCacheManager;
use crate::pipelines::PipelineSet;
use crate::post_processor::{self, PostFxPipelines, PostFxTextures};
use crate::render_action::DrawCall;
use crate::render_passes::{self, PassContext};
use crate::shadow_map::{aabb_from_scene, csm_light_view_projs, compute_csm_splits, primary_directional_light_dir, union_draw_call_aabbs};
use crate::shadow_omni::OmniShadowRenderer;
use crate::shadow_pass::{self, CsmShadowResources};
use crate::shader_permutation::ShaderVariantCache;
use crate::shader_reload::ShaderHotReload;
use crate::viewport::ViewportLayout;
use crate::sort_keys;
use crate::ssr_pass::SsrPass;
use crate::taa::{TaaJitter, TaaPass};
use crate::texture_cache::TextureCache;
use crate::vertex::{InstanceData, LineVertex, CSM_CASCADE_COUNT, MAX_INSTANCES};
use crate::volumetric_fog::VolumetricFogPass;
use crate::ibl::IblPreset;
use glam::{Mat4, Vec3};
use rc3d_core::DisplayMode;
use rc3d_scene::SceneGraph;

const PERFORMANCE_MODE_TRIANGLE_THRESHOLD: u64 = 2_000_000;
const CLUSTER_PIPELINE_GENERATION: u32 = 5;

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub visible_triangles: u64,
    pub visible_draw_calls: usize,
    pub culled_draw_calls: usize,
    /// Approximate GPU time per pass in microseconds: [shadow, solid, post, total]
    pub gpu_pass_times_us: Option<[f64; 4]>,
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
    pub assets: GpuAssetManager,
    pub materials: MaterialLibrary,
    pub frame_counter: u64,
    pub performance_mode_active: bool,
    pub hud: Option<HudRenderer>,
    pub hud_enabled: bool,
    pub(super) adaptive_quality: AdaptiveQuality,
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

        let mut shader_cache = ShaderVariantCache::new();
        let pipelines = PipelineSet::create(&device, config.format, &mut shader_cache);
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
        let ibl_path = Path::new("test_data/studio.hdr");
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
            &ibl_sampler, ibl_path, ibl_preset,
        );
        let ibl_diffuse = ibl_res.ibl_diffuse;
        let ibl_specular = ibl_res.ibl_specular;

        // Combined IBL + instance bind group (group 3, 4 bindings)
        let ibl_instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL + Instance BG"),
            layout: &pipelines.ibl_instance_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&ibl_res.env_map_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ibl_res.brdf_lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&ibl_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: instance_buffer.as_entire_binding() },
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
            last_hud_update_frame: 0,
            outline_width: 0.022,
            outline_color: [0.0, 0.0, 0.0, 1.0],
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
            enable_cluster_lights: false,
            enable_omni_shadows: false,
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

    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        self.global_display_mode = mode;
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
        let ibl_path = Path::new("test_data/studio.hdr");
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
            ibl_path,
            preset,
        );
        self.ibl_diffuse = ibl_res.ibl_diffuse;
        self.ibl_specular = ibl_res.ibl_specular;
        self.ibl_instance_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL + Instance BG"),
            layout: &self.pipelines.ibl_instance_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&ibl_res.env_map_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ibl_res.brdf_lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&ibl_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.instance_buffer.as_entire_binding() },
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

    pub fn report_frame_time_ms(&mut self, frame_time_ms: f32) {
        let previous = self.adaptive_quality;
        self.adaptive_quality = self.adaptive_quality.update(frame_time_ms);
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
        self.assets.invalidate_all();
    }

    pub fn render_draw_calls(&mut self, draw_calls: &[DrawCall], scene: &SceneGraph) -> FrameStats {
        self.frame_counter = self.frame_counter.wrapping_add(1);

        // ── Shader hot-reload: check for changed .wgsl files ──
        let _changed = self.shader_reload.check_and_reload();

        // ── Pipeline cache: save periodically (every 300 frames ≈ 5s) ──
        if self.frame_counter % 300 == 0 {
            if let Some(ref mut pc) = self.pipeline_cache {
                pc.save_to_disk();
            }
        }

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
            let handle = if dc.vertices.is_empty() {
                if let Some(md) = dc.meshlet_data.as_ref() {
                    // Fallback-only path: use meshlet-expanded geometry when no standard vertices exist.
                    let ptr = Arc::as_ptr(md) as u64;
                    if let Some((mesh_id, last_used)) = self.assets.mesh_cache.get_mut(&ptr) {
                        *last_used = self.frame_counter;
                        Some(*mesh_id)
                    } else {
                        if self.assets.mesh_cache.len() >= MESH_CACHE_MAX {
                            self.prune_mesh_cache();
                        }
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
                        let mesh_id = self.gpu_meshes.upload_mesh(
                            &self.device,
                            &verts,
                            Some(&md.indices),
                            &[],
                        );
                        self.assets.mesh_cache.insert(ptr, (mesh_id, self.frame_counter));
                        Some(mesh_id)
                    }
                } else {
                    None
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
                if let Some((mesh_id, last_used)) = self.assets.mesh_cache.get_mut(&hash) {
                    *last_used = self.frame_counter;
                    Some(*mesh_id)
                } else {
                    if self.assets.mesh_cache.len() >= MESH_CACHE_MAX {
                        self.prune_mesh_cache();
                    }
                    let mesh_id = self.gpu_meshes.upload_mesh(
                        &self.device,
                        &dc.vertices,
                        dc.indices.as_ref().map(|a| a.as_slice()),
                        &dc.edge_positions,
                    );
                    self.assets.mesh_cache.insert(hash, (mesh_id, self.frame_counter));
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

    let camera_pos_vec: Vec3 = draw_calls.first().map(|dc| dc.camera_pos).unwrap_or(Vec3::ZERO);
    let mut transparent_order: Vec<usize> = (0..draw_calls.len())
        .filter(|&i| draw_calls[i].opacity < 1.0)
        .collect();

    transparent_order.sort_unstable_by(|&a, &b| {
        let pos_a = draw_calls[a].model_matrix.w_axis.truncate();
        let pos_b = draw_calls[b].model_matrix.w_axis.truncate();
        let da = pos_a.distance(camera_pos_vec);
        let db = pos_b.distance(camera_pos_vec);
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

        // Collect meshlet visible indices and upload ClusterSets
        let mut meshlet_indices: Vec<usize> = Vec::new();
        for (i, dc) in visible.iter().enumerate() {
            if dc.meshlet_data.is_some() {
                meshlet_indices.push(i);
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

        // Upload cluster sets after cluster renderer is ready
        let bgls = self.cluster_renderer.as_ref().map(|cr| cr.bind_group_layouts());
        for &idx in &meshlet_indices {
            let dc = visible[idx];
            let md = dc.meshlet_data.as_ref().unwrap();
            let ptr = Arc::as_ptr(md) as u64;
            if !self.assets.cluster_cache.contains_key(&ptr) {
                if let Some((cs_bgl, cmp_bgl, fin_bgl)) = bgls {
                    let cs = ClusterSet::from_meshlet_data(
                        &self.device, md, cs_bgl, cmp_bgl, fin_bgl,
                    );
                    self.assets.cluster_cache.insert(ptr, cs);
                }
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

        let _camera_vp = first.mvp * first.model_matrix.inverse();
        // Build a default perspective projection for SSAO (fov=60°, aspect from config)
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let near = 0.1f32;
        let far = 1000.0f32;
        let fov = 60.0f32.to_radians();
        let f = 1.0 / (fov / 2.0).tan();
        let camera_proj = Mat4::from_cols_array_2d(&[
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far / (far - near), 1.0],
            [0.0, 0.0, -near * far / (far - near), 0.0],
        ]);
        let camera_inv_proj = camera_proj.inverse();

        let mut csm_view_proj = [Mat4::IDENTITY; CSM_CASCADE_COUNT];
        let mut csm_split_depths = [0.0f32; CSM_CASCADE_COUNT];
        let mut shadow_params = [0.0_f32, 0.0004, 0.0, 0.0];
        let mut run_shadow_pass = false;

        if solid_wants_shadow {
            if let Some(dir) = primary_directional_light_dir(scene) {
                let aabb = aabb_from_scene(scene).or_else(|| union_draw_call_aabbs(visible.iter().copied()));
                if let Some(_aabb) = aabb {
                    let sm_size = match self.adaptive_quality {
                        AdaptiveQuality::High => 2048,
                        AdaptiveQuality::Medium => 1024,
                        AdaptiveQuality::Low => 512,
                    };
                    let cascade_count = match self.adaptive_quality {
                        AdaptiveQuality::High => 4,
                        AdaptiveQuality::Medium => 3,
                        AdaptiveQuality::Low => 1,
                    };
                    self.ensure_csm_shadow(sm_size, cascade_count);

                    let vp = first.mvp * first.model_matrix.inverse();
                    let camera_near = 0.1f32;
                    let camera_far = 1000.0f32;
                    let splits = compute_csm_splits(camera_near, camera_far, cascade_count, 0.5);
                    let inv_vp = vp.inverse();
                    let light_vps = csm_light_view_projs(dir, inv_vp, &splits, 8.0);

                    for (i, lvp) in light_vps.iter().enumerate() {
                        if i < CSM_CASCADE_COUNT {
                            csm_view_proj[i] = *lvp;
                        }
                    }
                    // Convert split depths to [0,1] range for shader comparison (view-space depth / far)
                    let far_range = camera_far - camera_near;
                    for i in 0..cascade_count as usize {
                        if i + 1 < splits.len() {
                            csm_split_depths[i] = (splits[i + 1] - camera_near) / far_range;
                        }
                    }
                    // Fill remaining with far
                    for i in cascade_count as usize..CSM_CASCADE_COUNT {
                        csm_split_depths[i] = 1.0;
                    }

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
            transparent_order: &transparent_order,
            mesh_handles: &mesh_handles,
            mode,
            run_outline,
            // Darker background for stronger silhouette/material contrast in sRGB output.
            bg_color: wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 },
            performance_mode_active: self.performance_mode_active,
            wireframe_supported: self.wireframe_supported,
            adaptive_quality: self.adaptive_quality,
            outline_width: self.outline_width,
            outline_color: self.outline_color,
            meshlet_indices: &meshlet_indices,
            camera_pos: [first.camera_pos.x, first.camera_pos.y, first.camera_pos.z],
            depth_reversed_z: first.depth_reversed_z,
            csm_view_proj,
            csm_split_depths,
            shadow_params,
            run_shadow_pass,
            camera_proj,
            camera_inv_proj,
        };

        let mut stats = render_passes::execute_passes(self, &ctx, draw_calls, self.frame_counter);
        stats.gpu_pass_times_us = self.read_gpu_timestamps();
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
            hud.update_text(&self.device, &self.queue, fps, frame_time_ms, stats, &hud_mode_name);
        }
    }

    pub(super) fn get_mesh(&self, mesh_id: crate::gpu_resource::MeshId) -> Option<&GpuMesh> {
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
        self.assets
            .prune_stale_meshes(self.frame_counter, &mut self.gpu_meshes);
    }

    pub(super) fn write_gpu_timestamp(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ref qs) = self.gpu_query_set {
            if self.gpu_query_slots < 16 {
                encoder.write_timestamp(qs, self.gpu_query_slots);
                self.gpu_query_slots += 1;
            }
        }
    }

    pub(super) fn resolve_gpu_timestamps(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(ref qs), Some(ref qb)) = (&self.gpu_query_set, &self.gpu_query_buffer) {
            let written = self.gpu_query_slots.min(16);
            if written > 0 {
                encoder.resolve_query_set(qs, 0..written, qb, 0);
            }
        }
    }

    pub(super) fn read_gpu_timestamps(&mut self) -> Option<[f64; 4]> {
        let qb = self.gpu_query_buffer.as_ref()?;
        let written = (self.gpu_query_slots.min(16)) as usize;
        if written < 8 { return None; } // need at least 4 pairs

        // Use a staging buffer for async readback (simplified: try immediate map)
        // For production, use map_async with a callback. For now, skip readback.
        let _ = qb;
        let _ = written;
        // Reset for next frame
        self.gpu_query_slots = 0;
        None // simplified: no readback yet; extend later with staging
    }
}
