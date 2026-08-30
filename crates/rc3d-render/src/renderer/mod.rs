use std::path::{Path, PathBuf};

use wgpu::util::DeviceExt;

mod types;
mod helpers;
mod internals;
mod skinning;
mod resource_lifecycle;
mod frame_scheduler;
mod pass_orchestration;
mod presentation;

pub use types::*;
pub use internals::CadDisplayTier;
pub(crate) use internals::{GpuTier, TierConfig};

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
use self::internals::{DrawBatchBufs, FrameState, GpuInternals};
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
    pub volumetric_fog_settings: VolumetricFogSettings,
    pub enable_cluster_lights: bool,
    pub enable_omni_shadows: bool,
    pub enable_ldr_fxaa: bool,
    /// Enable SMAA (Subpixel Morphological Anti-Aliasing) instead of FXAA.
    /// Mutually exclusive with enable_ldr_fxaa for LDR path.
    pub enable_ldr_smaa: bool,
    pub hdr_post_processing: bool,
    /// Enable WBOIT (Weighted Blended Order-Independent Transparency) for
    /// transparent objects instead of traditional back-to-front alpha blend.
    /// Works on both LDR (CAD) and HDR shade targets.
    pub enable_wboit: bool,
    pub global_display_mode: DisplayMode,
    /// Display mode explicitly chosen by the user (may differ from global_display_mode
    /// when a tier forces flat shading). Used to restore the user's choice when leaving
    /// a flat-shading tier.
    user_display_mode: DisplayMode,
    /// Effective CAD tier: allow CSM shadow pass when geometry/display mode allow it.
    pub(crate) tier_wants_shadow: bool,
    /// Effective CAD tier: allow outline/edge overlay passes when display mode allows it.
    pub(crate) tier_wants_edges: bool,
    /// When true, edge overlay uses all triangle edges (wireframe) instead of feature edges.
    pub wireframe_overlay: bool,
    pub grid_enabled: bool,
    pub hud_enabled: bool,
    /// Pixel region used as the 3D film (egui central hole). Full surface when unset.
    pub scene_region: crate::viewport::ViewportRect,
    /// Size of the color target for the pass currently being encoded.
    pub(crate) pass_target_size: (u32, u32),
    pub outline_width: f32,
    /// Selection/bbox outline color. Default: orange.
    pub outline_color: [f32; 4],
    /// Feature edge color (ShadedWithEdges / FlatWithEdge). Default: red.
    pub feature_edge_color: [f32; 4],
    /// Full wireframe overlay edge color. Default: dark blue.
    pub wireframe_edge_color: [f32; 4],
    /// Occluded HiddenLine dashes (Fast HLR). Default: mid gray.
    pub hidden_edge_color: [f32; 4],
    /// Unlit HiddenLine face fill (paper). Default: dark gray.
    pub hidden_line_fill_color: [f32; 4],
    /// Face fill color for flat-shading mode. When `Some`, overrides material color.
    pub flat_face_color: Option<[f32; 4]>,
    /// Resolution scale factor during interaction (0.25..1.0). Default 0.5 = 50% per axis.
    pub interaction_render_scale: f32,
    /// When true, use screen-space edge detection instead of geometry edges during interaction.
    pub screen_space_edges: bool,
    /// When true, skip depth prepass during interaction.
    pub skip_prepass_interaction: bool,
    /// Sobel gradient threshold for screen-space edge detection (depth units). Default 0.015.
    pub ss_edge_threshold: f32,
    pub xray_mode: bool,
    /// HOOPS Isolate/Ghost: unselected filled geometry is drawn translucent.
    pub ghost_unselected: bool,
    pub ghost_opacity: f32,
    /// World-space gizmo line batches (Engine / editor overlay).
    pub gizmo_line_batches: Vec<(Vec<crate::vertex::LineVertex>, [f32; 4])>,
    /// Depth of field: world-space focus distance (default 5.0).
    pub dof_focus_distance: f32,
    /// Depth of field: lens aperture size (default 2.0).
    pub dof_aperture: f32,
    pub screen_space_selection_outline: bool,
    pub ibl_preset: IblPreset,
    /// Optional equirectangular HDR/image used as the IBL environment map.
    ibl_env_path: Option<PathBuf>,
    pub post_fx_params: crate::post_processor::PostEffectParams,

    /// When true, the renderer skips expensive passes (shadows, edges, SSAO)
    /// even below the triangle threshold. Set by the app during camera orbit/pan/zoom.
    pub interaction_active: bool,

    /// Set when `set_display_tier` runs: tier-driven toggles are enforced each frame via
    /// `reapply_cad_tier_constraints` so display mode / HDR / post paths cannot bypass CAD tier.
    pub(crate) cad_tier_authoritative: bool,

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
    pub fn ensure_bg_pass(&mut self) {
        if self.gpu.bg_pass.is_none() {
            self.gpu.bg_pass = Some(crate::background::BgPass::new(
                &self.device, &self.queue, self.config.format,
            ));
        }
    }

    pub fn set_background(&mut self, settings: crate::background::BgSettings) {
        self.ensure_bg_pass();
        if let Some(ref mut bg) = self.gpu.bg_pass {
            if let Some(ref img_path) = settings.image_path {
                bg.set_image(&self.device, &self.queue, img_path);
            }
            if settings.cube_faces.iter().any(|f| f.is_some()) {
                bg.set_cube_faces(&self.device, &self.queue, &settings.cube_faces);
            }
        }
        self.gpu.bg_settings = settings;
    }

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

    pub fn ensure_custom_shader_pass(&mut self) {
        if self.gpu.custom_shader_pass.is_none() {
            self.gpu.custom_shader_pass = Some(crate::custom_shader::CustomShaderPass::new(
                &self.device, self.config.format,
            ));
        }
    }

    /// Set effect commands for this frame (called by app layer after traversal).
    pub fn set_effect_commands(&mut self, cmds: crate::render_passes::pass_effects::EffectCommands) {
        self.frame.effect_commands = cmds;
    }

    /// Scene camera view-projection (`projection * view`) from the last graph traversal.
    pub fn set_scene_view_projection(&mut self, view: Mat4, projection: Mat4) {
        let vp = projection * view;
        self.frame.scene_vp = vp;
        self.frame.scene_vp_inv = vp.inverse();
        self.frame.scene_depth_reversed_z =
            rc3d_core::depth_reversed_z_from_projection(projection);
    }

    /// Reconfigure cascaded shadow map resolution and cascade count.
    ///
    /// Recreates shadow resources if the requested parameters differ from current.
    pub fn set_csm_shadow(&mut self, resolution: u32, cascade_count: u32) {
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
            self.gpu.global_frame_buffer.as_ref().unwrap(),
            self.gpu.omni_shadow_map.as_ref().expect("omni shadow map not initialized"),
            resolution,
            cascade_count,
        ));
    }

    /// Set the CAD display quality tier. Clamped by GPU capability on Basic tier.
    pub fn set_display_tier(&mut self, tier: CadDisplayTier) {
        self.cad_tier_authoritative = true;
        let max_tier = match self.gpu.gpu_capability.tier {
            GpuTier::Basic => CadDisplayTier::Visualization,
            GpuTier::Standard => CadDisplayTier::IndustrialDisplay,
            GpuTier::Enhanced => CadDisplayTier::ProductRendering,
        };
        let requested = if tier > max_tier { max_tier } else { tier };
        if tier != requested {
            log::warn!(
                "Requested tier {:?} exceeds GPU capability (max {:?}); clamping to {:?}",
                tier, max_tier, requested
            );
        }
        if self.gpu.requested_tier != requested {
            self.gpu.requested_tier = requested;
            self.gpu.effective_tier = requested;
            self.apply_tier_config(true);
        } else if self.gpu.effective_tier != requested {
            // Effective can lag behind requested during orbit degrade/recovery; choosing the
            // same tier again (or repeating a tier hotkey) must snap visuals without changing request.
            self.gpu.effective_tier = requested;
            self.gpu.tier_cooldown_frames = 0;
            self.apply_tier_config(true);
        }
    }

    /// Called once per frame. Updates effective tier with degradation and recovery.
    /// Reads `self.interaction_active` (set by app during camera orbit/pan/zoom).
    pub fn update_tier(&mut self) {
        let requested = self.gpu.requested_tier;
        let max_tier = match self.gpu.gpu_capability.tier {
            GpuTier::Basic => CadDisplayTier::Visualization,
            GpuTier::Standard => CadDisplayTier::IndustrialDisplay,
            GpuTier::Enhanced => CadDisplayTier::ProductRendering,
        };
        let clamped = if requested > max_tier { max_tier } else { requested };

        // Detect interaction stop: start cooldown for recovery
        if self.gpu.interaction_active && !self.interaction_active {
            self.gpu.tier_cooldown_frames = 30; // ~500ms at 60fps
        }
        self.gpu.interaction_active = self.interaction_active;

        // Degrade during interaction (tiers 2+ only)
        if self.interaction_active {
            let degraded = match clamped {
                CadDisplayTier::ProductRendering => CadDisplayTier::Visualization,
                CadDisplayTier::IndustrialDisplay => CadDisplayTier::Visualization,
                other => other,
            };
            if self.gpu.effective_tier != degraded {
                self.gpu.effective_tier = degraded;
                self.gpu.tier_cooldown_frames = 0;
                self.apply_tier_config(false);
            }
            return;
        }

        // Snap down when effective exceeds allowed target (requested lowered or clamp tightened).
        if self.gpu.effective_tier > clamped {
            self.gpu.effective_tier = clamped;
            self.gpu.tier_cooldown_frames = 0;
            self.apply_tier_config(false);
        }

        // Recovery: step up one tier per cooldown period
        if self.gpu.effective_tier < clamped {
            if self.gpu.tier_cooldown_frames > 0 {
                self.gpu.tier_cooldown_frames -= 1;
            } else {
                let next = self.gpu.effective_tier as u32 + 1;
                let next_tier = CadDisplayTier::from_u32(next);
                if next_tier <= clamped {
                    self.gpu.effective_tier = next_tier;
                    self.gpu.tier_cooldown_frames = 30;
                    self.apply_tier_config(false);
                }
            }
        }
    }

    /// Apply feature toggles for the current effective tier.
    ///
    /// When `user_initiated` is true (user changed tier via UI), the tier's
    /// display-mode semantics are also applied (e.g. DesignCreation → Flat).
    /// When false (automatic degradation/recovery), display mode is preserved
    /// so the user's explicit choice is not overridden.
    fn apply_tier_config(&mut self, user_initiated: bool) {
        let cfg = TierConfig::for_tier(self.gpu.effective_tier);
        log::debug!(
            "Tier config applied: {:?} (user={}) | HDR={} SSAO={} TAA={} SSR={} DOF={} Fog={}",
            self.gpu.effective_tier, user_initiated,
            cfg.hdr_post, cfg.ssao, cfg.taa, cfg.ssr, cfg.dof, cfg.volumetric_fog
        );
        // Apply display-mode semantics on user-initiated tier switch.
        // Entering a flat-shading tier forces Flat; leaving restores user's choice.
        if user_initiated {
            self.global_display_mode = if cfg.flat_shading {
                if cfg.edges { DisplayMode::FlatWithEdge } else { DisplayMode::Flat }
            } else if cfg.edges {
                DisplayMode::ShadedWithEdges
            } else {
                DisplayMode::Shaded
            };
        }
        self.tier_wants_shadow = cfg.shadows;
        self.tier_wants_edges = cfg.edges;
        // Derive feature toggles from tier config
        self.enable_taa = cfg.taa;
        self.enable_motion_blur = cfg.motion_blur && self.gpu.motion_blur.is_some();
        self.enable_ssr = cfg.ssr;
        self.enable_color_grading = cfg.color_grading;
        self.enable_dof = cfg.dof;
        self.enable_volumetric_fog = cfg.volumetric_fog;
        if cfg.hdr_post && self.gpu.post_fx.is_none() {
            self.ensure_post_fx_targets();
        }
        self.hdr_post_processing = cfg.hdr_post;
        // Motion blur needs TAA reference + HDR; guard against missing resources
        if self.enable_motion_blur && !self.hdr_post_processing {
            self.enable_motion_blur = false;
        }
        if self.enable_motion_blur {
            self.enable_taa = true;
        }
    }

    /// Flat-shading tiers (e.g. DesignCreation) always render as Flat; display-mode changes
    /// still update `user_display_mode` so leaving the tier restores the user's choice.
    fn clamp_global_display_mode_for_flat_shading_tier(&mut self) {
        let cfg = TierConfig::for_tier(self.gpu.effective_tier);
        if cfg.flat_shading {
            // Wireframe is an explicit user choice; don't overwrite it.
            if self.global_display_mode == DisplayMode::Wireframe {
                return;
            }
            self.global_display_mode = if cfg.edges {
                DisplayMode::FlatWithEdge
            } else {
                DisplayMode::Flat
            };
        }
    }

    /// Re-apply pipeline toggles for the current effective CAD tier (used when tier is
    /// authoritative so interaction overrides / inspector cannot bypass tier presets).
    pub fn reapply_cad_tier_constraints(&mut self) {
        self.apply_tier_config(false);
        self.clamp_global_display_mode_for_flat_shading_tier();
    }

    /// Whether CAD tier was explicitly chosen via [`Self::set_display_tier`].
    pub fn cad_tier_authoritative(&self) -> bool {
        self.cad_tier_authoritative
    }

    pub fn set_wboit(&mut self, enabled: bool) {
        self.enable_wboit = enabled;
        if !enabled {
            self.gpu.wboit_targets = None;
        }
    }

    pub(crate) fn ensure_wboit_targets(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if let Some(t) = &self.gpu.wboit_targets {
            if t.width == w && t.height == h {
                return;
            }
        }
        let accum_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("WBOIT Accum"),
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
        let revealage_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("WBOIT Revealage"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.gpu.wboit_targets = Some(crate::renderer::internals::WboitTargets {
            width: w,
            height: h,
            accum_view: accum_tex.create_view(&wgpu::TextureViewDescriptor::default()),
            revealage_view: revealage_tex.create_view(&wgpu::TextureViewDescriptor::default()),
            accum_tex,
            revealage_tex,
        });
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
            | wgpu::Features::PIPELINE_CACHE
            | wgpu::Features::MULTI_DRAW_INDIRECT;
        let features = adapter.features() & requested_features;
        let wireframe_supported = true; // wireframe pass uses line-list edges, not PolygonMode::Line
        let timing_supported = features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let multi_draw_indirect_supported = features.contains(wgpu::Features::MULTI_DRAW_INDIRECT);

        let adapter_info = adapter.get_info();
        let is_integrated = matches!(
            adapter_info.device_type,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu
        );
        // Basic tier only for CPU/software rasterizers. All real GPUs get
        // at least Standard. Integrated GPUs (e.g. Intel Arc) are capable
        // enough to run SSAO, TAA, HDR, etc.
        let tier = if matches!(adapter_info.device_type, wgpu::DeviceType::Cpu) {
            internals::GpuTier::Basic
        } else {
            internals::GpuTier::Standard
        };
        let meshlet_gpu_cull_enabled = !is_integrated;
        let gpu_capability = internals::GpuCapability {
            tier,
            is_integrated,
            max_draw_indirect_count: 0, // feature gated — use multi_draw_indirect_supported
            meshlet_gpu_cull_enabled,
        };
        log::info!("GPU tier: {:?} (integrated={})", tier, is_integrated);

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

        // Global frame uniform buffer — uploaded once per frame, bound in group 2.
        let global_frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Global Frame Uniforms"),
            size: std::mem::size_of::<crate::vertex::GlobalFrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Velocity buffer for motion blur + TAA (144 bytes: mat4x4 + mat4x4 + vec2 + vec2).
        let velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Velocity Params"),
            size: 144,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let selection_outline_pipelines =
            crate::selection_outline::SelectionOutlinePipelines::new(&device, &pipelines.flat_bgl);
        let phong_pool = GpuUniformPool::new_phong(&device, 65536);
        let shadow_pool = GpuUniformPool::new_shadow_pool(&device, &pipelines.shadow_draw_bgl, 32768);
        let flat_pool = GpuUniformPool::new_flat(&device, 32768);
        let section_cap_pool = GpuUniformPool::new_section_cap(&device, &pipelines.flat_bgl, 16384);
        let line_pool = GpuUniformPool::new_line(&device, &pipelines.flat_bgl, 65536);
        let texture_cache = TextureCache::new(&device, &queue);
        let shadow_compare_sampler = shadow_pass::create_shadow_compare_sampler(&device);
        let omni_renderer = crate::shadow_omni::OmniShadowRenderer::new(&device);
        let default_omni_shadow = crate::shadow_omni::OmniShadowMap::new(
            &device,
            omni_renderer.resource_bgl(),
            crate::shadow_omni::OMNISHADOW_RESOLUTION,
        );
        let csm_shadow = Some(shadow_pass::create_csm_shadow_resources(
            &device,
            &pipelines,
            &shadow_compare_sampler,
            &global_frame_buffer,
            &default_omni_shadow,
            1,
            1,
        ));
        let post_fx_pipelines = post_processor::create_post_fx_pipelines(&device, config.format, wgpu::TextureFormat::Rgba16Float);

        // Upscale pipeline for dynamic resolution interaction blit
        let upscale_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Upscale Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("../shaders/upscale.wgsl"))),
        });
        let upscale_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Upscale BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
        let upscale_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Upscale Pipeline Layout"),
            bind_group_layouts: &[&upscale_bgl],
            push_constant_ranges: &[],
        });
        let upscale_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Upscale Pipeline"),
            layout: Some(&upscale_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &upscale_shader,
                entry_point: Some("vs_upscale"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &upscale_shader,
                entry_point: Some("fs_upscale"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            depth_stencil: None,
            cache: None,
        });
        let upscale_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Upscale Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Screen-space edge detection pipeline
        let ss_edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Edge Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("../shaders/edge_detect.wgsl"))),
        });
        let ss_edge_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SS Edge Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let ss_edge_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Edge BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let ss_edge_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SS Edge Pipeline Layout"),
            bind_group_layouts: &[&ss_edge_bgl],
            push_constant_ranges: &[],
        });
        let ss_edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SS Edge Pipeline"),
            layout: Some(&ss_edge_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &ss_edge_shader,
                entry_point: Some("vs_sobel"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ss_edge_shader,
                entry_point: Some("fs_sobel"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            depth_stencil: None,
            cache: None,
        });
        let ss_edge_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SS Edge Uniform"),
            size: 32, // 8 floats (threshold + edge_rgb + texel_x/y + padding)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (ssao_noise_tex, ssao_noise_view) = post_processor::create_ssao_noise(&device, &queue);
        let instance_stride = std::mem::size_of::<InstanceData>() as u64;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance data SSBO"),
            size: instance_stride * MAX_INSTANCES as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let standard_indirect_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Standard Indirect Args"),
            size: 16384u64 * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
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
            volumetric_fog_settings: VolumetricFogSettings::default(),
            enable_cluster_lights: true,
            enable_omni_shadows: true,
            enable_ldr_fxaa: true,
            enable_ldr_smaa: false,
            hdr_post_processing: false,
            enable_wboit: true,
            global_display_mode: DisplayMode::Shaded,
            user_display_mode: DisplayMode::Shaded,
            tier_wants_shadow: true,
            tier_wants_edges: true,
            wireframe_overlay: false,
            grid_enabled: false,
            hud_enabled: true,
            scene_region: crate::viewport::ViewportRect::default(),
            pass_target_size: (1, 1),
            outline_width: 1.0,
            outline_color: [1.0, 0.5, 0.0, 1.0], // orange
            feature_edge_color: [0.9, 0.15, 0.1, 1.0], // red
            wireframe_edge_color: [0.15, 0.25, 0.7, 1.0], // dark blue
            hidden_edge_color: [0.62, 0.64, 0.68, 1.0],
            hidden_line_fill_color: [0.14, 0.14, 0.15, 1.0],
            flat_face_color: Some([0.68, 0.72, 0.78, 1.0]), // light blue-gray, CAD default
            interaction_render_scale: 1.0, // default off — explicit opt-in via set_interaction_render_scale
            screen_space_edges: true,
            skip_prepass_interaction: true,
            ss_edge_threshold: 0.015,
            xray_mode: false,
            ghost_unselected: false,
            ghost_opacity: crate::render_action::GHOST_UNSELECTED_OPACITY,
            gizmo_line_batches: Vec::new(),
            dof_focus_distance: 5.0,
            dof_aperture: 2.0,
            screen_space_selection_outline: true,
            ibl_preset,
            ibl_env_path: None,
            post_fx_params: crate::post_processor::PostEffectParams::default(),
            interaction_active: false,
            cad_tier_authoritative: false,
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
                annotation_world_labels: Vec::new(),
                cached_projected_markup: Vec::new(),
                cached_projected_labels: Vec::new(),
                clip_planes: Vec::new(),
                section_cap_tints: Vec::new(),
                scene_vp: Mat4::IDENTITY,
                scene_vp_inv: Mat4::IDENTITY,
                scene_depth_reversed_z: false,
                scene_camera_pos: Vec3::ZERO,
                animation_time_sec: 0.0,
                frame_counter: 0,
                performance_mode_active: false,
                perf_mode_cooldown: 0,
                last_diagnostics: None,
                last_hud_update_frame: 0,
                viewport_layout: ViewportLayout::new(),
                frame_stats: FrameStats::default(),
                effect_commands: crate::render_passes::pass_effects::EffectCommands::default(),
                cached_bvh: None,
                has_text_nodes: true,    // optimistic; auto-disabled after 2 empty frames
                has_effect_nodes: true,  // optimistic; auto-disabled after 2 empty frames
                has_lod_nodes: true,     // optimistic; auto-disabled after 2 frames with no LODs
                lod_scan_frames_since_seen: 0,
                gpu_cull_ready: false,
                gpu_cull_object_count: 0,
                parallel_traversal_enabled: false,
                bvh_fully_static: false,
                static_visible_indices: Vec::with_capacity(1024),
                static_frame_count: 0,
                last_vp: glam::Mat4::IDENTITY,
                bvh_out: Vec::with_capacity(1024),
                visible_indices: Vec::with_capacity(1024),
                solid_order_buf: Vec::with_capacity(1024),
                edge_order_buf: Vec::with_capacity(256),
                selected_order_buf: Vec::with_capacity(64),
                transparent_order_buf: Vec::with_capacity(128),
                light_hashes_buf: Vec::with_capacity(1024),
                meshlet_indices_buf: Vec::with_capacity(256),
                occlusion_capture_buf: None,
                occlusion_map_pending: None,
                occlusion_data: None,
                occlusion_dims: (0, 0, 0),
            },
            // GPU internals
            gpu: GpuInternals {
                pipelines,
                pbr_variant_cache: crate::pipelines::PbrVariantCache::new(),
                phong_pool,
                shadow_pool,
                flat_pool,
                section_cap_pool,
                line_pool,
                gpu_meshes: GpuResourceManager::new(),
                gpu_skinning_pass: None,
                skinned_mesh_resources: std::collections::HashMap::new(),
                depth_texture: None,
                assets: GpuAssetManager::new(),
                materials: MaterialLibrary::new(),
                hud: None,
                world_label_font: crate::world_label_font::WorldLabelFont::new(),
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
                sh_l2: [[0.0; 4]; 9],
                sh_intensity: [0.0; 4],
                ibl: Some(ibl_res),
                ibl_sampler,
                morph_dummy,
                morph_params_dummy,
                cube_camera: None,
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
                omni_shadow_map: None,
                bg_pass: None,
                bg_settings: crate::background::BgSettings::default(),
                decal_pass: None,
                volume_pass: None,
                point_cloud_pass: None,
                custom_shader_pass: None,
                selection_outline_pipelines: Some(selection_outline_pipelines),
                selection_outline_targets: None,
                ldr_shade_tex: None,
                ldr_shade_view: None,
                interaction_downscale_tex: None,
                interaction_downscale_view: None,
                interaction_downscale_depth: None,
                interaction_downscale_depth_view: None,
                interaction_downscale_depth_read_view: None,
                occlusion_downsample_pipeline: None,
                occlusion_downsample_bgl: None,
                occlusion_downsample_tex: None,
                upscale_pipeline: Some(upscale_pipeline),
                upscale_bgl: Some(upscale_bgl),
                upscale_sampler: Some(upscale_sampler),
                anaglyph_pipeline: None,
                anaglyph_bgl: None,
                ss_edge_pipeline: Some(ss_edge_pipeline),
                ss_edge_bgl: Some(ss_edge_bgl),
                ss_edge_uniform: Some(ss_edge_uniform),
                ss_edge_sampler: Some(ss_edge_sampler),
                transform_buffer: None,
                indirect_args_buffer: None,
                instance_indices_buffer: None,
                gpu_cull_pass: None,
                gpu_cull_bg: None,
                frustum_uniform: None,
                gpu_cull_staging: None,
                gpu_cull_enabled: false,
                max_gpu_cull_objects: 65536,
                multi_draw_indirect_supported,
                gpu_capability,
                requested_tier: internals::CadDisplayTier::Visualization,
                effective_tier: internals::CadDisplayTier::Visualization,
                interaction_active: false,
                tier_cooldown_frames: 0,
                global_frame_buffer: Some(global_frame_buffer),
                velocity_buffer: Some(velocity_buffer),
                quad_tiles: Vec::new(),
                wboit_targets: None,
                draw_bufs: DrawBatchBufs {
                    meshlet_bitmask: Vec::with_capacity(4096),
                    meshlet_draws: Vec::with_capacity(4096),
                    standard_draws: Vec::with_capacity(4096),
                    instances: Vec::with_capacity(4096),
                    mat_keys: Vec::with_capacity(4096),
                    standard_indirect_args: Vec::with_capacity(4096),
                    standard_indirect_buf: Some(standard_indirect_buf),
                },
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
        renderer.gpu.omni_shadow = Some(omni_renderer);
        renderer.gpu.omni_shadow_map = Some(default_omni_shadow);

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
        renderer.scene_region = crate::viewport::ViewportRect {
            x: 0,
            y: 0,
            width: renderer.config.width.max(1),
            height: renderer.config.height.max(1),
        };
        renderer.pass_target_size = (renderer.config.width.max(1), renderer.config.height.max(1));

        renderer.apply_tier_config(false);
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

    /// Create or resize intermediate HDR+Depth textures for dynamic resolution scaling.
    pub(super) fn ensure_interaction_downscale_targets(&mut self, full_w: u32, full_h: u32) {
        let scale = self.interaction_render_scale;
        let w = ((full_w as f32) * scale).max(64.0) as u32;
        let h = ((full_h as f32) * scale).max(64.0) as u32;
        let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };

        // Check if resize needed
        let need_resize = self.gpu.interaction_downscale_tex.as_ref().map_or(true, |t| {
            t.width() != w || t.height() != h
        });

        if need_resize {
            // Color texture (HDR-compatible when hdr_post_processing is on)
            let color_fmt = if self.hdr_post_processing {
                wgpu::TextureFormat::Rgba16Float
            } else {
                self.config.format
            };
            let color_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Interaction Downscale Color"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

            // Depth texture
            let depth_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Interaction Downscale Depth"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32FloatStencil8,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_read_view = depth_tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Interaction Downscale Depth readonly"),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0, mip_level_count: Some(1),
                base_array_layer: 0, array_layer_count: Some(1),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            });

            self.gpu.interaction_downscale_tex = Some(color_tex);
            self.gpu.interaction_downscale_view = Some(color_view);
            self.gpu.interaction_downscale_depth = Some(depth_tex);
            self.gpu.interaction_downscale_depth_view = Some(depth_view);
            self.gpu.interaction_downscale_depth_read_view = Some(depth_read_view);
        }
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
            self.scene_region = crate::viewport::ViewportRect {
                x: 0,
                y: 0,
                width,
                height,
            };
            self.pass_target_size = (width, height);
            self.create_depth_texture();
            self.gpu.selection_outline_targets = None;
            self.gpu.ldr_shade_tex = None;
            self.gpu.ldr_shade_view = None;
            self.gpu.wboit_targets = None;
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

    /// Upload L2 SH irradiance probe (last probe wins; intensity 0 disables).
    pub fn set_light_probe(&mut self, sh_l2: [[f32; 4]; 9], intensity: f32) {
        self.gpu.sh_l2 = sh_l2;
        self.gpu.sh_intensity = [intensity.max(0.0), 0.0, 0.0, 0.0];
    }

    /// Call after LOD scan to update auto-detection state.
    pub fn note_lod_scan(&mut self, lod_count: usize) {
        if lod_count > 0 {
            self.frame.has_lod_nodes = true;
            self.frame.lod_scan_frames_since_seen = 0;
        } else if self.frame.lod_scan_frames_since_seen >= 2 {
            self.frame.has_lod_nodes = false;
        } else {
            self.frame.lod_scan_frames_since_seen += 1;
        }
    }

    /// Whether the scene contains LOD nodes (fast-path skip when false).
    pub fn has_lod_nodes(&self) -> bool {
        self.frame.has_lod_nodes
    }

    /// Enable parallel scene traversal using rayon thread pool.
    /// Best for scenes with 10K+ objects and multiple dirty subtrees.
    pub fn set_parallel_traversal(&mut self, enabled: bool) {
        self.frame.parallel_traversal_enabled = enabled;
    }

    pub fn parallel_traversal_enabled(&self) -> bool {
        self.frame.parallel_traversal_enabled
    }

    pub fn gpu_culling_enabled(&self) -> bool {
        self.gpu.gpu_cull_enabled
    }

    /// Toggle GPU frustum culling. First enable allocates cull buffers.
    pub fn set_gpu_culling(&mut self, enabled: bool) {
        if enabled {
            if self.gpu.gpu_cull_pass.is_none() {
                let max = self.gpu.max_gpu_cull_objects.max(4096);
                self.enable_gpu_culling(max);
            } else {
                self.gpu.gpu_cull_enabled = true;
            }
        } else {
            self.gpu.gpu_cull_enabled = false;
        }
    }

    pub fn csm_shadow_params(&self) -> (u32, u32) {
        self.gpu
            .csm_shadow
            .as_ref()
            .map(|c| (c.resolution, c.cascade_count))
            .unwrap_or((2048, 4))
    }

    /// Enable GPU compute culling with buffers sized for max_objects.
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
            size: crate::gpu_culling::FRUSTUM_UNIFORM_SIZE,
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
        self.gpu.gpu_cull_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Cull Readback"),
            size: 4 + 4 * max_objects, // count (u32) + indices (u32 each)
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
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
        self.user_display_mode = settings.display.display_mode;
        self.clamp_global_display_mode_for_flat_shading_tier();
        self.grid_enabled = settings.display.grid_enabled;
        self.hud_enabled = settings.display.hud_enabled;
        self.outline_width = settings.display.outline_width;
        self.outline_color = settings.display.outline_color;
        self.xray_mode = settings.display.xray_mode;
        self.ghost_unselected = settings.display.ghost_unselected;
        self.ghost_opacity = settings.display.ghost_opacity;
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
        // Performance settings
        self.set_parallel_traversal(settings.performance.parallel_traversal);
        if settings.performance.mesh_pool_capacity > 0 {
            self.gpu.assets.enable_mesh_pool(settings.performance.mesh_pool_capacity);
            if let Some(ref mut pool) = self.gpu.assets.mesh_pool {
                pool.set_max_bytes(settings.performance.mesh_pool_max_mb * 1024 * 1024);
            }
        }
        self.settings = settings;
    }

    /// Read-only snapshot of current settings.
    pub fn settings(&self) -> &RenderSettings {
        &self.settings
    }

    pub fn vsync_enabled(&self) -> bool {
        matches!(self.config.present_mode, wgpu::PresentMode::AutoVsync)
    }

    fn is_vsync_enabled(&self) -> bool {
        self.vsync_enabled()
    }

    pub fn set_scene_region(&mut self, rect: crate::viewport::ViewportRect) {
        self.scene_region = rect.clamped_to(self.config.width, self.config.height);
    }

    /// GPU viewport for the current pass target.
    ///
    /// Swapchain / full-window intermediates keep the egui hole (`scene_region`).
    /// Offscreen tiles (quad views, screenshots) fill the whole target — applying
    /// window-space `scene_region` there clips geometry into a rectangle.
    pub(crate) fn current_pass_viewport(&self) -> crate::viewport::ViewportRect {
        let (tw, th) = (self.pass_target_size.0.max(1), self.pass_target_size.1.max(1));
        let (sw, sh) = (self.config.width.max(1), self.config.height.max(1));
        if tw == sw && th == sh {
            self.scene_region.clamped_to(tw, th)
        } else {
            crate::viewport::ViewportRect {
                x: 0,
                y: 0,
                width: tw,
                height: th,
            }
        }
    }

    pub fn apply_scene_viewport(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.current_pass_viewport().apply_to_pass(pass);
    }

    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        self.user_display_mode = mode;
        self.global_display_mode = mode;
        self.clamp_global_display_mode_for_flat_shading_tier();
    }

    pub fn display_tier(&self) -> CadDisplayTier {
        self.gpu.effective_tier
    }

    /// User-requested CAD tier (before interaction-time degradation).
    pub fn requested_display_tier(&self) -> CadDisplayTier {
        self.gpu.requested_tier
    }

    /// RGBA for mesh outline pass and selection edge/bbox lines. Default: orange [1.0, 0.5, 0.0, 1.0].
    pub fn set_outline_color(&mut self, rgba: [f32; 4]) {
        self.outline_color = rgba;
    }

    /// RGBA for feature (crease) edges in ShadedWithEdges / FlatWithEdge modes. Default: red.
    pub fn set_feature_edge_color(&mut self, rgba: [f32; 4]) {
        self.feature_edge_color = rgba;
    }

    /// RGBA for full wireframe overlay edges (F5). Default: dark blue.
    pub fn set_wireframe_edge_color(&mut self, rgba: [f32; 4]) {
        self.wireframe_edge_color = rgba;
    }

    /// RGBA for Fast Hidden Line dashed occluded edges. Default: mid gray.
    pub fn set_hidden_edge_color(&mut self, rgba: [f32; 4]) {
        self.hidden_edge_color = rgba;
    }

    /// Override face fill color in flat-shading mode. `None` uses material color. Default: `None`.
    pub fn set_flat_face_color(&mut self, rgba: Option<[f32; 4]>) {
        self.flat_face_color = rgba;
    }

    /// Set feature edge crease angle in degrees. Default: 12°.
    /// Lower values include more edges (smoother surfaces show more structure).
    /// Takes effect on next scene traversal (mesh cache rebuild).
    pub fn set_feature_edge_crease_angle(&mut self, deg: f32) {
        crate::render_action::set_feature_crease_angle(deg);
    }

    /// Set interaction render scale (0.25..1.0). Default: 0.5.
    /// Lower = faster interaction but softer image. 1.0 = no scaling.
    pub fn set_interaction_render_scale(&mut self, scale: f32) {
        self.interaction_render_scale = scale.clamp(0.25, 1.0);
    }

    /// Enable screen-space edge detection during interaction (replaces geometry edges).
    pub fn set_screen_space_edges(&mut self, enabled: bool) {
        self.screen_space_edges = enabled;
    }

    /// Set Sobel gradient threshold for screen-space edge detection. Default: 0.015.
    /// Higher = fewer detected edges.
    pub fn set_ss_edge_threshold(&mut self, threshold: f32) {
        self.ss_edge_threshold = threshold;
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

    pub fn set_ghost_unselected(&mut self, enabled: bool) {
        self.ghost_unselected = enabled;
    }

    pub fn set_ghost_opacity(&mut self, opacity: f32) {
        self.ghost_opacity = opacity.clamp(0.02, 0.95);
    }

    pub fn set_outline_width(&mut self, width: f32) {
        self.outline_width = width;
    }

    // ── Internal accessors for render_passes (pub(crate)) ──
    pub fn viewport_layout(&self) -> &ViewportLayout {
        &self.frame.viewport_layout
    }
    pub fn viewport_layout_mut(&mut self) -> &mut ViewportLayout {
        &mut self.frame.viewport_layout
    }

    /// Update post-processing effect parameters at runtime.
    /// Exposure is managed separately by auto_exposure and written per-frame.
    pub fn set_post_effect_params(&mut self, vignette: f32, chromatic: f32, bloom_str: f32, grain: f32) {
        self.post_fx_params.vignette = vignette;
        self.post_fx_params.chromatic = chromatic;
        self.post_fx_params.bloom_str = bloom_str;
        self.post_fx_params.grain = grain;
        self.upload_post_fx_params();
    }

    pub fn set_post_stylize(&mut self, halftone: f32, glitch: f32) {
        self.post_fx_params.halftone = halftone;
        self.post_fx_params.glitch = glitch;
        self.upload_post_fx_params();
    }

    fn upload_post_fx_params(&self) {
        self.queue.write_buffer(
            &self.gpu.post_fx_pipelines.post_params_buf,
            0,
            bytemuck::bytes_of(&self.post_fx_params),
        );
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
        self.rebuild_ibl();
        log::info!("IBL preset switched to {}", preset.name());
    }

    /// Load an equirectangular HDR/image as the IBL environment map.
    pub fn set_ibl_from_path(&mut self, path: PathBuf) {
        self.ibl_env_path = Some(path);
        self.rebuild_ibl();
        log::info!("IBL environment loaded from custom path");
    }

    fn rebuild_ibl(&mut self) {
        let preset = self.ibl_preset;
        let ibl_path = self
            .ibl_env_path
            .clone()
            .filter(|p| p.is_file())
            .unwrap_or_else(resolve_studio_hdr_path);
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
        self.gpu.ibl_instance_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL + Instance BG"),
            layout: &self.gpu.pipelines.ibl_instance_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&ibl_res.env_map_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ibl_res.brdf_lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&ibl_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.gpu.instance_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.gpu.morph_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.gpu.morph_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.gpu.morph_params_dummy.as_entire_binding() },
            ],
        });
        self.gpu.ibl_sampler = ibl_sampler;
        self.gpu.ibl = Some(ibl_res);
        if let Some(cc) = self.gpu.cube_camera.as_mut() {
            cc.mark_dirty();
        }
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

    pub fn force_adaptive_low(&mut self) {
        let prev = self.gpu.adaptive_quality;
        self.gpu.adaptive_quality = AdaptiveQuality::Low;
        if prev != AdaptiveQuality::Low {
            log::warn!(
                "Adaptive quality forced: {:?} -> Low (large scene)",
                prev
            );
        }
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

    pub fn set_clip_planes(
        &mut self,
        planes: Vec<[f32; 4]>,
        mut cap_tints: Vec<Option<rc3d_scene::SectionCapStyle>>,
    ) {
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

    /// Collect text strings from MarkupElement::Text nodes for HUD overlay rendering.
    pub fn collect_markup_text(&self, graph: &rc3d_scene::SceneGraph) -> Vec<String> {
        crate::render_passes::pass_markup::collect_markup_text_lines(graph)
    }

    /// Prepare HUD overlay: scene Text2/Text3 (annotation labels use world quads in pass_markup).
    pub fn prepare_hud_overlay_for_render(&mut self) {
        if let Some(ref mut hud) = self.gpu.hud {
            hud.positioned_texts.clone_from(&hud.scene_positioned_texts);
            hud.prepare_gpu_atlas_for_render(
                &self.device,
                &self.queue,
                self.frame.scene_depth_reversed_z,
            );
        }
    }

    pub fn has_overlay_elements(&self) -> bool {
        !self.frame.markup_vertices.is_empty() || self.hud_enabled
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

