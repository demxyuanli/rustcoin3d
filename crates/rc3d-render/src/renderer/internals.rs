use glam::{Mat4, Vec3};

use crate::adaptive_quality::AdaptiveQuality;
use crate::asset_manager::GpuAssetManager;
use crate::auto_exposure::AutoExposure;
use crate::cluster::ClusterRenderer;
use crate::cluster_lighting::{ClusterLightCuller, ClusterLightResources};
use crate::color_grading::ColorGradingPass;
use crate::dof_pass::DofPass;
use crate::gpu_resource::{GpuResourceManager, GpuUniformPool};
use crate::gpu_skinning::{GpuSkinningPass, GpuSkinningResources};
use crate::hud::HudRenderer;
use crate::hzb::{HzbBaker, HzbPyramids};
use crate::material_library::MaterialLibrary;
use crate::motion_blur::MotionBlurPass;
use crate::pipeline_cache::PipelineCacheManager;
use crate::pipelines::PipelineSet;
use crate::post_processor::{PostFxPipelines, PostFxTextures};
use crate::shadow_omni::OmniShadowRenderer;
use crate::shadow_pass::CsmShadowResources;
use crate::shader_permutation::ShaderVariantCache;
use crate::shader_reload::ShaderHotReload;
use crate::ssr_pass::SsrPass;
use crate::taa::{TaaJitter, TaaPass};
use crate::texture_cache::TextureCache;
use crate::vertex::MarkupVertex;
use crate::viewport::ViewportLayout;
use crate::volumetric_fog::VolumetricFogPass;

use super::types::{FrameDiagnostics, FrameStats};

/// CAD workflow display tier — controls which rendering passes are active.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CadDisplayTier {
    /// Model editing/creation: Flat shading, edges only. 60+ fps.
    DesignCreation = 0,
    /// Visualization/review: PBR + IBL + CSM shadows + edges.
    Visualization = 1,
    /// Industrial/engineering display: + SSAO, TAA/FXAA, color grading.
    IndustrialDisplay = 2,
    /// Product/marketing rendering: + SSR, volumetric fog, DOF.
    ProductRendering = 3,
}

impl CadDisplayTier {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::DesignCreation,
            1 => Self::Visualization,
            2 => Self::IndustrialDisplay,
            _ => Self::ProductRendering,
        }
    }
}

/// Per-tier rendering pass configuration.
#[derive(Clone, Debug)]
pub struct TierConfig {
    pub flat_shading: bool,
    pub shadows: bool,
    pub edges: bool,
    pub motion_blur: bool,
    pub ssao: bool,
    pub taa: bool,
    #[allow(dead_code)]
    pub fxaa: bool,
    pub color_grading: bool,
    pub ssr: bool,
    pub volumetric_fog: bool,
    pub dof: bool,
    pub hdr_post: bool,
}

impl TierConfig {
    pub fn for_tier(tier: CadDisplayTier) -> Self {
        match tier {
            CadDisplayTier::DesignCreation => Self {
                flat_shading: true,  shadows: false, edges: true,  motion_blur: false,
                ssao: false, taa: false, fxaa: false, color_grading: false,
                ssr: false, volumetric_fog: false, dof: false, hdr_post: false,
            },
            CadDisplayTier::Visualization => Self {
                flat_shading: false, shadows: true,  edges: true,  motion_blur: false,
                ssao: false, taa: false, fxaa: false, color_grading: false,
                ssr: false, volumetric_fog: false, dof: false, hdr_post: false,
            },
            CadDisplayTier::IndustrialDisplay => Self {
                flat_shading: false, shadows: true,  edges: false, motion_blur: false,
                ssao: false, taa: false, fxaa: false, color_grading: false,
                ssr: false, volumetric_fog: false, dof: false, hdr_post: true,
            },
            CadDisplayTier::ProductRendering => Self {
                flat_shading: false, shadows: true,  edges: false, motion_blur: false,
                ssao: false, taa: false, fxaa: false, color_grading: false,
                ssr: false, volumetric_fog: false, dof: false, hdr_post: true,
            },
        }
    }
}

/// GPU capability tier for runtime feature adaptation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuTier {
    /// Integrated GPU: meshlet path, no HZB occlusion, single LOD level.
    Basic,
    /// Mid-range discrete GPU: HZB occlusion, 2 LOD levels.
    Standard,
    /// High-end discrete GPU: HZB full precision, distance-based LOD.
    #[allow(dead_code)]
    Enhanced,
}

#[derive(Clone, Debug)]
pub struct GpuCapability {
    pub tier: GpuTier,
    #[allow(dead_code)]
    pub is_integrated: bool,
    #[allow(dead_code)]
    pub max_draw_indirect_count: u32,
    /// Whether meshlet GPU cull (frustum+HZB)→compact→draw_indexed_indirect is enabled.
    /// False on Basic tier (integrated GPU) where cull sync issues cause artifacts.
    pub meshlet_gpu_cull_enabled: bool,
}

pub(crate) struct FrameState {
    pub markup_vertices: Vec<MarkupVertex>,
    /// World-space annotation labels drawn in `pass_markup`.
    pub annotation_world_labels: Vec<crate::world_label::WorldLabelCommand>,
    /// Cached projected markup vertices (for static-frame fast path).
    pub cached_projected_markup: Vec<crate::vertex::MarkupVertex>,
    /// Cached world labels (for static-frame fast path).
    pub cached_projected_labels: Vec<crate::world_label::WorldLabelCommand>,
    /// GPU→CPU depth capture buffer (filled one frame, read the next).
    pub occlusion_capture_buf: Option<wgpu::Buffer>,
    /// Set when an async map of `occlusion_capture_buf` is in flight.
    /// 0 = in flight, 1 = mapped OK, 2 = mapping failed.
    pub occlusion_map_pending: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    /// Depth data from previous frame for annotation occlusion.
    pub occlusion_data: Option<(Vec<f32>, u32, u32)>,
    /// (tex_w, tex_h, aligned_row_bytes) for the capture buffer.
    pub occlusion_dims: (u32, u32, u32),
    pub clip_planes: Vec<[f32; 4]>,
    /// Per clip plane: cap fill color when `Some`, aligned with `clip_planes`.
    pub section_cap_tints: Vec<Option<[f32; 4]>>,
    pub scene_vp: Mat4,
    pub scene_vp_inv: Mat4,
    pub scene_depth_reversed_z: bool,
    pub scene_camera_pos: Vec3,
    pub animation_time_sec: f32,
    pub frame_counter: u64,
    pub performance_mode_active: bool,
    /// Hysteresis: frames to stay in performance mode after disabling trigger.
    pub perf_mode_cooldown: u8,
    pub last_diagnostics: Option<FrameDiagnostics>,
    pub last_hud_update_frame: u64,
    pub viewport_layout: ViewportLayout,
    #[allow(dead_code)]
    pub frame_stats: FrameStats,
    /// Effect commands collected during scene traversal (Decal, Volume, PointCloud).
    pub effect_commands: crate::render_passes::pass_effects::EffectCommands,
    /// Cached BVH + its items for incremental update (avoids full rebuild each frame).
    pub cached_bvh: Option<(rc3d_core::Bvh, Vec<(rc3d_core::Aabb, u32)>)>,
    /// Fast-path: skip full-scene text traversal when no Text2/Text3 nodes.
    pub has_text_nodes: bool,
    /// Fast-path: skip full-scene effect traversal when no Decal/Volume/PointCloud nodes.
    pub has_effect_nodes: bool,
    /// Fast-path: skip LOD distance updates when no Lod nodes in scene.
    pub has_lod_nodes: bool,
    /// How many frames since LOD nodes were last seen (auto-disable after 2 empty scans).
    pub lod_scan_frames_since_seen: u8,
    pub gpu_cull_ready: bool,
    /// Number of object transforms uploaded for GPU culling this frame
    /// (must match the dispatch size and the uniform's object_count).
    pub gpu_cull_object_count: u32,
    pub parallel_traversal_enabled: bool,
    /// Static frame: true when BVH had zero dirty AABBs this frame.
    pub bvh_fully_static: bool,
    /// Static frame: cached visible indices from previous static frame.
    pub static_visible_indices: Vec<usize>,
    /// Static frame: how many consecutive static frames we've had.
    pub static_frame_count: u64,
    /// Previous frame's view-projection for camera change detection.
    pub last_vp: glam::Mat4,
    /// Reusable allocations for frustum culling (cleared each frame, avoids re-allocation).
    pub bvh_out: Vec<u32>,
    pub visible_indices: Vec<usize>,
    /// Reusable draw-order Vecs (taken before sorting, restored after rendering).
    pub solid_order_buf: Vec<usize>,
    pub edge_order_buf: Vec<usize>,
    pub selected_order_buf: Vec<usize>,
    pub transparent_order_buf: Vec<usize>,
    pub light_hashes_buf: Vec<u64>,
    pub meshlet_indices_buf: Vec<usize>,
}

pub(crate) struct GpuInternals {
    pub pipelines: PipelineSet,
    #[allow(dead_code)]
    pub pbr_variant_cache: crate::pipelines::PbrVariantCache,
    pub phong_pool: GpuUniformPool,
    pub shadow_pool: GpuUniformPool,
    pub flat_pool: GpuUniformPool,
    pub line_pool: GpuUniformPool,
    pub section_cap_pool: GpuUniformPool,
    pub outline_pool: GpuUniformPool,
    pub gpu_meshes: GpuResourceManager,
    pub gpu_skinning_pass: Option<GpuSkinningPass>,
    pub skinned_mesh_resources:
        std::collections::HashMap<crate::gpu_resource::MeshId, GpuSkinningResources>,
    pub depth_texture: Option<(wgpu::Texture, wgpu::TextureView, wgpu::TextureView)>,
    pub assets: GpuAssetManager,
    pub materials: MaterialLibrary,
    pub hud: Option<HudRenderer>,
    pub world_label_font: crate::world_label_font::WorldLabelFont,
    pub adaptive_quality: AdaptiveQuality,
    pub adaptive_frame_time_ema_ms: f32,
    pub adaptive_switch_cooldown_frames: u8,
    pub cluster_renderer: Option<ClusterRenderer>,
    pub hzb: Option<HzbPyramids>,
    pub hzb_baker: Option<HzbBaker>,
    pub cluster_pipeline_generation: u32,
    pub depth_reversed_z_mismatch_warned: bool,
    pub texture_cache: TextureCache,
    pub ibl_diffuse: [f32; 4],
    pub ibl_specular: [f32; 4],
    pub shadow_compare_sampler: wgpu::Sampler,
    pub csm_shadow: Option<CsmShadowResources>,
    pub post_fx_pipelines: PostFxPipelines,
    pub post_fx: Option<PostFxTextures>,
    #[allow(dead_code)]
    pub ssao_noise_tex: wgpu::Texture,
    pub ssao_noise_view: wgpu::TextureView,
    pub instance_buffer: wgpu::Buffer,
    pub ibl_instance_bind_group: wgpu::BindGroup,
    pub timing_supported: bool,
    pub pipeline_cache: Option<PipelineCacheManager>,
    #[allow(dead_code)]
    pub shader_cache: ShaderVariantCache,
    pub shader_reload: ShaderHotReload,
    #[allow(dead_code)]
    pub auto_exposure: AutoExposure,
    pub taa_pass: Option<TaaPass>,
    #[allow(dead_code)]
    pub taa_jitter: TaaJitter,
    pub motion_blur: Option<MotionBlurPass>,
    pub ssr_pass: Option<SsrPass>,
    pub color_grading: Option<ColorGradingPass>,
    pub dof_pass: Option<DofPass>,
    pub volumetric_fog: Option<VolumetricFogPass>,
    pub cluster_lights: Option<ClusterLightResources>,
    pub cluster_light_culler: Option<ClusterLightCuller>,
    pub omni_shadow: Option<OmniShadowRenderer>,
    pub omni_shadow_map: Option<crate::shadow_omni::OmniShadowMap>,
    pub bg_pass: Option<crate::background::BgPass>,
    pub bg_settings: crate::background::BgSettings,
    pub decal_pass: Option<crate::render_passes::pass_effects::DecalPass>,
    pub volume_pass: Option<crate::render_passes::pass_effects::VolumePass>,
    pub point_cloud_pass: Option<crate::render_passes::pass_effects::PointCloudPass>,
    pub selection_outline_pipelines:
        Option<crate::selection_outline::SelectionOutlinePipelines>,
    pub selection_outline_targets: Option<crate::selection_outline::SelectionOutlineTargets>,
    pub ldr_shade_tex: Option<wgpu::Texture>,
    pub ldr_shade_view: Option<wgpu::TextureView>,
    /// Intermediate HDR texture for dynamic resolution scaling during interaction.
    pub interaction_downscale_tex: Option<wgpu::Texture>,
    pub interaction_downscale_view: Option<wgpu::TextureView>,
    pub interaction_downscale_depth: Option<wgpu::Texture>,
    pub interaction_downscale_depth_view: Option<wgpu::TextureView>,
    /// Read-only (DepthOnly aspect) view of `interaction_downscale_depth`.
    pub interaction_downscale_depth_read_view: Option<wgpu::TextureView>,
    /// Compute pipeline that downsamples the full depth buffer into a small
    /// R32Float grid for annotation occlusion (whole-screen coverage).
    pub occlusion_downsample_pipeline: Option<wgpu::ComputePipeline>,
    pub occlusion_downsample_bgl: Option<wgpu::BindGroupLayout>,
    /// Small R32Float target of the occlusion downsample (tex, storage view).
    pub occlusion_downsample_tex: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// Upscale pipeline + BGL + sampler for dynamic resolution interaction blit.
    pub upscale_pipeline: Option<wgpu::RenderPipeline>,
    pub upscale_bgl: Option<wgpu::BindGroupLayout>,
    pub upscale_sampler: Option<wgpu::Sampler>,
    /// Screen-space edge detection pipeline + BGL + uniform buffer.
    pub ss_edge_pipeline: Option<wgpu::RenderPipeline>,
    pub ss_edge_bgl: Option<wgpu::BindGroupLayout>,
    pub ss_edge_uniform: Option<wgpu::Buffer>,
    pub ss_edge_sampler: Option<wgpu::Sampler>,
    /// GPU compute culling: per-object transforms (STORAGE, updated each frame).
    pub transform_buffer: Option<wgpu::Buffer>,
    /// GPU compute culling: indirect draw args (STORAGE | INDIRECT).
    pub indirect_args_buffer: Option<wgpu::Buffer>,
    /// GPU compute culling: instance indices written by cull shader.
    pub instance_indices_buffer: Option<wgpu::Buffer>,
    pub gpu_cull_pass: Option<crate::gpu_culling::GpuCullPass>,
    pub gpu_cull_bg: Option<wgpu::BindGroup>,
    pub frustum_uniform: Option<wgpu::Buffer>,
    /// Staging buffer for reading back GPU cull instance count.
    pub gpu_cull_staging: Option<wgpu::Buffer>,
    pub gpu_cull_enabled: bool,
    pub max_gpu_cull_objects: u64,
    /// Whether multi-draw indirect is supported by the GPU adapter.
    pub multi_draw_indirect_supported: bool,
    /// Detected GPU capability tier for runtime feature adaptation.
    pub gpu_capability: GpuCapability,
    /// CAD display quality tier requested by user (may be clamped by GPU capability).
    pub requested_tier: CadDisplayTier,
    /// Effective tier after interaction degradation and recovery cooldown.
    pub effective_tier: CadDisplayTier,
    /// Whether user is currently interacting (orbit/pan/zoom).
    pub interaction_active: bool,
    /// Frames remaining in cooldown before tier recovery steps up.
    pub tier_cooldown_frames: u8,
    /// Global frame uniform buffer (lights, CSM, IBL, shadows) — uploaded once per frame, bound in group 2.
    pub global_frame_buffer: Option<wgpu::Buffer>,
    /// Velocity buffer for motion blur + TAA (144 bytes, updated each frame).
    pub velocity_buffer: Option<wgpu::Buffer>,
    pub draw_bufs: DrawBatchBufs,
}

pub(crate) struct DrawBatchBufs {
    pub meshlet_bitmask: Vec<bool>,
    pub meshlet_draws: Vec<usize>,
    pub standard_draws: Vec<usize>,
    pub instances: Vec<crate::vertex::InstanceData>,
    pub mat_keys: Vec<u64>,
    /// Pre-built indirect draw args, reset per frame.
    pub standard_indirect_args: Vec<wgpu::util::DrawIndexedIndirectArgs>,
    /// GPU buffer for indirect draw args (created once, written per frame).
    pub standard_indirect_buf: Option<wgpu::Buffer>,
}
