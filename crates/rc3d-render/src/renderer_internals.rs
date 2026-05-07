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
use crate::vertex::LineVertex;
use crate::viewport::ViewportLayout;
use crate::volumetric_fog::VolumetricFogPass;

use super::renderer_types::{FrameDiagnostics, FrameStats};

pub(crate) struct FrameState {
    pub markup_vertices: Vec<LineVertex>,
    pub clip_planes: Vec<[f32; 4]>,
    /// Per clip plane: cap fill color when `Some`, aligned with `clip_planes`.
    pub section_cap_tints: Vec<Option<[f32; 4]>>,
    pub scene_vp: Mat4,
    pub scene_camera_pos: Vec3,
    pub animation_time_sec: f32,
    pub frame_counter: u64,
    pub performance_mode_active: bool,
    pub last_diagnostics: Option<FrameDiagnostics>,
    pub last_hud_update_frame: u64,
    pub viewport_layout: ViewportLayout,
    #[allow(dead_code)]
    pub frame_stats: FrameStats,
    /// Effect commands collected during scene traversal (Decal, Volume, PointCloud).
    pub effect_commands: crate::render_passes::pass_effects::EffectCommands,
}

pub(crate) struct GpuInternals {
    pub pipelines: PipelineSet,
    #[allow(dead_code)]
    pub pbr_variant_cache: crate::pipelines::PbrVariantCache,
    pub phong_pool: GpuUniformPool,
    pub shadow_pool: GpuUniformPool,
    pub flat_pool: GpuUniformPool,
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
    pub decal_pass: Option<crate::render_passes::pass_effects::DecalPass>,
    pub volume_pass: Option<crate::render_passes::pass_effects::VolumePass>,
    pub point_cloud_pass: Option<crate::render_passes::pass_effects::PointCloudPass>,
    pub selection_outline_pipelines:
        Option<crate::selection_outline::SelectionOutlinePipelines>,
    pub selection_outline_targets: Option<crate::selection_outline::SelectionOutlineTargets>,
    pub ldr_shade_tex: Option<wgpu::Texture>,
    pub ldr_shade_view: Option<wgpu::TextureView>,
}
