#![allow(clippy::too_many_arguments, clippy::type_complexity)]
pub mod adaptive_quality;
pub mod asset_manager;
pub mod dirty_flags;
pub mod auto_exposure;
pub mod async_loader;
pub mod background;
pub mod cluster;
pub mod cluster_lighting;
pub mod cluster_tree;
pub mod color_grading;
pub mod dof_pass;
pub mod flat_draw_cache;
pub mod frustum;
pub mod global_tables;
pub mod gpu_culling;
pub mod gpu_resource;
pub mod gpu_skinning;
pub mod hud;
mod plane_text;
mod font_loader;
mod world_label;
mod world_label_font;
pub mod hzb;
pub mod offscreen;
pub mod parallel_traversal;
pub mod ibl;
pub mod light_set;
pub mod lod_state;
pub mod streaming_lod;
pub mod material_library;
pub mod mesh_pool;
pub mod motion_blur;
pub mod pass_graph;
pub mod pipeline_cache;
pub mod pipelines;
pub mod post_processor;
pub mod render_action;
pub mod render_graph;
pub mod render_passes;
pub mod renderer;
pub mod selection_outline;
pub mod settings;
pub mod shader_permutation;
pub mod shader_reload;
pub mod shadow_map;
pub mod shadow_omni;
pub mod shadow_pass;
pub mod sort_keys;
pub mod ss_edges;
pub mod ssr_pass;
pub mod taa;
pub mod texture_cache;
pub mod texture_format;
pub mod texture_streaming;
pub mod vertex;
pub mod viewport;
pub mod volumetric_fog;
pub mod profiler;

pub use asset_manager::GpuAssetManager;
pub use async_loader::{AssetHandle, AsyncAssetManager};
pub use auto_exposure::AutoExposure;
pub use pipeline_cache::PipelineCacheManager;
pub use shader_permutation::{ShaderFeatures, ShaderVariantCache, preprocess_wgsl};
pub use shader_reload::ShaderHotReload;
pub use ssr_pass::SsrPass;
pub use taa::{TaaJitter, TaaPass};
pub use frustum::Frustum;
pub use gpu_resource::{EdgeLineKind, GpuResourceManager, GpuUniformPool, MeshId};
pub use cluster_lighting::{ClusterLightCuller, ClusterLightResources, GpuPointLight, GpuSpotLight};
pub use color_grading::ColorGradingPass;
pub use dof_pass::DofPass;
pub use material_library::{MaterialId, MaterialLibrary};
pub use volumetric_fog::VolumetricFogPass;
pub use motion_blur::MotionBlurPass;
pub use hud::HudRenderer;
pub use render_passes::pass_effects::EffectCommands;
pub use offscreen::OffscreenTarget;
pub use pipelines::{DepthModePipelines, PipelineSet};
pub use render_action::{
    apply_world_camera, DrawCall, RenderCollector, SkinnedMeshDrawPayload,
};
pub use renderer::{
    AdaptiveControl, BatchAnalysis, FrameDiagnostics, FrameStats, MemoryBudget, NodeTypeDrawStat,
    Renderer,
};
pub use settings::{DisplaySettings, LightingSettings, PostEffectSettings, RenderSettings};
pub use texture_cache::{ibl_from_image_path, TextureCache, TextureHandle};
pub use vertex::{
    FlatUniforms, GlobalFrameUniforms, InstanceData, LineVertex, MarkupVertex, SceneUniforms,
    ShadowDrawUniforms, Vertex, WorldLabelVertex, MAX_INSTANCES, MAX_LIGHTS, CSM_CASCADE_COUNT,
};
pub use font_loader::{configure_font_system, new_label_font_system, LabelFont, ENV_FONT_DIR, ENV_FONT_PATH};
pub use world_label::WorldLabelCommand;
pub use viewport::{
    LayoutMode, ProjectionType, Viewport, ViewportId, ViewportLayout, ViewportRect, ViewportSplitAxis,
    VIEWPORT_SPLITTER_HIT_PX,
};
