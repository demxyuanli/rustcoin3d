use std::time::Instant;

use rc3d_core::DisplayMode;
use rc3d_effects::EffectGraph;
use rc3d_render::{FrameStats, Renderer};

use super::fps_tracker::FpsTracker;
use crate::adaptive_quality::AdaptiveQualityMode;
use crate::camera_controller::CameraController;
use crate::editor_ui::{EditorCommand, EditorUi};
use crate::scene_bridge::DynamicSurface;
use crate::viewport_camera::ViewportCameraSet;
use crate::world::World;

pub struct AppState {
    pub world: World,
    pub renderer: Option<Renderer>,
    pub window: Option<winit::window::Window>,
    /// Legacy camera controller; prefer viewport_cameras
    pub camera_controller: Option<CameraController>,
    pub viewport_cameras: ViewportCameraSet,
    pub initial_display_mode: DisplayMode,
    pub initial_post_effect_params: Option<(f32, f32, f32, f32)>,
    pub enable_hdr_post_processing: bool,
    pub adaptive_quality_mode: AdaptiveQualityMode,
    pub adaptive_last_interaction: Instant,
    pub continuous_redraw: bool,
    pub last_frame_time: Instant,
    pub last_frame_time_ms: f32,
    pub fps_tracker: FpsTracker,
    pub last_render_stats: FrameStats,
    pub window_title: String,
    pub editor_ui: Option<EditorUi>,
    pub editor_ui_enabled: bool,
    pub editor_commands: std::collections::VecDeque<EditorCommand>,
    pub hidden_nodes: std::collections::HashSet<rc3d_core::NodeId>,
    pub last_camera_eye: rc3d_core::math::Vec3,
    pub perf_mode_last: bool,
    pub bg_settings: Option<rc3d_render::background::BgSettings>,
    pub pending_effect_graph: Option<EffectGraph>,
    /// Dynamic NURBS surfaces that re-tessellate on camera movement.
    pub dynamic_surfaces: Vec<DynamicSurface>,
    /// Set by `render_interaction_frame` after inline render; cleared by `RedrawRequested`
    /// to skip the next full-quality frame (avoids double-rendering when both paths fire).
    pub interaction_frame_rendered: bool,
    /// Time of last inline render — used to rate-limit `render_interaction_frame`
    /// so CursorMoved floods (100s/sec on Windows) don't stall the event loop.
    pub last_inline_render_time: std::time::Instant,
    /// Counts completed inline renders for periodic diagnostic logging.
    pub interaction_render_count: u64,
}
