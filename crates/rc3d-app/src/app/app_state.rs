use std::time::Instant;

use rc3d_core::DisplayMode;
use rc3d_render::{FrameStats, Renderer};

use crate::adaptive_quality::AdaptiveQualityMode;
use crate::camera_controller::CameraController;
use crate::editor_ui::{EditorCommand, EditorUi};
use super::fps_tracker::FpsTracker;
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
}
