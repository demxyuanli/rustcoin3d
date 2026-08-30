use std::collections::HashSet;

use rc3d_core::NodeId;
use rc3d_gizmo::GizmoMode;
use rc3d_render::ibl::IblPreset;
use rc3d_render::renderer::CadDisplayTier;
use rc3d_render::viewport::LayoutMode;
use rc3d_render::FrameDiagnostics;

use crate::commands::AdaptiveQualityMode;
use crate::ui::i18n::UiLocale;
use crate::ui::theme::UiTheme;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditorDisplayMode {
    Wireframe,
    Shaded,
    ShadedWithEdges,
    HiddenLine,
    Flat,
    FlatWithEdge,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeDataType {
    Cube,
    Sphere,
    Cylinder,
    Cone,
    Separator,
    Transform,
    Material,
    DirectionalLight,
    PointLight,
    SpotLight,
    HemisphereLight,
    AreaLight,
    LightProbe,
    PerspectiveCamera,
    OrthographicCamera,
    StereoCamera,
    Text2,
    Text3,
    Font,
    Billboard,
    Sprite,
    Lod,
    Switch,
    Environment,
    SectionPlane,
    AnnotationSet,
    Measurement,
    Markup,
    InstancedMesh,
    BatchedMesh,
    Particles,
    All,
}

#[derive(Clone, Copy, Debug)]
pub struct RenderFeatureFlags {
    pub taa: bool,
    pub motion_blur: bool,
    pub ssr: bool,
    pub color_grading: bool,
    pub dof: bool,
    pub volumetric_fog: bool,
    pub cluster_lights: bool,
    pub omni_shadows: bool,
    pub xray: bool,
    pub ldr_fxaa: bool,
    pub screen_space_edges: bool,
    pub screen_space_selection_outline: bool,
    pub gpu_cull: bool,
    pub parallel_traversal: bool,
}

impl Default for RenderFeatureFlags {
    fn default() -> Self {
        Self {
            taa: true,
            motion_blur: false,
            ssr: false,
            color_grading: true,
            dof: false,
            volumetric_fog: false,
            cluster_lights: false,
            omni_shadows: false,
            xray: false,
            ldr_fxaa: true,
            screen_space_edges: false,
            screen_space_selection_outline: true,
            gpu_cull: false,
            parallel_traversal: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EditorUiContext {
    pub selected: HashSet<NodeId>,
    pub display_mode_label: String,
    pub ibl_label: String,
    pub ibl_preset: IblPreset,
    pub gizmo_mode: GizmoMode,
    pub layout_mode: LayoutMode,
    pub layout_mode_label: String,
    pub active_viewport_label: String,
    pub smoothed_fps: f32,
    pub frame_time_ms: f32,
    pub diagnostics: Option<FrameDiagnostics>,
    pub hidden_nodes: HashSet<NodeId>,
    pub render_features: RenderFeatureFlags,
    pub hdr_enabled: bool,
    pub vsync_enabled: bool,
    pub grid_enabled: bool,
    pub hud_enabled: bool,
    pub outline_width: f32,
    pub outline_color: [f32; 4],
    pub xray_mode: bool,
    pub ghost_unselected: bool,
    pub ghost_opacity: f32,
    pub feature_edge_color: [f32; 4],
    pub wireframe_edge_color: [f32; 4],
    pub hidden_edge_color: [f32; 4],
    pub crease_angle: f32,
    pub ss_edge_threshold: f32,
    pub wboit_enabled: bool,
    pub adaptive_quality_mode: AdaptiveQualityMode,
    pub adaptive_quality_name: String,
    pub cad_display_tier: CadDisplayTier,
    pub bookmarks: [(bool, &'static str); 9],
    pub selected_count: usize,
    pub document_path: Option<std::path::PathBuf>,
    pub document_dirty: bool,
    pub bg_mode: rc3d_render::background::BgMode,
    pub bg_top: [f32; 4],
    pub bg_bot: [f32; 4],
    pub post_vignette: f32,
    pub post_chromatic: f32,
    pub post_bloom: f32,
    pub post_grain: f32,
    pub post_halftone: f32,
    pub post_glitch: f32,
    pub csm_resolution: u32,
    pub csm_cascades: u32,
    pub interaction_render_scale: f32,
    pub walk_mode: bool,
    pub camera_from: [f32; 3],
    pub camera_up: [f32; 3],
    pub ui_theme: UiTheme,
    pub ui_locale: UiLocale,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptionAction {
    Minimize,
    ToggleMaximize,
    Close,
    Drag,
    ShowSystemMenu,
}

pub struct CaptionBarState {
    pub enabled: bool,
    pub maximized: bool,
    pub title: String,
    pub action: Option<CaptionAction>,
    pub dirty: bool,
    pub close_prompt: bool,
    pub close_after_save: bool,
}

impl Default for CaptionBarState {
    fn default() -> Self {
        Self {
            enabled: false,
            maximized: false,
            title: String::new(),
            action: None,
            dirty: false,
            close_prompt: false,
            close_after_save: false,
        }
    }
}

/// Host chrome flags and last layout holes (points, then converted to pixels).
pub struct EditorChromeState {
    pub document_open: bool,
    pub document_html: bool,
    pub scene_rect_points: Option<[f32; 4]>,
    pub document_rect_points: Option<[f32; 4]>,
    pub nav_cube_rect_points: Option<[f32; 4]>,
    pub nav_cube_dragging: bool,
    pub caption: CaptionBarState,
}

impl Default for EditorChromeState {
    fn default() -> Self {
        Self {
            document_open: true,
            document_html: false,
            scene_rect_points: None,
            document_rect_points: None,
            nav_cube_rect_points: None,
            nav_cube_dragging: false,
            caption: CaptionBarState::default(),
        }
    }
}

