use std::collections::HashSet;

use rc3d_core::NodeId;
use rc3d_gizmo::GizmoMode;
use rc3d_render::ibl::IblPreset;
use rc3d_render::renderer::CadDisplayTier;
use rc3d_render::viewport::LayoutMode;
use rc3d_render::FrameDiagnostics;

use crate::adaptive_quality::AdaptiveQualityMode;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditorDisplayMode {
    Wireframe,
    Shaded,
    ShadedWithEdges,
    HiddenLine,
    FlatWithEdge,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeDataType {
    Cube,
    Sphere,
    Cylinder,
    Cone,
    Separator,
    DirectionalLight,
    PointLight,
    SpotLight,
    PerspectiveCamera,
    OrthographicCamera,
    Text2,
    Text3,
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
    pub adaptive_quality_mode: AdaptiveQualityMode,
    pub adaptive_quality_name: String,
    pub cad_display_tier: CadDisplayTier,
    pub bookmarks: [(bool, &'static str); 9],
    pub selected_count: usize,
}
