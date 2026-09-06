use std::collections::HashSet;

use rc3d_actions::MarkupTool;
use rc3d_core::NodeId;
use rc3d_engine_api::camera::ViewPreset;
use rc3d_gizmo::GizmoMode;
use rc3d_render::ibl::IblPreset;
use rc3d_render::renderer::CadDisplayTier;
use rc3d_render::viewport::{LayoutMode, ViewportSplitAxis};
use rc3d_render::FrameDiagnostics;
use rc3d_scene::node_data::MeasurementType;

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
    pub viewport_h_split: f32,
    pub viewport_v_split: f32,
    pub viewport_split_hover: Option<ViewportSplitAxis>,
    pub viewport_split_drag: Option<ViewportSplitAxis>,
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
    pub select_kind: SelectKind,
    pub canvas_tool: CanvasTool,
    pub measurement_type: Option<MeasurementType>,
    pub markup_tool: MarkupTool,
    pub measure_label: String,
    pub measure_points: usize,
    pub measure_needed: usize,
    pub keymap: crate::keymap::Keymap,
    pub locked_nodes: HashSet<NodeId>,
    pub history_undo: Vec<String>,
    pub history_redo: Vec<String>,
    pub file_dialog_dir: Option<std::path::PathBuf>,
    /// Studio case-library catalog entries (empty when host has none).
    pub case_catalog: Vec<CaseListItem>,
    pub active_case_id: Option<String>,
    pub case_params: Vec<CaseParamView>,
    pub case_steps: Vec<CaseStepView>,
}

/// Param vs process case classification for Assets / Cases UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaseKind {
    Param,
    Process,
    Both,
}

#[derive(Clone, Debug)]
pub struct CaseListItem {
    pub id: String,
    pub title_key: &'static str,
    pub kind: CaseKind,
    pub category: String,
}

#[derive(Clone, Debug)]
pub struct CaseParamView {
    pub id: String,
    pub label_key: &'static str,
    pub value: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Debug)]
pub struct CaseStepView {
    pub index: usize,
    pub label_key: &'static str,
    pub active: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl PixelRect {
    pub fn contains(self, px: f32, py: f32) -> bool {
        px >= self.x as f32
            && py >= self.y as f32
            && px < (self.x + self.width) as f32
            && py < (self.y + self.height) as f32
    }
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BottomTab {
    Document,
    Compositor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SideTab {
    Hierarchy,
    Render,
    History,
    Assets,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PropsTab {
    Object,
    Material,
    Display,
    Camera,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Workspace {
    Model,
    LookDev,
    Compositor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectKind {
    Pick,
    Box,
    Lasso,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanvasTool {
    Select,
    Transform,
    Measure,
    Section,
    Markup,
    Walk,
}

/// Last-used flyout variants for the left tool strip (Photoshop-style).
pub struct ToolStripState {
    pub select: SelectKind,
    pub measure: MeasurementType,
    pub markup: MarkupTool,
    pub view: ViewPreset,
    pub create: NodeDataType,
}

impl Default for ToolStripState {
    fn default() -> Self {
        Self {
            select: SelectKind::Pick,
            measure: MeasurementType::Distance,
            markup: MarkupTool::Line,
            view: ViewPreset::Iso,
            create: NodeDataType::Cube,
        }
    }
}

pub struct EditorChromeState {
    pub bottom_tab: Option<BottomTab>,
    pub side_tab: Option<SideTab>,
    pub props_tab: PropsTab,
    pub workspace: Workspace,
    pub outliner_filter: String,
    pub keymap_capture: Option<crate::keymap::KeyAction>,
    pub tool_strip_pos: Option<[f32; 2]>,
    pub tool_strip_rect_points: Option<[f32; 4]>,
    pub side_dock_width: Option<f32>,
    pub bottom_dock_height: Option<f32>,
    /// Split ratio of the Hierarchy tab: tree on top, inspector below.
    /// `None` until the user drags the divider; persisted to prefs.
    pub inspector_ratio: Option<f32>,
    /// Transient: side dock resize handle is being dragged this frame.
    pub side_dock_resizing: bool,
    /// Transient: bottom dock resize handle is being dragged this frame.
    pub bottom_dock_resizing: bool,
    /// Filter text for the in-tab case library (Assets tab).
    pub cases_filter: String,
    pub tools: ToolStripState,
    pub scene_rect_points: Option<[f32; 4]>,
    pub nav_cube_rect_points: Option<[f32; 4]>,
    pub nav_cube_dragging: bool,
    pub nav_cube_hover_slot: Option<u32>,
    /// Updated each egui frame; used to route scene pointer without blocking the empty viewport.
    pub pointer_over_egui: bool,
    pub caption: CaptionBarState,
}

impl Default for EditorChromeState {
    fn default() -> Self {
        Self {
            bottom_tab: Some(BottomTab::Document),
            side_tab: Some(SideTab::Hierarchy),
            props_tab: PropsTab::Object,
            workspace: Workspace::Model,
            outliner_filter: String::new(),
            keymap_capture: None,
            tool_strip_pos: None,
            tool_strip_rect_points: None,
            side_dock_width: None,
            bottom_dock_height: None,
            inspector_ratio: None,
            side_dock_resizing: false,
            bottom_dock_resizing: false,
            cases_filter: String::new(),
            tools: ToolStripState::default(),
            scene_rect_points: None,
            nav_cube_rect_points: None,
            nav_cube_dragging: false,
            nav_cube_hover_slot: None,
            pointer_over_egui: false,
            caption: CaptionBarState::default(),
        }
    }
}

impl EditorChromeState {
    pub fn set_bottom_tab(&mut self, tab: BottomTab, open: bool) {
        if open {
            self.bottom_tab = Some(tab);
        } else if self.bottom_tab == Some(tab) {
            self.bottom_tab = None;
        }
    }

    pub fn set_side_tab(&mut self, tab: SideTab, open: bool) {
        if open {
            self.side_tab = Some(tab);
        } else if self.side_tab == Some(tab) {
            self.side_tab = None;
        }
    }

    pub fn document_tab_open(&self) -> bool {
        self.bottom_tab == Some(BottomTab::Document)
    }

    pub fn compositor_tab_open(&self) -> bool {
        self.bottom_tab == Some(BottomTab::Compositor)
    }

    pub fn apply_workspace(&mut self, ws: Workspace) {
        self.workspace = ws;
        match ws {
            Workspace::Model => {
                self.side_tab = Some(SideTab::Hierarchy);
                self.bottom_tab = Some(BottomTab::Document);
            }
            Workspace::LookDev => {
                // Properties live inside the Hierarchy tab (tree on top,
                // inspector below), so LookDev points there too.
                self.side_tab = Some(SideTab::Hierarchy);
                self.bottom_tab = None;
            }
            Workspace::Compositor => {
                self.side_tab = Some(SideTab::Render);
                self.bottom_tab = Some(BottomTab::Compositor);
            }
        }
    }
}
