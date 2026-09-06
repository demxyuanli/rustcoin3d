use rc3d_core::math::Mat4;
use rc3d_core::NodeId;
use rc3d_engine_api::Engine;
use rc3d_render::viewport::ViewportSplitAxis;
use rc3d_scene::node_data::MeasurementType;
use std::collections::VecDeque;

use crate::commands::EditorCommand;
use crate::ui::types::{CanvasTool, SelectKind};

/// Editor-local interaction state (pointer, gizmo, measurement, box-select).
pub struct EditorInteractionState {
    pub gizmo_dragging: bool,
    pub gizmo_pending_transform: Option<(NodeId, Mat4)>,
    pub measurement_mode: bool,
    pub measurement_type: Option<MeasurementType>,
    pub section_edit_mode: bool,
    pub box_select_drag: bool,
    pub box_select_anchor: (f32, f32),
    pub lasso_drag: bool,
    pub lasso_points: Vec<(f32, f32)>,
    pub section_hovered: Option<NodeId>,
    pub section_drag: Option<(NodeId, [f32; 4])>,
    pub view_split_hover: Option<ViewportSplitAxis>,
    pub view_split_drag: Option<ViewportSplitAxis>,
    pub left_pick_arm_pos: Option<(f64, f64)>,
    pub left_drag_suppresses_pick: bool,
    pub markup: rc3d_actions::MarkupAction,
    pub markup_preview_live: bool,
    pub measure: rc3d_actions::MeasurementAction,
    pub measure_label: String,
    pub locked_nodes: std::collections::HashSet<NodeId>,
    pub select_kind: SelectKind,
    pub canvas: CanvasTool,
}

impl Default for EditorInteractionState {
    fn default() -> Self {
        Self {
            gizmo_dragging: false,
            gizmo_pending_transform: None,
            measurement_mode: false,
            measurement_type: None,
            section_edit_mode: false,
            box_select_drag: false,
            box_select_anchor: (0.0, 0.0),
            lasso_drag: false,
            lasso_points: Vec::new(),
            section_hovered: None,
            section_drag: None,
            view_split_hover: None,
            view_split_drag: None,
            left_pick_arm_pos: None,
            left_drag_suppresses_pick: false,
            markup: rc3d_actions::MarkupAction::new(),
            markup_preview_live: false,
            measure: rc3d_actions::MeasurementAction::new(rc3d_actions::MeasurementMode::Distance),
            measure_label: String::new(),
            locked_nodes: std::collections::HashSet::new(),
            select_kind: SelectKind::Pick,
            canvas: CanvasTool::Select,
        }
    }
}

/// Runtime editor context wrapping the engine and all editor-local state.
pub struct EditorContext<'a> {
    pub engine: &'a mut Engine,
    pub commands: VecDeque<EditorCommand>,
    pub interaction: EditorInteractionState,
}

impl<'a> EditorContext<'a> {
    pub fn new(engine: &'a mut Engine) -> Self {
        Self::with_interaction(engine, EditorInteractionState::default())
    }

    pub fn with_interaction(engine: &'a mut Engine, interaction: EditorInteractionState) -> Self {
        Self {
            engine,
            commands: VecDeque::new(),
            interaction,
        }
    }

    pub fn push_command(&mut self, cmd: EditorCommand) {
        self.commands.push_back(cmd);
    }

    pub fn pop_command(&mut self) -> Option<EditorCommand> {
        self.commands.pop_front()
    }

    /// Access the scene graph through the engine.
    pub fn scene(&self) -> &rc3d_scene::SceneGraph {
        self.engine.scene()
    }

    /// Mutable access to the scene graph.
    pub fn scene_mut(&mut self) -> &mut rc3d_scene::SceneGraph {
        self.engine.scene_mut()
    }
}
