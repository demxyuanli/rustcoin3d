use rc3d_core::math::Mat4;
use rc3d_core::NodeId;
use rc3d_engine_api::Engine;
use rc3d_render::viewport::ViewportSplitAxis;
use rc3d_scene::node_data::MeasurementType;
use std::collections::{HashSet, VecDeque};

use crate::commands::EditorCommand;
use crate::selection::Selection;
use crate::ui::types::{EditorDisplayMode, NodeDataType, RenderFeatureFlags};

/// Editor-local interaction state (mirrors rc3d-app's EditorSession fields).
pub struct EditorInteractionState {
    pub gizmo_dragging: bool,
    pub gizmo_pending_transform: Option<(NodeId, Mat4)>,
    pub measurement_mode: bool,
    pub measurement_type: Option<MeasurementType>,
    pub section_edit_mode: bool,
    pub box_select_drag: bool,
    pub box_select_anchor: (f32, f32),
    pub view_split_drag: Option<ViewportSplitAxis>,
    pub left_pick_arm_pos: Option<(f64, f64)>,
    pub left_drag_suppresses_pick: bool,
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
            view_split_drag: None,
            left_pick_arm_pos: None,
            left_drag_suppresses_pick: false,
        }
    }
}

/// Runtime editor context wrapping the engine and all editor-local state.
pub struct EditorContext<'a> {
    pub engine: &'a mut Engine,
    pub commands: VecDeque<EditorCommand>,
    pub display_mode: EditorDisplayMode,
    pub selected_nodes: HashSet<NodeId>,
    pub hidden_nodes: HashSet<NodeId>,
    pub gizmo_mode: rc3d_gizmo::GizmoMode,
    pub render_features: RenderFeatureFlags,
    pub node_data_type: NodeDataType,
    pub interaction: EditorInteractionState,
    pub selection: Selection,
}

impl<'a> EditorContext<'a> {
    pub fn new(engine: &'a mut Engine) -> Self {
        Self::with_interaction(engine, EditorInteractionState::default())
    }

    pub fn with_interaction(
        engine: &'a mut Engine,
        interaction: EditorInteractionState,
    ) -> Self {
        Self {
            engine,
            commands: VecDeque::new(),
            display_mode: EditorDisplayMode::Shaded,
            selected_nodes: HashSet::new(),
            hidden_nodes: HashSet::new(),
            gizmo_mode: rc3d_gizmo::GizmoMode::Translate,
            render_features: RenderFeatureFlags::default(),
            node_data_type: NodeDataType::All,
            interaction,
            selection: Selection::new(),
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
