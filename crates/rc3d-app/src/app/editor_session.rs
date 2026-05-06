use rc3d_actions::{CommandHistory, MarkupAction};
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_gizmo::Gizmo;
use rc3d_render::viewport::{ViewportId, ViewportSplitAxis};
use rc3d_scene::node_data::MeasurementType;

pub struct EditorSession {
    pub gizmo: Gizmo,
    pub gizmo_dragging: bool,
    pub gizmo_pending_transform: Option<(NodeId, Mat4)>,
    pub command_history: CommandHistory,
    pub markup_action: MarkupAction,
    pub measurement_mode: bool,
    pub measurement_type: Option<MeasurementType>,
    pub measurement_first_point: Option<Vec3>,
    pub measurements: Vec<(Vec3, Vec3, f32)>,
    pub axis_clip: [bool; 3],
    pub grid_enabled: bool,
    pub box_select_drag: bool,
    pub box_select_anchor: (f32, f32),
    pub section_edit_mode: bool,
    pub view_split_drag: Option<ViewportSplitAxis>,
    pub orbit_drag_viewport_id: Option<ViewportId>,
    pub left_orbit_drag_viewport_id: Option<ViewportId>,
    pub pan_drag_viewport_id: Option<ViewportId>,
    pub left_pick_arm_pos: Option<(f64, f64)>,
    pub left_drag_suppresses_pick: bool,
}
