use std::path::PathBuf;

use rc3d_actions::MarkupTool;
use rc3d_core::NodeId;
use rc3d_gizmo::GizmoMode;
use rc3d_render::ibl::IblPreset;
use rc3d_render::renderer::CadDisplayTier;
use rc3d_render::viewport::LayoutMode;
use rc3d_scene::node_data::MeasurementType;

use crate::ui::types::{EditorDisplayMode, NodeDataType};

/// Editor-local adaptive quality mode (mirrors rc3d-app::adaptive_quality).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveQualityMode {
    Off,
    On,
    AutoIdleLock,
}

#[derive(Debug)]
pub enum EditorCommand {
    Undo,
    Redo,
    FitSelection,
    ToggleMeasurement,
    ToggleSectionEdit,
    CycleIbl,
    CycleViewportLayout,
    CycleActiveViewport,
    SetDisplayMode(EditorDisplayMode),
    SetSelection(Option<NodeId>),
    SetNodeVisibility(NodeId, bool),
    PreviewTransformTranslation(NodeId, [f32; 3]),
    CommitTransformTranslation {
        node: NodeId,
        old: [f32; 3],
        new: [f32; 3],
    },
    PreviewTransformScale(NodeId, [f32; 3]),
    CommitTransformScale {
        node: NodeId,
        old: [f32; 3],
        new: [f32; 3],
    },
    PreviewTransformRotationQuat(NodeId, [f32; 4]),
    CommitTransformRotationQuat {
        node: NodeId,
        old: [f32; 4],
        new: [f32; 4],
    },
    ImportPath(PathBuf),
    ExportIvPath(PathBuf),
    ExportDiagnosticsJsonPath(PathBuf),
    SetGizmoMode(GizmoMode),
    SetRenderFeature {
        feature_name: &'static str,
        enabled: bool,
    },
    SetHdrPostProcessing(bool),
    SetIblPreset(IblPreset),
    SetAdaptiveQualityMode(AdaptiveQualityMode),
    SetOutlineWidth(f32),
    SetOutlineColor([f32; 4]),
    SetXrayMode(bool),
    SetViewportLayoutMode(LayoutMode),
    SetViewPreset(rc3d_engine_api::camera::ViewPreset),
    SetGridEnabled(bool),
    SetHudEnabled(bool),
    SetVsyncEnabled(bool),
    SetCadDisplayTier(CadDisplayTier),
    SetBaseColor(NodeId, [f32; 3]),
    SetMetallic(NodeId, f32),
    SetRoughness(NodeId, f32),
    SetOpacity(NodeId, f32),
    SetLightColor(NodeId, [f32; 3]),
    SetLightIntensity(NodeId, f32),
    SetLightDirection(NodeId, [f32; 3]),
    SetCameraFov(NodeId, f32),
    SetCameraNear(NodeId, f32),
    SetCameraFar(NodeId, f32),
    SetCameraReverseDepth(NodeId, bool),
    SetOrthoHeight(NodeId, f32),
    SetSectionPlaneEnabled(NodeId, bool),
    SetSectionPlaneEquation(NodeId, [f32; 4]),
    CreateNode {
        node_type: NodeDataType,
        parent: Option<NodeId>,
    },
    DeleteNode(NodeId),
    DuplicateNode(NodeId),
    RenameNode(NodeId, String),
    SetMeasurementMode(Option<MeasurementType>),
    SaveBookmark(usize),
    RecallBookmark(usize),
    SetMarkupTool(MarkupTool),
    MarkupMouseDown {
        screen_pos: [f32; 2],
    },
    MarkupMouseMove {
        screen_pos: [f32; 2],
    },
    MarkupMouseUp {
        screen_pos: [f32; 2],
    },
    ClearAllMarkup {
        node: NodeId,
    },
}
