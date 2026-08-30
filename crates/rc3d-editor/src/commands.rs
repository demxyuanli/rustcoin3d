use std::path::PathBuf;

use rc3d_actions::MarkupTool;
use rc3d_core::{EdgeStyle, FillStyle, NodeId};
use rc3d_gizmo::GizmoMode;
use rc3d_render::background::BgMode;
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
    FitAll,
    ToggleMeasurement,
    ToggleSectionEdit,
    CycleIbl,
    CycleViewportLayout,
    CycleActiveViewport,
    SetDisplayMode(EditorDisplayMode),
    SetSelection(Option<NodeId>),
    SetSelectionMany(Vec<NodeId>),
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
    NewScene,
    OpenScene(PathBuf),
    SaveScene,
    SaveSceneAs(PathBuf),
    ImportPath(PathBuf),
    ExportIvPath(PathBuf),
    ExportDiagnosticsJsonPath(PathBuf),
    ExportScreenshot(PathBuf),
    ExportHiddenLineSvg(PathBuf),
    ExportQuadPack(PathBuf),
    LoadIblHdr(PathBuf),
    SetGpuCulling(bool),
    SetParallelTraversal(bool),
    SetCsmShadow {
        resolution: u32,
        cascade_count: u32,
    },
    SetInteractionRenderScale(f32),
    SetWalkMode(bool),
    SetBgMode(BgMode),
    SetBgTopColor([f32; 4]),
    SetBgBotColor([f32; 4]),
    SetBgImage(PathBuf),
    SetPostEffects {
        vignette: f32,
        chromatic: f32,
        bloom: f32,
        grain: f32,
    },
    SetPostStylize {
        halftone: f32,
        glitch: f32,
    },
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
    SetGhostUnselected(bool),
    SetGhostOpacity(f32),
    SetFillStyle(FillStyle),
    SetEdgeStyle(EdgeStyle),
    SetFeatureEdgeColor([f32; 4]),
    SetWireframeEdgeColor([f32; 4]),
    SetHiddenEdgeColor([f32; 4]),
    SetCreaseAngle(f32),
    SetSsEdgeThreshold(f32),
    SetWboit(bool),
    ApplyVisualStyle(String),
    SetViewportLayoutMode(LayoutMode),
    SetViewPreset(rc3d_engine_api::camera::ViewPreset),
    SetUiTheme(crate::ui::theme::UiTheme),
    SetUiLocale(crate::ui::i18n::UiLocale),
    SetViewFromDirection([f32; 3]),
    OrbitView { dx: f32, dy: f32 },
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
    ReparentNodes {
        ids: Vec<NodeId>,
        parent: Option<NodeId>,
        index: usize,
    },
    RenameNode(NodeId, String),
    SetMeasurementMode(Option<MeasurementType>),
    SaveBookmark(usize),
    RecallBookmark(usize),
    SetNodeField {
        node: NodeId,
        field_index: u16,
        value: rc3d_fields::FieldValue,
    },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_command_display_mode_variant_constructs() {
        let cmd = EditorCommand::SetDisplayMode(EditorDisplayMode::Shaded);
        // Verify construction succeeds without panic (the primary goal).
        // Also verify the variant matches.
        match &cmd {
            EditorCommand::SetDisplayMode(mode) => assert_eq!(*mode, EditorDisplayMode::Shaded),
            _ => panic!("expected SetDisplayMode, got different variant"),
        }
    }

    #[test]
    fn editor_command_all_display_mode_variants() {
        for mode in &[
            EditorDisplayMode::Wireframe,
            EditorDisplayMode::Shaded,
            EditorDisplayMode::ShadedWithEdges,
            EditorDisplayMode::HiddenLine,
            EditorDisplayMode::Flat,
            EditorDisplayMode::FlatWithEdge,
        ] {
            let cmd = EditorCommand::SetDisplayMode(*mode);
            match cmd {
                EditorCommand::SetDisplayMode(m) => assert_eq!(m, *mode),
                _ => panic!("unexpected variant"),
            }
        }
    }
}
