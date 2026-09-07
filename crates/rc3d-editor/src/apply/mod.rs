mod io;
mod render;
mod scene;
mod tools;

use rc3d_actions::CommandHistory;
use rc3d_engine_api::Engine;
use rc3d_scene::SceneGraph;

use crate::commands::{AdaptiveQualityMode, EditorCommand};
use crate::context::EditorInteractionState;
use crate::document;

/// Document undo stack plus markup tool state owned by the host.
pub struct EditorSession {
    pub history: CommandHistory,
    pub adaptive_quality_mode: AdaptiveQualityMode,
    pub document_path: Option<std::path::PathBuf>,
    pub dirty: bool,
    pub background: rc3d_engine_api::background::BackgroundSettings,
    pub ui_theme: crate::ui::theme::UiTheme,
    pub ui_locale: crate::ui::i18n::UiLocale,
    pub keymap: crate::keymap::Keymap,
    pub file_dialog_dir: Option<std::path::PathBuf>,
}

impl Default for EditorSession {
    fn default() -> Self {
        Self {
            history: CommandHistory::new(128),
            adaptive_quality_mode: AdaptiveQualityMode::Off,
            document_path: None,
            dirty: false,
            background: rc3d_engine_api::background::BackgroundSettings::default(),
            ui_theme: crate::ui::theme::UiTheme::Dark,
            ui_locale: crate::ui::i18n::UiLocale::En,
            keymap: crate::keymap::Keymap::standard(),
            file_dialog_dir: None,
        }
    }
}

impl EditorSession {
    pub fn execute_edit(&mut self, cmd: Box<dyn rc3d_actions::Command>, graph: &mut SceneGraph) {
        self.history.execute(cmd, graph);
        self.dirty = true;
    }

    pub fn window_title(&self) -> String {
        document::window_title(self.document_path.as_deref(), self.dirty)
    }
}

/// Apply an editor command to the engine, interaction, and document history.
pub fn apply_command(
    engine: &mut Engine,
    interaction: &mut EditorInteractionState,
    session: &mut EditorSession,
    cmd: EditorCommand,
) {
    match cmd {
        EditorCommand::NewScene
        | EditorCommand::OpenScene(_)
        | EditorCommand::SaveScene
        | EditorCommand::SaveSceneAs(_)
        | EditorCommand::ImportPath(_)
        | EditorCommand::ExportIvPath(_)
        | EditorCommand::Export3dPdf { .. }
        | EditorCommand::ExportDiagnosticsJsonPath(_)
        | EditorCommand::ExportScreenshot(_)
        | EditorCommand::ExportHiddenLineSvg(_)
        | EditorCommand::ExportQuadPack(_)
        | EditorCommand::LoadIblHdr(_) => io::apply(engine, interaction, session, cmd),

        EditorCommand::ApplyVisualStyle(_)
        | EditorCommand::SetViewportLayoutMode(_)
        | EditorCommand::CycleViewportLayout
        | EditorCommand::CycleActiveViewport
        | EditorCommand::SetViewPreset(_)
        | EditorCommand::SetViewFromDirection(_)
        | EditorCommand::OrbitView { .. }
        | EditorCommand::SetWboit(_)
        | EditorCommand::SetXrayMode(_)
        | EditorCommand::SetGhostUnselected(_)
        | EditorCommand::SetGhostOpacity(_)
        | EditorCommand::SetFillStyle(_)
        | EditorCommand::SetEdgeStyle(_)
        | EditorCommand::SetFeatureEdgeColor(_)
        | EditorCommand::SetWireframeEdgeColor(_)
        | EditorCommand::SetHiddenEdgeColor(_)
        | EditorCommand::SetCreaseAngle(_)
        | EditorCommand::SetSsEdgeThreshold(_)
        | EditorCommand::SetDisplayMode(_)
        | EditorCommand::SetGridEnabled(_)
        | EditorCommand::SetHudEnabled(_)
        | EditorCommand::SetVsyncEnabled(_)
        | EditorCommand::SetHdrPostProcessing(_)
        | EditorCommand::CycleIbl
        | EditorCommand::SetIblPreset(_)
        | EditorCommand::SetOutlineWidth(_)
        | EditorCommand::SetOutlineColor(_)
        | EditorCommand::SetCadDisplayTier(_)
        | EditorCommand::SetAdaptiveQualityMode(_)
        | EditorCommand::SetRenderFeature { .. }
        | EditorCommand::SetGpuCulling(_)
        | EditorCommand::SetParallelTraversal(_)
        | EditorCommand::SetCsmShadow { .. }
        | EditorCommand::SetInteractionRenderScale(_)
        | EditorCommand::SetBgMode(_)
        | EditorCommand::SetBgTopColor(_)
        | EditorCommand::SetBgBotColor(_)
        | EditorCommand::SetBgImage(_)
        | EditorCommand::SetPostEffects { .. }
        | EditorCommand::SetPostStylize { .. }
        | EditorCommand::CycleDisplayMode => render::apply(engine, interaction, session, cmd),

        EditorCommand::Undo
        | EditorCommand::Redo
        | EditorCommand::SetUiTheme(_)
        | EditorCommand::SetUiLocale(_)
        | EditorCommand::SetSelection(_)
        | EditorCommand::SetSelectionMany(_)
        | EditorCommand::FitSelection
        | EditorCommand::FitAll
        | EditorCommand::SetNodeVisibility(_, _)
        | EditorCommand::PreviewTransformTranslation(_, _)
        | EditorCommand::PreviewTransformScale(_, _)
        | EditorCommand::PreviewTransformRotationQuat(_, _)
        | EditorCommand::CommitTransformTranslation { .. }
        | EditorCommand::CommitTransformScale { .. }
        | EditorCommand::CommitTransformRotationQuat { .. }
        | EditorCommand::SetSectionPlaneEquation(_, _)
        | EditorCommand::SetSectionPlaneEnabled(_, _)
        | EditorCommand::SetBaseColor(_, _)
        | EditorCommand::SetMetallic(_, _)
        | EditorCommand::SetRoughness(_, _)
        | EditorCommand::SetOpacity(_, _)
        | EditorCommand::SetLightColor(_, _)
        | EditorCommand::SetLightIntensity(_, _)
        | EditorCommand::SetLightDirection(_, _)
        | EditorCommand::SetCameraFov(_, _)
        | EditorCommand::SetCameraNear(_, _)
        | EditorCommand::SetCameraFar(_, _)
        | EditorCommand::SetCameraReverseDepth(_, _)
        | EditorCommand::SetOrthoHeight(_, _)
        | EditorCommand::SetNodeField { .. }
        | EditorCommand::CreateNode { .. }
        | EditorCommand::DeleteNode(_)
        | EditorCommand::DuplicateNode(_)
        | EditorCommand::ReparentNodes { .. }
        | EditorCommand::RenameNode(_, _) => scene::apply(engine, interaction, session, cmd),

        EditorCommand::SetGizmoMode(_)
        | EditorCommand::SetWalkMode(_)
        | EditorCommand::ToggleSectionEdit
        | EditorCommand::SetSectionEdit(_)
        | EditorCommand::SetSelectKind(_)
        | EditorCommand::ToggleMeasurement
        | EditorCommand::SetMeasurementMode(_)
        | EditorCommand::SaveBookmark(_)
        | EditorCommand::RecallBookmark(_)
        | EditorCommand::SetMarkupTool(_)
        | EditorCommand::MarkupMouseDown { .. }
        | EditorCommand::MarkupMouseMove { .. }
        | EditorCommand::MarkupMouseUp { .. }
        | EditorCommand::ClearAllMarkup { .. }
        | EditorCommand::MeasurementPick { .. }
        | EditorCommand::CancelTool
        | EditorCommand::HideSelected
        | EditorCommand::IsolateSelected
        | EditorCommand::RevealHidden
        | EditorCommand::ToggleLockSelected
        | EditorCommand::ToggleLockNode(_)
        | EditorCommand::HistoryJump { .. }
        | EditorCommand::BindKey { .. } => tools::apply(engine, interaction, session, cmd),

        // Hosted by Studio (`cases::load_case` / param / step). No-op in library apply.
        EditorCommand::LoadCase(_)
        | EditorCommand::SetCaseParam { .. }
        | EditorCommand::RunCaseStep(_) => {}
    }
}
