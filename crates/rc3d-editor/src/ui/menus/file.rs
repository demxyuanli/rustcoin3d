//! File / Edit / Tools / Bookmarks / Settings menus.

use rc3d_gizmo::GizmoMode;

use crate::commands::EditorCommand;
use crate::ui::i18n::{t, UiLocale};
use crate::ui::theme::UiTheme;
use crate::ui::types::{EditorChromeState, EditorUiContext};

use super::km_item;

pub(super) fn file_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.file"), |ui| {
        km_item(ui, ui_ctx, crate::keymap::KeyAction::NewScene, push);
        if km_item(ui, ui_ctx, crate::keymap::KeyAction::OpenScene, push) {
            if let Some(path) = crate::document::pick_open_scene(ui_ctx.file_dialog_dir.as_deref())
            {
                push(EditorCommand::OpenScene(path));
            }
        }
        if ui.button(t(loc, "file.import_mesh")).clicked() {
            if let Some(path) = crate::document::pick_import_mesh(ui_ctx.file_dialog_dir.as_deref())
            {
                push(EditorCommand::ImportPath(path));
            }
        }
        ui.separator();
        if km_item(ui, ui_ctx, crate::keymap::KeyAction::SaveScene, push) {
            if ui_ctx.document_path.is_some() {
                push(EditorCommand::SaveScene);
            } else if let Some(path) =
                crate::document::pick_save_scene(ui_ctx.file_dialog_dir.as_deref())
            {
                push(EditorCommand::SaveSceneAs(path));
            }
        }
        if km_item(ui, ui_ctx, crate::keymap::KeyAction::SaveSceneAs, push) {
            if let Some(path) = crate::document::pick_save_scene(ui_ctx.file_dialog_dir.as_deref())
            {
                push(EditorCommand::SaveSceneAs(path));
            }
        }
        ui.separator();
        ui.menu_button(t(loc, "file.export"), |ui| {
            if ui.button(t(loc, "file.export_3d_pdf")).clicked() {
                // Remembered folder (last open/save/export, persisted across
                // sessions) seeds the picker; remembered view options seed
                // the options dialog that follows.
                let mut dlg = rfd::FileDialog::new()
                    .add_filter(t(loc, "filter.pdf"), &["pdf"])
                    .set_file_name("export.pdf");
                if let Some(dir) = ui_ctx
                    .file_dialog_dir
                    .as_deref()
                    .filter(|p| p.is_dir())
                {
                    dlg = dlg.set_directory(dir);
                }
                if let Some(mut path) = dlg.save_file() {
                    if path.extension().is_none() {
                        path.set_extension("pdf");
                    }
                    chrome.export3d = Some(crate::ui::types::Export3dPdfDialogState {
                        path,
                        options: chrome.last_pdf_options,
                    });
                }
            }
            if ui.button(t(loc, "file.export_scene")).clicked() {
                if let Some(path) =
                    crate::document::pick_save_scene(ui_ctx.file_dialog_dir.as_deref())
                {
                    push(EditorCommand::ExportIvPath(path));
                }
            }
            if ui.button(t(loc, "file.export_diagnostics")).clicked() {
                if let Some(path) = rfd::FileDialog::new().save_file() {
                    push(EditorCommand::ExportDiagnosticsJsonPath(path));
                }
            }
            if ui.button(t(loc, "file.export_screenshot")).clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter(t(loc, "filter.png"), &["png"])
                    .save_file()
                {
                    push(EditorCommand::ExportScreenshot(path));
                }
            }
            if ui.button(t(loc, "file.export_svg")).clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter(t(loc, "filter.svg"), &["svg"])
                    .save_file()
                {
                    push(EditorCommand::ExportHiddenLineSvg(path));
                }
            }
            if ui.button(t(loc, "file.export_quad")).clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter(t(loc, "filter.png"), &["png"])
                    .save_file()
                {
                    push(EditorCommand::ExportQuadPack(path));
                }
            }
        });
        if chrome.caption.enabled {
            ui.separator();
            if ui.button(t(loc, "file.exit")).clicked() {
                if chrome.caption.dirty {
                    chrome.caption.close_prompt = true;
                } else {
                    chrome.caption.action = Some(crate::ui::types::CaptionAction::Close);
                }
            }
        }
    });
}

pub(super) fn edit_menu(ui: &mut egui::Ui, ui_ctx: &EditorUiContext, push: &mut impl FnMut(EditorCommand)) {
    let loc = ui_ctx.ui_locale;
    use crate::keymap::KeyAction;
    ui.menu_button(t(loc, "menu.edit"), |ui| {
        km_item(ui, ui_ctx, KeyAction::Undo, push);
        km_item(ui, ui_ctx, KeyAction::Redo, push);
        ui.separator();
        km_item(ui, ui_ctx, KeyAction::FitSelection, push);
        km_item(ui, ui_ctx, KeyAction::FitAll, push);
        ui.separator();
        km_item(ui, ui_ctx, KeyAction::HideSelected, push);
        km_item(ui, ui_ctx, KeyAction::IsolateSelected, push);
        km_item(ui, ui_ctx, KeyAction::RevealHidden, push);
        km_item(ui, ui_ctx, KeyAction::ToggleLock, push);
    });
}

pub(super) fn tools_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    _chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.tools"), |ui| {
        if ui.button(t(loc, "tools.measure")).clicked() {
            push(EditorCommand::ToggleMeasurement);
        }
        if ui.button(t(loc, "tools.section")).clicked() {
            push(EditorCommand::ToggleSectionEdit);
        }
        ui.menu_button(t(loc, "tools.measurement"), |ui| {
            if ui.button(t(loc, "tools.off")).clicked() {
                push(EditorCommand::SetMeasurementMode(None));
            }
            for (m, key) in [
                (
                    rc3d_scene::node_data::MeasurementType::Distance,
                    "tools.distance",
                ),
                (rc3d_scene::node_data::MeasurementType::Angle, "tools.angle"),
                (
                    rc3d_scene::node_data::MeasurementType::Radius,
                    "tools.radius",
                ),
                (
                    rc3d_scene::node_data::MeasurementType::Diameter,
                    "tools.diameter",
                ),
            ] {
                if ui.button(t(loc, key)).clicked() {
                    push(EditorCommand::SetMeasurementMode(Some(m)));
                }
            }
        });
        ui.menu_button(t(loc, "tools.markup"), |ui| {
            for (tool, key) in [
                (rc3d_actions::MarkupTool::Select, "tools.select"),
                (rc3d_actions::MarkupTool::Line, "tools.line"),
                (rc3d_actions::MarkupTool::Rect, "tools.rect"),
                (rc3d_actions::MarkupTool::Circle, "tools.circle"),
                (rc3d_actions::MarkupTool::Freehand, "tools.freehand"),
            ] {
                if ui.button(t(loc, key)).clicked() {
                    push(EditorCommand::SetMarkupTool(tool));
                }
            }
        });
        ui.separator();
        let mut walk = ui_ctx.walk_mode;
        if ui.checkbox(&mut walk, t(loc, "tools.walk")).changed() {
            push(EditorCommand::SetWalkMode(walk));
        }
        if ui.button(t(loc, "edit.fit_all")).clicked() {
            push(EditorCommand::FitAll);
        }
        ui.menu_button(t(loc, "tools.gizmo"), |ui| {
            if ui.button(t(loc, "tools.move")).clicked() {
                push(EditorCommand::SetGizmoMode(GizmoMode::Translate));
            }
            if ui.button(t(loc, "tools.rotate")).clicked() {
                push(EditorCommand::SetGizmoMode(GizmoMode::Rotate));
            }
            if ui.button(t(loc, "tools.scale")).clicked() {
                push(EditorCommand::SetGizmoMode(GizmoMode::Scale));
            }
        });
    });
}

/// Menu-bar Settings entry. Rendered only while the caption bar is disabled
/// (see [`crate::ui::menus::menu_bar`]); Studio-style hosts open the same
/// content from the caption-bar gear in a small panel window.
pub(super) fn settings_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.settings"), |ui| {
        settings_body(ui, ui_ctx, chrome, push, KeymapListHeight::Fixed(240.0));
    });
}

/// Height budget for the keymap shortcut list in the shared settings body.
#[derive(Clone, Copy, Debug)]
pub(in crate::ui) enum KeymapListHeight {
    /// Menu popup: fixed cap; the list scrolls beyond it.
    Fixed(f32),
    /// Caption-bar panel: fill whatever window height is left.
    Fill,
}

/// Shared Settings content (theme / language / keymap). Used both inside the
/// menu-bar popup and in the caption-bar gear panel window.
pub(in crate::ui) fn settings_body(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
    keymap_height: KeymapListHeight,
) {
    let loc = ui_ctx.ui_locale;
    ui.set_min_width(300.0);
    ui.menu_button(t(loc, "settings.theme"), |ui| {
        for (theme, key) in [
            (UiTheme::Dark, "settings.theme.dark"),
            (UiTheme::Light, "settings.theme.light"),
        ] {
            let on = ui_ctx.ui_theme == theme;
            if ui.selectable_label(on, t(loc, key)).clicked() {
                push(EditorCommand::SetUiTheme(theme));
            }
        }
    });
    ui.menu_button(t(loc, "settings.language"), |ui| {
        for (locale, key) in [
            (UiLocale::En, "settings.language.en"),
            (UiLocale::ZhHans, "settings.language.zh"),
        ] {
            let on = ui_ctx.ui_locale == locale;
            if ui.selectable_label(on, t(loc, key)).clicked() {
                push(EditorCommand::SetUiLocale(locale));
            }
        }
    });
    ui.separator();
    ui.label(t(loc, "settings.keymap"));
    if let Some(action) = chrome.keymap_capture {
        ui.label(format!(
            "{}: {}",
            t(loc, "settings.keymap_press"),
            t(loc, action.i18n_key())
        ));
    }
    let mut scroll = egui::ScrollArea::vertical();
    scroll = match keymap_height {
        KeymapListHeight::Fixed(max) => scroll.max_height(max),
        KeymapListHeight::Fill => scroll
            .auto_shrink([false, false])
            .max_height(ui.available_height().max(160.0)),
    };
    scroll.show(ui, |ui| {
        for action in crate::keymap::KeyAction::ALL {
            ui.horizontal(|ui| {
                ui.label(t(ui_ctx.ui_locale, action.i18n_key()));
                let chord = ui_ctx
                    .keymap
                    .shortcut_label(*action)
                    .unwrap_or_else(|| "-".into());
                if ui.small_button(chord).clicked() {
                    chrome.keymap_capture = Some(*action);
                }
            });
        }
    });
}

pub(super) fn bookmarks_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.bookmarks"), |ui| {
        ui.label(t(loc, "bm.hint"));
        ui.separator();
        for i in 0..9 {
            let has = ui_ctx.bookmarks[i].0;
            let label = ui_ctx.bookmarks[i].1;
            ui.horizontal(|ui| {
                if ui.button(format!("{} {i}", t(loc, "bm.save"))).clicked() {
                    push(EditorCommand::SaveBookmark(i));
                }
                if ui
                    .button(format!("{} {i}  {label}", t(loc, "bm.recall")))
                    .clicked()
                {
                    push(EditorCommand::RecallBookmark(i));
                }
                if has {
                    ui.label("\u{25CF}");
                }
            });
        }
    });
}
