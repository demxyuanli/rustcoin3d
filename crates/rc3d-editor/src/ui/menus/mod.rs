//! Top menu bar: File / Edit / View / Display / Render / Tools / Bookmarks / Create / Settings.

mod create;
mod display;
mod file;
mod labels;
mod render;
mod view;

use crate::commands::EditorCommand;
use crate::ui::i18n::{t, UiLocale};
use crate::ui::types::{EditorChromeState, EditorUiContext};

pub(in crate::ui) use create::create_node_menu;
pub(in crate::ui) use file::{settings_body, KeymapListHeight};
pub(in crate::ui) use labels::{
    aq_mode_label, cad_tier_label, hierarchy_node_context_actions, layout_mode_label,
    selection_context_actions, view_preset_label, view_render_features_menu,
};

pub(super) fn menu_bar(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    egui::MenuBar::new().ui(ui, |ui| {
        file::file_menu(ui, ui_ctx, chrome, push);
        file::edit_menu(ui, ui_ctx, push);
        view::view_menu(ui, ui_ctx, chrome, push);
        display::display_menu(ui, ui_ctx, push);
        render::render_menu(ui, ui_ctx, push);
        file::tools_menu(ui, ui_ctx, chrome, push);
        file::bookmarks_menu(ui, ui_ctx, push);
        ui.menu_button(t(loc, "menu.create"), |ui| {
            create_node_menu(ui, None, loc, push);
        });
        // With a caption bar the Settings gear on the title bar replaces the
        // menu entry; without one (embedded/library hosts) keep it here.
        if !chrome.caption.enabled {
            file::settings_menu(ui, ui_ctx, chrome, push);
        }
    });
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Button row that pushes a fixed command when clicked.
pub(super) fn cmd_item(
    ui: &mut egui::Ui,
    loc: UiLocale,
    key: &'static str,
    cmd: EditorCommand,
    push: &mut impl FnMut(EditorCommand),
) {
    if ui.button(t(loc, key)).clicked() {
        push(cmd);
    }
}

/// Button row labeled via the keymap (label + shortcut), pushing the action's
/// default command. Returns true when clicked.
pub(super) fn km_item(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    action: crate::keymap::KeyAction,
    push: &mut impl FnMut(EditorCommand),
) -> bool {
    if ui
        .button(ui_ctx.keymap.menu_label(ui_ctx.ui_locale, action))
        .clicked()
    {
        if let Some(cmd) = crate::keymap::command_for(action) {
            push(cmd);
        }
        return true;
    }
    false
}

/// Slider row bound to a fixed range, reporting the new value.
pub(super) fn post_slider(
    ui: &mut egui::Ui,
    value: f32,
    label: &str,
    min: f32,
    max: f32,
    mut on_change: impl FnMut(f32),
) {
    let mut v = value;
    if ui
        .add(egui::Slider::new(&mut v, min..=max).text(label))
        .changed()
    {
        on_change(v);
    }
}

/// Color edit row reporting the new RGBA value.
pub(super) fn color_cmd(
    ui: &mut egui::Ui,
    rgba: [f32; 4],
    label: &str,
    mut on_change: impl FnMut([f32; 4]),
) {
    let mut c = rgba;
    ui.horizontal(|ui| {
        ui.label(label);
        if ui.color_edit_button_rgba_unmultiplied(&mut c).changed() {
            on_change(c);
        }
    });
}
