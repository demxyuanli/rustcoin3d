//! View menu: camera, layout, overlays, panels, workspace, background.

use rc3d_render::viewport::LayoutMode;

use crate::commands::EditorCommand;
use crate::ui::i18n::{t, UiLocale};
use crate::ui::types::{BottomTab, EditorChromeState, EditorUiContext, SideTab};

use super::color_cmd;

pub(super) fn view_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.view"), |ui| {
        ui.menu_button(t(loc, "view.camera"), |ui| {
            view_camera_items(ui, loc, push);
        });
        ui.menu_button(t(loc, "view.layout"), |ui| {
            view_layout_items(ui, ui_ctx, loc, push);
        });
        ui.menu_button(t(loc, "view.overlays"), |ui| {
            view_overlay_items(ui, ui_ctx, loc, push);
        });
        ui.menu_button(t(loc, "view.panels"), |ui| {
            view_panel_items(ui, chrome, loc);
        });
        ui.menu_button(t(loc, "view.workspace"), |ui| {
            view_workspace_items(ui, chrome, loc);
        });
        ui.menu_button(t(loc, "view.background"), |ui| {
            view_background_items(ui, ui_ctx, loc, push);
        });
    });
}

fn view_camera_items(ui: &mut egui::Ui, loc: UiLocale, push: &mut impl FnMut(EditorCommand)) {
    use rc3d_engine_api::camera::ViewPreset;
    for (preset, key) in [
        (ViewPreset::Top, "view.top"),
        (ViewPreset::Front, "view.front"),
        (ViewPreset::Right, "view.right"),
        (ViewPreset::Iso, "view.iso"),
        (ViewPreset::Bottom, "view.bottom"),
        (ViewPreset::Back, "view.back"),
        (ViewPreset::Left, "view.left"),
    ] {
        super::cmd_item(ui, loc, key, EditorCommand::SetViewPreset(preset), push);
    }
}

fn view_layout_items(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    for (mode, key) in [
        (LayoutMode::Single, "layout.single"),
        (LayoutMode::Quad, "layout.quad"),
        (LayoutMode::LeftRight, "layout.leftright"),
        (LayoutMode::TopBottom, "layout.topbottom"),
    ] {
        let on = ui_ctx.layout_mode == mode;
        if ui.selectable_label(on, t(loc, key)).clicked() {
            push(EditorCommand::SetViewportLayoutMode(mode));
        }
    }
    ui.separator();
    if ui.button(t(loc, "view.next_vp")).clicked() {
        push(EditorCommand::CycleActiveViewport);
    }
}

fn view_overlay_items(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let mut grid = ui_ctx.grid_enabled;
    if ui.checkbox(&mut grid, t(loc, "view.grid")).changed() {
        push(EditorCommand::SetGridEnabled(grid));
    }
    let mut hud = ui_ctx.hud_enabled;
    if ui.checkbox(&mut hud, t(loc, "view.hud")).changed() {
        push(EditorCommand::SetHudEnabled(hud));
    }
}

fn view_panel_items(ui: &mut egui::Ui, chrome: &mut EditorChromeState, loc: UiLocale) {
    side_tab_check(ui, chrome, SideTab::Hierarchy, t(loc, "view.hierarchy"));
    side_tab_check(ui, chrome, SideTab::Render, t(loc, "view.render_panel"));
    side_tab_check(ui, chrome, SideTab::History, t(loc, "view.history"));
    side_tab_check(ui, chrome, SideTab::Assets, t(loc, "view.assets"));
    ui.separator();
    bottom_tab_check(ui, chrome, BottomTab::Document, t(loc, "view.document"));
    bottom_tab_check(ui, chrome, BottomTab::Compositor, t(loc, "view.compositor"));
}

fn view_workspace_items(ui: &mut egui::Ui, chrome: &mut EditorChromeState, loc: UiLocale) {
    for (ws, key) in [
        (crate::ui::types::Workspace::Model, "ws.model"),
        (crate::ui::types::Workspace::LookDev, "ws.lookdev"),
        (crate::ui::types::Workspace::Compositor, "ws.compositor"),
    ] {
        if ui
            .selectable_label(chrome.workspace == ws, t(loc, key))
            .clicked()
        {
            chrome.apply_workspace(ws);
        }
    }
}

fn view_background_items(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    for (mode, key) in [
        (rc3d_render::background::BgMode::Solid, "view.bg.solid"),
        (
            rc3d_render::background::BgMode::VerticalGradient,
            "view.bg.vgrad",
        ),
        (
            rc3d_render::background::BgMode::HorizontalGradient,
            "view.bg.hgrad",
        ),
        (rc3d_render::background::BgMode::SkyGround, "view.bg.sky"),
        (rc3d_render::background::BgMode::Image, "view.bg.image"),
    ] {
        let on = ui_ctx.bg_mode == mode;
        if ui.selectable_label(on, t(loc, key)).clicked() {
            push(EditorCommand::SetBgMode(mode));
        }
    }
    ui.separator();
    color_cmd(ui, ui_ctx.bg_top, t(loc, "view.bg.top"), |c| {
        push(EditorCommand::SetBgTopColor(c));
    });
    color_cmd(ui, ui_ctx.bg_bot, t(loc, "view.bg.bot"), |c| {
        push(EditorCommand::SetBgBotColor(c));
    });
    if ui.button(t(loc, "view.bg.image_pick")).clicked() {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(
                t(loc, "filter.image"),
                &["png", "jpg", "jpeg", "hdr", "exr"],
            )
            .pick_file()
        {
            push(EditorCommand::SetBgImage(path));
        }
    }
}

fn bottom_tab_check(
    ui: &mut egui::Ui,
    chrome: &mut EditorChromeState,
    tab: BottomTab,
    label: &str,
) {
    let mut on = chrome.bottom_tab == Some(tab);
    if ui.checkbox(&mut on, label).changed() {
        chrome.set_bottom_tab(tab, on);
    }
}

fn side_tab_check(ui: &mut egui::Ui, chrome: &mut EditorChromeState, tab: SideTab, label: &str) {
    let mut on = chrome.side_tab == Some(tab);
    if ui.checkbox(&mut on, label).changed() {
        chrome.set_side_tab(tab, on);
    }
}
