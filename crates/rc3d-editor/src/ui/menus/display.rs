//! Display menu: shading, fill, edges, visual styles, visibility.

use rc3d_core::{EdgeStyle, FillStyle};

use crate::commands::EditorCommand;
use crate::ui::i18n::{t, UiLocale};
use crate::ui::types::{EditorDisplayMode, EditorUiContext};

use super::color_cmd;

pub(super) fn display_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.display"), |ui| {
        ui.menu_button(t(loc, "display.shading"), |ui| {
            display_shading_items(ui, loc, push);
        });
        ui.menu_button(t(loc, "display.fill"), |ui| {
            display_fill_items(ui, loc, push);
        });
        ui.menu_button(t(loc, "display.edges"), |ui| {
            ui.menu_button(t(loc, "display.edges.style"), |ui| {
                display_edge_style_items(ui, loc, push);
            });
            ui.menu_button(t(loc, "display.edges.colors"), |ui| {
                display_edge_color_items(ui, ui_ctx, loc, push);
            });
            ui.menu_button(t(loc, "display.edges.screen"), |ui| {
                display_edge_screen_items(ui, ui_ctx, loc, push);
            });
        });
        ui.menu_button(t(loc, "display.visual_style"), |ui| {
            for name in rc3d_core::VisualStyleLibrary::builtin().names() {
                if ui.button(name).clicked() {
                    push(EditorCommand::ApplyVisualStyle(name.to_string()));
                }
            }
        });
        ui.menu_button(t(loc, "display.visibility"), |ui| {
            display_visibility_items(ui, ui_ctx, loc, push);
        });
    });
}

fn display_shading_items(ui: &mut egui::Ui, loc: UiLocale, push: &mut impl FnMut(EditorCommand)) {
    for (mode, key) in [
        (EditorDisplayMode::Wireframe, "shade.wireframe"),
        (EditorDisplayMode::Shaded, "shade.shaded"),
        (EditorDisplayMode::ShadedWithEdges, "shade.shaded_edges"),
        (EditorDisplayMode::HiddenLine, "shade.hidden"),
        (EditorDisplayMode::Flat, "shade.flat"),
        (EditorDisplayMode::FlatWithEdge, "shade.flat_edges"),
    ] {
        super::cmd_item(ui, loc, key, EditorCommand::SetDisplayMode(mode), push);
    }
}

fn display_fill_items(ui: &mut egui::Ui, loc: UiLocale, push: &mut impl FnMut(EditorCommand)) {
    for (fill, key) in [
        (FillStyle::Shaded, "fill.shaded"),
        (FillStyle::Flat, "fill.flat"),
        (FillStyle::HiddenLine, "fill.hidden"),
        (FillStyle::None, "fill.none"),
    ] {
        super::cmd_item(ui, loc, key, EditorCommand::SetFillStyle(fill), push);
    }
}

fn display_edge_style_items(
    ui: &mut egui::Ui,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    for (edges, key) in [
        (EdgeStyle::None, "edge.none"),
        (EdgeStyle::Crease, "edge.crease"),
        (EdgeStyle::Silhouette, "edge.silhouette"),
        (EdgeStyle::Full, "edge.full"),
        (EdgeStyle::Perimeter, "edge.perimeter"),
        (EdgeStyle::Hard, "edge.hard"),
        (EdgeStyle::Adjacent, "edge.adjacent"),
    ] {
        super::cmd_item(ui, loc, key, EditorCommand::SetEdgeStyle(edges), push);
    }
}

fn display_edge_color_items(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    color_cmd(ui, ui_ctx.feature_edge_color, t(loc, "edge.feature"), |c| {
        push(EditorCommand::SetFeatureEdgeColor(c));
    });
    color_cmd(
        ui,
        ui_ctx.wireframe_edge_color,
        t(loc, "edge.wireframe"),
        |c| {
            push(EditorCommand::SetWireframeEdgeColor(c));
        },
    );
    color_cmd(ui, ui_ctx.hidden_edge_color, t(loc, "edge.hidden"), |c| {
        push(EditorCommand::SetHiddenEdgeColor(c));
    });
}

fn display_edge_screen_items(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let mut crease = ui_ctx.crease_angle;
    if ui
        .add(egui::Slider::new(&mut crease, 1.0_f32..=90.0_f32).text(t(loc, "edge.crease_deg")))
        .changed()
    {
        push(EditorCommand::SetCreaseAngle(crease));
    }
    let mut ss = ui_ctx.render_features.screen_space_edges;
    if ui.checkbox(&mut ss, t(loc, "edge.ss")).changed() {
        push(EditorCommand::SetRenderFeature {
            feature_name: "screen_space_edges",
            enabled: ss,
        });
    }
    let mut outline = ui_ctx.render_features.screen_space_selection_outline;
    if ui
        .checkbox(&mut outline, t(loc, "edge.ss_outline"))
        .changed()
    {
        push(EditorCommand::SetRenderFeature {
            feature_name: "screen_space_selection_outline",
            enabled: outline,
        });
    }
    let mut thr = ui_ctx.ss_edge_threshold;
    if ui
        .add(egui::Slider::new(&mut thr, 0.001_f32..=0.08_f32).text(t(loc, "edge.ss_thr")))
        .changed()
    {
        push(EditorCommand::SetSsEdgeThreshold(thr));
    }
}

fn display_visibility_items(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let mut xray = ui_ctx.xray_mode;
    if ui.checkbox(&mut xray, t(loc, "vis.xray")).changed() {
        push(EditorCommand::SetXrayMode(xray));
    }
    let mut ghost = ui_ctx.ghost_unselected;
    if ui.checkbox(&mut ghost, t(loc, "vis.ghost")).changed() {
        push(EditorCommand::SetGhostUnselected(ghost));
    }
    let mut go = ui_ctx.ghost_opacity;
    if ui
        .add(egui::Slider::new(&mut go, 0.02_f32..=0.95_f32).text(t(loc, "vis.ghost_opacity")))
        .changed()
    {
        push(EditorCommand::SetGhostOpacity(go));
    }
}
