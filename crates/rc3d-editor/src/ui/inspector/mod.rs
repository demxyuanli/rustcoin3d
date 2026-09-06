//! Inspector side tab: props tabs (object/material/display/camera) plus the
//! render settings panel (`render.rs`) and node field table (`fields.rs`).

mod fields;
mod render;

use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

use crate::commands::EditorCommand;
use crate::ui::i18n::{t, UiLocale};
use crate::ui::icons::{self, Icon, PANEL_BTN};
use crate::ui::types::{EditorChromeState, EditorDisplayMode, EditorUiContext, PropsTab};

pub(in crate::ui) use render::draw_render_panel;

pub(super) fn draw_inspector_tab(
    ui: &mut egui::Ui,
    graph: &SceneGraph,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    let pal = ui_ctx.ui_theme.palette();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0_f32;
        for (tab, key) in [
            (PropsTab::Object, "props.object"),
            (PropsTab::Material, "props.material"),
            (PropsTab::Display, "props.display"),
            (PropsTab::Camera, "props.camera"),
        ] {
            if icons::tab_button(
                ui,
                icons::props_tab_icon(tab),
                t(loc, key),
                chrome.props_tab == tab,
                &pal,
            )
            .clicked()
            {
                chrome.props_tab = tab;
            }
        }
    });
    ui.separator();
    egui::ScrollArea::vertical()
        .id_salt("inspector_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let Some(id) = ui_ctx.selected.iter().next().copied() else {
                ui.label(t(loc, "panel.no_sel"));
                return;
            };
            let Some(entry) = graph.get(id) else {
                return;
            };
            ui.label(format!("{} {:?}", t(loc, "panel.node"), id));
            match chrome.props_tab {
                PropsTab::Object => {
                    inspector_object(ui, id, &entry.data, graph, loc, push);
                }
                PropsTab::Material => {
                    inspector_kind(ui, id, &entry.data, graph, loc, push, is_material)
                }
                PropsTab::Display => {
                    inspector_display(ui, id, &entry.data, ui_ctx, loc, push);
                }
                PropsTab::Camera => {
                    inspector_kind(ui, id, &entry.data, graph, loc, push, is_camera)
                }
            }
            if let Some(d) = ui_ctx.diagnostics.as_ref() {
                ui.collapsing(t(loc, "panel.diagnostics"), |ui| {
                    ui.monospace(format!("frame {}", d.frame_index));
                    for (n, t) in &d.gpu_pass_timings_us {
                        ui.monospace(format!("{n}: {t:.1} us"));
                    }
                });
            }
        });
}

fn is_material(data: &NodeData) -> bool {
    matches!(data, NodeData::Material(_))
}

fn is_camera(data: &NodeData) -> bool {
    matches!(
        data,
        NodeData::PerspectiveCamera(_)
            | NodeData::OrthographicCamera(_)
            | NodeData::StereoCamera(_)
    )
}

fn inspector_kind(
    ui: &mut egui::Ui,
    id: NodeId,
    data: &NodeData,
    graph: &SceneGraph,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
    ok: fn(&NodeData) -> bool,
) {
    if !ok(data) {
        ui.label(t(loc, "props.wrong_type"));
        return;
    }
    fields::inspector_fields(ui, id, data, graph, push);
}

fn inspector_object(
    ui: &mut egui::Ui,
    id: NodeId,
    data: &NodeData,
    graph: &SceneGraph,
    _loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    fields::inspector_fields(ui, id, data, graph, push);
}

fn inspector_display(
    ui: &mut egui::Ui,
    id: NodeId,
    data: &NodeData,
    ui_ctx: &EditorUiContext,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let mut vis = match data {
        NodeData::Switch(sw) => sw.which_child >= 0,
        _ => !ui_ctx.hidden_nodes.contains(&id),
    };
    if ui.checkbox(&mut vis, t(loc, "insp.visible")).changed() {
        push(EditorCommand::SetNodeVisibility(id, vis));
    }
    let locked_on = ui_ctx.locked_nodes.contains(&id);
    let pal = ui_ctx.ui_theme.palette();
    if icons::icon_button(
        ui,
        Icon::Lock,
        t(loc, "key.lock"),
        locked_on,
        PANEL_BTN,
        &pal,
    )
    .clicked()
    {
        push(EditorCommand::ToggleLockNode(id));
    }
    ui.separator();
    ui.label(t(loc, "display.shading"));
    for (mode, key) in [
        (EditorDisplayMode::Wireframe, "shade.wireframe"),
        (EditorDisplayMode::Shaded, "shade.shaded"),
        (EditorDisplayMode::ShadedWithEdges, "shade.shaded_edges"),
        (EditorDisplayMode::HiddenLine, "shade.hidden"),
        (EditorDisplayMode::Flat, "shade.flat"),
        (EditorDisplayMode::FlatWithEdge, "shade.flat_edges"),
    ] {
        if ui.button(t(loc, key)).clicked() {
            push(EditorCommand::SetDisplayMode(mode));
        }
    }
    ui.separator();
    let mut xray = ui_ctx.xray_mode;
    if ui.checkbox(&mut xray, t(loc, "vis.xray")).changed() {
        push(EditorCommand::SetXrayMode(xray));
    }
    let mut ghost = ui_ctx.ghost_unselected;
    if ui.checkbox(&mut ghost, t(loc, "vis.ghost")).changed() {
        push(EditorCommand::SetGhostUnselected(ghost));
    }
}
