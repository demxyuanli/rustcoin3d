//! Bottom compositor: Blender-style node canvas via egui-snarl.

mod labels;
mod sync;
mod view;
mod viewer;
mod widgets;

use egui::{pos2, vec2, Color32, Id, Key, Ui};
use egui_snarl::{
    ui::{get_selected_nodes, SnarlWidget},
    Snarl,
};

use rc3d_render::{CompMenuGroup, CompNode, CompOp, CompositorGraph};

use crate::ui::i18n::{t, UiLocale};
use crate::ui::icons::{self, Icon, PANEL_BTN};
use crate::ui::theme::ThemePalette;
use sync::{export_graph, protected, snarl_from_graph};
use view::{ViewCmd, ViewState};
use viewer::CompViewer;

const SNARL_ID: &str = "rc3d-compositor-snarl";
/// Target width (graph units) for the center parameter column of a node. The
/// socket strips on the left / right edges add a bit on each side, so total
/// node width ends up slightly wider.
pub(super) const NODE_W: f32 = 180.0;
const DOT_STEP: f32 = 24.0;
const NODE_FILL: Color32 = Color32::from_rgb(0x2A, 0x2A, 0x2A);
const NODE_R: u8 = 6;

const ADD_MENU: [(CompMenuGroup, &str); 6] = [
    (CompMenuGroup::Input, "comp.group.input"),
    (CompMenuGroup::Color, "comp.group.color"),
    (CompMenuGroup::Filter, "comp.group.filter"),
    (CompMenuGroup::Transform, "comp.group.transform"),
    (CompMenuGroup::Converter, "comp.group.converter"),
    (CompMenuGroup::CadPass, "comp.group.cad"),
];

/// Full op table: ADD_OPS + CAD-only extras (Ssr/Fog + CAD2 group).
const ADD_OPS: [(CompOp, &str); 29] = [
    // Input / Color
    (CompOp::RenderLayers, "comp.layers"),
    (CompOp::Rgb, "comp.rgb"),
    (CompOp::Value, "comp.value"),
    (CompOp::Mix, "comp.mix"),
    (CompOp::AlphaOver, "comp.alpha_over"),
    (CompOp::BrightContrast, "comp.bc"),
    (CompOp::Exposure, "comp.exposure"),
    (CompOp::Gamma, "comp.gamma"),
    (CompOp::HueSat, "comp.hue_sat"),
    (CompOp::Invert, "comp.invert"),
    (CompOp::ColorRamp, "comp.ramp"),
    // Filter
    (CompOp::Blur, "comp.blur"),
    (CompOp::DilateErode, "comp.dilate"),
    // Transform
    (CompOp::Translate, "comp.translate"),
    (CompOp::Rotate, "comp.rotate"),
    (CompOp::Scale, "comp.scale"),
    (CompOp::Crop, "comp.crop"),
    // Converter
    (CompOp::Math, "comp.math"),
    // CadPass (render features)
    (CompOp::Ssao, "comp.ssao"),
    (CompOp::Fxaa, "comp.fxaa"),
    (CompOp::Taa, "comp.taa"),
    (CompOp::ColorGrade, "comp.grade"),
    (CompOp::Bloom, "comp.bloom"),
    (CompOp::Dof, "comp.dof"),
    (CompOp::Ssr, "comp.ssr"),
    (CompOp::Fog, "comp.fog"),
    (CompOp::Edges, "comp.edges"),
    (CompOp::HiddenLine, "comp.hlr"),
    (CompOp::Xray, "comp.xray"),
];

const ADD_OPS_CAD2: [(CompOp, &str); 2] = [
    (CompOp::Grid, "comp.grid"),
    (CompOp::Shadows, "comp.shadows"),
];

const PRESETS: [(&str, fn() -> CompositorGraph); 7] = [
    ("comp.preset.identity", CompositorGraph::identity),
    ("comp.preset.industrial", CompositorGraph::preset_industrial),
    ("comp.preset.product", CompositorGraph::preset_product),
    ("comp.preset.hlr", CompositorGraph::preset_hidden_line),
    ("comp.preset.xray_edges", CompositorGraph::preset_xray_edges),
    ("comp.preset.ssao_taa", CompositorGraph::preset_ssao_taa),
    ("comp.preset.edges", CompositorGraph::preset_edges_only),
];

pub struct CompositorEditor {
    pub graph: CompositorGraph,
    snarl: Snarl<CompNode>,
    bar_status: String,
    view: ViewState,
}

impl Default for CompositorEditor {
    fn default() -> Self {
        let graph = CompositorGraph::identity();
        let snarl = snarl_from_graph(&graph);
        let bar_status = look_status(&graph);
        Self {
            graph,
            snarl,
            bar_status,
            view: ViewState::default(),
        }
    }
}

impl CompositorEditor {
    fn load_graph(&mut self, graph: CompositorGraph) {
        self.graph = graph;
        self.snarl = snarl_from_graph(&self.graph);
        self.bar_status = look_status(&self.graph);
        self.view.pending = Some(ViewCmd::Fit);
    }
}

pub fn draw_panel(ui: &mut Ui, state: &mut CompositorEditor, loc: UiLocale) {
    let pal = if ui.visuals().dark_mode {
        ThemePalette::dark()
    } else {
        ThemePalette::light()
    };
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0_f32;
        if state.graph.has_cycle {
            ui.colored_label(Color32::from_rgb(220, 80, 80), t(loc, "comp.cycle"));
        }
        let add = icons::icon_button(ui, Icon::Add, t(loc, "comp.add"), false, PANEL_BTN, &pal);
        egui::Popup::menu(&add)
            .id(ui.make_persistent_id("rc3d_comp_add"))
            .close_behavior(egui::PopupCloseBehavior::CloseOnClick)
            .show(|ui| {
                add_menu(ui, state, loc);
            });
        let preset = icons::icon_button(
            ui,
            Icon::Preset,
            t(loc, "comp.preset"),
            false,
            PANEL_BTN,
            &pal,
        );
        egui::Popup::menu(&preset)
            .id(ui.make_persistent_id("rc3d_comp_preset"))
            .close_behavior(egui::PopupCloseBehavior::CloseOnClick)
            .show(|ui| {
                for (key, build) in PRESETS {
                    if ui.button(t(loc, key)).clicked() {
                        state.load_graph(build());
                        ui.close();
                    }
                }
            });
        if icons::icon_button(
            ui,
            Icon::Delete,
            t(loc, "comp.delete"),
            false,
            PANEL_BTN,
            &pal,
        )
        .clicked()
        {
            remove_selected(&mut state.snarl, ui);
        }
        ui.separator();
        draw_view_toolbar(ui, state, loc, &pal);
        ui.separator();
        ui.weak(&state.bar_status);
        ui.weak(format!("{:.0}%", state.view.last_scale * 100.0));
        ui.weak(t(loc, state.view.color_scheme.label_key()));
    });
    ui.separator();

    let avail = ui.available_rect_before_wrap();
    state.view.begin_frame(avail);
    let canvas_size = vec2(avail.width().max(64.0), avail.height().max(120.0));

    let scheme = state.view.color_scheme;
    let mut viewer = CompViewer {
        loc,
        view: &mut state.view,
        color_scheme: scheme,
    };
    SnarlWidget::new()
        .id(Id::new(SNARL_ID))
        .style(viewer::compositor_style())
        .min_size(canvas_size)
        .max_size(canvas_size)
        .show(&mut state.snarl, &mut viewer, ui);

    if ui.input(|i| i.key_pressed(Key::Delete)) {
        remove_selected(&mut state.snarl, ui);
    }

    let (nodes, edges) = export_graph(&state.snarl);
    state.graph.replace(nodes, edges);
    let _ = state.graph.compile();
    state.bar_status = look_status(&state.graph);
}

fn draw_view_toolbar(ui: &mut Ui, state: &mut CompositorEditor, loc: UiLocale, pal: &ThemePalette) {
    if icons::icon_button(
        ui,
        Icon::ZoomIn,
        t(loc, "comp.view.zoom_in"),
        false,
        PANEL_BTN,
        pal,
    )
    .clicked()
    {
        state.view.pending = Some(ViewCmd::ZoomIn);
    }
    if icons::icon_button(
        ui,
        Icon::ZoomOut,
        t(loc, "comp.view.zoom_out"),
        false,
        PANEL_BTN,
        pal,
    )
    .clicked()
    {
        state.view.pending = Some(ViewCmd::ZoomOut);
    }
    if icons::icon_button(
        ui,
        Icon::FitView,
        t(loc, "comp.view.fit"),
        false,
        PANEL_BTN,
        pal,
    )
    .clicked()
    {
        state.view.pending = Some(ViewCmd::Fit);
    }
    if icons::icon_button(
        ui,
        Icon::CenterView,
        t(loc, "comp.view.center"),
        false,
        PANEL_BTN,
        pal,
    )
    .clicked()
    {
        state.view.pending = Some(ViewCmd::Center);
    }
    if icons::icon_button(
        ui,
        Icon::ColorPalette,
        t(loc, "comp.view.color"),
        false,
        PANEL_BTN,
        pal,
    )
    .clicked()
    {
        state.view.color_scheme = state.view.color_scheme.next();
    }
}

fn look_status(graph: &CompositorGraph) -> String {
    let look = graph.cad_look();
    if !look.active {
        return "CAD: identity / tier".to_owned();
    }
    let mut flags: Vec<&str> = Vec::new();
    if look.ssao {
        flags.push("SSAO");
    }
    if look.taa {
        flags.push("TAA");
    }
    if look.fxaa && !look.taa {
        flags.push("FXAA");
    }
    if look.color_grading {
        flags.push("CG");
    }
    if look.bloom {
        flags.push("Bloom");
    }
    if look.dof {
        flags.push("DOF");
    }
    if look.ssr {
        flags.push("SSR");
    }
    if look.fog {
        flags.push("Fog");
    }
    if look.hidden_line {
        flags.push("HLR");
    } else if look.edges {
        flags.push("Edges");
    }
    if look.xray {
        flags.push("XRay");
    }
    if look.grid {
        flags.push("Grid");
    }
    if look.shadows {
        flags.push("Shadow");
    }
    if flags.is_empty() {
        "CAD: comp".to_owned()
    } else {
        format!("CAD: comp {}", flags.join(" "))
    }
}

fn add_menu(ui: &mut Ui, state: &mut CompositorEditor, loc: UiLocale) {
    // Insert new nodes near the visible canvas centre (graph space) so they
    // never land off-screen or pile onto existing nodes. Minor stagger avoids
    // overlap when several ops are added in a row.
    pick_add_op(ui, loc, |op| {
        let (c, scale) = (state.view.panel_rect.center(), state.view.last_scale.max(0.1));
        let base = pos2(c.x - state.view.pan.x, c.y - state.view.pan.y) / scale;
        let count = state.snarl.nodes().count() as f32;
        let pos = pos2(base.x + (count % 5.0) * 36.0, base.y + (count / 5.0).floor() * 26.0);
        state
            .snarl
            .insert_node(pos, CompNode::new(0, op, [pos.x, pos.y]));
    });
}

pub(super) fn pick_add_op(ui: &mut Ui, loc: UiLocale, mut on_pick: impl FnMut(CompOp)) {
    // Two-level menu: hover a group to open its sub-menu of ops.
    // `menu_button` inside a popup/menu renders as a sub-menu button.
    for (group, key) in ADD_MENU {
        ui.menu_button(t(loc, key), |ui| {
            for (op, op_key) in ADD_OPS.iter().chain(ADD_OPS_CAD2.iter()) {
                if op.menu_group() != group {
                    continue;
                }
                if ui.button(t(loc, op_key)).clicked() {
                    on_pick(*op);
                    ui.close();
                }
            }
        });
    }
}

fn remove_selected(snarl: &mut Snarl<CompNode>, ui: &Ui) {
    let ids: Vec<egui_snarl::NodeId> = get_selected_nodes(Id::new(SNARL_ID), ui.ctx());
    for id in ids {
        if snarl.get_node(id).is_some_and(|n| protected(n.op)) {
            continue;
        }
        if snarl.get_node(id).is_some() {
            let _ = snarl.remove_node(id);
        }
    }
}
