//! Photoshop-style left tool strip: last-used face icon, nub / right-click flyout.

use rc3d_actions::MarkupTool;
use rc3d_engine_api::camera::ViewPreset;
use rc3d_gizmo::GizmoMode;
use rc3d_scene::node_data::MeasurementType;

use crate::commands::EditorCommand;
use crate::ui::i18n::{t, UiLocale};
use crate::ui::icons::{
    gizmo_icon, icon_button, markup_icon, select_icon, view_preset_icon, Icon, STRIP_BTN,
};
use crate::ui::theme::ThemePalette;
use crate::ui::types::{
    CanvasTool, EditorChromeState, EditorDisplayMode, EditorUiContext, NodeDataType, SelectKind,
};

pub(super) fn draw(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    ui.spacing_mut().item_spacing = egui::vec2(0.0_f32, 2.0_f32);
    let loc = ui_ctx.ui_locale;
    let pal = ui_ctx.ui_theme.palette();
    let canvas = ui_ctx.canvas_tool;

    ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
        draw_grip(ui, &pal, loc);
        select_flyout(ui, ui_ctx, chrome, canvas, &pal, loc, push);
        transform_flyout(ui, ui_ctx, canvas, &pal, loc, push);
        measure_flyout(ui, ui_ctx, chrome, canvas, &pal, loc, push);
        if icon_button(
            ui,
            Icon::Section,
            t(loc, "tools.section"),
            canvas == CanvasTool::Section,
            STRIP_BTN,
            &pal,
        )
        .clicked()
        {
            push(EditorCommand::SetSectionEdit(canvas != CanvasTool::Section));
        }
        markup_flyout(ui, ui_ctx, chrome, canvas, &pal, loc, push);
        create_flyout(ui, chrome, &pal, loc, push);
        if icon_button(
            ui,
            Icon::Walk,
            t(loc, "tools.walk"),
            canvas == CanvasTool::Walk,
            STRIP_BTN,
            &pal,
        )
        .clicked()
        {
            push(EditorCommand::SetWalkMode(canvas != CanvasTool::Walk));
        }
        view_flyout(ui, chrome, &pal, loc, push);
        display_flyout(ui, ui_ctx, &pal, loc, push);
        strip_sep(ui, &pal);
        overlay_flyout(ui, ui_ctx, chrome, &pal, loc, push);
    });
}

fn draw_grip(ui: &mut egui::Ui, pal: &ThemePalette, loc: UiLocale) {
    // Hover-only so Area::movable can own the drag (Sense::drag on the grip steals it).
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(STRIP_BTN, 14.0_f32), egui::Sense::hover());
    let c = pal.text_secondary.gamma_multiply(0.75);
    let y = rect.center().y;
    for row in 0..2 {
        let yy = y + (row as f32 - 0.5) * 4.0;
        for i in 0..3 {
            let x = rect.center().x + (i as f32 - 1.0) * 5.0;
            ui.painter().circle_filled(egui::pos2(x, yy), 1.25, c);
        }
    }
    let _ = resp.on_hover_text(t(loc, "tools.strip_drag"));
}

fn select_flyout(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    canvas: CanvasTool,
    pal: &ThemePalette,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let last = chrome.tools.select;
    let tip = flyout_tip(loc, select_key(last));
    let (resp, activate, open) = flyout_button(
        ui,
        select_icon(last),
        &tip,
        canvas == CanvasTool::Select,
        pal,
    );
    if activate {
        push(EditorCommand::SetSelectKind(last));
    }
    show_flyout(&resp, open, |ui| {
        for kind in [SelectKind::Pick, SelectKind::Box, SelectKind::Lasso] {
            let on = ui_ctx.select_kind == kind && canvas == CanvasTool::Select;
            if ui.selectable_label(on, t(loc, select_key(kind))).clicked() {
                chrome.tools.select = kind;
                push(EditorCommand::SetSelectKind(kind));
            }
        }
    });
}

fn transform_flyout(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    canvas: CanvasTool,
    pal: &ThemePalette,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let mode = ui_ctx.gizmo_mode;
    let tip = flyout_tip(loc, gizmo_key(mode));
    let (resp, activate, open) = flyout_button(
        ui,
        gizmo_icon(mode),
        &tip,
        canvas == CanvasTool::Transform,
        pal,
    );
    if activate {
        push(EditorCommand::SetGizmoMode(mode));
    }
    show_flyout(&resp, open, |ui| {
        for m in [GizmoMode::Translate, GizmoMode::Rotate, GizmoMode::Scale] {
            let on = ui_ctx.gizmo_mode == m && canvas == CanvasTool::Transform;
            if ui.selectable_label(on, t(loc, gizmo_key(m))).clicked() {
                push(EditorCommand::SetGizmoMode(m));
            }
        }
    });
}

fn measure_flyout(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    canvas: CanvasTool,
    pal: &ThemePalette,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let last = chrome.tools.measure;
    let tip = flyout_tip(loc, measure_key(last));
    let (resp, activate, open) =
        flyout_button(ui, Icon::Measure, &tip, canvas == CanvasTool::Measure, pal);
    if activate {
        push(EditorCommand::SetMeasurementMode(Some(last)));
    }
    show_flyout(&resp, open, |ui| {
        for m in [
            MeasurementType::Distance,
            MeasurementType::Angle,
            MeasurementType::Radius,
            MeasurementType::Diameter,
        ] {
            let on = ui_ctx.measurement_type == Some(m) && canvas == CanvasTool::Measure;
            if ui.selectable_label(on, t(loc, measure_key(m))).clicked() {
                chrome.tools.measure = m;
                push(EditorCommand::SetMeasurementMode(Some(m)));
            }
        }
    });
}

fn markup_flyout(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    canvas: CanvasTool,
    pal: &ThemePalette,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let last = chrome.tools.markup;
    let tip = flyout_tip(loc, markup_key(last));
    let (resp, activate, open) = flyout_button(
        ui,
        markup_icon(last),
        &tip,
        canvas == CanvasTool::Markup,
        pal,
    );
    if activate {
        push(EditorCommand::SetMarkupTool(last));
    }
    show_flyout(&resp, open, |ui| {
        for tool in [
            MarkupTool::Line,
            MarkupTool::Rect,
            MarkupTool::Circle,
            MarkupTool::Freehand,
        ] {
            let on = ui_ctx.markup_tool == tool && canvas == CanvasTool::Markup;
            if ui.selectable_label(on, t(loc, markup_key(tool))).clicked() {
                chrome.tools.markup = tool;
                push(EditorCommand::SetMarkupTool(tool));
            }
        }
    });
}

fn create_flyout(
    ui: &mut egui::Ui,
    chrome: &mut EditorChromeState,
    pal: &ThemePalette,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let last = chrome.tools.create;
    let tip = flyout_tip(loc, create_key(last));
    let (resp, activate, open) = flyout_button(ui, Icon::Shape, &tip, false, pal);
    if activate {
        push(EditorCommand::CreateNode {
            node_type: last,
            parent: None,
        });
    }
    show_flyout(&resp, open, |ui| {
        for ty in [
            NodeDataType::Cube,
            NodeDataType::Sphere,
            NodeDataType::Cylinder,
            NodeDataType::Cone,
        ] {
            if ui
                .selectable_label(last == ty, t(loc, create_key(ty)))
                .clicked()
            {
                chrome.tools.create = ty;
                push(EditorCommand::CreateNode {
                    node_type: ty,
                    parent: None,
                });
            }
        }
    });
}

fn view_flyout(
    ui: &mut egui::Ui,
    chrome: &mut EditorChromeState,
    pal: &ThemePalette,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let last = chrome.tools.view;
    let tip = flyout_tip(loc, view_key(last));
    let (resp, activate, open) = flyout_button(ui, view_preset_icon(last), &tip, false, pal);
    if activate {
        push(EditorCommand::SetViewPreset(last));
    }
    show_flyout(&resp, open, |ui| {
        for preset in [
            ViewPreset::Top,
            ViewPreset::Front,
            ViewPreset::Right,
            ViewPreset::Iso,
            ViewPreset::Bottom,
            ViewPreset::Back,
            ViewPreset::Left,
        ] {
            if ui
                .selectable_label(last == preset, t(loc, view_key(preset)))
                .clicked()
            {
                chrome.tools.view = preset;
                push(EditorCommand::SetViewPreset(preset));
            }
        }
    });
}

fn display_flyout(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    pal: &ThemePalette,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let tip = t(loc, "tools.shading");
    let on = matches!(
        ui_ctx.display_mode_label.as_str(),
        "Wireframe" | "Shaded" | "ShadedWithEdges" | "HiddenLine" | "Flat" | "FlatWithEdge"
    );
    let (resp, activate, open) = flyout_button(ui, Icon::Shape, tip, on, pal);
    if activate {
        push(EditorCommand::CycleDisplayMode);
    }
    show_flyout(&resp, open, |ui| {
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
    });
}

fn overlay_flyout(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    _chrome: &mut EditorChromeState,
    pal: &ThemePalette,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    let (resp, _, open) = flyout_button(ui, Icon::Grid, t(loc, "view.overlays"), false, pal);
    show_flyout(&resp, open, |ui| {
        let mut grid = ui_ctx.grid_enabled;
        if ui.checkbox(&mut grid, t(loc, "view.grid")).changed() {
            push(EditorCommand::SetGridEnabled(grid));
        }
        let mut hud = ui_ctx.hud_enabled;
        if ui.checkbox(&mut hud, t(loc, "view.hud")).changed() {
            push(EditorCommand::SetHudEnabled(hud));
        }
        let mut xray = ui_ctx.xray_mode;
        if ui.checkbox(&mut xray, t(loc, "vis.xray")).changed() {
            push(EditorCommand::SetXrayMode(xray));
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
    });
}

fn flyout_button(
    ui: &mut egui::Ui,
    icon: Icon,
    tip: &str,
    selected: bool,
    pal: &ThemePalette,
) -> (egui::Response, bool, bool) {
    let resp = icon_button(ui, icon, tip, selected, STRIP_BTN, pal);
    paint_nub(ui, resp.rect, pal);
    let in_nub = resp
        .hover_pos()
        .is_some_and(|p| nub_rect(resp.rect).contains(p));
    let activate = resp.clicked() && !in_nub;
    let open = resp.secondary_clicked() || (resp.clicked() && in_nub);
    (resp, activate, open)
}

fn show_flyout(resp: &egui::Response, open: bool, popup: impl FnOnce(&mut egui::Ui)) {
    egui::Popup::from_response(resp)
        .kind(egui::PopupKind::Menu)
        .align(egui::RectAlign::RIGHT_START)
        .gap(4.0_f32)
        .layout(egui::Layout::top_down_justified(egui::Align::Min))
        .open_memory(open.then_some(egui::SetOpenCommand::Bool(true)))
        .show(popup);
}

fn nub_rect(btn: egui::Rect) -> egui::Rect {
    egui::Rect::from_min_max(
        btn.right_bottom() - egui::vec2(10.0_f32, 10.0_f32),
        btn.right_bottom(),
    )
}

fn paint_nub(ui: &mut egui::Ui, btn: egui::Rect, pal: &ThemePalette) {
    let p = btn.right_bottom() - egui::vec2(2.0_f32, 2.0_f32);
    let tri = vec![
        p - egui::vec2(6.0_f32, 0.0_f32),
        p,
        p - egui::vec2(0.0_f32, 6.0_f32),
    ];
    ui.painter().add(egui::Shape::convex_polygon(
        tri,
        pal.text_secondary.gamma_multiply(0.85),
        egui::Stroke::NONE,
    ));
}

fn strip_sep(ui: &mut egui::Ui, pal: &ThemePalette) {
    ui.add_space(4.0_f32);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(20.0_f32, 1.0_f32), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 0.0, pal.text_secondary.gamma_multiply(0.35));
    ui.add_space(4.0_f32);
}

fn flyout_tip(loc: UiLocale, key: &'static str) -> String {
    format!("{}\n{}", t(loc, key), t(loc, "tools.flyout_hint"))
}

fn select_key(kind: SelectKind) -> &'static str {
    match kind {
        SelectKind::Pick => "tools.pick",
        SelectKind::Box => "tools.box_select",
        SelectKind::Lasso => "tools.lasso",
    }
}

fn gizmo_key(mode: GizmoMode) -> &'static str {
    match mode {
        GizmoMode::Translate => "tools.move",
        GizmoMode::Rotate => "tools.rotate",
        GizmoMode::Scale => "tools.scale",
    }
}

fn measure_key(m: MeasurementType) -> &'static str {
    match m {
        MeasurementType::Distance => "tools.distance",
        MeasurementType::Angle => "tools.angle",
        MeasurementType::Radius => "tools.radius",
        MeasurementType::Diameter => "tools.diameter",
    }
}

fn markup_key(tool: MarkupTool) -> &'static str {
    match tool {
        MarkupTool::Select => "tools.select",
        MarkupTool::Line => "tools.line",
        MarkupTool::Rect => "tools.rect",
        MarkupTool::Circle => "tools.circle",
        MarkupTool::Freehand => "tools.freehand",
    }
}

fn create_key(ty: NodeDataType) -> &'static str {
    match ty {
        NodeDataType::Cube => "create.cube",
        NodeDataType::Sphere => "create.sphere",
        NodeDataType::Cylinder => "create.cylinder",
        NodeDataType::Cone => "create.cone",
        _ => "tools.create_geom",
    }
}

fn view_key(preset: ViewPreset) -> &'static str {
    match preset {
        ViewPreset::Top => "view.top",
        ViewPreset::Front => "view.front",
        ViewPreset::Right => "view.right",
        ViewPreset::Iso => "view.iso",
        ViewPreset::Bottom => "view.bottom",
        ViewPreset::Back => "view.back",
        ViewPreset::Left => "view.left",
    }
}
