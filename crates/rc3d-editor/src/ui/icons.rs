//! Fluent 2 toolbar and tree glyphs from Segoe Fluent Icons (system font).

use rc3d_engine_api::camera::ViewPreset;
use rc3d_gizmo::GizmoMode;

use crate::commands::EditorCommand;
use crate::ui::i18n::t;
use crate::ui::theme::{paint_icon_glyph, TOOL_GLYPH, TREE_GLYPH};
use crate::ui::types::EditorUiContext;

const TOOL_BTN: f32 = 24.0;
const TREE_BTN: f32 = 16.0;

#[derive(Clone, Copy)]
pub(super) enum Icon {
    Translate,
    Rotate,
    Scale,
    ViewTop,
    ViewFront,
    ViewRight,
    ViewIso,
    ViewBottom,
    ViewBack,
    ViewLeft,
    Grid,
    Hud,
    Add,
    Duplicate,
    Delete,
    ChevronRight,
    ChevronDown,
    Folder,
    Shape,
    Light,
    Camera,
}

impl Icon {
    fn codepoint(self) -> u32 {
        match self {
            Icon::Translate => 0xE7C2,
            Icon::Rotate => 0xE7AD,
            Icon::Scale => 0xE744,
            Icon::ViewTop => 0xE809,
            Icon::ViewBottom => 0xE80A,
            Icon::ViewFront => 0xE890,
            Icon::ViewBack => 0xE72B,
            Icon::ViewRight => 0xE90D,
            Icon::ViewLeft => 0xE90C,
            Icon::ViewIso => 0xE774,
            Icon::Grid => 0xF0E2,
            Icon::Hud => 0xF246,
            Icon::Add => 0xE710,
            Icon::Duplicate => 0xE8C8,
            Icon::Delete => 0xE74D,
            Icon::ChevronRight => 0xE76C,
            Icon::ChevronDown => 0xE70D,
            Icon::Folder => 0xE8B7,
            Icon::Shape => 0xE80F,
            Icon::Light => 0xE793,
            Icon::Camera => 0xE722,
        }
    }
}

pub(super) fn draw_toolbar(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    push: &mut impl FnMut(EditorCommand),
) {
    ui.spacing_mut().item_spacing = egui::vec2(2.0_f32, 0.0_f32);
    let loc = ui_ctx.ui_locale;
    ui.horizontal_centered(|ui| {
        let pal = ui_ctx.ui_theme.palette();
        let gm = ui_ctx.gizmo_mode;
        if icon_button(
            ui,
            Icon::Translate,
            t(loc, "tools.move"),
            gm == GizmoMode::Translate,
            TOOL_BTN,
            &pal,
        )
        .clicked()
        {
            push(EditorCommand::SetGizmoMode(GizmoMode::Translate));
        }
        if icon_button(
            ui,
            Icon::Rotate,
            t(loc, "tools.rotate"),
            gm == GizmoMode::Rotate,
            TOOL_BTN,
            &pal,
        )
        .clicked()
        {
            push(EditorCommand::SetGizmoMode(GizmoMode::Rotate));
        }
        if icon_button(
            ui,
            Icon::Scale,
            t(loc, "tools.scale"),
            gm == GizmoMode::Scale,
            TOOL_BTN,
            &pal,
        )
        .clicked()
        {
            push(EditorCommand::SetGizmoMode(GizmoMode::Scale));
        }

        group_sep(ui, &pal);

        for (preset, icon, key) in [
            (ViewPreset::Top, Icon::ViewTop, "view.top"),
            (ViewPreset::Front, Icon::ViewFront, "view.front"),
            (ViewPreset::Right, Icon::ViewRight, "view.right"),
            (ViewPreset::Iso, Icon::ViewIso, "view.iso"),
            (ViewPreset::Bottom, Icon::ViewBottom, "view.bottom"),
            (ViewPreset::Back, Icon::ViewBack, "view.back"),
            (ViewPreset::Left, Icon::ViewLeft, "view.left"),
        ] {
            if icon_button(ui, icon, t(loc, key), false, TOOL_BTN, &pal).clicked() {
                push(EditorCommand::SetViewPreset(preset));
            }
        }

        group_sep(ui, &pal);

        if icon_button(ui, Icon::Grid, t(loc, "view.grid"), ui_ctx.grid_enabled, TOOL_BTN, &pal)
            .clicked()
        {
            push(EditorCommand::SetGridEnabled(!ui_ctx.grid_enabled));
        }
        if icon_button(ui, Icon::Hud, t(loc, "view.hud"), ui_ctx.hud_enabled, TOOL_BTN, &pal)
            .clicked()
        {
            push(EditorCommand::SetHudEnabled(!ui_ctx.hud_enabled));
        }
    });
}

pub(super) fn paint_tree_icon(ui: &mut egui::Ui, icon: Icon, pal: &crate::ui::theme::ThemePalette) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(TREE_BTN, TREE_BTN), egui::Sense::hover());
    paint_icon_glyph(ui.painter(), rect, icon.codepoint(), TREE_GLYPH, pal.text_secondary);
}

pub(super) fn paint_tree_closer(ui: &mut egui::Ui, open: bool, pal: &crate::ui::theme::ThemePalette) {
    let icon = if open {
        Icon::ChevronDown
    } else {
        Icon::ChevronRight
    };
    paint_tree_icon(ui, icon, pal);
}

fn group_sep(ui: &mut egui::Ui, pal: &crate::ui::theme::ThemePalette) {
    ui.add_space(6.0_f32);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0_f32, 16.0_f32), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 0.0, pal.text_secondary.gamma_multiply(0.35));
    ui.add_space(6.0_f32);
}

pub(super) fn icon_button(
    ui: &mut egui::Ui,
    icon: Icon,
    tip: &str,
    selected: bool,
    size: f32,
    pal: &crate::ui::theme::ThemePalette,
) -> egui::Response {
    let (rect, mut resp) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click());
    let hovered = resp.hovered();
    let down = resp.is_pointer_button_down_on();
    let fill = if selected {
        pal.accent_fill
    } else if down {
        if pal.dark {
            egui::Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 0x1A)
        } else {
            egui::Color32::from_rgba_unmultiplied(0x00, 0x00, 0x00, 0x1A)
        }
    } else if hovered {
        pal.hover
    } else {
        egui::Color32::TRANSPARENT
    };
    ui.painter()
        .rect_filled(rect, egui::CornerRadius::same(4), fill);
    if selected {
        ui.painter().rect_stroke(
            rect.shrink(0.5_f32),
            egui::CornerRadius::same(4),
            egui::Stroke::new(1.0_f32, pal.accent),
            egui::StrokeKind::Inside,
        );
    }
    let color = if selected { pal.text } else { pal.text_secondary };
    let glyph = if size <= TREE_BTN + 0.5 { TREE_GLYPH } else { TOOL_GLYPH };
    paint_icon_glyph(ui.painter(), rect, icon.codepoint(), glyph, color);
    if !tip.is_empty() {
        resp = resp.on_hover_text(tip);
    }
    resp
}
