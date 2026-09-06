//! Fluent 2 toolbar and tree glyphs from Segoe Fluent Icons (system font).

use rc3d_actions::MarkupTool;
use rc3d_engine_api::camera::ViewPreset;
use rc3d_gizmo::GizmoMode;

use crate::ui::theme::{paint_icon_glyph, TOOL_GLYPH, TREE_GLYPH};
use crate::ui::types::{BottomTab, PropsTab, SelectKind, SideTab};

pub(super) const STRIP_BTN: f32 = 28.0;
pub(super) const PANEL_BTN: f32 = 22.0;
const TREE_BTN: f32 = 16.0;

// Chevron codepoints shared with custom-painted headers (compositor nodes).
pub(super) const CODEPOINT_CHEVRON_DOWN: u32 = 0xE70D;
pub(super) const CODEPOINT_CHEVRON_RIGHT: u32 = 0xE76C;

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
    Add,
    Duplicate,
    Delete,
    ChevronRight,
    ChevronDown,
    Folder,
    Shape,
    Light,
    Camera,
    Cursor,
    BoxSelect,
    Lasso,
    Measure,
    Section,
    Markup,
    MarkupLine,
    MarkupRect,
    MarkupCircle,
    MarkupFreehand,
    Walk,
    Eye,
    EyeOff,
    Lock,
    ZoomIn,
    ZoomOut,
    FitView,
    CenterView,
    ColorPalette,
    Close,
    Open,
    Hierarchy,
    RenderPanel,
    History,
    Document,
    Compositor,
    Clear,
    Preset,
    Undo,
    Redo,
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
            Icon::Add => 0xE710,
            Icon::Duplicate => 0xE8C8,
            Icon::Delete => 0xE74D,
            Icon::ChevronRight => 0xE76C,
            Icon::ChevronDown => 0xE70D,
            Icon::Folder => 0xE8B7,
            Icon::Shape => 0xE80F,
            Icon::Light => 0xE793,
            Icon::Camera => 0xE722,
            Icon::Cursor => 0xE8B0,
            Icon::BoxSelect => 0xE739,
            Icon::Lasso => 0xED5A,
            Icon::Measure => 0xE8EF,
            Icon::Section => 0xE7A8,
            Icon::Markup => 0xE70F,
            Icon::MarkupLine => 0xE76B,
            Icon::MarkupRect => 0xE739,
            Icon::MarkupCircle => 0xEA3A,
            Icon::MarkupFreehand => 0xE76D,
            Icon::Walk => 0xE805,
            Icon::Eye => 0xE7B3,
            Icon::EyeOff => 0xED1A,
            Icon::Lock => 0xE72E,
            Icon::ZoomIn => 0xE8A3,
            Icon::ZoomOut => 0xE71F,
            Icon::FitView => 0xE9A6,
            Icon::CenterView => 0xE7B5,
            Icon::ColorPalette => 0xE790,
            Icon::Close => 0xE711,
            Icon::Open => 0xE8E5,
            Icon::Hierarchy => 0xE8FD,
            Icon::RenderPanel => 0xE91B,
            Icon::History => 0xE81C,
            Icon::Document => 0xE8A5,
            Icon::Compositor => 0xE9F9,
            Icon::Clear => 0xE894,
            Icon::Preset => 0xE728,
            Icon::Undo => 0xE7A7,
            Icon::Redo => 0xE7A6,
        }
    }
}

pub(super) fn side_tab_icon(tab: SideTab) -> Icon {
    match tab {
        SideTab::Hierarchy => Icon::Hierarchy,
        SideTab::Render => Icon::RenderPanel,
        SideTab::History => Icon::History,
        SideTab::Assets => Icon::Folder,
    }
}

pub(super) fn bottom_tab_icon(tab: BottomTab) -> Icon {
    match tab {
        BottomTab::Document => Icon::Document,
        BottomTab::Compositor => Icon::Compositor,
    }
}

pub(super) fn props_tab_icon(tab: PropsTab) -> Icon {
    match tab {
        PropsTab::Object => Icon::Shape,
        PropsTab::Material => Icon::ColorPalette,
        PropsTab::Display => Icon::Eye,
        PropsTab::Camera => Icon::Camera,
    }
}

pub(super) fn view_preset_icon(preset: ViewPreset) -> Icon {
    match preset {
        ViewPreset::Top => Icon::ViewTop,
        ViewPreset::Bottom => Icon::ViewBottom,
        ViewPreset::Front => Icon::ViewFront,
        ViewPreset::Back => Icon::ViewBack,
        ViewPreset::Right => Icon::ViewRight,
        ViewPreset::Left => Icon::ViewLeft,
        ViewPreset::Iso => Icon::ViewIso,
    }
}

pub(super) fn gizmo_icon(mode: GizmoMode) -> Icon {
    match mode {
        GizmoMode::Translate => Icon::Translate,
        GizmoMode::Rotate => Icon::Rotate,
        GizmoMode::Scale => Icon::Scale,
    }
}

pub(super) fn select_icon(kind: SelectKind) -> Icon {
    match kind {
        SelectKind::Pick => Icon::Cursor,
        SelectKind::Box => Icon::BoxSelect,
        SelectKind::Lasso => Icon::Lasso,
    }
}

pub(super) fn markup_icon(tool: MarkupTool) -> Icon {
    match tool {
        MarkupTool::Select => Icon::Markup,
        MarkupTool::Line => Icon::MarkupLine,
        MarkupTool::Rect => Icon::MarkupRect,
        MarkupTool::Circle => Icon::MarkupCircle,
        MarkupTool::Freehand => Icon::MarkupFreehand,
    }
}

pub(super) fn paint_tree_icon(ui: &mut egui::Ui, icon: Icon, pal: &crate::ui::theme::ThemePalette) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(TREE_BTN, TREE_BTN), egui::Sense::hover());
    paint_icon_glyph(
        ui.painter(),
        rect,
        icon.codepoint(),
        TREE_GLYPH,
        pal.text_secondary,
    );
}

pub(super) fn paint_tree_closer(
    ui: &mut egui::Ui,
    open: bool,
    pal: &crate::ui::theme::ThemePalette,
) {
    let icon = if open {
        Icon::ChevronDown
    } else {
        Icon::ChevronRight
    };
    paint_tree_icon(ui, icon, pal);
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
    let color = if selected {
        pal.text
    } else {
        pal.text_secondary
    };
    let glyph = if size <= TREE_BTN + 0.5 {
        TREE_GLYPH
    } else {
        TOOL_GLYPH
    };
    paint_icon_glyph(ui.painter(), rect, icon.codepoint(), glyph, color);
    if !tip.is_empty() {
        resp = resp.on_hover_text(tip);
    }
    resp
}

/// Tab button: icon + text label, styled like a selectable tab.
pub(super) fn tab_button(
    ui: &mut egui::Ui,
    icon: Icon,
    label: &str,
    selected: bool,
    pal: &crate::ui::theme::ThemePalette,
) -> egui::Response {
    let icon_sz = TREE_GLYPH + 1.0_f32;
    let padding = egui::vec2(8.0_f32, 4.0_f32);
    let galley = ui.painter().layout_no_wrap(
        label.to_owned(),
        egui::FontId::new(
            ui.style().text_styles[&egui::TextStyle::Button].size,
            ui.style().text_styles[&egui::TextStyle::Button]
                .family
                .clone(),
        ),
        egui::Color32::WHITE,
    );
    let btn_w = padding.x + icon_sz + 4.0_f32 + galley.size().x + padding.x;
    let btn_h = (icon_sz + padding.y * 2.0_f32)
        .max(galley.size().y + padding.y * 2.0_f32)
        .max(22.0_f32);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(btn_w, btn_h), egui::Sense::click());
    let bg = if selected {
        pal.accent_fill
    } else if resp.hovered() {
        pal.hover
    } else {
        egui::Color32::TRANSPARENT
    };
    ui.painter()
        .rect_filled(rect, egui::CornerRadius::same(4), bg);
    if selected {
        ui.painter().rect_stroke(
            rect.shrink(0.5_f32),
            egui::CornerRadius::same(4),
            egui::Stroke::new(1.0_f32, pal.accent),
            egui::StrokeKind::Inside,
        );
    }
    let icon_x = rect.left() + padding.x + icon_sz * 0.5;
    let icon_rect = egui::Rect::from_center_size(
        egui::pos2(icon_x, rect.center().y),
        egui::vec2(icon_sz, icon_sz),
    );
    let text_color = if selected {
        pal.text
    } else {
        pal.text_secondary
    };
    paint_icon_glyph(
        ui.painter(),
        icon_rect,
        icon.codepoint(),
        icon_sz,
        text_color,
    );
    ui.painter().text(
        egui::pos2(icon_x + icon_sz * 0.5 + 4.0_f32, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::TextStyle::Button.resolve(ui.style()),
        text_color,
    );
    resp
}
pub(super) fn icon_menu_button(
    ui: &mut egui::Ui,
    icon: Icon,
    label: &str,
    pal: &crate::ui::theme::ThemePalette,
) -> egui::Response {
    let row_h = ui.spacing().interact_size.y.max(22.0_f32);
    let width = ui.available_width().max(148.0_f32);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, row_h), egui::Sense::click());
    let visuals = ui.style().interact(&resp);
    if resp.hovered() || resp.highlighted() || resp.has_focus() {
        ui.painter()
            .rect_filled(rect, visuals.corner_radius, visuals.bg_fill);
    }
    let icon_sz = 16.0_f32;
    let icon_rect = egui::Rect::from_center_size(
        egui::pos2(rect.left() + 8.0_f32 + icon_sz * 0.5, rect.center().y),
        egui::vec2(icon_sz, icon_sz),
    );
    paint_icon_glyph(
        ui.painter(),
        icon_rect,
        icon.codepoint(),
        TREE_GLYPH + 2.0_f32,
        pal.text_secondary,
    );
    let text_color = if resp.hovered() {
        pal.text
    } else {
        visuals.text_color()
    };
    ui.painter().text(
        egui::pos2(rect.left() + 8.0_f32 + icon_sz + 8.0_f32, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::TextStyle::Body.resolve(ui.style()),
        text_color,
    );
    resp
}
