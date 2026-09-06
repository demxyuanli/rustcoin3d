//! Blender-like parameter widgets (Principled BSDF style).

use egui::{pos2, vec2, Color32, CornerRadius, Rect, Stroke, Ui};

// Blender theme slider: dark track + blue progress fill.
pub(super) const SLIDER_H: f32 = 13.0;
pub(super) const SLIDER_BG: Color32 = Color32::from_rgb(0x54, 0x54, 0x54);
const SLIDER_BG_HOVER: Color32 = Color32::from_rgb(0x5E, 0x5E, 0x5E);
const SLIDER_FILL: Color32 = Color32::from_rgb(0x47, 0x72, 0xB3);
const SLIDER_R: u8 = 3;
const PILL_FILL_HOVER: Color32 = Color32::from_rgb(0x54, 0x54, 0x54);
const LABEL_TEXT: Color32 = Color32::from_rgb(0xD0, 0xD0, 0xD0);

/// Right-aligned muted label (used for plain socket-only rows: Image / A / B).
pub(super) fn param_label(ui: &mut Ui, text: &str) {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        ui.style_mut().visuals.widgets.inactive.fg_stroke = Stroke::new(0.0, LABEL_TEXT);
        ui.label(egui::RichText::new(text).size(10.0));
    });
}

/// Blender node slider: full-width dark track, blue progress fill, label left +
/// value right overlaid. Click sets absolute; drag adjusts.
pub(super) fn blender_slider(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
) -> egui::Response {
    let width = ui.available_width().max(64.0);
    let (rect, mut resp) =
        ui.allocate_exact_size(vec2(width, SLIDER_H), egui::Sense::click_and_drag());
    resp = resp.on_hover_cursor(egui::CursorIcon::ResizeHorizontal);

    let lo = *range.start();
    let hi = *range.end();
    let span = (hi - lo).abs().max(1e-6);

    if let Some(pos) = resp.interact_pointer_pos() {
        if resp.clicked() || resp.dragged() {
            let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            *value = lo + t * span;
            resp.mark_changed();
        }
    }

    let hovered = resp.hovered() || resp.dragged();
    let bg = if hovered { SLIDER_BG_HOVER } else { SLIDER_BG };
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(SLIDER_R), bg);

    let t = ((*value - lo) / span).clamp(0.0, 1.0);
    if t > 0.001 {
        let mut fill = rect;
        fill.set_width(rect.width() * t);
        // Keep left corners rounded, right edge flat when partially filled.
        let radius = if t > 0.98 {
            CornerRadius::same(SLIDER_R)
        } else {
            CornerRadius {
                nw: SLIDER_R,
                ne: 0,
                sw: SLIDER_R,
                se: 0,
            }
        };
        painter.rect_filled(fill, radius, SLIDER_FILL);
    }

    if resp.has_focus() {
        painter.rect_stroke(
            rect,
            CornerRadius::same(SLIDER_R),
            Stroke::new(1.0, SLIDER_FILL),
            egui::StrokeKind::Inside,
        );
    }

    let text_color = Color32::from_rgba_unmultiplied(255, 255, 255, 230);
    let font = egui::FontId::proportional(10.0);
    let pad = 5.0;
    painter.text(
        pos2(rect.left() + pad, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        font.clone(),
        text_color,
    );
    let value_text = format_slider_value(*value, span);
    painter.text(
        pos2(rect.right() - pad, rect.center().y),
        egui::Align2::RIGHT_CENTER,
        value_text,
        font,
        text_color,
    );

    resp
}

fn format_slider_value(v: f32, span: f32) -> String {
    if span <= 2.0 {
        format!("{v:.3}")
    } else if span <= 20.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.1}")
    }
}

/// Blender dropdown row: full-width dark track, label left, small triangle
/// at the right edge (GGX / Random Walk style). Click opens a popup menu.
pub(super) fn blender_dropdown(
    ui: &mut Ui,
    label: &str,
    popup_id: egui::Id,
    menu: impl FnOnce(&mut Ui),
) -> egui::Response {
    let width = ui.available_width().max(64.0);
    let (rect, resp) = ui.allocate_exact_size(vec2(width, SLIDER_H), egui::Sense::click());
    let open = egui::Popup::is_id_open(ui.ctx(), popup_id);

    let painter = ui.painter();
    let hovered = resp.hovered();
    let bg = if open || hovered {
        PILL_FILL_HOVER
    } else {
        SLIDER_BG
    };
    painter.rect_filled(rect, CornerRadius::same(SLIDER_R), bg);
    if open {
        painter.rect_stroke(
            rect,
            CornerRadius::same(SLIDER_R),
            Stroke::new(1.0, SLIDER_FILL),
            egui::StrokeKind::Inside,
        );
    }
    let text_color = Color32::from_rgba_unmultiplied(255, 255, 255, 230);
    let font = egui::FontId::proportional(10.0);
    painter.text(
        pos2(rect.left() + 5.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        font,
        text_color,
    );
    // Right-edge triangle, Blender dropdown affordance.
    let tri = rect.right_center() - vec2(6.0, 0.0);
    let points = [
        pos2(tri.x - 3.5, tri.y - 1.5),
        pos2(tri.x + 3.5, tri.y - 1.5),
        pos2(tri.x, tri.y + 2.5),
    ];
    painter.add(egui::Shape::convex_polygon(
        points.to_vec(),
        text_color,
        Stroke::NONE,
    ));

    // from_toggle_button_response toggles the popup on click and shows it
    // while its id is open; the `open` state read above only styles the row
    // (one-frame lag, imperceptible).
    egui::Popup::from_toggle_button_response(&resp)
        .id(popup_id)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClick)
        .show(menu);
    resp
}

/// Blender color-parameter row: dark track with the label inside, solid color
/// swatch at the right edge; clicking the swatch opens the color picker.
pub(super) fn blender_color_row_rgb(ui: &mut Ui, label: &str, color: &mut [f32; 4], popup_id: egui::Id) {
    let width = ui.available_width().max(64.0);
    let (rect, resp) = ui.allocate_exact_size(vec2(width, SLIDER_H), egui::Sense::click());
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(SLIDER_R), SLIDER_BG);
    // Swatch occupies the right quarter, inset like Blender's Base Color.
    let sw = rect.height() - 4.0;
    let swatch = Rect::from_min_size(
        pos2(rect.right() - sw - 2.0, rect.center().y - sw / 2.0),
        vec2(sw, sw),
    );
    let fill = Color32::from_rgba_unmultiplied(
        (color[0] * 255.0) as u8,
        (color[1] * 255.0) as u8,
        (color[2] * 255.0) as u8,
        (color[3] * 255.0) as u8,
    );
    painter.rect_filled(swatch, CornerRadius::same(2), fill);
    painter.rect_stroke(
        swatch,
        CornerRadius::same(2),
        Stroke::new(1.0, Color32::from_rgb(0x0A, 0x0A, 0x0A)),
        egui::StrokeKind::Inside,
    );
    painter.text(
        pos2(rect.left() + 5.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(10.0),
        Color32::from_rgba_unmultiplied(255, 255, 255, 230),
    );
    // from_toggle_button_response toggles the popup on click and shows it
    // while its id is open; the `open` state read above only styles the row.
    egui::Popup::from_toggle_button_response(&resp)
        .id(popup_id)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.spacing_mut().slider_width = 200.0;
            ui.color_edit_button_rgba_unmultiplied(color);
        });
}

/// Full-width Blender slider row (label + value live inside the track).
pub(super) fn param_row(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
) {
    blender_slider(ui, label, value, range);
}
