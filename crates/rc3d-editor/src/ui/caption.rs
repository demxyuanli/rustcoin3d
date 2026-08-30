use crate::ui::theme::{paint_icon_glyph, CAPTION_GLYPH, ThemePalette};
use crate::ui::types::{CaptionAction, EditorChromeState, EditorUiContext};

const CAPTION_H: f32 = 32.0;
const BTN_W: f32 = 46.0;

pub(super) fn draw_caption(ctx: &egui::Context, chrome: &mut EditorChromeState, ui_ctx: &EditorUiContext) {
    if !chrome.caption.enabled {
        return;
    }
    let pal = ui_ctx.ui_theme.palette();
    let maximized = chrome.caption.maximized;
    let title = chrome.caption.title.clone();
    let mut action = None;

    egui::TopBottomPanel::top("rc3d_caption")
        .exact_height(CAPTION_H)
        .frame(
            egui::Frame::NONE
                .fill(pal.layer)
                .inner_margin(egui::Margin::ZERO),
        )
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
            ui.set_min_height(CAPTION_H);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if caption_button(ui, CaptionIcon::Close, &pal).clicked() {
                    if chrome.caption.dirty {
                        chrome.caption.close_prompt = true;
                    } else {
                        action = Some(CaptionAction::Close);
                    }
                }
                let max_icon = if maximized {
                    CaptionIcon::Restore
                } else {
                    CaptionIcon::Maximize
                };
                if caption_button(ui, max_icon, &pal).clicked() {
                    action = Some(CaptionAction::ToggleMaximize);
                }
                if caption_button(ui, CaptionIcon::Minimize, &pal).clicked() {
                    action = Some(CaptionAction::Minimize);
                }

                let (rect, resp) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
                let accent = egui::Rect::from_min_size(
                    egui::pos2(rect.left() + 10.0, rect.center().y - 6.0),
                    egui::vec2(4.0, 12.0),
                );
                ui.painter()
                    .rect_filled(accent, egui::CornerRadius::same(1), pal.accent);
                ui.painter().text(
                    egui::pos2(rect.left() + 20.0, rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    &title,
                    egui::FontId::proportional(13.0),
                    pal.text,
                );
                if resp.double_clicked() {
                    action = Some(CaptionAction::ToggleMaximize);
                } else if resp.secondary_clicked() {
                    action = Some(CaptionAction::ShowSystemMenu);
                } else if resp.drag_started() {
                    action = Some(CaptionAction::Drag);
                }
            });
        });

    if action.is_some() {
        chrome.caption.action = action;
    }
}

#[derive(Clone, Copy)]
enum CaptionIcon {
    Minimize,
    Maximize,
    Restore,
    Close,
}

fn caption_button(ui: &mut egui::Ui, icon: CaptionIcon, pal: &ThemePalette) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(BTN_W, CAPTION_H), egui::Sense::click());
    let close = matches!(icon, CaptionIcon::Close);
    let fill = if resp.is_pointer_button_down_on() {
        if close {
            darken(pal.close_hover)
        } else {
            overlay(pal.dark, 0x1A)
        }
    } else if resp.hovered() {
        if close {
            pal.close_hover
        } else {
            pal.hover
        }
    } else {
        egui::Color32::TRANSPARENT
    };
    ui.painter()
        .rect_filled(rect, egui::CornerRadius::ZERO, fill);
    let icon_color = if close && resp.hovered() {
        egui::Color32::WHITE
    } else {
        pal.text_secondary
    };
    paint_icon_glyph(ui.painter(), rect, icon.codepoint(), CAPTION_GLYPH, icon_color);
    resp
}

impl CaptionIcon {
    fn codepoint(self) -> u32 {
        match self {
            CaptionIcon::Minimize => 0xE921,
            CaptionIcon::Maximize => 0xE922,
            CaptionIcon::Restore => 0xE923,
            CaptionIcon::Close => 0xE8BB,
        }
    }
}

fn darken(c: egui::Color32) -> egui::Color32 {
    egui::Color32::from_rgb(
        c.r().saturating_sub(20),
        c.g().saturating_sub(20),
        c.b().saturating_sub(20),
    )
}

fn overlay(dark: bool, alpha: u8) -> egui::Color32 {
    if dark {
        egui::Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, alpha)
    } else {
        egui::Color32::from_rgba_unmultiplied(0x00, 0x00, 0x00, alpha)
    }
}
