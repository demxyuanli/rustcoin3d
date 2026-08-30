//! Fluent 2 visuals (dark / light) and font install for the studio host.

use std::sync::Arc;

use egui::{Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Shadow, Stroke, Visuals};

pub const ICON_FONT: &str = "studio_icons";
pub const TOOL_GLYPH: f32 = 16.0;
pub const TREE_GLYPH: f32 = 12.0;
pub const CAPTION_GLYPH: f32 = 10.0;

/// Kept as dark-theme aliases for call sites that have no theme handle yet.
pub const LAYER: Color32 = Color32::from_rgb(0x20, 0x20, 0x20);
pub const CARD: Color32 = Color32::from_rgb(0x2C, 0x2C, 0x2C);
pub const SOLID: Color32 = Color32::from_rgb(0x1C, 0x1C, 0x1C);
pub const TEXT: Color32 = Color32::from_rgb(0xFF, 0xFF, 0xFF);
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(0xC5, 0xC5, 0xC5);
pub const ACCENT: Color32 = Color32::from_rgb(0x60, 0xCD, 0xFF);
pub const ACCENT_FILL: Color32 = Color32::from_rgb(0x00, 0x78, 0xD4);
pub const STROKE: Color32 = Color32::from_rgba_premultiplied(15, 15, 15, 15);
pub const CLOSE_HOVER: Color32 = Color32::from_rgb(0xC4, 0x2B, 0x1C);
pub const CAPTION_FILL: Color32 = LAYER;
pub const HOVER_FILL: Color32 = Color32::from_rgba_premultiplied(18, 18, 18, 18);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum UiTheme {
    #[default]
    Dark,
    Light,
}

impl UiTheme {
    pub fn as_id(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
        }
    }

    pub fn from_id(id: &str) -> Self {
        match id {
            "light" => Self::Light,
            _ => Self::Dark,
        }
    }

    pub fn palette(self) -> ThemePalette {
        match self {
            Self::Dark => ThemePalette::dark(),
            Self::Light => ThemePalette::light(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ThemePalette {
    pub layer: Color32,
    pub card: Color32,
    pub solid: Color32,
    pub text: Color32,
    pub text_secondary: Color32,
    pub accent: Color32,
    pub accent_fill: Color32,
    pub stroke: Color32,
    pub close_hover: Color32,
    pub hover: Color32,
    pub dark: bool,
}

impl ThemePalette {
    pub fn dark() -> Self {
        Self {
            layer: LAYER,
            card: CARD,
            solid: SOLID,
            text: TEXT,
            text_secondary: TEXT_SECONDARY,
            accent: ACCENT,
            accent_fill: ACCENT_FILL,
            stroke: STROKE,
            close_hover: CLOSE_HOVER,
            hover: HOVER_FILL,
            dark: true,
        }
    }

    pub fn light() -> Self {
        Self {
            layer: Color32::from_rgb(0xF3, 0xF3, 0xF3),
            card: Color32::from_rgb(0xFF, 0xFF, 0xFF),
            solid: Color32::from_rgb(0xF9, 0xF9, 0xF9),
            text: Color32::from_rgb(0x1A, 0x1A, 0x1A),
            text_secondary: Color32::from_rgb(0x5D, 0x5D, 0x5D),
            accent: Color32::from_rgb(0x00, 0x5F, 0xB8),
            accent_fill: Color32::from_rgb(0x00, 0x78, 0xD4),
            stroke: Color32::from_rgba_unmultiplied(0, 0, 0, 0x14),
            close_hover: CLOSE_HOVER,
            hover: Color32::from_rgba_unmultiplied(0, 0, 0, 0x0F),
            dark: false,
        }
    }
}

pub fn icon_family() -> FontFamily {
    FontFamily::Name(ICON_FONT.into())
}

pub fn paint_icon_glyph(
    painter: &egui::Painter,
    rect: egui::Rect,
    codepoint: u32,
    size: f32,
    color: Color32,
) {
    let Some(ch) = char::from_u32(codepoint) else {
        return;
    };
    let galley = painter.layout_no_wrap(ch.to_string(), FontId::new(size, icon_family()), color);
    let pos = egui::pos2(
        rect.center().x - galley.size().x * 0.5,
        rect.center().y - galley.size().y * 0.5,
    );
    painter.galley(pos, galley, color);
}

fn stroke(color: Color32) -> Stroke {
    Stroke::new(1.0_f32, color)
}

fn overlay(dark: bool, alpha: u8) -> Color32 {
    if dark {
        Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, alpha)
    } else {
        Color32::from_rgba_unmultiplied(0x00, 0x00, 0x00, alpha)
    }
}

pub fn apply_fluent_dark(ctx: &egui::Context) {
    apply_theme(ctx, UiTheme::Dark, true);
}

pub fn apply_theme(ctx: &egui::Context, theme: UiTheme, load_fonts: bool) {
    if load_fonts {
        install_fonts(ctx);
    }
    let pal = theme.palette();
    let mut visuals = if pal.dark {
        Visuals::dark()
    } else {
        Visuals::light()
    };
    visuals.dark_mode = pal.dark;
    visuals.override_text_color = Some(pal.text);
    visuals.window_fill = pal.layer;
    visuals.panel_fill = pal.layer;
    visuals.faint_bg_color = pal.card;
    visuals.extreme_bg_color = pal.solid;
    visuals.code_bg_color = pal.solid;
    visuals.hyperlink_color = pal.accent;
    visuals.selection.bg_fill = pal.accent_fill;
    visuals.selection.stroke = stroke(pal.accent);
    visuals.window_corner_radius = CornerRadius::same(8);
    visuals.menu_corner_radius = CornerRadius::same(8);
    visuals.window_shadow = Shadow::NONE;
    visuals.popup_shadow = Shadow::NONE;
    visuals.window_stroke = stroke(pal.stroke);
    visuals.widgets.noninteractive.bg_fill = pal.card;
    visuals.widgets.noninteractive.weak_bg_fill = Color32::TRANSPARENT;
    visuals.widgets.noninteractive.bg_stroke = stroke(pal.stroke);
    visuals.widgets.noninteractive.fg_stroke = stroke(pal.text_secondary);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(4);
    visuals.widgets.inactive.bg_fill = overlay(pal.dark, 0x0F);
    visuals.widgets.inactive.weak_bg_fill = overlay(pal.dark, 0x0A);
    visuals.widgets.inactive.bg_stroke = stroke(pal.stroke);
    visuals.widgets.inactive.fg_stroke = stroke(pal.text);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(4);
    visuals.widgets.hovered.bg_fill = pal.hover;
    visuals.widgets.hovered.weak_bg_fill = pal.hover;
    visuals.widgets.hovered.bg_stroke = stroke(overlay(pal.dark, 0x1A));
    visuals.widgets.hovered.fg_stroke = stroke(pal.text);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(4);
    visuals.widgets.active.bg_fill = overlay(pal.dark, 0x18);
    visuals.widgets.active.weak_bg_fill = overlay(pal.dark, 0x14);
    visuals.widgets.active.fg_stroke = stroke(pal.text);
    visuals.widgets.active.corner_radius = CornerRadius::same(4);
    visuals.widgets.open.bg_fill = pal.card;
    visuals.widgets.open.corner_radius = CornerRadius::same(4);
    ctx.set_visuals(visuals);

    ctx.style_mut(|style| {
        style.spacing.item_spacing = egui::vec2(8.0_f32, 6.0_f32);
        style.spacing.button_padding = egui::vec2(10.0_f32, 5.0_f32);
        style.spacing.menu_margin = egui::Margin::same(6);
        style.spacing.window_margin = egui::Margin::same(8);
    });
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let mut loaded = false;

    if let Ok(bytes) = std::fs::read(r"C:\Windows\Fonts\segoeui.ttf") {
        fonts.font_data.insert(
            "segoe_ui".to_owned(),
            Arc::new(FontData::from_owned(bytes)),
        );
        if let Some(proportional) = fonts.families.get_mut(&FontFamily::Proportional) {
            proportional.insert(0, "segoe_ui".to_owned());
        }
        if let Some(monospace) = fonts.families.get_mut(&FontFamily::Monospace) {
            monospace.push("segoe_ui".to_owned());
        }
        loaded = true;
    }

    // CJK coverage for zh-Hans (YaHei / fallback TTC).
    for (path, name, index) in [
        (r"C:\Windows\Fonts\msyh.ttc", "yahei", 0u32),
        (r"C:\Windows\Fonts\msyh.ttf", "yahei", 0u32),
        (r"C:\Windows\Fonts\simhei.ttf", "simhei", 0u32),
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            let mut data = FontData::from_owned(bytes);
            data.index = index;
            fonts.font_data.insert(name.to_owned(), Arc::new(data));
            if let Some(proportional) = fonts.families.get_mut(&FontFamily::Proportional) {
                proportional.push(name.to_owned());
            }
            if let Some(monospace) = fonts.families.get_mut(&FontFamily::Monospace) {
                monospace.push(name.to_owned());
            }
            loaded = true;
            break;
        }
    }

    for path in [
        r"C:\Windows\Fonts\SegoeIcons.ttf",
        r"C:\Windows\Fonts\segmdl2.ttf",
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            fonts.font_data.insert(
                ICON_FONT.to_owned(),
                Arc::new(FontData::from_owned(bytes)),
            );
            fonts.families.insert(
                FontFamily::Name(ICON_FONT.into()),
                vec![ICON_FONT.to_owned()],
            );
            loaded = true;
            break;
        }
    }

    if loaded {
        ctx.set_fonts(fonts);
    }
}
