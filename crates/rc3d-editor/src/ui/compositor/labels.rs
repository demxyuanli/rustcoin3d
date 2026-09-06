//! Node titles, header colors and socket colors (Blender-like palette).

use egui::Color32;
use egui_snarl::ui::PinInfo;

use rc3d_render::{CompMenuGroup, CompOp, MathOp, MixBlend};

use crate::ui::i18n::{t, UiLocale};

use super::view::NodeColorScheme;

pub(super) fn op_title(loc: UiLocale, op: CompOp) -> &'static str {
    match op {
        CompOp::RenderLayers => t(loc, "comp.layers"),
        CompOp::Mix => t(loc, "comp.mix"),
        CompOp::Blur => t(loc, "comp.blur"),
        CompOp::BrightContrast => t(loc, "comp.bc"),
        CompOp::ColorRamp => t(loc, "comp.ramp"),
        CompOp::Rgb => t(loc, "comp.rgb"),
        CompOp::Value => t(loc, "comp.value"),
        CompOp::Math => t(loc, "comp.math"),
        CompOp::Exposure => t(loc, "comp.exposure"),
        CompOp::Gamma => t(loc, "comp.gamma"),
        CompOp::HueSat => t(loc, "comp.hue_sat"),
        CompOp::Invert => t(loc, "comp.invert"),
        CompOp::AlphaOver => t(loc, "comp.alpha_over"),
        CompOp::Translate => t(loc, "comp.translate"),
        CompOp::Rotate => t(loc, "comp.rotate"),
        CompOp::Scale => t(loc, "comp.scale"),
        CompOp::Crop => t(loc, "comp.crop"),
        CompOp::DilateErode => t(loc, "comp.dilate"),
        CompOp::Ssao => t(loc, "comp.ssao"),
        CompOp::Fxaa => t(loc, "comp.fxaa"),
        CompOp::Taa => t(loc, "comp.taa"),
        CompOp::ColorGrade => t(loc, "comp.grade"),
        CompOp::Bloom => t(loc, "comp.bloom"),
        CompOp::Dof => t(loc, "comp.dof"),
        CompOp::Ssr => t(loc, "comp.ssr"),
        CompOp::Fog => t(loc, "comp.fog"),
        CompOp::Edges => t(loc, "comp.edges"),
        CompOp::HiddenLine => t(loc, "comp.hlr"),
        CompOp::Xray => t(loc, "comp.xray"),
        CompOp::Grid => t(loc, "comp.grid"),
        CompOp::Shadows => t(loc, "comp.shadows"),
        CompOp::Viewer => t(loc, "comp.viewer"),
    }
}

/// Blender node-editor-like header palette. Hue follows the node category
/// (`menu_group`); light theme uses darker, desaturated tones for contrast.
pub(super) fn header_color(op: CompOp, scheme: NodeColorScheme) -> Color32 {
    let dark = match scheme {
        NodeColorScheme::Default => true,
        NodeColorScheme::Cool => true,
        NodeColorScheme::Warm => true,
        NodeColorScheme::HighContrast => false,
    };
    match op.menu_group() {
        // Input: muted red-brown (Render Layers / RGB / Value).
        CompMenuGroup::Input => {
            if dark {
                Color32::from_rgb(0x83, 0x3A, 0x33)
            } else {
                Color32::from_rgb(0xC0, 0x6A, 0x60)
            }
        }
        // Color: warm yellow-brown (Mix, ramps, HSV nodes).
        CompMenuGroup::Color => {
            if dark {
                Color32::from_rgb(0x82, 0x64, 0x33)
            } else {
                Color32::from_rgb(0xC8, 0xA0, 0x5A)
            }
        }
        // Filter: green (Blur, Dilate/Erode).
        CompMenuGroup::Filter => {
            if dark {
                Color32::from_rgb(0x4B, 0x70, 0x3A)
            } else {
                Color32::from_rgb(0x8A, 0xB0, 0x6A)
            }
        }
        // Transform: steel blue (Translate/Rotate/Scale/Crop).
        CompMenuGroup::Transform => {
            if dark {
                Color32::from_rgb(0x3A, 0x62, 0x83)
            } else {
                Color32::from_rgb(0x6A, 0xA0, 0xC8)
            }
        }
        // Converter: teal (Math).
        CompMenuGroup::Converter => {
            if dark {
                Color32::from_rgb(0x3A, 0x7A, 0x74)
            } else {
                Color32::from_rgb(0x6A, 0xB8, 0xB0)
            }
        }
        // CAD passes: violet.
        CompMenuGroup::CadPass => {
            if dark {
                Color32::from_rgb(0x6A, 0x4B, 0x7E)
            } else {
                Color32::from_rgb(0xA8, 0x82, 0xC0)
            }
        }
    }
}

pub(super) fn viewer_color(scheme: NodeColorScheme) -> Color32 {
    match scheme {
        NodeColorScheme::Default => Color32::from_rgb(0x5A, 0x28, 0x3A),
        NodeColorScheme::Cool => Color32::from_rgb(0x3A, 0x3A, 0x6A),
        NodeColorScheme::Warm => Color32::from_rgb(0x7A, 0x3A, 0x30),
        NodeColorScheme::HighContrast => Color32::from_rgb(0xD0, 0x40, 0x90),
    }
}

pub(super) fn image_pin() -> PinInfo {
    // Blender "Color/Image" socket: bright yellow-green.
    PinInfo::circle()
        .with_fill(Color32::from_rgb(0xE7, 0xE1, 0x55))
        .with_wire_color(Color32::from_rgb(0xC9, 0xC3, 0x4C))
}

pub(super) fn value_pin() -> PinInfo {
    // Blender "Value" socket: neutral gray.
    PinInfo::circle()
        .with_fill(Color32::from_rgb(0x8F, 0x8F, 0x8F))
        .with_wire_color(Color32::from_rgb(0x8F, 0x8F, 0x8F))
}

pub(super) fn factor_pin() -> PinInfo {
    // Blender "Factor" socket: bright green.
    PinInfo::circle()
        .with_fill(Color32::from_rgb(0x7F, 0xE0, 0x50))
        .with_wire_color(Color32::from_rgb(0x6F, 0xC8, 0x45))
}

/// Output socket label + color per op, Blender-style (Image / Value / Color).
pub(super) fn output_pin_info(op: CompOp, loc: UiLocale) -> (String, PinInfo) {
    match op {
        CompOp::Math | CompOp::Value => (t(loc, "comp.socket.value").to_owned(), value_pin()),
        _ => (t(loc, "comp.socket.image").to_owned(), image_pin()),
    }
}

pub(super) const MATH_OPS: [MathOp; 14] = [
    MathOp::Add,
    MathOp::Subtract,
    MathOp::Multiply,
    MathOp::Divide,
    MathOp::Power,
    MathOp::Minimum,
    MathOp::Maximum,
    MathOp::LessThan,
    MathOp::GreaterThan,
    MathOp::Absolute,
    MathOp::Floor,
    MathOp::Ceiling,
    MathOp::Sine,
    MathOp::Cosine,
];

pub(super) fn blend_label(loc: UiLocale, b: MixBlend) -> &'static str {
    match b {
        MixBlend::Mix => t(loc, "comp.blend.mix"),
        MixBlend::Add => t(loc, "comp.blend.add"),
        MixBlend::Subtract => t(loc, "comp.blend.sub"),
        MixBlend::Multiply => t(loc, "comp.blend.mul"),
        MixBlend::Screen => t(loc, "comp.blend.screen"),
        MixBlend::Divide => t(loc, "comp.blend.divide"),
        MixBlend::Difference => t(loc, "comp.blend.difference"),
        MixBlend::Darken => t(loc, "comp.blend.darken"),
        MixBlend::Lighten => t(loc, "comp.blend.lighten"),
        MixBlend::Overlay => t(loc, "comp.blend.overlay"),
        MixBlend::Dodge => t(loc, "comp.blend.dodge"),
        MixBlend::Burn => t(loc, "comp.blend.burn"),
        MixBlend::Hue => t(loc, "comp.blend.hue"),
        MixBlend::Saturation => t(loc, "comp.blend.sat"),
        MixBlend::Value => t(loc, "comp.blend.value"),
        MixBlend::Color => t(loc, "comp.blend.color"),
    }
}

pub(super) fn math_label(loc: UiLocale, m: MathOp) -> &'static str {
    match m {
        MathOp::Add => t(loc, "comp.math.add"),
        MathOp::Subtract => t(loc, "comp.math.subtract"),
        MathOp::Multiply => t(loc, "comp.math.multiply"),
        MathOp::Divide => t(loc, "comp.math.divide"),
        MathOp::Power => t(loc, "comp.math.power"),
        MathOp::Minimum => t(loc, "comp.math.min"),
        MathOp::Maximum => t(loc, "comp.math.max"),
        MathOp::LessThan => t(loc, "comp.math.less"),
        MathOp::GreaterThan => t(loc, "comp.math.greater"),
        MathOp::Absolute => t(loc, "comp.math.abs"),
        MathOp::Floor => t(loc, "comp.math.floor"),
        MathOp::Ceiling => t(loc, "comp.math.ceil"),
        MathOp::Sine => t(loc, "comp.math.sin"),
        MathOp::Cosine => t(loc, "comp.math.cos"),
    }
}
