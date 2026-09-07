//! egui-snarl viewer: node rendering, style and body widgets.
//!
//! Layout follows the Blender compositor convention:
//! - inputs as sockets on the node's left edge, outputs on the right edge
//!   (`NodeLayoutKind::Coil`);
//! - each socket row is a thin label strip on the edge column while parameter
//!   rows (dropdowns / sliders / fields) sit in the wider center column;
//! - wires are smooth cubic Beziers whose curvature adapts to the distance
//!   between sockets.

use egui::{
    emath::TSTransform, pos2, vec2, Color32, CornerRadius, Frame, Margin, Painter, Pos2, Rect,
    Shadow, Stroke, Style, Ui,
};
use egui_snarl::{
    ui::{
        AnyPins, BackgroundPattern, PinInfo, SelectionStyle, SnarlStyle, SnarlViewer, WireStyle,
    },
    InPin, InPinId, NodeId, OutPin, OutPinId, Snarl,
};

use rc3d_render::{CompNode, CompOp, MixBlend};

use crate::ui::i18n::{t, UiLocale};
use crate::ui::icons::{self, Icon, PANEL_BTN};
use crate::ui::theme::{paint_icon_glyph, ThemePalette};

use super::labels::{
    blend_label, header_color, input_pin, input_socket_label, math_label, op_title,
    output_pin_info, value_pin, viewer_color, MATH_OPS,
};
use super::sync::{first_image_pin, graph_slot, protected, ui_input_count};
use super::view::{NodeColorScheme, ViewState, MIN_SCALE};
use super::widgets::{
    blender_color_row_rgb, blender_dropdown, blender_slider, param_row, socket_field, SLIDER_H,
};
use super::{pick_add_op, DOT_STEP, NODE_FILL, NODE_R, NODE_W};

/// Horizontal gap between the socket (which sits on the node edge) and the
/// first piece of text in the same socket row.
const SOCKET_GAP: f32 = 7.0;
/// Muted white used for socket labels.
const SOCKET_TEXT: Color32 = Color32::from_rgb(0xC8, 0xC8, 0xC8);

pub(super) fn force_node_width(ui: &mut Ui) {
    ui.set_min_width(NODE_W);
    ui.set_max_width(NODE_W);
}

pub(super) fn compositor_style() -> SnarlStyle {
    let mut layout = egui_snarl::ui::NodeLayout::coil();
    // Rows in the socket columns are slightly taller than the widgets so the
    // dots breathe like Blender's socket strips.
    layout.min_pin_row_height = SLIDER_H + 8.0;
    // Blender-like socket: dark outline around the colored circle.
    let pin_stroke = Stroke::new(1.4, Color32::from_rgb(0x16, 0x16, 0x16));
    SnarlStyle {
        node_layout: Some(layout),
        pin_placement: Some(egui_snarl::ui::PinPlacement::Edge),
        pin_size: Some(5.5),
        pin_stroke: Some(pin_stroke),
        // Noticeable smooth connections: adaptive cubic Beziers.
        wire_width: Some(1.6),
        wire_style: Some(WireStyle::Bezier3),
        wire_smoothness: Some(150.0),
        collapsible: Some(false),
        min_scale: Some(MIN_SCALE),
        max_scale: Some(super::view::MAX_SCALE),
        crisp_magnified_text: Some(true),
        centering: Some(true),
        bg_pattern: Some(BackgroundPattern::NoPattern),
        // Node body: dark slab with a soft drop shadow, like Blender nodes.
        node_frame: Some(Frame {
            inner_margin: Margin::symmetric(4, 2),
            outer_margin: Margin::ZERO,
            corner_radius: CornerRadius::same(NODE_R),
            fill: NODE_FILL,
            stroke: Stroke::new(1.0, Color32::from_rgb(0x0A, 0x0A, 0x0A)),
            shadow: Shadow {
                offset: [0, 4],
                blur: 12,
                spread: 0,
                color: Color32::from_black_alpha(110),
            },
        }),
        // Header: colored strip, flat (shadow lives on the node frame).
        header_frame: Some(Frame {
            inner_margin: Margin::symmetric(2, 2),
            outer_margin: Margin::ZERO,
            corner_radius: CornerRadius {
                nw: NODE_R,
                ne: NODE_R,
                sw: 0,
                se: 0,
            },
            fill: NODE_FILL,
            stroke: Stroke::NONE,
            shadow: Shadow::NONE,
        }),
        // Header drag space: compact so the title starts close to the edge.
        header_drag_space: Some(egui::vec2(0.0, 0.0)),
        bg_frame: Some(Frame {
            inner_margin: Margin::ZERO,
            outer_margin: Margin::ZERO,
            corner_radius: CornerRadius::ZERO,
            fill: Color32::from_rgb(0x1D, 0x1D, 0x1D),
            stroke: Stroke::NONE,
            shadow: Shadow::NONE,
        }),
        // Blender-style selection: soft dark-orange outline.
        select_style: Some(SelectionStyle {
            margin: Margin::same(4),
            rounding: CornerRadius::same(NODE_R + 2),
            fill: Color32::TRANSPARENT,
            stroke: Stroke::new(2.0, Color32::from_rgb(0xFF, 0xAA, 0x40)),
        }),
        ..SnarlStyle::new()
    }
}

const GROUP_BASIC: [MixBlend; 4] = [
    MixBlend::Mix,
    MixBlend::Add,
    MixBlend::Subtract,
    MixBlend::Multiply,
];
const GROUP_MUL: [MixBlend; 8] = [
    MixBlend::Screen,
    MixBlend::Divide,
    MixBlend::Difference,
    MixBlend::Darken,
    MixBlend::Lighten,
    MixBlend::Overlay,
    MixBlend::Dodge,
    MixBlend::Burn,
];
const GROUP_HSV: [MixBlend; 4] = [
    MixBlend::Hue,
    MixBlend::Saturation,
    MixBlend::Value,
    MixBlend::Color,
];

/// One parameter row rendered inside the center column of a node.
fn draw_node_body(ui: &mut Ui, n: &mut CompNode, loc: UiLocale) {
    match n.op {
        CompOp::Mix => {
            let popup_id = ui.make_persistent_id("comp-blend-modes");
            blender_dropdown(
                ui,
                blend_label(loc, n.mix_blend),
                popup_id,
                |ui| {
                    ui.set_min_width(150.0);
                    let groups: [(&str, &[MixBlend]); 3] = [
                        ("comp.blend.grp_basic", &GROUP_BASIC),
                        ("comp.blend.grp_mul", &GROUP_MUL),
                        ("comp.blend.grp_hsv", &GROUP_HSV),
                    ];
                    for (label, group) in groups {
                        ui.label(t(loc, label));
                        for b in group {
                            if ui
                                .selectable_label(n.mix_blend == *b, blend_label(loc, *b))
                                .clicked()
                            {
                                n.mix_blend = *b;
                                ui.close();
                            }
                        }
                        ui.separator();
                    }
                },
            );
            param_row(ui, &t(loc, "comp.fac"), &mut n.fac, 0.0..=1.0);
        }
        CompOp::AlphaOver => {
            param_row(ui, &t(loc, "comp.fac"), &mut n.fac, 0.0..=1.0);
        }
        CompOp::BrightContrast => {
            param_row(ui, &t(loc, "comp.brightness"), &mut n.brightness, -1.0..=1.0);
            param_row(ui, &t(loc, "comp.contrast"), &mut n.contrast, -1.0..=1.0);
        }
        CompOp::Blur => {
            param_row(ui, &t(loc, "comp.radius"), &mut n.blur_radius, 0.0..=24.0);
        }
        CompOp::Bloom => {
            param_row(ui, &t(loc, "comp.fac"), &mut n.fac, 0.0..=1.0);
        }
        CompOp::Rgb => {
            // Blender RGB node: color row with right swatch + picker popup.
            let popup_id = ui.make_persistent_id("comp-rgb-picker");
            blender_color_row_rgb(ui, &t(loc, "comp.color"), &mut n.color, popup_id);
            param_row(ui, &t(loc, "comp.alpha"), &mut n.color[3], 0.0..=1.0);
        }
        CompOp::Value => {
            param_row(ui, &t(loc, "comp.value"), &mut n.value, -100.0..=100.0);
        }
        CompOp::Math => {
            let popup_id = ui.make_persistent_id("comp-math-op");
            blender_dropdown(
                ui,
                math_label(loc, n.math_op),
                popup_id,
                |ui| {
                    ui.set_min_width(120.0);
                    for m in MATH_OPS {
                        if ui
                            .selectable_label(n.math_op == m, math_label(loc, m))
                            .clicked()
                        {
                            n.math_op = m;
                            ui.close();
                        }
                    }
                },
            );
        }
        CompOp::HueSat => {
            param_row(ui, &t(loc, "comp.hue"), &mut n.hsv[0], 0.0..=1.0);
            param_row(ui, &t(loc, "comp.saturation"), &mut n.hsv[1], 0.0..=2.0);
            param_row(ui, &t(loc, "comp.value"), &mut n.hsv[2], 0.0..=2.0);
        }
        CompOp::Invert => {
            param_row(ui, &t(loc, "comp.fac"), &mut n.invert_fac, 0.0..=1.0);
        }
        CompOp::Exposure => {
            param_row(ui, &t(loc, "comp.exposure"), &mut n.value, -10.0..=10.0);
        }
        CompOp::Gamma => {
            param_row(ui, &t(loc, "comp.gamma"), &mut n.value, 0.01..=10.0);
        }
        CompOp::Translate => {
            param_row(ui, &t(loc, "comp.offset.x"), &mut n.transform[0], -2.0..=2.0);
            param_row(ui, &t(loc, "comp.offset.y"), &mut n.transform[1], -2.0..=2.0);
        }
        CompOp::Rotate => {
            param_row(
                ui,
                &t(loc, "comp.angle"),
                &mut n.transform[2],
                -std::f32::consts::TAU..=std::f32::consts::TAU,
            );
        }
        CompOp::Scale => {
            param_row(ui, &t(loc, "comp.scale"), &mut n.transform[3], 0.05..=8.0);
        }
        CompOp::DilateErode => {
            let mut dist = n.morph_amount as f32;
            param_row(ui, &t(loc, "comp.distance"), &mut dist, -16.0..=16.0);
            n.morph_amount = dist.round() as i32;
        }
        CompOp::Crop => {
            blender_slider(ui, &t(loc, "comp.crop.left"), &mut n.crop[0], 0.0..=1.0);
            blender_slider(ui, &t(loc, "comp.crop.top"), &mut n.crop[1], 0.0..=1.0);
            blender_slider(ui, &t(loc, "comp.crop.right"), &mut n.crop[2], 0.0..=1.0);
            blender_slider(ui, &t(loc, "comp.crop.bottom"), &mut n.crop[3], 0.0..=1.0);
        }
        CompOp::ColorRamp => {
            for i in 0..n.ramp_count.min(4) as usize {
                let popup_id = ui.make_persistent_id("comp-ramp-picker").with(i);
                let mut rgba = [n.ramp[i].1[0], n.ramp[i].1[1], n.ramp[i].1[2], 1.0_f32];
                blender_color_row_rgb(ui, "Pos", &mut rgba, popup_id);
                n.ramp[i].1 = [rgba[0], rgba[1], rgba[2]];
                blender_slider(ui, "Pos", &mut n.ramp[i].0, 0.0..=1.0);
            }
        }
        _ => {}
    }
}

/// Muted socket name next to a pin (left column inputs / right column outputs).
fn socket_label(ui: &mut Ui, text: &str) {
    ui.label(egui::RichText::new(text).size(10.0).color(SOCKET_TEXT));
}

pub(super) struct CompViewer<'a> {
    pub(super) loc: UiLocale,
    pub(super) view: &'a mut ViewState,
    pub(super) color_scheme: NodeColorScheme,
}

impl SnarlViewer<CompNode> for CompViewer<'_> {
    fn title(&mut self, node: &CompNode) -> String {
        op_title(self.loc, node.op).to_owned()
    }

    fn current_transform(&mut self, to_global: &mut TSTransform, snarl: &mut Snarl<CompNode>) {
        self.view.apply_transform(to_global, snarl);
    }

    fn inputs(&mut self, node: &CompNode) -> usize {
        ui_input_count(node.op)
    }

    fn outputs(&mut self, node: &CompNode) -> usize {
        match node.op {
            CompOp::Viewer => 0,
            _ => 1,
        }
    }

    fn show_header(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<CompNode>,
    ) {
        // NOTE: only expand min_rect here. `set_max_width` would rewind the
        // horizontal cursor back to the node's left edge and paint the title
        // on top of the collapse arrow snarl just allocated.
        ui.set_min_width(NODE_W);
        ui.set_visuals(egui::Visuals::dark());
        ui.style_mut().visuals.widgets.inactive.fg_stroke =
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 230));
        ui.style_mut()
            .text_styles
            .insert(egui::TextStyle::Body, egui::FontId::proportional(10.0));
        // Character chevron replaces snarl's oversized triangle icon; small
        // click target toggles node openness (style.collapsible is disabled).
        // Glyph comes from Segoe Fluent Icons (same as the rest of the UI);
        // the generic ▾/▸ chars rendered as tofu boxes in egui's fonts.
        let open = snarl.get_node_info(node).is_some_and(|n| n.open);
        let codepoint = if open {
            icons::CODEPOINT_CHEVRON_DOWN
        } else {
            icons::CODEPOINT_CHEVRON_RIGHT
        };
        let (hit, btn_resp) = ui.allocate_exact_size(vec2(9.0, 12.0), egui::Sense::click());
        paint_icon_glyph(
            ui.painter(),
            hit,
            codepoint,
            7.0,
            Color32::from_rgba_unmultiplied(255, 255, 255, 230),
        );
        let btn_resp = btn_resp.on_hover_text(t(self.loc, "comp.toggle"));
        if btn_resp.clicked() {
            snarl.open_node(node, !open);
            // Keep the CompNode value in sync so export_graph sees the change
            // even before the next snarl rebuild.
            if let Some(info) = snarl.get_node_info_mut(node) {
                info.value.open = !open;
            }
        }
        ui.strong(op_title(self.loc, snarl[node].op));
    }

    #[allow(refining_impl_trait)]
    fn show_input(&mut self, pin: &InPin, ui: &mut Ui, snarl: &mut Snarl<CompNode>) -> PinInfo {
        // Input socket rows live in the narrow strip along the node's left
        // edge. Row content is the socket name (A/B for multi-input ops) plus,
        // for scalar operands that fall back to a constant when unlinked, a
        // compact inline field.
        ui.spacing_mut().item_spacing.x = 3.0;
        ui.spacing_mut().item_spacing.y = 1.0;
        let loc = self.loc;
        let n = &mut snarl[pin.id.node];
        let linked = !pin.remotes.is_empty();
        // Keep the label clear of the socket circle that sits on the edge.
        ui.add_space(SOCKET_GAP);
        match (n.op, pin.id.input) {
            (CompOp::Math, 0) => {
                socket_label(ui, t(loc, "comp.socket.a"));
                if !linked {
                    socket_field(ui, &mut n.value_a, -100.0..=100.0);
                }
                value_pin()
            }
            (CompOp::Math, 1) => {
                socket_label(ui, t(loc, "comp.socket.b"));
                if !linked {
                    socket_field(ui, &mut n.value, -100.0..=100.0);
                }
                value_pin()
            }
            _ => {
                let text = input_socket_label(n.op, pin.id.input, loc)
                    .unwrap_or(t(loc, "comp.socket.image"));
                socket_label(ui, text);
                input_pin(n.op, pin.id.input)
            }
        }
    }

    #[allow(refining_impl_trait)]
    fn show_output(&mut self, _pin: &OutPin, ui: &mut Ui, snarl: &mut Snarl<CompNode>) -> PinInfo {
        // Output socket rows live along the node's right edge: label to the
        // left of the socket, mirrored from the input column.
        ui.spacing_mut().item_spacing.x = 3.0;
        let n = &snarl[_pin.id.node];
        let (label, pin) = output_pin_info(n.op, self.loc);
        // Right-to-left allocation: the gap ends up right next to the socket,
        // the label just left of it.
        ui.add_space(SOCKET_GAP);
        socket_label(ui, &label);
        pin
    }

    fn node_frame(
        &mut self,
        mut default: Frame,
        _node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        _snarl: &Snarl<CompNode>,
    ) -> Frame {
        default.corner_radius = CornerRadius::same(NODE_R);
        default.shadow = Shadow::NONE;
        default
    }

    fn header_frame(
        &mut self,
        mut default: Frame,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        snarl: &Snarl<CompNode>,
    ) -> Frame {
        let n = &snarl[node];
        default.fill = if n.op == CompOp::Viewer {
            viewer_color(self.color_scheme)
        } else {
            header_color(n.op, self.color_scheme)
        };
        default.corner_radius = CornerRadius {
            nw: NODE_R,
            ne: NODE_R,
            sw: 0,
            se: 0,
        };
        default.stroke = Stroke::NONE;
        default.shadow = Shadow::NONE;
        default
    }

    fn has_body(&mut self, node: &CompNode) -> bool {
        matches!(
            node.op,
            CompOp::Mix
                | CompOp::AlphaOver
                | CompOp::BrightContrast
                | CompOp::Blur
                | CompOp::Bloom
                | CompOp::ColorRamp
                | CompOp::Rgb
                | CompOp::Value
                | CompOp::Math
                | CompOp::HueSat
                | CompOp::Invert
                | CompOp::Crop
                | CompOp::Exposure
                | CompOp::Gamma
                | CompOp::Translate
                | CompOp::Rotate
                | CompOp::Scale
                | CompOp::DilateErode
        )
    }

    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<CompNode>,
    ) {
        force_node_width(ui);
        draw_node_body(ui, &mut snarl[node], self.loc);
    }

    fn draw_background(
        &mut self,
        _background: Option<&BackgroundPattern>,
        viewport: &Rect,
        _snarl_style: &SnarlStyle,
        _style: &Style,
        painter: &Painter,
        _snarl: &Snarl<CompNode>,
    ) {
        // Blender-like dot grid: spacing is fixed in graph space and scales
        // with zoom; dots stay aligned to the graph origin via the view
        // translation. Dot spacing halves between zoom levels to keep density
        // constant on screen while the step visibly changes (Blender style).
        let scale = self.view.last_scale.max(0.05);
        // Densify when zoomed out, thin out when zoomed in (screen-space clamp).
        let mut step = DOT_STEP * scale;
        while step < 14.0 {
            step *= 2.0;
        }
        while step > 56.0 {
            step *= 0.5;
        }
        // Two dot intensities: strong dots on the current level, faint dots on
        // the level below (they render where the coarse dots would land at
        // 2x spacing), which reads like Blender's two-tier grid.
        let strong = Color32::from_rgb(0x46, 0x46, 0x46);
        let faint = Color32::from_rgb(0x32, 0x32, 0x32);
        let radius = (step * 0.045).clamp(0.6, 1.8);
        let pan = self.view.pan;
        let mut y = (viewport.min.y - pan.y).rem_euclid(step) + viewport.min.y;
        while y <= viewport.max.y {
            let mut x = (viewport.min.x - pan.x).rem_euclid(step) + viewport.min.x;
            while x <= viewport.max.x {
                painter.circle_filled(pos2(x, y), radius, strong);
                x += step;
            }
            y += step;
        }
        // Faint dots at the 2x grid (alternate levels).
        let coarse = step * 2.0;
        let mut y = (viewport.min.y - pan.y).rem_euclid(coarse) + viewport.min.y;
        while y <= viewport.max.y {
            let mut x = (viewport.min.x - pan.x).rem_euclid(coarse) + viewport.min.x;
            while x <= viewport.max.x {
                painter.circle_filled(pos2(x, y), radius * 0.8, faint);
                x += coarse;
            }
            y += coarse;
        }
    }

    fn has_graph_menu(&mut self, _pos: Pos2, _snarl: &mut Snarl<CompNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: Pos2, ui: &mut Ui, snarl: &mut Snarl<CompNode>) {
        ui.label(t(self.loc, "comp.add_node"));
        pick_add_op(ui, self.loc, |op| {
            snarl.insert_node(pos, CompNode::new(0, op, [pos.x, pos.y]));
        });
    }

    fn has_dropped_wire_menu(
        &mut self,
        _src_pins: AnyPins<'_>,
        _snarl: &mut Snarl<CompNode>,
    ) -> bool {
        true
    }

    fn show_dropped_wire_menu(
        &mut self,
        pos: Pos2,
        ui: &mut Ui,
        src_pins: AnyPins<'_>,
        snarl: &mut Snarl<CompNode>,
    ) {
        ui.label(t(self.loc, "comp.add_node"));
        match src_pins {
            AnyPins::Out(pins) => {
                let from = pins.first().copied();
                pick_add_op(ui, self.loc, |op| {
                    let id = snarl.insert_node(pos, CompNode::new(0, op, [pos.x, pos.y]));
                    if let Some(from) = from {
                        let input = first_image_pin(snarl[id].op);
                        if ui_input_count(snarl[id].op) > input {
                            let _ = snarl.connect(from, InPinId { node: id, input });
                        }
                    }
                });
            }
            AnyPins::In(pins) => {
                let to = pins.first().copied();
                pick_add_op(ui, self.loc, |op| {
                    let id = snarl.insert_node(pos, CompNode::new(0, op, [pos.x, pos.y]));
                    if let Some(to) = to {
                        let _ = snarl.connect(
                            OutPinId {
                                node: id,
                                output: 0,
                            },
                            to,
                        );
                    }
                });
            }
        }
    }

    fn has_node_menu(&mut self, node: &CompNode) -> bool {
        !protected(node.op)
    }

    fn show_node_menu(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<CompNode>,
    ) {
        let pal = if ui.visuals().dark_mode {
            ThemePalette::dark()
        } else {
            ThemePalette::light()
        };
        if icons::icon_button(
            ui,
            Icon::Delete,
            t(self.loc, "comp.delete"),
            false,
            PANEL_BTN,
            &pal,
        )
        .clicked()
        {
            let _ = snarl.remove_node(node);
            ui.close();
        }
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<CompNode>) {
        if from.id.node == to.id.node {
            return;
        }
        let Some(dst) = snarl.get_node(to.id.node) else {
            return;
        };
        if to.id.input >= ui_input_count(dst.op) {
            return;
        }
        if graph_slot(dst.op, to.id.input).is_none() {
            return;
        }
        for &remote in &to.remotes {
            let _ = snarl.disconnect(remote, to.id);
        }
        let _ = snarl.connect(from.id, to.id);
    }
}
