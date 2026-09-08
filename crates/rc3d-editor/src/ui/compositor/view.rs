//! Compositor graph view transform helpers (pan/zoom/fit without mutating node sizes).

use egui::{emath::TSTransform, pos2, vec2, Pos2, Rect, Vec2};
use egui_snarl::Snarl;

use rc3d_render::{CompNode, CompOp};

use super::{sync::ui_input_count, NODE_W};

/// Header height (collapse button + title), in graph units.
const HEADER_EST: f32 = 26.0;
/// Height of one body parameter row (slider / dropdown + spacing).
const ROW_EST: f32 = 18.0;
/// Height of one socket strip row (`min_pin_row_height` in the node style).
const SOCKET_ROW_EST: f32 = 15.0;
/// Width of one bare-dot socket strip on a node edge.
const STRIP_W_EST: f32 = 20.0;
/// Bottom slack below the payload when estimating node height.
const NODE_END_PAD: f32 = 8.0;
/// Extra graph margin used by the exact "fit all" view command.
const BB_PAD: f32 = 32.0;
pub(super) const MIN_SCALE: f32 = 0.25;
pub(super) const MAX_SCALE: f32 = 2.0;
const ZOOM_STEP: f32 = 1.15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ViewCmd {
    ZoomIn,
    ZoomOut,
    Fit,
    Center,
}

/// Whether an op renders a parameter body in the node's center column.
/// Mirrors the `has_body` viewer hook (kept in one place so fit estimates and
/// the actual node layout cannot drift apart).
pub(super) fn op_has_body(op: CompOp) -> bool {
    matches!(
        op,
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

/// Number of parameter rows a node shows in its body (drives the fit estimate).
fn body_rows(op: CompOp) -> f32 {
    match op {
        CompOp::Mix => 2.0,          // blend type + Fac
        CompOp::AlphaOver => 1.0,    // Fac
        CompOp::Rgb => 2.0,          // color + alpha
        CompOp::Value => 1.0,
        CompOp::Math => 1.0,         // operator
        CompOp::BrightContrast => 2.0,
        CompOp::Blur => 1.0,
        CompOp::Bloom => 1.0,
        CompOp::Exposure | CompOp::Gamma | CompOp::Invert => 1.0,
        CompOp::HueSat => 3.0,
        CompOp::Translate | CompOp::Rotate | CompOp::Scale => 2.0,
        CompOp::Crop => 4.0,
        CompOp::DilateErode => 1.0,
        CompOp::ColorRamp => 4.0,
        _ => 0.0,
    }
}

/// Estimated node width in graph units. The header sets a `NODE_W` floor on
/// every node (bodyless render ops are about that wide), while ops with a
/// parameter body add the two bare-dot socket strips around the `NODE_W` body
/// column.
fn node_width_est(op: CompOp) -> f32 {
    if op_has_body(op) {
        NODE_W + STRIP_W_EST * 2.0
    } else {
        NODE_W
    }
}

/// Estimated node height in graph units. Payload height is the taller of the
/// body rows and the socket strip block, both starting under the header.
fn node_height_est(op: CompOp) -> f32 {
    let in_rows = ui_input_count(op);
    let out_rows = usize::from(op != CompOp::Viewer);
    let body_h = if op_has_body(op) {
        body_rows(op) * ROW_EST
    } else {
        0.0
    };
    let pin_h = in_rows.max(out_rows).max(1) as f32 * SOCKET_ROW_EST;
    HEADER_EST + body_h.max(pin_h) + NODE_END_PAD
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum NodeColorScheme {
    #[default]
    Default,
    Cool,
    Warm,
    HighContrast,
}

impl NodeColorScheme {
    pub(super) fn next(self) -> Self {
        match self {
            Self::Default => Self::Cool,
            Self::Cool => Self::Warm,
            Self::Warm => Self::HighContrast,
            Self::HighContrast => Self::Default,
        }
    }

    pub(super) fn label_key(self) -> &'static str {
        match self {
            Self::Default => "comp.view.color_default",
            Self::Cool => "comp.view.color_cool",
            Self::Warm => "comp.view.color_warm",
            Self::HighContrast => "comp.view.color_hc",
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ViewState {
    pub panel_rect: Rect,
    last_center: Option<Pos2>,
    last_size: Option<Vec2>,
    pub pending: Option<ViewCmd>,
    pub color_scheme: NodeColorScheme,
    pub last_scale: f32,
    /// Live view translation, updated each frame for background drawing.
    pub pan: Vec2,
    /// First-frame auto-fit (keeps node graph on screen after load / resize).
    fitted_once: bool,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            panel_rect: Rect::NOTHING,
            last_center: None,
            last_size: None,
            pending: None,
            color_scheme: NodeColorScheme::Default,
            last_scale: 1.0,
            pan: Vec2::ZERO,
            fitted_once: false,
        }
    }
}

impl ViewState {
    pub(super) fn begin_frame(&mut self, panel: Rect) {
        self.panel_rect = panel;
    }

    /// Apply resize compensation + pending view commands. Keeps node graph sizes
    /// (scale) unchanged on panel resize; only adjusts pan translation.
    pub(super) fn apply_transform(&mut self, to_global: &mut TSTransform, snarl: &Snarl<CompNode>) {
        let panel = self.panel_rect;
        if !panel.is_finite() || panel.width() < 8.0 || panel.height() < 8.0 {
            return;
        }

        if let (Some(prev_c), Some(prev_s)) = (self.last_center, self.last_size) {
            let size_delta = (panel.size() - prev_s).abs();
            if size_delta.x > 0.5 || size_delta.y > 0.5 {
                // Keep the graph point that sat under the old center under the new center.
                let delta = panel.center() - prev_c;
                to_global.translation += delta;
            }
        }

        // Auto-fit once the canvas has a real size: keeps freshly loaded
        // graphs (or the initial identity chain) centred and fully visible.
        if !self.fitted_once {
            self.fitted_once = true;
            if let Some(bb) = nodes_bb(snarl) {
                fit_view(to_global, bb, panel);
            }
        }

        if let Some(cmd) = self.pending.take() {
            match cmd {
                ViewCmd::ZoomIn => zoom_at(to_global, ZOOM_STEP, panel.center()),
                ViewCmd::ZoomOut => zoom_at(to_global, 1.0 / ZOOM_STEP, panel.center()),
                ViewCmd::Fit => {
                    if let Some(bb) = nodes_bb(snarl) {
                        fit_view(to_global, bb, panel);
                    }
                }
                ViewCmd::Center => {
                    if let Some(bb) = nodes_bb(snarl) {
                        center_view(to_global, bb, panel);
                    }
                }
            }
        }

        clamp_scale(to_global, panel.center());
        self.last_scale = to_global.scaling;
        self.pan = to_global.translation;
        self.last_center = Some(panel.center());
        self.last_size = Some(panel.size());
    }
}

fn nodes_bb(snarl: &Snarl<CompNode>) -> Option<Rect> {
    let mut bb = Rect::NOTHING;
    for (_id, pos, value) in snarl.nodes_pos_ids() {
        bb.extend_with(pos);
        bb.extend_with(pos2(
            pos.x + node_width_est(value.op),
            pos.y + node_height_est(value.op),
        ));
    }
    if bb.is_finite() {
        Some(bb)
    } else {
        None
    }
}

fn zoom_at(t: &mut TSTransform, factor: f32, screen_anchor: Pos2) {
    let new_scale = (t.scaling * factor).clamp(MIN_SCALE, MAX_SCALE);
    *t = scale_at(t, new_scale, screen_anchor);
}

fn fit_view(t: &mut TSTransform, view: Rect, panel: Rect) {
    let view = view.expand(BB_PAD);
    let size = view.size().max(vec2(1.0, 1.0));
    let scaling = (panel.size() / size).min_elem().clamp(MIN_SCALE, MAX_SCALE);
    *t = transform_matching_points(view.center(), panel.center(), scaling);
}

fn center_view(t: &mut TSTransform, view: Rect, panel: Rect) {
    let s = t.scaling.clamp(MIN_SCALE, MAX_SCALE);
    *t = transform_matching_points(view.center(), panel.center(), s);
}

fn clamp_scale(t: &mut TSTransform, anchor: Pos2) {
    let s = t.scaling.clamp(MIN_SCALE, MAX_SCALE);
    if (s - t.scaling).abs() > f32::EPSILON {
        *t = scale_at(t, s, anchor);
    }
}

fn scale_at(t: &TSTransform, new_scale: f32, screen_anchor: Pos2) -> TSTransform {
    let graph_anchor = t.inverse() * screen_anchor;
    TSTransform {
        scaling: new_scale,
        translation: screen_anchor.to_vec2() - graph_anchor.to_vec2() * new_scale,
    }
}

fn transform_matching_points(from: Pos2, to: Pos2, scaling: f32) -> TSTransform {
    TSTransform {
        scaling,
        translation: to.to_vec2() - from.to_vec2() * scaling,
    }
}
