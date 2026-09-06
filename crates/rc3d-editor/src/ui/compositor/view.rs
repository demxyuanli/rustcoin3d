//! Compositor graph view transform helpers (pan/zoom/fit without mutating node sizes).

use egui::{emath::TSTransform, pos2, vec2, Pos2, Rect, Vec2};
use egui_snarl::Snarl;

use rc3d_render::CompNode;

use super::NODE_W;

const NODE_H_EST: f32 = 110.0;
const BB_PAD: f32 = 48.0;
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
    for (_id, pos, _value) in snarl.nodes_pos_ids() {
        bb.extend_with(pos);
        bb.extend_with(pos2(pos.x + NODE_W, pos.y + NODE_H_EST));
    }
    if bb.is_finite() {
        Some(bb.expand(BB_PAD))
    } else {
        None
    }
}

fn zoom_at(t: &mut TSTransform, factor: f32, screen_anchor: Pos2) {
    let new_scale = (t.scaling * factor).clamp(MIN_SCALE, MAX_SCALE);
    *t = scale_at(t, new_scale, screen_anchor);
}

fn fit_view(t: &mut TSTransform, view: Rect, panel: Rect) {
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
