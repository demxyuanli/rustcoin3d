//! Shared menu labels and context-menu action rows reused by inspector /
//! hierarchy / viewport context menus.

use rc3d_core::NodeId;
use rc3d_render::renderer::CadDisplayTier;
use rc3d_render::viewport::LayoutMode;

use crate::commands::{AdaptiveQualityMode, EditorCommand};
use crate::ui::i18n::{t, UiLocale};
use crate::ui::icons::{self, Icon};
use crate::ui::theme::ThemePalette;
use crate::ui::types::EditorUiContext;

pub(in crate::ui) fn view_render_features_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    let mut feat = |label_key: &'static str, field: bool, name: &'static str| {
        let mut v = field;
        if ui.checkbox(&mut v, t(loc, label_key)).changed() {
            push(EditorCommand::SetRenderFeature {
                feature_name: name,
                enabled: v,
            });
        }
    };
    let f = ui_ctx.render_features;
    feat("feat.taa", f.taa, "taa");
    feat("feat.motion_blur", f.motion_blur, "motion_blur");
    feat("feat.ssr", f.ssr, "ssr");
    feat("feat.color_grading", f.color_grading, "color_grading");
    feat("feat.dof", f.dof, "dof");
    feat("feat.fog", f.volumetric_fog, "volumetric_fog");
    feat("feat.cluster", f.cluster_lights, "cluster_lights");
    feat("feat.omni", f.omni_shadows, "omni_shadows");
    feat("feat.fxaa", f.ldr_fxaa, "ldr_fxaa");
    feat("feat.gpu_cull", f.gpu_cull, "gpu_cull");
    feat("feat.parallel", f.parallel_traversal, "parallel_traversal");
}

pub(in crate::ui) fn cad_tier_label(loc: UiLocale, tier: CadDisplayTier) -> &'static str {
    match tier {
        CadDisplayTier::DesignCreation => t(loc, "tier.design"),
        CadDisplayTier::Visualization => t(loc, "tier.viz"),
        CadDisplayTier::IndustrialDisplay => t(loc, "tier.industrial"),
        CadDisplayTier::ProductRendering => t(loc, "tier.product"),
    }
}

pub(in crate::ui) fn layout_mode_label(loc: UiLocale, mode: LayoutMode) -> &'static str {
    match mode {
        LayoutMode::Single => t(loc, "layout.single"),
        LayoutMode::Quad => t(loc, "layout.quad"),
        LayoutMode::LeftRight => t(loc, "layout.leftright"),
        LayoutMode::TopBottom => t(loc, "layout.topbottom"),
    }
}

pub(in crate::ui) fn aq_mode_label(loc: UiLocale, mode: AdaptiveQualityMode) -> &'static str {
    match mode {
        AdaptiveQualityMode::Off => t(loc, "render.aq.off"),
        AdaptiveQualityMode::On => t(loc, "render.aq.on"),
        AdaptiveQualityMode::AutoIdleLock => t(loc, "render.aq.idle"),
    }
}

pub(in crate::ui) fn view_preset_label(
    loc: UiLocale,
    preset: rc3d_engine_api::camera::ViewPreset,
) -> &'static str {
    use rc3d_engine_api::camera::ViewPreset;
    match preset {
        ViewPreset::Top => t(loc, "view.top"),
        ViewPreset::Bottom => t(loc, "view.bottom"),
        ViewPreset::Front => t(loc, "view.front"),
        ViewPreset::Back => t(loc, "view.back"),
        ViewPreset::Right => t(loc, "view.right"),
        ViewPreset::Left => t(loc, "view.left"),
        ViewPreset::Iso => t(loc, "view.iso"),
    }
}

/// Hierarchy / viewport context-menu rows for selection visibility.
/// Returns true if any row was clicked (caller may close a floating menu).
pub(in crate::ui) fn selection_context_actions(
    ui: &mut egui::Ui,
    loc: UiLocale,
    pal: &ThemePalette,
    push: &mut impl FnMut(EditorCommand),
) -> bool {
    let mut hit = false;
    if icons::icon_menu_button(ui, Icon::EyeOff, t(loc, "ctx.hide"), pal).clicked() {
        push(EditorCommand::HideSelected);
        hit = true;
    }
    if icons::icon_menu_button(ui, Icon::FitView, t(loc, "ctx.isolate"), pal).clicked() {
        push(EditorCommand::IsolateSelected);
        hit = true;
    }
    if icons::icon_menu_button(ui, Icon::Eye, t(loc, "ctx.reveal"), pal).clicked() {
        push(EditorCommand::RevealHidden);
        hit = true;
    }
    if icons::icon_menu_button(ui, Icon::Lock, t(loc, "ctx.lock"), pal).clicked() {
        push(EditorCommand::ToggleLockSelected);
        hit = true;
    }
    hit
}

pub(in crate::ui) fn hierarchy_node_context_actions(
    ui: &mut egui::Ui,
    nodes: &[NodeId],
    loc: UiLocale,
    pal: &ThemePalette,
    push: &mut impl FnMut(EditorCommand),
) {
    if icons::icon_menu_button(ui, Icon::Duplicate, t(loc, "hier.duplicate"), pal).clicked() {
        for &id in nodes {
            push(EditorCommand::DuplicateNode(id));
        }
    }
    if icons::icon_menu_button(ui, Icon::Delete, t(loc, "hier.delete"), pal).clicked() {
        for &id in nodes {
            push(EditorCommand::DeleteNode(id));
        }
    }
}
