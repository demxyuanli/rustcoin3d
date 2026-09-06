//! Render settings side tab (shading toggles, IBL, CSM, layout, quality).

use rc3d_render::ibl::IblPreset;
use rc3d_render::renderer::CadDisplayTier;
use rc3d_render::viewport::LayoutMode;

use crate::commands::{AdaptiveQualityMode, EditorCommand};
use crate::ui::i18n::{t, tf};
use crate::ui::icons::{Icon, PANEL_BTN};
use crate::ui::types::EditorUiContext;

pub(in crate::ui) fn draw_render_panel(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    push: &mut impl FnMut(EditorCommand),
) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        render_panel(ui, ui_ctx, push);
    });
}

/// Checkbox bound to a ui_ctx flag; on change pushes `cmd(new_value)`.
fn check_cmd(
    ui: &mut egui::Ui,
    loc: crate::ui::i18n::UiLocale,
    value: &mut bool,
    key: &'static str,
    cmd: impl FnOnce(bool) -> EditorCommand,
    push: &mut impl FnMut(EditorCommand),
) {
    if ui.checkbox(value, t(loc, key)).changed() {
        push(cmd(*value));
    }
}

fn render_panel(ui: &mut egui::Ui, ui_ctx: &EditorUiContext, push: &mut impl FnMut(EditorCommand)) {
    let loc = ui_ctx.ui_locale;
    let mut hdr = ui_ctx.hdr_enabled;
    check_cmd(ui, loc, &mut hdr, "render.hdr", EditorCommand::SetHdrPostProcessing, push);
    let mut vsync = ui_ctx.vsync_enabled;
    check_cmd(ui, loc, &mut vsync, "render.vsync", EditorCommand::SetVsyncEnabled, push);
    let mut grid = ui_ctx.grid_enabled;
    check_cmd(ui, loc, &mut grid, "view.grid", EditorCommand::SetGridEnabled, push);
    let mut hud = ui_ctx.hud_enabled;
    check_cmd(ui, loc, &mut hud, "view.hud", EditorCommand::SetHudEnabled, push);
    let mut xray = ui_ctx.xray_mode;
    check_cmd(ui, loc, &mut xray, "vis.xray", EditorCommand::SetXrayMode, push);
    let mut ghost = ui_ctx.ghost_unselected;
    check_cmd(ui, loc, &mut ghost, "vis.ghost", EditorCommand::SetGhostUnselected, push);
    let mut go = ui_ctx.ghost_opacity;
    if ui
        .add(egui::Slider::new(&mut go, 0.02_f32..=0.95_f32).text(t(loc, "vis.ghost_opacity")))
        .changed()
    {
        push(EditorCommand::SetGhostOpacity(go));
    }
    let mut wboit = ui_ctx.wboit_enabled;
    check_cmd(ui, loc, &mut wboit, "render.wboit", EditorCommand::SetWboit, push);

    ui.label(t(loc, "render.ibl.preset"));
    let mut preset = ui_ctx.ibl_preset;
    egui::ComboBox::from_id_salt("ibl_preset")
        .selected_text(preset.name())
        .show_ui(ui, |ui| {
            for p in [IblPreset::Neutral, IblPreset::Studio, IblPreset::Warm] {
                if ui.selectable_value(&mut preset, p, p.name()).clicked() {
                    push(EditorCommand::SetIblPreset(p));
                }
            }
        });
    let pal = ui_ctx.ui_theme.palette();
    if crate::ui::icons::icon_button(
        ui,
        Icon::Open,
        t(loc, "render.ibl.load_short"),
        false,
        PANEL_BTN,
        &pal,
    )
    .clicked()
    {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(t(loc, "filter.hdr"), &["hdr", "exr", "png", "jpg", "jpeg"])
            .pick_file()
        {
            push(EditorCommand::LoadIblHdr(path));
        }
    }

    let mut walk = ui_ctx.walk_mode;
    check_cmd(ui, loc, &mut walk, "tools.walk", EditorCommand::SetWalkMode, push);
    let mut gpu_cull = ui_ctx.render_features.gpu_cull;
    check_cmd(ui, loc, &mut gpu_cull, "feat.gpu_cull", EditorCommand::SetGpuCulling, push);
    let mut parallel = ui_ctx.render_features.parallel_traversal;
    check_cmd(
        ui,
        loc,
        &mut parallel,
        "feat.parallel",
        EditorCommand::SetParallelTraversal,
        push,
    );
    let mut scale = ui_ctx.interaction_render_scale;
    if ui
        .add(egui::Slider::new(&mut scale, 0.25_f32..=1.0).text(t(loc, "render.interaction_scale")))
        .changed()
    {
        push(EditorCommand::SetInteractionRenderScale(scale));
    }
    ui.label(t(loc, "csm.label"));
    for res in [1024_u32, 2048, 4096] {
        let on = ui_ctx.csm_resolution == res;
        if ui
            .selectable_label(on, tf(loc, "render.shadow_map", res))
            .clicked()
        {
            push(EditorCommand::SetCsmShadow {
                resolution: res,
                cascade_count: ui_ctx.csm_cascades,
            });
        }
    }
    for n in [1_u32, 2, 3, 4] {
        let on = ui_ctx.csm_cascades == n;
        if ui
            .selectable_label(on, tf(loc, "render.cascades", n))
            .clicked()
        {
            push(EditorCommand::SetCsmShadow {
                resolution: ui_ctx.csm_resolution,
                cascade_count: n,
            });
        }
    }

    ui.label(t(loc, "layout.label"));
    let mut lm = ui_ctx.layout_mode;
    egui::ComboBox::from_id_salt("layout_mode")
        .selected_text(crate::ui::menus::layout_mode_label(loc, lm))
        .show_ui(ui, |ui| {
            for m in [
                LayoutMode::Single,
                LayoutMode::Quad,
                LayoutMode::LeftRight,
                LayoutMode::TopBottom,
            ] {
                if ui
                    .selectable_value(&mut lm, m, crate::ui::menus::layout_mode_label(loc, m))
                    .clicked()
                {
                    push(EditorCommand::SetViewportLayoutMode(m));
                }
            }
        });

    ui.label(t(loc, "aq.label"));
    let mut aq = ui_ctx.adaptive_quality_mode;
    egui::ComboBox::from_id_salt("adaptive_q")
        .selected_text(crate::ui::menus::aq_mode_label(loc, aq))
        .show_ui(ui, |ui| {
            for m in [
                AdaptiveQualityMode::Off,
                AdaptiveQualityMode::On,
                AdaptiveQualityMode::AutoIdleLock,
            ] {
                if ui
                    .selectable_value(&mut aq, m, crate::ui::menus::aq_mode_label(loc, m))
                    .clicked()
                {
                    push(EditorCommand::SetAdaptiveQualityMode(m));
                }
            }
        });

    ui.label(t(loc, "tier.label"));
    let mut tier = ui_ctx.cad_display_tier;
    egui::ComboBox::from_id_salt("cad_tier")
        .selected_text(crate::ui::menus::cad_tier_label(loc, tier))
        .show_ui(ui, |ui| {
            for t in [
                CadDisplayTier::DesignCreation,
                CadDisplayTier::Visualization,
                CadDisplayTier::IndustrialDisplay,
                CadDisplayTier::ProductRendering,
            ] {
                if ui
                    .selectable_value(&mut tier, t, crate::ui::menus::cad_tier_label(loc, t))
                    .clicked()
                {
                    push(EditorCommand::SetCadDisplayTier(t));
                }
            }
        });

    ui.label(t(loc, "preset.label"));
    use rc3d_engine_api::camera::ViewPreset;
    let mut vp = ViewPreset::Iso;
    egui::ComboBox::from_id_salt("view_preset")
        .selected_text(crate::ui::menus::view_preset_label(loc, vp))
        .show_ui(ui, |ui| {
            for p in [
                ViewPreset::Top,
                ViewPreset::Bottom,
                ViewPreset::Front,
                ViewPreset::Back,
                ViewPreset::Right,
                ViewPreset::Left,
                ViewPreset::Iso,
            ] {
                if ui
                    .selectable_value(&mut vp, p, crate::ui::menus::view_preset_label(loc, p))
                    .clicked()
                {
                    push(EditorCommand::SetViewPreset(p));
                }
            }
        });

    let mut ow = ui_ctx.outline_width;
    if ui
        .add(egui::Slider::new(&mut ow, 1.0..=4.0).text(t(loc, "edge.outline")))
        .changed()
    {
        push(EditorCommand::SetOutlineWidth(ow));
    }
    let mut oc = ui_ctx.outline_color;
    if ui.color_edit_button_rgba_unmultiplied(&mut oc).changed() {
        push(EditorCommand::SetOutlineColor(oc));
    }

    ui.separator();
    ui.label(t(loc, "render.features"));
    crate::ui::menus::view_render_features_menu(ui, ui_ctx, push);
}
