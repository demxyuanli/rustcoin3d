//! Render menu: HDR/vsync/WBOIT, IBL, quality, shadows, features, post.

use rc3d_render::ibl::IblPreset;
use rc3d_render::renderer::CadDisplayTier;

use crate::commands::{AdaptiveQualityMode, EditorCommand};
use crate::ui::i18n::{t, tf, UiLocale};
use crate::ui::types::EditorUiContext;

use super::{post_slider, labels::cad_tier_label};

pub(super) fn render_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.render"), |ui| {
        let mut hdr = ui_ctx.hdr_enabled;
        if ui.checkbox(&mut hdr, t(loc, "render.hdr")).changed() {
            push(EditorCommand::SetHdrPostProcessing(hdr));
        }
        let mut vsync = ui_ctx.vsync_enabled;
        if ui.checkbox(&mut vsync, t(loc, "render.vsync")).changed() {
            push(EditorCommand::SetVsyncEnabled(vsync));
        }
        let mut wboit = ui_ctx.wboit_enabled;
        if ui.checkbox(&mut wboit, t(loc, "render.wboit")).changed() {
            push(EditorCommand::SetWboit(wboit));
        }
        ui.menu_button(t(loc, "render.ibl"), |ui| {
            if ui.button(t(loc, "render.ibl.cycle")).clicked() {
                push(EditorCommand::CycleIbl);
            }
            for p in [IblPreset::Neutral, IblPreset::Studio, IblPreset::Warm] {
                let on = ui_ctx.ibl_preset == p;
                if ui.selectable_label(on, p.name()).clicked() {
                    push(EditorCommand::SetIblPreset(p));
                }
            }
            if ui.button(t(loc, "render.ibl.load")).clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter(t(loc, "filter.hdr"), &["hdr", "exr", "png", "jpg", "jpeg"])
                    .pick_file()
                {
                    push(EditorCommand::LoadIblHdr(path));
                }
            }
        });
        ui.menu_button(t(loc, "render.quality"), |ui| {
            for (mode, key) in [
                (AdaptiveQualityMode::Off, "render.aq.off"),
                (AdaptiveQualityMode::On, "render.aq.on"),
                (AdaptiveQualityMode::AutoIdleLock, "render.aq.idle"),
            ] {
                let on = ui_ctx.adaptive_quality_mode == mode;
                if ui.selectable_label(on, t(loc, key)).clicked() {
                    push(EditorCommand::SetAdaptiveQualityMode(mode));
                }
            }
            ui.separator();
            for tier in [
                CadDisplayTier::DesignCreation,
                CadDisplayTier::Visualization,
                CadDisplayTier::IndustrialDisplay,
                CadDisplayTier::ProductRendering,
            ] {
                let on = ui_ctx.cad_display_tier == tier;
                if ui.selectable_label(on, cad_tier_label(loc, tier)).clicked() {
                    push(EditorCommand::SetCadDisplayTier(tier));
                }
            }
            ui.separator();
            post_slider(
                ui,
                ui_ctx.interaction_render_scale,
                t(loc, "render.interaction_scale"),
                0.25,
                1.0,
                |v| {
                    push(EditorCommand::SetInteractionRenderScale(v));
                },
            );
        });
        ui.menu_button(t(loc, "render.shadows"), |ui| {
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
            ui.separator();
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
        });
        ui.menu_button(t(loc, "render.features"), |ui| {
            ui.menu_button(t(loc, "render.effects"), |ui| {
                render_feature_group(
                    ui,
                    loc,
                    push,
                    &[
                        ("feat.taa", ui_ctx.render_features.taa, "taa"),
                        (
                            "feat.motion_blur",
                            ui_ctx.render_features.motion_blur,
                            "motion_blur",
                        ),
                        ("feat.ssr", ui_ctx.render_features.ssr, "ssr"),
                        (
                            "feat.color_grading",
                            ui_ctx.render_features.color_grading,
                            "color_grading",
                        ),
                        ("feat.dof", ui_ctx.render_features.dof, "dof"),
                        (
                            "feat.fog",
                            ui_ctx.render_features.volumetric_fog,
                            "volumetric_fog",
                        ),
                        ("feat.fxaa", ui_ctx.render_features.ldr_fxaa, "ldr_fxaa"),
                    ],
                );
            });
            ui.menu_button(t(loc, "render.lighting"), |ui| {
                render_feature_group(
                    ui,
                    loc,
                    push,
                    &[
                        (
                            "feat.cluster",
                            ui_ctx.render_features.cluster_lights,
                            "cluster_lights",
                        ),
                        (
                            "feat.omni",
                            ui_ctx.render_features.omni_shadows,
                            "omni_shadows",
                        ),
                    ],
                );
            });
            ui.menu_button(t(loc, "render.perf"), |ui| {
                render_feature_group(
                    ui,
                    loc,
                    push,
                    &[
                        ("feat.gpu_cull", ui_ctx.render_features.gpu_cull, "gpu_cull"),
                        (
                            "feat.parallel",
                            ui_ctx.render_features.parallel_traversal,
                            "parallel_traversal",
                        ),
                    ],
                );
            });
        });
        ui.menu_button(t(loc, "render.post"), |ui| {
            post_slider(
                ui,
                ui_ctx.post_vignette,
                t(loc, "post.vignette"),
                0.0,
                1.0,
                |v| {
                    push(EditorCommand::SetPostEffects {
                        vignette: v,
                        chromatic: ui_ctx.post_chromatic,
                        bloom: ui_ctx.post_bloom,
                        grain: ui_ctx.post_grain,
                    });
                },
            );
            post_slider(
                ui,
                ui_ctx.post_chromatic,
                t(loc, "post.chromatic"),
                0.0,
                1.0,
                |v| {
                    push(EditorCommand::SetPostEffects {
                        vignette: ui_ctx.post_vignette,
                        chromatic: v,
                        bloom: ui_ctx.post_bloom,
                        grain: ui_ctx.post_grain,
                    });
                },
            );
            post_slider(ui, ui_ctx.post_bloom, t(loc, "post.bloom"), 0.0, 2.0, |v| {
                push(EditorCommand::SetPostEffects {
                    vignette: ui_ctx.post_vignette,
                    chromatic: ui_ctx.post_chromatic,
                    bloom: v,
                    grain: ui_ctx.post_grain,
                });
            });
            post_slider(ui, ui_ctx.post_grain, t(loc, "post.grain"), 0.0, 1.0, |v| {
                push(EditorCommand::SetPostEffects {
                    vignette: ui_ctx.post_vignette,
                    chromatic: ui_ctx.post_chromatic,
                    bloom: ui_ctx.post_bloom,
                    grain: v,
                });
            });
            post_slider(
                ui,
                ui_ctx.post_halftone,
                t(loc, "post.halftone"),
                0.0,
                1.0,
                |v| {
                    push(EditorCommand::SetPostStylize {
                        halftone: v,
                        glitch: ui_ctx.post_glitch,
                    });
                },
            );
            post_slider(
                ui,
                ui_ctx.post_glitch,
                t(loc, "post.glitch"),
                0.0,
                1.0,
                |v| {
                    push(EditorCommand::SetPostStylize {
                        halftone: ui_ctx.post_halftone,
                        glitch: v,
                    });
                },
            );
        });
    });
}

pub(super) fn render_feature_group(
    ui: &mut egui::Ui,
    loc: UiLocale,
    push: &mut impl FnMut(EditorCommand),
    items: &[(&'static str, bool, &'static str)],
) {
    for &(key, field, name) in items {
        let mut v = field;
        if ui.checkbox(&mut v, t(loc, key)).changed() {
            push(EditorCommand::SetRenderFeature {
                feature_name: name,
                enabled: v,
            });
        }
    }
}
