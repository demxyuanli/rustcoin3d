//! Top menu bar: File / Edit / View / Display / Render / Tools / Bookmarks / Create.

use rc3d_core::{EdgeStyle, FillStyle, NodeId};
use rc3d_gizmo::GizmoMode;
use rc3d_render::ibl::IblPreset;
use rc3d_render::renderer::CadDisplayTier;
use rc3d_render::viewport::LayoutMode;

use crate::commands::{AdaptiveQualityMode, EditorCommand};
use crate::ui::i18n::{t, tf, UiLocale};
use crate::ui::theme::UiTheme;
use crate::ui::types::{EditorChromeState, EditorDisplayMode, EditorUiContext, NodeDataType};

pub(super) fn menu_bar(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    egui::MenuBar::new().ui(ui, |ui| {
        file_menu(ui, ui_ctx, chrome, push);
        edit_menu(ui, loc, push);
        view_menu(ui, ui_ctx, chrome, push);
        display_menu(ui, ui_ctx, push);
        render_menu(ui, ui_ctx, push);
        tools_menu(ui, ui_ctx, push);
        bookmarks_menu(ui, ui_ctx, push);
        ui.menu_button(t(loc, "menu.create"), |ui| {
            create_node_menu(ui, None, loc, push);
        });
    });
}

fn file_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.file"), |ui| {
        if ui.button(t(loc, "file.new")).clicked() {
            push(EditorCommand::NewScene);
        }
        if ui.button(t(loc, "file.open")).clicked() {
            if let Some(path) = crate::document::pick_open_scene() {
                push(EditorCommand::OpenScene(path));
            }
        }
        if ui.button(t(loc, "file.import_mesh")).clicked() {
            if let Some(path) = crate::document::pick_import_mesh() {
                push(EditorCommand::ImportPath(path));
            }
        }
        ui.separator();
        if ui.button(t(loc, "file.save")).clicked() {
            if ui_ctx.document_path.is_some() {
                push(EditorCommand::SaveScene);
            } else if let Some(path) = crate::document::pick_save_scene() {
                push(EditorCommand::SaveSceneAs(path));
            }
        }
        if ui.button(t(loc, "file.save_as")).clicked() {
            if let Some(path) = crate::document::pick_save_scene() {
                push(EditorCommand::SaveSceneAs(path));
            }
        }
        ui.separator();
        if ui.button(t(loc, "file.export_scene")).clicked() {
            if let Some(path) = crate::document::pick_save_scene() {
                push(EditorCommand::ExportIvPath(path));
            }
        }
        if ui.button(t(loc, "file.export_diagnostics")).clicked() {
            if let Some(path) = rfd::FileDialog::new().save_file() {
                push(EditorCommand::ExportDiagnosticsJsonPath(path));
            }
        }
        if ui.button(t(loc, "file.export_screenshot")).clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter(t(loc, "filter.png"), &["png"])
                .save_file()
            {
                push(EditorCommand::ExportScreenshot(path));
            }
        }
        if ui.button(t(loc, "file.export_svg")).clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter(t(loc, "filter.svg"), &["svg"])
                .save_file()
            {
                push(EditorCommand::ExportHiddenLineSvg(path));
            }
        }
        if ui.button(t(loc, "file.export_quad")).clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter(t(loc, "filter.png"), &["png"])
                .save_file()
            {
                push(EditorCommand::ExportQuadPack(path));
            }
        }
        if chrome.caption.enabled {
            ui.separator();
            if ui.button(t(loc, "file.exit")).clicked() {
                if chrome.caption.dirty {
                    chrome.caption.close_prompt = true;
                } else {
                    chrome.caption.action = Some(crate::ui::types::CaptionAction::Close);
                }
            }
        }
    });
}

fn edit_menu(ui: &mut egui::Ui, loc: UiLocale, push: &mut impl FnMut(EditorCommand)) {
    ui.menu_button(t(loc, "menu.edit"), |ui| {
        if ui.button(t(loc, "edit.undo")).clicked() {
            push(EditorCommand::Undo);
        }
        if ui.button(t(loc, "edit.redo")).clicked() {
            push(EditorCommand::Redo);
        }
        ui.separator();
        if ui.button(t(loc, "edit.fit_sel")).clicked() {
            push(EditorCommand::FitSelection);
        }
        if ui.button(t(loc, "edit.fit_all")).clicked() {
            push(EditorCommand::FitAll);
        }
    });
}

fn view_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.view"), |ui| {
        use rc3d_engine_api::camera::ViewPreset;
        ui.label(t(loc, "view.camera"));
        for (preset, key) in [
            (ViewPreset::Top, "view.top"),
            (ViewPreset::Front, "view.front"),
            (ViewPreset::Right, "view.right"),
            (ViewPreset::Iso, "view.iso"),
            (ViewPreset::Bottom, "view.bottom"),
            (ViewPreset::Back, "view.back"),
            (ViewPreset::Left, "view.left"),
        ] {
            if ui.button(t(loc, key)).clicked() {
                push(EditorCommand::SetViewPreset(preset));
            }
        }
        ui.separator();
        ui.label(t(loc, "view.layout"));
        for (mode, key) in [
            (LayoutMode::Single, "layout.single"),
            (LayoutMode::Quad, "layout.quad"),
            (LayoutMode::LeftRight, "layout.leftright"),
            (LayoutMode::TopBottom, "layout.topbottom"),
        ] {
            let on = ui_ctx.layout_mode == mode;
            if ui.selectable_label(on, t(loc, key)).clicked() {
                push(EditorCommand::SetViewportLayoutMode(mode));
            }
        }
        if ui.button(t(loc, "view.next_vp")).clicked() {
            push(EditorCommand::CycleActiveViewport);
        }
        ui.separator();
        ui.menu_button(t(loc, "view.theme"), |ui| {
            for (theme, key) in [
                (UiTheme::Dark, "view.theme.dark"),
                (UiTheme::Light, "view.theme.light"),
            ] {
                let on = ui_ctx.ui_theme == theme;
                if ui.selectable_label(on, t(loc, key)).clicked() {
                    push(EditorCommand::SetUiTheme(theme));
                }
            }
        });
        ui.menu_button(t(loc, "view.language"), |ui| {
            for (locale, key) in [
                (UiLocale::En, "view.language.en"),
                (UiLocale::ZhHans, "view.language.zh"),
            ] {
                let on = ui_ctx.ui_locale == locale;
                if ui.selectable_label(on, t(loc, key)).clicked() {
                    push(EditorCommand::SetUiLocale(locale));
                }
            }
        });
        ui.separator();
        ui.label(t(loc, "view.overlays"));
        let mut grid = ui_ctx.grid_enabled;
        if ui.checkbox(&mut grid, t(loc, "view.grid")).changed() {
            push(EditorCommand::SetGridEnabled(grid));
        }
        let mut hud = ui_ctx.hud_enabled;
        if ui.checkbox(&mut hud, t(loc, "view.hud")).changed() {
            push(EditorCommand::SetHudEnabled(hud));
        }
        ui.checkbox(&mut chrome.document_open, t(loc, "view.document"));
        ui.separator();
        ui.label(t(loc, "view.background"));
        for (mode, key) in [
            (rc3d_render::background::BgMode::Solid, "view.bg.solid"),
            (rc3d_render::background::BgMode::VerticalGradient, "view.bg.vgrad"),
            (rc3d_render::background::BgMode::HorizontalGradient, "view.bg.hgrad"),
            (rc3d_render::background::BgMode::SkyGround, "view.bg.sky"),
            (rc3d_render::background::BgMode::Image, "view.bg.image"),
        ] {
            let on = ui_ctx.bg_mode == mode;
            if ui.selectable_label(on, t(loc, key)).clicked() {
                push(EditorCommand::SetBgMode(mode));
            }
        }
        color_cmd(ui, ui_ctx.bg_top, t(loc, "view.bg.top"), |c| {
            push(EditorCommand::SetBgTopColor(c));
        });
        color_cmd(ui, ui_ctx.bg_bot, t(loc, "view.bg.bot"), |c| {
            push(EditorCommand::SetBgBotColor(c));
        });
        if ui.button(t(loc, "view.bg.image_pick")).clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter(t(loc, "filter.image"), &["png", "jpg", "jpeg", "hdr", "exr"])
                .pick_file()
            {
                push(EditorCommand::SetBgImage(path));
            }
        }
    });
}

fn display_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.display"), |ui| {
        ui.menu_button(t(loc, "display.shading"), |ui| {
            for (mode, key) in [
                (EditorDisplayMode::Wireframe, "shade.wireframe"),
                (EditorDisplayMode::Shaded, "shade.shaded"),
                (EditorDisplayMode::ShadedWithEdges, "shade.shaded_edges"),
                (EditorDisplayMode::HiddenLine, "shade.hidden"),
                (EditorDisplayMode::Flat, "shade.flat"),
                (EditorDisplayMode::FlatWithEdge, "shade.flat_edges"),
            ] {
                if ui.button(t(loc, key)).clicked() {
                    push(EditorCommand::SetDisplayMode(mode));
                }
            }
        });
        ui.menu_button(t(loc, "display.fill"), |ui| {
            for (fill, key) in [
                (FillStyle::Shaded, "fill.shaded"),
                (FillStyle::Flat, "fill.flat"),
                (FillStyle::HiddenLine, "fill.hidden"),
                (FillStyle::None, "fill.none"),
            ] {
                if ui.button(t(loc, key)).clicked() {
                    push(EditorCommand::SetFillStyle(fill));
                }
            }
        });
        ui.menu_button(t(loc, "display.edges"), |ui| {
            for (edges, key) in [
                (EdgeStyle::None, "edge.none"),
                (EdgeStyle::Crease, "edge.crease"),
                (EdgeStyle::Silhouette, "edge.silhouette"),
                (EdgeStyle::Full, "edge.full"),
                (EdgeStyle::Perimeter, "edge.perimeter"),
                (EdgeStyle::Hard, "edge.hard"),
                (EdgeStyle::Adjacent, "edge.adjacent"),
            ] {
                if ui.button(t(loc, key)).clicked() {
                    push(EditorCommand::SetEdgeStyle(edges));
                }
            }
            ui.separator();
            color_cmd(ui, ui_ctx.feature_edge_color, t(loc, "edge.feature"), |c| {
                push(EditorCommand::SetFeatureEdgeColor(c));
            });
            color_cmd(ui, ui_ctx.wireframe_edge_color, t(loc, "edge.wireframe"), |c| {
                push(EditorCommand::SetWireframeEdgeColor(c));
            });
            color_cmd(ui, ui_ctx.hidden_edge_color, t(loc, "edge.hidden"), |c| {
                push(EditorCommand::SetHiddenEdgeColor(c));
            });
            let mut crease = ui_ctx.crease_angle;
            if ui
                .add(egui::Slider::new(&mut crease, 1.0_f32..=90.0_f32).text(t(loc, "edge.crease_deg")))
                .changed()
            {
                push(EditorCommand::SetCreaseAngle(crease));
            }
            let mut ss = ui_ctx.render_features.screen_space_edges;
            if ui.checkbox(&mut ss, t(loc, "edge.ss")).changed() {
                push(EditorCommand::SetRenderFeature {
                    feature_name: "screen_space_edges",
                    enabled: ss,
                });
            }
            let mut outline = ui_ctx.render_features.screen_space_selection_outline;
            if ui.checkbox(&mut outline, t(loc, "edge.ss_outline")).changed() {
                push(EditorCommand::SetRenderFeature {
                    feature_name: "screen_space_selection_outline",
                    enabled: outline,
                });
            }
            let mut thr = ui_ctx.ss_edge_threshold;
            if ui
                .add(egui::Slider::new(&mut thr, 0.001_f32..=0.08_f32).text(t(loc, "edge.ss_thr")))
                .changed()
            {
                push(EditorCommand::SetSsEdgeThreshold(thr));
            }
        });
        ui.menu_button(t(loc, "display.visual_style"), |ui| {
            for name in rc3d_core::VisualStyleLibrary::builtin().names() {
                if ui.button(name).clicked() {
                    push(EditorCommand::ApplyVisualStyle(name.to_string()));
                }
            }
        });
        ui.menu_button(t(loc, "display.visibility"), |ui| {
            let mut xray = ui_ctx.xray_mode;
            if ui.checkbox(&mut xray, t(loc, "vis.xray")).changed() {
                push(EditorCommand::SetXrayMode(xray));
            }
            let mut ghost = ui_ctx.ghost_unselected;
            if ui.checkbox(&mut ghost, t(loc, "vis.ghost")).changed() {
                push(EditorCommand::SetGhostUnselected(ghost));
            }
            let mut go = ui_ctx.ghost_opacity;
            if ui
                .add(egui::Slider::new(&mut go, 0.02_f32..=0.95_f32).text(t(loc, "vis.ghost_opacity")))
                .changed()
            {
                push(EditorCommand::SetGhostOpacity(go));
            }
        });
    });
}

fn color_cmd(ui: &mut egui::Ui, rgba: [f32; 4], label: &str, mut on_change: impl FnMut([f32; 4])) {
    let mut c = rgba;
    ui.horizontal(|ui| {
        ui.label(label);
        if ui.color_edit_button_rgba_unmultiplied(&mut c).changed() {
            on_change(c);
        }
    });
}

fn render_menu(
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
                if ui.selectable_label(on, tf(loc, "render.cascades", n)).clicked() {
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
                        ("feat.motion_blur", ui_ctx.render_features.motion_blur, "motion_blur"),
                        ("feat.ssr", ui_ctx.render_features.ssr, "ssr"),
                        ("feat.color_grading", ui_ctx.render_features.color_grading, "color_grading"),
                        ("feat.dof", ui_ctx.render_features.dof, "dof"),
                        ("feat.fog", ui_ctx.render_features.volumetric_fog, "volumetric_fog"),
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
                        ("feat.cluster", ui_ctx.render_features.cluster_lights, "cluster_lights"),
                        ("feat.omni", ui_ctx.render_features.omni_shadows, "omni_shadows"),
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
                        ("feat.parallel", ui_ctx.render_features.parallel_traversal, "parallel_traversal"),
                    ],
                );
            });
        });
        ui.menu_button(t(loc, "render.post"), |ui| {
            post_slider(ui, ui_ctx.post_vignette, t(loc, "post.vignette"), 0.0, 1.0, |v| {
                push(EditorCommand::SetPostEffects {
                    vignette: v,
                    chromatic: ui_ctx.post_chromatic,
                    bloom: ui_ctx.post_bloom,
                    grain: ui_ctx.post_grain,
                });
            });
            post_slider(ui, ui_ctx.post_chromatic, t(loc, "post.chromatic"), 0.0, 1.0, |v| {
                push(EditorCommand::SetPostEffects {
                    vignette: ui_ctx.post_vignette,
                    chromatic: v,
                    bloom: ui_ctx.post_bloom,
                    grain: ui_ctx.post_grain,
                });
            });
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
            post_slider(ui, ui_ctx.post_halftone, t(loc, "post.halftone"), 0.0, 1.0, |v| {
                push(EditorCommand::SetPostStylize {
                    halftone: v,
                    glitch: ui_ctx.post_glitch,
                });
            });
            post_slider(ui, ui_ctx.post_glitch, t(loc, "post.glitch"), 0.0, 1.0, |v| {
                push(EditorCommand::SetPostStylize {
                    halftone: ui_ctx.post_halftone,
                    glitch: v,
                });
            });
        });
    });
}

fn render_feature_group(
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

fn post_slider(
    ui: &mut egui::Ui,
    value: f32,
    label: &str,
    min: f32,
    max: f32,
    mut on_change: impl FnMut(f32),
) {
    let mut v = value;
    if ui.add(egui::Slider::new(&mut v, min..=max).text(label)).changed() {
        on_change(v);
    }
}

fn tools_menu(ui: &mut egui::Ui, ui_ctx: &EditorUiContext, push: &mut impl FnMut(EditorCommand)) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.tools"), |ui| {
        if ui.button(t(loc, "tools.measure")).clicked() {
            push(EditorCommand::ToggleMeasurement);
        }
        if ui.button(t(loc, "tools.section")).clicked() {
            push(EditorCommand::ToggleSectionEdit);
        }
        ui.separator();
        ui.label(t(loc, "tools.measurement"));
        if ui.button(t(loc, "tools.off")).clicked() {
            push(EditorCommand::SetMeasurementMode(None));
        }
        for (m, key) in [
            (rc3d_scene::node_data::MeasurementType::Distance, "tools.distance"),
            (rc3d_scene::node_data::MeasurementType::Angle, "tools.angle"),
            (rc3d_scene::node_data::MeasurementType::Radius, "tools.radius"),
            (rc3d_scene::node_data::MeasurementType::Diameter, "tools.diameter"),
        ] {
            if ui.button(t(loc, key)).clicked() {
                push(EditorCommand::SetMeasurementMode(Some(m)));
            }
        }
        ui.separator();
        ui.label(t(loc, "tools.markup"));
        for (tool, key) in [
            (rc3d_actions::MarkupTool::Select, "tools.select"),
            (rc3d_actions::MarkupTool::Line, "tools.line"),
            (rc3d_actions::MarkupTool::Rect, "tools.rect"),
            (rc3d_actions::MarkupTool::Circle, "tools.circle"),
            (rc3d_actions::MarkupTool::Freehand, "tools.freehand"),
        ] {
            if ui.button(t(loc, key)).clicked() {
                push(EditorCommand::SetMarkupTool(tool));
            }
        }
        ui.separator();
        let mut walk = ui_ctx.walk_mode;
        if ui.checkbox(&mut walk, t(loc, "tools.walk")).changed() {
            push(EditorCommand::SetWalkMode(walk));
        }
        if ui.button(t(loc, "edit.fit_all")).clicked() {
            push(EditorCommand::FitAll);
        }
        ui.separator();
        ui.label(t(loc, "tools.gizmo"));
        if ui.button(t(loc, "tools.move")).clicked() {
            push(EditorCommand::SetGizmoMode(GizmoMode::Translate));
        }
        if ui.button(t(loc, "tools.rotate")).clicked() {
            push(EditorCommand::SetGizmoMode(GizmoMode::Rotate));
        }
        if ui.button(t(loc, "tools.scale")).clicked() {
            push(EditorCommand::SetGizmoMode(GizmoMode::Scale));
        }
    });
}

fn bookmarks_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.menu_button(t(loc, "menu.bookmarks"), |ui| {
        ui.label(t(loc, "bm.hint"));
        ui.separator();
        for i in 0..9 {
            let has = ui_ctx.bookmarks[i].0;
            let label = ui_ctx.bookmarks[i].1;
            ui.horizontal(|ui| {
                if ui.button(format!("{} {i}", t(loc, "bm.save"))).clicked() {
                    push(EditorCommand::SaveBookmark(i));
                }
                if ui
                    .button(format!("{} {i}  {label}", t(loc, "bm.recall")))
                    .clicked()
                {
                    push(EditorCommand::RecallBookmark(i));
                }
                if has {
                    ui.label("●");
                }
            });
        }
    });
}

pub(super) fn create_node_menu<F: FnMut(EditorCommand)>(
    ui: &mut egui::Ui,
    parent: Option<NodeId>,
    loc: UiLocale,
    push: &mut F,
) {
    let add = |ui: &mut egui::Ui, key: &'static str, ty: NodeDataType, push: &mut F| {
        if ui.button(t(loc, key)).clicked() {
            push(EditorCommand::CreateNode {
                node_type: ty,
                parent,
            });
        }
    };
    add(ui, "create.cube", NodeDataType::Cube, push);
    add(ui, "create.sphere", NodeDataType::Sphere, push);
    add(ui, "create.cylinder", NodeDataType::Cylinder, push);
    add(ui, "create.cone", NodeDataType::Cone, push);
    ui.separator();
    add(ui, "create.separator", NodeDataType::Separator, push);
    add(ui, "create.transform", NodeDataType::Transform, push);
    add(ui, "create.material", NodeDataType::Material, push);
    add(ui, "create.switch", NodeDataType::Switch, push);
    add(ui, "create.lod", NodeDataType::Lod, push);
    add(ui, "create.environment", NodeDataType::Environment, push);
    ui.separator();
    add(ui, "create.dir_light", NodeDataType::DirectionalLight, push);
    add(ui, "create.point_light", NodeDataType::PointLight, push);
    add(ui, "create.spot_light", NodeDataType::SpotLight, push);
    add(ui, "create.hemi_light", NodeDataType::HemisphereLight, push);
    add(ui, "create.area_light", NodeDataType::AreaLight, push);
    add(ui, "create.light_probe", NodeDataType::LightProbe, push);
    ui.separator();
    add(ui, "create.persp_cam", NodeDataType::PerspectiveCamera, push);
    add(ui, "create.ortho_cam", NodeDataType::OrthographicCamera, push);
    add(ui, "create.stereo_cam", NodeDataType::StereoCamera, push);
    ui.separator();
    add(ui, "create.text2", NodeDataType::Text2, push);
    add(ui, "create.text3", NodeDataType::Text3, push);
    add(ui, "create.font", NodeDataType::Font, push);
    add(ui, "create.billboard", NodeDataType::Billboard, push);
    add(ui, "create.sprite", NodeDataType::Sprite, push);
    ui.separator();
    add(ui, "create.section", NodeDataType::SectionPlane, push);
    add(ui, "create.annotation", NodeDataType::AnnotationSet, push);
    add(ui, "create.measurement", NodeDataType::Measurement, push);
    add(ui, "create.markup", NodeDataType::Markup, push);
    add(ui, "create.particles", NodeDataType::Particles, push);
    add(ui, "create.instanced", NodeDataType::InstancedMesh, push);
    add(ui, "create.batched", NodeDataType::BatchedMesh, push);
}

pub(super) fn view_render_features_menu(
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

pub(super) fn cad_tier_label(loc: UiLocale, t: CadDisplayTier) -> &'static str {
    match t {
        CadDisplayTier::DesignCreation => crate::ui::i18n::t(loc, "tier.design"),
        CadDisplayTier::Visualization => crate::ui::i18n::t(loc, "tier.viz"),
        CadDisplayTier::IndustrialDisplay => crate::ui::i18n::t(loc, "tier.industrial"),
        CadDisplayTier::ProductRendering => crate::ui::i18n::t(loc, "tier.product"),
    }
}

pub(super) fn layout_mode_label(loc: UiLocale, mode: LayoutMode) -> &'static str {
    match mode {
        LayoutMode::Single => t(loc, "layout.single"),
        LayoutMode::Quad => t(loc, "layout.quad"),
        LayoutMode::LeftRight => t(loc, "layout.leftright"),
        LayoutMode::TopBottom => t(loc, "layout.topbottom"),
    }
}

pub(super) fn aq_mode_label(loc: UiLocale, mode: AdaptiveQualityMode) -> &'static str {
    match mode {
        AdaptiveQualityMode::Off => t(loc, "render.aq.off"),
        AdaptiveQualityMode::On => t(loc, "render.aq.on"),
        AdaptiveQualityMode::AutoIdleLock => t(loc, "render.aq.idle"),
    }
}

pub(super) fn view_preset_label(
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
