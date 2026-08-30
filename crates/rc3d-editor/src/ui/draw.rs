use std::collections::{HashSet, VecDeque};

use rc3d_core::NodeId;
use rc3d_render::ibl::IblPreset;
use rc3d_render::renderer::CadDisplayTier;
use rc3d_render::viewport::LayoutMode;
use rc3d_scene::node_data::MeasurementType;
use rc3d_scene::{NodeData, SceneGraph};

use crate::commands::{AdaptiveQualityMode, EditorCommand};
use crate::ui::i18n::{t, tf};
use crate::ui::types::{EditorChromeState, EditorDisplayMode, EditorUiContext};

pub(super) fn build_ui(
    ctx: &mut egui::Ui,
    graph: &SceneGraph,
    ui_ctx: &EditorUiContext,
    context_menu_pos: &mut Option<egui::Pos2>,
    show_console: &mut bool,
    console_entries: &[String],
    chrome: &mut EditorChromeState,
    q: &mut VecDeque<EditorCommand>,
) {
    let mut push = |c: EditorCommand| q.push_back(c);
    let loc = ui_ctx.ui_locale;

    crate::ui::caption::draw_caption(ctx, chrome, ui_ctx);

    egui::Panel::top("rc3d_menu").show(ctx, |ui| {
        super::menus::menu_bar(ui, ui_ctx, chrome, &mut push);
    });

    egui::Panel::top("rc3d_toolbar")
        .exact_size(32.0_f32)
        .show(ctx, |ui| {
            super::icons::draw_toolbar(ui, ui_ctx, &mut push);
        });

    egui::Panel::bottom("rc3d_status").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let doc = crate::document::document_display_name(ui_ctx.document_path.as_deref());
            let dirty = if ui_ctx.document_dirty { "*" } else { "" };
            ui.label(format!("{doc}{dirty}"));
            ui.separator();
            ui.label(format!(
                "{}: {:.1}  {}: {:.2} ms  {}: {}  {}: {}  {}: {}  {}: {}  {}: {}  {}: {} ({})  {}: {}",
                t(loc, "status.fps"),
                ui_ctx.smoothed_fps,
                t(loc, "status.frame"),
                ui_ctx.frame_time_ms,
                t(loc, "status.mode"),
                ui_ctx.display_mode_label,
                t(loc, "status.ibl"),
                ui_ctx.ibl_label,
                t(loc, "status.layout"),
                ui_ctx.layout_mode_label,
                t(loc, "status.vp"),
                ui_ctx.active_viewport_label,
                t(loc, "status.sel"),
                ui_ctx.selected_count,
                t(loc, "status.aq"),
                super::menus::aq_mode_label(loc, ui_ctx.adaptive_quality_mode),
                ui_ctx.adaptive_quality_name,
                t(loc, "status.tier"),
                super::menus::cad_tier_label(loc, ui_ctx.cad_display_tier),
            ));
        });
    });

    egui::Panel::left("hierarchy")
        .default_size(240.0)
        .show(ctx, |ui| {
            super::hierarchy::draw_hierarchy(ui, graph, ui_ctx, &mut push);
        });

    egui::Panel::right("inspector")
        .default_size(280.0)
        .show(ctx, |ui| {
            ui.heading(t(loc, "panel.inspector"));
            ui.separator();
            if let Some(id) = ui_ctx.selected.iter().next().copied() {
                if let Some(entry) = graph.get(id) {
                    ui.label(format!("{} {:?}", t(loc, "panel.node"), id));
                    inspector(ui, id, &entry.data, &ui_ctx.hidden_nodes, graph, loc, &mut push);
                }
            } else {
                ui.label(t(loc, "panel.no_sel"));
            }
            ui.separator();
            ui.collapsing(t(loc, "panel.render"), |ui| {
                render_panel(ui, ui_ctx, &mut push);
            });
            if let Some(d) = ui_ctx.diagnostics.as_ref() {
                ui.collapsing(t(loc, "panel.diagnostics"), |ui| {
                    ui.monospace(format!("frame {}", d.frame_index));
                    for (n, t) in &d.gpu_pass_timings_us {
                        ui.monospace(format!("{n}: {t:.1} us"));
                    }
                });
            }
        });

    if chrome.document_open {
        egui::Panel::bottom("document")
            .resizable(true)
            .default_size(180.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(t(loc, "panel.document"));
                    ui.selectable_value(&mut chrome.document_html, false, t(loc, "panel.markdown"));
                    ui.selectable_value(&mut chrome.document_html, true, t(loc, "panel.html"));
                });
                ui.separator();
                let inner = ui.available_rect_before_wrap();
                chrome.document_rect_points =
                    Some([inner.min.x, inner.min.y, inner.width(), inner.height()]);
                if chrome.document_html {
                    ui.label(t(loc, "panel.html_hint"));
                } else {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        markdown_ui(ui, STUDIO_HELP_MD);
                    });
                }
            });
    }

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            let inner = ui.available_rect_before_wrap();
            chrome.scene_rect_points =
                Some([inner.min.x, inner.min.y, inner.width(), inner.height()]);
            crate::ui::nav_cube::draw_nav_cube(ui, inner, ui_ctx, chrome, &mut push);
            // Right-click context menu detection
            if ui.input(|i| i.pointer.button_clicked(egui::PointerButton::Secondary)) {
                if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                    *context_menu_pos = Some(pos);
                }
            }
        });

    if chrome.caption.close_prompt {
        egui::Window::new(t(loc, "dialog.unsaved"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0_f32, 0.0_f32))
            .show(ctx, |ui| {
                ui.label(t(loc, "dialog.unsaved_body"));
                ui.add_space(8.0_f32);
                ui.horizontal(|ui| {
                    if ui.button(t(loc, "dialog.save")).clicked() {
                        if ui_ctx.document_path.is_some() {
                            push(EditorCommand::SaveScene);
                            chrome.caption.close_after_save = true;
                            chrome.caption.close_prompt = false;
                        } else if let Some(path) = crate::document::pick_save_scene() {
                            push(EditorCommand::SaveSceneAs(path));
                            chrome.caption.close_after_save = true;
                            chrome.caption.close_prompt = false;
                        }
                    }
                    if ui.button(t(loc, "dialog.dont_save")).clicked() {
                        chrome.caption.close_prompt = false;
                        chrome.caption.action = Some(crate::ui::types::CaptionAction::Close);
                    }
                    if ui.button(t(loc, "dialog.cancel")).clicked() {
                        chrome.caption.close_prompt = false;
                    }
                });
            });
    }

    // Right-click context menu window
    if let Some(pos) = context_menu_pos {
        egui::Window::new("##context_menu")
            .fixed_pos(*pos)
            .resizable(false)
            .title_bar(false)
            .auto_sized()
            .show(ctx, |ui| {
                ui.set_min_width(180.0);
                ui.menu_button(t(loc, "menu.add"), |ui| {
                    super::menus::create_node_menu(ui, None, loc, &mut push);
                });
                ui.menu_button(t(loc, "menu.view"), |ui| {
                    use rc3d_engine_api::camera::ViewPreset;
                    let presets = [
                        (ViewPreset::Top, "view.top"),
                        (ViewPreset::Front, "view.front"),
                        (ViewPreset::Right, "view.right"),
                        (ViewPreset::Iso, "view.iso"),
                    ];
                    for (preset, key) in presets {
                        if ui.button(t(loc, key)).clicked() {
                            push(EditorCommand::SetViewPreset(preset));
                            *context_menu_pos = None;
                        }
                    }
                });
                if ui.button(t(loc, "ctx.toggle_grid")).clicked() {
                    push(EditorCommand::SetGridEnabled(!ui_ctx.grid_enabled));
                    *context_menu_pos = None;
                }
                if ui.button(t(loc, "ctx.toggle_wire")).clicked() {
                    let mode = EditorDisplayMode::Wireframe;
                    push(EditorCommand::SetDisplayMode(mode));
                    *context_menu_pos = None;
                }
            });

        // Close on click outside
        if ctx.input(|i| {
            i.pointer.button_clicked(egui::PointerButton::Primary)
                || i.pointer.button_clicked(egui::PointerButton::Secondary)
        }) && context_menu_pos.is_some()
        {
            *context_menu_pos = None;
        }
    }

    // Console panel (toggled with backtick key)
    if *show_console {
        egui::Window::new(t(loc, "panel.console"))
            .collapsible(false)
            .resizable(true)
            .default_size([600.0, 200.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button(t(loc, "panel.clear")).clicked() {
                        // Handled by clearing the buffer
                    }
                    if ui.button(t(loc, "panel.close")).clicked() {
                        *show_console = false;
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for entry in console_entries.iter().rev().take(100) {
                            ui.label(entry);
                        }
                    });
            });
    }
}

fn inspector(
    ui: &mut egui::Ui,
    id: NodeId,
    data: &NodeData,
    hidden: &HashSet<NodeId>,
    graph: &SceneGraph,
    loc: crate::ui::i18n::UiLocale,
    push: &mut impl FnMut(EditorCommand),
) {
    ui.label(data.type_name());
    let descriptors = data.field_descriptors();
    if descriptors.is_empty() {
        ui.label(super::hierarchy::node_type_tag(data));
    } else {
        for desc in descriptors {
            field_row(ui, graph, id, desc.name, desc.field_index, push);
        }
    }

    let mut vis = match data {
        NodeData::Switch(sw) => sw.which_child >= 0,
        _ => !hidden.contains(&id),
    };
    if ui.checkbox(&mut vis, t(loc, "insp.visible")).changed() {
        push(EditorCommand::SetNodeVisibility(id, vis));
    }
}

fn field_row(
    ui: &mut egui::Ui,
    graph: &SceneGraph,
    id: NodeId,
    name: &str,
    field_index: u16,
    push: &mut impl FnMut(EditorCommand),
) {
    let Some(value) = rc3d_scene::read_node_field(graph, id, field_index) else {
        ui.label(format!("{name}: (unbound)"));
        return;
    };
    match value {
        rc3d_fields::FieldValue::Bool(mut b) => {
            if ui.checkbox(&mut b, name).changed() {
                push(EditorCommand::SetNodeField {
                    node: id,
                    field_index,
                    value: rc3d_fields::FieldValue::Bool(b),
                });
            }
        }
        rc3d_fields::FieldValue::Int32(mut i) => {
            ui.horizontal(|ui| {
                ui.label(name);
                if ui.add(egui::DragValue::new(&mut i)).changed() {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::Int32(i),
                    });
                }
            });
        }
        rc3d_fields::FieldValue::Float(mut f) => {
            ui.horizontal(|ui| {
                ui.label(name);
                if ui.add(egui::DragValue::new(&mut f).speed(0.02)).changed() {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::Float(f),
                    });
                }
            });
        }
        rc3d_fields::FieldValue::Vec3f(v) => {
            let mut arr = v.to_array();
            ui.horizontal(|ui| {
                ui.label(name);
                let mut changed = false;
                changed |= ui.add(egui::DragValue::new(&mut arr[0]).speed(0.02)).changed();
                changed |= ui.add(egui::DragValue::new(&mut arr[1]).speed(0.02)).changed();
                changed |= ui.add(egui::DragValue::new(&mut arr[2]).speed(0.02)).changed();
                if name.to_ascii_lowercase().contains("color") {
                    changed |= ui.color_edit_button_rgb(&mut arr).changed();
                }
                if changed {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::Vec3f(rc3d_core::math::Vec3::from_array(arr)),
                    });
                }
            });
        }
        rc3d_fields::FieldValue::Vec2f(v) => {
            let mut arr = v.to_array();
            ui.horizontal(|ui| {
                ui.label(name);
                let mut changed = false;
                changed |= ui.add(egui::DragValue::new(&mut arr[0]).speed(0.02)).changed();
                changed |= ui.add(egui::DragValue::new(&mut arr[1]).speed(0.02)).changed();
                if changed {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::Vec2f(rc3d_core::math::Vec2::from_array(arr)),
                    });
                }
            });
        }
        rc3d_fields::FieldValue::Vec4f(v) => {
            let mut arr = v.to_array();
            ui.horizontal(|ui| {
                ui.label(name);
                let mut changed = false;
                for c in &mut arr {
                    changed |= ui.add(egui::DragValue::new(c).speed(0.02)).changed();
                }
                if changed {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::Vec4f(rc3d_core::math::Vec4::from_array(arr)),
                    });
                }
            });
        }
        rc3d_fields::FieldValue::String(mut s) => {
            ui.horizontal(|ui| {
                ui.label(name);
                if ui.text_edit_singleline(&mut s).changed() {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::String(s),
                    });
                }
            });
        }
        other => {
            ui.label(format!("{name}: {other:?}"));
        }
    }
}

fn render_panel(ui: &mut egui::Ui, ui_ctx: &EditorUiContext, push: &mut impl FnMut(EditorCommand)) {
    let loc = ui_ctx.ui_locale;
    let mut hdr = ui_ctx.hdr_enabled;
    if ui.checkbox(&mut hdr, t(loc, "render.hdr")).changed() {
        push(EditorCommand::SetHdrPostProcessing(hdr));
    }
    let mut vsync = ui_ctx.vsync_enabled;
    if ui.checkbox(&mut vsync, t(loc, "render.vsync")).changed() {
        push(EditorCommand::SetVsyncEnabled(vsync));
    }
    let mut grid = ui_ctx.grid_enabled;
    if ui.checkbox(&mut grid, t(loc, "view.grid")).changed() {
        push(EditorCommand::SetGridEnabled(grid));
    }
    let mut hud = ui_ctx.hud_enabled;
    if ui.checkbox(&mut hud, t(loc, "view.hud")).changed() {
        push(EditorCommand::SetHudEnabled(hud));
    }
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
    let mut wboit = ui_ctx.wboit_enabled;
    if ui.checkbox(&mut wboit, t(loc, "render.wboit")).changed() {
        push(EditorCommand::SetWboit(wboit));
    }

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
    if ui.button(t(loc, "render.ibl.load_short")).clicked() {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(t(loc, "filter.hdr"), &["hdr", "exr", "png", "jpg", "jpeg"])
            .pick_file()
        {
            push(EditorCommand::LoadIblHdr(path));
        }
    }

    let mut walk = ui_ctx.walk_mode;
    if ui.checkbox(&mut walk, t(loc, "tools.walk")).changed() {
        push(EditorCommand::SetWalkMode(walk));
    }
    let mut gpu_cull = ui_ctx.render_features.gpu_cull;
    if ui.checkbox(&mut gpu_cull, t(loc, "feat.gpu_cull")).changed() {
        push(EditorCommand::SetGpuCulling(gpu_cull));
    }
    let mut parallel = ui_ctx.render_features.parallel_traversal;
    if ui.checkbox(&mut parallel, t(loc, "feat.parallel")).changed() {
        push(EditorCommand::SetParallelTraversal(parallel));
    }
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
        if ui.selectable_label(on, tf(loc, "render.shadow_map", res)).clicked() {
            push(EditorCommand::SetCsmShadow {
                resolution: res,
                cascade_count: ui_ctx.csm_cascades,
            });
        }
    }
    for n in [1_u32, 2, 3, 4] {
        let on = ui_ctx.csm_cascades == n;
        if ui.selectable_label(on, tf(loc, "render.cascades", n)).clicked() {
            push(EditorCommand::SetCsmShadow {
                resolution: ui_ctx.csm_resolution,
                cascade_count: n,
            });
        }
    }

    ui.label(t(loc, "layout.label"));
    let mut lm = ui_ctx.layout_mode;
    egui::ComboBox::from_id_salt("layout_mode")
        .selected_text(super::menus::layout_mode_label(loc, lm))
        .show_ui(ui, |ui| {
            for m in [
                LayoutMode::Single,
                LayoutMode::Quad,
                LayoutMode::LeftRight,
                LayoutMode::TopBottom,
            ] {
                if ui
                    .selectable_value(&mut lm, m, super::menus::layout_mode_label(loc, m))
                    .clicked()
                {
                    push(EditorCommand::SetViewportLayoutMode(m));
                }
            }
        });

    ui.label(t(loc, "aq.label"));
    let mut aq = ui_ctx.adaptive_quality_mode;
    egui::ComboBox::from_id_salt("adaptive_q")
        .selected_text(super::menus::aq_mode_label(loc, aq))
        .show_ui(ui, |ui| {
            for m in [
                AdaptiveQualityMode::Off,
                AdaptiveQualityMode::On,
                AdaptiveQualityMode::AutoIdleLock,
            ] {
                if ui
                    .selectable_value(&mut aq, m, super::menus::aq_mode_label(loc, m))
                    .clicked()
                {
                    push(EditorCommand::SetAdaptiveQualityMode(m));
                }
            }
        });

    ui.label(t(loc, "tier.label"));
    let mut tier = ui_ctx.cad_display_tier;
    egui::ComboBox::from_id_salt("cad_tier")
        .selected_text(super::menus::cad_tier_label(loc, tier))
        .show_ui(ui, |ui| {
            for t in [
                CadDisplayTier::DesignCreation,
                CadDisplayTier::Visualization,
                CadDisplayTier::IndustrialDisplay,
                CadDisplayTier::ProductRendering,
            ] {
                if ui
                    .selectable_value(&mut tier, t, super::menus::cad_tier_label(loc, t))
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
        .selected_text(super::menus::view_preset_label(loc, vp))
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
                    .selectable_value(&mut vp, p, super::menus::view_preset_label(loc, p))
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
    super::menus::view_render_features_menu(ui, ui_ctx, push);

    ui.separator();
    ui.label(t(loc, "tools.measurement"));
    if ui.selectable_label(false, t(loc, "tools.off")).clicked() {
        push(EditorCommand::SetMeasurementMode(None));
    }
    for (m, key) in [
        (MeasurementType::Distance, "tools.distance"),
        (MeasurementType::Angle, "tools.angle"),
        (MeasurementType::Radius, "tools.radius"),
        (MeasurementType::Diameter, "tools.diameter"),
    ] {
        if ui.selectable_label(false, t(loc, key)).clicked() {
            push(EditorCommand::SetMeasurementMode(Some(m)));
        }
    }
}

const STUDIO_HELP_MD: &str = r#"# rustcoin3d Studio

## Viewport
- Middle drag: orbit
- Right drag: pan
- Wheel: zoom
- Left click: pick
- Ctrl+drag: box select
- Alt+drag: lasso select
- T / R / G: translate / rotate / scale
- P: section-plane handles
- F: fit selection

## Display
- W / S / E / H / L: wireframe / shaded / shaded+edges / hidden line / flat+edges
- Display menu: Flat, fill/edge on selection, visual styles, X-ray, ghost, edge colors
- View menu: camera presets, layout, Grid, HUD, document panel
- Render menu: HDR, IBL + custom HDR, quality, TAA / SSR / fog / FXAA, GPU cull, CSM
- Tools: measurement types, markup, walk/FPS, Fit all
- File: screenshot PNG, hidden-line SVG, quad-pack PNG

## Document
- Markdown tab renders this panel in egui
- HTML tab overlays a native WebView on this rectangle
- File: New / Open JSON / Import mesh / Save / Save As
- Ctrl+N New, Ctrl+O Open, Ctrl+S Save, Ctrl+Shift+S Save As
- Caption shows Untitled or the file name; * means unsaved

## Window
- Drag the caption bar to move; double-click to maximize
- Caption buttons: minimize / maximize / close
"#;

fn markdown_ui(ui: &mut egui::Ui, src: &str) {
    for line in src.lines() {
        if let Some(rest) = line.strip_prefix("### ") {
            ui.heading(rest);
        } else if let Some(rest) = line.strip_prefix("## ") {
            ui.heading(rest);
        } else if let Some(rest) = line.strip_prefix("# ") {
            ui.heading(rest);
        } else if let Some(rest) = line.strip_prefix("- ") {
            ui.label(format!("- {rest}"));
        } else if line.trim().is_empty() {
            ui.add_space(6.0);
        } else {
            ui.label(line);
        }
    }
}

