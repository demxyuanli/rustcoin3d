use std::collections::{HashSet, VecDeque};

use rc3d_core::math::Quat;
use rc3d_core::NodeId;
use rc3d_gizmo::GizmoMode;
use rc3d_render::ibl::IblPreset;
use rc3d_render::viewport::LayoutMode;
use rc3d_scene::node_data::MeasurementType;
use rc3d_scene::{NodeData, SceneGraph};

use crate::adaptive_quality::AdaptiveQualityMode;
use crate::camera_controller::ViewPreset;

use super::commands::EditorCommand;
use super::types::{EditorDisplayMode, EditorUiContext, NodeDataType};

pub(super) fn build_ui(
    ctx: &egui::Context,
    graph: &SceneGraph,
    ui_ctx: &EditorUiContext,
    context_menu_pos: &mut Option<egui::Pos2>,
    show_console: &mut bool,
    console_entries: &[String],
    q: &mut VecDeque<EditorCommand>,
) {
    let mut push = |c: EditorCommand| q.push_back(c);

    // ── Menu bar ──
    egui::TopBottomPanel::top("rc3d_menu").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open…").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_file() {
                        push(EditorCommand::ImportPath(path));
                    }
                }
                if ui.button("Export IV…").clicked() {
                    if let Some(path) = rfd::FileDialog::new().save_file() {
                        push(EditorCommand::ExportIvPath(path));
                    }
                }
                if ui.button("Export diagnostics JSON…").clicked() {
                    if let Some(path) = rfd::FileDialog::new().save_file() {
                        push(EditorCommand::ExportDiagnosticsJsonPath(path));
                    }
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui.button("Undo  (Ctrl+Z)").clicked() {
                    push(EditorCommand::Undo);
                }
                if ui.button("Redo  (Ctrl+Y)").clicked() {
                    push(EditorCommand::Redo);
                }
                ui.separator();
                if ui.button("Fit selection  (F)").clicked() {
                    push(EditorCommand::FitSelection);
                }
            });
            ui.menu_button("View", |ui| {
                ui.label("Display mode");
                if ui.button("Wireframe  (W)").clicked() {
                    push(EditorCommand::SetDisplayMode(EditorDisplayMode::Wireframe));
                }
                if ui.button("Shaded  (S)").clicked() {
                    push(EditorCommand::SetDisplayMode(EditorDisplayMode::Shaded));
                }
                if ui.button("Shaded + edges  (E)").clicked() {
                    push(EditorCommand::SetDisplayMode(
                        EditorDisplayMode::ShadedWithEdges,
                    ));
                }
                if ui.button("Hidden line  (H)").clicked() {
                    push(EditorCommand::SetDisplayMode(EditorDisplayMode::HiddenLine));
                }
                ui.separator();
                ui.label("Display toggles");
                let mut grid = ui_ctx.grid_enabled;
                if ui.checkbox(&mut grid, "Grid  (N)").changed() {
                    push(EditorCommand::SetGridEnabled(grid));
                }
                let mut hud = ui_ctx.hud_enabled;
                if ui.checkbox(&mut hud, "HUD").changed() {
                    push(EditorCommand::SetHudEnabled(hud));
                }
                let mut xray = ui_ctx.xray_mode;
                if ui.checkbox(&mut xray, "X-ray").changed() {
                    push(EditorCommand::SetXrayMode(xray));
                }
                ui.separator();
                ui.label("Render features");
                view_render_features_menu(ui, ui_ctx, &mut push);
                ui.separator();
                if ui.button("Cycle IBL  (I)").clicked() {
                    push(EditorCommand::CycleIbl);
                }
                if ui.button("Cycle viewport layout  (C)").clicked() {
                    push(EditorCommand::CycleViewportLayout);
                }
                if ui.button("Cycle active viewport  (Tab)").clicked() {
                    push(EditorCommand::CycleActiveViewport);
                }
            });
            ui.menu_button("Tools", |ui| {
                if ui.button("Toggle measurement  (M)").clicked() {
                    push(EditorCommand::ToggleMeasurement);
                }
                if ui.button("Toggle section edit  (P)").clicked() {
                    push(EditorCommand::ToggleSectionEdit);
                }
                ui.separator();
                ui.label("Gizmo");
                if ui.button("Move  (T)").clicked() {
                    push(EditorCommand::SetGizmoMode(GizmoMode::Translate));
                }
                if ui.button("Rotate  (R)").clicked() {
                    push(EditorCommand::SetGizmoMode(GizmoMode::Rotate));
                }
                if ui.button("Scale  (G)").clicked() {
                    push(EditorCommand::SetGizmoMode(GizmoMode::Scale));
                }
            });
            ui.menu_button("Bookmarks", |ui| {
                ui.label("Ctrl+Digit = save | Digit = recall");
                ui.separator();
                for i in 0..9 {
                    let has = ui_ctx.bookmarks[i].0;
                    let label = ui_ctx.bookmarks[i].1;
                    ui.horizontal(|ui| {
                        if ui.button(format!("Save {i}")).clicked() {
                            push(EditorCommand::SaveBookmark(i));
                        }
                        if ui.button(format!("Recall {i}  {label}")).clicked() {
                            push(EditorCommand::RecallBookmark(i));
                        }
                        if has {
                            ui.label("●");
                        }
                    });
                }
            });
            ui.menu_button("Create", |ui| {
                create_node_menu(ui, None, &mut push);
            });
        });
    });

    // ── Toolbar ──
    egui::TopBottomPanel::top("rc3d_toolbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Gizmo:");
            let gm = ui_ctx.gizmo_mode;
            let mut selected = gm == GizmoMode::Translate;
            if ui.selectable_label(selected, "↕ Move").clicked() {
                push(EditorCommand::SetGizmoMode(GizmoMode::Translate));
            }
            selected = gm == GizmoMode::Rotate;
            if ui.selectable_label(selected, "↻ Rotate").clicked() {
                push(EditorCommand::SetGizmoMode(GizmoMode::Rotate));
            }
            selected = gm == GizmoMode::Scale;
            if ui.selectable_label(selected, "↔ Scale").clicked() {
                push(EditorCommand::SetGizmoMode(GizmoMode::Scale));
            }

            ui.separator();

            ui.label("View:");
            let presets = [
                (ViewPreset::Top, "Top"),
                (ViewPreset::Front, "Front"),
                (ViewPreset::Right, "Right"),
                (ViewPreset::Iso, "Iso"),
                (ViewPreset::Bottom, "Bot"),
                (ViewPreset::Back, "Back"),
                (ViewPreset::Left, "Left"),
            ];
            for (preset, label) in presets {
                if ui.small_button(label).clicked() {
                    push(EditorCommand::SetViewPreset(preset));
                }
            }

            ui.separator();

            let mut grid = ui_ctx.grid_enabled;
            if ui.toggle_value(&mut grid, "Grid").changed() {
                push(EditorCommand::SetGridEnabled(grid));
            }
            let mut hud = ui_ctx.hud_enabled;
            if ui.toggle_value(&mut hud, "HUD").changed() {
                push(EditorCommand::SetHudEnabled(hud));
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!(
                    "FPS: {:.1}  {:.1} ms",
                    ui_ctx.smoothed_fps, ui_ctx.frame_time_ms
                ));
                ui.label(format!(
                    "Mode: {}  IBL: {}",
                    ui_ctx.display_mode_label, ui_ctx.ibl_label
                ));
            });
        });
    });

    egui::TopBottomPanel::bottom("rc3d_status").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label(format!(
                "FPS: {:.1}  Frame: {:.2} ms  Mode: {}  IBL: {}  Layout: {}  Active VP: {}  Sel: {}  AQ: {:?} ({})",
                ui_ctx.smoothed_fps,
                ui_ctx.frame_time_ms,
                ui_ctx.display_mode_label,
                ui_ctx.ibl_label,
                ui_ctx.layout_mode_label,
                ui_ctx.active_viewport_label,
                ui_ctx.selected_count,
                ui_ctx.adaptive_quality_mode,
                ui_ctx.adaptive_quality_name,
            ));
        });
    });

    egui::SidePanel::left("hierarchy")
        .default_width(220.0)
        .show(ctx, |ui| {
            ui.heading("Hierarchy");
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                for &root in graph.roots() {
                    hierarchy_node(ui, graph, root, &ui_ctx.selected, &mut push);
                }
            });
        });

    egui::SidePanel::right("inspector")
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.heading("Inspector");
            ui.separator();
            if let Some(id) = ui_ctx.selected.iter().next().copied() {
                if let Some(entry) = graph.get(id) {
                    ui.label(format!("Node {:?}", id));
                    inspector(ui, id, &entry.data, &ui_ctx.hidden_nodes, &mut push);
                }
            } else {
                ui.label("No selection");
            }
            ui.separator();
            ui.collapsing("Render", |ui| {
                render_panel(ui, ui_ctx, &mut push);
            });
            if let Some(d) = ui_ctx.diagnostics.as_ref() {
                ui.collapsing("Diagnostics", |ui| {
                    ui.monospace(format!("frame {}", d.frame_index));
                    for (n, t) in &d.gpu_pass_timings_us {
                        ui.monospace(format!("{n}: {t:.1} us"));
                    }
                });
            }
        });

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            // Right-click context menu detection
            if ui.input(|i| i.pointer.button_clicked(egui::PointerButton::Secondary)) {
                if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                    *context_menu_pos = Some(pos);
                }
            }
        });

    // Right-click context menu window
    if let Some(pos) = context_menu_pos {
        egui::Window::new("##context_menu")
            .fixed_pos(*pos)
            .resizable(false)
            .title_bar(false)
            .auto_sized()
            .show(ctx, |ui| {
                ui.set_min_width(180.0);
                ui.menu_button("Add", |ui| {
                    create_node_menu(ui, None, &mut push);
                });
                ui.menu_button("View", |ui| {
                    let presets = [
                        (ViewPreset::Top, "Top"),
                        (ViewPreset::Front, "Front"),
                        (ViewPreset::Right, "Right"),
                        (ViewPreset::Iso, "Iso"),
                    ];
                    for (preset, label) in presets {
                        if ui.button(label).clicked() {
                            push(EditorCommand::SetViewPreset(preset));
                            *context_menu_pos = None;
                        }
                    }
                });
                if ui.button("Toggle Grid").clicked() {
                    push(EditorCommand::SetGridEnabled(!ui_ctx.grid_enabled));
                    *context_menu_pos = None;
                }
                if ui.button("Toggle Wireframe").clicked() {
                    let mode = EditorDisplayMode::Wireframe;
                    push(EditorCommand::SetDisplayMode(mode));
                    *context_menu_pos = None;
                }
            });

        // Close on click outside
        if ctx.input(|i| {
            i.pointer.button_clicked(egui::PointerButton::Primary)
                || i.pointer.button_clicked(egui::PointerButton::Secondary)
        }) {
            if context_menu_pos.is_some() {
                *context_menu_pos = None;
            }
        }
    }

    // Console panel (toggled with ` backtick key)
    if *show_console {
        egui::Window::new("Console")
            .collapsible(false)
            .resizable(true)
            .default_size([600.0, 200.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Clear").clicked() {
                        // Handled by clearing the buffer — can't mutate console_entries here
                    }
                    if ui.button("Close").clicked() {
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

fn create_node_menu<F: FnMut(EditorCommand)>(
    ui: &mut egui::Ui,
    parent: Option<NodeId>,
    push: &mut F,
) {
    let add = |ui: &mut egui::Ui, label: &str, t: NodeDataType, push: &mut F| {
        if ui.button(label).clicked() {
            push(EditorCommand::CreateNode {
                node_type: t,
                parent,
            });
        }
    };
    add(ui, "Cube", NodeDataType::Cube, push);
    add(ui, "Sphere", NodeDataType::Sphere, push);
    add(ui, "Cylinder", NodeDataType::Cylinder, push);
    add(ui, "Cone", NodeDataType::Cone, push);
    add(ui, "Separator", NodeDataType::Separator, push);
    add(
        ui,
        "Directional light",
        NodeDataType::DirectionalLight,
        push,
    );
    add(ui, "Point light", NodeDataType::PointLight, push);
    add(ui, "Spot light", NodeDataType::SpotLight, push);
    add(
        ui,
        "Perspective camera",
        NodeDataType::PerspectiveCamera,
        push,
    );
    add(
        ui,
        "Orthographic camera",
        NodeDataType::OrthographicCamera,
        push,
    );
    add(ui, "Text2", NodeDataType::Text2, push);
    add(ui, "Text3", NodeDataType::Text3, push);
}

fn hierarchy_node(
    ui: &mut egui::Ui,
    graph: &SceneGraph,
    id: NodeId,
    selected: &HashSet<NodeId>,
    push: &mut impl FnMut(EditorCommand),
) {
    let Some(entry) = graph.get(id) else {
        return;
    };
    let label = entry
        .name
        .as_deref()
        .unwrap_or_else(|| node_type_tag(&entry.data));
    let is_sel = selected.contains(&id);
    ui.horizontal(|ui| {
        let resp = ui.selectable_label(is_sel, format!("{label} [{id:?}]"));
        if resp.clicked() {
            push(EditorCommand::SetSelection(Some(id)));
        }
        if ui.small_button("Del").clicked() {
            push(EditorCommand::DeleteNode(id));
        }
        if ui.small_button("Dup").clicked() {
            push(EditorCommand::DuplicateNode(id));
        }
        ui.menu_button("+", |ui| {
            create_node_menu(ui, Some(id), push);
        });
    });
    if let Some(children) = graph.children(id) {
        ui.indent(id, |ui| {
            for &c in children.iter() {
                hierarchy_node(ui, graph, c, selected, push);
            }
        });
    }
}

fn node_type_tag(data: &NodeData) -> &'static str {
    match data {
        NodeData::Separator(_) => "Separator",
        NodeData::Group(_)
        | NodeData::Environment(_)
        | NodeData::ShapeHints(_)
        | NodeData::Annotation(_)
        | NodeData::ResetTransform(_)
        | NodeData::Texture2Transform(_)
        | NodeData::MaterialBinding(_)
        | NodeData::IndexedLineSet(_)
        | NodeData::File(_)
        | NodeData::Decal(_)
        | NodeData::ExplodedView(_)
        | NodeData::ReflectionPlane(_) => "Group",
        NodeData::Billboard(_) => "Billboard",
        NodeData::Transform(_) => "Transform",
        NodeData::Material(_) => "Material",
        NodeData::Triangle(_) => "Triangle",
        NodeData::Cube(_) => "Cube",
        NodeData::Sphere(_) => "Sphere",
        NodeData::Cone(_) => "Cone",
        NodeData::Cylinder(_) => "Cylinder",
        NodeData::IndexedFaceSet(_) => "IFS",
        NodeData::SkinnedMesh(_) => "SkinnedMesh",
        NodeData::MorphTarget(_) => "Morph",
        NodeData::RayTracing(_) => "RayTracing",
        NodeData::Volume(_) => "Volume",
        NodeData::PointCloud(_) => "PtCloud",
        NodeData::PerspectiveCamera(_) => "PerspCam",
        NodeData::StereoCamera(_) => "StereoCam",
        NodeData::OrthographicCamera(_) => "OrthoCam",
        NodeData::DirectionalLight(_) => "DirLight",
        NodeData::PointLight(_) => "PointLight",
        NodeData::SpotLight(_) | NodeData::AreaLight(_) => "SpotLight",
        NodeData::HandlerNode(_) => "Handler",
        NodeData::EventCallback(_) => "EventCb",
        NodeData::PickStyle(_) => "PickStyle",
        NodeData::Lod(_) => "LOD",
        NodeData::Switch(_) => "Switch",
        NodeData::MultipleCopy(_) => "MultiCopy",
        NodeData::SectionPlane(_) => "Section",
        NodeData::Text2(_) => "Text2",
        NodeData::Text3(_) => "Text3",
        NodeData::Measurement(_) => "Measure",
        NodeData::Markup(_) => "Markup",
        NodeData::Coordinate3(_) => "Coord3",
        NodeData::TextureCoordinate2(_) => "TexCoord2",
        NodeData::Normal(_) => "Normal",
        NodeData::Custom(_, d) => d.type_name(),
    }
}

fn inspector(
    ui: &mut egui::Ui,
    id: NodeId,
    data: &NodeData,
    hidden: &HashSet<NodeId>,
    push: &mut impl FnMut(EditorCommand),
) {
    match data {
        NodeData::Transform(tf) => {
            ui.label(format!("translation {:?}", tf.translation));
            ui.label(format!("scale {:?}", tf.scale));
            let q = Quat::from_mat4(&tf.rotation);
            ui.label(format!("rotation quat {:?}", (q.x, q.y, q.z, q.w)));
        }
        NodeData::Material(m) => {
            let mut c = m.base_color.to_array();
            if ui.color_edit_button_rgb(&mut c).changed() {
                push(EditorCommand::SetBaseColor(id, c));
            }
            let mut met = m.metallic;
            if ui
                .add(egui::Slider::new(&mut met, 0.0..=1.0).text("metallic"))
                .changed()
            {
                push(EditorCommand::SetMetallic(id, met));
            }
            let mut rough = m.roughness;
            if ui
                .add(egui::Slider::new(&mut rough, 0.0..=1.0).text("roughness"))
                .changed()
            {
                push(EditorCommand::SetRoughness(id, rough));
            }
            let mut op = m.opacity;
            if ui
                .add(egui::Slider::new(&mut op, 0.0..=1.0).text("opacity"))
                .changed()
            {
                push(EditorCommand::SetOpacity(id, op));
            }
        }
        NodeData::DirectionalLight(l) => {
            let mut col = l.color.to_array();
            if ui.color_edit_button_rgb(&mut col).changed() {
                push(EditorCommand::SetLightColor(id, col));
            }
            let mut inten = l.intensity;
            if ui
                .add(egui::DragValue::new(&mut inten).speed(0.05))
                .changed()
            {
                push(EditorCommand::SetLightIntensity(id, inten));
            }
            let mut dir = l.direction.to_array();
            let mut dir_changed = false;
            ui.horizontal(|ui| {
                ui.label("dir");
                dir_changed |= ui
                    .add(egui::DragValue::new(&mut dir[0]).speed(0.02))
                    .changed();
                dir_changed |= ui
                    .add(egui::DragValue::new(&mut dir[1]).speed(0.02))
                    .changed();
                dir_changed |= ui
                    .add(egui::DragValue::new(&mut dir[2]).speed(0.02))
                    .changed();
            });
            if dir_changed {
                push(EditorCommand::SetLightDirection(id, dir));
            }
        }
        NodeData::PointLight(l) => {
            let mut col = l.color.to_array();
            if ui.color_edit_button_rgb(&mut col).changed() {
                push(EditorCommand::SetLightColor(id, col));
            }
            let mut inten = l.intensity;
            if ui
                .add(egui::DragValue::new(&mut inten).speed(0.05))
                .changed()
            {
                push(EditorCommand::SetLightIntensity(id, inten));
            }
        }
        NodeData::SpotLight(l) => {
            let mut col = l.color.to_array();
            if ui.color_edit_button_rgb(&mut col).changed() {
                push(EditorCommand::SetLightColor(id, col));
            }
            let mut inten = l.intensity;
            if ui
                .add(egui::DragValue::new(&mut inten).speed(0.05))
                .changed()
            {
                push(EditorCommand::SetLightIntensity(id, inten));
            }
            let mut dir = l.direction.to_array();
            let mut dir_changed = false;
            ui.horizontal(|ui| {
                ui.label("dir");
                dir_changed |= ui
                    .add(egui::DragValue::new(&mut dir[0]).speed(0.02))
                    .changed();
                dir_changed |= ui
                    .add(egui::DragValue::new(&mut dir[1]).speed(0.02))
                    .changed();
                dir_changed |= ui
                    .add(egui::DragValue::new(&mut dir[2]).speed(0.02))
                    .changed();
            });
            if dir_changed {
                push(EditorCommand::SetLightDirection(id, dir));
            }
        }
        NodeData::PerspectiveCamera(c) => {
            let mut fov = c.fov;
            if ui
                .add(egui::Slider::new(&mut fov, 0.1..=3.0).text("fov"))
                .changed()
            {
                push(EditorCommand::SetCameraFov(id, fov));
            }
            let mut near = c.near;
            if ui
                .add(
                    egui::DragValue::new(&mut near)
                        .speed(0.01)
                        .range(0.001..=10.0),
                )
                .changed()
            {
                push(EditorCommand::SetCameraNear(id, near));
            }
            let mut far = c.far;
            if ui
                .add(
                    egui::DragValue::new(&mut far)
                        .speed(1.0)
                        .range(1.0..=1_000_000.0),
                )
                .changed()
            {
                push(EditorCommand::SetCameraFar(id, far));
            }
            let mut rd = c.reverse_depth;
            if ui.checkbox(&mut rd, "reverse depth").changed() {
                push(EditorCommand::SetCameraReverseDepth(id, rd));
            }
        }
        NodeData::OrthographicCamera(c) => {
            let mut h = c.height;
            if ui
                .add(
                    egui::DragValue::new(&mut h)
                        .speed(0.05)
                        .range(0.01..=1_000_000.0),
                )
                .changed()
            {
                push(EditorCommand::SetOrthoHeight(id, h));
            }
            let mut near = c.near;
            if ui
                .add(
                    egui::DragValue::new(&mut near)
                        .speed(0.01)
                        .range(0.001..=10.0),
                )
                .changed()
            {
                push(EditorCommand::SetCameraNear(id, near));
            }
            let mut far = c.far;
            if ui
                .add(
                    egui::DragValue::new(&mut far)
                        .speed(1.0)
                        .range(1.0..=1_000_000.0),
                )
                .changed()
            {
                push(EditorCommand::SetCameraFar(id, far));
            }
            let mut rd = c.reverse_depth;
            if ui.checkbox(&mut rd, "reverse depth").changed() {
                push(EditorCommand::SetCameraReverseDepth(id, rd));
            }
        }
        NodeData::SectionPlane(sp) => {
            let mut en = sp.enabled;
            if ui.checkbox(&mut en, "enabled").changed() {
                push(EditorCommand::SetSectionPlaneEnabled(id, en));
            }
            let mut p = sp.plane;
            ui.horizontal(|ui| {
                ui.label("plane");
                ui.add(egui::DragValue::new(&mut p[0]).speed(0.02));
                ui.add(egui::DragValue::new(&mut p[1]).speed(0.02));
                ui.add(egui::DragValue::new(&mut p[2]).speed(0.02));
                ui.add(egui::DragValue::new(&mut p[3]).speed(0.02));
            });
            if ui.button("Apply plane").clicked() {
                push(EditorCommand::SetSectionPlaneEquation(id, p));
            }
        }
        NodeData::Switch(sw) => {
            let visible = sw.which_child >= 0;
            let mut v = visible;
            if ui.checkbox(&mut v, "switch visible").changed() {
                push(EditorCommand::SetNodeVisibility(id, v));
            }
        }
        _ => {
            ui.label(node_type_tag(data));
        }
    }

    let mut vis = match data {
        NodeData::Transform(_)
        | NodeData::Group(_)
        | NodeData::Environment(_)
        | NodeData::ShapeHints(_)
        | NodeData::Annotation(_)
        | NodeData::ResetTransform(_)
        | NodeData::Texture2Transform(_)
        | NodeData::MaterialBinding(_)
        | NodeData::IndexedLineSet(_)
        | NodeData::File(_)
        | NodeData::Decal(_)
        | NodeData::ExplodedView(_)
        | NodeData::ReflectionPlane(_)
        | NodeData::StereoCamera(_)
        | NodeData::RayTracing(_)
        | NodeData::Volume(_)
        | NodeData::PointCloud(_)
        | NodeData::Separator(_) => !hidden.contains(&id),
        NodeData::Switch(sw) => sw.which_child >= 0,
        _ => true,
    };
    if ui.checkbox(&mut vis, "visible (app)").changed() {
        push(EditorCommand::SetNodeVisibility(id, vis));
    }
}

/// Render feature checkboxes (used by both the View menu and the Render panel).
fn view_render_features_menu(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    push: &mut impl FnMut(EditorCommand),
) {
    let mut feat = |label: &str, field: bool, name: &'static str| {
        let mut v = field;
        if ui.checkbox(&mut v, label).changed() {
            push(EditorCommand::SetRenderFeature {
                feature_name: name,
                enabled: v,
            });
        }
    };
    let f = ui_ctx.render_features;
    feat("TAA", f.taa, "taa");
    feat("Motion blur", f.motion_blur, "motion_blur");
    feat("SSR", f.ssr, "ssr");
    feat("Color grading", f.color_grading, "color_grading");
    feat("DoF", f.dof, "dof");
    feat("Volumetric fog", f.volumetric_fog, "volumetric_fog");
    feat("Cluster lights", f.cluster_lights, "cluster_lights");
    feat("Omni shadows", f.omni_shadows, "omni_shadows");
}

fn render_panel(ui: &mut egui::Ui, ui_ctx: &EditorUiContext, push: &mut impl FnMut(EditorCommand)) {
    let mut hdr = ui_ctx.hdr_enabled;
    if ui.checkbox(&mut hdr, "HDR post").changed() {
        push(EditorCommand::SetHdrPostProcessing(hdr));
    }
    let mut vsync = ui_ctx.vsync_enabled;
    if ui.checkbox(&mut vsync, "VSync").changed() {
        push(EditorCommand::SetVsyncEnabled(vsync));
    }
    let mut grid = ui_ctx.grid_enabled;
    if ui.checkbox(&mut grid, "Grid").changed() {
        push(EditorCommand::SetGridEnabled(grid));
    }
    let mut hud = ui_ctx.hud_enabled;
    if ui.checkbox(&mut hud, "HUD").changed() {
        push(EditorCommand::SetHudEnabled(hud));
    }
    let mut xray = ui_ctx.xray_mode;
    if ui.checkbox(&mut xray, "X-ray").changed() {
        push(EditorCommand::SetXrayMode(xray));
    }

    ui.label("IBL preset");
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

    ui.label("Viewport layout");
    let mut lm = ui_ctx.layout_mode;
    egui::ComboBox::from_id_salt("layout_mode")
        .selected_text(format!("{lm:?}"))
        .show_ui(ui, |ui| {
            for m in [
                LayoutMode::Single,
                LayoutMode::Quad,
                LayoutMode::LeftRight,
                LayoutMode::TopBottom,
            ] {
                if ui.selectable_value(&mut lm, m, format!("{m:?}")).clicked() {
                    push(EditorCommand::SetViewportLayoutMode(m));
                }
            }
        });

    ui.label("Adaptive quality");
    let mut aq = ui_ctx.adaptive_quality_mode;
    egui::ComboBox::from_id_salt("adaptive_q")
        .selected_text(format!("{aq:?}"))
        .show_ui(ui, |ui| {
            for m in [
                AdaptiveQualityMode::Off,
                AdaptiveQualityMode::On,
                AdaptiveQualityMode::AutoIdleLock,
            ] {
                if ui.selectable_value(&mut aq, m, format!("{m:?}")).clicked() {
                    push(EditorCommand::SetAdaptiveQualityMode(m));
                }
            }
        });

    ui.label("View preset");
    let mut vp = ViewPreset::Iso;
    egui::ComboBox::from_id_salt("view_preset")
        .selected_text(format!("{vp:?}"))
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
                if ui.selectable_value(&mut vp, p, format!("{p:?}")).clicked() {
                    push(EditorCommand::SetViewPreset(p));
                }
            }
        });

    let mut ow = ui_ctx.outline_width;
    if ui
        .add(egui::Slider::new(&mut ow, 0.0..=8.0).text("outline width"))
        .changed()
    {
        push(EditorCommand::SetOutlineWidth(ow));
    }
    let mut oc = ui_ctx.outline_color;
    if ui.color_edit_button_rgba_unmultiplied(&mut oc).changed() {
        push(EditorCommand::SetOutlineColor(oc));
    }

    ui.separator();
    ui.label("Features");
    view_render_features_menu(ui, ui_ctx, push);

    ui.separator();
    ui.label("Measurement");
    if ui.selectable_label(false, "Off").clicked() {
        push(EditorCommand::SetMeasurementMode(None));
    }
    for m in [
        MeasurementType::Distance,
        MeasurementType::Angle,
        MeasurementType::Radius,
        MeasurementType::Diameter,
    ] {
        if ui.selectable_label(false, format!("{m:?}")).clicked() {
            push(EditorCommand::SetMeasurementMode(Some(m)));
        }
    }
}
