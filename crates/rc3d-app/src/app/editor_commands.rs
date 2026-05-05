use rc3d_actions::{
    AddChildCommand, RemoveChildCommand, SetFieldCommand,
    SetRotationCommand, SetScaleCommand, SetTranslationCommand,
};
use rc3d_core::math::{Mat4, Quat, Vec2, Vec3};
use rc3d_core::DisplayMode;
use rc3d_scene::NodeData;
use crate::editor_ui::{EditorCommand, EditorDisplayMode, NodeDataType};

use super::gizmo_support;
use super::App;

pub(crate) fn apply_editor_commands(app: &mut App) {
    while let Some(cmd) = app.editor_commands.pop_front() {
        match cmd {
            EditorCommand::Undo => {
                let _ = app.command_history.undo(&mut app.world.graph);
            }
            EditorCommand::Redo => {
                let _ = app.command_history.redo(&mut app.world.graph);
            }
            EditorCommand::FitSelection => app.fit_selection_to_view(),
            EditorCommand::ToggleMeasurement => {
                app.measurement_mode = !app.measurement_mode;
                app.measurement_first_point = None;
            }
            EditorCommand::ToggleSectionEdit => {
                app.section_edit_mode = !app.section_edit_mode;
            }
            EditorCommand::CycleIbl => {
                if let Some(renderer) = &mut app.renderer {
                    renderer.cycle_ibl_preset();
                }
            }
            EditorCommand::CycleViewportLayout => {
                if let Some(renderer) = &mut app.renderer {
                    let w = renderer.config.width;
                    let h = renderer.config.height;
                    renderer.viewport_layout.cycle_layout(w, h);
                }
                app.sync_viewport_camera_ids();
            }
            EditorCommand::CycleActiveViewport => {
                if let Some(renderer) = &mut app.renderer {
                    renderer.viewport_layout.cycle_active();
                    app.viewport_cameras.active_viewport = renderer.viewport_layout.active_id;
                }
            }
            EditorCommand::SetDisplayMode(mode) => {
                if let Some(renderer) = &mut app.renderer {
                    let mode = match mode {
                        EditorDisplayMode::Wireframe => DisplayMode::Wireframe,
                        EditorDisplayMode::Shaded => DisplayMode::Shaded,
                        EditorDisplayMode::ShadedWithEdges => DisplayMode::ShadedWithEdges,
                        EditorDisplayMode::HiddenLine => DisplayMode::HiddenLine,
                    };
                    renderer.set_display_mode(mode);
                }
            }
            EditorCommand::SetSelection(id) => {
                app.world.graph.clear_selection();
                if let Some(id) = id {
                    app.world.graph.select(id);
                }
                gizmo_support::sync_gizmo_from_selection(&mut app.gizmo, &app.world.graph);
            }
            EditorCommand::SetNodeVisibility(id, visible) => {
                if let Some(entry) = app.world.graph.get_mut(id) {
                    match &mut entry.data {
                        rc3d_scene::NodeData::Transform(_)
                        | rc3d_scene::NodeData::Group(_)
                        | rc3d_scene::NodeData::Separator(_) => {
                            if visible {
                                app.hidden_nodes.remove(&id);
                            } else {
                                app.hidden_nodes.insert(id);
                            }
                        }
                        rc3d_scene::NodeData::Switch(sw) => {
                            sw.which_child = if visible { -1 } else { -2 };
                        }
                        rc3d_scene::NodeData::SectionPlane(sp) => {
                            sp.enabled = visible;
                        }
                        rc3d_scene::NodeData::Markup(m) => {
                            m.visible = visible;
                        }
                        _ => {}
                    }
                }
            }
            EditorCommand::PreviewTransformTranslation(node, t) => {
                if let Some(entry) = app.world.graph.get_mut(node) {
                    if let rc3d_scene::NodeData::Transform(tf) = &mut entry.data {
                        tf.translation = Vec3::new(t[0], t[1], t[2]);
                    }
                }
            }
            EditorCommand::CommitTransformTranslation { node, old, new } => {
                let old_value = Vec3::new(old[0], old[1], old[2]);
                let new_value = Vec3::new(new[0], new[1], new[2]);
                if (new_value - old_value).length_squared() > 1e-10 {
                    app.command_history.execute(
                        Box::new(SetTranslationCommand {
                            node,
                            old_value,
                            new_value,
                        }),
                        &mut app.world.graph,
                    );
                }
            }
            EditorCommand::PreviewTransformScale(node, s) => {
                if let Some(entry) = app.world.graph.get_mut(node) {
                    if let rc3d_scene::NodeData::Transform(tf) = &mut entry.data {
                        tf.scale = Vec3::new(s[0], s[1], s[2]);
                    }
                }
            }
            EditorCommand::CommitTransformScale { node, old, new } => {
                let old_value = Vec3::new(old[0], old[1], old[2]);
                let new_value = Vec3::new(new[0], new[1], new[2]);
                if (new_value - old_value).length_squared() > 1e-10 {
                    app.command_history.execute(
                        Box::new(SetScaleCommand {
                            node,
                            old_value,
                            new_value,
                        }),
                        &mut app.world.graph,
                    );
                }
            }
            EditorCommand::PreviewTransformRotationQuat(node, quat) => {
                if let Some(entry) = app.world.graph.get_mut(node) {
                    if let rc3d_scene::NodeData::Transform(tf) = &mut entry.data {
                        tf.rotation = Mat4::from_quat(Quat::from_xyzw(
                            quat[0], quat[1], quat[2], quat[3],
                        ));
                    }
                }
            }
            EditorCommand::CommitTransformRotationQuat { node, old, new } => {
                let old_value = Mat4::from_quat(Quat::from_xyzw(old[0], old[1], old[2], old[3]));
                let new_value = Mat4::from_quat(Quat::from_xyzw(new[0], new[1], new[2], new[3]));
                if old_value.to_cols_array_2d() != new_value.to_cols_array_2d() {
                    app.command_history.execute(
                        Box::new(SetRotationCommand {
                            node,
                            old_value,
                            new_value,
                        }),
                        &mut app.world.graph,
                    );
                }
            }
            EditorCommand::ImportPath(path) => match rc3d_io::import_file(path.as_path()) {
                Ok(graph) => {
                    app.world.graph = graph;
                    if let Some(renderer) = &mut app.renderer {
                        app.world.invalidate_caches(renderer);
                    } else {
                        app.world.collector.invalidate_mesh_cache();
                    }
                    app.world.graph.clear_selection();
                    app.measurements.clear();
                    app.measurement_first_point = None;
                    gizmo_support::sync_gizmo_from_selection(&mut app.gizmo, &app.world.graph);
                    log::info!("Imported scene: {}", path.display());
                }
                Err(err) => {
                    log::error!("Import failed: {}", err);
                }
            },
            EditorCommand::ExportIvPath(path) => {
                let content = rc3d_io::write_iv(&app.world.graph);
                if let Err(err) = std::fs::write(path.as_path(), content) {
                    log::error!("Export IV failed ({}): {}", path.display(), err);
                } else {
                    log::info!("Exported IV: {}", path.display());
                }
            }
            EditorCommand::ExportDiagnosticsJsonPath(path) => {
                if let Some(diag) = app.last_render_stats.diagnostics.as_ref() {
                    let content = diag.to_json_pretty();
                    if let Err(err) = std::fs::write(path.as_path(), content) {
                        log::error!(
                            "Export diagnostics JSON failed ({}): {}",
                            path.display(),
                            err
                        );
                    } else {
                        log::info!("Exported diagnostics JSON: {}", path.display());
                    }
                } else {
                    log::warn!("No diagnostics available to export");
                }
            }
            EditorCommand::SetGizmoMode(mode) => {
                app.gizmo.mode = mode;
            }
            EditorCommand::SetRenderFeature { feature_name, enabled } => {
                if let Some(r) = &mut app.renderer {
                    match feature_name {
                        "taa" => r.enable_taa = enabled,
                        "motion_blur" => r.enable_motion_blur = enabled,
                        "ssr" => r.enable_ssr = enabled,
                        "color_grading" => r.enable_color_grading = enabled,
                        "dof" => r.enable_dof = enabled,
                        "volumetric_fog" => r.enable_volumetric_fog = enabled,
                        "cluster_lights" => r.enable_cluster_lights = enabled,
                        "omni_shadows" => r.enable_omni_shadows = enabled,
                        _ => {}
                    }
                }
            }
            EditorCommand::SetHdrPostProcessing(enabled) => {
                if let Some(r) = &mut app.renderer {
                    r.hdr_post_processing = enabled;
                }
            }
            EditorCommand::SetIblPreset(preset) => {
                if let Some(r) = &mut app.renderer {
                    r.set_ibl_preset(preset);
                }
            }
            EditorCommand::SetAdaptiveQualityMode(mode) => {
                app.adaptive_quality_mode = mode;
            }
            EditorCommand::SetOutlineWidth(w) => {
                if let Some(r) = &mut app.renderer {
                    r.outline_width = w;
                }
            }
            EditorCommand::SetOutlineColor(c) => {
                if let Some(r) = &mut app.renderer {
                    r.set_outline_color(c);
                }
            }
            EditorCommand::SetXrayMode(enabled) => {
                if let Some(r) = &mut app.renderer {
                    r.xray_mode = enabled;
                }
            }
            EditorCommand::SetViewportLayoutMode(lm) => {
                if let Some(r) = &mut app.renderer {
                    r.viewport_layout.layout_mode = lm;
                    let w = r.config.width;
                    let h = r.config.height;
                    r.viewport_layout.rebuild(w, h);
                }
                app.sync_viewport_camera_ids();
            }
            EditorCommand::SetViewPreset(preset) => {
                if let Some(ctrl) = app.active_camera_controller_mut() {
                    ctrl.set_view_preset(preset);
                }
            }
            EditorCommand::SetGridEnabled(enabled) => {
                app.grid_enabled = enabled;
                if let Some(r) = &mut app.renderer {
                    r.grid_enabled = enabled;
                }
            }
            EditorCommand::SetHudEnabled(enabled) => {
                if let Some(r) = &mut app.renderer {
                    r.hud_enabled = enabled;
                }
            }
            EditorCommand::SetVsyncEnabled(enabled) => {
                if let Some(r) = &mut app.renderer {
                    r.set_vsync(enabled);
                }
            }
            EditorCommand::SetBaseColor(node, color) => {
                let new_value = Vec3::new(color[0], color[1], color[2]);
                let old_value = app.world.graph.get(node)
                    .and_then(|e| if let NodeData::Material(m) = &e.data { Some(m.base_color) } else { None });
                if let Some(old) = old_value {
                    if (new_value - old).length_squared() > 1e-10 {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, new_value, "SetBaseColor",
                                |entry, v| if let NodeData::Material(m) = &mut entry.data { m.base_color = v })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetMetallic(node, v) => {
                let old = app.world.graph.get(node)
                    .and_then(|e| if let NodeData::Material(m) = &e.data { Some(m.metallic) } else { None });
                if let Some(old) = old {
                    if (v - old).abs() > 1e-10 {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, v, "SetMetallic",
                                |entry, v| if let NodeData::Material(m) = &mut entry.data { m.metallic = v })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetRoughness(node, v) => {
                let old = app.world.graph.get(node)
                    .and_then(|e| if let NodeData::Material(m) = &e.data { Some(m.roughness) } else { None });
                if let Some(old) = old {
                    if (v - old).abs() > 1e-10 {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, v, "SetRoughness",
                                |entry, v| if let NodeData::Material(m) = &mut entry.data { m.roughness = v })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetOpacity(node, v) => {
                let old = app.world.graph.get(node)
                    .and_then(|e| if let NodeData::Material(m) = &e.data { Some(m.opacity) } else { None });
                if let Some(old) = old {
                    if (v - old).abs() > 1e-10 {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, v, "SetOpacity",
                                |entry, v| if let NodeData::Material(m) = &mut entry.data { m.opacity = v })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetLightColor(node, color) => {
                let new_value = Vec3::new(color[0], color[1], color[2]);
                let old_value = app.world.graph.get(node).and_then(|e| match &e.data {
                    NodeData::DirectionalLight(l) => Some(l.color),
                    NodeData::PointLight(l) => Some(l.color),
                    NodeData::SpotLight(l) => Some(l.color),
                    _ => None,
                });
                if let Some(old) = old_value {
                    if (new_value - old).length_squared() > 1e-10 {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, new_value, "SetLightColor",
                                |entry, v| match &mut entry.data {
                                    NodeData::DirectionalLight(l) => l.color = v,
                                    NodeData::PointLight(l) => l.color = v,
                                    NodeData::SpotLight(l) => l.color = v,
                                    _ => {}
                                })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetLightIntensity(node, v) => {
                let old = app.world.graph.get(node).and_then(|e| match &e.data {
                    NodeData::DirectionalLight(l) => Some(l.intensity),
                    NodeData::PointLight(l) => Some(l.intensity),
                    NodeData::SpotLight(l) => Some(l.intensity),
                    _ => None,
                });
                if let Some(old) = old {
                    if (v - old).abs() > 1e-10 {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, v, "SetLightIntensity",
                                |entry, v| match &mut entry.data {
                                    NodeData::DirectionalLight(l) => l.intensity = v,
                                    NodeData::PointLight(l) => l.intensity = v,
                                    NodeData::SpotLight(l) => l.intensity = v,
                                    _ => {}
                                })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetLightDirection(node, dir) => {
                let new_value = Vec3::new(dir[0], dir[1], dir[2]).normalize();
                let old_value = app.world.graph.get(node).and_then(|e| match &e.data {
                    NodeData::DirectionalLight(l) => Some(l.direction),
                    NodeData::SpotLight(l) => Some(l.direction),
                    _ => None,
                });
                if let Some(old) = old_value {
                    if (new_value - old).length_squared() > 1e-10 {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, new_value, "SetLightDirection",
                                |entry, v| match &mut entry.data {
                                    NodeData::DirectionalLight(l) => l.direction = v,
                                    NodeData::SpotLight(l) => l.direction = v,
                                    _ => {}
                                })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetCameraFov(node, v) => {
                let new_fov = v.clamp(0.01, std::f32::consts::PI - 0.01);
                let old = app.world.graph.get(node)
                    .and_then(|e| if let NodeData::PerspectiveCamera(c) = &e.data { Some(c.fov) } else { None });
                if let Some(old) = old {
                    if (new_fov - old).abs() > 1e-6 {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, new_fov, "SetCameraFov",
                                |entry, v| if let NodeData::PerspectiveCamera(c) = &mut entry.data { c.fov = v })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetCameraNear(node, v) => {
                let new_val = v.max(0.001);
                let old = app.world.graph.get(node).and_then(|e| match &e.data {
                    NodeData::PerspectiveCamera(c) => Some(c.near),
                    NodeData::OrthographicCamera(c) => Some(c.near),
                    _ => None,
                });
                if let Some(old) = old {
                    if (new_val - old).abs() > 1e-6 {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, new_val, "SetCameraNear",
                                |entry, v| match &mut entry.data {
                                    NodeData::PerspectiveCamera(c) => c.near = v,
                                    NodeData::OrthographicCamera(c) => c.near = v,
                                    _ => {}
                                })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetCameraFar(node, v) => {
                let new_val = v.max(1.0);
                let old = app.world.graph.get(node).and_then(|e| match &e.data {
                    NodeData::PerspectiveCamera(c) => Some(c.far),
                    NodeData::OrthographicCamera(c) => Some(c.far),
                    _ => None,
                });
                if let Some(old) = old {
                    if (new_val - old).abs() > 1e-6 {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, new_val, "SetCameraFar",
                                |entry, v| match &mut entry.data {
                                    NodeData::PerspectiveCamera(c) => c.far = v,
                                    NodeData::OrthographicCamera(c) => c.far = v,
                                    _ => {}
                                })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetCameraReverseDepth(node, enabled) => {
                let old = app.world.graph.get(node).and_then(|e| match &e.data {
                    NodeData::PerspectiveCamera(c) => Some(c.reverse_depth),
                    NodeData::OrthographicCamera(c) => Some(c.reverse_depth),
                    _ => None,
                });
                if let Some(old) = old {
                    if enabled != old {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, enabled, "SetCameraReverseDepth",
                                |entry, v| match &mut entry.data {
                                    NodeData::PerspectiveCamera(c) => c.reverse_depth = v,
                                    NodeData::OrthographicCamera(c) => c.reverse_depth = v,
                                    _ => {}
                                })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetOrthoHeight(node, v) => {
                let new_height = v.max(0.001);
                let old = app.world.graph.get(node)
                    .and_then(|e| if let NodeData::OrthographicCamera(c) = &e.data { Some(c.height) } else { None });
                if let Some(old) = old {
                    if (new_height - old).abs() > 1e-6 {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, new_height, "SetOrthoHeight",
                                |entry, v| if let NodeData::OrthographicCamera(c) = &mut entry.data { c.height = v })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetSectionPlaneEnabled(node, enabled) => {
                let old = app.world.graph.get(node)
                    .and_then(|e| if let NodeData::SectionPlane(sp) = &e.data { Some(sp.enabled) } else { None });
                if let Some(old) = old {
                    if enabled != old {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, enabled, "SetSectionPlaneEnabled",
                                |entry, v| if let NodeData::SectionPlane(sp) = &mut entry.data { sp.enabled = v })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::SetSectionPlaneEquation(node, eq) => {
                let old = app.world.graph.get(node)
                    .and_then(|e| if let NodeData::SectionPlane(sp) = &e.data { Some(sp.plane) } else { None });
                if let Some(old) = old {
                    if (eq[0] - old[0]).abs() > 1e-10 || (eq[1] - old[1]).abs() > 1e-10
                        || (eq[2] - old[2]).abs() > 1e-10 || (eq[3] - old[3]).abs() > 1e-10
                    {
                        app.command_history.execute(
                            Box::new(SetFieldCommand::new(node, old, eq, "SetSectionPlaneEquation",
                                |entry, v| if let NodeData::SectionPlane(sp) = &mut entry.data { sp.plane = v })),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::CreateNode { node_type, parent } => {
                let data = node_type_to_node_data(node_type);
                let (id, parent_id) = if let Some(p) = parent {
                    (app.world.graph.add_child(p, data), p)
                } else {
                    let id = app.world.graph.add_root(data);
                    (id, id) // root node has self as parent for undo tracking
                };
                app.world.graph.clear_selection();
                app.world.graph.select(id);
                app.command_history.execute(
                    Box::new(AddChildCommand::new(parent_id, id)),
                    &mut app.world.graph,
                );
            }
            EditorCommand::DeleteNode(node) => {
                if let Some(parent) = app.world.graph.get(node).and_then(|e| e.parent) {
                    let cmd = RemoveChildCommand::new(parent, node, &app.world.graph);
                    app.command_history.execute(Box::new(cmd), &mut app.world.graph);
                } else {
                    app.world.graph.remove(node);
                }
            }
            EditorCommand::DuplicateNode(node) => {
                let data = app.world.graph.get(node).map(|e| e.data.clone());
                if let Some(data) = data {
                    let parent = app.world.graph.get(node)
                        .and_then(|e| e.parent)
                        .or_else(|| app.world.graph.roots().first().copied());
                    let clone = match parent {
                        Some(p) => app.world.graph.add_child(p, data),
                        None => app.world.graph.add_root(data),
                    };
                    app.world.graph.clear_selection();
                    app.world.graph.select(clone);
                    if let Some(p) = parent {
                        app.command_history.execute(
                            Box::new(AddChildCommand::new(p, clone)),
                            &mut app.world.graph,
                        );
                    }
                }
            }
            EditorCommand::RenameNode(_node, _name) => {}
            EditorCommand::SetMeasurementMode(mode) => {
                let is_active = mode.is_some();
                app.measurement_type = mode;
                app.measurement_mode = is_active;
                app.measurement_first_point = None;
                app.measurements.clear();
            }
            EditorCommand::SetMarkupTool(tool) => {
                app.markup_action.set_tool(tool);
                let root = app.world.graph.roots().first().copied().unwrap_or_default();
                app.markup_action.ensure_target_node(&mut app.world.graph, root);
            }
            EditorCommand::MarkupMouseDown { screen_pos } => {
                let pos = Vec2::new(screen_pos[0], screen_pos[1]);
                app.markup_action.on_mouse_down(pos, &mut app.world.graph);
            }
            EditorCommand::MarkupMouseMove { screen_pos } => {
                app.markup_action.on_mouse_move(Vec2::new(screen_pos[0], screen_pos[1]));
            }
            EditorCommand::MarkupMouseUp { screen_pos } => {
                if let Some(element) =
                    app.markup_action.on_mouse_up(Vec2::new(screen_pos[0], screen_pos[1]))
                {
                    if let Some(target) = app.markup_action.target_node {
                        if let Some(entry) = app.world.graph.get_mut(target) {
                            if let NodeData::Markup(m) = &mut entry.data {
                                m.elements.push(element);
                            }
                        }
                    }
                }
            }
            EditorCommand::ClearAllMarkup { node } => {
                if let Some(entry) = app.world.graph.get_mut(node) {
                    if let NodeData::Markup(m) = &mut entry.data {
                        m.elements.clear();
                    }
                }
            }
            EditorCommand::SaveBookmark(slot) => {
                if let Some(ctrl) = app.active_camera_controller_mut() {
                    ctrl.save_bookmark(slot, "");
                }
            }
            EditorCommand::RecallBookmark(slot) => {
                if let Some(ctrl) = app.active_camera_controller_mut() {
                    ctrl.recall_bookmark(slot);
                }
            }
        }
    }
}

fn node_type_to_node_data(node_type: NodeDataType) -> rc3d_scene::NodeData {
    use rc3d_scene::node_data::*;
    match node_type {
        NodeDataType::Cube => NodeData::Cube(CubeNode::default()),
        NodeDataType::Sphere => NodeData::Sphere(SphereNode::default()),
        NodeDataType::Cylinder => NodeData::Cylinder(CylinderNode::default()),
        NodeDataType::Cone => NodeData::Cone(ConeNode::default()),
        NodeDataType::Separator => NodeData::Separator(SeparatorNode),
        NodeDataType::DirectionalLight => {
            NodeData::DirectionalLight(DirectionalLightNode::default())
        }
        NodeDataType::PointLight => NodeData::PointLight(PointLightNode::default()),
        NodeDataType::SpotLight => NodeData::SpotLight(SpotLightNode::default()),
        NodeDataType::PerspectiveCamera => {
            NodeData::PerspectiveCamera(PerspectiveCameraNode::default())
        }
        NodeDataType::OrthographicCamera => {
            NodeData::OrthographicCamera(OrthographicCameraNode::default())
        }
        NodeDataType::Text2 => NodeData::Text2(Text2Node::default()),
        NodeDataType::Text3 => NodeData::Text3(Text3Node::default()),
    }
}
