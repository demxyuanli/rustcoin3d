use std::path::Path;

use rc3d_actions::{
    Command, CommandHistory, CreateNodeCommand, DeleteNodeCommand, ReparentNodesCommand,
    SetFieldCommand, SetRotationCommand, SetScaleCommand, SetTranslationCommand,
};
use rc3d_core::math::{Mat4, Quat, Vec3};
use rc3d_core::{DisplayMode, NodeId};
use rc3d_engine_api::{CameraController, Engine};
use rc3d_fields::FieldValue;
use rc3d_render::AdaptiveControl;
use rc3d_scene::node_data::NodeData;
use rc3d_scene::{read_node_field, write_node_field, SceneGraph};

use crate::commands::{AdaptiveQualityMode, EditorCommand};
use crate::context::EditorInteractionState;
use crate::document;
use crate::node_factory::node_data_for_type;
use crate::ui::types::EditorDisplayMode;

/// Document undo stack plus markup tool state owned by the host.
pub struct EditorSession {
    pub history: CommandHistory,
    pub adaptive_quality_mode: AdaptiveQualityMode,
    pub document_path: Option<std::path::PathBuf>,
    pub dirty: bool,
    pub background: rc3d_engine_api::background::BackgroundSettings,
    pub ui_theme: crate::ui::theme::UiTheme,
    pub ui_locale: crate::ui::i18n::UiLocale,
}

impl Default for EditorSession {
    fn default() -> Self {
        Self {
            history: CommandHistory::new(128),
            adaptive_quality_mode: AdaptiveQualityMode::Off,
            document_path: None,
            dirty: false,
            background: rc3d_engine_api::background::BackgroundSettings::default(),
            ui_theme: crate::ui::theme::UiTheme::Dark,
            ui_locale: crate::ui::i18n::UiLocale::En,
        }
    }
}

impl EditorSession {
    pub fn execute_edit(&mut self, cmd: Box<dyn Command>, graph: &mut SceneGraph) {
        self.history.execute(cmd, graph);
        self.dirty = true;
    }

    pub fn window_title(&self) -> String {
        document::window_title(self.document_path.as_deref(), self.dirty)
    }
}

/// Apply an editor command to the engine, interaction, and document history.
pub fn apply_command(
    engine: &mut Engine,
    interaction: &mut EditorInteractionState,
    session: &mut EditorSession,
    cmd: EditorCommand,
) {
    match cmd {
        EditorCommand::Undo => {
            session.history.undo(&mut engine.world.graph);
            rc3d_engine_api::sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
        }
        EditorCommand::Redo => {
            session.history.redo(&mut engine.world.graph);
            rc3d_engine_api::sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
        }
        EditorCommand::ApplyVisualStyle(name) => {
            engine.apply_visual_style_to_selected(&name);
        }
        EditorCommand::SetViewportLayoutMode(mode) => {
            engine.set_layout_mode(mode);
        }
        EditorCommand::CycleViewportLayout => {
            engine.cycle_layout_mode();
        }
        EditorCommand::CycleActiveViewport => {
            engine.cycle_active_viewport();
        }
        EditorCommand::SetViewPreset(preset) => {
            engine.set_view_preset(preset);
        }
        EditorCommand::SetUiTheme(theme) => {
            session.ui_theme = theme;
        }
        EditorCommand::SetUiLocale(locale) => {
            session.ui_locale = locale;
        }
        EditorCommand::SetViewFromDirection(from) => {
            engine.set_view_from_direction(Vec3::from_array(from));
        }
        EditorCommand::OrbitView { dx, dy } => {
            engine.orbit_view(dx, dy);
        }
        EditorCommand::SetWboit(enabled) => {
            engine.set_wboit(enabled);
        }
        EditorCommand::SetGizmoMode(mode) => {
            engine.set_gizmo_mode(mode);
        }
        EditorCommand::SetSelection(Some(id)) => {
            engine.world.graph.clear_selection();
            engine.world.graph.select(id);
            rc3d_engine_api::sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
        }
        EditorCommand::SetSelection(None) => {
            engine.world.graph.clear_selection();
            rc3d_engine_api::sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
        }
        EditorCommand::SetSelectionMany(ids) => {
            engine.world.graph.clear_selection();
            engine.world.graph.select_many(ids);
            rc3d_engine_api::sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
        }
        EditorCommand::FitSelection => {
            crate::interaction::fit_selection_to_view(engine);
        }
        EditorCommand::FitAll => {
            crate::interaction::fit_scene_to_view(engine);
        }
        EditorCommand::SetWalkMode(enabled) => {
            engine.controller.walk_mode = enabled;
            if let Some(vc) = engine.viewport_cameras.active_mut() {
                vc.controller.walk_mode = enabled;
            }
        }
        EditorCommand::SetXrayMode(enabled) => {
            engine.set_xray_mode(enabled);
        }
        EditorCommand::SetGhostUnselected(enabled) => {
            engine.set_ghost_unselected(enabled);
        }
        EditorCommand::SetGhostOpacity(opacity) => {
            engine.set_ghost_opacity(opacity);
        }
        EditorCommand::SetFillStyle(fill) => {
            apply_to_selected_or_roots(engine, |graph, id| graph.set_fill_style(id, fill));
            session.dirty = true;
        }
        EditorCommand::SetEdgeStyle(edges) => {
            apply_to_selected_or_roots(engine, |graph, id| graph.set_edge_style(id, edges));
            session.dirty = true;
        }
        EditorCommand::SetFeatureEdgeColor(c) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_feature_edge_color(c);
            }
        }
        EditorCommand::SetWireframeEdgeColor(c) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_wireframe_edge_color(c);
            }
        }
        EditorCommand::SetHiddenEdgeColor(c) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_hidden_edge_color(c);
            }
        }
        EditorCommand::SetCreaseAngle(deg) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_feature_edge_crease_angle(deg);
            }
            engine.world.collector.invalidate_mesh_cache();
        }
        EditorCommand::SetSsEdgeThreshold(t) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_ss_edge_threshold(t);
            }
        }
        EditorCommand::ToggleSectionEdit => {
            interaction.section_edit_mode = !interaction.section_edit_mode;
            if !interaction.section_edit_mode {
                interaction.section_drag = None;
                interaction.section_hovered = None;
            }
        }
        EditorCommand::ToggleMeasurement => {
            interaction.measurement_mode = !interaction.measurement_mode;
        }
        EditorCommand::SetMeasurementMode(mode) => {
            interaction.measurement_mode = mode.is_some();
            interaction.measurement_type = mode;
        }
        EditorCommand::SetSectionPlaneEquation(id, plane) => {
            let old = match engine.world.graph.get(id) {
                Some(e) => match &e.data {
                    NodeData::SectionPlane(sp) => sp.plane,
                    _ => plane,
                },
                None => plane,
            };
            session.execute_edit(
                Box::new(NodeFieldCommand {
                    node: id,
                    field_index: 0,
                    old: FieldValue::Vec4f(rc3d_core::math::Vec4::from_array(old)),
                    new: FieldValue::Vec4f(rc3d_core::math::Vec4::from_array(plane)),
                }),
                &mut engine.world.graph,
            );
        }
        EditorCommand::SetSectionPlaneEnabled(id, enabled) => {
            let old = match engine.world.graph.get(id) {
                Some(e) => match &e.data {
                    NodeData::SectionPlane(sp) => sp.enabled,
                    _ => enabled,
                },
                None => enabled,
            };
            session.execute_edit(
                Box::new(NodeFieldCommand {
                    node: id,
                    field_index: 1,
                    old: FieldValue::Bool(old),
                    new: FieldValue::Bool(enabled),
                }),
                &mut engine.world.graph,
            );
        }
        EditorCommand::SetDisplayMode(mode) => {
            engine.set_display_mode(display_mode_from_editor(mode));
        }
        EditorCommand::SetGridEnabled(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_grid_enabled(enabled);
            }
        }
        EditorCommand::SetHudEnabled(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_hud_enabled(enabled);
            }
        }
        EditorCommand::SetVsyncEnabled(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_vsync(enabled);
            }
        }
        EditorCommand::SetHdrPostProcessing(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_hdr_post_processing(enabled);
            }
        }
        EditorCommand::CycleIbl => {
            if let Some(r) = engine.renderer.as_mut() {
                r.cycle_ibl_preset();
            }
        }
        EditorCommand::SetIblPreset(preset) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_ibl_preset(preset);
            }
        }
        EditorCommand::SetOutlineWidth(w) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_outline_width(w);
            }
        }
        EditorCommand::SetOutlineColor(c) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_outline_color(c);
            }
        }
        EditorCommand::SetCadDisplayTier(tier) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_display_tier(tier);
            }
        }
        EditorCommand::SetAdaptiveQualityMode(mode) => {
            session.adaptive_quality_mode = mode;
            engine.set_adaptive_quality(adaptive_from_editor(mode));
        }
        EditorCommand::SetRenderFeature {
            feature_name,
            enabled,
        } => {
            if let Some(r) = engine.renderer.as_mut() {
                match feature_name {
                    "taa" => r.set_taa(enabled),
                    "motion_blur" => r.set_motion_blur(enabled),
                    "ssr" => r.set_ssr(enabled),
                    "color_grading" => r.set_color_grading(enabled),
                    "dof" => r.set_dof(enabled),
                    "volumetric_fog" => r.set_volumetric_fog(enabled),
                    "cluster_lights" => r.set_cluster_lights(enabled),
                    "omni_shadows" => r.set_omni_shadows(enabled),
                    "ldr_fxaa" => r.set_ldr_fxaa(enabled),
                    "screen_space_edges" => r.set_screen_space_edges(enabled),
                    "screen_space_selection_outline" => {
                        r.set_screen_space_selection_outline(enabled)
                    }
                    "gpu_cull" => r.set_gpu_culling(enabled),
                    "parallel_traversal" => r.set_parallel_traversal(enabled),
                    _ => {}
                }
            }
        }
        EditorCommand::SetNodeVisibility(id, visible) => {
            if visible {
                engine.hidden_nodes.remove(&id);
            } else {
                engine.hidden_nodes.insert(id);
            }
        }
        EditorCommand::PreviewTransformTranslation(id, t) => {
            if let Some(e) = engine.world.graph.get_mut(id) {
                if let NodeData::Transform(tf) = &mut e.data {
                    tf.translation = Vec3::from_array(t);
                }
            }
        }
        EditorCommand::PreviewTransformScale(id, s) => {
            if let Some(e) = engine.world.graph.get_mut(id) {
                if let NodeData::Transform(tf) = &mut e.data {
                    tf.scale = Vec3::from_array(s);
                }
            }
        }
        EditorCommand::PreviewTransformRotationQuat(id, q) => {
            if let Some(e) = engine.world.graph.get_mut(id) {
                if let NodeData::Transform(tf) = &mut e.data {
                    tf.rotation = Mat4::from_quat(Quat::from_xyzw(q[0], q[1], q[2], q[3]));
                }
            }
        }
        EditorCommand::CommitTransformTranslation { node, old, new } => {
            session.execute_edit(
                Box::new(SetTranslationCommand {
                    node,
                    old_value: Vec3::from_array(old),
                    new_value: Vec3::from_array(new),
                }),
                &mut engine.world.graph,
            );
            rc3d_engine_api::sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
        }
        EditorCommand::CommitTransformScale { node, old, new } => {
            session.execute_edit(
                Box::new(SetScaleCommand {
                    node,
                    old_value: Vec3::from_array(old),
                    new_value: Vec3::from_array(new),
                }),
                &mut engine.world.graph,
            );
        }
        EditorCommand::CommitTransformRotationQuat { node, old, new } => {
            session.execute_edit(
                Box::new(SetRotationCommand {
                    node,
                    old_value: Mat4::from_quat(Quat::from_xyzw(old[0], old[1], old[2], old[3])),
                    new_value: Mat4::from_quat(Quat::from_xyzw(new[0], new[1], new[2], new[3])),
                }),
                &mut engine.world.graph,
            );
        }
        EditorCommand::NewScene => {
            install_scene(engine, session, interaction, document::blank_scene(), None);
        }
        EditorCommand::OpenScene(path) => match document::load_native_scene(&path) {
            Ok(graph) => {
                install_scene(engine, session, interaction, graph, Some(path));
            }
            Err(e) => log::warn!("open scene failed: {e}"),
        },
        EditorCommand::SaveScene => {
            if let Some(path) = session.document_path.clone() {
                save_bound_scene(engine, session, &path);
            } else {
                log::warn!("save scene: no document path");
            }
        }
        EditorCommand::SaveSceneAs(path) => {
            let path = document::with_json_extension(path);
            if save_bound_scene(engine, session, &path) {
                session.document_path = Some(path);
            }
        }
        EditorCommand::ImportPath(path) => {
            import_path(engine, &path);
            session.dirty = true;
        }
        EditorCommand::ExportIvPath(path) => {
            export_scene_json(engine.scene(), &path);
        }
        EditorCommand::ExportDiagnosticsJsonPath(path) => {
            export_diagnostics(engine, &path);
        }
        EditorCommand::ExportScreenshot(path) => {
            export_screenshot(engine, &path);
        }
        EditorCommand::ExportHiddenLineSvg(path) => {
            if let Err(e) = engine.export_hidden_line_svg(&path) {
                log::warn!("hidden-line SVG export failed: {e}");
            }
        }
        EditorCommand::ExportQuadPack(path) => {
            export_quad_pack(engine, &path);
        }
        EditorCommand::LoadIblHdr(path) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_ibl_from_path(path);
            }
        }
        EditorCommand::SetGpuCulling(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_gpu_culling(enabled);
            }
        }
        EditorCommand::SetParallelTraversal(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_parallel_traversal(enabled);
            }
        }
        EditorCommand::SetCsmShadow {
            resolution,
            cascade_count,
        } => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_csm_shadow(resolution, cascade_count);
            }
        }
        EditorCommand::SetInteractionRenderScale(scale) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_interaction_render_scale(scale);
            }
        }
        EditorCommand::SetBgMode(mode) => {
            session.background.mode = mode;
            engine.set_background(session.background.clone());
        }
        EditorCommand::SetBgTopColor(c) => {
            session.background.top_color = c;
            session.background.clear_color = c;
            engine.set_background(session.background.clone());
        }
        EditorCommand::SetBgBotColor(c) => {
            session.background.bot_color = c;
            engine.set_background(session.background.clone());
        }
        EditorCommand::SetBgImage(path) => {
            session.background.mode = rc3d_render::background::BgMode::Image;
            session.background.image_path = Some(path.to_string_lossy().into_owned());
            engine.set_background(session.background.clone());
        }
        EditorCommand::SetPostEffects {
            vignette,
            chromatic,
            bloom,
            grain,
        } => {
            engine.set_post_effects(vignette, chromatic, bloom, grain);
        }
        EditorCommand::SetPostStylize { halftone, glitch } => {
            engine.set_post_stylize(halftone, glitch);
        }
        EditorCommand::SetBaseColor(id, c) => {
            mutate_material(engine, session, id, c, |m, c| {
                m.base_color = Vec3::from_array(c);
                m.diffuse_color = Vec3::from_array(c);
            });
        }
        EditorCommand::SetMetallic(id, v) => {
            let old = material_metallic(&engine.world.graph, id).unwrap_or(v);
            apply_scalar_field(engine, session, id, old, v, "metallic", |e, v| {
                if let NodeData::Material(m) = &mut e.data {
                    m.metallic = v;
                }
            });
        }
        EditorCommand::SetRoughness(id, v) => {
            let old = material_roughness(&engine.world.graph, id).unwrap_or(v);
            apply_scalar_field(engine, session, id, old, v, "roughness", |e, v| {
                if let NodeData::Material(m) = &mut e.data {
                    m.roughness = v;
                }
            });
        }
        EditorCommand::SetOpacity(id, v) => {
            let old = material_opacity(&engine.world.graph, id).unwrap_or(v);
            apply_scalar_field(engine, session, id, old, v, "opacity", |e, v| {
                if let NodeData::Material(m) = &mut e.data {
                    m.opacity = v;
                }
            });
        }
        EditorCommand::SetLightColor(id, c) => {
            apply_vec3_light(engine, session, id, c);
        }
        EditorCommand::SetLightIntensity(id, v) => {
            let old = light_intensity(&engine.world.graph, id).unwrap_or(v);
            apply_scalar_field(engine, session, id, old, v, "intensity", |e, v| {
                match &mut e.data {
                    NodeData::DirectionalLight(l) => l.intensity = v,
                    NodeData::PointLight(l) => l.intensity = v,
                    NodeData::SpotLight(l) => l.intensity = v,
                    NodeData::AreaLight(l) => l.intensity = v,
                    NodeData::HemisphereLight(l) => l.intensity = v,
                    _ => {}
                }
            });
        }
        EditorCommand::SetLightDirection(id, d) => {
            let new = Vec3::from_array(d);
            let old = light_direction(&engine.world.graph, id).unwrap_or(new);
            apply_scalar_field(engine, session, id, old, new, "direction", |e, v| {
                match &mut e.data {
                    NodeData::DirectionalLight(l) => l.direction = v,
                    NodeData::SpotLight(l) => l.direction = v,
                    NodeData::HemisphereLight(l) => l.direction = v,
                    _ => {}
                }
            });
        }
        EditorCommand::SetCameraFov(id, v) => {
            let old = camera_fov(&engine.world.graph, id).unwrap_or(v);
            apply_scalar_field(engine, session, id, old, v, "fov", |e, v| {
                if let NodeData::PerspectiveCamera(c) = &mut e.data {
                    c.fov = v;
                }
            });
        }
        EditorCommand::SetCameraNear(id, v) => {
            let old = camera_near(&engine.world.graph, id).unwrap_or(v);
            apply_scalar_field(engine, session, id, old, v, "near", |e, v| {
                match &mut e.data {
                    NodeData::PerspectiveCamera(c) => c.near = v,
                    NodeData::OrthographicCamera(c) => c.near = v,
                    _ => {}
                }
            });
        }
        EditorCommand::SetCameraFar(id, v) => {
            let old = camera_far(&engine.world.graph, id).unwrap_or(v);
            apply_scalar_field(engine, session, id, old, v, "far", |e, v| {
                match &mut e.data {
                    NodeData::PerspectiveCamera(c) => c.far = v,
                    NodeData::OrthographicCamera(c) => c.far = v,
                    _ => {}
                }
            });
        }
        EditorCommand::SetCameraReverseDepth(id, v) => {
            let old = camera_reverse_depth(&engine.world.graph, id).unwrap_or(v);
            apply_scalar_field(engine, session, id, old, v, "reverse_depth", |e, v| {
                match &mut e.data {
                    NodeData::PerspectiveCamera(c) => c.reverse_depth = v,
                    NodeData::OrthographicCamera(c) => c.reverse_depth = v,
                    _ => {}
                }
            });
        }
        EditorCommand::SetOrthoHeight(id, v) => {
            let old = ortho_height(&engine.world.graph, id).unwrap_or(v);
            apply_scalar_field(engine, session, id, old, v, "height", |e, v| {
                if let NodeData::OrthographicCamera(c) = &mut e.data {
                    c.height = v;
                }
            });
        }
        EditorCommand::SetNodeField {
            node,
            field_index,
            value,
        } => {
            let old = read_node_field(&engine.world.graph, node, field_index)
                .unwrap_or_else(|| value.clone());
            session.execute_edit(
                Box::new(NodeFieldCommand {
                    node,
                    field_index,
                    old,
                    new: value,
                }),
                &mut engine.world.graph,
            );
        }
        EditorCommand::CreateNode { node_type, parent } => {
            let data = node_data_for_type(node_type);
            session.execute_edit(
                Box::new(CreateNodeCommand::new(NodeId::default(), parent, data)),
                &mut engine.world.graph,
            );
        }
        EditorCommand::DeleteNode(id) => {
            session.execute_edit(
                Box::new(DeleteNodeCommand::new(id, &engine.world.graph)),
                &mut engine.world.graph,
            );
            engine.world.graph.clear_selection();
            rc3d_engine_api::sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
        }
        EditorCommand::DuplicateNode(id) => {
            duplicate_node(engine, session, id);
        }
        EditorCommand::ReparentNodes {
            ids,
            parent,
            index,
        } => {
            session.execute_edit(
                Box::new(ReparentNodesCommand::new(ids, parent, index)),
                &mut engine.world.graph,
            );
        }
        EditorCommand::RenameNode(id, name) => {
            engine.world.graph.set_name(id, name);
            session.dirty = true;
        }
        EditorCommand::SaveBookmark(slot) => {
            engine.controller.save_bookmark(slot, bookmark_name(slot));
        }
        EditorCommand::RecallBookmark(slot) => {
            engine.controller.recall_bookmark(slot);
        }
        EditorCommand::SetMarkupTool(tool) => {
            interaction.markup.set_tool(tool);
        }
        EditorCommand::MarkupMouseDown { screen_pos } => {
            markup_down(engine, interaction, screen_pos);
        }
        EditorCommand::MarkupMouseMove { screen_pos } => {
            interaction
                .markup
                .on_mouse_move(rc3d_core::math::Vec2::from_array(screen_pos));
        }
        EditorCommand::MarkupMouseUp { screen_pos } => {
            markup_up(engine, session, interaction, screen_pos);
        }
        EditorCommand::ClearAllMarkup { node } => {
            if let Some(e) = engine.world.graph.get_mut(node) {
                if let NodeData::Markup(m) = &mut e.data {
                    m.elements.clear();
                    session.dirty = true;
                }
            }
        }
    }
}

#[derive(Debug)]
struct NodeFieldCommand {
    node: NodeId,
    field_index: u16,
    old: FieldValue,
    new: FieldValue,
}

impl Command for NodeFieldCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        write_node_field(graph, self.node, self.field_index, &self.new);
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        write_node_field(graph, self.node, self.field_index, &self.old);
    }
    fn description(&self) -> &str {
        "SetField"
    }
}

fn apply_scalar_field<T: Clone + std::fmt::Debug + Send + Sync + 'static>(
    engine: &mut Engine,
    session: &mut EditorSession,
    node: NodeId,
    old_value: T,
    new_value: T,
    desc: &str,
    apply: impl Fn(&mut rc3d_scene::NodeEntry, T) + Send + Sync + 'static,
) {
    session.execute_edit(
        Box::new(SetFieldCommand::new(
            node,
            old_value,
            new_value,
            desc,
            apply,
        )),
        &mut engine.world.graph,
    );
}

fn mutate_material(
    engine: &mut Engine,
    session: &mut EditorSession,
    id: NodeId,
    c: [f32; 3],
    apply: impl Fn(&mut rc3d_scene::node_data::MaterialNode, [f32; 3]) + Send + Sync + 'static,
) {
    let old = if let Some(e) = engine.world.graph.get(id) {
        if let NodeData::Material(m) = &e.data {
            m.base_color.to_array()
        } else {
            c
        }
    } else {
        c
    };
    session.execute_edit(
        Box::new(SetFieldCommand::new(
            id,
            old,
            c,
            "base_color",
            move |entry, v| {
                if let NodeData::Material(m) = &mut entry.data {
                    apply(m, v);
                }
            },
        )),
        &mut engine.world.graph,
    );
}

fn apply_vec3_light(engine: &mut Engine, session: &mut EditorSession, id: NodeId, c: [f32; 3]) {
    let new = Vec3::from_array(c);
    let old = light_color(&engine.world.graph, id).unwrap_or(new);
    apply_scalar_field(engine, session, id, old, new, "color", |e, v| {
        match &mut e.data {
            NodeData::DirectionalLight(l) => l.color = v,
            NodeData::PointLight(l) => l.color = v,
            NodeData::SpotLight(l) => l.color = v,
            NodeData::AreaLight(l) => l.color = v,
            _ => {}
        }
    });
}

fn apply_to_selected_or_roots(
    engine: &mut Engine,
    mut f: impl FnMut(&mut SceneGraph, NodeId),
) {
    let selected: Vec<NodeId> = engine.world.graph.selected_nodes().iter().copied().collect();
    let targets = if selected.is_empty() {
        engine.world.graph.roots().to_vec()
    } else {
        selected
    };
    for id in targets {
        f(&mut engine.world.graph, id);
    }
}

fn display_mode_from_editor(mode: EditorDisplayMode) -> DisplayMode {
    match mode {
        EditorDisplayMode::Wireframe => DisplayMode::Wireframe,
        EditorDisplayMode::Shaded => DisplayMode::Shaded,
        EditorDisplayMode::ShadedWithEdges => DisplayMode::ShadedWithEdges,
        EditorDisplayMode::HiddenLine => DisplayMode::HiddenLine,
        EditorDisplayMode::Flat => DisplayMode::Flat,
        EditorDisplayMode::FlatWithEdge => DisplayMode::FlatWithEdge,
    }
}

fn adaptive_from_editor(mode: AdaptiveQualityMode) -> AdaptiveControl {
    match mode {
        AdaptiveQualityMode::Off => AdaptiveControl::Disabled,
        AdaptiveQualityMode::On => AdaptiveControl::Dynamic {
            allow_downgrade: true,
        },
        AdaptiveQualityMode::AutoIdleLock => AdaptiveControl::Locked,
    }
}

fn duplicate_node(engine: &mut Engine, session: &mut EditorSession, id: NodeId) {
    let Some(entry) = engine.world.graph.get(id) else {
        return;
    };
    let data = entry.data.clone();
    let parent = entry.parent;
    session.execute_edit(
        Box::new(CreateNodeCommand::new(NodeId::default(), parent, data)),
        &mut engine.world.graph,
    );
}

fn import_path(engine: &mut Engine, path: &Path) {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "json" => match document::load_native_scene(path) {
            Ok(graph) => engine.load_scene(graph),
            Err(e) => log::warn!("scene load failed: {e}"),
        },
        _ => match engine.import(path) {
            Ok(_) => {
                engine.world.collector.invalidate_mesh_cache();
            }
            Err(e) => log::warn!("import failed: {e}"),
        },
    }
}

fn save_bound_scene(engine: &Engine, session: &mut EditorSession, path: &Path) -> bool {
    match document::save_native_scene(engine.scene(), path) {
        Ok(()) => {
            session.dirty = false;
            true
        }
        Err(e) => {
            log::warn!("save scene failed: {e}");
            false
        }
    }
}

fn install_scene(
    engine: &mut Engine,
    session: &mut EditorSession,
    interaction: &mut EditorInteractionState,
    graph: SceneGraph,
    path: Option<std::path::PathBuf>,
) {
    engine.load_scene(graph);
    engine.controller = CameraController::new(Vec3::ZERO, 10.0);
    engine.hidden_nodes.clear();
    *interaction = EditorInteractionState::default();
    session.history.clear();
    session.document_path = path;
    session.dirty = false;
    rc3d_engine_api::sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
}

fn export_scene_json(graph: &SceneGraph, path: &Path) {
    match serde_json::to_string_pretty(graph) {
        Ok(text) => {
            if let Err(e) = std::fs::write(path, text) {
                log::warn!("export failed: {e}");
            }
        }
        Err(e) => log::warn!("serialize scene failed: {e}"),
    }
}

fn export_diagnostics(engine: &Engine, path: &Path) {
    let text = engine
        .renderer
        .as_ref()
        .and_then(|r| r.last_diagnostics())
        .map(|d| format!("{d:#?}"))
        .unwrap_or_else(|| "{}".into());
    if let Err(e) = std::fs::write(path, text) {
        log::warn!("diagnostics export failed: {e}");
    }
}

fn export_screenshot(engine: &mut Engine, path: &Path) {
    let Some(r) = engine.renderer.as_mut() else {
        return;
    };
    let (w, h) = r.surface_size();
    let (w, h, pixels) = r.render_to_image(
        &engine.world.cached_draw_calls,
        &engine.world.graph,
        w.max(1),
        h.max(1),
    );
    save_rgba_png(path, w, h, pixels);
}

fn export_quad_pack(engine: &mut Engine, path: &Path) {
    let (w, h) = engine
        .renderer
        .as_ref()
        .map(|r| r.surface_size())
        .unwrap_or((1280, 720));
    let (w, h, pixels) = engine.render_quad_pack_image(w.max(1), h.max(1));
    save_rgba_png(path, w, h, pixels);
}

fn save_rgba_png(path: &Path, w: u32, h: u32, pixels: Vec<u8>) {
    match image::RgbaImage::from_raw(w, h, pixels) {
        Some(img) => {
            if let Err(e) = img.save(path) {
                log::warn!("image save failed: {e}");
            }
        }
        None => log::warn!("image encode failed"),
    }
}

fn bookmark_name(slot: usize) -> &'static str {
    match slot {
        0 => "1",
        1 => "2",
        2 => "3",
        3 => "4",
        4 => "5",
        5 => "6",
        6 => "7",
        7 => "8",
        _ => "9",
    }
}

fn markup_down(engine: &mut Engine, interaction: &mut EditorInteractionState, screen_pos: [f32; 2]) {
    let root = engine.world.graph.roots().first().copied();
    if let Some(root) = root {
        interaction
            .markup
            .ensure_target_node(&mut engine.world.graph, root);
    }
    let _ = interaction
        .markup
        .on_mouse_down(rc3d_core::math::Vec2::from_array(screen_pos), &mut engine.world.graph);
}

fn markup_up(
    engine: &mut Engine,
    session: &mut EditorSession,
    interaction: &mut EditorInteractionState,
    screen_pos: [f32; 2],
) {
    if let Some(el) = interaction
        .markup
        .on_mouse_up(rc3d_core::math::Vec2::from_array(screen_pos))
    {
        if let Some(id) = interaction.markup.target_node {
            if let Some(e) = engine.world.graph.get_mut(id) {
                if let NodeData::Markup(m) = &mut e.data {
                    m.elements.push(el);
                    session.dirty = true;
                }
            }
        }
    }
}

fn material_metallic(graph: &SceneGraph, id: NodeId) -> Option<f32> {
    match &graph.get(id)?.data {
        NodeData::Material(m) => Some(m.metallic),
        _ => None,
    }
}

fn material_roughness(graph: &SceneGraph, id: NodeId) -> Option<f32> {
    match &graph.get(id)?.data {
        NodeData::Material(m) => Some(m.roughness),
        _ => None,
    }
}

fn material_opacity(graph: &SceneGraph, id: NodeId) -> Option<f32> {
    match &graph.get(id)?.data {
        NodeData::Material(m) => Some(m.opacity),
        _ => None,
    }
}

fn light_intensity(graph: &SceneGraph, id: NodeId) -> Option<f32> {
    match &graph.get(id)?.data {
        NodeData::DirectionalLight(l) => Some(l.intensity),
        NodeData::PointLight(l) => Some(l.intensity),
        NodeData::SpotLight(l) => Some(l.intensity),
        NodeData::AreaLight(l) => Some(l.intensity),
        NodeData::HemisphereLight(l) => Some(l.intensity),
        _ => None,
    }
}

fn light_direction(graph: &SceneGraph, id: NodeId) -> Option<Vec3> {
    match &graph.get(id)?.data {
        NodeData::DirectionalLight(l) => Some(l.direction),
        NodeData::SpotLight(l) => Some(l.direction),
        NodeData::HemisphereLight(l) => Some(l.direction),
        _ => None,
    }
}

fn light_color(graph: &SceneGraph, id: NodeId) -> Option<Vec3> {
    match &graph.get(id)?.data {
        NodeData::DirectionalLight(l) => Some(l.color),
        NodeData::PointLight(l) => Some(l.color),
        NodeData::SpotLight(l) => Some(l.color),
        NodeData::AreaLight(l) => Some(l.color),
        _ => None,
    }
}

fn camera_fov(graph: &SceneGraph, id: NodeId) -> Option<f32> {
    match &graph.get(id)?.data {
        NodeData::PerspectiveCamera(c) => Some(c.fov),
        _ => None,
    }
}

fn camera_near(graph: &SceneGraph, id: NodeId) -> Option<f32> {
    match &graph.get(id)?.data {
        NodeData::PerspectiveCamera(c) => Some(c.near),
        NodeData::OrthographicCamera(c) => Some(c.near),
        _ => None,
    }
}

fn camera_far(graph: &SceneGraph, id: NodeId) -> Option<f32> {
    match &graph.get(id)?.data {
        NodeData::PerspectiveCamera(c) => Some(c.far),
        NodeData::OrthographicCamera(c) => Some(c.far),
        _ => None,
    }
}

fn camera_reverse_depth(graph: &SceneGraph, id: NodeId) -> Option<bool> {
    match &graph.get(id)?.data {
        NodeData::PerspectiveCamera(c) => Some(c.reverse_depth),
        NodeData::OrthographicCamera(c) => Some(c.reverse_depth),
        _ => None,
    }
}

fn ortho_height(graph: &SceneGraph, id: NodeId) -> Option<f32> {
    match &graph.get(id)?.data {
        NodeData::OrthographicCamera(c) => Some(c.height),
        _ => None,
    }
}
