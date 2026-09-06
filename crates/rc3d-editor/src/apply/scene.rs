use rc3d_actions::{
    Command, CreateNodeCommand, DeleteNodeCommand, ReparentNodesCommand, SetFieldCommand,
    SetRotationCommand, SetScaleCommand, SetTranslationCommand,
};
use rc3d_core::math::{Mat4, Quat, Vec3};
use rc3d_core::NodeId;
use rc3d_engine_api::Engine;
use rc3d_fields::FieldValue;
use rc3d_scene::node_data::NodeData;
use rc3d_scene::{read_node_field, write_node_field, SceneGraph};

use crate::commands::EditorCommand;
use crate::context::EditorInteractionState;
use crate::node_factory::node_data_for_type;

use super::EditorSession;

pub(super) fn apply(
    engine: &mut Engine,
    _interaction: &mut EditorInteractionState,
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
        EditorCommand::SetUiTheme(theme) => {
            session.ui_theme = theme;
        }
        EditorCommand::SetUiLocale(locale) => {
            session.ui_locale = locale;
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
            apply_scalar_field(
                engine,
                session,
                id,
                old,
                v,
                "intensity",
                |e, v| match &mut e.data {
                    NodeData::DirectionalLight(l) => l.intensity = v,
                    NodeData::PointLight(l) => l.intensity = v,
                    NodeData::SpotLight(l) => l.intensity = v,
                    NodeData::AreaLight(l) => l.intensity = v,
                    NodeData::HemisphereLight(l) => l.intensity = v,
                    _ => {}
                },
            );
        }
        EditorCommand::SetLightDirection(id, d) => {
            let new = Vec3::from_array(d);
            let old = light_direction(&engine.world.graph, id).unwrap_or(new);
            apply_scalar_field(
                engine,
                session,
                id,
                old,
                new,
                "direction",
                |e, v| match &mut e.data {
                    NodeData::DirectionalLight(l) => l.direction = v,
                    NodeData::SpotLight(l) => l.direction = v,
                    NodeData::HemisphereLight(l) => l.direction = v,
                    _ => {}
                },
            );
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
            apply_scalar_field(
                engine,
                session,
                id,
                old,
                v,
                "reverse_depth",
                |e, v| match &mut e.data {
                    NodeData::PerspectiveCamera(c) => c.reverse_depth = v,
                    NodeData::OrthographicCamera(c) => c.reverse_depth = v,
                    _ => {}
                },
            );
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
        EditorCommand::ReparentNodes { ids, parent, index } => {
            session.execute_edit(
                Box::new(ReparentNodesCommand::new(ids, parent, index)),
                &mut engine.world.graph,
            );
        }
        EditorCommand::RenameNode(id, name) => {
            engine.world.graph.set_name(id, name);
            session.dirty = true;
        }
        _ => {}
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
            node, old_value, new_value, desc, apply,
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
    apply_scalar_field(
        engine,
        session,
        id,
        old,
        new,
        "color",
        |e, v| match &mut e.data {
            NodeData::DirectionalLight(l) => l.color = v,
            NodeData::PointLight(l) => l.color = v,
            NodeData::SpotLight(l) => l.color = v,
            NodeData::AreaLight(l) => l.color = v,
            _ => {}
        },
    );
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
