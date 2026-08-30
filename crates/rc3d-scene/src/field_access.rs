//! Read/write typed node fields by `FieldDescriptor` index.

use rc3d_core::math::{Vec2, Vec4};
use rc3d_core::NodeId;
use rc3d_fields::FieldValue;

use crate::node_data::{
    DraggerKind, FontStyle, ManipMode, ManipSpace, NodeData, RotationAxis, StereoMode,
};
use crate::node_entry::dirty_flags;
use crate::SceneGraph;

pub fn field_as_f32(v: &FieldValue) -> Option<f32> {
    match v {
        FieldValue::Float(x) => Some(*x),
        FieldValue::Float64(x) => Some(*x as f32),
        FieldValue::Int32(x) => Some(*x as f32),
        FieldValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

pub fn field_as_i32(v: &FieldValue) -> Option<i32> {
    match v {
        FieldValue::Int32(x) => Some(*x),
        FieldValue::Float(x) => Some(*x as i32),
        FieldValue::Float64(x) => Some(*x as i32),
        FieldValue::Bool(b) => Some(i32::from(*b)),
        _ => None,
    }
}

pub fn field_as_bool(v: &FieldValue) -> Option<bool> {
    match v {
        FieldValue::Bool(b) => Some(*b),
        FieldValue::Int32(x) => Some(*x != 0),
        FieldValue::Float(x) => Some(*x != 0.0),
        FieldValue::Float64(x) => Some(*x != 0.0),
        _ => None,
    }
}

pub fn read_node_field(graph: &SceneGraph, node: NodeId, field_index: u16) -> Option<FieldValue> {
    let entry = graph.get(node)?;
    match (&entry.data, field_index) {
        (NodeData::Transform(t), 0) => Some(FieldValue::Vec3f(t.translation)),
        (NodeData::Transform(t), 1) => Some(FieldValue::Mat4f(t.rotation)),
        (NodeData::Transform(t), 2) => Some(FieldValue::Vec3f(t.scale)),
        (NodeData::Transform(t), 3) => Some(FieldValue::Vec3f(t.center)),
        (NodeData::Rotation(r), 0) => Some(FieldValue::Vec3f(r.axis)),
        (NodeData::Rotation(r), 1) => Some(FieldValue::Float(r.angle)),
        (NodeData::RotationXYZ(r), 0) => Some(FieldValue::Int32(rotation_axis_i32(r.axis))),
        (NodeData::RotationXYZ(r), 1) => Some(FieldValue::Float(r.angle)),
        (NodeData::Material(m), 0) => Some(FieldValue::Vec3f(m.diffuse_color)),
        (NodeData::Material(m), 1) => Some(FieldValue::Vec3f(m.specular_color)),
        (NodeData::Material(m), 2) => Some(FieldValue::Float(m.shininess)),
        (NodeData::Material(m), 3) => Some(FieldValue::Float(m.opacity)),
        (NodeData::DirectionalLight(l), 0) => Some(FieldValue::Vec3f(l.direction)),
        (NodeData::DirectionalLight(l), 1) => Some(FieldValue::Vec3f(l.color)),
        (NodeData::DirectionalLight(l), 2) => Some(FieldValue::Float(l.intensity)),
        (NodeData::PointLight(l), 0) => Some(FieldValue::Vec3f(l.location)),
        (NodeData::PointLight(l), 1) => Some(FieldValue::Vec3f(l.color)),
        (NodeData::PointLight(l), 2) => Some(FieldValue::Float(l.intensity)),
        (NodeData::SpotLight(l), 0) => Some(FieldValue::Vec3f(l.location)),
        (NodeData::SpotLight(l), 1) => Some(FieldValue::Vec3f(l.direction)),
        (NodeData::SpotLight(l), 2) => Some(FieldValue::Vec3f(l.color)),
        (NodeData::SpotLight(l), 3) => Some(FieldValue::Float(l.intensity)),
        (NodeData::SpotLight(l), 4) => Some(FieldValue::Float(l.cut_off_angle)),
        (NodeData::SpotLight(l), 5) => Some(FieldValue::Float(l.drop_off_rate)),
        (NodeData::AreaLight(l), 0) => Some(FieldValue::Vec3f(l.color)),
        (NodeData::AreaLight(l), 1) => Some(FieldValue::Float(l.intensity)),
        (NodeData::AreaLight(l), 2) => Some(FieldValue::Float(l.width)),
        (NodeData::AreaLight(l), 3) => Some(FieldValue::Float(l.height)),
        (NodeData::HemisphereLight(l), 0) => Some(FieldValue::Vec3f(l.sky_color)),
        (NodeData::HemisphereLight(l), 1) => Some(FieldValue::Vec3f(l.ground_color)),
        (NodeData::HemisphereLight(l), 2) => Some(FieldValue::Float(l.intensity)),
        (NodeData::HemisphereLight(l), 3) => Some(FieldValue::Vec3f(l.direction)),
        (NodeData::LightProbe(l), 0) => Some(FieldValue::Float(l.intensity)),
        (NodeData::Sprite(s), 0) => Some(FieldValue::Float(s.size)),
        (NodeData::Sprite(s), 1) => Some(FieldValue::Float(s.opacity)),
        (NodeData::Sprite(s), 2) => Some(FieldValue::String(s.texture_path.clone())),
        (NodeData::PerspectiveCamera(c), 0) => Some(FieldValue::Float(c.fov)),
        (NodeData::PerspectiveCamera(c), 1) => Some(FieldValue::Float(c.near)),
        (NodeData::PerspectiveCamera(c), 2) => Some(FieldValue::Float(c.far)),
        (NodeData::PerspectiveCamera(c), 3) => Some(FieldValue::Bool(c.reverse_depth)),
        (NodeData::StereoCamera(c), 0) => Some(FieldValue::Float(c.interocular_distance)),
        (NodeData::StereoCamera(c), 1) => Some(FieldValue::Float(c.convergence_distance)),
        (NodeData::StereoCamera(c), 2) => Some(FieldValue::Int32(stereo_mode_i32(c.mode))),
        (NodeData::OrthographicCamera(c), 0) => Some(FieldValue::Float(c.height)),
        (NodeData::OrthographicCamera(c), 1) => Some(FieldValue::Float(c.near)),
        (NodeData::OrthographicCamera(c), 2) => Some(FieldValue::Float(c.far)),
        (NodeData::OrthographicCamera(c), 3) => Some(FieldValue::Bool(c.reverse_depth)),
        (NodeData::CubeCamera(c), 0) => Some(FieldValue::Vec3f(c.position)),
        (NodeData::CubeCamera(c), 1) => Some(FieldValue::Float(c.near)),
        (NodeData::CubeCamera(c), 2) => Some(FieldValue::Float(c.far)),
        (NodeData::CubeCamera(c), 3) => Some(FieldValue::Int32(c.resolution as i32)),
        (NodeData::SectionPlane(p), 0) => Some(FieldValue::Vec4f(Vec4::from_array(p.plane))),
        (NodeData::SectionPlane(p), 1) => Some(FieldValue::Bool(p.enabled)),
        (NodeData::SectionPlane(p), 2) => Some(FieldValue::Vec4f(Vec4::from_array(p.cap_color))),
        (NodeData::SectionPlane(p), 3) => Some(FieldValue::Bool(p.cap_enabled)),
        (NodeData::SectionPlane(p), 4) => Some(FieldValue::Bool(p.hatch_enabled)),
        (NodeData::SectionPlane(p), 5) => Some(FieldValue::Float(p.hatch_spacing)),
        (NodeData::SectionPlane(p), 6) => Some(FieldValue::Float(p.hatch_angle_deg)),
        (NodeData::Lod(l), 0) => Some(FieldValue::Int32(l.current_level as i32)),
        (NodeData::Lod(l), 1) => Some(FieldValue::Float(l.range_scale)),
        (NodeData::Switch(s), 0) => Some(FieldValue::Int32(s.which_child)),
        (NodeData::Text2(t), 0) => Some(FieldValue::String(t.string.clone())),
        (NodeData::Text2(t), 1) => Some(FieldValue::Vec2f(Vec2::from_array(t.position))),
        (NodeData::Text2(t), 2) => Some(FieldValue::Float(t.size)),
        (NodeData::Text2(t), 3) => Some(FieldValue::Vec4f(Vec4::from_array(t.color))),
        (NodeData::Text3(t), 0) => Some(FieldValue::String(t.string.clone())),
        (NodeData::Text3(t), 1) => Some(FieldValue::Vec3f(t.position)),
        (NodeData::Text3(t), 2) => Some(FieldValue::Float(t.size)),
        (NodeData::Text3(t), 3) => Some(FieldValue::Vec4f(Vec4::from_array(t.color))),
        (NodeData::Font(f), 0) => Some(FieldValue::String(f.name.clone())),
        (NodeData::Font(f), 1) => Some(FieldValue::Float(f.size)),
        (NodeData::Font(f), 2) => Some(FieldValue::Int32(font_style_i32(f.style))),
        (NodeData::EventCallback(e), 0) => Some(FieldValue::Bool(e.enabled)),
        (NodeData::EventCallback(e), 1) => Some(FieldValue::Bool(e.consume)),
        (NodeData::PickStyle(p), 0) => Some(FieldValue::Bool(p.pickable)),
        (NodeData::Markup(m), 0) => Some(FieldValue::Bool(m.visible)),
        (NodeData::Markup(m), 1) => Some(FieldValue::String(m.layer_name.clone())),
        (NodeData::Measurement(m), 0) => Some(FieldValue::Float(m.value)),
        (NodeData::Measurement(m), 1) => Some(FieldValue::String(m.label.clone())),
        (NodeData::Measurement(m), 2) => Some(FieldValue::Vec4f(Vec4::from_array(m.color))),
        (NodeData::Decal(d), 0) => Some(FieldValue::Float(d.opacity)),
        (NodeData::TransformManip(m), 0) => Some(FieldValue::Int32(manip_mode_i32(m.mode))),
        (NodeData::TransformManip(m), 1) => Some(FieldValue::Int32(manip_space_i32(m.space))),
        (NodeData::TransformManip(m), 2) => Some(FieldValue::Bool(m.enabled)),
        (NodeData::TransformManip(m), 3) => Some(FieldValue::Float(m.size)),
        (NodeData::Dragger(d), 0) => Some(FieldValue::Int32(dragger_kind_i32(d.kind))),
        (NodeData::Dragger(d), 1) => Some(FieldValue::Bool(d.enabled)),
        _ => None,
    }
}

pub fn write_node_field(graph: &mut SceneGraph, node: NodeId, field_index: u16, value: &FieldValue) {
    let Some(entry) = graph.get_mut(node) else {
        return;
    };
    match (&mut entry.data, field_index) {
        (NodeData::Transform(t), 0) => {
            if let FieldValue::Vec3f(v) = value {
                t.translation = *v;
                entry.dirty_flags |= dirty_flags::TRANSFORM;
            }
        }
        (NodeData::Transform(t), 1) => {
            if let FieldValue::Mat4f(m) = value {
                t.rotation = *m;
                entry.dirty_flags |= dirty_flags::TRANSFORM;
            }
        }
        (NodeData::Transform(t), 2) => {
            if let FieldValue::Vec3f(v) = value {
                t.scale = *v;
                entry.dirty_flags |= dirty_flags::TRANSFORM;
            }
        }
        (NodeData::Transform(t), 3) => {
            if let FieldValue::Vec3f(v) = value {
                t.center = *v;
                entry.dirty_flags |= dirty_flags::TRANSFORM;
            }
        }
        (NodeData::Rotation(r), 0) => {
            if let FieldValue::Vec3f(v) = value {
                r.axis = *v;
                entry.dirty_flags |= dirty_flags::TRANSFORM;
            }
        }
        (NodeData::Rotation(r), 1) => {
            if let Some(a) = field_as_f32(value) {
                r.angle = a;
                entry.dirty_flags |= dirty_flags::TRANSFORM;
            }
        }
        (NodeData::RotationXYZ(r), 0) => {
            if let Some(i) = field_as_i32(value) {
                r.axis = rotation_axis_from_i32(i);
                entry.dirty_flags |= dirty_flags::TRANSFORM;
            }
        }
        (NodeData::RotationXYZ(r), 1) => {
            if let Some(a) = field_as_f32(value) {
                r.angle = a;
                entry.dirty_flags |= dirty_flags::TRANSFORM;
            }
        }
        (NodeData::Material(m), 0) => {
            if let FieldValue::Vec3f(v) = value {
                m.diffuse_color = *v;
                m.base_color = *v;
                entry.dirty_flags |= dirty_flags::MATERIAL;
            }
        }
        (NodeData::Material(m), 1) => {
            if let FieldValue::Vec3f(v) = value {
                m.specular_color = *v;
                entry.dirty_flags |= dirty_flags::MATERIAL;
            }
        }
        (NodeData::Material(m), 2) => {
            if let Some(s) = field_as_f32(value) {
                m.shininess = s;
                entry.dirty_flags |= dirty_flags::MATERIAL;
            }
        }
        (NodeData::Material(m), 3) => {
            if let Some(o) = field_as_f32(value) {
                m.opacity = o;
                entry.dirty_flags |= dirty_flags::MATERIAL;
            }
        }
        (NodeData::DirectionalLight(l), 0) => {
            if let FieldValue::Vec3f(v) = value {
                l.direction = *v;
            }
        }
        (NodeData::DirectionalLight(l), 1) => {
            if let FieldValue::Vec3f(v) = value {
                l.color = *v;
            }
        }
        (NodeData::DirectionalLight(l), 2) => {
            if let Some(i) = field_as_f32(value) {
                l.intensity = i;
            }
        }
        (NodeData::PointLight(l), 0) => {
            if let FieldValue::Vec3f(v) = value {
                l.location = *v;
            }
        }
        (NodeData::PointLight(l), 1) => {
            if let FieldValue::Vec3f(v) = value {
                l.color = *v;
            }
        }
        (NodeData::PointLight(l), 2) => {
            if let Some(i) = field_as_f32(value) {
                l.intensity = i;
            }
        }
        (NodeData::SpotLight(l), 0) => {
            if let FieldValue::Vec3f(v) = value {
                l.location = *v;
            }
        }
        (NodeData::SpotLight(l), 1) => {
            if let FieldValue::Vec3f(v) = value {
                l.direction = *v;
            }
        }
        (NodeData::SpotLight(l), 2) => {
            if let FieldValue::Vec3f(v) = value {
                l.color = *v;
            }
        }
        (NodeData::SpotLight(l), 3) => {
            if let Some(i) = field_as_f32(value) {
                l.intensity = i;
            }
        }
        (NodeData::SpotLight(l), 4) => {
            if let Some(a) = field_as_f32(value) {
                l.cut_off_angle = a;
            }
        }
        (NodeData::SpotLight(l), 5) => {
            if let Some(a) = field_as_f32(value) {
                l.drop_off_rate = a;
            }
        }
        (NodeData::AreaLight(l), 0) => {
            if let FieldValue::Vec3f(v) = value {
                l.color = *v;
            }
        }
        (NodeData::AreaLight(l), 1) => {
            if let Some(i) = field_as_f32(value) {
                l.intensity = i;
            }
        }
        (NodeData::AreaLight(l), 2) => {
            if let Some(w) = field_as_f32(value) {
                l.width = w;
            }
        }
        (NodeData::AreaLight(l), 3) => {
            if let Some(h) = field_as_f32(value) {
                l.height = h;
            }
        }
        (NodeData::HemisphereLight(l), 0) => {
            if let FieldValue::Vec3f(v) = value {
                l.sky_color = *v;
            }
        }
        (NodeData::HemisphereLight(l), 1) => {
            if let FieldValue::Vec3f(v) = value {
                l.ground_color = *v;
            }
        }
        (NodeData::HemisphereLight(l), 2) => {
            if let Some(i) = field_as_f32(value) {
                l.intensity = i;
            }
        }
        (NodeData::HemisphereLight(l), 3) => {
            if let FieldValue::Vec3f(v) = value {
                l.direction = *v;
            }
        }
        (NodeData::LightProbe(l), 0) => {
            if let Some(i) = field_as_f32(value) {
                l.intensity = i;
            }
        }
        (NodeData::Sprite(s), 0) => {
            if let Some(v) = field_as_f32(value) {
                s.size = v;
            }
        }
        (NodeData::Sprite(s), 1) => {
            if let Some(v) = field_as_f32(value) {
                s.opacity = v;
            }
        }
        (NodeData::Sprite(s), 2) => {
            if let FieldValue::String(p) = value {
                s.texture_path = p.clone();
            }
        }
        (NodeData::PerspectiveCamera(c), 0) => {
            if let Some(v) = field_as_f32(value) {
                c.fov = v;
            }
        }
        (NodeData::PerspectiveCamera(c), 1) => {
            if let Some(v) = field_as_f32(value) {
                c.near = v;
            }
        }
        (NodeData::PerspectiveCamera(c), 2) => {
            if let Some(v) = field_as_f32(value) {
                c.far = v;
            }
        }
        (NodeData::PerspectiveCamera(c), 3) => {
            if let Some(b) = field_as_bool(value) {
                c.reverse_depth = b;
            }
        }
        (NodeData::StereoCamera(c), 0) => {
            if let Some(v) = field_as_f32(value) {
                c.interocular_distance = v;
            }
        }
        (NodeData::StereoCamera(c), 1) => {
            if let Some(v) = field_as_f32(value) {
                c.convergence_distance = v;
            }
        }
        (NodeData::StereoCamera(c), 2) => {
            if let Some(i) = field_as_i32(value) {
                c.mode = stereo_mode_from_i32(i);
            }
        }
        (NodeData::OrthographicCamera(c), 0) => {
            if let Some(v) = field_as_f32(value) {
                c.height = v;
            }
        }
        (NodeData::OrthographicCamera(c), 1) => {
            if let Some(v) = field_as_f32(value) {
                c.near = v;
            }
        }
        (NodeData::OrthographicCamera(c), 2) => {
            if let Some(v) = field_as_f32(value) {
                c.far = v;
            }
        }
        (NodeData::OrthographicCamera(c), 3) => {
            if let Some(b) = field_as_bool(value) {
                c.reverse_depth = b;
            }
        }
        (NodeData::CubeCamera(c), 0) => {
            if let FieldValue::Vec3f(v) = value {
                c.position = *v;
            }
        }
        (NodeData::CubeCamera(c), 1) => {
            if let Some(v) = field_as_f32(value) {
                c.near = v;
            }
        }
        (NodeData::CubeCamera(c), 2) => {
            if let Some(v) = field_as_f32(value) {
                c.far = v;
            }
        }
        (NodeData::CubeCamera(c), 3) => {
            if let Some(i) = field_as_i32(value) {
                c.resolution = i.max(32) as u32;
            }
        }
        (NodeData::SectionPlane(p), 0) => {
            if let FieldValue::Vec4f(v) = value {
                p.plane = v.to_array();
                entry.dirty_flags |= dirty_flags::GEOMETRY;
            }
        }
        (NodeData::SectionPlane(p), 1) => {
            if let Some(b) = field_as_bool(value) {
                p.enabled = b;
                entry.dirty_flags |= dirty_flags::GEOMETRY;
            }
        }
        (NodeData::SectionPlane(p), 2) => {
            if let FieldValue::Vec4f(v) = value {
                p.cap_color = v.to_array();
            }
        }
        (NodeData::SectionPlane(p), 3) => {
            if let Some(b) = field_as_bool(value) {
                p.cap_enabled = b;
            }
        }
        (NodeData::SectionPlane(p), 4) => {
            if let Some(b) = field_as_bool(value) {
                p.hatch_enabled = b;
            }
        }
        (NodeData::SectionPlane(p), 5) => {
            if let Some(v) = field_as_f32(value) {
                p.hatch_spacing = v;
            }
        }
        (NodeData::SectionPlane(p), 6) => {
            if let Some(v) = field_as_f32(value) {
                p.hatch_angle_deg = v;
            }
        }
        (NodeData::Lod(l), 0) => {
            if let Some(i) = field_as_i32(value) {
                l.current_level = i.max(0) as usize;
                entry.dirty_flags |= dirty_flags::CHILDREN;
            }
        }
        (NodeData::Lod(l), 1) => {
            if let Some(v) = field_as_f32(value) {
                l.range_scale = v.max(1.0e-4);
            }
        }
        (NodeData::Switch(s), 0) => {
            if let Some(i) = field_as_i32(value) {
                s.which_child = i;
                entry.dirty_flags |= dirty_flags::CHILDREN;
            }
        }
        (NodeData::Text2(t), 0) => {
            if let FieldValue::String(s) = value {
                t.string = s.clone();
            }
        }
        (NodeData::Text2(t), 1) => {
            if let FieldValue::Vec2f(v) = value {
                t.position = v.to_array();
            }
        }
        (NodeData::Text2(t), 2) => {
            if let Some(v) = field_as_f32(value) {
                t.size = v;
            }
        }
        (NodeData::Text2(t), 3) => {
            if let FieldValue::Vec4f(v) = value {
                t.color = v.to_array();
            }
        }
        (NodeData::Text3(t), 0) => {
            if let FieldValue::String(s) = value {
                t.string = s.clone();
            }
        }
        (NodeData::Text3(t), 1) => {
            if let FieldValue::Vec3f(v) = value {
                t.position = *v;
            }
        }
        (NodeData::Text3(t), 2) => {
            if let Some(v) = field_as_f32(value) {
                t.size = v;
            }
        }
        (NodeData::Text3(t), 3) => {
            if let FieldValue::Vec4f(v) = value {
                t.color = v.to_array();
            }
        }
        (NodeData::Font(f), 0) => {
            if let FieldValue::String(s) = value {
                f.name = s.clone();
            }
        }
        (NodeData::Font(f), 1) => {
            if let Some(v) = field_as_f32(value) {
                f.size = v;
            }
        }
        (NodeData::Font(f), 2) => {
            if let Some(i) = field_as_i32(value) {
                f.style = font_style_from_i32(i);
            }
        }
        (NodeData::EventCallback(e), 0) => {
            if let Some(b) = field_as_bool(value) {
                e.enabled = b;
            }
        }
        (NodeData::EventCallback(e), 1) => {
            if let Some(b) = field_as_bool(value) {
                e.consume = b;
            }
        }
        (NodeData::PickStyle(p), 0) => {
            if let Some(b) = field_as_bool(value) {
                p.pickable = b;
            }
        }
        (NodeData::Markup(m), 0) => {
            if let Some(b) = field_as_bool(value) {
                m.visible = b;
            }
        }
        (NodeData::Markup(m), 1) => {
            if let FieldValue::String(s) = value {
                m.layer_name = s.clone();
            }
        }
        (NodeData::Measurement(m), 0) => {
            if let Some(v) = field_as_f32(value) {
                m.value = v;
            }
        }
        (NodeData::Measurement(m), 1) => {
            if let FieldValue::String(s) = value {
                m.label = s.clone();
            }
        }
        (NodeData::Measurement(m), 2) => {
            if let FieldValue::Vec4f(v) = value {
                m.color = v.to_array();
            }
        }
        (NodeData::Decal(d), 0) => {
            if let Some(v) = field_as_f32(value) {
                d.opacity = v;
            }
        }
        (NodeData::TransformManip(m), 0) => {
            if let Some(i) = field_as_i32(value) {
                m.mode = manip_mode_from_i32(i);
            }
        }
        (NodeData::TransformManip(m), 1) => {
            if let Some(i) = field_as_i32(value) {
                m.space = manip_space_from_i32(i);
            }
        }
        (NodeData::TransformManip(m), 2) => {
            if let Some(b) = field_as_bool(value) {
                m.enabled = b;
            }
        }
        (NodeData::TransformManip(m), 3) => {
            if let Some(v) = field_as_f32(value) {
                m.size = v;
            }
        }
        (NodeData::Dragger(d), 0) => {
            if let Some(i) = field_as_i32(value) {
                d.kind = dragger_kind_from_i32(i);
            }
        }
        (NodeData::Dragger(d), 1) => {
            if let Some(b) = field_as_bool(value) {
                d.enabled = b;
            }
        }
        _ => {}
    }
}

fn rotation_axis_i32(axis: RotationAxis) -> i32 {
    match axis {
        RotationAxis::X => 0,
        RotationAxis::Y => 1,
        RotationAxis::Z => 2,
    }
}

fn rotation_axis_from_i32(v: i32) -> RotationAxis {
    match v {
        1 => RotationAxis::Y,
        2 => RotationAxis::Z,
        _ => RotationAxis::X,
    }
}

fn font_style_i32(style: FontStyle) -> i32 {
    match style {
        FontStyle::Sans => 0,
        FontStyle::Serif => 1,
        FontStyle::Typewriter => 2,
    }
}

fn font_style_from_i32(v: i32) -> FontStyle {
    match v {
        1 => FontStyle::Serif,
        2 => FontStyle::Typewriter,
        _ => FontStyle::Sans,
    }
}

fn stereo_mode_i32(mode: StereoMode) -> i32 {
    match mode {
        StereoMode::SideBySide => 0,
        StereoMode::TopBottom => 1,
        StereoMode::Anaglyph => 2,
    }
}

fn stereo_mode_from_i32(v: i32) -> StereoMode {
    match v {
        1 => StereoMode::TopBottom,
        2 => StereoMode::Anaglyph,
        _ => StereoMode::SideBySide,
    }
}

fn manip_mode_i32(mode: ManipMode) -> i32 {
    match mode {
        ManipMode::Translate => 0,
        ManipMode::Rotate => 1,
        ManipMode::Scale => 2,
    }
}

fn manip_mode_from_i32(v: i32) -> ManipMode {
    match v {
        1 => ManipMode::Rotate,
        2 => ManipMode::Scale,
        _ => ManipMode::Translate,
    }
}

fn manip_space_i32(space: ManipSpace) -> i32 {
    match space {
        ManipSpace::World => 0,
        ManipSpace::Local => 1,
    }
}

fn manip_space_from_i32(v: i32) -> ManipSpace {
    match v {
        1 => ManipSpace::Local,
        _ => ManipSpace::World,
    }
}

fn dragger_kind_i32(kind: DraggerKind) -> i32 {
    match kind {
        DraggerKind::TranslateX => 0,
        DraggerKind::TranslateY => 1,
        DraggerKind::TranslateZ => 2,
        DraggerKind::TranslateXY => 3,
        DraggerKind::TranslateYZ => 4,
        DraggerKind::TranslateZX => 5,
        DraggerKind::RotateX => 6,
        DraggerKind::RotateY => 7,
        DraggerKind::RotateZ => 8,
        DraggerKind::ScaleX => 9,
        DraggerKind::ScaleY => 10,
        DraggerKind::ScaleZ => 11,
        DraggerKind::ScaleUniform => 12,
    }
}

fn dragger_kind_from_i32(v: i32) -> DraggerKind {
    match v {
        1 => DraggerKind::TranslateY,
        2 => DraggerKind::TranslateZ,
        3 => DraggerKind::TranslateXY,
        4 => DraggerKind::TranslateYZ,
        5 => DraggerKind::TranslateZX,
        6 => DraggerKind::RotateX,
        7 => DraggerKind::RotateY,
        8 => DraggerKind::RotateZ,
        9 => DraggerKind::ScaleX,
        10 => DraggerKind::ScaleY,
        11 => DraggerKind::ScaleZ,
        12 => DraggerKind::ScaleUniform,
        _ => DraggerKind::TranslateX,
    }
}
