//! Read/write typed node fields by `FieldDescriptor` index.

use rc3d_core::NodeId;
use rc3d_fields::FieldValue;

use crate::node_data::NodeData;
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
        (NodeData::RotationXYZ(r), 1) => Some(FieldValue::Float(r.angle)),
        (NodeData::Switch(s), 0) => Some(FieldValue::Int32(s.which_child)),
        (NodeData::Material(m), 3) => Some(FieldValue::Float(m.opacity)),
        (NodeData::SectionPlane(p), 1) => Some(FieldValue::Bool(p.enabled)),
        _ => None,
    }
}

pub fn write_node_field(graph: &mut SceneGraph, node: NodeId, field_index: u16, value: &FieldValue) {
    let Some(entry) = graph.get_mut(node) else {
        return;
    };
    match (&mut entry.data, field_index, value) {
        (NodeData::Transform(t), 0, FieldValue::Vec3f(v)) => {
            t.translation = *v;
            entry.dirty_flags |= dirty_flags::TRANSFORM;
        }
        (NodeData::Transform(t), 1, FieldValue::Mat4f(m)) => {
            t.rotation = *m;
            entry.dirty_flags |= dirty_flags::TRANSFORM;
        }
        (NodeData::Transform(t), 2, FieldValue::Vec3f(v)) => {
            t.scale = *v;
            entry.dirty_flags |= dirty_flags::TRANSFORM;
        }
        (NodeData::Transform(t), 3, FieldValue::Vec3f(v)) => {
            t.center = *v;
            entry.dirty_flags |= dirty_flags::TRANSFORM;
        }
        (NodeData::Rotation(r), 0, FieldValue::Vec3f(v)) => {
            r.axis = *v;
            entry.dirty_flags |= dirty_flags::TRANSFORM;
        }
        (NodeData::Rotation(r), 1, FieldValue::Float(a)) => {
            r.angle = *a;
            entry.dirty_flags |= dirty_flags::TRANSFORM;
        }
        (NodeData::RotationXYZ(r), 1, FieldValue::Float(a)) => {
            r.angle = *a;
            entry.dirty_flags |= dirty_flags::TRANSFORM;
        }
        (NodeData::Switch(s), 0, v) => {
            if let Some(i) = field_as_i32(v) {
                s.which_child = i;
                entry.dirty_flags |= dirty_flags::CHILDREN;
            }
        }
        (NodeData::Material(m), 3, v) => {
            if let Some(o) = field_as_f32(v) {
                m.opacity = o;
                entry.dirty_flags |= dirty_flags::MATERIAL;
            }
        }
        (NodeData::SectionPlane(p), 1, v) => {
            if let Some(b) = field_as_bool(v) {
                p.enabled = b;
                entry.dirty_flags |= dirty_flags::GEOMETRY;
            }
        }
        _ => {}
    }
}
