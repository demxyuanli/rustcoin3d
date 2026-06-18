//! TopoDS_Shape-like handles.

use rc3d_core::math::PMat4;
use slotmap::new_key_type;

use crate::topo::{EdgeKey, FaceKey, Orientation, ShellKey, SolidKey, VertexKey, WireKey};

new_key_type! { pub struct ShapeId; }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeType {
    Vertex,
    Edge,
    Wire,
    Face,
    Shell,
    Solid,
    Compound,
    CompSolid,
}

#[derive(Clone, Copy)]
pub struct Shape {
    pub id: ShapeId,
}

impl Shape {
    pub fn null() -> Self {
        Self {
            id: ShapeId::default(),
        }
    }

    pub fn is_null(self) -> bool {
        self.id == ShapeId::default()
    }
}

#[derive(Debug, Clone)]
pub struct ShapeNode {
    pub kind: ShapeKind,
    pub orientation: Orientation,
    pub location: PMat4,
    pub parent: Option<ShapeId>,
    pub children: Vec<ShapeId>,
}

#[derive(Debug, Clone)]
pub enum ShapeKind {
    Vertex(VertexKey),
    Edge(EdgeKey),
    Wire(WireKey),
    Face(FaceKey),
    Shell(ShellKey),
    Solid(SolidKey),
    Compound,
    CompSolid,
}

impl ShapeKind {
    pub fn shape_type(&self) -> ShapeType {
        match self {
            ShapeKind::Vertex(_) => ShapeType::Vertex,
            ShapeKind::Edge(_) => ShapeType::Edge,
            ShapeKind::Wire(_) => ShapeType::Wire,
            ShapeKind::Face(_) => ShapeType::Face,
            ShapeKind::Shell(_) => ShapeType::Shell,
            ShapeKind::Solid(_) => ShapeType::Solid,
            ShapeKind::Compound => ShapeType::Compound,
            ShapeKind::CompSolid => ShapeType::CompSolid,
        }
    }

    pub fn solid_key(&self) -> Option<SolidKey> {
        match self {
            ShapeKind::Solid(sk) => Some(*sk),
            _ => None,
        }
    }
}
