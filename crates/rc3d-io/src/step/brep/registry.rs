//! B-Rep shape registry with SlotMap storage. T1.5-T1.6

use std::collections::HashMap;
use slotmap::SlotMap;
use rc3d_core::math::Vec3;
use super::topo::*;
use super::geom::CurveGeom;

pub struct BRepRegistry {
    pub vertices: SlotMap<VertexKey, BRepVertex>,
    pub edges: SlotMap<EdgeKey, BRepEdge>,
    pub wires: SlotMap<WireKey, BRepWire>,
    pub faces: SlotMap<FaceKey, BRepFace>,
    pub shells: SlotMap<ShellKey, BRepShell>,
    pub solids: SlotMap<SolidKey, BRepSolid>,
    pub vertex_hash_index: HashMap<[u32; 3], VertexKey>,
}

impl BRepRegistry {
    pub fn new() -> Self {
        Self {
            vertices: SlotMap::with_key(),
            edges: SlotMap::with_key(),
            wires: SlotMap::with_key(),
            faces: SlotMap::with_key(),
            shells: SlotMap::with_key(),
            solids: SlotMap::with_key(),
            vertex_hash_index: HashMap::new(),
        }
    }

    pub fn find_or_add_vertex(&mut self, position: Vec3, tolerance: f32) -> VertexKey {
        todo!("T1.5")
    }

    pub fn add_edge_with_pcurve(&mut self, curve: CurveGeom, tolerance: f32, face: FaceKey, pcurve: CurveGeom) -> EdgeKey {
        todo!("T1.5")
    }

    pub fn find_shared_edges(&self, _face_a: FaceKey, _face_b: FaceKey) -> Vec<EdgeKey> {
        todo!("T1.6")
    }

    pub fn iter_faces(&self) -> impl Iterator<Item = (FaceKey, &BRepFace)> {
        self.faces.iter()
    }
}
