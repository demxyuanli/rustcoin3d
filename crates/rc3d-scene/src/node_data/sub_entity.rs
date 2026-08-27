//! Per-face / per-edge tint records (HOOPS sub-entity color). No new `NodeData` variant.

use serde::{Deserialize, Serialize};

/// Color one CAD face (id from [`super::IndexedFaceSetNode::face_ids`] or cube face `tri / 2`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceTint {
    pub id: u32,
    pub color: [f32; 4],
}

/// Color one triangle edge (`edge` is 0..2 on `triangle`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeTint {
    pub triangle: u32,
    pub edge: u8,
    pub color: [f32; 4],
}

impl FaceTint {
    pub fn new(id: u32, color: [f32; 4]) -> Self {
        Self { id, color }
    }
}

impl EdgeTint {
    pub fn new(triangle: u32, edge: u8, color: [f32; 4]) -> Self {
        Self {
            triangle,
            edge: edge.min(2),
            color,
        }
    }
}
