//! Grouping nodes (Separator, Group, Billboard).
use serde::{Deserialize, Serialize};

/// Behavioral marker: saves/restores all state elements during traversal.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SeparatorNode;

/// Ordered container of children (no state save/restore).
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct GroupNode;

/// Billboard node: renders children always facing the active camera.
/// During traversal, the model matrix is adjusted to cancel the camera
/// rotation, keeping the children screen-aligned.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct BillboardNode {
    /// If true, only the Y-axis rotation is cancelled (cylindrical billboard).
    /// If false, full camera orientation is cancelled (spherical billboard).
    pub axis_aligned: bool,
}

/// Camera-facing textured quad (three.js Sprite).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SpriteNode {
    pub texture_path: String,
    pub color: [f32; 4],
    pub size: f32,
    /// When false, world size grows with camera distance (constant screen size).
    pub size_attenuation: bool,
    pub center: [f32; 2],
    pub opacity: f32,
}

impl Default for SpriteNode {
    fn default() -> Self {
        Self {
            texture_path: String::new(),
            color: [1.0, 1.0, 1.0, 1.0],
            size: 1.0,
            size_attenuation: true,
            center: [0.5, 0.5],
            opacity: 1.0,
        }
    }
}
