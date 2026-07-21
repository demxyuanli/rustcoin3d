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
