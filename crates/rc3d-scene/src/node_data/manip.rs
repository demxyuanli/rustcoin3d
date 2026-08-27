//! Coin3D-style transform manipulator and composable dragger nodes.
use rc3d_core::NodeId;
use serde::{Deserialize, Serialize};

/// Interaction mode for [`TransformManipNode`] (mirrors gizmo translate/rotate/scale).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ManipMode {
    #[default]
    Translate,
    Rotate,
    Scale,
}

/// Axis space for manipulator handles.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ManipSpace {
    #[default]
    World,
    Local,
}

/// Composable dragger part (Coin3D `SoTranslate1Dragger` / plane / rotate / scale).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DraggerKind {
    #[default]
    TranslateX,
    TranslateY,
    TranslateZ,
    TranslateXY,
    TranslateYZ,
    TranslateZX,
    RotateX,
    RotateY,
    RotateZ,
    ScaleX,
    ScaleY,
    ScaleZ,
    ScaleUniform,
}

/// Scene-graph transform manipulator (three.js `TransformControls` / Coin3D `SoTransformManip`).
///
/// Bind to a `Transform` via [`Self::target`], or the preceding sibling `Transform`
/// under the same parent when `target` is `None`. Child [`DraggerNode`]s filter
/// visible handles; with no enabled children, the engine gizmo uses its current mode.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct TransformManipNode {
    pub mode: ManipMode,
    pub space: ManipSpace,
    pub enabled: bool,
    /// Handle size in world units. `0` = auto from target bounding box.
    pub size: f32,
    /// Explicit transform to drive. `None` = preceding sibling Transform.
    pub target: Option<NodeId>,
}

impl Default for TransformManipNode {
    fn default() -> Self {
        Self {
            mode: ManipMode::Translate,
            space: ManipSpace::World,
            enabled: true,
            size: 0.0,
            target: None,
        }
    }
}

/// One dragger part under a [`TransformManipNode`].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct DraggerNode {
    pub kind: DraggerKind,
    pub enabled: bool,
}

impl Default for DraggerNode {
    fn default() -> Self {
        Self {
            kind: DraggerKind::TranslateX,
            enabled: true,
        }
    }
}
