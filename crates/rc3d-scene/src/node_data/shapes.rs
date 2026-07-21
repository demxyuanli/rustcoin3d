//! Shape and mesh nodes.
use rc3d_core::math::Vec3;
use serde::{Deserialize, Serialize};
use crate::animation::{AnimationClip, Skeleton, VertexSkinData};

/// Shape: renders the first 3 coordinates as a triangle.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TriangleNode;

/// Shape: axis-aligned box.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CubeNode {
    pub width: f32,
    pub height: f32,
    pub depth: f32,
}

impl Default for CubeNode {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 1.0,
            depth: 1.0,
        }
    }
}

/// Shape: UV sphere.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SphereNode {
    pub radius: f32,
}

impl Default for SphereNode {
    fn default() -> Self {
        Self { radius: 1.0 }
    }
}

/// Shape: cone.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConeNode {
    pub bottom_radius: f32,
    pub height: f32,
}

impl Default for ConeNode {
    fn default() -> Self {
        Self {
            bottom_radius: 1.0,
            height: 2.0,
        }
    }
}

/// Shape: cylinder.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CylinderNode {
    pub radius: f32,
    pub height: f32,
}

impl Default for CylinderNode {
    fn default() -> Self {
        Self {
            radius: 1.0,
            height: 2.0,
        }
    }
}

/// Shape: torus (donut).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TorusNode {
    /// Distance from the center of the torus to the center of the tube.
    pub major_radius: f32,
    /// Radius of the tube.
    pub minor_radius: f32,
}

impl Default for TorusNode {
    fn default() -> Self {
        Self {
            major_radius: 1.0,
            minor_radius: 0.25,
        }
    }
}

/// Shape: arbitrary triangle mesh from vertex/index arrays.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct IndexedFaceSetNode {
    pub coord_index: Vec<i32>,
}

/// Shape: line segments from vertex/index arrays (Coin3D SoIndexedLineSet).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IndexedLineSetNode {
    pub coord_index: Vec<i32>,
    pub line_width: f32,
    /// RGBA color for the edge overlay pass.
    #[serde(default = "default_indexed_line_color")]
    pub color: [f32; 4],
}

fn default_indexed_line_color() -> [f32; 4] {
    [0.5, 0.5, 0.5, 1.0]
}

impl Default for IndexedLineSetNode {
    fn default() -> Self {
        Self {
            coord_index: Vec::new(),
            line_width: 1.0,
            color: default_indexed_line_color(),
        }
    }
}

/// Skeletal skinning data for mesh geometry under the same separator (Coin-style sidecar).
/// Place before [`Coordinate3`](Coordinate3Node) / [`IndexedFaceSet`](IndexedFaceSetNode) so the
/// render collector can attach weights to the generated draw call.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SkinnedMeshNode {
    pub skeleton: Skeleton,
    pub skin_data: Vec<VertexSkinData>,
    pub clip: Option<AnimationClip>,
}

/// A single morph target (blend shape) storing per-vertex deltas.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MorphTarget {
    pub name: String,
    /// Per-vertex position deltas (same length as Coordinate3Node::point).
    pub position_deltas: Vec<Vec3>,
    /// Per-vertex normal deltas (same length as NormalNode::vector, if present).
    pub normal_deltas: Option<Vec<Vec3>>,
    /// Per-vertex tangent deltas (same length as the tangent array, if present).
    pub tangent_deltas: Option<Vec<[f32; 4]>>,
}

/// Stores morph target (blend shape) data and per-instance weights.
/// Parent this node alongside the geometry it affects within a Separator.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct MorphTargetNode {
    /// The morph targets (blend shapes) for this mesh.
    pub targets: Vec<MorphTarget>,
    /// Current weight for each target (same length as targets).
    pub weights: Vec<f32>,
}
