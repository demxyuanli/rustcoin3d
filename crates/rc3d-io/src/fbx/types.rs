use std::collections::HashMap;

use rc3d_core::math::{Mat4, Vec3};
use rc3d_scene::animation::{AnimationClip, VertexSkinData};

/// All parsed FBX data before conversion to SceneGraph.
#[derive(Default)]
pub struct FbxData {
    pub objects: HashMap<i64, FbxObject>,
    pub connections: Vec<FbxConnection>,
}

#[derive(Clone, Debug)]
pub enum FbxObject {
    Geometry(FbxGeometry),
    Material(rc3d_scene::node_data::MaterialNode),
    Model(FbxModel),
    Deformer(FbxDeformer),
    AnimationStack(FbxAnimationStack),
    AnimationLayer(FbxAnimationLayer),
    AnimationCurveNode(FbxAnimationCurveNode),
    AnimationCurve(FbxAnimationCurve),
}

#[derive(Clone, Debug, Default)]
pub struct FbxGeometry {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct FbxModel {
    pub name: String,
    pub model_type: String,
    pub local_transform: Mat4,
    pub translation: Option<Vec3>,
    pub rotation: Option<Vec3>,
    pub scaling: Option<Vec3>,
    /// Pre-rotation (FBX bone pre-rotation in Euler degrees).
    pub pre_rotation: Option<Vec3>,
    /// Post-rotation (FBX bone post-rotation in Euler degrees).
    pub post_rotation: Option<Vec3>,
}

impl FbxModel {
    pub fn is_limb_node(&self) -> bool {
        self.model_type == "LimbNode" || self.model_type == "Limb"
    }
}

#[derive(Clone, Debug)]
pub enum FbxDeformer {
    Skin {
        /// Cluster IDs connected to this skin deformer.
        clusters: Vec<i64>,
    },
    Cluster {
        /// bone_index within the skeleton (resolved later).
        bone_id: i64,
        /// Per-vertex indices into the mesh.
        indices: Vec<i32>,
        /// Per-vertex weights (same length as indices).
        weights: Vec<f64>,
        /// Transform: link (bone) to mesh bind-pose matrix.
        transform: Mat4,
        /// TransformLink: mesh to link (bone) bind-pose matrix.
        transform_link: Mat4,
    },
    BlendShape {
        /// Blend shape channel indices.
        channels: Vec<i64>,
    },
}

#[derive(Clone, Debug, Default)]
pub struct FbxAnimationStack {
    pub name: String,
}

#[derive(Clone, Debug, Default)]
pub struct FbxAnimationLayer {
    pub name: String,
}

#[derive(Clone, Debug, Default)]
pub struct FbxAnimationCurveNode {
    pub name: String,
    /// Target property: "T" (translation), "R" (rotation), "S" (scaling).
    pub target_property: String,
    /// Connected model (bone) ID.
    pub target_model_id: Option<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct FbxAnimationCurve {
    /// Keyframe times (seconds).
    pub times: Vec<f64>,
    /// Keyframe values.
    pub values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct FbxConnection {
    pub child: i64,
    pub parent: i64,
}

/// Resolved skeleton data for a mesh.
pub struct ResolvedSkeleton {
    pub skeleton: rc3d_scene::animation::Skeleton,
    pub skin_data: Vec<VertexSkinData>,
}

/// Resolved animation data.
pub struct ResolvedAnimation {
    pub clips: Vec<AnimationClip>,
}
