//! 3D point references for annotations (local or bound to scene-graph nodes).

use rc3d_core::NodeId;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A point in annotation geometry: either local to the `AnnotationSet` transform or bound to a node.
#[derive(Clone, Debug, PartialEq)]
pub struct AnnotationPoint {
    /// When set, `local` is expressed in this node's coordinate system.
    pub node: Option<NodeId>,
    pub local: [f32; 3],
}

impl AnnotationPoint {
    pub fn local(pos: [f32; 3]) -> Self {
        Self { node: None, local: pos }
    }

    pub fn on_node(node: NodeId, local: [f32; 3]) -> Self {
        Self {
            node: Some(node),
            local,
        }
    }

    /// Resolved or author-local coordinates in `AnnotationSet` space.
    pub fn coords(&self) -> [f32; 3] {
        self.local
    }
}

impl From<[f32; 3]> for AnnotationPoint {
    fn from(local: [f32; 3]) -> Self {
        Self::local(local)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum AnnotationPointSerde {
    Local([f32; 3]),
    Bound { node: NodeId, local: [f32; 3] },
}

impl Serialize for AnnotationPoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.node {
            None => self.local.serialize(serializer),
            Some(node) => AnnotationPointSerde::Bound {
                node,
                local: self.local,
            }
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for AnnotationPoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match AnnotationPointSerde::deserialize(deserializer)? {
            AnnotationPointSerde::Local(local) => Ok(Self::local(local)),
            AnnotationPointSerde::Bound { node, local } => Ok(Self::on_node(node, local)),
        }
    }
}
