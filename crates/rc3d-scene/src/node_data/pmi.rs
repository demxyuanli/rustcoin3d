//! STEP-style PMI semantic records (no new `NodeData` variant).
//!
//! Visual geometry stays on [`super::control::AnnotationSetNode::elements`];
//! these records bind an annotation to named parts / faces / edges.

use rc3d_core::NodeId;
use serde::{Deserialize, Serialize};

/// Kind of product-manufacturing information (AP242-style).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PmiKind {
    #[default]
    LinearDimension,
    AngularDimension,
    RadialDimension,
    DiameterDimension,
    Gdt,
    Datum,
    Note,
    SurfaceFinish,
    Weld,
}

/// Where the semantic record came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PmiSource {
    #[default]
    Manual,
    StepAp242,
    Json,
}

/// Geometry reference: resolved `NodeId` and/or authoring name + sub-entity.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PmiBinding {
    #[serde(default)]
    pub node: Option<NodeId>,
    /// Resolved by [`crate::annotation::resolve_pmi_bindings`] via `SceneGraph::find_named`.
    #[serde(default)]
    pub node_name: Option<String>,
    #[serde(default)]
    pub face: Option<u32>,
    #[serde(default)]
    pub edge: Option<u32>,
}

impl PmiBinding {
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            node: None,
            node_name: Some(name.into()),
            face: None,
            edge: None,
        }
    }

    pub fn on_node(node: NodeId) -> Self {
        Self {
            node: Some(node),
            node_name: None,
            face: None,
            edge: None,
        }
    }

    pub fn with_face(mut self, face: u32) -> Self {
        self.face = Some(face);
        self
    }
}

/// One PMI semantic attached to an [`super::control::AnnotationSetNode`] element.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PmiRecord {
    pub id: String,
    pub kind: PmiKind,
    /// Index into the parent set's `elements` (visual).
    #[serde(default)]
    pub element_index: usize,
    #[serde(default)]
    pub bindings: Vec<PmiBinding>,
    #[serde(default)]
    pub nominal: Option<f32>,
    #[serde(default)]
    pub upper_tol: Option<f32>,
    #[serde(default)]
    pub lower_tol: Option<f32>,
    #[serde(default)]
    pub unit: String,
    #[serde(default)]
    pub datum_refs: Vec<String>,
    #[serde(default)]
    pub source: PmiSource,
}

impl Default for PmiRecord {
    fn default() -> Self {
        Self {
            id: String::new(),
            kind: PmiKind::LinearDimension,
            element_index: 0,
            bindings: Vec::new(),
            nominal: None,
            upper_tol: None,
            lower_tol: None,
            unit: String::new(),
            datum_refs: Vec::new(),
            source: PmiSource::Manual,
        }
    }
}

impl PmiRecord {
    pub fn new(id: impl Into<String>, kind: PmiKind, element_index: usize) -> Self {
        Self {
            id: id.into(),
            kind,
            element_index,
            ..Self::default()
        }
    }

    pub fn bound_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.bindings.iter().filter_map(|b| b.node)
    }
}
