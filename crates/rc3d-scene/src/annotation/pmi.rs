//! Bind STEP-style PMI records on [`AnnotationSetNode`] to named geometry.

use rc3d_core::NodeId;
use serde::{Deserialize, Serialize};

use crate::node_data::{AnnotationElement, NodeData, PmiRecord};
use crate::SceneGraph;

use super::point::AnnotationPoint;

/// Interchange document (JSON sidecar; not a STEP file).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PmiDocument {
    #[serde(default = "pmi_doc_version")]
    pub version: u32,
    #[serde(default)]
    pub records: Vec<PmiRecord>,
}

fn pmi_doc_version() -> u32 {
    1
}

impl PmiDocument {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Resolve `node_name` → `NodeId` via [`SceneGraph::find_named`].
pub fn resolve_pmi_bindings(graph: &SceneGraph, records: &mut [PmiRecord]) {
    for rec in records.iter_mut() {
        for binding in rec.bindings.iter_mut() {
            if binding.node.is_none() {
                if let Some(name) = binding.node_name.as_deref() {
                    binding.node = graph.find_named(name);
                }
            }
        }
    }
}

fn stamp_pt(point: &mut AnnotationPoint, node: NodeId) {
    if point.node.is_none() {
        point.node = Some(node);
    }
}

fn stamp_unbound_points(element: &mut AnnotationElement, node: NodeId) {
    match element {
        AnnotationElement::Dimension { start, end, .. } => {
            stamp_pt(start, node);
            stamp_pt(end, node);
        }
        AnnotationElement::AngleDimension {
            center, arm1, arm2, ..
        } => {
            stamp_pt(center, node);
            stamp_pt(arm1, node);
            stamp_pt(arm2, node);
        }
        AnnotationElement::RadialDimension {
            center, perimeter, ..
        } => {
            stamp_pt(center, node);
            stamp_pt(perimeter, node);
        }
        AnnotationElement::DiameterDimension { center, p1, p2, .. } => {
            stamp_pt(center, node);
            stamp_pt(p1, node);
            stamp_pt(p2, node);
        }
        AnnotationElement::Leader { anchor, .. } => stamp_pt(anchor, node),
        AnnotationElement::Callout { anchor, .. } => stamp_pt(anchor, node),
        AnnotationElement::Datum { position, .. } => stamp_pt(position, node),
        AnnotationElement::GdtFeatureControlFrame {
            position,
            leader_target,
            ..
        } => {
            stamp_pt(position, node);
            if let Some(lt) = leader_target.as_mut() {
                stamp_pt(lt, node);
            }
        }
        AnnotationElement::GdtDatumTarget { position, .. } => stamp_pt(position, node),
        AnnotationElement::ChamferDimension { start, end, .. } => {
            stamp_pt(start, node);
            stamp_pt(end, node);
        }
        AnnotationElement::OrdinateDimension { feature, datum, .. } => {
            stamp_pt(feature, node);
            stamp_pt(datum, node);
        }
        AnnotationElement::SurfaceFinish { position, .. } => stamp_pt(position, node),
        AnnotationElement::WeldSymbol { position, .. } => stamp_pt(position, node),
        AnnotationElement::DatumIdentifier { position, .. } => stamp_pt(position, node),
    }
}

/// Resolve names on `set_id` and stamp unbound annotation points to bound nodes.
/// Returns how many records successfully stamped an element.
pub fn apply_pmi_to_set(graph: &mut SceneGraph, set_id: NodeId) -> usize {
    let mut records = match graph.get(set_id) {
        Some(entry) => match &entry.data {
            NodeData::AnnotationSet(set) => set.pmi.clone(),
            _ => return 0,
        },
        None => return 0,
    };
    resolve_pmi_bindings(graph, &mut records);

    let Some(entry) = graph.get_mut(set_id) else {
        return 0;
    };
    let NodeData::AnnotationSet(set) = &mut entry.data else {
        return 0;
    };
    let mut stamped = 0;
    for rec in &records {
        let Some(node) = rec.bindings.iter().find_map(|b| b.node) else {
            continue;
        };
        if let Some(element) = set.elements.get_mut(rec.element_index) {
            stamp_unbound_points(element, node);
            stamped += 1;
        }
    }
    set.pmi = records;
    stamped
}

/// Append `doc.records` onto the set, then apply bindings.
pub fn apply_pmi_document(graph: &mut SceneGraph, set_id: NodeId, doc: &PmiDocument) -> usize {
    if let Some(entry) = graph.get_mut(set_id) {
        if let NodeData::AnnotationSet(set) = &mut entry.data {
            set.pmi.extend(doc.records.iter().cloned());
        }
    }
    apply_pmi_to_set(graph, set_id)
}

/// Walk every `AnnotationSet` and apply PMI bindings.
pub fn bind_scene_pmi(graph: &mut SceneGraph) -> usize {
    let ids = graph.all_node_ids();
    let mut total = 0;
    for id in ids {
        if matches!(
            graph.get(id).map(|e| &e.data),
            Some(NodeData::AnnotationSet(_))
        ) {
            total += apply_pmi_to_set(graph, id);
        }
    }
    total
}

/// First record whose `id` matches (owned clone).
pub fn find_pmi(graph: &SceneGraph, id: &str) -> Option<(NodeId, PmiRecord)> {
    for set_id in graph.all_node_ids() {
        if let Some(entry) = graph.get(set_id) {
            if let NodeData::AnnotationSet(set) = &entry.data {
                if let Some(rec) = set.pmi.iter().find(|r| r.id == id) {
                    return Some((set_id, rec.clone()));
                }
            }
        }
    }
    None
}

/// Records that bind to `node` (after resolve).
pub fn pmi_for_node(graph: &SceneGraph, node: NodeId) -> Vec<(NodeId, PmiRecord)> {
    let mut out = Vec::new();
    for set_id in graph.all_node_ids() {
        if let Some(entry) = graph.get(set_id) {
            if let NodeData::AnnotationSet(set) = &entry.data {
                for rec in &set.pmi {
                    if rec.bound_nodes().any(|n| n == node) {
                        out.push((set_id, rec.clone()));
                    }
                }
            }
        }
    }
    out
}
