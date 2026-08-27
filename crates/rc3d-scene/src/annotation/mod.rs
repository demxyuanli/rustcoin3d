//! 3D viewport annotation: geometry, formatting, and label resolution.
//!
//! Annotations are defined in model-local 3D coordinates on a single plane,
//! projected each frame by the renderer (`pass_markup`).

mod format;
mod geometry;
mod label;
mod measurement;
mod pmi;
mod point;
mod resolve;
mod types;

pub use format::*;
pub use geometry::*;
pub use label::*;
pub use measurement::*;
pub use pmi::{
    apply_pmi_document, apply_pmi_to_set, bind_scene_pmi, find_pmi, pmi_for_node,
    resolve_pmi_bindings, PmiDocument,
};
pub use point::AnnotationPoint;
pub use resolve::{
    bound_node_in_element, effective_annotation_model, localize_element_points,
    node_world_matrix, prepare_annotation_for_render, resolve_element, resolve_point,
};
pub use types::*;

pub use crate::node_data::{
    GdtSymbol, GdtMaterialCondition, DatumTargetType, WeldType,
};
