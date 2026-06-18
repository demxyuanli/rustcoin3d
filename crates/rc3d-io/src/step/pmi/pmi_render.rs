//! Convert PMI data to scene graph annotation nodes.
//! Maps PmiData → AnnotationElement[] → AnnotationSetNode → SceneGraph.
//! Rendering handled by existing pass_markup + plane_text pipeline.

use rc3d_core::NodeId;
use rc3d_scene::SceneGraph;
use rc3d_scene::node_data::{
    AnnotationElement, AnnotationSetNode, NodeData,
};
use rc3d_scene::annotation::{AnnotationLabelMode, AnnotationPoint, AnnotationStyle};

use super::pmi_extract::{PmiData, PmiDimension, PmiDatum, PmiToleranceFrame};
use super::pmi_types::PmiDataSet;

/// Default annotation style for imported PMI.
fn pmi_style() -> AnnotationStyle {
    AnnotationStyle {
        decimals: 3,
        unit_suffix: " mm".to_string(),
        font_size: 14.0,
        ..AnnotationStyle::default()
    }
}

/// Helper: PVec3 to [f32; 3] for annotation rendering boundary.
fn to_arr(v: &rc3d_core::math::PVec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

/// Convert a PmiDimension to an AnnotationElement.
fn dimension_to_element(dim: &PmiDimension) -> AnnotationElement {
    AnnotationElement::Dimension {
        start: AnnotationPoint::local(to_arr(&dim.start)),
        end: AnnotationPoint::local(to_arr(&dim.end)),
        offset_dir: to_arr(&dim.offset_dir),
        extension_len: 0.3,
        arrow_size: 0.15,
        label: dim.text.clone(),
        label_mode: AnnotationLabelMode::Fixed,
        color: [1.0, 1.0, 0.0, 1.0], // yellow
    }
}

/// Convert a PmiDatum to an AnnotationElement.
fn datum_to_element(datum: &PmiDatum) -> AnnotationElement {
    AnnotationElement::Datum {
        position: AnnotationPoint::local(to_arr(&datum.origin)),
        size: 0.3,
        color: [1.0, 0.5, 0.0, 1.0], // orange
    }
}

/// Convert a PmiToleranceFrame to an AnnotationElement (GdtFeatureControlFrame).
/// Returns None if the tolerance symbol is not recognized.
fn tolerance_to_element(tol: &PmiToleranceFrame) -> Option<AnnotationElement> {
    let symbol = match tol.symbol {
        Some(s) => s,
        None => {
            log::warn!("[PMI] unrecognized tolerance symbol for '{}', skipping", tol.text);
            return None;
        }
    };
    Some(AnnotationElement::GdtFeatureControlFrame {
        symbol,
        tolerance: tol.value as f32,
        diameter: tol.diameter,
        datum_primary: tol.datum_primary.clone(),
        datum_secondary: tol.datum_secondary.clone(),
        material_condition: tol.material_condition,
        position: AnnotationPoint::local(to_arr(&tol.origin)),
        leader_target: tol.leader_points.first().map(|p| {
            AnnotationPoint::local(to_arr(p))
        }),
        color: [0.3, 0.8, 1.0, 1.0], // blue
    })
}

/// Convert all PMI data into elements.
fn pmi_to_elements(pmi: &PmiData) -> Vec<AnnotationElement> {
    let mut elements = Vec::new();
    for dim in &pmi.dimensions {
        elements.push(dimension_to_element(dim));
    }
    for datum in &pmi.datums {
        elements.push(datum_to_element(datum));
    }
    for tol in &pmi.tolerances {
        if let Some(elem) = tolerance_to_element(tol) {
            elements.push(elem);
        }
    }
    elements
}

/// Attach PMI annotations to the scene graph as an AnnotationSet node.
/// Returns the NodeId of the created AnnotationSet node.
pub fn attach_pmi_to_scene(
    graph: &mut SceneGraph,
    parent: NodeId,
    pmi: &PmiData,
) -> NodeId {
    let elements = pmi_to_elements(pmi);
    if elements.is_empty() {
        return parent;
    }
    let set = AnnotationSetNode {
        elements,
        visible: true,
        style: pmi_style(),
    };
    graph.add_child(parent, NodeData::AnnotationSet(set))
}

/// Attach full PMI data set (including surface finishes) to the scene graph.
/// Returns the NodeId of the created AnnotationSet node.
pub fn attach_pmi_full_to_scene(
    graph: &mut SceneGraph,
    parent: NodeId,
    pmi: &PmiDataSet,
) -> NodeId {
    let mut elements = Vec::new();
    for dim in &pmi.dimensions {
        elements.push(dimension_to_element(dim));
    }
    for datum in &pmi.datums {
        elements.push(datum_to_element(datum));
    }
    for tol in &pmi.tolerances {
        if let Some(elem) = tolerance_to_element(tol) {
            elements.push(elem);
        }
    }
    for finish in &pmi.surface_finishes {
        elements.push(surface_finish_to_element(finish));
    }
    if elements.is_empty() {
        return parent;
    }
    let set = AnnotationSetNode {
        elements,
        visible: true,
        style: pmi_style(),
    };
    graph.add_child(parent, NodeData::AnnotationSet(set))
}

/// Convert a PmiSurfaceFinish to an AnnotationElement.
fn surface_finish_to_element(finish: &super::pmi_types::PmiSurfaceFinish) -> AnnotationElement {
    let ra = finish.ra_value.unwrap_or(0.0);
    AnnotationElement::SurfaceFinish {
        ra_value: ra as f32,
        note: finish.note.clone(),
        position: AnnotationPoint::local(to_arr(&finish.anchor_point)),
        direction: [0.0, 1.0, 0.0],
        color: [0.6, 0.9, 0.6, 1.0], // light green
    }
}
