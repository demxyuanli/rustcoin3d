//! Shared primary-record selection for complex STEP instances (adapter + model).

use crate::step::model::{ComplexMapping, Record, StepInstance};
use crate::step::value::StepValue;

pub const PRIORITY_TYPES: &[&str] = &[
    "B_SPLINE_CURVE_WITH_KNOTS",
    "B_SPLINE_CURVE",
    "RATIONAL_B_SPLINE_CURVE",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "B_SPLINE_SURFACE",
    "RATIONAL_B_SPLINE_SURFACE",
    "LINE",
    "CIRCLE",
    "ELLIPSE",
    "HYPERBOLA",
    "PARABOLA",
    "POLYLINE",
    "TRIMMED_CURVE",
    "COMPOSITE_CURVE",
    "SEAM_CURVE",
    "INTERSECTION_CURVE",
    "OFFSET_CURVE_3D",
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "SURFACE_OF_LINEAR_EXTRUSION",
    "SURFACE_OF_REVOLUTION",
    "RECTANGULAR_TRIMMED_SURFACE",
    "CURVE_BOUNDED_SURFACE",
    "FACE_SURFACE",
    "ADVANCED_FACE",
    "FACE_OUTER_BOUND",
    "FACE_BOUND",
    "CLOSED_SHELL",
    "OPEN_SHELL",
    "SHELL",
    "EDGE_CURVE",
    "ORIENTED_EDGE",
    "EDGE_LOOP",
    "VERTEX_POINT",
    "CARTESIAN_POINT",
    "DIRECTION",
    "VECTOR",
    "AXIS2_PLACEMENT_3D",
    "AXIS2_PLACEMENT_2D",
    "CURVE",
    "SURFACE",
    "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
    "PRODUCT_DEFINITION_SHAPE",
    "SHAPE_DEFINITION_REPRESENTATION",
    "ITEM_DEFINED_TRANSFORMATION",
    "MANIFOLD_SOLID_BREP",
    "BREP_WITH_VOIDS",
];

/// Primary keyword for display / inventory (same rules as adapter flatten).
pub fn primary_keyword(inst: &StepInstance) -> Option<&str> {
    let leaf_index = match inst.mapping {
        ComplexMapping::Simple => 0,
        ComplexMapping::Internal { leaf_index } => leaf_index,
        ComplexMapping::External => inst.records.len().saturating_sub(1),
    };
    let pairs: Vec<(String, String)> = inst
        .records
        .iter()
        .map(record_param_text)
        .collect();
    let (idx, _) = select_primary_record(&pairs, leaf_index).ok()?;
    inst.records.get(idx).map(|r| r.keyword.as_str())
}

pub fn select_primary_record(
    pairs: &[(String, String)],
    leaf_index: usize,
) -> Result<(usize, String), String> {
    let mut best_idx = leaf_index.min(pairs.len().saturating_sub(1));
    let mut best_priority = usize::MAX;
    for (idx, (kw, _)) in pairs.iter().enumerate() {
        if let Some(prio) = PRIORITY_TYPES.iter().position(|t| t == kw) {
            if prio < best_priority {
                best_priority = prio;
                best_idx = idx;
            }
        }
    }
    if best_priority == usize::MAX {
        best_idx = pairs
            .iter()
            .rposition(|(_, pd)| !pd.is_empty() && pd != "''" && pd != "*" && pd != "$")
            .unwrap_or(best_idx);
    }
    Ok((best_idx, pairs[best_idx].0.clone()))
}

fn record_param_text(r: &Record) -> (String, String) {
    let text = params_to_comma_text(&r.params);
    (r.keyword.clone(), text)
}

fn params_to_comma_text(params: &StepValue) -> String {
    match params.as_list() {
        Some(list) if !list.is_empty() => list
            .iter()
            .map(value_to_text)
            .collect::<Vec<_>>()
            .join(","),
        _ => String::new(),
    }
}

fn value_to_text(v: &StepValue) -> String {
    match v {
        StepValue::Integer(n) => n.to_string(),
        StepValue::Real(x) => x.to_string(),
        StepValue::String(s) => format!("'{}'", s.replace('\'', "''")),
        StepValue::Enum(e) => e.clone(),
        StepValue::Ref(id) => format!("#{id}"),
        StepValue::Omitted => "$".to_string(),
        StepValue::Typed(name, inner) => {
            format!("{}({})", name, params_to_comma_text(inner))
        }
        StepValue::List(items) => {
            let inner = items.iter().map(value_to_text).collect::<Vec<_>>().join(",");
            format!("({inner})")
        }
    }
}
