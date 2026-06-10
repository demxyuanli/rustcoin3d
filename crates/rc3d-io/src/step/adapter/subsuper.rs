//! Map fidelity `StepInstance` records to flat entity params (OCC Transfer-style).

use crate::step::model::{ComplexMapping, Record, StepInstance};
use crate::step::primary_keyword::PRIORITY_TYPES;
use crate::step::value::StepValue;

use super::AdapterMode;

pub fn instance_to_name_and_params(
    inst: &StepInstance,
    mode: AdapterMode,
) -> Result<(String, StepValue), String> {
    match inst.mapping {
        ComplexMapping::Simple => {
            let r = inst.records.first().ok_or("empty instance")?;
            Ok((r.keyword.clone(), r.params.clone()))
        }
        ComplexMapping::Internal { leaf_index } => {
            let (name, params) = flatten_complex(inst, leaf_index, mode)?;
            Ok((name, params))
        }
        ComplexMapping::External => {
            let leaf = inst.records.len().saturating_sub(1);
            let (name, params) = flatten_complex(inst, leaf, mode)?;
            Ok((name, params))
        }
    }
}

fn flatten_complex(
    inst: &StepInstance,
    leaf_index: usize,
    mode: AdapterMode,
) -> Result<(String, StepValue), String> {
    let pairs_struct: Vec<(String, &StepValue)> = inst
        .records
        .iter()
        .map(|r| (r.keyword.clone(), &r.params))
        .collect();

    if pairs_struct.is_empty() {
        return Err(format!("entity #{}: no records", inst.id));
    }

    let (best_idx, name) = select_primary_record_structured(&pairs_struct, leaf_index)?;

    let params = match mode {
        AdapterMode::CompatMerge => merge_all_params_structured(
            &pairs_struct.iter().map(|(k, v)| (k.as_str(), *v)).collect::<Vec<_>>(),
            best_idx,
        ),
        AdapterMode::StrictFidelity => pairs_struct[best_idx].1.clone(),
    };

    Ok((name, params))
}

#[allow(dead_code)]
fn record_param_text(r: &Record) -> (String, String) {
    let text = params_to_comma_text(&r.params);
    (r.keyword.clone(), text)
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
fn merge_all_param_texts(pairs: &[(String, String)]) -> String {
    let mut parts = Vec::new();
    for (_, pd) in pairs {
        let trimmed = pd.trim();
        if !trimmed.is_empty() && trimmed != "''" && trimmed != "*" && trimmed != "$" {
            parts.push(trimmed.to_string());
        }
    }
    parts.join(",")
}

/// Select the primary record from structured (keyword, StepValue) pairs.
///
/// Strategy:
/// 1-2. Prioritise concrete geometric types (B_SPLINE_SURFACE, B_SPLINE_CURVE, etc.)
///      over abstract supertypes (REPRESENTATION_ITEM, SURFACE, CURVE, etc.) using
///      `PRIORITY_TYPES`, with a structural sanity check (first param must be
///      Integer or Ref to avoid empty supertype stubs).
/// 3.  Fallback: trust the STEP-assigned `leaf_index` (points to the most-derived
///      subtype).
fn select_primary_record_structured(
    pairs: &[(String, &StepValue)],
    leaf_index: usize,
) -> Result<(usize, String), String> {
    if pairs.is_empty() {
        return Err("no records".to_string());
    }
    if pairs.len() == 1 {
        return Ok((0, pairs[0].0.clone()));
    }

    // Rule 1-2: priority-based with structural check
    for prio_type in PRIORITY_TYPES {
        for (i, (name, params)) in pairs.iter().enumerate() {
            if name.as_str() != *prio_type {
                continue;
            }
            // Structural check: first param must be Integer (degree) or Ref (placement)
            let has_struct = params
                .as_list()
                .and_then(|l| l.first())
                .is_some_and(|v| {
                    matches!(v, StepValue::Integer(_) | StepValue::Ref(_))
                });
            if has_struct {
                return Ok((i, name.clone()));
            }
        }
    }

    // Rule 3: fallback to leaf_index
    let idx = leaf_index.min(pairs.len() - 1);
    Ok((idx, pairs[idx].0.clone()))
}

/// Merge all record parameters as structured StepValue lists.
/// Each record contributes its parameter list entries; Omitted values are skipped
/// to avoid contaminating the primary record's parameter count with supertype placeholders.
fn merge_all_params_structured(
    pairs: &[(&str, &StepValue)],
    _primary_idx: usize,
) -> StepValue {
    let mut merged: Vec<StepValue> = Vec::new();
    for (_keyword, params) in pairs {
        if let Some(list) = params.as_list() {
            for val in list {
                if matches!(val, StepValue::Omitted) {
                    continue;
                }
                // NOTE: Deduplication was removed because it breaks the positional layout
                // of merged params. Supertype and subtype records can contribute identical
                // StepValues (e.g. two `.F.` LOGICALs in B_SPLINE_SURFACE) that occupy
                // different parameter slots. Removing them shifts indices, causing the
                // downstream B-rep builder to read wrong values for knots, weights, etc.
                merged.push(val.clone());
            }
        }
    }
    StepValue::List(merged)
}
