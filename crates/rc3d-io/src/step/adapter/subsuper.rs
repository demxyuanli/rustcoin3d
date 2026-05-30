//! Map fidelity `StepInstance` records to flat entity params (OCC Transfer-style).

use crate::step::model::{ComplexMapping, Record, StepInstance};
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
/// Strategy: trust the STEP-assigned leaf_index (points to the most-derived
/// subtype). Falls back to the last record when leaf_index is out of range.
///
/// NOTE: A previous attempt delegated to `primary_keyword::PRIORITY_TYPES` for
/// better geometric-type selection, but this broke the NURBS build pipeline for
/// production STEP files because the priority-selected record's parameters have
/// different semantics than the STEP-ordered leaf record's parameters.
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
                // Deduplicate: skip if same value already in merged
                let is_dup = merged.iter().any(|existing| {
                    step_values_equal(existing, val)
                });
                if !is_dup {
                    merged.push(val.clone());
                }
            }
        }
    }
    StepValue::List(merged)
}

/// Approximate equality check for StepValue deduplication during merge.
fn step_values_equal(a: &StepValue, b: &StepValue) -> bool {
    match (a, b) {
        (StepValue::Integer(n1), StepValue::Integer(n2)) => n1 == n2,
        (StepValue::Real(r1), StepValue::Real(r2)) => (r1 - r2).abs() < 1e-10,
        (StepValue::Ref(id1), StepValue::Ref(id2)) => id1 == id2,
        (StepValue::Enum(e1), StepValue::Enum(e2)) => e1 == e2,
        (StepValue::String(s1), StepValue::String(s2)) => s1 == s2,
        (StepValue::Omitted, StepValue::Omitted) => true,
        (StepValue::Typed(n1, i1), StepValue::Typed(n2, i2)) => {
            n1 == n2 && step_values_equal(i1, i2)
        }
        (StepValue::List(l1), StepValue::List(l2)) => {
            l1.len() == l2.len()
                && l1.iter().zip(l2.iter()).all(|(a, b)| step_values_equal(a, b))
        }
        // Integer/Real cross-type: compare numerically
        (StepValue::Integer(n), StepValue::Real(r)) | (StepValue::Real(r), StepValue::Integer(n)) => {
            (*n as f64 - *r).abs() < 1e-10
        }
        _ => false,
    }
}
