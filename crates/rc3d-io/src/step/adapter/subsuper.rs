//! Map fidelity `StepInstance` records to flat entity params (OCC Transfer-style).

use crate::step::model::{ComplexMapping, Record, StepInstance};
use crate::step::part21::params::parse_param_list;
use crate::step::value::StepValue;

use super::AdapterMode;
use crate::step::primary_keyword::select_primary_record;

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
    let pairs: Vec<(String, String)> = inst
        .records
        .iter()
        .map(record_param_text)
        .collect();

    if pairs.is_empty() {
        return Err(format!("entity #{}: no records", inst.id));
    }

    let (best_idx, name) = select_primary_record(&pairs, leaf_index)?;

    let merged = match mode {
        AdapterMode::CompatMerge => merge_all_param_texts(&pairs),
        AdapterMode::StrictFidelity => pairs[best_idx].1.clone(),
    };

    let (params, _) =
        parse_param_list(&merged).map_err(|e| format!("entity #{}: {}", inst.id, e))?;
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
