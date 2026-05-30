//! Shared STEP param helpers.
use super::super::value::StepValue;

// ── Helpers ─────────────────────────────────────────────────

pub(crate) fn nth_ref(params: &StepValue, index: usize) -> Option<u64> {
    params.nth_param(index)?.as_ref_id()
}

pub(crate) fn nth_bool(params: &StepValue, index: usize) -> bool {
    match params.nth_param(index) {
        Some(StepValue::Enum(s)) => s == ".T.",
        _ => true,
    }
}

pub(crate) fn nth_enum(params: &StepValue, index: usize) -> Option<String> {
    match params.nth_param(index) {
        Some(StepValue::Enum(s)) => Some(s.clone()),
        _ => None,
    }
}

pub(crate) fn nth_list_refs(params: &StepValue, index: usize) -> Option<Vec<u64>> {
    let list = params.nth_param(index)?.as_list()?;
    Some(list.iter().filter_map(|v| v.as_ref_id()).collect())
}

pub(crate) fn nth_list_params(params: &StepValue, index: usize) -> Option<Vec<StepValue>> {
    match params.nth_param(index) {
        Some(StepValue::List(v)) => Some(v.clone()),
        _ => None,
    }
}
