//! AP schema validation on Part21 `StepInstance` records (pre-adapter transfer).

pub mod ap242;

use crate::step::adapter::AdapterMode;
use crate::step::model::Exchange;
use crate::step::validate::SchemaViolation;

/// Validate all instances in a fidelity exchange (geometry/topology subset).
pub fn validate_exchange(exchange: &Exchange) -> Vec<SchemaViolation> {
    validate_exchange_with_mode(exchange, AdapterMode::CompatMerge)
}

pub fn validate_exchange_with_mode(
    exchange: &Exchange,
    mode: AdapterMode,
) -> Vec<SchemaViolation> {
    let keywords: std::collections::HashMap<u64, String> = exchange
        .instances()
        .filter_map(|inst| ap242::instance_keyword(inst, mode).map(|k| (inst.id, k)))
        .collect();
    exchange
        .instances()
        .flat_map(|inst| ap242::validate_instance(inst, mode, &keywords))
        .collect()
}
