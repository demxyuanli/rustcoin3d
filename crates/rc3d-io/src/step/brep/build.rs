//! STEP → B-Rep builder. T1.7-T1.9

use crate::step::parser::EntityIndex;
use crate::step::StepError;
use super::registry::BRepRegistry;
use super::topo::{SolidKey, ShellKey};

pub struct BRepBuildResult {
    pub registry: BRepRegistry,
    pub root_solids: Vec<SolidKey>,
}

/// Build a full B-Rep from STEP entities.
pub fn build_brep(_entities: &EntityIndex) -> Result<BRepBuildResult, StepError> {
    todo!("T1.7-T1.9")
}
