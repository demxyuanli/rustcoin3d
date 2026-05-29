//! STEP entity instance (#id = ...).

use super::record::Record;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexMapping {
    Simple,
    Internal { leaf_index: usize },
    External,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepInstance {
    pub id: u64,
    pub records: Vec<Record>,
    pub mapping: ComplexMapping,
}

impl StepInstance {
    pub fn primary_keyword(&self) -> Option<&str> {
        crate::step::primary_keyword::primary_keyword(self)
    }
}
