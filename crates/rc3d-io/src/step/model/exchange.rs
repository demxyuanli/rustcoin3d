//! Full STEP exchange structure.

use super::conformance::ImplementationLevel;
use super::instance::StepInstance;
use crate::step::header::HeaderInfo;

#[derive(Debug, Clone, Default)]
pub struct Exchange {
    pub header: Option<HeaderInfo>,
    pub implementation_level: Option<ImplementationLevel>,
    pub anchor_raw: Option<String>,
    pub reference_raw: Option<String>,
    pub data_sections: Vec<DataSection>,
    pub signatures: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DataSection {
    pub instances: Vec<StepInstance>,
}

impl Exchange {
    pub fn instances(&self) -> impl Iterator<Item = &StepInstance> {
        self.data_sections.iter().flat_map(|s| s.instances.iter())
    }

    pub fn instance_count(&self) -> usize {
        self.data_sections.iter().map(|s| s.instances.len()).sum()
    }
}
