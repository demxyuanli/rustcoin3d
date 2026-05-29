//! One EXPRESS record inside a STEP instance.

use crate::step::value::StepValue;

#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub keyword: String,
    pub params: StepValue,
}
