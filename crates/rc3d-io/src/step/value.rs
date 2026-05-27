#[derive(Debug, Clone, PartialEq)]
pub enum StepValue {
    Integer(i64),
    Real(f64),
    String(String),
    Enum(String),
    Ref(u64),
    List(Vec<StepValue>),
    Typed(String, Box<StepValue>),
    Omitted,
}

impl StepValue {
    pub fn as_ref_id(&self) -> Option<u64> {
        match self {
            StepValue::Ref(id) => Some(*id),
            StepValue::Typed(_, inner) => inner.as_ref_id(),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[StepValue]> {
        match self {
            StepValue::List(v) => Some(v),
            StepValue::Typed(_, inner) => inner.as_list(),
            _ => None,
        }
    }

    pub fn nth_param(&self, index: usize) -> Option<&StepValue> {
        self.as_list().and_then(|v| v.get(index))
    }

    pub fn as_real(&self) -> Option<f64> {
        match self {
            StepValue::Real(v) => Some(*v),
            StepValue::Integer(v) => Some(*v as f64),
            StepValue::Typed(_, inner) => inner.as_real(),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            StepValue::Integer(v) => Some(*v),
            StepValue::Real(v) => Some(*v as i64),
            StepValue::Typed(_, inner) => inner.as_int(),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            StepValue::String(s) => Some(s),
            StepValue::Typed(_, inner) => inner.as_string(),
            _ => None,
        }
    }
}
