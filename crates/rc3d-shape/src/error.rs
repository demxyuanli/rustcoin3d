#[derive(Debug, thiserror::Error)]
pub enum ShapeError {
    #[error("null shape")]
    NullShape,
    #[error("invalid subshape: expected {expected:?}, got {got:?}")]
    InvalidSubshape {
        expected: &'static str,
        got: &'static str,
    },
    #[error("B-Rep build failed: {0}")]
    BuildFailed(String),
    #[error("tessellation failed: {0}")]
    TessellationFailed(String),
    #[error("dedup conflict at STEP entity #{entity_id}")]
    DedupConflict { entity_id: u64 },
}
