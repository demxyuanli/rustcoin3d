//! Shared distance tolerances for topology construction, heal, and meshing.

use rc3d_core::math::Real;
/// Numerical floor for spatial-index cell sizes and dedup radii.
pub const TOLERANCE_FLOOR: Real = 1e-6;

/// Default model distance tolerance when STEP uncertainty is absent.
pub const DEFAULT_MODEL_TOLERANCE: Real = 1e-4;

/// Upper clamp for STEP `UNCERTAINTY_MEASURE_WITH_UNIT` values.
pub const MAX_MODEL_TOLERANCE: Real = 0.01;

/// Central tolerance bundle propagated from STEP import through mesh/heal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToleranceContext {
    /// Model distance tolerance (G0), typically from STEP uncertainty.
    pub model: Real,
}

impl Default for ToleranceContext {
    fn default() -> Self {
        Self::from_model(DEFAULT_MODEL_TOLERANCE)
    }
}

impl ToleranceContext {
    pub fn from_model(model: Real) -> Self {
        Self {
            model: model.clamp(TOLERANCE_FLOOR, MAX_MODEL_TOLERANCE),
        }
    }

    /// Vertex dedup radius for `BRepStore::find_or_add_vertex`.
    pub fn vertex_dedup(&self) -> Real {
        self.model.max(TOLERANCE_FLOOR)
    }

    /// Spatial-index cell size backing the vertex pool.
    pub fn vertex_cell_size(&self) -> Real {
        self.vertex_dedup()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_model_clamps_extremes() {
        let low = ToleranceContext::from_model(1e-8);
        assert_eq!(low.model, TOLERANCE_FLOOR);
        let high = ToleranceContext::from_model(1.0);
        assert_eq!(high.model, MAX_MODEL_TOLERANCE);
    }
}
