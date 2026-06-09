//! Shared distance tolerances for topology construction, heal, and meshing.

/// Numerical floor for spatial-index cell sizes and dedup radii.
pub const TOLERANCE_FLOOR: f32 = 1e-6;

/// Default model distance tolerance when STEP uncertainty is absent.
pub const DEFAULT_MODEL_TOLERANCE: f32 = 1e-4;

/// Upper clamp for STEP `UNCERTAINTY_MEASURE_WITH_UNIT` values.
pub const MAX_MODEL_TOLERANCE: f32 = 0.01;

/// Central tolerance bundle propagated from STEP import through mesh/heal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToleranceContext {
    /// Model distance tolerance (G0), typically from STEP uncertainty.
    pub model: f32,
}

impl Default for ToleranceContext {
    fn default() -> Self {
        Self::from_model(DEFAULT_MODEL_TOLERANCE)
    }
}

impl ToleranceContext {
    pub fn from_model(model: f32) -> Self {
        Self {
            model: model.clamp(TOLERANCE_FLOOR, MAX_MODEL_TOLERANCE),
        }
    }

    /// Vertex dedup radius for `BRepStore::find_or_add_vertex`.
    pub fn vertex_dedup(&self) -> f32 {
        self.model.max(TOLERANCE_FLOOR)
    }

    /// Spatial-index cell size backing the vertex pool.
    pub fn vertex_cell_size(&self) -> f32 {
        self.vertex_dedup()
    }

    /// Boundary vertex dedup radius for shell meshing spatial pools.
    pub fn boundary_dedup(&self, mesh: &crate::mesh::BRepMeshConfig) -> f32 {
        self.model
            .max(mesh.weld_tolerance)
            .max(mesh.same_parameter_tol)
            .max(TOLERANCE_FLOOR)
    }

    /// Apply model tolerance to mesh config fields that default to hard-coded values.
    pub fn apply_to_mesh_config(&self, mesh: &mut crate::mesh::BRepMeshConfig) {
        mesh.same_parameter_tol = mesh.same_parameter_tol.max(self.model);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::BRepMeshConfig;

    #[test]
    fn boundary_dedup_composes_mesh_and_model() {
        let ctx = ToleranceContext::from_model(2e-4);
        let mut mesh = BRepMeshConfig::default();
        mesh.weld_tolerance = 1e-5;
        mesh.same_parameter_tol = 1e-4;
        assert!((ctx.boundary_dedup(&mesh) - 2e-4).abs() < 1e-8);
    }

    #[test]
    fn from_model_clamps_extremes() {
        let low = ToleranceContext::from_model(1e-8);
        assert_eq!(low.model, TOLERANCE_FLOOR);
        let high = ToleranceContext::from_model(1.0);
        assert_eq!(high.model, MAX_MODEL_TOLERANCE);
    }
}
