//! UV loop builders: single-source strategies for constructing FaceUvLoops.
//!
//! The trait-based design allows the strategy chain to select and retry
//! different UV sources without scattering face-specific logic.
//!
//! Phase 1 will add concrete builders (Pcurve, NativeSurface, Projection, Repair);
//! Phase 0 defines the contract only.

use crate::store::BRepStore;
use crate::topo::{BRepFace, FaceKey};

use super::face_uv::FaceUvLoops;

/// Outcome of UV loop construction.
#[derive(Debug, Clone)]
pub enum UvLoopOutcome {
    /// Successfully built UV loops.
    Ok(FaceUvLoops),
    /// Could not build with this strategy — try next.
    Unavailable,
    /// Builder succeeded but loops are degenerate (retry with repair).
    Degenerate(FaceUvLoops),
}

/// Builds UV loops for a face from a single source (PCURVE, native, projection, repair).
///
/// Each builder is responsible for one UV source strategy. The strategy chain
/// tries builders in priority order until one returns `Ok`.
///
/// Concrete implementations (Phase 1+):
/// - `PcurveLoopBuilder` — default, uses STEP PCurves
/// - `NativeSurfaceUvBuilder` — analytic/NURBS native UV domain
/// - `ProjectionLoopBuilder` — surface projection (last resort)
/// - `CylinderRepairBuilder` / `RevolutionRepairBuilder` — specialized repairs
pub trait UvLoopBuilder: std::fmt::Debug {
    /// Human-readable identifier for diagnostics.
    fn name(&self) -> &'static str;

    /// Build UV loops for the face.
    fn build(
        &self,
        face_key: FaceKey,
        face: &BRepFace,
        reg: &BRepStore,
    ) -> UvLoopOutcome;
}

// ── Builder registry placeholder ─────────────────────────────────────

/// Ordered list of UV loop builders to try (first success wins).
///
/// Phase 1 will populate with concrete builders.
pub fn default_uv_builders() -> Vec<Box<dyn UvLoopBuilder>> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uv_builder_registry_empty() {
        let builders = default_uv_builders();
        assert!(builders.is_empty());
    }
}
