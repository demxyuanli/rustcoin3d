//! Per-face strategy chain: plan → try → retry → quality gate.
//!
//! Replaces the inline if-else fallback tree in `shell_impl.rs`.
//! The chain owns a `FaceMesherRegistry` and drives the face through
//! its strategy sequence, applying retry hints and enforcing quality gates.

use crate::mesh::config::TessellationPolicy;
use crate::mesh::face_dispatch::plan_face_mesh;
use crate::mesh::face_mesher::{
    FaceContext, FaceFailReason, FaceMeshMutContext, FaceMeshOutcome, FaceMesherRegistry, RetryHint,
};
use crate::mesh::face_uv::FaceUvLoops;
use crate::topo::{BRepFace, FaceKey};

/// Drives a single face through the meshing strategy chain.
///
/// Owns the strategy registry; constructed once per shell, invoked per face.
pub struct FaceStrategyChain {
    registry: FaceMesherRegistry,
    policy: TessellationPolicy,
}

impl FaceStrategyChain {
    pub fn new(registry: FaceMesherRegistry, policy: TessellationPolicy) -> Self {
        Self { registry, policy }
    }

    /// Mesh one face. Returns the outcome and a flag indicating whether
    /// the face should be skipped (true = no mesh, continue to next face).
    pub fn mesh_face(
        &self,
        face_key: FaceKey,
        face: &BRepFace,
        loops: &FaceUvLoops,
        wire_empty: bool,
        heal_skipped: bool,
        ctx: &mut FaceMeshMutContext<'_>,
    ) -> (FaceMeshOutcome, bool) {
        let plan = plan_face_mesh(face, loops, wire_empty, heal_skipped);
        // Clone once — FaceContext is read-only and shared across all strategy attempts.
        let face_ctx = FaceContext {
            face_key,
            face: face.clone(),
            loops: loops.clone(),
            plan,
            policy: self.policy.clone(),
        };

        // Try each strategy in priority order.
        for strategy in self.registry.iter() {
            if !strategy.can_handle(plan, &face_ctx) {
                continue;
            }
            let outcome = strategy.mesh(&face_ctx, ctx);
            match &outcome {
                FaceMeshOutcome::Ok(_) => return (outcome, false),
                FaceMeshOutcome::Failed(_) => return (outcome, true),
                FaceMeshOutcome::Unavailable => continue,
                FaceMeshOutcome::Retry(hint) => {
                    if let Some(retry_outcome) =
                        self.retry_with_hint(strategy.as_ref(), *hint, &face_ctx, ctx)
                    {
                        return (retry_outcome, false);
                    }
                    continue;
                }
            }
        }

        // All strategies exhausted.
        let reason = if heal_skipped {
            FaceFailReason::InvalidUvDomain
        } else {
            FaceFailReason::CdtConstraintFailed
        };
        (FaceMeshOutcome::Failed(reason), true)
    }

    /// Apply a retry hint and re-invoke the strategy once.
    fn retry_with_hint(
        &self,
        strategy: &dyn crate::mesh::face_mesher::FaceMesher,
        hint: RetryHint,
        face_ctx: &FaceContext,
        ctx: &mut FaceMeshMutContext<'_>,
    ) -> Option<FaceMeshOutcome> {
        match hint {
            RetryHint::SubdivideBoundaryEdges => {
                for poly in ctx.edge_polygons.values_mut() {
                    poly.subdivide();
                }
                Some(strategy.mesh(face_ctx, ctx))
            }
            RetryHint::IncreaseSteiner => {
                let orig = ctx.config.max_cdt_vertices;
                ctx.config.max_cdt_vertices = (orig * 2).min(16384);
                let outcome = strategy.mesh(face_ctx, ctx);
                ctx.config.max_cdt_vertices = orig;
                Some(outcome)
            }
            RetryHint::FinerParametricGrid => {
                let orig = ctx.config.parameter_division_max_depth;
                ctx.config.parameter_division_max_depth = (orig + 2).min(10);
                let outcome = strategy.mesh(face_ctx, ctx);
                ctx.config.parameter_division_max_depth = orig;
                Some(outcome)
            }
            RetryHint::RebuildUvLoops => None,
        }
    }

    /// Whether the chain has any registered strategies.
    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_empty_registry() {
        let policy = TessellationPolicy::default();
        let registry = FaceMesherRegistry::new();
        let chain = FaceStrategyChain::new(registry, policy);
        assert!(chain.is_empty());
    }
}
