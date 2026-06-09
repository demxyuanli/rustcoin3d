//! Face mesher trait + strategy registry (OCC BRepMesh_MeshAlgoFactory replacement).
//!
//! Replaces the static `FaceMeshAlgo` enum dispatch with a trait-based registry
//! that supports retry hints and per-tier fallback configuration.

use std::collections::HashMap;

use rc3d_core::math::Vec3;

use crate::mesh::config::TessellationPolicy;
use crate::mesh::face_dispatch::FaceMeshPlan;
use crate::mesh_result::MeshResult;
use crate::store::BRepStore;
use crate::topo::{BRepFace, EdgeKey, FaceKey};

use super::edge_disc::EdgePolygon;
use super::face_fill::FaceMeshRange;
use super::face_uv::FaceUvLoops;

// ── Strategy identifiers ─────────────────────────────────────────────

/// Registered mesh strategy IDs (ordered by priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaceMeshStrategyId {
    ClosedParametric,
    TrimmedCdt,
    RuledStrip,
    ParametricGrid,
}

// ── Context types ────────────────────────────────────────────────────

/// Read-only context passed to `can_handle`.
#[derive(Debug, Clone)]
pub struct FaceContext {
    pub face_key: FaceKey,
    pub face: BRepFace,
    pub loops: FaceUvLoops,
    pub plan: FaceMeshPlan,
    pub policy: TessellationPolicy,
}

/// Mutable context for mesh execution.
pub struct FaceMeshMutContext<'a> {
    pub reg: &'a BRepStore,
    pub mesh: &'a mut MeshResult,
    pub global_vertices: &'a mut Vec<Vec3>,
    pub global_normals: &'a mut Vec<Vec3>,
    /// Output index buffer (appended per face).
    pub all_indices: &'a mut Vec<i32>,
    /// Maps edge keys to discretized polygon points.
    pub edge_polygons: &'a mut HashMap<EdgeKey, EdgePolygon>,
    /// Edge → boundary vertex index mapping for ruled meshes.
    pub edge_boundary_idx: &'a super::edge_pool::FaceEdgeBoundaryIdx,
    /// Boundary point deduplication index (quantized 3D position → vertex index).
    pub boundary_pos_to_idx: &'a mut super::boundary::BoundaryPosIndex,
    /// Fill config for this shell's meshing.
    pub config: &'a mut super::face_fill::FaceFillConfig,
    /// Read-only boundary pool (None for first chunk).
    pub shared_boundary: Option<&'a super::boundary::SharedBoundaryPool>,
}

/// Phase 3 migration: FaceStrategyChain expects bundled context.
/// Callers that currently pass individual `&mut` refs can transition to
/// `FaceMeshMutContext::from_parts(...)` to enable per-face strategy dispatch.
impl<'a> FaceMeshMutContext<'a> {
    /// Construct from individual mutable borrows (same lifetime).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        reg: &'a BRepStore,
        mesh: &'a mut MeshResult,
        global_vertices: &'a mut Vec<Vec3>,
        global_normals: &'a mut Vec<Vec3>,
        all_indices: &'a mut Vec<i32>,
        edge_polygons: &'a mut HashMap<EdgeKey, EdgePolygon>,
        edge_boundary_idx: &'a super::edge_pool::FaceEdgeBoundaryIdx,
        boundary_pos_to_idx: &'a mut super::boundary::BoundaryPosIndex,
        config: &'a mut super::face_fill::FaceFillConfig,
        shared_boundary: Option<&'a super::boundary::SharedBoundaryPool>,
    ) -> Self {
        Self { reg, mesh, global_vertices, global_normals, all_indices,
               edge_polygons, edge_boundary_idx, boundary_pos_to_idx, config, shared_boundary }
    }
}

// ── Outcome types ────────────────────────────────────────────────────

/// Result of a face meshing attempt.
#[derive(Debug, Clone)]
pub enum FaceMeshOutcome {
    /// Successfully meshed the face.
    Ok(FaceMeshRange),
    /// Strategy cannot handle this face — try next.
    Unavailable,
    /// Strategy can handle but failed — use retry hint before giving up.
    Retry(RetryHint),
    /// Strategy failed permanently for this face.
    Failed(FaceFailReason),
}

/// Suggested remedial action before retrying meshing.
#[derive(Debug, Clone, Copy)]
pub enum RetryHint {
    /// Increase edge discretization density and retry.
    SubdivideBoundaryEdges,
    /// Rebuild UV loops with a different source.
    RebuildUvLoops,
    /// Increase Steiner point budget for CDT.
    IncreaseSteiner,
    /// Use a finer parametric grid.
    FinerParametricGrid,
}

/// Why a face mesh attempt failed definitively.
#[derive(Debug, Clone)]
pub enum FaceFailReason {
    /// UV loops are degenerate and none of the repair strategies succeeded.
    InvalidUvDomain,
    /// CDT constraint insertion failed (self-intersecting trim).
    CdtConstraintFailed,
    /// Surface projection failure rate too high.
    ProjectionFailure(f32),
    /// Quality gate rejected the output mesh.
    QualityGateRejected(String),
}

// ── FaceMesher trait ─────────────────────────────────────────────────

/// A pluggable face meshing strategy.
///
/// Implementations register in the strategy chain via `FaceMesherRegistry`.
/// Each impl handles one algorithm (ClosedParametric, TrimmedCdt, RuledStrip, etc.).
pub trait FaceMesher: std::fmt::Debug {
    /// Which strategy this implements.
    fn id(&self) -> FaceMeshStrategyId;

    /// Quick check: can this strategy attempt to mesh the given face?
    fn can_handle(&self, plan: FaceMeshPlan, ctx: &FaceContext) -> bool;

    /// Execute meshing with face geometry and mutable mesh context.
    fn mesh(&self, face_ctx: &FaceContext, mesh_ctx: &mut FaceMeshMutContext<'_>) -> FaceMeshOutcome;
}

// ── Strategy registry ────────────────────────────────────────────────

/// Ordered registry of face meshing strategies.
///
/// Strategies are tried in insertion order (highest priority first).
/// The chain stops at the first `Ok` or `Failed`; `Unavailable` moves
/// to the next strategy; `Retry` loops back with the hint applied.
pub struct FaceMesherRegistry {
    strategies: Vec<Box<dyn FaceMesher>>,
    #[allow(dead_code)]
    max_retries_per_hint: usize,
}

impl FaceMesherRegistry {
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            max_retries_per_hint: 2,
        }
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            strategies: Vec::with_capacity(n),
            max_retries_per_hint: 2,
        }
    }

    /// Register a strategy (appended to priority chain).
    pub fn register(&mut self, strategy: Box<dyn FaceMesher>) {
        self.strategies.push(strategy);
    }

    /// Number of registered strategies.
    pub fn len(&self) -> usize {
        self.strategies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strategies.is_empty()
    }

    /// Iterate strategies in priority order.
    pub fn iter(&self) -> impl Iterator<Item = &Box<dyn FaceMesher>> {
        self.strategies.iter()
    }
}

impl Default for FaceMesherRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_empty() {
        let reg = FaceMesherRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_with_capacity() {
        let reg = FaceMesherRegistry::with_capacity(4);
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }
}
