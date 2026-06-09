//! Pipeline orchestration and reporting types for S0–S7.
//!
//! Phase 2: types defined, wiring to import_step_file_with_options
//! deferred to Phase 3 when strategy chain is fully operational.

use rc3d_shape::mesh::config::TessellationTier;
use rc3d_shape::FaceKey;

// ── Pipeline stages ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    Parse,
    Validate,
    Build,
    Transfer,
    Heal,
    Tessellate,
    Emit,
}

// ── Warning codes ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarningCode {
    ParseSkippedEntity,
    ParseRecoveredEntity,
    BuildGeomFallback,
    BuildMissingSurface,
    HealSkipFace,
    TessellateSkipFace,
    TessellateDegradedQuality,
    ValidateDanglingRef,
    ValidateTopoAnomaly,
}

impl WarningCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            WarningCode::ParseSkippedEntity => "parse_skipped_entity",
            WarningCode::ParseRecoveredEntity => "parse_recovered_entity",
            WarningCode::BuildGeomFallback => "build_geom_fallback",
            WarningCode::BuildMissingSurface => "build_missing_surface",
            WarningCode::HealSkipFace => "heal_skip_face",
            WarningCode::TessellateSkipFace => "tessellate_skip_face",
            WarningCode::TessellateDegradedQuality => "tessellate_degraded_quality",
            WarningCode::ValidateDanglingRef => "validate_dangling_ref",
            WarningCode::ValidateTopoAnomaly => "validate_topo_anomaly",
        }
    }
}

// ── Pipeline warning ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PipelineWarning {
    pub stage: PipelineStage,
    pub code: WarningCode,
    pub step_entity_id: Option<u64>,
    pub face_key: Option<FaceKey>,
    pub message: String,
}

impl PipelineWarning {
    pub fn new(stage: PipelineStage, code: WarningCode, message: impl Into<String>) -> Self {
        Self {
            stage,
            code,
            step_entity_id: None,
            face_key: None,
            message: message.into(),
        }
    }

    pub fn with_entity(mut self, id: u64) -> Self {
        self.step_entity_id = Some(id);
        self
    }

    pub fn with_face(mut self, fk: FaceKey) -> Self {
        self.face_key = Some(fk);
        self
    }
}

// ── Stage outcome ────────────────────────────────────────────────────

/// Result of a single pipeline stage.
#[derive(Debug)]
pub enum StageOutcome<T> {
    /// Stage completed successfully.
    Ok(T),
    /// Stage completed with non-fatal issues.
    Degraded(T, Vec<PipelineWarning>),
    /// Stage failed completely.
    Failed(PipelineStage, String),
}

impl<T> StageOutcome<T> {
    pub fn is_ok(&self) -> bool {
        matches!(self, StageOutcome::Ok(_) | StageOutcome::Degraded(_, _))
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, StageOutcome::Failed(_, _))
    }

    /// Extract the inner value, panicking if Failed.
    /// Prefer `into_result()` for non-panicking error propagation.
    pub fn unwrap(self) -> T {
        match self {
            StageOutcome::Ok(v) | StageOutcome::Degraded(v, _) => v,
            StageOutcome::Failed(stage, msg) => {
                panic!("pipeline stage {:?} failed: {}", stage, msg)
            }
        }
    }

    /// Convert to `Result`, mapping `Failed` to an error string.
    pub fn into_result(self) -> Result<(T, Vec<PipelineWarning>), (PipelineStage, String)> {
        match self {
            StageOutcome::Ok(v) => Ok((v, Vec::new())),
            StageOutcome::Degraded(v, warnings) => Ok((v, warnings)),
            StageOutcome::Failed(stage, msg) => Err((stage, msg)),
        }
    }
}

// ── Aggregated report ────────────────────────────────────────────────

/// Summary of a complete S0–S7 pipeline run.
///
/// Carries diagnostics from each stage plus the tessellation tier used.
/// This is the primary output of `import_step_file_with_options`.
#[derive(Debug, Clone)]
pub struct ImportPipelineReport {
    pub tier: TessellationTier,
    /// Number of STEP entities parsed.
    pub entity_count: usize,
    /// Number of root solids found.
    pub solid_count: usize,
    /// Total faces across all solids.
    pub total_faces: usize,
    /// Faces successfully meshed.
    pub faces_meshed: usize,
    /// Faces skipped (heal or tessellation failure).
    pub faces_skipped: usize,
    /// Total output triangles.
    pub total_tris: usize,
    /// Warnings collected across all stages.
    pub warnings: Vec<PipelineWarning>,
    /// Per-stage timing in seconds (stage name → seconds).
    pub stage_timings: Vec<(PipelineStage, f32)>,
}

impl ImportPipelineReport {
    pub fn new(tier: TessellationTier) -> Self {
        Self {
            tier,
            entity_count: 0,
            solid_count: 0,
            total_faces: 0,
            faces_meshed: 0,
            faces_skipped: 0,
            total_tris: 0,
            warnings: Vec::new(),
            stage_timings: Vec::new(),
        }
    }

    /// Fraction of faces successfully meshed.
    pub fn mesh_success_rate(&self) -> f32 {
        if self.total_faces == 0 {
            0.0
        } else {
            self.faces_meshed as f32 / self.total_faces as f32
        }
    }

    /// True when all faces were meshed (no skip).
    pub fn is_watertight(&self) -> bool {
        self.faces_skipped == 0 && self.faces_meshed == self.total_faces
    }

    /// Number of warnings at a specific stage.
    pub fn warning_count_at(&self, stage: PipelineStage) -> usize {
        self.warnings.iter().filter(|w| w.stage == stage).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_default() {
        let r = ImportPipelineReport::new(TessellationTier::Standard);
        assert_eq!(r.entity_count, 0);
        assert_eq!(r.mesh_success_rate(), 0.0);
        // Empty shell is vacuously watertight (0 skipped, 0 faces)
        assert!(r.is_watertight());
    }

    #[test]
    fn test_warning_chain() {
        let w = PipelineWarning::new(
            PipelineStage::Build,
            WarningCode::BuildGeomFallback,
            "plane fallback for #42",
        )
        .with_entity(42);
        assert_eq!(w.stage, PipelineStage::Build);
        assert_eq!(w.step_entity_id, Some(42));
    }

    #[test]
    fn test_stage_outcome_ok() {
        let o: StageOutcome<usize> = StageOutcome::Ok(42);
        assert!(o.is_ok());
        assert_eq!(o.unwrap(), 42);
    }

    #[test]
    fn test_stage_outcome_failed() {
        let o: StageOutcome<usize> =
            StageOutcome::Failed(PipelineStage::Tessellate, "no triangles".into());
        assert!(o.is_failed());
    }
}
