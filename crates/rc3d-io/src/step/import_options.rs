//! STEP import strictness and quality reporting.

use rc3d_scene::SceneGraph;
use rc3d_shape::ShapeDocument;
use rc3d_shape::mesh::config::TessellationTier;

use crate::step::adapter::{AdapterMode, AdapterOptions};
use crate::step::brep::heal::HealLevel;
use rc3d_shape::topo::FaceKey;
use crate::step::parser::EntityIndex;

/// How strictly STEP import treats parse/build/heal issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepImportMode {
    /// Fail on skipped entities, geometry substitution, validation/heal errors, void shells.
    Strict,
    /// Best-effort preview: recover parse errors and substitute missing geometry.
    Preview,
}

impl Default for StepImportMode {
    fn default() -> Self {
        Self::Preview
    }
}

#[derive(Debug, Clone)]
pub struct StepImportOptions {
    pub mode: StepImportMode,
    /// Auto-heal level for B-Rep shells (default Advanced for industrial STEP).
    pub heal_level: HealLevel,
    /// When > 0, mesh deflection scales with shell bbox diagonal (OCC relative mode).
    pub mesh_relative_deflection: f32,
    /// Run AP242 WR subset on Part21 instances before adapter transfer.
    pub strict_schema: bool,
    /// Complex-entity flattening mode for Part21 → `EntityIndex` adapter.
    pub adapter_mode: AdapterMode,
    /// Radial preview separation for multi-solid assemblies (0 = off).
    /// In Preview mode, overlapping solid bboxes still auto-apply 0.35 when this is zero.
    pub assembly_preview_explode: f32,
    /// Skip visualization-only work (edge overlay, wireframe, mesh properties).
    /// Set true for non-interactive use cases like STL export.
    pub skip_visualization: bool,
    /// When true, attempt to boolean-subtract void shells from outer shells
    /// by projecting void wires as inner wires (holes) on matching outer faces.
    pub strict_voids: bool,
    /// Enable fast-export mesh mode: skips post-refine, optimization, chord checks.
    pub fast_export: bool,
    /// Tessellation quality tier — drives heal level, mesh config, and fallback policy.
    /// When set, overrides `heal_level` and `fast_export` with tier-appropriate values.
    pub tessellation_tier: Option<TessellationTier>,
}

/// Full STEP import output (scene + shape document + assembly metadata).
pub struct StepImportResult {
    pub document: ShapeDocument,
    pub graph: SceneGraph,
    pub report: StepImportReport,
    pub entities: EntityIndex,
}

impl StepImportOptions {
    pub fn strict() -> Self {
        Self {
            mode: StepImportMode::Strict,
            heal_level: HealLevel::Standard,
            mesh_relative_deflection: 0.0,
            strict_schema: false,
            adapter_mode: AdapterMode::CompatMerge,
            assembly_preview_explode: 0.0,
            skip_visualization: false,
            strict_voids: false,
            fast_export: false,
            tessellation_tier: Some(TessellationTier::Precision),
        }
    }

    pub fn preview() -> Self {
        Self {
            mode: StepImportMode::Preview,
            heal_level: HealLevel::Standard,
            mesh_relative_deflection: 0.0,
            strict_schema: false,
            adapter_mode: AdapterMode::CompatMerge,
            assembly_preview_explode: 0.0,
            skip_visualization: false,
            strict_voids: false,
            fast_export: false,
            tessellation_tier: Some(TessellationTier::Preview),
        }
    }

    /// Create options for a specific tessellation tier.
    pub fn for_tier(tier: TessellationTier) -> Self {
        let mode = match tier {
            TessellationTier::Precision => StepImportMode::Strict,
            _ => StepImportMode::Preview,
        };
        let heal_level = match tier {
            TessellationTier::Preview => HealLevel::Basic,
            _ => HealLevel::Standard,
        };
        Self {
            mode,
            heal_level,
            mesh_relative_deflection: 0.0,
            strict_schema: tier == TessellationTier::Precision,
            adapter_mode: AdapterMode::CompatMerge,
            assembly_preview_explode: 0.0,
            skip_visualization: false,
            strict_voids: tier == TessellationTier::Precision,
            fast_export: tier == TessellationTier::Preview,
            tessellation_tier: Some(tier),
        }
    }

    /// Resolve the effective tessellation tier (explicit or derived from mode).
    pub fn effective_tier(&self) -> TessellationTier {
        self.tessellation_tier.unwrap_or(match self.mode {
            StepImportMode::Strict => TessellationTier::Precision,
            StepImportMode::Preview => TessellationTier::Preview,
        })
    }

    pub fn adapter_options(&self) -> AdapterOptions {
        AdapterOptions {
            mode: self.adapter_mode,
        }
    }

    pub fn recover_skipped_entities(&self) -> bool {
        matches!(self.mode, StepImportMode::Preview)
    }

    pub fn allow_geometry_fallback(&self) -> bool {
        matches!(self.mode, StepImportMode::Preview)
    }

    pub fn allow_void_shells_unmeshed(&self) -> bool {
        matches!(self.mode, StepImportMode::Preview)
    }

    pub fn fail_on_validation_errors(&self) -> bool {
        matches!(self.mode, StepImportMode::Strict)
    }

    /// Heal check errors are logged in `StepImportReport` but do not fail import:
    /// many valid industrial STEP files report non-manifold/heal diagnostics.
    pub fn fail_on_heal_check_errors(&self) -> bool {
        false
    }
}

impl Default for StepImportOptions {
    fn default() -> Self {
        Self::preview()
    }
}

#[derive(Debug, Default, Clone)]
pub struct StepImportReport {
    pub skipped_parse_entities: usize,
    pub skipped_faces: usize,
    pub skipped_edges: usize,
    pub void_shell_count: usize,
    pub validation_errors: usize,
    pub heal_check_errors: usize,
    pub unknown_entity_count: usize,
    pub continuity_defects: usize,
    /// Detected AP schema (e.g. "AP242", "AP203", "AP214", or None if unrecognized).
    pub ap_schema: Option<String>,
    /// Product nodes in assembly tree with at least one shell.
    pub assembly_node_count: usize,
    /// NAUO children referenced from more than one parent (DAG, not tree).
    pub assembly_multi_parent_pd_count: usize,
    /// NAUO edges ignored by legacy single-parent transform chain.
    pub assembly_dropped_parent_link_count: usize,
    /// Distinct (shell, transform) placements emitted for rendering/export.
    pub assembly_shell_instance_count: usize,
    /// Part21 DATA section count (when fidelity reader used).
    pub data_section_count: usize,
    /// Complex entities using external mapping (Part21 reader).
    pub complex_external_count: usize,
    /// EXPRESS WR violations from pre-transfer schema check (Part21 path).
    pub schema_violations: usize,
    /// Face orientation counters from STEP ORIENTED_FACE/ADVANCED_FACE mapping.
    pub oriented_forward_faces: usize,
    pub oriented_reversed_faces: usize,
    /// Number of void faces successfully punched as inner wires (strict_voids mode).
    pub void_shells_subtracted: usize,
    /// Faces skipped by the heal pipeline (passed to mesh tessellation).
    pub heal_skip_face_keys: Vec<FaceKey>,
    /// Unknown surfaces/curves replaced with plane/line fallback (Preview mode).
    pub geometry_fallback_count: usize,
}
