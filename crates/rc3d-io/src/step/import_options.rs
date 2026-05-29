//! STEP import strictness and quality reporting.

use rc3d_scene::SceneGraph;

use crate::step::adapter::{AdapterMode, AdapterOptions};
use crate::step::brep::heal::HealLevel;
use crate::step::parser::EntityIndex;
use crate::step::tree::AssemblyTree;

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
}

/// Full STEP import output (scene + assembly metadata for editor/tools).
pub struct StepImportResult {
    pub graph: SceneGraph,
    pub report: StepImportReport,
    pub assembly_tree: AssemblyTree,
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
        }
    }

    pub fn preview() -> Self {
        Self {
            mode: StepImportMode::Preview,
            heal_level: HealLevel::Standard,
            mesh_relative_deflection: 0.0,
            strict_schema: false,
            adapter_mode: AdapterMode::CompatMerge,
        }
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
    /// Part21 DATA section count (when fidelity reader used).
    pub data_section_count: usize,
    /// Complex entities using external mapping (Part21 reader).
    pub complex_external_count: usize,
    /// EXPRESS WR violations from pre-transfer schema check (Part21 path).
    pub schema_violations: usize,
}
