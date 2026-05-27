//! STEP import strictness and quality reporting.

use crate::step::brep::heal::HealLevel;

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
}

impl StepImportOptions {
    pub fn strict() -> Self {
        Self {
            mode: StepImportMode::Strict,
            heal_level: HealLevel::Standard,
            mesh_relative_deflection: 0.0,
        }
    }

    pub fn preview() -> Self {
        Self {
            mode: StepImportMode::Preview,
            heal_level: HealLevel::Standard,
            mesh_relative_deflection: 0.0,
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
}
