//! STEP import orchestration helpers (heal, continuity checks, mesh props).

use rc3d_core::math::Vec3;

use crate::step::brep::heal::{auto_heal_shell, check_shell_continuity, HealLevel, HealReport};
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::SolidKey;
use crate::step::import_options::StepImportReport;
use crate::step::mesh_result::MeshResult;

/// Run auto-heal on all root solids and aggregate reports.
pub fn run_heal_pipeline(
    reg: &mut BRepRegistry,
    root_solids: &[SolidKey],
    heal_level: HealLevel,
    max_iterations: usize,
) -> HealReport {
    let mut total_heal = HealReport::default();
    for &sk in root_solids {
        let shell_keys: Vec<_> = reg
            .solids
            .get(sk)
            .map(|s| {
                let mut keys = Vec::with_capacity(1 + s.void_shells.len());
                keys.push(s.outer_shell);
                keys.extend(s.void_shells.iter().copied());
                keys
            })
            .unwrap_or_default();
        for shell_key in shell_keys {
            total_heal.merge(auto_heal_shell(
                shell_key,
                reg,
                heal_level,
                max_iterations,
            ));
        }
    }
    log::info!("[STEP] healed: {:?}", total_heal);
    total_heal
}

/// Continuity check pass; updates `import_report.continuity_defects`.
pub fn run_continuity_checks(
    reg: &BRepRegistry,
    root_solids: &[SolidKey],
    g0_tol: f32,
    import_report: &mut StepImportReport,
) {
    for &sk in root_solids {
        let shell_keys: Vec<_> = reg
            .solids
            .get(sk)
            .map(|s| {
                let mut keys = Vec::with_capacity(1 + s.void_shells.len());
                keys.push(s.outer_shell);
                keys.extend(s.void_shells.iter().copied());
                keys
            })
            .unwrap_or_default();
        for shell_key in shell_keys {
            let defects = check_shell_continuity(shell_key, reg, g0_tol, 5.0);
            import_report.continuity_defects += defects.len();
            if !defects.is_empty() {
                log::info!(
                    "[STEP] shell {:?}: {} continuity defect(s)",
                    shell_key,
                    defects.len()
                );
            }
        }
    }
}

/// Public helper for aggregating mesh properties from emit plan meshes.
pub fn append_props_mesh_public(
    mesh: &MeshResult,
    props_vertices: &mut Vec<Vec3>,
    props_indices: &mut Vec<i32>,
) {
    let base_offset = props_vertices.len() as i32;
    props_vertices.extend_from_slice(&mesh.vertices);
    for chunk in mesh.indices.chunks(4) {
        if chunk.len() >= 3 {
            props_indices.extend_from_slice(&[
                chunk[0] + base_offset,
                chunk[1] + base_offset,
                chunk[2] + base_offset,
                -1,
            ]);
        }
    }
}
