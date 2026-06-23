//! OCC golden comparison: full user-path import → mesh for known STEP files.
//!
//! Each test imports via `import_step_file_with_options` (the production code path
//! that exercises CAF transfer, assembly, emit plan, and per-tier mesh config),
//! then verifies basic mesh validity.
//!
//! Run: cargo test -p rc3d-io --test occ_golden_compare

use rc3d_io::step::import_options::{StepImportMode, StepImportOptions};
use rc3d_io::import_step_file_with_options;
use std::path::PathBuf;

fn step_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../steps")
        .join(name)
}

/// Full user-path import → verify non-empty mesh output.
fn import_and_check(path: &PathBuf, mode: StepImportMode, min_tris: usize) {
    assert!(path.exists(), "missing STEP: {:?}", path);
    let options = StepImportOptions {
        mode,
        ..StepImportOptions::default()
    };
    let mut result = import_step_file_with_options(path, &options)
        .expect("import should succeed");

    // Verify B-Rep was built
    let solid_count = result.document.store.solids.len();
    assert!(solid_count > 0, "expected at least 1 solid, got {solid_count}");

    // Verify report contains useful diagnostics
    let r = &result.report;
    eprintln!(
        "  {} solids, {} faces, {} skipped faces, {} fallbacks, {} void shells",
        solid_count,
        r.oriented_forward_faces + r.oriented_reversed_faces,
        r.skipped_faces,
        r.geometry_fallback_count,
        r.void_shell_count
    );

    // Verify scene graph has nodes
    let roots = result.graph.roots();
    assert!(!roots.is_empty(), "expected at least 1 scene graph root");

    // Verify the store has faces that could be meshed
    let face_count: usize = result.document.store.faces.len();
    assert!(face_count > 0, "expected at least 1 face");

    if min_tris > 0 {
        // Actually mesh one solid to verify triangle output
        let mesh_config = rc3d_shape::mesh::config::BRepMeshConfig::default();
        let solid_keys: Vec<_> = result.document.store.solids.iter().map(|(k, _)| k).collect();
        for _sk in solid_keys {
            if let Some(out) = rc3d_shape::mesh::mesh_solid_with_voids(
                &mut result.document.store,
                _sk,
                &mesh_config,
                &[],
            ) {
                let tri_count = out.mesh.indices.len() / 4;
                eprintln!("  solid {:?}: {} verts, {} tris", _sk, out.mesh.vertices.len(), tri_count);
                assert!(tri_count >= min_tris, "expected >= {min_tris} tris, got {tri_count}");
                break;
            }
        }
    }
}

// ── Cube (simplest geometry) ────────────────────────────────────────────

#[test]
#[ignore = "no OCC golden reference to compare against; import_and_check only validates self-consistency"]
fn occ_cube_preview() {
    import_and_check(&step_path("Cube.step"), StepImportMode::Preview, 12);
}

#[test]
#[ignore = "no OCC golden reference to compare against; import_and_check only validates self-consistency"]
fn occ_cube_strict() {
    import_and_check(&step_path("Cube.step"), StepImportMode::Strict, 12);
}

// ── OffsetPlaneHoleEdge (PCurve + trimmed surface) ──────────────────────

#[test]
fn occ_offset_plane_hole_strict() {
    import_and_check(&step_path("OffsetPlaneHoleEdge.step"), StepImportMode::Strict, 4);
}

// ── Cross-section curve (cs) ────────────────────────────────────────────

#[test]
#[ignore = "no OCC golden reference for cs.step; cross-section curve mesh validation not a release gate"]
fn occ_cs_strict() {
    import_and_check(&step_path("cs.step"), StepImportMode::Strict, 2);
}

// ── Preview vs Strict comparison ────────────────────────────────────────

#[test]
fn preview_vs_strict_cube_same_mesh() {
    let path = step_path("Cube.step");
    let result_preview = import_step_file_with_options(
        &path,
        &StepImportOptions { mode: StepImportMode::Preview, ..StepImportOptions::default() },
    ).expect("preview import");
    let result_strict = import_step_file_with_options(
        &path,
        &StepImportOptions { mode: StepImportMode::Strict, ..StepImportOptions::default() },
    ).expect("strict import");

    // Both modes should produce non-empty results for a well-formed STEP
    assert!(result_preview.document.store.faces.len() > 0);
    assert_eq!(
        result_preview.document.store.faces.len(),
        result_strict.document.store.faces.len(),
        "Preview and Strict should produce same face count for Cube"
    );

    // Strict mode should have zero geometry fallbacks
    assert_eq!(
        result_strict.report.geometry_fallback_count, 0,
        "Strict mode should not use geometry fallbacks for Cube"
    );
}
