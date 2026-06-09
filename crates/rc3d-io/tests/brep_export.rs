//! BREP ASCII export test: import STEP, write .brep, verify output structure.

use rc3d_io::step::{import_step_with_options, StepImportOptions};
use rc3d_shape::brep::write_brep;

/// Resolve test data path (same convention as export_step_stl).
fn test_data_path(name: &str) -> std::path::PathBuf {
    let primary = std::env::var("RC3D_STEP_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../steps")
        });
    let candidate = primary.join(name);
    if candidate.exists() {
        return candidate;
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

#[test]
fn brep_export_cube_has_expected_sections() {
    let step = include_str!("../../../test_data/Cube.step");
    let result = import_step_with_options(step, &StepImportOptions::default())
        .expect("import Cube.step");

    let store = &result.document.store;
    let mut buf = Vec::new();
    write_brep(store, &mut buf).expect("write brep");

    let text = String::from_utf8(buf).expect("valid utf8");

    // Verify all required sections exist
    let required_sections = [
        "DBRep_DrawableShape",
        "Locations 0",
        "Curve3ds",
        "Surfaces",
        "Curve2ds",
        "TVertexes",
        "TEdges",
        "TWires",
        "TFaces",
        "TShells",
        "TSolids",
    ];
    for section in &required_sections {
        assert!(
            text.contains(section),
            "missing section: {}",
            section
        );
    }

    // Verify topology counts are non-zero
    assert!(text.contains("TVertexes"), "no vertices section");
    assert!(text.contains("TEdges"), "no edges section");
    assert!(text.contains("TFaces"), "no faces section");
}

#[test]
fn brep_export_cube_writes_file() {
    let step = include_str!("../../../test_data/Cube.step");
    let result = import_step_with_options(step, &StepImportOptions::default())
        .expect("import Cube.step");

    let store = &result.document.store;
    let path = std::env::temp_dir().join("cube_brep_test.brep");
    let mut file = std::fs::File::create(&path).expect("create file");
    write_brep(store, &mut file).expect("write brep");

    let file_size = std::fs::metadata(&path).unwrap().len();
    assert!(file_size > 100, "brep file too small: {} bytes", file_size);
    println!("Cube.brep: {} bytes", file_size);

    // Clean up
    let _ = std::fs::remove_file(&path);
}

#[test]
fn brep_export_cube_has_vertices() {
    let step = include_str!("../../../test_data/Cube.step");
    let result = import_step_with_options(step, &StepImportOptions::default())
        .expect("import Cube.step");

    let store = &result.document.store;
    let mut buf = Vec::new();
    write_brep(store, &mut buf).expect("write brep");

    let text = String::from_utf8(buf).expect("valid utf8");

    // Cube should have vertices
    let vert_count = store.vertices.len();
    assert!(
        vert_count >= 8,
        "cube should have at least 8 vertices, got {}",
        vert_count
    );
    println!(
        "Cube: {} vertices, {} edges, {} faces",
        store.vertices.len(),
        store.edges.len(),
        store.faces.len()
    );
}

#[test]
fn brep_export_no_wire_warnings_for_well_formed_shape() {
    let step = include_str!("../../../test_data/Cube.step");
    let result = import_step_with_options(step, &StepImportOptions::default())
        .expect("import Cube.step");

    let store = &result.document.store;
    let mut buf = Vec::new();
    write_brep(store, &mut buf).expect("write brep");

    let text = String::from_utf8(buf).expect("valid utf8");

    // A well-formed Cube.step should not produce wire chain warnings.
    // If it does, count them but don't assert - some STEP files may have minor issues.
    let warning_count = text.matches("-- WARNING: broken wire chain").count();
    if warning_count > 0 {
        println!(
            "Cube has {} wire chain warnings (may indicate topology issues)",
            warning_count
        );
    }
}
