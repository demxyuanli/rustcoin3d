//! BREP verification loop: batch import STEP files, write .brep, diff topology.

use rc3d_io::step::{import_step_file_with_options, StepImportOptions};
use rc3d_shape::brep::write_brep;
use std::path::PathBuf;

// Re-use the diff tools defined in brep_diff.rs
mod brep_diff;
use brep_diff::TopoCounts;

/// Files used for verification (from export_step_stl corpus).
const CORPUS: &[&str] = &[
    "Shape.step",
    "Shape-1.step",
    "Shape-2.step",
    "Cube.step",
    "cs.step",
    "OffsetPlaneHoleEdge.step",
    "rev.step",
    "asse.step",
    "HoledPlate.step",
];

/// Resolve test data path (same convention as export_step_stl).
fn test_data_path(name: &str) -> PathBuf {
    let primary = std::env::var("RC3D_STEP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../steps")
        });
    let candidate = primary.join(name);
    if candidate.exists() {
        return candidate;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

#[test]
fn brep_verify_all_import() {
    println!("\n=== BREP Import Verification ({}) ===\n", CORPUS.len());
    let mut success = 0;
    let mut failed = 0;
    let mut counts_list: Vec<(&str, TopoCounts)> = Vec::new();

    for name in CORPUS {
        let path = test_data_path(name);
        if !path.exists() {
            eprintln!("SKIP: {} not found", name);
            continue;
        }
        match import_step_file_with_options(&path, &StepImportOptions::default()) {
            Ok(result) => {
                let store = &result.document.store;
                let counts = TopoCounts {
                    vertices: store.vertices.len(),
                    edges: store.edges.len(),
                    wires: store.wires.len(),
                    faces: store.faces.len(),
                    shells: store.shells.len(),
                    solids: store.solids.len(),
                    compounds: store.compounds.len(),
                    pcurves_total: store.edges.values().map(|e| e.pcurves.len()).sum(),
                };
                println!(
                    "  ✓ {}  V={} E={} W={} F={} Sh={} So={} PC={}",
                    name,
                    counts.vertices,
                    counts.edges,
                    counts.wires,
                    counts.faces,
                    counts.shells,
                    counts.solids,
                    counts.pcurves_total
                );
                counts_list.push((name, counts));
                success += 1;
            }
            Err(e) => {
                println!("  ✗ {}  FAILED: {}", name, e);
                failed += 1;
            }
        }
    }

    println!(
        "\n=== Summary: {} success, {} failed ===",
        success, failed
    );
    assert!(success > 0, "at least one file must import successfully");
}

#[test]
fn brep_write_all_corpus() {
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_output/brep");
    let _ = std::fs::create_dir_all(&out_dir);

    println!("\n=== BREP Export ({}) ===\n", CORPUS.len());
    let mut written = 0;

    for name in CORPUS {
        let path = test_data_path(name);
        if !path.exists() {
            continue;
        }
        let result = match import_step_file_with_options(&path, &StepImportOptions::default()) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  SKIP {}: import failed ({})", name, e);
                continue;
            }
        };

        let brep_path = out_dir.join(name).with_extension("brep");
        let mut file = match std::fs::File::create(&brep_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  SKIP {}: cannot create file ({})", name, e);
                continue;
            }
        };

        match write_brep(&result.document.store, &mut file) {
            Ok(()) => {
                let size = std::fs::metadata(&brep_path).unwrap().len();
                println!("  ✓ {} -> {} ({} bytes)", name, brep_path.display(), size);
                written += 1;
            }
            Err(e) => {
                eprintln!("  ✗ {}: write failed ({})", name, e);
            }
        }
    }

    println!(
        "\n=== {} BREP files written to {} ===",
        written,
        out_dir.display()
    );
    assert!(written > 0, "at least one BREP file must be written");
}

#[test]
fn brep_corpus_topology_summary() {
    println!("\n=== Corpus Topology Summary ===\n");
    println!(
        "{:<30} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "File", "V", "E", "F", "Sh", "So", "PC"
    );
    println!("{}", "-".repeat(65));

    for name in CORPUS {
        let path = test_data_path(name);
        if !path.exists() {
            println!("{:<30} {:>5}", name, "N/A");
            continue;
        }
        match import_step_file_with_options(&path, &StepImportOptions::default()) {
            Ok(result) => {
                let store = &result.document.store;
                let pc = store.edges.values().map(|e| e.pcurves.len()).sum::<usize>();
                println!(
                    "{:<30} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
                    name,
                    store.vertices.len(),
                    store.edges.len(),
                    store.faces.len(),
                    store.shells.len(),
                    store.solids.len(),
                    pc
                );
            }
            Err(_) => {
                println!("{:<30} {:>5}", name, "FAIL");
            }
        }
    }
    println!();
}
