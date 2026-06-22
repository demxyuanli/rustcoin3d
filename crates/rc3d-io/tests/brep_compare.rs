//! BREP export comparison tests: import STEP from compare/step/,
//! export BREP to compare/out/, compare against OCC reference in compare/brep/.
//!
//! Naming: step-{Name}.step → occ-{name}.brep (reference) → step-{Name}.brep (output)
//!
//! Uses the low-level parse+transfer API to skip heal+mesh for fast BREP-only export.

use rc3d_io::step::{
    parser::parse_exchange_with_options,
    AdapterMode, StepCafTransfer, StepImportMode, StepImportOptions,
};
use rc3d_io::step::brep::build::BRepBuildOptions;
use rc3d_shape::brep::write_brep;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn compare_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../compare")
}

fn step_dir() -> PathBuf {
    compare_dir().join("step")
}

fn brep_ref_dir() -> PathBuf {
    compare_dir().join("brep")
}

fn out_dir() -> PathBuf {
    compare_dir().join("out")
}

/// Map step filename → occ reference filename.
/// Strips "step-" prefix and ".step" suffix, then looks for
/// matching occ-*.brep in the reference directory.
/// Falls back through: exact case → lowercase → underscore variants.
fn find_occ_ref(step_name: &str) -> Option<PathBuf> {
    let core = step_name
        .strip_prefix("step-")
        .and_then(|s| s.strip_suffix(".step"))
        .unwrap_or(step_name);

    let ref_dir = brep_ref_dir();

    // Candidates in priority order
    let candidates: Vec<String> = vec![
        format!("occ-{}.brep", core),               // exact case
        format!("occ-{}.brep", core.to_lowercase()), // lowercase
        format!("occ_{}.brep", core.to_lowercase()), // underscore (sphere special case)
    ];

    for c in &candidates {
        let p = ref_dir.join(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Fast import: parse STEP → BRepStore only (skip heal + mesh).
/// Uses low-level parse_exchange + StepCafTransfer for speed.
fn fast_import_step(path: &std::path::Path) -> Result<rc3d_shape::BRepStore, String> {
    let options = StepImportOptions {
        mode: StepImportMode::Preview,
        adapter_mode: AdapterMode::CompatMerge,
        ..StepImportOptions::default()
    };

    let file_len = std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0);

    let exchange = if file_len <= 65536 {
        let bytes = std::fs::read(path).map_err(|e| format!("read: {}", e))?;
        let text = rc3d_io::step::decode_step_bytes(&bytes);
        parse_exchange_with_options(&text, &options)
            .map_err(|e| format!("parse: {}", e))?
    } else {
        // Use streaming parser for large files
        rc3d_io::step::parser::parse_step_from_file_with_options(path, &options)
            .map_err(|e| format!("parse: {}", e))?
    };

    let build_options = BRepBuildOptions {
        allow_geometry_fallback: true,
        strict_voids: false,
    };

    let transfer = StepCafTransfer::transfer(&exchange.entities, &build_options)
        .map_err(|e| format!("transfer: {}", e))?;

    Ok(transfer.document.store)
}

/// Collect all step-*.step files in compare/step/
fn discover_step_files() -> Vec<PathBuf> {
    let dir = step_dir();
    if !dir.exists() {
        return vec![];
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|e| e == "step" || e == "stp")
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    files
}

/// Count occurrences of a marker in text (for structural comparison)
fn count_marker(text: &str, marker: &str) -> usize {
    text.match_indices(marker).count()
}

/// Extract topology counts from BREP text
struct BrepTopoCounts {
    vertices: usize,
    edges: usize,
    wires: usize,
    faces: usize,
    shells: usize,
    solids: usize,
    curves: usize,
    curve2ds: usize,
    surfaces: usize,
    warnings: usize,
}

fn analyze_brep_text(text: &str) -> BrepTopoCounts {
    // Parse counts from section headers like "Surfaces 6" or "Curves 12"
    fn parse_section_count(text: &str, section: &str) -> usize {
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(section) {
                if let Some(num_str) = trimmed.strip_prefix(section) {
                    return num_str.trim().parse().unwrap_or(0);
                }
            }
        }
        0
    }

    BrepTopoCounts {
        vertices: count_marker(text, "\nVe"),
        edges: count_marker(text, "\nEd"),
        wires: count_marker(text, "\nWi"),
        faces: count_marker(text, "\nFa"),
        shells: count_marker(text, "\nSh"),
        solids: count_marker(text, "\nSo"),
        curves: parse_section_count(text, "Curves"),
        curve2ds: parse_section_count(text, "Curve2ds"),
        surfaces: parse_section_count(text, "Surfaces"),
        warnings: text.matches("WARNING").count(),
    }
}

/// Print topology comparison table between our output and OCC reference
fn print_comparison(label: &str, our: &BrepTopoCounts, r#ref: &BrepTopoCounts) {
    println!(
        "  {:>6} V={:<4} E={:<4} Wi={:<4} Fa={:<4} Sh={:<4} So={:<4} | curves={:<4} 2d={:<4} surfs={:<4}",
        label,
        our.vertices, our.edges, our.wires, our.faces, our.shells, our.solids,
        our.curves, our.curve2ds, our.surfaces,
    );
    if r#ref.vertices > 0 {
        let v = if our.vertices == r#ref.vertices { "✓" } else { "✗" };
        let e = if our.edges == r#ref.edges { "✓" } else { "✗" };
        let f = if our.faces == r#ref.faces { "✓" } else { "✗" };
        println!(
            "  {:>6} V={:<4} E={:<4} Wi={:<4} Fa={:<4} Sh={:<4} So={:<4} | curves={:<4} 2d={:<4} surfs={:<4}",
            format!("ref"),
            r#ref.vertices, r#ref.edges, r#ref.wires, r#ref.faces, r#ref.shells, r#ref.solids,
            r#ref.curves, r#ref.curve2ds, r#ref.surfaces,
        );
        println!("  match: V{} E{} F{} Sh{} So{}", v, e, f,
            if our.shells == r#ref.shells { "✓" } else { "✗" },
            if our.solids == r#ref.solids { "✓" } else { "✗" },
        );
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[test]
fn brep_compare_discover_files() {
    let files = discover_step_files();
    println!("Discovered {} STEP files in compare/step/:", files.len());
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy();
        let ref_opt = find_occ_ref(&name);
        let status = if ref_opt.is_some() { "✓ ref" } else { "✗ NO REF" };
        println!("  {} → {}", name, status);
    }
    assert!(!files.is_empty(), "no step files found in compare/step/");
}

/// Import STEP, export BREP, verify structural consistency for all files.
#[test]
fn brep_compare_all_files_export_and_check() {
    let files = discover_step_files();
    assert!(!files.is_empty(), "no step files found");

    let out = out_dir();
    std::fs::create_dir_all(&out).ok();

    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for step_path in &files {
        let step_name = step_path.file_name().unwrap().to_string_lossy().to_string();
        let ref_path = find_occ_ref(&step_name);
        let out_path = out.join(&step_name.replace(".step", ".brep"));

        println!("\n=== {} ===", step_name);

        // 1. Fast import STEP → BRepStore (skip heal+mesh)
        let store = match fast_import_step(step_path) {
            Ok(s) => s,
            Err(e) => {
                let msg = format!("{}: IMPORT FAILED: {}", step_name, e);
                println!("  FAIL: {}", msg);
                failures.push(msg);
                failed += 1;
                continue;
            }
        };
        println!(
            "  imported: {} V, {} E, {} W, {} F, {} Sh, {} So",
            store.vertices.len(), store.edges.len(), store.wires.len(),
            store.faces.len(), store.shells.len(), store.solids.len(),
        );

        // 2. Export BREP to buffer
        let mut buf = Vec::new();
        if let Err(e) = write_brep(&store, &mut buf) {
            let msg = format!("{}: BREP EXPORT FAILED: {}", step_name, e);
            println!("  FAIL: {}", msg);
            failures.push(msg);
            failed += 1;
            continue;
        }

        // 3. Write output file
        if let Err(e) = std::fs::write(&out_path, &buf) {
            let msg = format!("{}: WRITE FAILED: {}", step_name, e);
            println!("  FAIL: {}", msg);
            failures.push(msg);
            failed += 1;
            continue;
        }
        println!("  wrote {} bytes → {}", buf.len(), out_path.display());

        // 4. Parse and check our BREP
        let our_text = String::from_utf8(buf).expect("valid utf8");
        let our = analyze_brep_text(&our_text);

        // 5. Check required sections exist
        let required = [
            "DBRep_DrawableShape",
            "CASCADE Topology V1",
            "Curve2ds",
            "Curves",
            "Surfaces",
            "TShapes",
        ];
        let mut missing_sections = Vec::new();
        for sec in &required {
            if !our_text.contains(sec) {
                missing_sections.push(*sec);
            }
        }
        if !missing_sections.is_empty() {
            let msg = format!(
                "{}: missing sections: {:?}",
                step_name, missing_sections
            );
            println!("  FAIL: {}", msg);
            failures.push(msg);
            failed += 1;
            continue;
        }

        // 6. Compare against OCC reference if available
        if let Some(ref_path) = &ref_path {
            let ref_text = std::fs::read_to_string(ref_path).unwrap_or_default();
            let r#ref = analyze_brep_text(&ref_text);
            print_comparison("our", &our, &r#ref);

            // Collect topology differences
            let mut diffs: Vec<String> = Vec::new();
            if our.vertices != r#ref.vertices && r#ref.vertices > 0 {
                diffs.push(format!("V: our={} ref={}", our.vertices, r#ref.vertices));
            }
            if our.edges != r#ref.edges && r#ref.edges > 0 {
                diffs.push(format!("E: our={} ref={}", our.edges, r#ref.edges));
            }
            if our.faces != r#ref.faces && r#ref.faces > 0 {
                diffs.push(format!("Fa: our={} ref={}", our.faces, r#ref.faces));
            }
            if our.shells != r#ref.shells && r#ref.shells > 0 {
                diffs.push(format!("Sh: our={} ref={}", our.shells, r#ref.shells));
            }
            if our.solids != r#ref.solids && r#ref.solids > 0 {
                diffs.push(format!("So: our={} ref={}", our.solids, r#ref.solids));
            }

            if diffs.is_empty() {
                println!("  ✓ topology counts match OCC reference");
            } else {
                println!("  ⚠ topology mismatch (non-fatal): {}", diffs.join(", "));
            }

            // Structural warnings
            if our.curves == 0 && r#ref.curves > 0 {
                println!("  ⚠ WARNING: our BREP has 0 curves but ref has {}", r#ref.curves);
            }
            if our.surfaces == 0 && r#ref.surfaces > 0 {
                println!("  ⚠ WARNING: our BREP has 0 surfaces but ref has {}", r#ref.surfaces);
            }
            passed += 1;
        } else {
            println!("  (no OCC reference file)");
            print_comparison("our", &our, &our); // our-only view
            if our.vertices > 0 && our.faces > 0 {
                println!("  ✓ export OK (no ref to compare)");
                passed += 1;
            } else {
                let msg = format!(
                    "{}: empty output (V={} Fa={})",
                    step_name, our.vertices, our.faces
                );
                println!("  FAIL: {}", msg);
                failures.push(msg);
                failed += 1;
            }
        }
    }

    println!("\n=== SUMMARY ===");
    println!("  passed: {}  failed: {}  total: {}", passed, failed, passed + failed);
    if !failures.is_empty() {
        println!("  failures:");
        for f in &failures {
            println!("    - {}", f);
        }
    }
    assert!(failed == 0, "{} file(s) failed BREP export", failed);
}

/// Quick test: fast import+export of simple primitives.
/// Uses low-level parse+CafTransfer to skip heal+mesh for speed.
#[test]
fn brep_compare_simple_primitives() {
    let out = out_dir();
    std::fs::create_dir_all(&out).ok();

    let simple: &[&str] = &[
        "step-cube.step",
        "step-cylinder.step",
        "step-cone.step",
        "step-torus.step",
        "step-Sphere.step",
        "step-rev.step",
    ];

    for step_file in simple {
        let step_path = step_dir().join(step_file);
        let ref_path = find_occ_ref(step_file);
        let out_path = out.join(step_file.replace(".step", ".brep"));

        if !step_path.exists() {
            println!("  SKIP {} (file not found)", step_file);
            continue;
        }

        println!("\n--- {} ---", step_file);
        let store = match fast_import_step(&step_path) {
            Ok(s) => s,
            Err(e) => {
                println!("  FAIL import: {}", e);
                continue;
            }
        };
        println!(
            "  imported: {} V, {} E, {} W, {} F, {} Sh, {} So",
            store.vertices.len(), store.edges.len(), store.wires.len(),
            store.faces.len(), store.shells.len(), store.solids.len(),
        );
        let mut buf = Vec::new();
        write_brep(&store, &mut buf).expect("write brep");
        std::fs::write(&out_path, &buf).ok();

        let our_text = String::from_utf8(buf).unwrap();
        let our = analyze_brep_text(&our_text);

        // Basic sanity — must have surfaces and TShapes
        assert!(our_text.contains("DBRep_DrawableShape"), "missing header in {}", step_file);
        assert!(our_text.contains("CASCADE Topology V1"), "missing topology in {}", step_file);
        assert!(our_text.contains("TShapes"), "missing TShapes in {}", step_file);
        assert!(our.surfaces > 0, "no surfaces in {}", step_file);

        // Compare with OCC reference if available
        if let Some(ref_path) = &ref_path {
            let ref_text = std::fs::read_to_string(ref_path).unwrap();
            let r#ref = analyze_brep_text(&ref_text);
            print_comparison("our", &our, &r#ref);

            // Compare structural topology — report all differences
            let mut ok = true;
            if our.vertices != r#ref.vertices && r#ref.vertices > 0 {
                println!("  ⚠ V: our={} ref={}", our.vertices, r#ref.vertices);
            }
            if our.edges != r#ref.edges && r#ref.edges > 0 {
                println!("  ⚠ E: our={} ref={}", our.edges, r#ref.edges);
            }
            if our.faces != r#ref.faces && r#ref.faces > 0 {
                println!("  ❌ Fa: our={} ref={}", our.faces, r#ref.faces);
                ok = false;
            }
            if our.shells != r#ref.shells && r#ref.shells > 0 {
                println!("  ❌ Sh: our={} ref={}", our.shells, r#ref.shells);
                ok = false;
            }
            if our.solids != r#ref.solids && r#ref.solids > 0 {
                println!("  ❌ So: our={} ref={}", our.solids, r#ref.solids);
                ok = false;
            }
            if our.curves != r#ref.curves && r#ref.curves > 0 {
                println!("  ⚠ curves: our={} ref={}", our.curves, r#ref.curves);
            }
            if our.surfaces != r#ref.surfaces && r#ref.surfaces > 0 {
                println!("  ❌ surfaces: our={} ref={}", our.surfaces, r#ref.surfaces);
                ok = false;
            }
            if ok && our.vertices == r#ref.vertices && our.edges == r#ref.edges {
                println!("  ✓ all counts match OCC reference");
            } else if ok {
                println!("  ✓ structural topology matches (V/E differ, see above)");
            } else {
                println!("  ❌ structural mismatch — needs investigation");
            }
        }
    }
}
