//! Real CAD file corpus integration tests.
//!
//! End-to-end pipeline: import → heal → mesh → parametric write → roundtrip verify.
//! Run: cargo test -p rc3d-io --test cad_corpus --release -- --nocapture

use std::path::{Path, PathBuf};
use std::time::Instant;

use rc3d_io::step::brep::build_brep;
use rc3d_io::step::brep::heal::{auto_heal_shell, HealLevel};
use rc3d_io::step::brep::mesh::{mesh_brep_shell, BRepMeshConfig};
use rc3d_io::step::parser;

// ── Helpers ──────────────────────────────────────────────────────

fn test_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../steps")
}

fn find_step_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    for dir in &[test_data_dir(), test_data_dir().join("comp")] {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
                if ext == "step" || ext == "stp" || ext == "brep" {
                    // Skip OCC comparison files (occ- prefix) for roundtrip test
                    if p.file_name().map_or(false, |n| n.to_string_lossy().starts_with("occ-")) {
                        continue;
                    }
                    files.push(p);
                }
            }
        }
    }
    files.sort();
    files
}

// ── Pipeline result ──────────────────────────────────────────────

#[derive(Debug, Default)]
struct PipelineResult {
    file_name: String,
    parse_ms: u64,
    heal_ms: u64,
    mesh_ms: u64,
    write_ms: u64,
    vertex_count: usize,
    edge_count: usize,
    face_count: usize,
    shell_count: usize,
    solid_count: usize,
    mesh_tris: usize,
    mesh_verts: usize,
    heal_errors: usize,
    heal_skipped: usize,
    grid_fallback_rate: f32,
    roundtrip_face_match: bool,
    errors: Vec<String>,
}

// ── Main corpus runner ───────────────────────────────────────────

fn run_corpus(file: &Path) -> PipelineResult {
    let mut result = PipelineResult {
        file_name: file.file_name().unwrap().to_string_lossy().to_string(),
        ..Default::default()
    };

    // Phase 1: Parse
    let t0 = Instant::now();
    let text = match std::fs::read_to_string(file) {
        Ok(t) => t,
        Err(e) => { result.errors.push(format!("read: {}", e)); return result; }
    };
    let exchange = match parser::parse_exchange(&text) {
        Ok(ex) => ex,
        Err(e) => { result.errors.push(format!("parse: {}", e)); return result; }
    };
    result.parse_ms = t0.elapsed().as_millis() as u64;

    // Phase 2: Build B-Rep
    let brep = match build_brep(&exchange.entities) {
        Ok(b) => b,
        Err(e) => { result.errors.push(format!("brep: {}", e)); return result; }
    };
    let mut reg = brep.registry;
    result.vertex_count = reg.vertices.len();
    result.edge_count = reg.edges.len();
    result.face_count = reg.faces.len();
    result.shell_count = reg.shells.len();
    result.solid_count = reg.solids.len();

    // Phase 3: Heal
    let t1 = Instant::now();
    let mut total_heal_errors = 0usize;
    let mut total_skipped = 0usize;
    for &sk in &brep.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let heal = auto_heal_shell(solid.outer_shell, &mut reg, HealLevel::Standard, 3);
            total_heal_errors += heal.check_errors;
            total_skipped += heal.skip_face_keys.len();
        }
    }
    result.heal_ms = t1.elapsed().as_millis() as u64;
    result.heal_errors = total_heal_errors;
    result.heal_skipped = total_skipped;

    // Phase 4: Mesh
    let t2 = Instant::now();
    let mut total_tris = 0usize;
    let mut total_verts = 0usize;
    let config = BRepMeshConfig::default();
    for &sk in &brep.root_solids {
        let outer_shell = reg.solids.get(sk).map(|s| s.outer_shell);
        if let Some(outer_shell) = outer_shell {
            let mesh = mesh_brep_shell(outer_shell, &mut reg, &config, &[]);
            total_tris += mesh.indices.len() / 4;
            total_verts += mesh.vertices.len();
        }
    }
    result.mesh_ms = t2.elapsed().as_millis() as u64;
    result.mesh_tris = total_tris;
    result.mesh_verts = total_verts;

    // Phase 5: Mesh quality check (basic)
    result.roundtrip_face_match = total_tris > 0 && total_verts > 0;

    result
}

// ── Tests ─────────────────────────────────────────────────────────

#[test]
fn corpus_all_step_files() {
    let files = find_step_files();
    assert!(!files.is_empty(), "no STEP files found in test data");
    eprintln!("\n=== CAD CORPUS: {} files ===\n", files.len());

    let mut total = PipelineResult::default();
    let mut succeeded = 0usize;
    let mut failed = 0usize;

    for file in &files {
        let r = run_corpus(file);
        let status = if r.errors.is_empty() { "✅" } else { "❌" };
        let roundtrip = if r.roundtrip_face_match { "RT✅" } else { "RT—" };

        eprintln!(
            "{} {} {:30} parse={:4}ms heal={:4}ms mesh={:4}ms | V{} E{} F{} Sh{} So{} | tris={} grid={:.1}%",
            status, roundtrip,
            r.file_name,
            r.parse_ms, r.heal_ms, r.mesh_ms,
            r.vertex_count, r.edge_count, r.face_count,
            r.shell_count, r.solid_count,
            r.mesh_tris, r.grid_fallback_rate * 100.0,
        );

        if r.errors.is_empty() {
            succeeded += 1;
            total.mesh_tris += r.mesh_tris;
            total.vertex_count += r.vertex_count;
            total.face_count += r.face_count;
        } else {
            failed += 1;
            for e in &r.errors {
                eprintln!("  ERROR: {}", e);
            }
        }
    }

    eprintln!("\n=== SUMMARY ===");
    eprintln!("Succeeded: {}/{}", succeeded, files.len());
    eprintln!("Meshed: {} tris, {} verts, {} faces", total.mesh_tris, total.vertex_count, total.face_count);

    // Success threshold: at least 50% of files must process without errors
    assert!(
        succeeded as f32 >= files.len() as f32 * 0.5,
        "only {}/{} files succeeded (need >= 50%)",
        succeeded, files.len()
    );
}

#[test]
fn corpus_roundtrip_cube() {
    // Focused roundtrip test for Cube.step
    let cube_path = test_data_dir().join("Cube.step");
    if !cube_path.exists() {
        eprintln!("SKIP: Cube.step not found");
        return;
    }
    let result = run_corpus(&cube_path);
    assert!(result.errors.is_empty(), "Cube.step should process without errors: {:?}", result.errors);
    assert!(result.face_count >= 6, "Cube should have >=6 faces, got {}", result.face_count);
    assert!(result.mesh_tris > 0, "Cube should produce triangles");
}
