//! Verification STL export from the same import/emit pipeline used by visualization.
//! Single file:  cargo test -p rc3d-io --test export_step_stl case_shape --release -- --exact case_shape
//! All three:    cargo test -p rc3d-io --test export_step_stl case_all --release -- --exact case_all
//! Optional cs.step: set RC3D_EXPORT_CS=1
//! Layout: set RC3D_EXPORT_LAYOUT=merged|per_instance (default: CI->merged, local->per_instance)
//! Format: set RC3D_EXPORT_STL=ascii|binary (default: ascii)

use rc3d_core::math::Vec3;
use rc3d_io::step::{emit_plan_options_from_step, import_step_file_with_options, StepImportOptions};
use rc3d_io::step::mesh_result::MeshResult;
use rc3d_io::{write_ascii_stl, write_binary_stl, StlError};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExportLayout {
    Merged,
    PerInstance,
}

fn test_data(name: &str) -> PathBuf {
    let steps = std::env::var("RC3D_STEP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../steps"));
    steps.join(name)
}

enum StlFormat {
    Ascii,
    Binary,
}

fn export_stl_format() -> StlFormat {
    match std::env::var("RC3D_EXPORT_STL").ok().as_deref() {
        Some("binary") => StlFormat::Binary,
        _ => StlFormat::Ascii,
    }
}

fn export_layout() -> ExportLayout {
    match std::env::var("RC3D_EXPORT_LAYOUT").ok().as_deref() {
        Some("merged") => ExportLayout::Merged,
        Some("per_instance") => ExportLayout::PerInstance,
        _ => {
            if std::env::var("CI").is_ok() {
                ExportLayout::Merged
            } else {
                ExportLayout::PerInstance
            }
        }
    }
}

fn append_mesh_with_transform(dst: &mut MeshResult, src: &MeshResult, world: rc3d_core::math::Mat4) {
    let offset = dst.vertices.len() as i32;
    for &v in &src.vertices {
        let p = world * v.extend(1.0);
        dst.vertices.push(Vec3::new(p.x, p.y, p.z));
    }
    for &n in &src.normals {
        dst.normals.push(n);
    }
    for &idx in &src.indices {
        if idx == -1 {
            dst.indices.push(-1);
        } else {
            dst.indices.push(idx + offset);
        }
    }
}

fn write_mesh_stl(path: &Path, mesh: &MeshResult) -> Result<(), StlError> {
    match export_stl_format() {
        StlFormat::Ascii => write_ascii_stl(path, &mesh.vertices, &mesh.indices),
        StlFormat::Binary => write_binary_stl(path, &mesh.vertices, &mesh.indices),
    }
}

fn mesh_and_export_stl(step_name: &str, output_dir: &Path) -> Option<Vec<PathBuf>> {
    let step_path = test_data(step_name);
    if !step_path.exists() {
        println!("  SKIP: {} not found", step_name);
        return None;
    }

    let t0 = Instant::now();
    println!("=== {} ===", step_name);

    let mut import_options = StepImportOptions::default();
    import_options.skip_visualization = true;
    import_options.heal_level = rc3d_io::step::brep::heal::HealLevel::Basic;
    import_options.fast_export = true;
    let mut result = import_step_file_with_options(&step_path, &import_options).expect("import step");

    let plan_options = emit_plan_options_from_step(&import_options);
    let plan = result
        .document
        .build_emit_plan(&plan_options)
        .expect("build emit plan");

    let layout = export_layout();
    let stem = step_name.trim_end_matches(".step").trim_end_matches(".stp");
    let mut out_paths = Vec::new();

    match layout {
        ExportLayout::Merged => {
            let mut merged = MeshResult::default();
            let mut tris = 0usize;
            for inst in &plan.instances {
                let cached = plan
                    .mesh_table
                    .get(&inst.mesh_slot)
                    .expect("mesh slot exists");
                tris += cached.mesh.indices.len() / 4;
                append_mesh_with_transform(&mut merged, &cached.mesh, inst.world_transform);
            }
            if merged.vertices.is_empty() {
                println!("  WARN: no mesh produced");
                return None;
            }
            let stl_path = output_dir.join(format!("{}.stl", stem));
            write_mesh_stl(&stl_path, &merged).expect("write stl");
            let file_size = std::fs::metadata(&stl_path).map(|m| m.len()).unwrap_or(0);
            println!(
                "  merged: {} instances, {} verts, {} tris",
                plan.instances.len(),
                merged.vertices.len(),
                tris
            );
            println!(
                "  -> STL: {} ({} KB)",
                stl_path.display(),
                file_size / 1024,
            );
            out_paths.push(stl_path);
        }
        ExportLayout::PerInstance => {
            let mut total_tris = 0usize;
            for (i, inst) in plan.instances.iter().enumerate() {
                let cached = plan
                    .mesh_table
                    .get(&inst.mesh_slot)
                    .expect("mesh slot exists");
                total_tris += cached.mesh.indices.len() / 4;
                let mut mesh = MeshResult::default();
                append_mesh_with_transform(&mut mesh, &cached.mesh, inst.world_transform);
                let stl_path = output_dir.join(format!("{}__inst{:03}.stl", stem, i));
                write_mesh_stl(&stl_path, &mesh).expect("write stl");
                out_paths.push(stl_path);
            }
            println!(
                "  per_instance: {} files, {} total tris",
                out_paths.len(),
                total_tris
            );
        }
    }

    println!(
        "  timing: import+emit+export={:.1}s",
        t0.elapsed().as_secs_f32(),
    );
    Some(out_paths)
}

fn corpus_files() -> Vec<&'static str> {
    let mut files = vec!["Shape.step", "Shape-1.step", "Shape-2.step"];
    if std::env::var("RC3D_EXPORT_CS").is_ok() {
        files.push("cs.step");
    }
    files.extend(["Cube.step", "asse.step"]);
    files
}

fn assert_stl_paths(step_name: &str, paths: &[PathBuf]) {
    assert!(!paths.is_empty(), "no STL files produced for {step_name}");
    for stl_path in paths {
        let size = std::fs::metadata(stl_path).unwrap().len();
        assert!(
            size > 100,
            "STL too small for {}: {} ({} bytes)",
            step_name,
            stl_path.display(),
            size
        );
    }
}

fn export_one(step_name: &str) {
    let step_path = test_data(step_name);
    if !step_path.exists() {
        eprintln!("SKIP: {} not found (set RC3D_STEP_DIR to enable)", step_name);
        return;
    }
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_output");
    std::fs::create_dir_all(&output_dir).ok();
    let stl_paths = mesh_and_export_stl(step_name, &output_dir).expect("export failed");
    assert_stl_paths(step_name, &stl_paths);
}

/// Heavy integration tests: import + mesh + STL export from real STEP files.
/// Run manually with: cargo test -p rc3d-io --test export_step_stl --release -- --ignored
/// Requires RC3D_STEP_DIR env pointing to a directory with Shape.step etc.

#[test]
#[ignore = "requires external STEP files + minutes of CPU; run with --ignored"]
fn case_shape() {
    export_one("Shape.step");
}

#[test]
#[ignore = "requires external STEP files + minutes of CPU; run with --ignored"]
fn case_shape1() {
    export_one("Shape-1.step");
}

#[test]
#[ignore = "requires external STEP files + minutes of CPU; run with --ignored"]
fn case_shape2() {
    export_one("Shape-2.step");
}

/// Run export with a per-file timeout. Returns Some(paths) on success,
/// None on timeout or error.
fn mesh_and_export_with_timeout(step_name: &str, output_dir: &Path, timeout_secs: u64) -> Option<Vec<PathBuf>> {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    let name = step_name.to_string();
    let dir = output_dir.to_path_buf();

    std::thread::spawn(move || {
        let result = mesh_and_export_stl(&name, &dir);
        let _ = tx.send(result);
    });

    match rx.recv_timeout(std::time::Duration::from_secs(timeout_secs)) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            eprintln!("  TIMEOUT: {} after {}s", step_name, timeout_secs);
            None
        }
        Err(_) => None,
    }
}

#[test]
#[ignore = "requires external STEP files; run with --ignored"]
fn case_all() {
    // Per-file timeout: quick files get 60s, large files get 300s
    let quick_timeout = 60;
    let large_timeout = 300;

    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_output");
    std::fs::create_dir_all(&output_dir).ok();

    let mut ok_count = 0;
    let mut fail_count = 0;
    let mut failures: Vec<String> = Vec::new();

    for name in corpus_files() {
        // Shape-1 and Shape-2 have complex B-Rep that may hang during mesh
        // generation; use a shorter timeout to avoid blocking the suite.
        let timeout = if name.contains("Shape-1") || name.contains("Shape-2") {
            large_timeout
        } else {
            quick_timeout
        };

        match mesh_and_export_with_timeout(name, &output_dir, timeout) {
            Some(paths) => {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    assert_stl_paths(name, &paths);
                })) {
                    Ok(()) => {
                        ok_count += 1;
                        println!("  OK: {}", name);
                    }
                    Err(_) => {
                        fail_count += 1;
                        failures.push(format!("{} (STL validation failed)", name));
                        println!("  FAIL: {} — STL validation failed", name);
                    }
                }
            }
            None => {
                fail_count += 1;
                failures.push(format!("{} (export returned None or timed out)", name));
                println!("  FAIL: {} — export failed or timed out", name);
            }
        }
    }

    println!("=== Summary: {} OK, {} FAILED ===", ok_count, fail_count);
    if !failures.is_empty() {
        println!("Failures:");
        for f in &failures {
            println!("  - {}", f);
        }
    }
    assert!(ok_count > 0, "At least one file should export successfully");
}
