//! Quick test: export STL from Shape corpus STEP files.
//! Single file:  cargo test -p rc3d-io --test export_step_stl case_shape --release -- --exact case_shape
//! All three:    cargo test -p rc3d-io --test export_step_stl case_all --release -- --exact case_all
//! Optional cs.step: set RC3D_EXPORT_CS=1

use rc3d_io::step::brep::build_brep;
use rc3d_io::step::brep::heal::{auto_heal_shell, HealLevel};
use rc3d_io::step::brep::mesh::{mesh_brep_shell_with_report, BRepMeshConfig};
use rc3d_io::step::mesh_result::MeshResult;
use rc3d_io::step::parser;
use rc3d_io::write_binary_stl;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn test_data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

fn export_mesh_config() -> BRepMeshConfig {
    BRepMeshConfig::preview()
}

fn mesh_and_export_stl(step_name: &str, output_dir: &Path) -> Option<PathBuf> {
    let step_path = test_data(step_name);
    if !step_path.exists() {
        println!("  SKIP: {} not found", step_name);
        return None;
    }

    let t0 = Instant::now();
    println!("=== {} ===", step_name);

    let text = std::fs::read_to_string(&step_path).expect("read step");
    let exchange = parser::parse_exchange(&text).expect("parse step");
    let brep = build_brep(&exchange.entities).expect("build brep");
    let mut reg = brep.registry;
    let t_parse = t0.elapsed();

    let mut skip_face_keys = Vec::new();
    let root_solids: Vec<_> = brep.root_solids.clone();
    for &sk in &root_solids {
        let outer_shell = match reg.solids.get(sk) {
            Some(solid) => solid.outer_shell,
            None => continue,
        };
        let heal = auto_heal_shell(outer_shell, &mut reg, HealLevel::Basic, 2);
        let skip_count = heal.skip_face_keys.len();
        skip_face_keys.extend(heal.skip_face_keys);
        let face_count = reg
            .shells
            .get(outer_shell)
            .map(|s| s.faces.len())
            .unwrap_or(0);
        println!(
            "  solid {:?}: {} faces, skip={}",
            sk,
            face_count,
            skip_count
        );
    }
    let t_heal = t0.elapsed();

    let mesh_config = export_mesh_config();
    let mut combined_mesh = MeshResult::default();
    let mut total_tris = 0usize;

    for &sk in &brep.root_solids {
        let Some(solid) = reg.solids.get(sk) else {
            continue;
        };
        let out = mesh_brep_shell_with_report(
            solid.outer_shell,
            &reg,
            &mesh_config,
            &skip_face_keys,
        );
        total_tris += out.report.total_tris;
        println!(
            "  mesh: {} verts, {} tris, meshed_faces={}, grid_fallback={}",
            out.mesh.vertices.len(),
            out.report.total_tris,
            out.report.meshed_faces,
            out.report.grid_fallback_count,
        );
        let offset = combined_mesh.vertices.len() as i32;
        combined_mesh.vertices.extend_from_slice(&out.mesh.vertices);
        if !out.mesh.normals.is_empty() {
            combined_mesh.normals.extend_from_slice(&out.mesh.normals);
        }
        for &idx in &out.mesh.indices {
            if idx == -1 {
                combined_mesh.indices.push(-1);
            } else {
                combined_mesh.indices.push(idx + offset);
            }
        }
    }
    let t_mesh = t0.elapsed();

    if combined_mesh.vertices.is_empty() {
        println!("  WARN: no mesh produced");
        return None;
    }

    let stem = step_name.trim_end_matches(".step").trim_end_matches(".stp");
    let stl_path = output_dir.join(format!("{}.stl", stem));
    write_binary_stl(&stl_path, &combined_mesh.vertices, &combined_mesh.indices)
        .expect("write binary stl");
    let file_size = std::fs::metadata(&stl_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  -> STL: {} ({} KB, {} verts, {} tris)",
        stl_path.display(),
        file_size / 1024,
        combined_mesh.vertices.len(),
        total_tris,
    );
    println!(
        "  timing: parse={:.1}s heal={:.1}s mesh={:.1}s total={:.1}s",
        t_parse.as_secs_f32(),
        (t_heal - t_parse).as_secs_f32(),
        (t_mesh - t_heal).as_secs_f32(),
        t0.elapsed().as_secs_f32(),
    );
    Some(stl_path)
}

fn corpus_files() -> Vec<&'static str> {
    let mut files = vec!["Shape.step", "Shape-1.step", "Shape-2.step"];
    if std::env::var("RC3D_EXPORT_CS").is_ok() {
        files.push("cs.step");
    }
    files
}

fn export_one(step_name: &str) {
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_output");
    std::fs::create_dir_all(&output_dir).ok();
    let stl_path = mesh_and_export_stl(step_name, &output_dir).expect("export failed");
    let size = std::fs::metadata(&stl_path).unwrap().len();
    assert!(
        size > 100,
        "STL too small: {} ({} bytes)",
        stl_path.display(),
        size
    );
}

#[test]
fn case_shape() {
    export_one("Shape.step");
}

#[test]
fn case_shape1() {
    export_one("Shape-1.step");
}

#[test]
fn case_shape2() {
    export_one("Shape-2.step");
}

#[test]
fn case_all() {
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_output");
    std::fs::create_dir_all(&output_dir).ok();

    for name in corpus_files() {
        let path = mesh_and_export_stl(name, &output_dir).expect("export failed");
        let size = std::fs::metadata(&path).unwrap().len();
        assert!(
            size > 100,
            "STL too small for {}: {} ({} bytes)",
            name,
            path.display(),
            size
        );
    }
}
