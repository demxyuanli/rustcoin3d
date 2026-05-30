//! Quick test: export STL from Shape, Shape-1, Shape-2, cs STEP files.
//! Run: cargo test -p rc3d-io --test export_step_stl --release -- --nocapture

use rc3d_io::step::brep::build_brep;
use rc3d_io::step::brep::heal::{auto_heal_shell, HealLevel};
use rc3d_io::step::brep::mesh::{mesh_brep_shell_with_report, BRepMeshConfig};
use rc3d_io::step::mesh_result::MeshResult;
use rc3d_io::step::parser;
use rc3d_io::write_ascii_stl;
use std::path::Path;

fn test_data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

fn mesh_and_export_stl(step_name: &str, output_dir: &Path) {
    let step_path = test_data(step_name);
    if !step_path.exists() {
        println!("  SKIP: {} not found", step_name);
        return;
    }

    println!("=== {} ===", step_name);

    let text = std::fs::read_to_string(&step_path).expect("read step");
    let exchange = parser::parse_exchange(&text).expect("parse step");
    let brep = build_brep(&exchange.entities).expect("build brep");
    let mut reg = brep.registry;

    // Heal
    let mut skip_face_keys = Vec::new();
    let root_solids: Vec<_> = brep.root_solids.clone();
    for &sk in &root_solids {
        let outer_shell = match reg.solids.get(sk) {
            Some(solid) => solid.outer_shell,
            None => continue,
        };
        let heal = auto_heal_shell(outer_shell, &mut reg, HealLevel::Standard, 5);
        let skip_count = heal.skip_face_keys.len();
        skip_face_keys.extend(heal.skip_face_keys);
        let face_count = reg.shells.get(outer_shell).map(|s| s.faces.len()).unwrap_or(0);
        println!("  solid {:?}: {} faces, skip={}", sk, face_count, skip_count);
    }

    // Mesh
    let mesh_config = BRepMeshConfig::default();
    let mut combined_mesh = MeshResult::default();
    let mut total_tris = 0usize;

    for &sk in &brep.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let out = mesh_brep_shell_with_report(
                solid.outer_shell, &reg, &mesh_config, &skip_face_keys,
            );
            total_tris += out.report.total_tris;
            println!("  mesh: {} verts, {} tris, meshed_faces={}, grid_fallback={}",
                out.mesh.vertices.len(),
                out.report.total_tris,
                out.report.meshed_faces,
                out.report.grid_fallback_count,
            );
            // Merge into combined
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
    }

    if combined_mesh.vertices.is_empty() {
        println!("  WARN: no mesh produced");
        return;
    }

    // Write STL
    let stem = step_name.trim_end_matches(".step").trim_end_matches(".stp");
    let stl_path = output_dir.join(format!("{}.stl", stem));
    match write_ascii_stl(&stl_path, &combined_mesh.vertices, &combined_mesh.indices) {
        Ok(()) => {
            let file_size = std::fs::metadata(&stl_path).map(|m| m.len()).unwrap_or(0);
            println!("  -> STL: {} ({} KB, {} verts, {} tris)",
                stl_path.display(),
                file_size / 1024,
                combined_mesh.vertices.len(),
                total_tris,
            );
        }
        Err(e) => println!("  FAIL: write STL error: {}", e),
    }
}

#[test]
fn export_shape_stl() {
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_output");
    std::fs::create_dir_all(&output_dir).ok();

    let files = ["Shape.step", "Shape-1.step", "Shape-2.step", "cs.step"];
    for &name in &files {
        mesh_and_export_stl(name, &output_dir);
        println!();
    }

    // Verify all STLs exist
    for name in &files {
        let stem = name.trim_end_matches(".step");
        let stl_path = output_dir.join(format!("{}.stl", stem));
        assert!(stl_path.exists(), "STL not created: {}", stl_path.display());
        let size = std::fs::metadata(&stl_path).unwrap().len();
        assert!(size > 100, "STL too small: {} ({} bytes)", stl_path.display(), size);
    }
}
