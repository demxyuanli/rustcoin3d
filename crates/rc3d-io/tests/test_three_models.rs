//! Test three STEP models: Cube.step, cs.step, Shape.step
//! Run: cargo test -p rc3d-io --test test_three_models --release -- --nocapture

use rc3d_core::math::Vec3;
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

fn output_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_output")
}

fn test_step_model(name: &str) {
    println!("\n========== Testing {} ==========", name);
    let path = test_data(name);

    // 1. Parse
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            println!("  SKIP: cannot read {:?}: {}", path, e);
            return;
        }
    };
    let exchange = match parser::parse_exchange(&text) {
        Ok(e) => e,
        Err(e) => {
            println!("  FAIL: parse error: {:?}", e);
            return;
        }
    };
    println!("  Parsed: {} entities", exchange.entities.len());

    // 2. Build B-Rep
    let brep = match build_brep(&exchange.entities) {
        Ok(b) => b,
        Err(e) => {
            println!("  FAIL: build_brep error: {:?}", e);
            return;
        }
    };
    let mut reg = brep.registry;
    println!(
        "  B-Rep: {} solids, {} faces, {} edges, {} vertices",
        brep.root_solids.len(),
        reg.faces.len(),
        reg.edges.len(),
        reg.vertices.len()
    );

    // 3. Heal
    for &sk in &brep.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            auto_heal_shell(solid.outer_shell, &mut reg, HealLevel::Standard, 5);
        }
    }

    // 4. Mesh + export STL (merge all solids into one file)
    let mesh_config = BRepMeshConfig::default();
    let mut combined = MeshResult::default();
    for &sk in &brep.root_solids {
        let solid = match reg.solids.get(sk) {
            Some(s) => s,
            None => continue,
        };
        let shell_key = solid.outer_shell;
        let _shell = match reg.shells.get(shell_key) {
            Some(s) => s,
            None => continue,
        };
        let skip: Vec<_> = Vec::new();
        let output = mesh_brep_shell_with_report(shell_key, &reg, &mesh_config, &skip);
        let mesh = &output.mesh;
        let report = &output.report;
        println!(
            "  Mesh: {} vertices, {} triangles, {} faces meshed, grid_fallback {:.1}%",
            mesh.vertices.len(),
            mesh.indices.len() / 4,
            report.meshed_faces,
            if report.meshed_faces > 0 {
                report.grid_fallback_count as f32 / report.meshed_faces as f32 * 100.0
            } else {
                0.0
            }
        );

        // Check for degenerate triangles
        let mut degenerate = 0usize;
        let mut nan_verts = 0usize;
        for v in &mesh.vertices {
            if v.x.is_nan() || v.y.is_nan() || v.z.is_nan() {
                nan_verts += 1;
            }
        }
        let chunks = mesh.indices.len() / 4;
        for i in 0..chunks {
            let i0 = mesh.indices[i * 4] as usize;
            let i1 = mesh.indices[i * 4 + 1] as usize;
            let i2 = mesh.indices[i * 4 + 2] as usize;
            if i0 < mesh.vertices.len() && i1 < mesh.vertices.len() && i2 < mesh.vertices.len() {
                let a = mesh.vertices[i0];
                let b = mesh.vertices[i1];
                let c = mesh.vertices[i2];
                let area = (b - a).cross(c - a).length() * 0.5;
                if area < 1e-12 {
                    degenerate += 1;
                }
            }
        }
        if nan_verts > 0 {
            println!("  WARNING: {} NaN vertices!", nan_verts);
        }
        if degenerate > 0 {
            println!("  WARNING: {} degenerate triangles", degenerate);
        }
        println!(
            "  Quality: {} nan_verts, {} degenerate_tris / {} total",
            nan_verts,
            degenerate,
            chunks
        );

        combined.append_from(mesh);
    }

    if !combined.vertices.is_empty() && !combined.indices.is_empty() {
        let stem = name.trim_end_matches(".step");
        let stl_path = output_dir().join(format!("{}.stl", stem));
        std::fs::create_dir_all(stl_path.parent().unwrap()).unwrap();
        write_ascii_stl(&stl_path, &combined.vertices, &combined.indices).expect("write stl");
        println!(
            "  ASCII STL written: {} ({} tris, merged {} solids)",
            stl_path.display(),
            combined.indices.len() / 4,
            brep.root_solids.len()
        );
    }
    println!("  OK");
}

#[test]
fn test_cube() {
    test_step_model("Cube.step");
}

#[test]
fn test_cs() {
    test_step_model("cs.step");
}

#[test]
fn test_shape() {
    test_step_model("Shape.step");
}

#[test]
fn test_shape1() {
    test_step_model("Shape-1.step");
}

#[test]
fn test_shape2() {
    test_step_model("Shape-2.step");
}
