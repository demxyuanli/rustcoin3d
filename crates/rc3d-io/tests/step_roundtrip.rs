//! Test STEP roundtrip: parse → export → re-parse → compare
//! Run: cargo test -p rc3d-io --test step_roundtrip --release -- --nocapture

use rc3d_core::math::PVec3;
use rc3d_io::step::brep::build_brep;
use rc3d_io::step::brep::heal::{auto_heal_shell, HealLevel};
use rc3d_io::step::brep::mesh::{mesh_brep_shell_with_report, BRepMeshConfig};
use rc3d_io::step::brep::hausdorff_meshes;
use rc3d_io::step::parser;
use rc3d_io::step::mesh_result::MeshResult;
use rc3d_io::parse_stl_triangles;
use std::path::Path;

fn test_data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

fn mesh_step_file(step_path: &Path) -> MeshResult {
    let text = std::fs::read_to_string(step_path).expect("read");
    let exchange = parser::parse_exchange(&text).expect("parse");
    let brep = build_brep(&exchange.entities).expect("brep");
    let mut reg = brep.registry;
    let mut skip_face_keys = Vec::new();
    for &sk in &brep.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let heal = auto_heal_shell(solid.outer_shell, &mut reg, HealLevel::Standard, 5);
            skip_face_keys.extend(heal.skip_face_keys);
        }
    }
    let mesh_config = BRepMeshConfig::default();
    let mut mesh = MeshResult::default();
    for &sk in &brep.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let out = mesh_brep_shell_with_report(solid.outer_shell, &reg, &mesh_config, &skip_face_keys);
            mesh = out.mesh;
        }
    }
    mesh
}

fn bbox(verts: &[PVec3]) -> (PVec3, PVec3) {
    let mut mn = PVec3::splat(f64::MAX);
    let mut mx = PVec3::splat(f64::MIN);
    for v in verts { mn = mn.min(*v); mx = mx.max(*v); }
    (mn, mx)
}

#[test]
fn step_roundtrip_shape() {
    let step_path = test_data("Shape.step");

    // 1. Original mesh
    let orig_mesh = mesh_step_file(&step_path);
    let (orig_min, orig_max) = bbox(&orig_mesh.vertices);
    println!("\nOriginal: {} tris, {} verts, diag={:.2}",
        orig_mesh.indices.len()/4, orig_mesh.vertices.len(),
        (orig_max - orig_min).length());

    // 2. Export to STEP
    let text = std::fs::read_to_string(&step_path).expect("read");
    let exchange = parser::parse_exchange(&text).expect("parse original");

    let export_path = test_data("Shape_roundtrip.step");
    let step_text = rc3d_io::step::write::write_step_from_entities(&exchange.entities);
    std::fs::write(&export_path, &step_text).expect("write STEP");
    println!("Exported: {:?} ({} bytes)", export_path,
        std::fs::metadata(&export_path).map(|m| m.len()).unwrap_or(0));

    // 3. Re-parse exported STEP
    let reimported = mesh_step_file(&export_path);
    let (re_min, re_max) = bbox(&reimported.vertices);
    println!("Reimported: {} tris, {} verts, diag={:.2}",
        reimported.indices.len()/4, reimported.vertices.len(),
        (re_max - re_min).length());

    // 4. Compare bounding boxes
    println!("\nBBox original:  {:?} → {:?}", orig_min, orig_max);
    println!("BBox reimport: {:?} → {:?}", re_min, re_max);
    let diag_ratio = (orig_max - orig_min).length() / (re_max - re_min).length();
    println!("Diag ratio: {:.4}", diag_ratio);

    // 5. Hausdorff between original and reimported meshes
    let h = hausdorff_meshes(&orig_mesh, &reimported, 512);
    println!("\nHausdorff (orig ↔ reimport):");
    println!("  sym_p95={:.4}  sym_max={:.4}", h.symmetric_p95, h.symmetric_max);
    println!("  orig→reimport p95={:.4}", h.p95_a_to_b);
    println!("  reimport→orig p95={:.4}", h.p95_b_to_a);

    // 6. Compare vs OCCT reference
    let ref_path = test_data("shape-tri.stl");
    if ref_path.exists() {
        let data = std::fs::read(&ref_path).expect("read ref stl");
        let tris = parse_stl_triangles(&data).expect("parse ref");
        let mut ref_mesh = MeshResult::default();
        for tri in tris {
            let base = ref_mesh.vertices.len() as i32;
            ref_mesh.vertices.push(PVec3::from(tri.vertices[0]));
            ref_mesh.vertices.push(PVec3::from(tri.vertices[1]));
            ref_mesh.vertices.push(PVec3::from(tri.vertices[2]));
            ref_mesh.indices.extend_from_slice(&[base, base + 1, base + 2, -1]);
        }
        let h_ref = hausdorff_meshes(&reimported, &ref_mesh, 512);
        println!("\nHausdorff (reimport ↔ OCCT ref):");
        println!("  sym_p95={:.4}  sym_max={:.4}", h_ref.symmetric_p95, h_ref.symmetric_max);
        println!("  reimport→ref p95={:.4}", h_ref.p95_a_to_b);
    }

    // Basic sanity: diag should be within 5%
    assert!(diag_ratio > 0.95 && diag_ratio < 1.05,
        "Roundtrip BBox changed significantly: ratio={:.4}", diag_ratio);
}
