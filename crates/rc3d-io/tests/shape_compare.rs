//! Compare our engine's STEP→mesh output against an OCCT-generated reference STL.
//! Run: cargo test -p rc3d-io --test shape_compare --release -- --nocapture

use rc3d_core::math::Vec3;
use rc3d_io::step::brep::build_brep;
use rc3d_io::step::brep::mesh::{mesh_brep_shell_with_report, BRepMeshConfig};
use rc3d_io::step::brep::mesh::report::ShellMeshReport;
use rc3d_io::step::brep::hausdorff_meshes;
use rc3d_io::step::import_options::StepImportOptions;
use rc3d_io::step::parser;
use rc3d_io::step::mesh_result::MeshResult;
use rc3d_io::parse_stl_triangles;
use std::path::Path;

fn test_data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

#[test]
#[ignore = "requires OCCT-generated reference STL (shape-tri.stl) not in repository"]
fn compare_shape_vs_occt() {
    // 1. Parse Shape.step and build mesh
    let step_path = test_data("Shape.step");
    let text = std::fs::read_to_string(&step_path).expect("read step");
    let exchange =
        parser::parse_exchange_with_options(&text, &StepImportOptions::default()).expect("parse");
    let brep = build_brep(&exchange.entities).expect("brep");
    let reg = brep.registry;
    let skip_face_keys = Vec::new();
    // Skip heal — it incorrectly removes circle edges (v_start==v_end) from wires,
    // collapsing 4-edge loops to 2-edge degenerate loops.
    // for &sk in &brep.root_solids {
    //     if let Some(solid) = reg.solids.get(sk) {
    //         let heal = auto_heal_shell(solid.outer_shell, &mut reg, HealLevel::Standard, 5);
    //         skip_face_keys.extend(heal.skip_face_keys);
    //     }
    // }
    let mesh_config = BRepMeshConfig::default();
    let mut our_mesh = MeshResult::default();
    let mut our_report: Option<ShellMeshReport> = None;
    for &sk in &brep.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let out = mesh_brep_shell_with_report(solid.outer_shell, &reg, &mesh_config, &skip_face_keys);
            our_report = Some(out.report);
            our_mesh = out.mesh;
        }
    }

    // 2. Load OCCT reference STL
    let ref_path = test_data("shape-tri.stl");
    let data = std::fs::read(&ref_path).expect("read reference stl");
    let tris = parse_stl_triangles(&data).expect("parse reference stl");
    let mut ref_mesh = MeshResult::default();
    for tri in tris {
        let base = ref_mesh.vertices.len() as i32;
        ref_mesh.vertices.push(Vec3::from(tri.vertices[0]));
        ref_mesh.vertices.push(Vec3::from(tri.vertices[1]));
        ref_mesh.vertices.push(Vec3::from(tri.vertices[2]));
        ref_mesh.indices.extend_from_slice(&[base, base + 1, base + 2, -1]);
    }

    // 3. Bounding box comparison
    let bbox = |verts: &[Vec3]| -> (Vec3, Vec3) {
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for v in verts { mn = mn.min(*v); mx = mx.max(*v); }
        (mn, mx)
    };
    let (our_min, our_max) = bbox(&our_mesh.vertices);
    let (ref_min, ref_max) = bbox(&ref_mesh.vertices);

    println!("\n=== Bounding Box ===");
    println!("  Ours:  min={:?}  max={:?}", our_min, our_max);
    println!("  OCCT:  min={:?}  max={:?}", ref_min, ref_max);
    println!("  Ours diag={:.2}  OCCT diag={:.2}", (our_max - our_min).length(), (ref_max - ref_min).length());
    println!("  Our tris: {}  OCCT tris: {}", our_mesh.indices.len() / 4, ref_mesh.indices.len() / 4);

    // 4. Hausdorff distance
    let h = hausdorff_meshes(&our_mesh, &ref_mesh, 1024);
    println!("\n=== Hausdorff (1024 directions) ===");
    println!("  sym_p95={:.4}  sym_max={:.4}", h.symmetric_p95, h.symmetric_max);
    println!("  our→ref p95={:.4} max={:.4}", h.p95_a_to_b, h.max_a_to_b);
    println!("  ref→our p95={:.4} max={:.4}", h.p95_b_to_a, h.max_b_to_a);

    // 5. Per-face report
    if let Some(report) = &our_report {
        println!("\n=== Per-face ===");
        for fs in &report.faces {
            let face = reg.faces.get(fs.face_key).unwrap();
            let kind = format!("{:?}", std::mem::discriminant(&face.surface));
            println!("  {:?} kind={} tris={} chord={:.4} fb={} uv={:?}",
                fs.face_key, kind, fs.tri_count, fs.max_chord_error, fs.grid_fallback, fs.uv_source);
        }
    }
}
