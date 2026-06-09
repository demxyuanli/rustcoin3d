//! Multi-hole plate STEP solid test: square hole + true circular hole.
//!
//! Reads `steps/HoledPlate.step` (hand-written closed-shell BRep) and verifies
//! top/bottom face triangles do not fall inside the holes after meshing.

use rc3d_io::step::brep::heal::HealLevel;
use rc3d_io::step::{
    emit_plan_options_from_step, import_step_file_with_options, StepImportOptions,
};
use rc3d_shape::ToleranceContext;
use std::path::PathBuf;

fn steps_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("steps")
}

#[test]
fn test_multi_hole_step_import_and_mesh() {
    let step_path = steps_dir().join("HoledPlate.step");
    assert!(step_path.exists(), "HoledPlate.step not found at {:?}", step_path);

    let mut import_options = StepImportOptions::default();
    import_options.heal_level = HealLevel::Basic;
    import_options.skip_visualization = true;

    let mut result = import_step_file_with_options(&step_path, &import_options)
        .expect("import HoledPlate.step");

    let mut plan_options = emit_plan_options_from_step(&import_options);
    ToleranceContext::from_model(result.document.store.tolerance.model)
        .apply_to_mesh_config(&mut plan_options.mesh_config);
    let plan = result
        .document
        .build_emit_plan(&plan_options)
        .expect("build emit plan");

    let mut total_tris = 0usize;
    for inst in &plan.instances {
        let cached = plan
            .mesh_table
            .get(&inst.mesh_slot)
            .expect("mesh slot exists");
        total_tris += cached.mesh.indices.len() / 4;
    }
    assert!(total_tris > 0, "expected mesh triangles from holed plate STEP");

    // Holes: square + circle approximated by 32-gon for point-in-trim test.
    let holes: Vec<Vec<(f32, f32)>> = vec![
        vec![(19.0,34.0),(19.0,46.0),(31.0,46.0),(31.0,34.0)],
        (0..32).map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / 32.0;
            (50.0 + angle.cos() * 8.0, 40.0 + angle.sin() * 8.0)
        }).collect(),
    ];
    let outer_poly = vec![(0.0,0.0),(100.0,0.0),(100.0,80.0),(0.0,80.0)];

    for inst in &plan.instances {
        let cached = plan.mesh_table.get(&inst.mesh_slot).unwrap();
        let mesh = &cached.mesh;
        for chunk in mesh.indices.chunks(4) {
            if chunk.len() < 4 {
                continue;
            }
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
                continue;
            }
            let p0 = mesh.vertices[i0];
            let p1 = mesh.vertices[i1];
            let p2 = mesh.vertices[i2];
            let cz = (p0.z + p1.z + p2.z) / 3.0;
            // Only check top/bottom face triangles; skip vertical side faces.
            if cz < 0.5 || cz > 4.5 {
                let cu = (p0.x + p1.x + p2.x) / 3.0;
                let cv = (p0.y + p1.y + p2.y) / 3.0;
                let c = rc3d_core::math::Vec3::new(cu, cv, 0.0);
                let c_local = inst.world_transform.inverse() * c.extend(1.0);
                let cu_local = c_local.x;
                let cv_local = c_local.y;
                assert!(
                    rc3d_shape::mesh::face_uv::point_in_trim(cu_local, cv_local, &outer_poly, &holes),
                    "triangle centroid ({cu_local},{cv_local}) outside trim or inside hole"
                );
            }
        }
    }
}
