//! Scripted scene — create geometry from a scene description script.
//!
//! Usage: cargo run -p rc3d-examples --example scripted_scene

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    let script = r#"
        add_cube(0.0, 0.0, 0.0);
        add_cube(2.0, 0.0, 0.0);
        add_cube(0.0, 0.0, 2.0);
        add_sphere(0.0, 2.0, 0.0, 0.8);
        add_light(-1.0, -1.0, -1.0);
    "#;

    run_example("Scripted Scene", |engine| {
        let mut graph = rc3d_script::execute_scene_script(script)
            .expect("script execution failed");

        let root = graph.roots().first().copied().unwrap();
        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(3.0, 3.0, 5.0),
                Vec3::new(1.0, 0.5, 0.5),
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        engine.load_scene(graph);
    });
}
