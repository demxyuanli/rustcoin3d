use rc3d_app::{camera_controller_from_scene_bounds, App};
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();
    log::info!("=== {} demo ===", "scripted_scene");
    let script = r#"
        add_cube(0.0, 0.0, 0.0);
        add_cube(2.0, 0.0, 0.0);
        add_cube(0.0, 0.0, 2.0);
        add_sphere(0.0, 2.0, 0.0, 0.8);
        add_light(-1.0, -1.0, -1.0);
    "#;

    let mut graph = rc3d_script::execute_scene_script(script).expect("script execution failed");

    // Add camera
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

    let orbit = camera_controller_from_scene_bounds(&graph, Vec3::new(1.0, 0.5, 0.5), 8.0);
    let mut app = App::new(graph).with_camera_controller(orbit);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}
