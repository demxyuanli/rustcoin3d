//! Section plane cap — sphere clipped by a plane with filled cross-section.
//! Camera: left or middle drag = orbit, right drag = pan, wheel = zoom.

use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();
    print_section_help();

    let mut graph = rc3d_scene::SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    let eye = Vec3::new(3.0, 2.5, 4.0);
    let camera = PerspectiveCameraNode::look_at(
        eye,
        Vec3::ZERO,
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        800.0 / 600.0,
    );
    graph.add_child(root, NodeData::PerspectiveCamera(camera));

    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }),
    );

    // Material for the sphere
    graph.add_child(
        root,
        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.3, 0.6, 0.9))),
    );

    // Sphere to be clipped
    graph.add_child(root, NodeData::Sphere(SphereNode { radius: 1.5 }));

    // Section plane cutting through the sphere (Y=0 plane)
    graph.add_child(
        root,
        NodeData::SectionPlane(SectionPlaneNode {
            plane: [0.0, 1.0, 0.0, 0.3], // Y = -0.3
            enabled: true,
            cap_color: [0.8, 0.3, 0.3, 1.0], // Red cap
            cap_enabled: true,
        }),
    );

    let dist = eye.length();
    let mut ctrl = CameraController::new(Vec3::ZERO, dist);
    ctrl.yaw = eye.x.atan2(eye.z);
    ctrl.pitch = (eye.y / dist).asin();

    let mut app = App::new(graph)
        .with_camera_controller(ctrl)
        .with_window_title("section_caps");
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn print_section_help() {
    println!("Section Plane Cap example");
    println!("Usage: cargo run -p rc3d-app --example section_caps");
    println!("Features:");
    println!("  Sphere clipped by Y=-0.3 section plane");
    println!("  cap_enabled=true with red cap_color");
    println!("  The cut surface should appear filled (not hollow)");
    println!("Controls:");
    println!("  Left / middle drag: orbit camera");
    println!("  Right drag: pan");
    println!("  Mouse wheel: zoom");
}
