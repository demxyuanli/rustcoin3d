use rc3d_app::{App, CameraController};
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();
    log::info!("=== {} demo ===", "Cube");
    print_cube_help();

    let mut graph = rc3d_scene::SceneGraph::new();

    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera looking at origin from above-right-front
    let camera = PerspectiveCameraNode::look_at(
        Vec3::new(2.0, 2.0, 4.0),
        Vec3::ZERO,
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        800.0 / 600.0,
    );
    graph.add_child(root, NodeData::PerspectiveCamera(camera));

    // Light
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }),
    );

    // Material
    graph.add_child(
        root,
        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.5, 0.8))),
    );

    // Cube
    graph.add_child(root, NodeData::Cube(CubeNode::default()));

    let orbit = CameraController::new(Vec3::ZERO, 10.0);
    let mut app = App::new(graph).with_camera_controller(orbit);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn print_cube_help() {
    println!("Cube example");
    println!("Usage: cargo run -p rc3d-app --example cube");
    println!("Controls:");
    println!(
        "  Middle mouse drag: orbit | Right drag: pan | Scroll wheel: zoom — Left drag also orbits here"
    );
    println!("  W / S / E / H: wireframe / shaded / shaded+edges / hidden-line display mode");
    println!("  Escape: clear selection | Close window to quit");
    println!("Feature switches:");
    println!("  Single directional light + single material baseline");
}
