use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_render::background::{BgMode, BgSettings};
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();
    log::info!("=== {} demo ===", "Walk Camera");
    print_walk_help();

    let mut graph = rc3d_scene::SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    let camera = PerspectiveCameraNode::look_at(
        Vec3::new(3.0, 1.5, 5.0),
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

    // Floor grid of cubes
    for x in -2..=2i32 {
        for z in -2..=2i32 {
            let tf = graph.add_child(
                root,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(
                    x as f32 * 2.0,
                    0.0,
                    z as f32 * 2.0,
                ))),
            );
            graph.add_child(
                tf,
                NodeData::Material(MaterialNode::from_diffuse(Vec3::new(
                    0.2,
                    0.4 + x as f32 * 0.1,
                    0.3 + z as f32 * 0.1,
                ))),
            );
            graph.add_child(tf, NodeData::Cube(CubeNode::default()));
        }
    }

    // Camera controller with walk mode enabled
    let mut ctrl = CameraController::new(Vec3::new(0.0, 1.0, 0.0), 5.0);
    ctrl.walk_mode = true;

    let mut app = App::new(graph)
        .with_camera_controller(ctrl)
        .with_background(BgSettings {
            mode: BgMode::VerticalGradient,
            top_color: [0.2, 0.35, 0.55, 1.0],
            bot_color: [0.03, 0.05, 0.12, 1.0],
            ..Default::default()
        });
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn print_walk_help() {
    println!("Walk Camera example");
    println!("Usage: cargo run -p rc3d-app --example walk_camera");
    println!("Features:");
    println!("  walk_mode=true on CameraController");
    println!("  walk(forward, right, up_down, speed) for WASD movement");
    println!("  turn(dx, dy) for mouse look");
    println!("  toggle_walk_mode() to switch between orbit and walk");
    println!("Note: keyboard input wiring to walk()/turn() is app-level");
    println!("  -- use editor example (cargo run -p rc3d-app --example editor)");
    println!("     for full keyboard binding support");
}
