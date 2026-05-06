use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();
    print_billboard_help();

    let mut graph = rc3d_scene::SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    let camera = PerspectiveCameraNode::look_at(
        Vec3::new(3.0, 2.0, 5.0),
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
        }),
    );

    // Billboard node: children always face the camera
    let billboard = graph.add_child(root, NodeData::Billboard(BillboardNode {
        axis_aligned: false, // spherical billboard
    }));

    // Text2 inside billboard stays screen-aligned
    graph.add_child(
        billboard,
        NodeData::Text2(Text2Node {
            string: "Facing Camera".into(),
            position: [0.0, 0.0],
            size: 24.0,
            color: [1.0, 0.8, 0.0, 1.0],
        }),
    );

    // Reference cube at origin
    graph.add_child(root, NodeData::Cube(CubeNode {
        width: 0.5, height: 0.5, depth: 0.5,
    }));

    let mut app = App::new(graph);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn print_billboard_help() {
    println!("Billboard example");
    println!("Usage: cargo run -p rc3d-app --example billboard");
    println!("Features:");
    println!("  BillboardNode (spherical) with Text2 child always facing camera");
    println!("  Orbit camera to see billboard effect");
}
