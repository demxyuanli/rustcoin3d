use rc3d_app::{App, CameraController};
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();
    log::info!("=== {} demo ===", "Markup Dimensions");
    print_markup_help();

    let mut graph = rc3d_scene::SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    let camera = PerspectiveCameraNode::look_at(
        Vec3::new(0.0, 0.0, 10.0),
        Vec3::new(200.0, 150.0, 0.0),
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        800.0 / 600.0,
    );
    graph.add_child(root, NodeData::PerspectiveCamera(camera));

    // Markup node with all dimension types
    graph.add_child(
        root,
        NodeData::Markup(MarkupNode {
            elements: vec![
                // Linear dimension
                MarkupElement::Dimension {
                    start: [50.0, 50.0],
                    end: [250.0, 50.0],
                    offset_dir: [0.0, -1.0],
                    extension_len: 20.0,
                    arrow_size: 8.0,
                    label: "200 px".into(),
                    color: [1.0, 0.5, 0.0, 1.0],
                },
                // Angle dimension
                MarkupElement::AngleDimension {
                    center: [50.0, 200.0],
                    arm1: [250.0, 200.0],
                    arm2: [50.0, 50.0],
                    radius: 60.0,
                    color: [0.0, 0.8, 0.0, 1.0],
                    label: "90°".into(),
                },
                // Radial dimension
                MarkupElement::RadialDimension {
                    center: [200.0, 150.0],
                    perimeter: [280.0, 150.0],
                    color: [0.0, 0.5, 1.0, 1.0],
                    label: "R=80".into(),
                },
                // Diameter dimension
                MarkupElement::DiameterDimension {
                    p1: [50.0, 250.0],
                    p2: [250.0, 250.0],
                    center: [150.0, 250.0],
                    color: [0.8, 0.0, 0.8, 1.0],
                    label: "D=200".into(),
                },
                // Leader line
                MarkupElement::Leader {
                    anchor: [300.0, 100.0],
                    label_pos: [350.0, 80.0],
                    text: "Note: corner".into(),
                    color: [0.5, 0.5, 0.5, 1.0],
                },
            ],
            layer_name: "dimensions".into(),
            visible: true,
        }),
    );

    let markup_center = Vec3::new(200.0, 150.0, 0.0);
    let orbit = CameraController::new(markup_center, 380.0);
    let mut app = App::new(graph).with_camera_controller(orbit);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn print_markup_help() {
    println!("Markup Dimensions example");
    println!("Usage: cargo run -p rc3d-app --example markup_dimensions");
    println!("Features:");
    println!("  Linear dimension with arrowheads");
    println!("  Angle dimension (arc + arms)");
    println!("  Radial dimension (center-to-perimeter)");
    println!("  Diameter dimension (through-center with cross)");
    println!("  Leader line with anchor dot");
    println!("Controls:");
    println!(
        "  Middle mouse drag: orbit | Right drag: pan | Scroll wheel: zoom — Left drag also orbits here"
    );
    println!("  W / S / E / H: display mode shortcuts | Escape: clear selection");
}
