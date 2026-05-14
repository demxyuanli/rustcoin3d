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
        Vec3::new(4.0, 3.0, 12.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        800.0 / 600.0,
    );
    graph.add_child(root, NodeData::PerspectiveCamera(camera));

    // ══  Cube (left, orange annotations)  ══
    let cube_x = -3.5;
    let cube_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        cube_sep,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(cube_x, 0.0, 0.0))),
    );
    let hw = 0.75; // half-width for 1.5 cube
    graph.add_child(cube_sep, NodeData::Material(MaterialNode::from_diffuse(Vec3::new(1.0, 0.6, 0.2))));
    graph.add_child(cube_sep, NodeData::Cube(CubeNode { width: 1.5, height: 1.5, depth: 1.5 }));

    // ══  Sphere (center, green annotations)  ══
    let sphere_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        sphere_sep,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 0.5, 0.0))),
    );
    graph.add_child(sphere_sep, NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.8, 0.3))));
    graph.add_child(sphere_sep, NodeData::Sphere(SphereNode { radius: 1.0 }));

    // ══  Cylinder (right, blue annotations)  ══
    let cyl_x = 3.5;
    let cyl_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        cyl_sep,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(cyl_x, -0.3, 0.0))),
    );
    let cr = 0.8; // cylinder radius
    let ch = 1.0; // half-height
    graph.add_child(cyl_sep, NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.5, 1.0))));
    graph.add_child(cyl_sep, NodeData::Cylinder(CylinderNode { radius: cr, height: 2.0 }));

    // ══  3D Annotations (inside Annotation grouping node → overlay rendering)  ══
    let ann_group = graph.add_child(root, NodeData::Annotation(AnnotationNode));

    // Cube annotations (placed relative to cube's local space)
    let cube_ann = graph.add_child(ann_group, NodeData::Separator(SeparatorNode));
    graph.add_child(
        cube_ann,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(cube_x, 0.0, 0.0))),
    );
    graph.add_child(
        cube_ann,
        NodeData::AnnotationSet(AnnotationSetNode {
            elements: vec![
                // Width dimension on bottom face (Y = -hw, centered Z) — single face: ∥ XZ
                AnnotationElement::Dimension {
                    start: [-hw, -hw, 0.0],
                    end: [hw, -hw, 0.0],
                    offset_dir: [0.0, -1.0, 0.0],
                    extension_len: 0.3,
                    arrow_size: 0.15,
                    label: "W=1.5".into(),
                    color: [1.0, 0.4, 0.0, 1.0], // orange
                },
                // Height dimension on front face (Z = -hw, left edge) — single face: ∥ XY
                AnnotationElement::Dimension {
                    start: [-hw, -hw, -hw],
                    end: [-hw, hw, -hw],
                    offset_dir: [-1.0, 0.0, 0.0],
                    extension_len: 0.3,
                    arrow_size: 0.15,
                    label: "H=1.5".into(),
                    color: [1.0, 0.4, 0.0, 1.0],
                },
                // Leader from front face center
                AnnotationElement::Leader {
                    anchor: [0.0, 0.0, -hw],
                    label_offset: [40.0, -30.0],
                    text: "Cube".into(),
                    color: [0.9, 0.4, 0.0, 1.0],
                },
                // Datum at bottom-front-left corner (front face)
                AnnotationElement::Datum {
                    position: [-hw, -hw, -hw],
                    size: 0.12,
                    color: [1.0, 0.6, 0.0, 1.0],
                },
            ],
            visible: true,
        }),
    );

    // Sphere annotations
    let sphere_ann = graph.add_child(ann_group, NodeData::Separator(SeparatorNode));
    graph.add_child(
        sphere_ann,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 0.5, 0.0))),
    );
    graph.add_child(
        sphere_ann,
        NodeData::AnnotationSet(AnnotationSetNode {
            elements: vec![
                // Vertical diameter dimension
                AnnotationElement::Dimension {
                    start: [0.0, -1.0, 0.0],
                    end: [0.0, 1.0, 0.0],
                    offset_dir: [-1.0, 0.0, 0.0],
                    extension_len: 0.25,
                    arrow_size: 0.15,
                    label: "D=2.0".into(),
                    color: [0.0, 0.7, 0.3, 1.0], // green
                },
                // Leader from sphere surface on equatorial plane (Y=0) — single plane: ∥ XZ
                AnnotationElement::Leader {
                    anchor: [1.0, 0.0, 0.0],
                    label_offset: [50.0, -20.0],
                    text: "Sphere R=1.0".into(),
                    color: [0.0, 0.7, 0.3, 1.0],
                },
                // Datum at sphere surface south pole — single plane: ∥ XZ
                AnnotationElement::Datum {
                    position: [0.0, -1.0, 0.0],
                    size: 0.1,
                    color: [0.0, 0.9, 0.3, 1.0],
                },
            ],
            visible: true,
        }),
    );

    // Cylinder annotations
    let cyl_ann = graph.add_child(ann_group, NodeData::Separator(SeparatorNode));
    graph.add_child(
        cyl_ann,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(cyl_x, -0.3, 0.0))),
    );
    graph.add_child(
        cyl_ann,
        NodeData::AnnotationSet(AnnotationSetNode {
            elements: vec![
                // Height dimension on the side
                AnnotationElement::Dimension {
                    start: [cr, -ch, 0.0],
                    end: [cr, ch, 0.0],
                    offset_dir: [1.0, 0.0, 0.0],
                    extension_len: 0.25,
                    arrow_size: 0.14,
                    label: "H=2.0".into(),
                    color: [0.1, 0.5, 1.0, 1.0], // blue
                },
                // Leader from cylinder top
                AnnotationElement::Leader {
                    anchor: [0.0, ch, 0.0],
                    label_offset: [40.0, -30.0],
                    text: "Cyl. R=0.8".into(),
                    color: [0.1, 0.5, 1.0, 1.0],
                },
                // Datum at top center
                AnnotationElement::Datum {
                    position: [0.0, ch, 0.0],
                    size: 0.1,
                    color: [0.2, 0.6, 1.0, 1.0],
                },
            ],
            visible: true,
        }),
    );

    let orbit = CameraController::new(Vec3::new(0.0, 0.0, 0.0), 12.0);
    let mut app = App::new(graph).with_camera_controller(orbit);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn print_markup_help() {
    println!("Markup Dimensions — 3D Annotation demo");
    println!("Usage: RUST_LOG=info cargo run -p rc3d-app --example markup_dimensions");
    println!("Features:");
    println!("  3D objects: Cube, Sphere, Cylinder");
    println!("  Cube: W dim (bottom face ∥XZ) + H dim (front face ∥XY) + leader + datum");
    println!("  Each annotation on a single plane parallel to coordinate axes");
    println!("Controls:");
    println!("  Middle mouse drag: orbit | Right drag: pan | Scroll wheel: zoom");
    println!("  W/S/E/H: display mode | Escape: clear selection");
    println!("  Rotate/zoom — annotations track in screen space");
}
