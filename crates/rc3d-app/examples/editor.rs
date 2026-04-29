//! Editor example — demonstrates gizmo, multi-viewport, undo/redo, and section planes.
//!
//! Keys:
//!   T/R/G: Gizmo Translate/Rotate/Scale
//!   Ctrl+Z/Y: Undo/Redo
//!   C: Cycle viewport layout (Single→Quad→Left+Right→TopBottom)
//!   Tab: Cycle active viewport
//!   P: Toggle section edit, [/]: nudge
//!   X/Y/Z: Toggle axis clip
//!   F: Fit selection
//!   Ctrl+Left-drag: Box select
//!   M: Measurement mode, Left-click place points
//!   Escape: Clear selection + measurements
//!   W/S/E/H: Cycle display mode
//!   I: Cycle IBL preset

use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d=info"),
    )
    .init();

    println!("rustcoin3d Editor");
    println!("  T/R/G: Gizmo  |  Ctrl+Z/Y: Undo/Redo  |  C: Viewport layout");
    println!("  F: Fit  |  P+[/]: Section  |  M: Measure  |  Escape: Clear");

    let graph = build_demo_scene();

    let mut app =
        App::new(graph).with_initial_display_mode(DisplayMode::ShadedWithEdges);

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn build_demo_scene() -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera
    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(5.0, 4.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // Lights
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity: 1.2,
        }),
    );
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(0.6, -0.4, 0.5).normalize(),
            color: Vec3::new(0.6, 0.7, 1.0),
            intensity: 0.5,
        }),
    );

    // Floor
    let floor_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        floor_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.4, 0.4, 0.45),
            ambient_color: Vec3::new(0.05, 0.05, 0.05),
            specular_color: Vec3::new(0.1, 0.1, 0.1),
            shininess: 4.0,
            base_color: Vec3::new(0.4, 0.4, 0.45),
            metallic: 0.0,
            roughness: 0.9,
            albedo_texture: None,
            opacity: 1.0,
        }),
    );
    graph.add_child(
        floor_sep,
        NodeData::Cube(CubeNode {
            width: 10.0,
            height: 0.2,
            depth: 10.0,
        }),
    );

    // Demo objects with transforms (selectable + gizmo-operable)
    let cube_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        cube_sep,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(-2.0, 1.0, 0.0))),
    );
    graph.add_child(
        cube_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.8, 0.3, 0.3),
            ambient_color: Vec3::new(0.1, 0.02, 0.02),
            specular_color: Vec3::new(0.3, 0.1, 0.1),
            shininess: 64.0,
            base_color: Vec3::new(0.8, 0.3, 0.3),
            metallic: 0.1,
            roughness: 0.4,
            albedo_texture: None,
            opacity: 1.0,
        }),
    );
    graph.add_child(
        cube_sep,
        NodeData::Cube(CubeNode {
            width: 1.5,
            height: 1.5,
            depth: 1.5,
        }),
    );

    let sphere_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        sphere_sep,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(2.0, 1.0, 0.0))),
    );
    graph.add_child(
        sphere_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.3, 0.6, 0.3),
            ambient_color: Vec3::new(0.02, 0.1, 0.02),
            specular_color: Vec3::new(0.2, 0.6, 0.2),
            shininess: 128.0,
            base_color: Vec3::new(0.3, 0.6, 0.3),
            metallic: 0.5,
            roughness: 0.2,
            albedo_texture: None,
            opacity: 1.0,
        }),
    );
    graph.add_child(
        sphere_sep,
        NodeData::Sphere(SphereNode { radius: 1.0 }),
    );

    let cyl_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        cyl_sep,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 1.0, -2.5))),
    );
    graph.add_child(
        cyl_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.3, 0.4, 0.8),
            ambient_color: Vec3::new(0.02, 0.02, 0.1),
            specular_color: Vec3::new(0.2, 0.3, 0.8),
            shininess: 96.0,
            base_color: Vec3::new(0.3, 0.4, 0.8),
            metallic: 0.3,
            roughness: 0.3,
            albedo_texture: None,
            opacity: 1.0,
        }),
    );
    graph.add_child(
        cyl_sep,
        NodeData::Cylinder(CylinderNode {
            radius: 0.7,
            height: 2.0,
        }),
    );

    graph
}
