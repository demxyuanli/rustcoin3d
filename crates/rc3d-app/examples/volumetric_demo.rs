//! Volumetric fog demo — scene with lights in foggy environment.
//!
//! Demonstrates:
//!   - Volumetric fog with height-based density
//!   - Directional light god rays
//!   - Point lights illuminating fog
//!   - Focus distance control
//!
//! Keys:
//!   1-4: Fog density presets (light/medium/heavy/off)
//!   5-6: Adjust height falloff
//!   Mouse drag: orbit camera
//!   Escape: clear selection | close window to quit
//!
use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::{math::Vec3, DisplayMode};
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();
    print_volumetric_demo_help();

    let scene = build_fog_scene();

    let ctrl = CameraController::new(Vec3::new(2.0, 1.5, 5.0), 8.0);

    let mut app = App::new(scene)
        .with_camera_controller(ctrl)
        .with_initial_display_mode(DisplayMode::Shaded)
        .with_hdr_post_processing(true);

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn print_volumetric_demo_help() {
    println!("Volumetric fog demo");
    println!("Usage: cargo run -p rc3d-app --example volumetric_demo");
    println!("Controls:");
    println!("  1-4: Fog density presets (light/medium/heavy/off)");
    println!("  5-6: Adjust fog height/falloff");
    println!("  Mouse drag: orbit camera");
    println!("  Escape: clear selection | Close window to quit");
    println!("Feature switches:");
    println!("  Volumetric fog + HDR post-processing enabled by default");
}

fn build_fog_scene() -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera (slightly elevated to see fog layering)
    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(5.0, 2.5, 3.0),
            Vec3::new(0.0, 1.0, -5.0),
            Vec3::Y,
            std::f32::consts::FRAC_PI_3,
            800.0 / 600.0,
        )),
    );

    // Strong directional light (creates god rays in fog)
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.4, -0.5, 0.6).normalize(),
            color: Vec3::new(1.0, 0.9, 0.7), // warm
            intensity: 3.0,
            light_group: None,
        }),
    );

    // Point lights scattered through the scene to illuminate fog
    let light_positions = [
        Vec3::new(-4.0, 2.0, -3.0),
        Vec3::new(4.0, 1.5, -4.0),
        Vec3::new(0.0, 3.0, -6.0),
        Vec3::new(-2.0, 0.8, -8.0),
        Vec3::new(2.0, 0.5, -8.0),
    ];
    let light_colors = [
        Vec3::new(1.0, 0.3, 0.2), // red
        Vec3::new(0.2, 0.4, 1.0), // blue
        Vec3::new(1.0, 0.8, 0.3), // yellow
        Vec3::new(0.3, 1.0, 0.3), // green
        Vec3::new(1.0, 0.5, 0.8), // pink
    ];
    for (i, (&pos, &color)) in light_positions.iter().zip(light_colors.iter()).enumerate() {
        let light_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            light_sep,
            NodeData::Transform(TransformNode::from_translation(pos)),
        );
        graph.add_child(
            light_sep,
            NodeData::PointLight(PointLightNode {
                location: Vec3::ZERO,
                color,
                intensity: 15.0,
            light_group: None,
            }),
        );
        // Small visible sphere at light position
        graph.add_child(
            light_sep,
            NodeData::Material(MaterialNode {
                diffuse_color: color,
                ambient_color: color * 0.5,
                specular_color: Vec3::ONE,
                shininess: 256.0,
                base_color: color,
                metallic: 0.0,
                roughness: 0.2,
                opacity: 1.0,
                ..Default::default()
            }),
        );
        graph.add_child(light_sep, NodeData::Sphere(SphereNode { radius: 0.15 }));
        let _ = i;
    }

    // Ground plane (large to catch fog)
    let ground_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        ground_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.3, 0.35, 0.4),
            ambient_color: Vec3::new(0.02, 0.02, 0.02),
            specular_color: Vec3::new(0.1, 0.1, 0.1),
            shininess: 16.0,
            base_color: Vec3::new(0.3, 0.35, 0.4),
            metallic: 0.1,
            roughness: 0.8,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(
        ground_sep,
        NodeData::Cube(CubeNode {
            width: 30.0,
            height: 0.2,
            depth: 30.0,
        }),
    );

    // Series of pillars to show fog depth
    for z in 0..8 {
        let dist = -2.0 - z as f32 * 1.5;
        for x in 0..3 {
            let pillar_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(
                pillar_sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(
                    x as f32 * 3.0 - 3.0,
                    1.5,
                    dist,
                ))),
            );
            graph.add_child(
                pillar_sep,
                NodeData::Material(MaterialNode {
                    diffuse_color: Vec3::new(0.6, 0.55, 0.5),
                    ambient_color: Vec3::new(0.03, 0.03, 0.03),
                    specular_color: Vec3::new(0.1, 0.1, 0.1),
                    shininess: 8.0,
                    base_color: Vec3::new(0.6, 0.55, 0.5),
                    metallic: 0.0,
                    roughness: 0.7,
                    opacity: 1.0,
                    ..Default::default()
                }),
            );
            graph.add_child(
                pillar_sep,
                NodeData::Cylinder(CylinderNode {
                    radius: 0.3,
                    height: 3.0,
                }),
            );
            let _ = x;
        }
    }

    graph
}
