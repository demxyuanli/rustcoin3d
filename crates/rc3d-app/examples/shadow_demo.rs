//! Shadow type demo — directional CSM + point light + spot light shadows.
//!
//! Scene: A room-like setup with objects casting shadows from multiple light types.
//!
//! Keys:
//!   1: Toggle directional light shadow
//!   2: Toggle point light shadow
//!   3: Toggle spot light shadow
//!   +/-: Adjust shadow bias
//!   Mouse drag: orbit camera

use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::{math::Vec3, DisplayMode};
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();
    print_shadow_demo_help();

    let scene = build_shadow_scene();

    let ctrl = CameraController::new(Vec3::new(0.0, 2.0, 5.0), 8.0);

    let mut app = App::new(scene)
        .with_camera_controller(ctrl)
        .with_initial_display_mode(DisplayMode::Shaded)
        .with_hdr_post_processing(true);

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn print_shadow_demo_help() {
    println!("Shadow demo");
    println!("Usage: cargo run -p rc3d-app --example shadow_demo");
    println!("Controls:");
    println!("  1: Toggle directional shadow (CSM)");
    println!("  2: Toggle point-light shadow");
    println!("  3: Toggle spot-light shadow");
    println!("  +/-: Adjust shadow bias");
    println!("  Mouse drag: orbit camera");
    println!("  ESC: exit");
    println!("Feature switches:");
    println!("  Directional + point + spot shadow types in one scene");
}

fn build_shadow_scene() -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera
    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(4.0, 6.0, 8.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // Directional light (casts CSM shadow)
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity: 1.2,
            light_group: None,
        }),
    );

    // Point light (casts cubemap shadow)
    let pt_light_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        pt_light_sep,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(3.0, 4.0, 0.0))),
    );
    graph.add_child(
        pt_light_sep,
        NodeData::PointLight(PointLightNode {
            location: Vec3::ZERO,
            color: Vec3::new(0.8, 0.5, 0.3),
            intensity: 30.0,
            light_group: None,
        }),
    );
    // Visible marker for point light
    graph.add_child(
        pt_light_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(1.0, 0.5, 0.2),
            ambient_color: Vec3::new(0.5, 0.25, 0.1),
            specular_color: Vec3::ONE,
            shininess: 256.0,
            base_color: Vec3::new(1.0, 0.5, 0.2),
            metallic: 0.0,
            roughness: 0.1,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(pt_light_sep, NodeData::Sphere(SphereNode { radius: 0.15 }));

    // Spot light (casts perspective shadow)
    let sp_light_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        sp_light_sep,
        NodeData::Transform(TransformNode {
            translation: Vec3::new(-3.0, 5.0, 3.0),
            rotation: rc3d_core::math::Mat4::look_at_rh(
                Vec3::new(-3.0, 5.0, 3.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::Y,
            ),
            scale: Vec3::ONE,
            center: Vec3::ZERO,
        }),
    );
    graph.add_child(
        sp_light_sep,
        NodeData::SpotLight(SpotLightNode {
            location: Vec3::ZERO,
            direction: Vec3::new(3.0, -5.0, -3.0).normalize(),
            color: Vec3::new(0.3, 0.7, 1.0),
            intensity: 40.0,
            light_group: None,
            cut_off_angle: 0.5,
            drop_off_rate: 4.0,
        }),
    );

    // Floor
    add_shadow_floor(&mut graph, root);

    // Central large sphere (main shadow caster)
    let center_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        center_sep,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 1.5, 0.0))),
    );
    graph.add_child(
        center_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.7, 0.6, 0.5),
            ambient_color: Vec3::new(0.05, 0.05, 0.05),
            specular_color: Vec3::new(0.3, 0.3, 0.3),
            shininess: 64.0,
            base_color: Vec3::new(0.7, 0.6, 0.5),
            metallic: 0.2,
            roughness: 0.3,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(center_sep, NodeData::Sphere(SphereNode { radius: 1.2 }));

    // Small spheres around to cast multiple shadows
    let offsets = [
        Vec3::new(2.5, 0.8, 1.5),
        Vec3::new(-2.0, 0.6, -1.0),
        Vec3::new(1.0, 0.5, -2.5),
        Vec3::new(-1.5, 0.9, 2.0),
    ];
    for &offset in &offsets {
        let small_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            small_sep,
            NodeData::Transform(TransformNode::from_translation(offset)),
        );
        graph.add_child(
            small_sep,
            NodeData::Material(MaterialNode {
                diffuse_color: Vec3::new(0.5, 0.7, 0.6),
                ambient_color: Vec3::new(0.03, 0.03, 0.03),
                specular_color: Vec3::new(0.2, 0.2, 0.2),
                shininess: 32.0,
                base_color: Vec3::new(0.5, 0.7, 0.6),
                metallic: 0.05,
                roughness: 0.5,
                opacity: 1.0,
                ..Default::default()
            }),
        );
        graph.add_child(small_sep, NodeData::Sphere(SphereNode { radius: 0.5 }));
    }

    graph
}

fn add_shadow_floor(graph: &mut SceneGraph, parent: rc3d_core::NodeId) {
    let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));
    graph.add_child(
        sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.8, 0.8, 0.8),
            ambient_color: Vec3::new(0.05, 0.05, 0.05),
            specular_color: Vec3::new(0.05, 0.05, 0.05),
            shininess: 4.0,
            base_color: Vec3::new(0.8, 0.8, 0.8),
            metallic: 0.0,
            roughness: 0.9,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(
        sep,
        NodeData::Cube(CubeNode {
            width: 16.0,
            height: 0.2,
            depth: 16.0,
        }),
    );
}
