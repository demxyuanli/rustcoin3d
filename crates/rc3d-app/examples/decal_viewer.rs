//! Decal demo — screen-space projected texture overlay on geometry.
use rc3d_app::{App, CameraController};
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();
    let mut g = rc3d_scene::SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    log::info!("=== {} demo ===", "Decal");
    g.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(3.0, 2.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );
    g.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }),
    );
    g.add_child(
        root,
        NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.5, 0.5, 0.5),
            roughness: 0.4,
            albedo_texture: Some("decal.png".into()),
            ..Default::default()
        }),
    );
    // Large flat ground to project decal onto
    let ground = g.add_child(
        root,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, -1.0, 0.0))),
    );
    g.add_child(
        ground,
        NodeData::Cube(CubeNode {
            width: 5.0,
            height: 0.1,
            depth: 5.0,
        }),
    );
    // Decal projected from above onto the ground
    g.add_child(
        root,
        NodeData::Decal(DecalNode {
            position: Vec3::new(0.0, 0.5, 0.0),
            direction: Vec3::NEG_Y,
            size: [2.0, 2.0],
            texture_path: "decal.png".into(),
            color: [1.0, 0.5, 0.0, 0.8],
            opacity: 0.8,
        }),
    );
    println!(
        "Decal shader at shaders/decal_project.wgsl — ensure_decal_pass() initializes pipeline"
    );
    let orbit = CameraController::new(Vec3::ZERO, 10.0);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut App::new(g).with_camera_controller(orbit))
        .expect("event loop");
}
