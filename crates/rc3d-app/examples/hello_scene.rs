//! Minimal scene using the new high-level API.
//!
//! Compare with `cube.rs` — this does the same thing with far less boilerplate.
//!
//! Usage: cargo run -p rc3d-app --example hello_scene

use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_scene_api::{Cube, DirectionalLight, Material, PerspectiveCamera, Scene};

fn main() {
    env_logger::init();
    println!("=== Hello Scene (new API) ===");

    let mut scene = Scene::new();

    // Camera
    scene.set_camera(PerspectiveCamera::look_at(
        Vec3::new(2.0, 2.0, 4.0),
        Vec3::ZERO,
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        800.0 / 600.0,
    ));

    // Light
    scene.add_light(DirectionalLight::sun(
        Vec3::new(-1.0, -1.0, -1.0),
        1.0,
    ));

    // Cube with material
    scene.add(
        Cube::default()
            .material(Material::pbr().base_color(0.2, 0.5, 0.8).roughness(0.3)),
    );

    let controller = CameraController::new(Vec3::ZERO, 10.0);
    let mut app = App::from_scene(scene).with_camera_controller(controller);

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}
