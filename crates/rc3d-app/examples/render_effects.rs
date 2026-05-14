//! Render effects demo using the new high-level API.
//!
//! Demonstrates EffectGraph configuration with shadows and post-processing.
//!
//! Usage: cargo run -p rc3d-app --example render_effects

use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_effects::{PostEffect, RenderConfig, Shadow};
use rc3d_scene_api::{Cube, DirectionalLight, Material, PerspectiveCamera, Scene, Sphere};

fn main() {
    env_logger::init();
    println!("=== Render Effects Demo (new API) ===");

    let mut scene = Scene::new();

    // Camera
    scene.set_camera(PerspectiveCamera::look_at(
        Vec3::new(3.0, 3.0, 8.0),
        Vec3::ZERO,
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        800.0 / 600.0,
    ));

    // Light
    scene.add_light(DirectionalLight::sun(
        Vec3::new(-0.5, -0.8, -0.3),
        1.2,
    ));

    // Two spheres and a cube with PBR materials
    scene.add(
        Sphere::default()
            .radius(1.5)
            .at(-2.5, 0.0, 0.0)
            .material(Material::pbr().base_color(0.8, 0.2, 0.2).roughness(0.3).metallic(0.1)),
    );
    scene.add(
        Sphere::default()
            .radius(1.5)
            .at(2.5, 0.0, 0.0)
            .material(Material::pbr().base_color(0.2, 0.6, 0.8).roughness(0.7).metallic(0.8)),
    );
    scene.add(
        Cube::default()
            .width(2.0)
            .height(1.0)
            .depth(2.0)
            .at(0.0, -2.5, 0.0)
            .material(Material::pbr().base_color(0.3, 0.7, 0.3).roughness(0.4)),
    );

    // Effects: 4-cascade CSM shadow + full post-processing chain
    let effects = RenderConfig::new()
        .shadow(Shadow::CSM(rc3d_effects::CsmConfig {
            cascade_count: 4,
            resolution: 2048,
            soft: true,
        }))
        .enable(PostEffect::SSAO)
        .enable(PostEffect::SSR)
        .enable(PostEffect::Bloom)
        .enable(PostEffect::TAA)
        .enable(PostEffect::Tonemap)
        .build();

    let controller = CameraController::new(Vec3::ZERO, 14.0);
    let mut app = App::from_scene(scene)
        .with_camera_controller(controller)
        .with_effects_config(effects);

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}
