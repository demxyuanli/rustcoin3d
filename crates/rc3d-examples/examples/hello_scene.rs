//! Minimal scene using the scene-api DSL, driven by the Engine facade.
//!
//! Usage: cargo run -p rc3d-examples --example hello_scene

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene_api::{Cube, DirectionalLight, Material, PerspectiveCamera, Scene};

fn main() {
    run_example("Hello Scene", |engine| {
        let mut scene = Scene::new();

        scene.set_camera(PerspectiveCamera::look_at(
            Vec3::new(2.0, 2.0, 4.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        ));

        scene.add_light(DirectionalLight::sun(
            Vec3::new(-1.0, -1.0, -1.0),
            1.0,
        ));

        scene.add(
            Cube::default()
                .material(Material::pbr().base_color(0.2, 0.5, 0.8).roughness(0.3)),
        );

        engine.load_scene(scene.build());
    });
}
