//! Render effects demo — scene with shadows and post-processing.
//!
//! Usage: cargo run -p rc3d-examples --example render_effects

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene_api::{Cube, DirectionalLight, Material, PerspectiveCamera, Scene, Sphere};

fn main() {
    run_example("Render Effects", |engine| {
        let mut scene = Scene::new();

        scene.set_camera(PerspectiveCamera::look_at(
            Vec3::new(3.0, 3.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        ));

        scene.add_light(DirectionalLight::sun(Vec3::new(-0.5, -0.8, -0.3), 1.2));

        scene.add(
            Sphere::default()
                .radius(1.5).at(-2.5, 0.0, 0.0)
                .material(Material::pbr().base_color(0.8, 0.2, 0.2).roughness(0.3).metallic(0.1)),
        );
        scene.add(
            Sphere::default()
                .radius(1.5).at(2.5, 0.0, 0.0)
                .material(Material::pbr().base_color(0.2, 0.6, 0.8).roughness(0.7).metallic(0.8)),
        );
        scene.add(
            Cube::default()
                .width(2.0).height(1.0).depth(2.0).at(0.0, -2.5, 0.0)
                .material(Material::pbr().base_color(0.3, 0.7, 0.3).roughness(0.4)),
        );

        engine.load_scene(scene.build());
    });
}
