//! StereoCamera demo — dual-eye render with interocular distance.
//!
//! Side-by-side by default (left | right). Objects at different depths show parallax.
//!
//! Usage: cargo run -p rc3d-examples --example stereo_camera

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene_api::{
    Cube, DirectionalLight, Material, PerspectiveCamera, Scene, Sphere, StereoCamera, StereoMode,
};

fn main() {
    run_example("Stereo Camera", |engine| {
        let mut scene = Scene::new();

        scene.set_camera(PerspectiveCamera::look_at(
            Vec3::new(3.0, 2.0, 6.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
        ));
        scene.add_stereo_camera(
            StereoCamera::side_by_side()
                .interocular(0.12)
                .convergence(4.0)
                .mode(StereoMode::SideBySide),
        );
        scene.add_light(DirectionalLight::sun(Vec3::new(-1.0, -1.0, -1.0), 1.0));

        scene.add(
            Sphere::default()
                .radius(0.85)
                .material(Material::pbr().base_color(0.5, 0.3, 0.8).roughness(0.25)),
        );
        scene.add(
            Cube::default()
                .width(0.7)
                .height(0.7)
                .depth(0.7)
                .at(-1.6, 0.0, -1.2)
                .material(Material::pbr().base_color(0.9, 0.35, 0.2).roughness(0.35)),
        );
        scene.add(
            Cube::default()
                .width(0.55)
                .height(0.55)
                .depth(0.55)
                .at(1.7, 0.0, 1.4)
                .material(Material::pbr().base_color(0.25, 0.7, 0.45).roughness(0.4)),
        );

        engine.load_scene(scene.build());
    });
}
