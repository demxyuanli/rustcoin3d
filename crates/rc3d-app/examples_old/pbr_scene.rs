//! PBR material showcase using the new high-level API.
//!
//! 7×7 sphere grid: rows = roughness, columns = metallic.
//!
//! Usage: cargo run -p rc3d-app --example pbr_scene

use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_scene_api::{Cube, DirectionalLight, Material, PerspectiveCamera, Scene, Sphere};

const GRID_SIZE: usize = 7;
const SPACING: f32 = 2.5;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d=info"),
    )
    .init();
    println!("=== PBR Material Showcase (new API) ===");

    let scene = build_pbr_scene();

    let controller = CameraController::new(
        Vec3::new(GRID_SIZE as f32 * SPACING * 0.5, 2.0, GRID_SIZE as f32 * SPACING * 0.8),
        GRID_SIZE as f32 * SPACING * 1.2,
    );

    let mut app = App::from_scene(scene).with_camera_controller(controller);

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn build_pbr_scene() -> Scene {
    let mut scene = Scene::new();

    // Camera
    scene.set_camera(PerspectiveCamera::look_at(
        Vec3::new(-2.0, 8.0, 14.0),
        Vec3::new(GRID_SIZE as f32 * SPACING * 0.5, 0.5, GRID_SIZE as f32 * SPACING * 0.5),
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        800.0 / 600.0,
    ));

    // Key light
    scene.add_light(DirectionalLight::sun(
        Vec3::new(-0.5, -0.8, -0.3),
        1.5,
    ));
    // Fill light
    scene.add_light(
        DirectionalLight::sun(Vec3::new(0.7, -0.3, 0.5), 0.6)
            .color(0.6, 0.7, 1.0),
    );
    // Point light for highlights
    scene.add_point_light(
        rc3d_scene_api::PointLight::new(
            Vec3::new(GRID_SIZE as f32 * SPACING * 0.3, 3.0, GRID_SIZE as f32 * SPACING * 0.3),
            Vec3::new(1.0, 0.9, 0.8),
            20.0,
        ),
    );

    // Floor
    let floor_size = GRID_SIZE as f32 * SPACING * 1.5;
    scene.add(
        Cube::default()
            .width(floor_size)
            .height(0.2)
            .depth(floor_size)
            .at(floor_size * 0.5, -0.5, floor_size * 0.5)
            .material(Material::pbr().base_color(0.5, 0.5, 0.5).roughness(0.9)),
    );

    // Sphere grid
    for row in 0..GRID_SIZE {
        let roughness = row as f32 / (GRID_SIZE - 1) as f32;
        for col in 0..GRID_SIZE {
            let metallic = col as f32 / (GRID_SIZE - 1) as f32;
            let hue = 0.12 + metallic * 0.05;
            let saturation = 0.3 + roughness * 0.4;
            let base = hsl_to_rgb(hue, saturation, 0.7);

            scene.add(
                Sphere::default()
                    .at(col as f32 * SPACING, 1.2, row as f32 * SPACING)
                    .material(
                        Material::pbr()
                            .base_color(base.x, base.y, base.z)
                            .metallic(metallic)
                            .roughness(roughness),
                    ),
            );
        }
    }

    scene
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Vec3 {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c * 0.5;
    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Vec3::new(r + m, g + m, b + m)
}
