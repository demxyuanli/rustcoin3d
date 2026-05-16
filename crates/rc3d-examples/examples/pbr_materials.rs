//! PBR material showcase — grid of spheres with metallic (rows) x roughness (cols) variation.
//!
//! Usage: cargo run -p rc3d-examples --example pbr_materials

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

const GRID_SIZE: usize = 7;
const SPACING: f32 = 2.5;

fn main() {
    run_example("PBR Materials", |engine| {
        engine.set_display_mode(rc3d_core::DisplayMode::Shaded);

        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        // Camera
        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(-2.0, 8.0, 14.0),
                Vec3::new(GRID_SIZE as f32 * SPACING * 0.5, 0.5, GRID_SIZE as f32 * SPACING * 0.5),
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        // Key light
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity: 1.5,
            light_group: None,
        }));
        // Fill light
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(0.7, -0.3, 0.5).normalize(),
            color: Vec3::new(0.6, 0.7, 1.0),
            intensity: 0.6,
            light_group: None,
        }));
        // Point light
        graph.add_child(root, NodeData::PointLight(PointLightNode {
            location: Vec3::new(GRID_SIZE as f32 * SPACING * 0.3, 3.0, GRID_SIZE as f32 * SPACING * 0.3),
            color: Vec3::new(1.0, 0.9, 0.8),
            intensity: 20.0,
            light_group: None,
        }));

        // Floor
        let floor_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        let floor_size = GRID_SIZE as f32 * SPACING * 1.5;
        graph.add_child(floor_sep, NodeData::Transform(TransformNode {
            translation: Vec3::new(floor_size * 0.5, -0.5, floor_size * 0.5),
            scale: Vec3::new(floor_size * 0.5, 0.1, floor_size * 0.5),
            ..Default::default()
        }));
        graph.add_child(floor_sep, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.5, 0.5, 0.5),
            roughness: 0.9,
            ..Default::default()
        }));
        graph.add_child(floor_sep, NodeData::Cube(CubeNode { width: 2.0, height: 1.0, depth: 2.0 }));

        // Sphere grid
        for row in 0..GRID_SIZE {
            let roughness = row as f32 / (GRID_SIZE - 1) as f32;
            for col in 0..GRID_SIZE {
                let metallic = col as f32 / (GRID_SIZE - 1) as f32;
                let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
                graph.add_child(
                    sep,
                    NodeData::Transform(TransformNode::from_translation(Vec3::new(
                        col as f32 * SPACING, 1.2, row as f32 * SPACING,
                    ))),
                );
                let hue = 0.12 + metallic * 0.05;
                let saturation = 0.3 + roughness * 0.4;
                let base = hsl_to_rgb(hue, saturation, 0.7);
                graph.add_child(sep, NodeData::Material(MaterialNode {
                    base_color: base,
                    metallic,
                    roughness,
                    ..Default::default()
                }));
                graph.add_child(sep, NodeData::Sphere(SphereNode { radius: 1.0 }));
            }
        }
    });
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Vec3 {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c * 0.5;
    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0), 1 => (x, c, 0.0), 2 => (0.0, c, x),
        3 => (0.0, x, c), 4 => (x, 0.0, c), _ => (c, 0.0, x),
    };
    Vec3::new(r + m, g + m, b + m)
}
