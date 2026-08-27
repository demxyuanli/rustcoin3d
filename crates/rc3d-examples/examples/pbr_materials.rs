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
        graph.add_child(root, NodeData::HemisphereLight(HemisphereLightNode {
            sky_color: Vec3::new(0.45, 0.65, 1.0),
            ground_color: Vec3::new(0.35, 0.22, 0.12),
            intensity: 0.45,
            direction: Vec3::Y,
        }));
        graph.add_child(
            root,
            NodeData::LightProbe(LightProbeNode::from_hemisphere(
                Vec3::new(0.35, 0.5, 0.85),
                Vec3::new(0.2, 0.12, 0.06),
                0.25,
            )),
        );

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

        let iri_x = GRID_SIZE as f32 * SPACING + 3.0;
        let iridescent: [(Vec3, f32, f32, f32, f32, f32); 3] = [
            (Vec3::new(0.06, 0.06, 0.08), 0.15, 1.0, 1.3, 100.0, 400.0),
            (Vec3::new(0.92, 0.92, 0.94), 0.06, 0.0, 1.33, 200.0, 400.0),
            (Vec3::new(0.72, 0.45, 0.2), 0.22, 1.0, 1.5, 100.0, 300.0),
        ];
        for (i, (base, roughness, metallic, ior, tmin, tmax)) in iridescent.iter().copied().enumerate() {
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(
                    iri_x,
                    1.2,
                    i as f32 * SPACING,
                ))),
            );
            graph.add_child(sep, NodeData::Material(MaterialNode {
                base_color: base,
                metallic,
                roughness,
                iridescence_factor: 1.0,
                iridescence_ior: ior,
                iridescence_thickness_min: tmin,
                iridescence_thickness_max: tmax,
                ..Default::default()
            }));
            graph.add_child(sep, NodeData::Sphere(SphereNode { radius: 1.0 }));
        }

        let extra_x = GRID_SIZE as f32 * SPACING + 6.0;
        let extras = [
            (Vec3::new(0.85, 0.35, 0.15), 3.0, false, false),
            (Vec3::new(0.6, 0.6, 0.65), 0.0, true, false),
            (Vec3::new(0.5, 0.5, 0.5), 0.0, false, true),
        ];
        for (i, (base, toon, normals, depth)) in extras.iter().copied().enumerate() {
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(
                    extra_x,
                    1.2,
                    i as f32 * SPACING,
                ))),
            );
            graph.add_child(sep, NodeData::Material(MaterialNode {
                base_color: base,
                roughness: 0.4,
                toon_steps: toon,
                visualize_normals: normals,
                visualize_depth: depth,
                ..Default::default()
            }));
            graph.add_child(sep, NodeData::Sphere(SphereNode { radius: 1.0 }));
        }

        let rot_x = extra_x + 3.0;
        let rot_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            rot_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(rot_x, 1.2, 0.0))),
        );
        graph.add_child(
            rot_sep,
            NodeData::RotationXYZ(RotationXYZNode {
                axis: RotationAxis::Z,
                angle: 0.5,
            }),
        );
        graph.add_child(
            rot_sep,
            NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.2, 0.55, 0.85),
                roughness: 0.35,
                ..Default::default()
            }),
        );
        graph.add_child(
            rot_sep,
            NodeData::Cube(CubeNode {
                width: 1.2,
                height: 1.2,
                depth: 1.2,
            }),
        );
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
