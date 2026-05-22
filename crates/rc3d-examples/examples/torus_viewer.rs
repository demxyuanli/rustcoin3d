//! Torus viewer — showcases the TorusNode primitive with various sizes and materials.
//!
//! Demonstrates:
//!   - TorusNode with different major/minor radii
//!   - PBR materials (metallic, roughness, color)
//!   - Ray-picking support for torus geometry
//!
//! Usage: cargo run -p rc3d-examples --example torus_viewer

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Torus Viewer", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        // Camera
        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(5.0, 4.0, 8.0),
                Vec3::ZERO,
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

        // Ground plane
        let ground_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(ground_sep, NodeData::Transform(TransformNode {
            translation: Vec3::new(0.0, -1.5, 0.0),
            ..Default::default()
        }));
        graph.add_child(ground_sep, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.4, 0.4, 0.42),
            roughness: 0.9,
            ..Default::default()
        }));
        graph.add_child(ground_sep, NodeData::Cube(CubeNode {
            width: 12.0, height: 0.1, depth: 12.0,
        }));

        // Row 1: Default torus (major_radius=1.0, minor_radius=0.4)
        let sep1 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep1, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(-3.0, 0.0, 0.0),
        )));
        graph.add_child(sep1, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.9, 0.3, 0.2),
            metallic: 0.1,
            roughness: 0.4,
            ..Default::default()
        }));
        graph.add_child(sep1, NodeData::Torus(TorusNode {
            major_radius: 1.0,
            minor_radius: 0.4,
        }));

        // Row 1: Thick ring (minor_radius close to major_radius)
        let sep2 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep2, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(0.0, 0.0, 0.0),
        )));
        graph.add_child(sep2, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.2, 0.5, 0.9),
            metallic: 0.8,
            roughness: 0.15,
            ..Default::default()
        }));
        graph.add_child(sep2, NodeData::Torus(TorusNode {
            major_radius: 1.0,
            minor_radius: 0.7,
        }));

        // Row 1: Thin ring (small minor_radius)
        let sep3 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep3, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(3.0, 0.0, 0.0),
        )));
        graph.add_child(sep3, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.1, 0.8, 0.3),
            metallic: 0.3,
            roughness: 0.3,
            ..Default::default()
        }));
        graph.add_child(sep3, NodeData::Torus(TorusNode {
            major_radius: 1.0,
            minor_radius: 0.15,
        }));

        // Row 2: Large torus (scaled up)
        let sep4 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep4, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(-3.0, 0.0, 3.5),
        )));
        graph.add_child(sep4, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.9, 0.8, 0.1),
            metallic: 0.6,
            roughness: 0.2,
            ..Default::default()
        }));
        graph.add_child(sep4, NodeData::Torus(TorusNode {
            major_radius: 1.5,
            minor_radius: 0.5,
        }));

        // Row 2: Golden torus — highly metallic
        let sep5 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep5, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(0.0, 0.0, 3.5),
        )));
        graph.add_child(sep5, NodeData::Material(MaterialNode {
            base_color: Vec3::new(1.0, 0.76, 0.34),
            metallic: 1.0,
            roughness: 0.05,
            ..Default::default()
        }));
        graph.add_child(sep5, NodeData::Torus(TorusNode {
            major_radius: 0.8,
            minor_radius: 0.3,
        }));

        // Row 2: Tilted torus via transform
        let sep6 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep6, NodeData::Transform(TransformNode {
            translation: Vec3::new(3.0, 0.0, 3.5),
            rotation: rc3d_core::math::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_3),
            ..Default::default()
        }));
        graph.add_child(sep6, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.7, 0.2, 0.8),
            metallic: 0.4,
            roughness: 0.5,
            ..Default::default()
        }));
        graph.add_child(sep6, NodeData::Torus(TorusNode {
            major_radius: 1.0,
            minor_radius: 0.35,
        }));

        // Enable picking for all shapes
        graph.add_child(root, NodeData::EventCallback(EventCallbackNode::default()));
    });
}
