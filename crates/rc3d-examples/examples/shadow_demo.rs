//! Shadow type demo — directional CSM + point light + spot light shadows.
//!
//! Usage: cargo run -p rc3d-examples --example shadow_demo

use rc3d_core::{math::Vec3, DisplayMode};
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Shadow Demo", |engine| {
        engine.set_display_mode(DisplayMode::Shaded);

        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        // Camera
        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(4.0, 6.0, 8.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        // Directional light (casts CSM shadow)
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity: 1.2,
            light_group: None,
        }));

        // Point light (casts cubemap shadow)
        let pt_light_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(pt_light_sep, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(3.0, 4.0, 0.0),
        )));
        graph.add_child(pt_light_sep, NodeData::PointLight(PointLightNode {
            location: Vec3::ZERO,
            color: Vec3::new(0.8, 0.5, 0.3),
            intensity: 30.0,
            light_group: None,
        }));
        graph.add_child(pt_light_sep, NodeData::Material(MaterialNode {
            base_color: Vec3::new(1.0, 0.5, 0.2),
            roughness: 0.1,
            ..Default::default()
        }));
        graph.add_child(pt_light_sep, NodeData::Sphere(SphereNode { radius: 0.15 }));

        // Spot light (casts perspective shadow)
        let sp_light_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sp_light_sep, NodeData::Transform(TransformNode {
            translation: Vec3::new(-3.0, 5.0, 3.0),
            rotation: rc3d_core::math::Mat4::look_at_rh(
                Vec3::new(-3.0, 5.0, 3.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::Y,
            ),
            ..Default::default()
        }));
        graph.add_child(sp_light_sep, NodeData::SpotLight(SpotLightNode {
            location: Vec3::ZERO,
            direction: Vec3::new(3.0, -5.0, -3.0).normalize(),
            color: Vec3::new(0.3, 0.7, 1.0),
            intensity: 40.0,
            light_group: None,
            cut_off_angle: 0.5,
            drop_off_rate: 4.0,
        }));

        // Floor
        let floor_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(floor_sep, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.8, 0.8, 0.8),
            roughness: 0.9,
            ..Default::default()
        }));
        graph.add_child(floor_sep, NodeData::Cube(CubeNode {
            width: 16.0, height: 0.2, depth: 16.0,
        }));

        // Central large sphere
        let center_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(center_sep, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(0.0, 1.5, 0.0),
        )));
        graph.add_child(center_sep, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.7, 0.6, 0.5),
            metallic: 0.2,
            roughness: 0.3,
            ..Default::default()
        }));
        graph.add_child(center_sep, NodeData::Sphere(SphereNode { radius: 1.2 }));

        // Small spheres around
        let offsets = [
            Vec3::new(2.5, 0.8, 1.5),
            Vec3::new(-2.0, 0.6, -1.0),
            Vec3::new(1.0, 0.5, -2.5),
            Vec3::new(-1.5, 0.9, 2.0),
        ];
        for &offset in &offsets {
            let s = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(s, NodeData::Transform(TransformNode::from_translation(offset)));
            graph.add_child(s, NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.5, 0.7, 0.6),
                roughness: 0.5,
                ..Default::default()
            }));
            graph.add_child(s, NodeData::Sphere(SphereNode { radius: 0.5 }));
        }
    });
}
