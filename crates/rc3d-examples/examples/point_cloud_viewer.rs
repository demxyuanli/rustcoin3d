//! PointCloud + GPU particles, ShaderMaterial, and transmission demo.
//!
//! Usage: cargo run -p rc3d-examples --example point_cloud_viewer

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;
use rc3d_scene::ParticleEmitter;

fn main() {
    run_example("Point Cloud", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(0.0, 2.0, 10.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        graph.add_child(
            root,
            NodeData::PointCloud(PointCloudNode::with_emitter(ParticleEmitter::fountain())),
        );

        let glass = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            glass,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(1.6, 0.0, 0.0))),
        );
        graph.add_child(glass, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.85, 0.95, 1.0),
            roughness: 0.04,
            metallic: 0.0,
            transmission_factor: 1.0,
            ior: 1.5,
            opacity: 1.0,
            alpha_mode: AlphaMode::Blend,
            ..Default::default()
        }));
        graph.add_child(glass, NodeData::Sphere(SphereNode { radius: 0.45 }));

        let custom = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            custom,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(-1.6, 0.0, 0.0))),
        );
        graph.add_child(custom, NodeData::Material(MaterialNode {
            custom_wgsl: Some(
                "fn material_fs(in: VertexOutput, u: PerDrawUniforms) -> vec4<f32> {\n    let n = normalize(in.world_normal) * 0.5 + 0.5;\n    return vec4<f32>(n * u.base_color.xyz, 1.0);\n}\n"
                    .into(),
            ),
            base_color: Vec3::new(1.0, 0.85, 0.4),
            ..Default::default()
        }));
        graph.add_child(custom, NodeData::Cube(CubeNode { width: 0.7, height: 0.7, depth: 0.7 }));
    });
}
