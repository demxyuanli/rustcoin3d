//! WBOIT (Weighted Blended Order-Independent Transparency) demo.
//!
//! Demonstrates overlapping transparent objects rendered with OIT.
//! Without WBOIT, the painter's algorithm produces incorrect blending when
//! transparent objects overlap; WBOIT resolves this by accumulating weighted
//! premultiplied colors and compositing the result.
//!
//! The scene shows multiple semi-transparent shapes (spheres, cubes, tori,
//! cylinders) at varying opacities, with an opaque ground plane behind them
//! to make the blending differences visible.
//!
//! Usage: cargo run -p rc3d-examples --example wboit_demo

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("WBOIT Demo", |engine| {
        // Enable HDR post-processing (required for WBOIT) and WBOIT itself
        if let Some(ref mut r) = engine.renderer {
            r.hdr_post_processing = true;
            r.enable_wboit = true;
        }

        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        // Camera — pulled back to see all spread-out objects
        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(0.0, 8.0, 12.0),
                Vec3::new(0.0, 1.5, 2.0),
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

        // Opaque ground plane (serves as backdrop for transparency)
        let ground_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(ground_sep, NodeData::Transform(TransformNode {
            translation: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        }));
        graph.add_child(ground_sep, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.45, 0.45, 0.48),
            roughness: 0.85,
            ..Default::default()
        }));
        graph.add_child(ground_sep, NodeData::Cube(CubeNode {
            width: 14.0, height: 0.1, depth: 14.0,
        }));

        // ── Row 1: Red transparent sphere (opacity 0.5) ──
        let sep1 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep1, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(-4.0, 1.0, 0.0),
        )));
        graph.add_child(sep1, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.9, 0.15, 0.1),
            metallic: 0.0,
            roughness: 0.3,
            opacity: 0.5,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            ..Default::default()
        }));
        graph.add_child(sep1, NodeData::Sphere(SphereNode { radius: 1.2 }));

        // ── Row 1: Blue transparent cube (opacity 0.4) ──
        let sep2 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep2, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(0.0, 1.0, 0.0),
        )));
        graph.add_child(sep2, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.1, 0.3, 0.9),
            metallic: 0.1,
            roughness: 0.2,
            opacity: 0.4,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            ..Default::default()
        }));
        graph.add_child(sep2, NodeData::Cube(CubeNode {
            width: 2.0, height: 2.0, depth: 2.0,
        }));

        // ── Row 1: Green transparent torus (opacity 0.6) ──
        let sep3 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep3, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(4.0, 1.0, 0.0),
        )));
        graph.add_child(sep3, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.1, 0.8, 0.2),
            metallic: 0.2,
            roughness: 0.4,
            opacity: 0.6,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            ..Default::default()
        }));
        graph.add_child(sep3, NodeData::Torus(TorusNode {
            major_radius: 1.2,
            minor_radius: 0.35,
        }));

        // ── Row 2: Yellow transparent cylinder (opacity 0.35) ──
        let sep4 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep4, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(-4.0, 1.0, 4.0),
        )));
        graph.add_child(sep4, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.95, 0.85, 0.1),
            metallic: 0.3,
            roughness: 0.25,
            opacity: 0.35,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            ..Default::default()
        }));
        graph.add_child(sep4, NodeData::Cylinder(CylinderNode {
            radius: 0.7,
            height: 2.5,
        }));

        // ── Row 2: Purple transparent sphere (opacity 0.3) ──
        let sep5 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep5, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(0.0, 1.0, 4.0),
        )));
        graph.add_child(sep5, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.7, 0.15, 0.85),
            metallic: 0.5,
            roughness: 0.15,
            opacity: 0.3,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            ..Default::default()
        }));
        graph.add_child(sep5, NodeData::Sphere(SphereNode { radius: 1.0 }));

        // ── Row 2: Cyan transparent cone (opacity 0.45) ──
        let sep6 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep6, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(4.0, 1.0, 4.0),
        )));
        graph.add_child(sep6, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.1, 0.85, 0.8),
            metallic: 0.0,
            roughness: 0.5,
            opacity: 0.45,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            ..Default::default()
        }));
        graph.add_child(sep6, NodeData::Cone(ConeNode {
            bottom_radius: 1.0,
            height: 2.0,
        }));

        // Picking support
        graph.add_child(root, NodeData::EventCallback(EventCallbackNode::default()));
    });
}
