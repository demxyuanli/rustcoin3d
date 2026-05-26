//! 3D annotation demo — annotations live under each body's `Transform` (same as mesh).
//!
//! Usage: cargo run -p rc3d-examples --example markup_dimensions

use rc3d_core::math::Vec3;
use rc3d_engine::{EngineRegistry, InterpolateVec3Engine};
use rc3d_engine_api::CameraController;
use rc3d_examples::common::run_example;
use rc3d_scene::annotation::{AnnotationLabelMode, AnnotationPoint, AnnotationStyle};
use rc3d_scene::node_data::*;

fn main() {
    run_example("Markup Dimensions", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(4.0, 3.0, 12.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        let style = AnnotationStyle {
            unit_suffix: " mm".into(),
            ..AnnotationStyle::default()
        };

        let cube_x = -3.5;
        let cube_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        let cube_tf = graph.add_child(
            cube_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(cube_x, 0.0, 0.0))),
        );
        let hw = 0.75;
        graph.add_child(
            cube_tf,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(1.0, 0.6, 0.2))),
        );
        graph.add_child(cube_tf, NodeData::Cube(CubeNode { width: 1.5, height: 1.5, depth: 1.5 }));
        graph.add_child(
            cube_tf,
            NodeData::AnnotationSet(AnnotationSetNode {
                style: style.clone(),
                elements: vec![
                    AnnotationElement::linear_auto(
                        [-hw, -hw, 0.0],
                        [hw, -hw, 0.0],
                        [0.0, -1.0, 0.0],
                        [1.0, 0.4, 0.0, 1.0],
                    ),
                    AnnotationElement::Dimension {
                        start: [-hw, -hw, -hw].into(),
                        end: [-hw, hw, -hw].into(),
                        offset_dir: [-1.0, 0.0, 0.0],
                        extension_len: 0.3,
                        arrow_size: 0.15,
                        label: "H=".into(),
                        label_mode: AnnotationLabelMode::Prefix,
                        color: [1.0, 0.4, 0.0, 1.0],
                    },
                    AnnotationElement::leader([0.0, 0.0, -hw], [40.0, -30.0], "立方体", [0.9, 0.4, 0.0, 1.0]),
                    AnnotationElement::Datum {
                        position: [-hw, -hw, -hw].into(),
                        size: 0.12,
                        color: [1.0, 0.6, 0.0, 1.0],
                    },
                ],
                visible: true,
            }),
        );

        // Node-bound annotation: follows the animated cube
        graph.add_child(
            cube_tf,
            NodeData::AnnotationSet(AnnotationSetNode {
                style: AnnotationStyle {
                    unit_suffix: " mm".into(),
                    ..AnnotationStyle::default()
                },
                elements: vec![AnnotationElement::Dimension {
                    start: AnnotationPoint::on_node(cube_tf, [hw, 0.0, hw]),
                    end: AnnotationPoint::on_node(cube_tf, [hw, 0.0, -hw]),
                    offset_dir: [1.0, 0.0, 0.0],
                    extension_len: 0.2,
                    arrow_size: 0.12,
                    label: "深=1.5".into(),
                    label_mode: AnnotationLabelMode::Prefix,
                    color: [1.0, 0.8, 0.0, 1.0],
                }],
                visible: true,
            }),
        );

        let sphere_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        let sphere_tf = graph.add_child(
            sphere_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 0.5, 0.0))),
        );
        graph.add_child(
            sphere_tf,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.8, 0.3))),
        );
        graph.add_child(sphere_tf, NodeData::Sphere(SphereNode { radius: 1.0 }));
        graph.add_child(
            sphere_tf,
            NodeData::AnnotationSet(AnnotationSetNode {
                style: style.clone(),
                elements: vec![
                    AnnotationElement::Dimension {
                        start: [0.0, -1.0, 0.0].into(),
                        end: [0.0, 1.0, 0.0].into(),
                        offset_dir: [-1.2, 0.0, 0.0],
                        extension_len: 0.25,
                        arrow_size: 0.15,
                        label: String::new(),
                        label_mode: AnnotationLabelMode::Auto,
                        color: [0.0, 0.7, 0.3, 1.0],
                    },
                    AnnotationElement::RadialDimension {
                        center: [0.0, 0.0, 0.0].into(),
                        perimeter: [1.0, 0.0, 0.0].into(),
                        label: String::new(),
                        label_mode: AnnotationLabelMode::Auto,
                        arrow_size: 0.12,
                        color: [0.0, 0.7, 0.3, 1.0],
                    },
                    AnnotationElement::AngleDimension {
                        center: [0.0, 0.0, 0.0].into(),
                        arm1: [1.0, 0.0, 0.0].into(),
                        arm2: [0.0, 1.0, 0.0].into(),
                        radius: 0.55,
                        label: String::new(),
                        label_mode: AnnotationLabelMode::Auto,
                        color: [0.0, 0.8, 0.4, 1.0],
                    },
                    AnnotationElement::Callout {
                        anchor: [0.0, 1.0, 0.0].into(),
                        label_offset: [50.0, -20.0],
                        text: "球体".into(),
                        radius: 18.0,
                        color: [0.0, 0.7, 0.3, 1.0],
                    },
                    AnnotationElement::leader([0.0, 0.0, 1.0], [-40.0, -20.0], "球体", [0.0, 0.7, 0.3, 1.0]),
                ],
                visible: true,
            }),
        );

        let cyl_x = 3.5;
        let cyl_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        let cyl_tf = graph.add_child(
            cyl_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(cyl_x, -0.3, 0.0))),
        );
        let cr = 0.8;
        let ch = 1.0;
        graph.add_child(
            cyl_tf,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.5, 1.0))),
        );
        graph.add_child(cyl_tf, NodeData::Cylinder(CylinderNode { radius: cr, height: 2.0 }));
        graph.add_child(
            cyl_tf,
            NodeData::AnnotationSet(AnnotationSetNode {
                style,
                elements: vec![
                    AnnotationElement::linear_auto(
                        [cr, -ch, 0.0],
                        [cr, ch, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.1, 0.5, 1.0, 1.0],
                    ),
                    AnnotationElement::DiameterDimension {
                        center: [0.0, 0.0, 0.0].into(),
                        p1: [cr, 0.0, 0.0].into(),
                        p2: [-cr, 0.0, 0.0].into(),
                        label: String::new(),
                        label_mode: AnnotationLabelMode::Auto,
                        arrow_size: 0.14,
                        color: [0.1, 0.5, 1.0, 1.0],
                    },
                    AnnotationElement::leader([0.0, ch, 0.0], [40.0, -30.0], "圆柱体", [0.1, 0.5, 1.0, 1.0]),
                    AnnotationElement::Datum {
                        position: [0.0, ch, 0.0].into(),
                        size: 0.1,
                        color: [0.2, 0.6, 1.0, 1.0],
                    },
                ],
                visible: true,
            }),
        );

        let mut engines = EngineRegistry::new();
        engines.add(InterpolateVec3Engine {
            transform_node: cube_tf,
            from: Vec3::new(cube_x, -0.35, 0.0),
            to: Vec3::new(cube_x, 0.35, 0.0),
            period_secs: 2.5,
        });
        engine.world_mut().engines = Some(engines);

        let orbit = CameraController::new(Vec3::new(0.0, 0.0, 0.0), 12.0);
        engine.controller = orbit;
    });
}
