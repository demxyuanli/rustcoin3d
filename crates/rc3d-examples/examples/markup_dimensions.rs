//! Markup Dimensions demo — 3D annotations on Cube/Sphere/Cylinder.
//!
//! Usage: cargo run -p rc3d-examples --example markup_dimensions

use rc3d_core::math::Vec3;
use rc3d_engine_api::CameraController;
use rc3d_examples::common::run_example;
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

        // ══  Cube (left, orange annotations)  ══
        let cube_x = -3.5;
        let cube_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(cube_sep, NodeData::Transform(TransformNode::from_translation(Vec3::new(cube_x, 0.0, 0.0))));
        let hw = 0.75; // half-width for 1.5 cube
        graph.add_child(cube_sep, NodeData::Material(MaterialNode::from_diffuse(Vec3::new(1.0, 0.6, 0.2))));
        graph.add_child(cube_sep, NodeData::Cube(CubeNode { width: 1.5, height: 1.5, depth: 1.5 }));

        // ══  Sphere (center, green annotations)  ══
        let sphere_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sphere_sep, NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 0.5, 0.0))));
        graph.add_child(sphere_sep, NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.8, 0.3))));
        graph.add_child(sphere_sep, NodeData::Sphere(SphereNode { radius: 1.0 }));

        // ══  Cylinder (right, blue annotations)  ══
        let cyl_x = 3.5;
        let cyl_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(cyl_sep, NodeData::Transform(TransformNode::from_translation(Vec3::new(cyl_x, -0.3, 0.0))));
        let cr = 0.8;
        let ch = 1.0;
        graph.add_child(cyl_sep, NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.5, 1.0))));
        graph.add_child(cyl_sep, NodeData::Cylinder(CylinderNode { radius: cr, height: 2.0 }));

        // ══  3D Annotations (inside Annotation grouping node → overlay rendering)  ══
        let ann_group = graph.add_child(root, NodeData::Annotation(AnnotationNode));

        // Cube annotations (placed relative to cube's local space)
        let cube_ann = graph.add_child(ann_group, NodeData::Separator(SeparatorNode));
        graph.add_child(cube_ann, NodeData::Transform(TransformNode::from_translation(Vec3::new(cube_x, 0.0, 0.0))));
        graph.add_child(cube_ann, NodeData::AnnotationSet(AnnotationSetNode {
            elements: vec![
                AnnotationElement::Dimension {
                    start: [-hw, -hw, 0.0],
                    end: [hw, -hw, 0.0],
                    offset_dir: [0.0, -1.0, 0.0],
                    extension_len: 0.3,
                    arrow_size: 0.15,
                    label: "W=1.5".into(),
                    color: [1.0, 0.4, 0.0, 1.0],
                },
                AnnotationElement::Dimension {
                    start: [-hw, -hw, -hw],
                    end: [-hw, hw, -hw],
                    offset_dir: [-1.0, 0.0, 0.0],
                    extension_len: 0.3,
                    arrow_size: 0.15,
                    label: "H=1.5".into(),
                    color: [1.0, 0.4, 0.0, 1.0],
                },
                AnnotationElement::Leader {
                    anchor: [0.0, 0.0, -hw],
                    label_offset: [40.0, -30.0],
                    text: "Cube".into(),
                    color: [0.9, 0.4, 0.0, 1.0],
                },
                AnnotationElement::Datum {
                    position: [-hw, -hw, -hw],
                    size: 0.12,
                    color: [1.0, 0.6, 0.0, 1.0],
                },
            ],
            visible: true,
        }));

        // Sphere annotations
        let sphere_ann = graph.add_child(ann_group, NodeData::Separator(SeparatorNode));
        graph.add_child(sphere_ann, NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 0.5, 0.0))));
        graph.add_child(sphere_ann, NodeData::AnnotationSet(AnnotationSetNode {
            elements: vec![
                AnnotationElement::Dimension {
                    start: [0.0, -1.0, 0.0],
                    end: [0.0, 1.0, 0.0],
                    offset_dir: [-1.0, 0.0, 0.0],
                    extension_len: 0.25,
                    arrow_size: 0.15,
                    label: "D=2.0".into(),
                    color: [0.0, 0.7, 0.3, 1.0],
                },
                AnnotationElement::Leader {
                    anchor: [1.0, 0.0, 0.0],
                    label_offset: [50.0, -20.0],
                    text: "Sphere R=1.0".into(),
                    color: [0.0, 0.7, 0.3, 1.0],
                },
                AnnotationElement::Datum {
                    position: [0.0, -1.0, 0.0],
                    size: 0.1,
                    color: [0.0, 0.9, 0.3, 1.0],
                },
            ],
            visible: true,
        }));

        // Cylinder annotations
        let cyl_ann = graph.add_child(ann_group, NodeData::Separator(SeparatorNode));
        graph.add_child(cyl_ann, NodeData::Transform(TransformNode::from_translation(Vec3::new(cyl_x, -0.3, 0.0))));
        graph.add_child(cyl_ann, NodeData::AnnotationSet(AnnotationSetNode {
            elements: vec![
                AnnotationElement::Dimension {
                    start: [cr, -ch, 0.0],
                    end: [cr, ch, 0.0],
                    offset_dir: [1.0, 0.0, 0.0],
                    extension_len: 0.25,
                    arrow_size: 0.14,
                    label: "H=2.0".into(),
                    color: [0.1, 0.5, 1.0, 1.0],
                },
                AnnotationElement::Leader {
                    anchor: [0.0, ch, 0.0],
                    label_offset: [40.0, -30.0],
                    text: "Cyl. R=0.8".into(),
                    color: [0.1, 0.5, 1.0, 1.0],
                },
                AnnotationElement::Datum {
                    position: [0.0, ch, 0.0],
                    size: 0.1,
                    color: [0.2, 0.6, 1.0, 1.0],
                },
            ],
            visible: true,
        }));

        let orbit = CameraController::new(Vec3::new(0.0, 0.0, 0.0), 12.0);
        engine.controller = orbit;
    });
}
