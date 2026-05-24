//! GD&T annotation demo: Feature Control Frames, Datum Targets, Chamfer & Ordinate dims.
//!
//! Usage: cargo run -p rc3d-examples --example gdt_demo
//!
//! Controls: Mouse orbit | Scroll zoom | Escape quit

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::annotation::{AnnotationLabelMode, AnnotationStyle};
use rc3d_scene::node_data::*;

fn main() {
    run_example("GD&T Demo — FCF, DatumTarget, Chamfer, Ordinate", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(4.0, 3.0, 8.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        graph.add_child(
            root,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
                color: Vec3::ONE,
                intensity: 1.0,
                light_group: None,
            }),
        );

        // Gray reference block
        graph.add_child(
            root,
            NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.5, 0.5, 0.6),
                roughness: 0.5,
                ..Default::default()
            }),
        );
        graph.add_child(root, NodeData::Cube(CubeNode { width: 2.0, height: 2.0, depth: 2.0 }));

        let style = AnnotationStyle {
            unit_suffix: " mm".into(),
            ..AnnotationStyle::default()
        };

        graph.add_child(
            root,
            NodeData::AnnotationSet(AnnotationSetNode {
                style,
                elements: vec![
                    // ── GD&T Feature Control Frame: flatness 0.05 on top face ──
                    AnnotationElement::GdtFeatureControlFrame {
                        symbol: GdtSymbol::Flatness,
                        tolerance: 0.05,
                        diameter: false,
                        datum_primary: None,
                        datum_secondary: None,
                        material_condition: None,
                        position: [0.0, 1.15, 0.0].into(),
                        leader_target: Some([0.0, 1.0, 0.0].into()),
                        color: [1.0, 0.3, 0.3, 1.0],
                    },
                    // ── FCF with datum reference: position Ø0.02 Ⓜ A ──
                    AnnotationElement::GdtFeatureControlFrame {
                        symbol: GdtSymbol::Position,
                        tolerance: 0.02,
                        diameter: true,
                        datum_primary: Some("A".into()),
                        datum_secondary: None,
                        material_condition: Some(GdtMaterialCondition::MaximumMaterial),
                        position: [1.3, -1.15, 0.0].into(),
                        leader_target: Some([1.0, -1.0, 0.0].into()),
                        color: [1.0, 0.3, 0.3, 1.0],
                    },
                    // ── Datum Target A1 ──
                    AnnotationElement::GdtDatumTarget {
                        position: [-1.0, -1.1, 0.0].into(),
                        label: "A1".into(),
                        target_type: DatumTargetType::Point,
                        size: 0.15,
                        color: [0.3, 0.3, 1.0, 1.0],
                    },
                    // ── Datum Target B1 (line) ──
                    AnnotationElement::GdtDatumTarget {
                        position: [0.0, 1.15, 0.5].into(),
                        label: "B1".into(),
                        target_type: DatumTargetType::Line,
                        size: 0.4,
                        color: [0.3, 0.6, 1.0, 1.0],
                    },
                    // ── Chamfer dimension ──
                    AnnotationElement::ChamferDimension {
                        start: [1.0, -1.0, -1.0].into(),
                        end: [1.0, -1.0, 1.0].into(),
                        offset_dir: [1.0, -1.0, 0.0],
                        extension_len: 0.25,
                        arrow_size: 0.12,
                        label: String::new(),
                        label_mode: AnnotationLabelMode::Auto,
                        color: [0.8, 0.5, 0.0, 1.0],
                    },
                    // ── Ordinate dimension ──
                    AnnotationElement::OrdinateDimension {
                        feature: [1.0, 1.0, -1.0].into(),
                        datum: [0.0, 1.0, -1.0].into(),
                        axis_dir: [1.0, 0.0, 0.0],
                        jog_length: 0.6,
                        offset: 0.5,
                        label: String::new(),
                        label_mode: AnnotationLabelMode::Auto,
                        color: [0.0, 0.7, 0.0, 1.0],
                    },
                ],
                visible: true,
            }),
        );
    });
}
