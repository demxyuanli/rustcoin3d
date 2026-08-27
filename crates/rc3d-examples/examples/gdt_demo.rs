//! GD&T annotation demo: Feature Control Frames, Datum Targets, Chamfer & Ordinate dims.
//!
//! PMI records on the AnnotationSet bind visuals to the named `housing` part
//! (face ids + tolerances). `SceneGraph::bind_pmi` resolves names to NodeId.
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
        let bound = {
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

            let housing = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.set_name(housing, "housing");
            graph.add_child(
                housing,
                NodeData::Material(MaterialNode {
                    base_color: Vec3::new(0.5, 0.5, 0.6),
                    roughness: 0.5,
                    ..Default::default()
                }),
            );
            graph.add_child(
                housing,
                NodeData::Cube(CubeNode {
                    width: 2.0,
                    height: 2.0,
                    depth: 2.0,
                }),
            );

            let style = AnnotationStyle {
                unit_suffix: " mm".into(),
                ..AnnotationStyle::default()
            };

            graph.add_child(
                root,
                NodeData::AnnotationSet(AnnotationSetNode {
                    style,
                    elements: vec![
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
                        AnnotationElement::GdtDatumTarget {
                            position: [-1.0, -1.1, 0.0].into(),
                            label: "A1".into(),
                            target_type: DatumTargetType::Point,
                            size: 0.15,
                            color: [0.3, 0.3, 1.0, 1.0],
                        },
                        AnnotationElement::GdtDatumTarget {
                            position: [0.0, 1.15, 0.5].into(),
                            label: "B1".into(),
                            target_type: DatumTargetType::Line,
                            size: 0.4,
                            color: [0.3, 0.6, 1.0, 1.0],
                        },
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
                    pmi: vec![
                        PmiRecord {
                            id: "flatness-top".into(),
                            kind: PmiKind::Gdt,
                            element_index: 0,
                            bindings: vec![PmiBinding::named("housing").with_face(0)],
                            nominal: None,
                            upper_tol: Some(0.05),
                            lower_tol: None,
                            unit: "mm".into(),
                            datum_refs: vec![],
                            source: PmiSource::Manual,
                        },
                        PmiRecord {
                            id: "position-mmc-a".into(),
                            kind: PmiKind::Gdt,
                            element_index: 1,
                            bindings: vec![PmiBinding::named("housing").with_face(1)],
                            nominal: None,
                            upper_tol: Some(0.02),
                            lower_tol: None,
                            unit: "mm".into(),
                            datum_refs: vec!["A".into()],
                            source: PmiSource::Manual,
                        },
                        PmiRecord {
                            id: "datum-a1".into(),
                            kind: PmiKind::Datum,
                            element_index: 2,
                            bindings: vec![PmiBinding::named("housing").with_face(2)],
                            ..PmiRecord::default()
                        },
                        PmiRecord {
                            id: "chamfer-front".into(),
                            kind: PmiKind::LinearDimension,
                            element_index: 4,
                            bindings: vec![PmiBinding::named("housing").with_face(3)],
                            nominal: Some(2.0),
                            unit: "mm".into(),
                            source: PmiSource::Manual,
                            ..PmiRecord::default()
                        },
                    ],
                }),
            );

            graph.bind_pmi()
        };
        engine.hud_text_hook = Some(Box::new(move || format!("PMI bindings: {bound}")));
    });
}
