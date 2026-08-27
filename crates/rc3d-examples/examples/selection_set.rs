//! SelectionSet + HOOPS Isolate/Ghost demo.
//!
//! Named selection groups for batch operations. Ghost mode keeps the selected
//! sphere shaded and fades everything else (other spheres + backing plate).
//! Each shape lives in its own Separator so sibling Transforms do not accumulate.
//!
//! Usage: cargo run -p rc3d-examples --example selection_set
//! Screenshot: --screenshot --screenshot-path target/selection_ghost.png --screenshot-exit

use rc3d_core::math::Vec3;
use rc3d_core::{DisplayMode, NodeId};
use rc3d_engine_api::background::BackgroundSettings;
use rc3d_examples::common::run_example;
use rc3d_render::background::BgMode;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

fn add_part(
    graph: &mut SceneGraph,
    root: NodeId,
    material: MaterialNode,
    translation: Vec3,
    shape: NodeData,
) -> NodeId {
    let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(sep, NodeData::Material(material));
    graph.add_child(sep, NodeData::Transform(TransformNode::from_translation(translation)));
    graph.add_child(sep, shape)
}

fn main() {
    run_example("Selection Set", |engine| {
        engine.set_display_mode(DisplayMode::ShadedWithEdges);
        engine.set_ghost_unselected(true);
        engine.set_ghost_opacity(0.45);
        engine.set_background(BackgroundSettings {
            mode: BgMode::VerticalGradient,
            clear_color: [0.78, 0.80, 0.84, 1.0],
            top_color: [0.82, 0.84, 0.88, 1.0],
            bot_color: [0.62, 0.64, 0.68, 1.0],
            ..Default::default()
        });

        engine.camera_mut().target = Vec3::ZERO;
        engine.camera_mut().distance = 11.0;
        engine.camera_mut().yaw = 0.35;
        engine.camera_mut().pitch = 0.28;

        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(0.0, 2.2, 11.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        graph.add_child(
            root,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-0.4, -1.0, -0.6).normalize(),
                color: Vec3::ONE,
                intensity: 1.15,
                light_group: None,
            }),
        );

        add_part(
            graph,
            root,
            MaterialNode {
                base_color: Vec3::new(0.55, 0.58, 0.62),
                roughness: 0.7,
                metallic: 0.05,
                ..Default::default()
            },
            Vec3::new(0.0, 0.0, -1.35),
            NodeData::Cube(CubeNode {
                width: 7.2,
                height: 2.6,
                depth: 0.1,
            }),
        );

        let mut ids = Vec::new();
        for i in 0..5i32 {
            let focused = i == 2;
            ids.push(add_part(
                graph,
                root,
                MaterialNode {
                    base_color: if focused {
                        Vec3::new(0.92, 0.42, 0.08)
                    } else {
                        Vec3::new(0.3 + i as f32 * 0.1, 0.22, 0.72)
                    },
                    roughness: if focused { 0.35 } else { 0.3 },
                    ..Default::default()
                },
                Vec3::new(i as f32 * 1.5 - 3.0, 0.0, 0.0),
                NodeData::Sphere(SphereNode { radius: 0.4 }),
            ));
        }

        graph.selection_set_add("spheres", &ids);
        graph.selection_set_add("focus", &[ids[2]]);
        graph.selection_set_select("focus");

        eprintln!(
            "HOOPS ghost: middle sphere shaded; other spheres + plate translucent"
        );
    });
}
