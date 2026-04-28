use rc3d_app::{App, CameraController};
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();

    let mut graph = rc3d_scene::SceneGraph::new();

    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera — position controlled by CameraController
    let camera_id = graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(0.0, 3.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // Light
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -1.0, -0.5).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
        }),
    );

    // Shapes with their own material nodes (for picking highlight)
    // Red cube
    let sep1 = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        sep1,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(-2.5, 0.0, 0.0))),
    );
    let mat1 = graph.add_child(
        sep1,
        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.9, 0.2, 0.2))),
    );
    graph.add_child(sep1, NodeData::Cube(CubeNode::default()));

    // Green sphere
    let sep2 = graph.add_child(root, NodeData::Separator(SeparatorNode));
    let mat2 = graph.add_child(
        sep2,
        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.8, 0.2))),
    );
    graph.add_child(sep2, NodeData::Sphere(SphereNode { radius: 0.8 }));

    // Blue cone
    let sep3 = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        sep3,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(2.5, 0.0, 0.0))),
    );
    let mat3 = graph.add_child(
        sep3,
        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.3, 0.9))),
    );
    graph.add_child(
        sep3,
        NodeData::Cone(ConeNode {
            bottom_radius: 0.7,
            height: 1.5,
        }),
    );

    // Yellow cylinder
    let sep4 = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        sep4,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(-1.25, 0.0, -2.5))),
    );
    let mat4 = graph.add_child(
        sep4,
        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.9, 0.8, 0.1))),
    );
    graph.add_child(
        sep4,
        NodeData::Cylinder(CylinderNode {
            radius: 0.5,
            height: 1.2,
        }),
    );

    // Purple cube
    let sep5 = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        sep5,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(1.25, 0.0, -2.5))),
    );
    let mat5 = graph.add_child(
        sep5,
        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.7, 0.2, 0.8))),
    );
    graph.add_child(sep5, NodeData::Cube(CubeNode::default()));

    let ctrl = CameraController::new(camera_id, Vec3::ZERO, 8.0);

    let mut app = App::new(graph)
        .with_camera_controller(ctrl)
        .on_pick(move |graph, hit_node, _point| {
            // Find the material sibling of the shape's separator
            fn find_material_in_parent(graph: &rc3d_scene::SceneGraph, node: rc3d_core::NodeId) -> Option<rc3d_core::NodeId> {
                let parent = graph.get(node)?.parent?;
                let entry = graph.get(parent)?;
                for &child in &entry.children {
                    if let Some(ce) = graph.get(child) {
                        if matches!(ce.data, NodeData::Material(_)) {
                            return Some(child);
                        }
                    }
                }
                None
            }

            if let Some(mat_id) = find_material_in_parent(graph, hit_node) {
                if let Some(entry) = graph.get_mut(mat_id) {
                    if let NodeData::Material(mat) = &mut entry.data {
                        // Highlight: set to white
                        mat.diffuse_color = Vec3::ONE;
                        mat.base_color = Vec3::ONE;
                        log::info!("Picked node, highlighted material");
                    }
                }
            }
        });

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}
