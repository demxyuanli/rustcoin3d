use rc3d_app::App;
use rc3d_scene::node_data::*;
use rc3d_core::math::Vec3;

fn main() {
    env_logger::init();

    let mut graph = rc3d_scene::SceneGraph::new();

    // Build: Separator { Coordinate3, Material, Triangle }
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    graph.add_child(
        root,
        NodeData::Coordinate3(Coordinate3Node::from_points(vec![
            Vec3::new(-0.5, -0.5, 0.0),
            Vec3::new(0.5, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
        ])),
    );
    graph.add_child(
        root,
        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(1.0, 0.0, 0.0))),
    );
    graph.add_child(root, NodeData::Triangle(TriangleNode));

    let mut app = App::new(graph);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}
