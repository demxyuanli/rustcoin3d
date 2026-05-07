//! MaterialBinding + ShapeHints + Texture2Transform demo.
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    log::info!("=== {} demo ===", "material_variants");
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    // Feature label
    g.add_child(root, NodeData::Text2(Text2Node { string: "Material Variants".into(), position: [10.0, 10.0], size: 18.0, color: [1.0, 0.9, 0.3, 1.0] }));
    // ShapeHints: set counter-clockwise vertex ordering
    g.add_child(root, NodeData::ShapeHints(ShapeHintsNode { vertex_ordering: VertexOrdering::CounterClockwise, ..Default::default() }));
    // MaterialBinding: per-vertex color
    g.add_child(root, NodeData::MaterialBinding(MaterialBindingNode { value: MaterialBinding::PerVertex }));
    // Three colored cubes with different materials
    for i in 0..3i32 {
        let c = [Vec3::new(0.9,0.2,0.2), Vec3::new(0.2,0.9,0.2), Vec3::new(0.2,0.2,0.9)][i as usize];
        g.add_child(root, NodeData::Material(MaterialNode { base_color: c, roughness: 0.3, metallic: i as f32 * 0.2, ..Default::default() }));
        g.add_child(root, NodeData::Transform(TransformNode::from_translation(Vec3::new(i as f32 * 2.0 - 2.0, 0.0, 0.0))));
        g.add_child(root, NodeData::Cube(CubeNode::default()));
    }
    // Texture2Transform: rotate texture coordinates
    g.add_child(root, NodeData::Texture2Transform(Texture2TransformNode { rotation: 0.5, ..Default::default() }));
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
