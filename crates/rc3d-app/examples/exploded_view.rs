//! ExplodedView demo — assembly explosion along direction.
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(2.0, 2.0, 4.0), Vec3::new(0.0, 0.8, 0.0), Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    // ExplodedView offsets children along Y with factor 1.5
    let ev = g.add_child(root, NodeData::ExplodedView(ExplodedViewNode { direction: Vec3::Y, factor: 1.5, center: Vec3::ZERO }));
    // Three boxes stacked before explosion
    for i in 0..3i32 {
        let c = [Vec3::new(0.9,0.2,0.2), Vec3::new(0.2,0.9,0.2), Vec3::new(0.2,0.2,0.9)][i as usize];
        g.add_child(ev, NodeData::Material(MaterialNode { base_color: c, roughness: 0.3, ..Default::default() }));
        g.add_child(ev, NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, i as f32*1.1, 0.0))));
        g.add_child(ev, NodeData::Cube(CubeNode { width: 1.0, height: 0.8, depth: 0.6 }));
    }
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
