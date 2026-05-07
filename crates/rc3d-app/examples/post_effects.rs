//! PostEffectParams demo — vignette, chromatic aberration, bloom, film grain.
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.6, 0.3, 0.3), roughness: 0.3, metallic: 0.5, ..Default::default() }));
    g.add_child(root, NodeData::Sphere(SphereNode { radius: 1.0 }));
    let mut app = App::new(g);
    // Apply post effects: vignette=0.4, chromatic=0.01, bloom=1.0, grain=0.02
    if let Some(renderer) = &mut app.state.renderer {
        renderer.set_post_effect_params(0.4, 0.01, 1.0, 0.02);
    }
    println!("Post effects applied: vignette=0.4, chromatic=0.01, bloom=1.0, grain=0.02");
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut app).expect("event loop");
}
