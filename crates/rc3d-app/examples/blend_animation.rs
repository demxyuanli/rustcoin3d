//! BlendNode animation demo — clip blending with weight parameter.
use rc3d_app::App; use rc3d_core::math::{Quat, Vec3};
use rc3d_scene::{animation::{BlendNode, AnimationClip, Joint, JointKeyframe, JointTrack, Skeleton}, node_data::*};

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.3, 0.7, 0.9), roughness: 0.4, ..Default::default() }));
    g.add_child(root, NodeData::Transform(TransformNode { translation: Vec3::new(0.0, 1.5, 0.0), ..Default::default() }));
    g.add_child(root, NodeData::Sphere(SphereNode { radius: 0.6 }));
    // Build a simple blend tree: idle clip blended with walk clip at 50%
    let skeleton = Skeleton::new(vec![Joint { name: "root".into(), parent: 1, bind_transform: Mat4::IDENTITY, inverse_bind_matrix: Mat4::IDENTITY }]);
    let idle = AnimationClip { name: "idle".into(), duration: 1.0, tracks: vec![] };
    let walk = AnimationClip { name: "walk".into(), duration: 0.5, tracks: vec![] };
    let blend = BlendNode::Blend { left: Box::new(BlendNode::Clip { clip: idle, speed: 1.0, start_time: 0.0 }), right: Box::new(BlendNode::Clip { clip: walk, speed: 1.0, start_time: 0.0 }), weight: 0.5 };
    if let Some(_pose) = blend.sample(0.0, &skeleton) { println!("Blend tree sampled successfully at time 0"); }
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
