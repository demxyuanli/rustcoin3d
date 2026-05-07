//! NURBS curve and surface demo — tessellation with adaptive subdivision.
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_nurbs::{curve::NurbsCurve, knot::open_uniform_knots}; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(0.0, 0.0, 8.0), Vec3::new(0.0, 1.5, 0.0), Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    // Create a NURBS curve (quadratic, 3 control points)
    let degree = 2; let n_ctrl = 3;
    let knots = open_uniform_knots(degree, n_ctrl);
    let ctrl_pts = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 3.0, 0.0), Vec3::new(4.0, 0.0, 0.0)];
    let curve = NurbsCurve::new(degree, knots, ctrl_pts);
    let tess = curve.tessellate(0.05);
    println!("NURBS curve: {} control points -> {} tessellated vertices", 3, tess.len());
    // Visualize with IndexedLineSet
    let mut coord = tess.clone(); let mut indices: Vec<i32> = (0..tess.len() as i32).collect();
    g.add_child(root, NodeData::Coordinate3(Coordinate3Node::from_points(coord)));
    g.add_child(root, NodeData::IndexedLineSet(IndexedLineSetNode { coord_index: indices, line_width: 2.0 }));
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
