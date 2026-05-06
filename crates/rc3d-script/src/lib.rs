use std::cell::RefCell;
use std::rc::Rc;

use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;
use rhai::Engine;

/// Execute a Rhai script that builds a scene graph.
///
/// The script has access to the following helper functions:
/// - `add_cube(x, y, z)` — cube at position with default size
/// - `add_sphere(x, y, z, radius)` — sphere at position
/// - `add_light(dx, dy, dz)` — directional light
pub fn execute_scene_script(script: &str) -> Result<SceneGraph, String> {
    let mut engine = Engine::new();
    let graph = Rc::new(RefCell::new(SceneGraph::new()));

    let g = graph.clone();
    engine.register_fn("add_cube", move |x: f32, y: f32, z: f32| {
        let mut graph = g.borrow_mut();
        let root = graph.roots().first().copied().unwrap_or_else(|| {
            graph.add_root(NodeData::Separator(SeparatorNode))
        });
        let tf = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(x, y, z))),
        );
        graph.add_child(tf, NodeData::Cube(CubeNode::default()));
    });

    let g = graph.clone();
    engine.register_fn("add_sphere", move |x: f32, y: f32, z: f32, r: f32| {
        let mut graph = g.borrow_mut();
        let root = graph.roots().first().copied().unwrap_or_else(|| {
            graph.add_root(NodeData::Separator(SeparatorNode))
        });
        let tf = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(x, y, z))),
        );
        graph.add_child(tf, NodeData::Sphere(SphereNode { radius: r }));
    });

    let g = graph.clone();
    engine.register_fn("add_light", move |dx: f32, dy: f32, dz: f32| {
        let mut graph = g.borrow_mut();
        let root = graph.roots().first().copied().unwrap_or_else(|| {
            graph.add_root(NodeData::Separator(SeparatorNode))
        });
        graph.add_child(
            root,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(dx, dy, dz).normalize(),
                color: Vec3::ONE,
                intensity: 1.0,
                light_group: None,
            }),
        );
    });

    engine.run(script).map_err(|e| e.to_string())?;
    drop(engine);

    let graph = Rc::try_unwrap(graph)
        .map_err(|_| "internal error: graph still referenced".to_string())?
        .into_inner();
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_script() {
        let script = r#"
            add_cube(0.0, 0.0, 0.0);
            add_sphere(2.0, 0.0, 0.0, 0.5);
            add_light(-1.0, -1.0, -1.0);
        "#;
        let result = execute_scene_script(script);
        assert!(result.is_ok());
        let graph = result.unwrap();
        assert!(!graph.roots().is_empty());
    }

    #[test]
    fn test_empty_script() {
        let script = "";
        let graph = execute_scene_script(script).unwrap();
        assert!(graph.roots().is_empty());
    }

    #[test]
    fn test_multiple_cubes() {
        let script = r#"
            add_cube(0.0, 0.0, 0.0);
            add_cube(1.0, 0.0, 0.0);
            add_cube(2.0, 0.0, 0.0);
            add_cube(3.0, 0.0, 0.0);
            add_cube(4.0, 0.0, 0.0);
        "#;
        let graph = execute_scene_script(script).unwrap();
        let root = graph.roots()[0];
        assert_eq!(graph.children(root).unwrap().len(), 5);
    }
}
