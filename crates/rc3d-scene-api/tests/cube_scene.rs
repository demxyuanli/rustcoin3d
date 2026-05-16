use rc3d_scene_api::{Cube, Scene};

#[test]
fn cube_scene_builds_and_has_nodes() {
    let mut scene = Scene::new();
    scene.add(Cube::default().at(0.0, 0.0, 0.0));
    let graph = scene.build();
    assert!(graph.node_count() > 0);
}
