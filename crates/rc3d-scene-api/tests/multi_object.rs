use rc3d_scene_api::{Cube, Scene};

#[test]
fn multi_object_scene_has_all_nodes() {
    let mut scene = Scene::new();
    scene.add(Cube::default().at(-2.0, 0.0, 0.0));
    scene.add(Cube::default().at(2.0, 0.0, 0.0));
    scene.add(Cube::default().at(0.0, 2.0, 0.0));
    let graph = scene.build();
    assert!(graph.node_count() >= 3);
}
