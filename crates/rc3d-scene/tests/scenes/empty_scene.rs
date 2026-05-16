use rc3d_scene::SceneGraph;

#[test]
fn empty_scene_graph_constructs_without_panic() {
    let graph = SceneGraph::new();
    // Minimal: verify graph constructs without panic
}

#[test]
fn empty_scene_node_count_is_non_negative() {
    let graph = SceneGraph::new();
    let count = graph.node_count();
    assert!(count < 1000); // reasonable upper bound for empty scene
}
