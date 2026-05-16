use rc3d_scene::node_data::{GroupNode, MarkupElement, MarkupNode, NodeData, SeparatorNode};
use rc3d_scene::SceneGraph;

use super::pass_markup::collect_markup_lines;

#[test]
fn empty_scene_produces_no_markup_lines() {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Group(GroupNode));
    let vertices = collect_markup_lines(&graph, root, 800, 600);
    assert!(
        vertices.is_empty(),
        "Expected no markup vertices from a graph with only a Group root"
    );
}

#[test]
fn markup_line_collection_does_not_panic_on_arbitrary_graph() {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    let child1 = graph.add_child(root, NodeData::Group(GroupNode));
    graph.add_child(
        child1,
        NodeData::Markup(MarkupNode {
            elements: vec![MarkupElement::Line {
                start: [0.0, 0.0],
                end: [100.0, 50.0],
                color: [1.0, 0.0, 0.0, 1.0],
                width: 2.0,
            }],
            layer_name: "test_layer".into(),
            visible: true,
        }),
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        collect_markup_lines(&graph, root, 800, 600);
    }));
    assert!(
        result.is_ok(),
        "collect_markup_lines panicked on a valid scene graph"
    );
}
