use rc3d_core::DisplayMode;
use rc3d_scene::SceneGraph;

use super::App;

#[test]
fn app_new_has_no_renderer_or_window() {
    let graph = SceneGraph::new();
    let app = App::new(graph);
    assert!(app.state.renderer.is_none());
    assert!(app.state.window.is_none());
    assert!(app.state.camera_controller.is_none());
}

#[test]
fn app_new_defaults_to_shaded_display_mode() {
    let graph = SceneGraph::new();
    let app = App::new(graph);
    assert_eq!(app.state.initial_display_mode, DisplayMode::Shaded);
}

#[test]
fn app_new_disables_editor_ui_by_default() {
    let graph = SceneGraph::new();
    let app = App::new(graph);
    assert!(!app.state.editor_ui_enabled);
}

#[test]
fn app_builder_with_editor_ui_enables_feature() {
    let graph = SceneGraph::new();
    let app = App::new(graph).with_editor_ui(true);
    assert!(app.state.editor_ui_enabled);
}

#[test]
fn app_state_holds_world_with_graph() {
    let graph = SceneGraph::new();
    let app = App::new(graph);
    // SceneGraph::new() creates an empty graph with no nodes.
    assert_eq!(app.state.world.graph.node_count(), 0);
}
