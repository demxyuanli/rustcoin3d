use rc3d_actions::Ray;
use rc3d_core::math::Mat4;
use rc3d_render::RenderCollector;
use rc3d_scene::SceneGraph;

pub(super) fn build_pick_ray(
    graph: &SceneGraph,
    cursor_pos: (f64, f64),
    window_size: (u32, u32),
) -> Option<Ray> {
    let (view, proj) = collect_view_projection(graph);
    Some(Ray::from_screen_point(
        cursor_pos.0 as f32,
        cursor_pos.1 as f32,
        window_size.0 as f32,
        window_size.1 as f32,
        view,
        proj,
    ))
}

fn collect_view_projection(graph: &SceneGraph) -> (Mat4, Mat4) {
    let mut collector = RenderCollector::new();
    for &root in graph.roots() {
        collector.traverse(graph, root);
    }
    (collector.view_matrix, collector.projection_matrix)
}
