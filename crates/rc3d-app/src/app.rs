use rc3d_actions::Ray;
use rc3d_core::math::Vec3;
use rc3d_engine::EngineRegistry;
use rc3d_render::{RenderCollector, Renderer};
use rc3d_scene::SceneGraph;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::WindowAttributes,
};

use crate::camera_controller::CameraController;

pub struct App {
    pub graph: SceneGraph,
    pub renderer: Option<Renderer>,
    pub window: Option<winit::window::Window>,
    pub camera_controller: Option<CameraController>,
    pub engines: Option<EngineRegistry>,
    pub on_pick: Option<Box<dyn FnMut(&mut SceneGraph, rc3d_core::NodeId, Vec3)>>,
    pub picked_node: Option<rc3d_core::NodeId>,
    cursor_pos: (f64, f64),
    start: std::time::Instant,
}

impl App {
    pub fn new(graph: SceneGraph) -> Self {
        Self {
            graph,
            renderer: None,
            window: None,
            camera_controller: None,
            engines: None,
            on_pick: None,
            picked_node: None,
            cursor_pos: (0.0, 0.0),
            start: std::time::Instant::now(),
        }
    }

    pub fn with_camera_controller(mut self, controller: CameraController) -> Self {
        self.camera_controller = Some(controller);
        self
    }

    pub fn with_engines(mut self, engines: EngineRegistry) -> Self {
        self.engines = Some(engines);
        self
    }

    pub fn on_pick(
        mut self,
        f: impl FnMut(&mut SceneGraph, rc3d_core::NodeId, Vec3) + 'static,
    ) -> Self {
        self.on_pick = Some(Box::new(f));
        self
    }

    pub fn scene_graph(&self) -> &SceneGraph {
        &self.graph
    }

    pub fn scene_graph_mut(&mut self) -> &mut SceneGraph {
        &mut self.graph
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("rustcoin3d")
                        .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
                )
                .expect("failed to create window");
            let renderer = pollster::block_on(Renderer::new(&window));
            self.window = Some(window);
            self.renderer = Some(renderer);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match &event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                let size = *physical_size;
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = (position.x, position.y);
            }
            WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                if self.camera_controller.is_none() {
                    self.do_pick();
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(window)) = (&self.renderer, &self.window) {
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    // Evaluate engines (animation)
                    if let Some(ref mut engines) = self.engines {
                        let time = self.start.elapsed().as_secs_f64();
                        engines.evaluate_all(&mut self.graph, time);
                    }

                    if let Some(ref ctrl) = self.camera_controller {
                        ctrl.update_camera(&mut self.graph, aspect);
                    }

                    let mut collector = RenderCollector::new();
                    for &root in self.graph.roots() {
                        collector.traverse(&self.graph, root);
                    }
                    if !collector.draw_calls.is_empty() {
                        renderer.render_draw_calls(&collector.draw_calls);
                    } else {
                        drop(renderer.surface.get_current_texture());
                    }
                    window.request_redraw();
                }
            }
            _ => {}
        }

        if let Some(ref mut ctrl) = self.camera_controller {
            ctrl.handle_event(&event);
        }
    }
}

impl App {
    fn do_pick(&mut self) {
        let (_renderer, window) = match (&self.renderer, &self.window) {
            (Some(r), Some(w)) => (r, w),
            _ => return,
        };
        let size = window.inner_size();

        let mut collector = RenderCollector::new();
        for &root in self.graph.roots() {
            collector.traverse(&self.graph, root);
        }
        let view = collector.state.view_matrix();
        let proj = collector.state.projection_matrix();

        let ray = Ray::from_screen_point(
            self.cursor_pos.0 as f32,
            self.cursor_pos.1 as f32,
            size.width as f32,
            size.height as f32,
            view,
            proj,
        );

        let mut picker = rc3d_actions::RayPickAction::new(ray);
        for &root in self.graph.roots() {
            picker.apply(&self.graph, root);
        }

        if let Some(hit) = picker.hits.first() {
            self.picked_node = Some(hit.node);
            if let Some(cb) = &mut self.on_pick {
                cb(&mut self.graph, hit.node, hit.point);
            }
        }
    }
}
