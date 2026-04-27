use rc3d_actions::Ray;
use rc3d_core::{math::{Mat4, Vec3}, DisplayMode};
use rc3d_engine::EngineRegistry;
use rc3d_render::{DrawCall, RenderCollector, Renderer};
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
    cursor_pos: (f64, f64),
    shift_pressed: bool,
    start: std::time::Instant,
    measurement_mode: bool,
    measurement_first_point: Option<Vec3>,
    measurements: Vec<(Vec3, Vec3, f32)>,
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
            cursor_pos: (0.0, 0.0),
            shift_pressed: false,
            start: std::time::Instant::now(),
            measurement_mode: false,
            measurement_first_point: None,
            measurements: Vec::new(),
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
            WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    state: winit::event::ElementState::Pressed,
                    physical_key: key,
                    ..
                },
                ..
            } => {
                match key {
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyW) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.set_display_mode(DisplayMode::Wireframe);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyS) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.set_display_mode(DisplayMode::Shaded);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyE) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.set_display_mode(DisplayMode::ShadedWithEdges);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyH) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.set_display_mode(DisplayMode::HiddenLine);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyX) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.toggle_clip_plane(0);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyY) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.toggle_clip_plane(1);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyZ) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.toggle_clip_plane(2);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) => {
                        self.graph.clear_selection();
                        self.measurements.clear();
                        self.measurement_first_point = None;
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyM) => {
                        self.measurement_mode = !self.measurement_mode;
                        self.measurement_first_point = None;
                        log::info!("Measurement mode: {}", self.measurement_mode);
                    }
                    _ => {}
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.shift_pressed = mods.state().shift_key();
            }
            WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                if self.measurement_mode {
                    self.do_measure_pick();
                } else if self.camera_controller.is_none() || self.shift_pressed {
                    self.do_pick();
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
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
                    collector.global_display_mode = renderer.display_mode();
                    for &root in self.graph.roots() {
                        collector.traverse(&self.graph, root);
                    }

                    // Add measurement line draw calls
                    if !self.measurements.is_empty() {
                        let vp = collector.projection_matrix * collector.view_matrix;
                        for &(p1, p2, _dist) in &self.measurements {
                            collector.draw_calls.push(DrawCall {
                                vertices: Vec::new(),
                                indices: None,
                                edge_positions: vec![p1.to_array(), p2.to_array()],
                                mvp: vp,
                                model_matrix: Mat4::IDENTITY,
                                camera_pos: collector.camera_pos,
                                light_dir: Vec3::ZERO,
                                light_color: Vec3::ZERO,
                                color: Vec3::ZERO,
                                aabb: None,
                                display_mode: DisplayMode::ShadedWithEdges,
                                selected: false,
                                overlay_color: Some([1.0, 1.0, 0.0, 1.0]),
                            });
                        }
                    }

                    if !collector.draw_calls.is_empty() {
                        renderer.render_draw_calls(&mut collector.draw_calls);
                    } else {
                        drop(renderer.surface.get_current_texture());
                    }
                    window.request_redraw();
                }
            }
            _ => {}
        }

        if let Some(ref mut ctrl) = self.camera_controller {
            if !self.measurement_mode {
                ctrl.handle_event(&event);
            }
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
        let view = collector.view_matrix;
        let proj = collector.projection_matrix;

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
            self.graph.toggle_selection(hit.node);
            log::info!(
                "Pick hit: node={:?}, point={:?}, selected={}",
                hit.node, hit.point, self.graph.is_selected(hit.node)
            );
            if let Some(cb) = &mut self.on_pick {
                cb(&mut self.graph, hit.node, hit.point);
            }
        } else {
            log::info!("Pick miss (no hit)");
            if !self.shift_pressed {
                self.graph.clear_selection();
            }
        }
    }

    fn do_measure_pick(&mut self) {
        let (_renderer, window) = match (&self.renderer, &self.window) {
            (Some(r), Some(w)) => (r, w),
            _ => return,
        };
        let size = window.inner_size();

        let mut collector = RenderCollector::new();
        for &root in self.graph.roots() {
            collector.traverse(&self.graph, root);
        }
        let view = collector.view_matrix;
        let proj = collector.projection_matrix;

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
            let point = hit.point;
            match self.measurement_first_point {
                None => {
                    self.measurement_first_point = Some(point);
                    log::info!("Measurement point A: {:?}", point);
                }
                Some(first) => {
                    let dist = (point - first).length();
                    self.measurements.push((first, point, dist));
                    self.measurement_first_point = None;
                    log::info!("Measurement: A={:?} B={:?} distance={:.4}", first, point, dist);
                }
            }
        }
    }
}
