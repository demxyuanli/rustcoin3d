//! Editor — Unity-style UI (menu, toolbar, Hierarchy, Inspector, status)
//! plus gizmo, viewports, undo, clipping, measurement.
//!
//! Camera: middle drag = orbit | right drag = pan | wheel = zoom.
//! Gizmo: toolbar Move / Rotate / Scale, or T / R / G, then drag handles.
//! Ctrl+Z / Ctrl+Y: undo / redo (also under Edit in the menu bar).
//! C: cycle layout (Single -> Quad -> Left/Right -> Top/Bottom)
//! Tab: cycle active viewport (when multi-viewport)
//! P: section edit on/off, [ / ]: nudge planes
//! X / Y / Z: axis clip toggles (without Ctrl)
//! F: fit camera to selection
//! M: measurement mode, click two points
//! Escape: clear selection and measurements
//! W / S / E / H: wireframe / shaded / shaded+edges / hidden-line
//! I: cycle IBL preset

use rc3d_editor::Editor;
use rc3d_engine_api::{CameraController, Engine};
use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;
use rc3d_examples::common::run_app;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowAttributes;

struct EditorApp {
    pending_graph: Option<SceneGraph>,
    engine: Option<Engine>,
    editor: Option<Editor>,
    window: Option<winit::window::Window>,
}

impl EditorApp {
    fn new(graph: SceneGraph) -> Self {
        Self {
            pending_graph: Some(graph),
            engine: None,
            editor: None,
            window: None,
        }
    }
}

impl ApplicationHandler for EditorApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = event_loop
            .create_window(
                WindowAttributes::default().with_title("rustcoin3d Editor"),
            )
            .expect("failed to create window");
        let graph = self.pending_graph.take().expect("demo scene");
        let mut engine = Engine::new(&window);
        engine.load_scene(graph);
        engine.controller = CameraController::new(Vec3::ZERO, 10.0);
        engine.set_display_mode(DisplayMode::Shaded);
        let editor = Editor::new(&window, &engine);
        self.engine = Some(engine);
        self.editor = Some(editor);
        self.window = Some(window);
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        let Some(engine) = self.engine.as_mut() else {
            return;
        };
        if editor.handle_event(window, &event) {
            return;
        }
        match event {
            WindowEvent::RedrawRequested => {
                engine.render();
                window.request_redraw();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                engine.resize(size.width, size.height);
                editor.resize(size.width, size.height, window.scale_factor() as f32);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d=info"),
    )
    .init();

    println!("rustcoin3d Editor (egui in-viewport, Unity-style layout)");
    println!("  Menu bar, tools row, Hierarchy / Inspector, bottom status.");
    println!("  Open Help from the Help menu for shortcuts.");

    run_app(EditorApp::new(build_demo_scene()));
}

fn build_demo_scene() -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera
    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(5.0, 4.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // Lights
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity: 1.2,
            light_group: None,
        }),
    );
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(0.6, -0.4, 0.5).normalize(),
            color: Vec3::new(0.6, 0.7, 1.0),
            intensity: 0.5,
            light_group: None,
        }),
    );

    // Floor
    let floor_sep =
        graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        floor_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.4, 0.4, 0.45),
            ambient_color: Vec3::new(0.05, 0.05, 0.05),
            specular_color: Vec3::new(0.1, 0.1, 0.1),
            shininess: 4.0,
            base_color: Vec3::new(0.4, 0.4, 0.45),
            metallic: 0.0,
            roughness: 0.9,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(
        floor_sep,
        NodeData::Cube(CubeNode {
            width: 10.0,
            height: 0.2,
            depth: 10.0,
        }),
    );

    // Point light above center for specular highlights
    graph.add_child(
        root,
        NodeData::PointLight(PointLightNode {
            location: Vec3::new(0.0, 5.0, 0.0),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 40.0,
            light_group: None,
        }),
    );

    // PBR material grid: rows = increasing roughness, cols = increasing metallic
    let grid_size = 5;
    let spacing = 2.5;
    for row in 0..grid_size {
        let roughness = row as f32 / (grid_size - 1) as f32;
        for col in 0..grid_size {
            let metallic = col as f32 / (grid_size - 1) as f32;
            let sep =
                graph.add_child(root, NodeData::Separator(SeparatorNode));
            let x = (col as f32 - 2.0) * spacing;
            let z = (row as f32 - 2.0) * spacing;
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(
                    Vec3::new(x, 1.2, z),
                )),
            );
            let hue = 0.12 + metallic * 0.08;
            let sat = 0.3 + roughness * 0.5;
            let lum = 0.55 + roughness * 0.2;
            let base = hsl_to_rgb(hue, sat, lum);
            graph.add_child(
                sep,
                NodeData::Material(MaterialNode {
                    diffuse_color: base,
                    ambient_color: base * 0.15,
                    specular_color: Vec3::new(0.04, 0.04, 0.04),
                    shininess: ((1.0 - roughness).max(0.01) * 128.0) as f32,
                    base_color: base,
                    metallic,
                    roughness,
                    opacity: 1.0,
                    ..Default::default()
                }),
            );
            graph
                .add_child(sep, NodeData::Sphere(SphereNode { radius: 0.8 }));
        }
    }

    graph.add_child(
        root,
        NodeData::EventCallback(EventCallbackNode::default()),
    );

    graph
}

/// Simple HSL to RGB for material color variation.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Vec3 {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c * 0.5;
    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Vec3::new(r + m, g + m, b + m)
}
