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

use rc3d_editor::{
    interaction, Editor, EditorCommand, EditorContext, EditorInteractionState,
};
use rc3d_engine_api::{
    sync_gizmo_from_selection, CameraController, Engine, EventRouteOpts,
};
use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_gizmo::{GizmoAxis, GizmoHandle, GizmoMode};
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;
use rc3d_examples::common::run_app;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowAttributes;

struct EditorApp {
    pending_graph: Option<SceneGraph>,
    engine: Option<Engine>,
    editor: Option<Editor>,
    window: Option<winit::window::Window>,
    interaction: EditorInteractionState,
}

impl EditorApp {
    fn new(graph: SceneGraph) -> Self {
        Self {
            pending_graph: Some(graph),
            engine: None,
            editor: None,
            window: None,
            interaction: EditorInteractionState::default(),
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
        maybe_verify_gizmo(&mut engine);
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
        let egui_consumed = editor.handle_event(window, &event);
        let is_pointer = matches!(
            event,
            WindowEvent::CursorMoved { .. }
                | WindowEvent::MouseInput { .. }
                | WindowEvent::MouseWheel { .. }
        );
        if egui_consumed && !is_pointer {
            return;
        }
        match event {
            WindowEvent::RedrawRequested => {
                for cmd in editor.take_commands() {
                    apply_editor_command(engine, cmd);
                }
                engine.render();
                window.request_redraw();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                engine.resize(size.width, size.height);
                editor.resize(size.width, size.height, window.scale_factor() as f32);
            }
            WindowEvent::CursorMoved { .. } => {
                engine.feed_input(&event);
                let input = engine.input;
                {
                    let mut ctx = EditorContext::with_interaction(
                        engine,
                        std::mem::take(&mut self.interaction),
                    );
                    interaction::on_cursor_moved(&mut ctx, window, &input);
                    self.interaction = ctx.interaction;
                }
                engine.dispatch_routed_event(&event, EventRouteOpts::editor());
            }
            WindowEvent::MouseInput { state, button, .. } => {
                engine.feed_input(&event);
                let input = engine.input;
                if button == MouseButton::Left {
                    let mut ctx = EditorContext::with_interaction(
                        engine,
                        std::mem::take(&mut self.interaction),
                    );
                    match state {
                        ElementState::Pressed => {
                            interaction::on_left_down(&mut ctx, window, &input, true);
                        }
                        ElementState::Released => {
                            interaction::on_left_up(&mut ctx, window, &input, true);
                        }
                    }
                    self.interaction = ctx.interaction;
                }
                engine.dispatch_routed_event(&event, EventRouteOpts::editor());
            }
            WindowEvent::MouseWheel { .. } => {
                engine.handle_window_event(&event, EventRouteOpts::editor());
            }
            WindowEvent::ModifiersChanged(_) => {
                engine.feed_input(&event);
            }
            WindowEvent::KeyboardInput { event: ref key, .. } if key.state == ElementState::Pressed => {
                engine.handle_window_event(
                    &event,
                    EventRouteOpts {
                        left_orbit: false,
                        pick_on_click: false,
                        camera: false,
                    },
                );
                match key.physical_key {
                    PhysicalKey::Code(KeyCode::KeyT) => engine.set_gizmo_mode(GizmoMode::Translate),
                    PhysicalKey::Code(KeyCode::KeyR) => engine.set_gizmo_mode(GizmoMode::Rotate),
                    PhysicalKey::Code(KeyCode::KeyG) => engine.set_gizmo_mode(GizmoMode::Scale),
                    PhysicalKey::Code(KeyCode::Escape) => {
                        engine.world.graph.clear_selection();
                        sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
                    }
                    _ => {}
                }
            }
            _ => {
                engine.handle_window_event(&event, EventRouteOpts::editor());
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn apply_editor_command(engine: &mut Engine, cmd: EditorCommand) {
    match cmd {
        EditorCommand::ApplyVisualStyle(name) => {
            engine.apply_visual_style_to_selected(&name);
        }
        EditorCommand::SetViewportLayoutMode(mode) => {
            engine.set_layout_mode(mode);
        }
        EditorCommand::CycleViewportLayout => {
            engine.cycle_layout_mode();
        }
        EditorCommand::CycleActiveViewport => {
            engine.cycle_active_viewport();
        }
        EditorCommand::SetViewPreset(preset) => {
            engine.set_view_preset(preset);
        }
        EditorCommand::SetWboit(enabled) => {
            engine.set_wboit(enabled);
        }
        EditorCommand::SetGizmoMode(mode) => {
            engine.set_gizmo_mode(mode);
        }
        EditorCommand::SetSelection(Some(id)) => {
            engine.world.graph.clear_selection();
            engine.world.graph.select(id);
            sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
        }
        EditorCommand::SetSelection(None) => {
            engine.world.graph.clear_selection();
            sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
        }
        EditorCommand::FitSelection => {
            interaction::fit_selection_to_view(engine);
        }
        _ => {}
    }
}

fn maybe_verify_gizmo(engine: &mut Engine) {
    if !std::env::args().any(|a| a == "--verify-gizmo") {
        return;
    }
    let mut shape = None;
    let mut xform = None;
    for id in engine.world.graph.all_node_ids() {
        if let Some(e) = engine.world.graph.get(id) {
            match &e.data {
                NodeData::Sphere(_) if shape.is_none() => shape = Some(id),
                NodeData::Transform(_) if xform.is_none() => xform = Some(id),
                _ => {}
            }
        }
    }
    let Some(shape) = shape else {
        eprintln!("gizmo verify failed: no Sphere");
        std::process::exit(1);
    };
    let Some(xform) = xform else {
        eprintln!("gizmo verify failed: no Transform");
        std::process::exit(1);
    };

    engine.world.graph.clear_selection();
    engine.world.graph.select(shape);
    sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
    if !engine.gizmo.visible || engine.gizmo.target_node != Some(xform) {
        eprintln!(
            "gizmo verify failed: pick shape did not bind sibling Transform (visible={} target={:?})",
            engine.gizmo.visible, engine.gizmo.target_node
        );
        std::process::exit(1);
    }
    let batches = engine.gizmo.generate_lines();
    let verts: usize = batches.iter().map(|(l, _)| l.len()).sum();
    if verts < 4 {
        eprintln!("gizmo verify failed: no handle lines ({verts} verts)");
        std::process::exit(1);
    }

    let old = match &engine.world.graph.get(xform).unwrap().data {
        NodeData::Transform(t) => t.translation,
        _ => unreachable!(),
    };
    if (engine.gizmo.position - old).length() > 2.0 {
        eprintln!(
            "gizmo verify failed: handle at {:?} not on object {:?}",
            engine.gizmo.position, old
        );
        std::process::exit(1);
    }
    let handle = engine.gizmo.position + Vec3::X * engine.gizmo.screen_scale.max(0.2) * 0.6;
    let eye = Vec3::new(5.0, 4.0, 8.0);
    let ray0 = rc3d_actions::Ray::new(eye, handle - eye);
    engine
        .gizmo
        .start_drag(&ray0, GizmoHandle::TranslateArrow(GizmoAxis::X));
    let ray1 = rc3d_actions::Ray::new(eye, handle + Vec3::X - eye);
    let Some(delta) = engine.gizmo.drag_delta(&ray1) else {
        eprintln!("gizmo verify failed: drag_delta returned None");
        std::process::exit(1);
    };
    let shift = delta.to_scale_rotation_translation().2;
    if let Some(e) = engine.world.graph.get_mut(xform) {
        if let NodeData::Transform(t) = &mut e.data {
            t.translation += shift;
        }
    }
    engine.gizmo.end_drag();
    let new = match &engine.world.graph.get(xform).unwrap().data {
        NodeData::Transform(t) => t.translation,
        _ => unreachable!(),
    };
    let moved = (new - old).length();
    if !moved.is_finite() || moved < 1e-4 {
        eprintln!("gizmo verify failed: drag did not move Transform ({old:?} -> {new:?})");
        std::process::exit(1);
    }
    engine.render();
    let (vw, vh) = engine
        .renderer
        .as_ref()
        .map(|r| (r.config.width.max(1) as f32, r.config.height.max(1) as f32))
        .unwrap_or((800.0, 600.0));
    let (view, proj) = rc3d_engine_api::scene_pick_matrices(engine);
    let center = match &engine.world.graph.get(xform).unwrap().data {
        NodeData::Transform(t) => t.translation,
        _ => unreachable!(),
    };
    let clip = proj * view * center.extend(1.0);
    if clip.w.abs() > 1e-5 {
        let ndc = clip.truncate() / clip.w;
        let sx = (ndc.x * 0.5 + 0.5) * vw;
        let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * vh;
        let ray = rc3d_actions::Ray::from_screen_point(sx, sy, vw, vh, view, proj);
        let mut picker = rc3d_actions::RayPickAction::new(ray);
        rc3d_actions::apply_to_all_roots(&mut picker, &engine.world.graph);
        match picker.hits.first() {
            Some(hit) if hit.node == shape => {}
            Some(hit) => {
                eprintln!(
                    "gizmo verify failed: pixel pick hit {:?} expected {:?}",
                    hit.node, shape
                );
                std::process::exit(1);
            }
            None => {
                eprintln!("gizmo verify failed: pixel pick missed shape");
                std::process::exit(1);
            }
        }
    }

    eprintln!(
        "gizmo verify ok: shape->Transform, {verts} handle verts, delta={}",
        (new - old).length()
    );
    if std::env::args().any(|a| a == "--verify-gizmo-exit") {
        std::process::exit(0);
    }
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d=info"),
    )
    .init();

    println!("rustcoin3d Editor (egui in-viewport, Unity-style layout)");
    println!("  Menu bar, tools row, Hierarchy / Inspector, bottom status.");
    println!("  Left click: pick (gizmo on Transform sibling). Drag handles to move.");
    println!("  T / R / G: translate / rotate / scale. Middle orbit, right pan.");
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

    // Always-on transform manipulator kit (Coin3D SoTransformManip + child draggers).
    let manip_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.set_name(manip_sep, "TransformManipKit");
    graph.add_child(
        manip_sep,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 3.5, 0.0))),
    );
    graph.add_child(
        manip_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.85, 0.85, 0.9),
            ambient_color: Vec3::new(0.1, 0.1, 0.12),
            specular_color: Vec3::new(0.04, 0.04, 0.04),
            shininess: 96.0,
            base_color: Vec3::new(0.85, 0.85, 0.9),
            metallic: 1.0,
            roughness: 0.15,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(
        manip_sep,
        NodeData::Cube(CubeNode {
            width: 0.8,
            height: 0.8,
            depth: 0.8,
        }),
    );
    let manip = graph.add_child(
        manip_sep,
        NodeData::TransformManip(TransformManipNode {
            mode: ManipMode::Translate,
            space: ManipSpace::World,
            enabled: true,
            size: 0.0,
            target: None,
        }),
    );
    for kind in [
        DraggerKind::TranslateX,
        DraggerKind::TranslateY,
        DraggerKind::TranslateZ,
    ] {
        graph.add_child(
            manip,
            NodeData::Dragger(DraggerNode {
                kind,
                enabled: true,
            }),
        );
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
