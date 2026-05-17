//! Render features demo — toggles post-processing effects via keyboard.
//!
//! Keys:
//!   1-9: Toggle effects (HDR/TAA/MotionBlur/SSR/DOF/Fog/LUT/Shadows/AO)
//!   W/A/S/D/Q/E: Camera movement
//!   Mouse drag: Orbit
//!   F: Cycle display mode (Shaded/ShadedWithEdges/Wireframe/HiddenLine)
//!   L: Cycle IBL preset
//!   R: Reset exposure
//!   +/-: Adjust exposure
//!   V: Toggle vsync
//!   H: Toggle HUD
//!   Escape: clear selection | close window to quit
//!
//! Usage: cargo run -p rc3d-examples --example render_features -- <file.gltf|file.glb|file.obj|file.stl>

use std::env;
use std::path::Path;
use std::sync::{Arc, Mutex};

use rc3d_actions::CameraFitConfig;
use rc3d_editor::{
    preset_for_render_features_panel, RenderFeaturePanelHandle,
    RenderFeaturePanelState,
};
use rc3d_engine_api::{CameraController, Engine};
use rc3d_core::NodeId;
use rc3d_core::{math::Vec3, DisplayMode};
use rc3d_engine::{Engine as EngineTrait, EngineRegistry};
use rc3d_render::AdaptiveControl;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowAttributes;

#[derive(Debug)]
struct RenderFeaturesPanelEngine {
    panel: Arc<Mutex<RenderFeaturePanelState>>,
    material_ids: Vec<NodeId>,
    demo_light_ids: [NodeId; 3],
    debug_overlay_root: NodeId,
    base_light_intensities: [f32; 3],
    phase: f32,
    last_time: Option<f64>,
    idle_mode: bool,
}

impl RenderFeaturesPanelEngine {
    fn new(
        panel: Arc<Mutex<RenderFeaturePanelState>>,
        material_ids: Vec<NodeId>,
        demo_light_ids: [NodeId; 3],
        debug_overlay_root: NodeId,
        base_light_intensities: [f32; 3],
        idle_mode: bool,
    ) -> Self {
        Self {
            panel,
            material_ids,
            demo_light_ids,
            debug_overlay_root,
            base_light_intensities,
            phase: 0.0,
            last_time: None,
            idle_mode,
        }
    }
}

impl EngineTrait for RenderFeaturesPanelEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, time: f64) {
        let mut s = self.panel.lock().expect("panel state lock");
        let dt = if let Some(prev) = self.last_time {
            (time - prev).max(0.0) as f32
        } else {
            0.0
        };
        self.last_time = Some(time);
        let step_dt = s.advance_time(dt);
        if !self.idle_mode {
            self.phase += step_dt;
        }

        let opacity = if s.show_model { 1.0 } else { 0.02 };
        for &mid in &self.material_ids {
            if let Some(entry) = graph.get_mut(mid) {
                if let NodeData::Material(mat) = &mut entry.data {
                    mat.opacity = opacity;
                }
            }
        }

        let mut weights = [
            s.channel_a_weight,
            s.channel_b_weight,
            s.channel_c_weight,
        ];
        if !s.channel_a_active {
            weights[0] = 0.0;
        }
        if !s.channel_b_active {
            weights[1] = 0.0;
        }
        if !s.channel_c_active {
            weights[2] = 0.0;
        }
        let wsum = weights[0] + weights[1] + weights[2];
        if wsum > 1e-6 {
            weights[0] /= wsum;
            weights[1] /= wsum;
            weights[2] /= wsum;
        }
        let debug_scale = if s.show_debug_overlay {
            Vec3::ONE
        } else {
            Vec3::splat(0.0001)
        };
        drop(s);

        let wobble = if self.idle_mode {
            0.0
        } else {
            (self.phase * 1.8).sin() * 0.25
        };
        for (i, &lid) in self.demo_light_ids.iter().enumerate() {
            if let Some(entry) = graph.get_mut(lid) {
                if let NodeData::PointLight(light) = &mut entry.data {
                    light.intensity = self.base_light_intensities[i] * weights[i];
                    if self.idle_mode {
                        light.location = Vec3::new(
                            -2.0 + i as f32 * 2.0,
                            1.5,
                            2.0,
                        );
                    } else {
                        let offset = i as f32 * 2.1;
                        light.location = Vec3::new(
                            (self.phase * (0.6 + i as f32 * 0.25) + offset).sin()
                                * 2.0,
                            1.5 + wobble,
                            (self.phase * (0.4 + i as f32 * 0.15) + offset).cos()
                                * 2.0,
                        );
                    }
                }
            }
        }

        if let Some(entry) = graph.get_mut(self.debug_overlay_root) {
            if let NodeData::Transform(tr) = &mut entry.data {
                tr.scale = debug_scale;
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();
    print_render_features_help();

    let args: Vec<String> = env::args().collect();
    let no_panel = args.iter().skip(1).any(|arg| arg == "--no-panel");
    let idle_mode = args.iter().skip(1).any(|arg| arg == "--idle");
    let adaptive_control = parse_adaptive_quality_mode(&args);
    let Some(path_arg) = args.iter().skip(1).find(|arg| !arg.starts_with("--")) else {
        eprintln!("Error: no input file specified.");
        eprintln!("Usage: render_features <file.stl|file.obj|file.gltf|file.iv> [--no-panel] [--idle]");
        return;
    };

    let path = Path::new(path_arg);
    let mut graph = match rc3d_io::import_file(path) {
        Ok(g) => {
            println!(
                "Loaded: {} ({} roots, format auto-detected)",
                path.display(),
                g.roots().len()
            );
            g
        }
        Err(e) => {
            eprintln!("Import error: {e}");
            return;
        }
    };

    graph = ensure_scene_setup(graph);
    rc3d_actions::fit_camera_to_scene(&mut graph, CameraFitConfig::default());
    let material_ids = collect_material_nodes(&graph);
    let (demo_light_ids, base_light_intensities) =
        attach_panel_demo_lights(&mut graph);
    let debug_overlay_root = attach_render_debug_overlay(&mut graph);

    let event_loop =
        EventLoop::new().expect("failed to create render_features event loop");
    let window = event_loop
        .create_window(
            WindowAttributes::default().with_title("Render Features"),
        )
        .expect("failed to create window");

    let panel_preset = preset_for_render_features_panel();
    let panel_title = panel_preset.config.title.clone();
    let panel_state = Arc::new(Mutex::new(panel_preset.state));

    if no_panel {
        println!("Panel: disabled | title: {panel_title}");
    } else {
        println!("Panel: enabled (embedded) | title: {panel_title}");
    }
    println!("Adaptive quality mode: {:?}", adaptive_control);
    println!("Idle mode: {}", if idle_mode { "on" } else { "off" });

    let mut engines = EngineRegistry::new();
    engines.add(RenderFeaturesPanelEngine::new(
        panel_state.clone(),
        material_ids,
        demo_light_ids,
        debug_overlay_root,
        base_light_intensities,
        idle_mode,
    ));

    let ctrl = CameraController::new(Vec3::ZERO, 5.0);

    let mut engine = Engine::new(&window);
    engine.load_scene(graph);
    engine.controller = ctrl;
    engine.world_mut().engines = Some(engines);
    engine.set_display_mode(DisplayMode::Shaded);
    engine.set_adaptive_quality(adaptive_control);
    engine.continuous_redraw = !idle_mode;
    if let Some(ref mut r) = engine.renderer {
        r.hdr_post_processing = true;
    }

    if !no_panel {
        let panel_for_text = panel_state.clone();
        let panel_for_key = panel_state.clone();
        let selected = Arc::new(Mutex::new(0usize));
        let selected_for_text = selected.clone();
        let selected_for_key = selected.clone();
        let selected_for_mouse = selected.clone();
        let panel_for_mouse = panel_for_key.clone();

        engine.hud_text_hook = Some(Box::new(move || {
            let sel =
                *selected_for_text.lock().expect("panel selected lock");
            build_render_features_panel_overlay_with_selection(
                &panel_for_text, sel,
            )
        }));
        engine.panel_overlay_key_hook = Some(Box::new(
            move |key: PhysicalKey| -> bool {
                apply_render_features_panel_key(
                    &panel_for_key,
                    &selected_for_key,
                    key,
                );
                false
            },
        ));
        engine.panel_overlay_mouse_hook = Some(Box::new(
            move |x, y, w, h| {
                apply_render_features_panel_mouse(
                    &panel_for_mouse,
                    &selected_for_mouse,
                    x, y, w, h,
                )
            },
        ));
    }

    println!("Render features demo ready:");
    println!("  HDR:ON  TAA:ON  MotionBlur:ON  SSR:ON  DOF:ON  Fog:ON  Shadows:ON");
    println!("  Press 1-9 to toggle effects");

    let mut cursor_pos: Option<(f32, f32)> = None;
    let mut window_size: (u32, u32) = (800, 600);

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::RedrawRequested => {
                engine.render();
                window.request_redraw();
            }
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(size) => {
                window_size = (size.width, size.height);
                engine.resize(size.width, size.height);
            }
            WindowEvent::CursorMoved { position, .. } => {
                cursor_pos =
                    Some((position.x as f32, position.y as f32));
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } => {
                if let Some((x, y)) = cursor_pos {
                    if let Some(ref mut hook) =
                        engine.panel_overlay_mouse_hook
                    {
                        if hook(x, y, window_size.0, window_size.1) {
                            window.request_redraw();
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let Some(ref mut hook) =
                        engine.panel_overlay_key_hook
                    {
                        if hook(event.physical_key) {
                            window.request_redraw();
                        }
                    }
                }
            }
            _ => {}
        },
        Event::AboutToWait => {
            if engine.continuous_redraw {
                window.request_redraw();
            }
        }
        _ => {}
    });
}

// -- Helper functions preserved from original example --

fn build_render_features_panel_overlay_with_selection(
    state: &Arc<Mutex<RenderFeaturePanelState>>,
    selected: usize,
) -> String {
    let s = state.lock().expect("panel state lock");
    let mark = |idx: usize| if idx == selected { ">" } else { " " };
    let check = |v: bool| if v { "[x]" } else { "[ ]" };
    let slider_a = make_slider(s.channel_a_weight, 0.0, 1.0, 14);
    let slider_b = make_slider(s.channel_b_weight, 0.0, 1.0, 14);
    let slider_c = make_slider(s.channel_c_weight, 0.0, 1.0, 14);
    let slider_rate = make_slider(s.time_scale, 0.0, 3.0, 14);
    format!(
        "+-- Embedded Control Panel ------------------+\n\
{} {} Show Model\n\
{} {} Show Debug Overlay\n\
{} {} Channel A\n\
{} {} Channel B\n\
{} {} Channel C\n\
{} [slider] Mix A {} {:.2}\n\
{} [slider] Mix B {} {:.2}\n\
{} [slider] Mix C {} {:.2}\n\
{} [slider] Global Rate {} {:.2}\n\
Mouse: click checkbox/slider | Keyboard: Up/Down Left/Right Enter\n\
+--------------------------------------------+",
        mark(0),
        check(s.show_model),
        mark(1),
        check(s.show_debug_overlay),
        mark(2),
        check(s.channel_a_active),
        mark(3),
        check(s.channel_b_active),
        mark(4),
        check(s.channel_c_active),
        mark(5),
        slider_a,
        s.channel_a_weight,
        mark(6),
        slider_b,
        s.channel_b_weight,
        mark(7),
        slider_c,
        s.channel_c_weight,
        mark(8),
        slider_rate,
        s.time_scale
    )
}

fn make_slider(value: f32, min: f32, max: f32, width: usize) -> String {
    let t = if max > min {
        rc3d_core::utils::math::remap(value, min, max)
    } else {
        0.0
    };
    let pos = (t * (width.saturating_sub(1)) as f32).round() as usize;
    let mut s = String::with_capacity(width + 2);
    s.push('[');
    for i in 0..width {
        s.push(if i == pos { '|' } else { '-' });
    }
    s.push(']');
    s
}

fn apply_render_features_panel_key(
    state: &Arc<Mutex<RenderFeaturePanelState>>,
    selected: &Arc<Mutex<usize>>,
    key: PhysicalKey,
) {
    let mut sel = selected.lock().expect("panel selected lock");
    let item_count = 9usize;
    match key {
        PhysicalKey::Code(KeyCode::ArrowUp) => {
            *sel = (*sel + item_count - 1) % item_count;
            return;
        }
        PhysicalKey::Code(KeyCode::ArrowDown) => {
            *sel = (*sel + 1) % item_count;
            return;
        }
        _ => {}
    }
    let mut s = state.lock().expect("panel state lock");
    match key {
        PhysicalKey::Code(KeyCode::F5) => s.show_model = !s.show_model,
        PhysicalKey::Code(KeyCode::F6) => {
            s.show_debug_overlay = !s.show_debug_overlay
        }
        PhysicalKey::Code(KeyCode::F7) => {
            s.channel_a_active = !s.channel_a_active
        }
        PhysicalKey::Code(KeyCode::F8) => {
            s.channel_b_active = !s.channel_b_active
        }
        PhysicalKey::Code(KeyCode::F9) => {
            s.channel_c_active = !s.channel_c_active
        }
        PhysicalKey::Code(KeyCode::BracketLeft) => {
            s.time_scale = (s.time_scale - 0.1).max(0.0)
        }
        PhysicalKey::Code(KeyCode::BracketRight) => {
            s.time_scale = (s.time_scale + 0.1).min(3.0)
        }
        PhysicalKey::Code(KeyCode::Enter)
        | PhysicalKey::Code(KeyCode::Space) => match *sel {
            0 => s.show_model = !s.show_model,
            1 => s.show_debug_overlay = !s.show_debug_overlay,
            2 => s.channel_a_active = !s.channel_a_active,
            3 => s.channel_b_active = !s.channel_b_active,
            4 => s.channel_c_active = !s.channel_c_active,
            _ => {}
        },
        PhysicalKey::Code(KeyCode::ArrowLeft) => match *sel {
            5 => {
                s.channel_a_weight =
                    (s.channel_a_weight - 0.05).clamp(0.0, 1.0)
            }
            6 => {
                s.channel_b_weight =
                    (s.channel_b_weight - 0.05).clamp(0.0, 1.0)
            }
            7 => {
                s.channel_c_weight =
                    (s.channel_c_weight - 0.05).clamp(0.0, 1.0)
            }
            8 => s.time_scale = (s.time_scale - 0.1).max(0.0),
            _ => {}
        },
        PhysicalKey::Code(KeyCode::ArrowRight) => match *sel {
            5 => {
                s.channel_a_weight =
                    (s.channel_a_weight + 0.05).clamp(0.0, 1.0)
            }
            6 => {
                s.channel_b_weight =
                    (s.channel_b_weight + 0.05).clamp(0.0, 1.0)
            }
            7 => {
                s.channel_c_weight =
                    (s.channel_c_weight + 0.05).clamp(0.0, 1.0)
            }
            8 => s.time_scale = (s.time_scale + 0.1).min(3.0),
            _ => {}
        },
        _ => {}
    }
}

fn apply_render_features_panel_mouse(
    state: &Arc<Mutex<RenderFeaturePanelState>>,
    selected: &Arc<Mutex<usize>>,
    x: f32,
    y: f32,
    _w: u32,
    _h: u32,
) -> bool {
    const HUD_LEFT: f32 = 12.0;
    const HUD_TOP: f32 = 12.0;
    const LINE_H: f32 = 22.0;
    const PANEL_START_LINE: i32 = 3;
    const ITEM_LINES: [i32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let line = ((y - HUD_TOP) / LINE_H).floor() as i32;
    let rel = line - PANEL_START_LINE;
    let Some(idx) = ITEM_LINES.iter().position(|v| *v == rel) else {
        return false;
    };
    *selected.lock().expect("panel selected lock") = idx;
    let mut s = state.lock().expect("panel state lock");
    let slider_left = HUD_LEFT + 200.0;
    let slider_right = slider_left + 120.0;
    let slider_norm =
        rc3d_core::utils::math::remap(x, slider_left, slider_right);
    match idx {
        0 => s.show_model = !s.show_model,
        1 => s.show_debug_overlay = !s.show_debug_overlay,
        2 => s.channel_a_active = !s.channel_a_active,
        3 => s.channel_b_active = !s.channel_b_active,
        4 => s.channel_c_active = !s.channel_c_active,
        5 => s.channel_a_weight = slider_norm,
        6 => s.channel_b_weight = slider_norm,
        7 => s.channel_c_weight = slider_norm,
        8 => s.time_scale = slider_norm * 3.0,
        _ => {}
    }
    true
}

fn print_render_features_help() {
    println!("Render features example");
    println!("Usage: cargo run -p rc3d-examples --example render_features -- <file.gltf|file.glb|file.obj|file.stl|file.iv> [--no-panel] [--idle] [--adaptive-quality=off|on|auto-idle-lock]");
    println!("Controls:");
    println!("  1-9: Toggle HDR/TAA/MotionBlur/SSR/DOF/Fog/LUT/Shadows/SSAO");
    println!("  F: Cycle display mode");
    println!("  L: Cycle IBL preset");
    println!("  +/-: Adjust DOF focus");
    println!("  V: Toggle vsync");
    println!("  H: Toggle HUD");
    println!("  Mouse drag: orbit camera");
    println!("  Embedded panel HUD: clickable checkbox/slider + F5/F6/F7/F8/F9 and [ / ]");
    println!("  Escape: clear selection | Close window to quit");
    println!("Feature switches:");
    println!("  Full post-processing feature toggle matrix");
    println!("  --no-panel");
    println!("  --idle");
    println!("  --adaptive-quality=off|on|auto-idle-lock");
    println!("  Shared panel: visibility, channel enable, update control, channel mix, global rate");
}

fn parse_adaptive_quality_mode(args: &[String]) -> AdaptiveControl {
    for arg in args.iter().skip(1) {
        if let Some(v) = arg.strip_prefix("--adaptive-quality=") {
            return match v {
                "off" => AdaptiveControl::Disabled,
                "on" => AdaptiveControl::Dynamic {
                    allow_downgrade: true,
                },
                "auto-idle-lock" => AdaptiveControl::Dynamic {
                    allow_downgrade: false,
                },
                _ => AdaptiveControl::Dynamic {
                    allow_downgrade: false,
                },
            };
        }
    }
    AdaptiveControl::Dynamic {
        allow_downgrade: false,
    }
}

fn ensure_scene_setup(mut graph: SceneGraph) -> SceneGraph {
    let has_camera = has_node_type(&graph, |d| {
        matches!(
            d,
            NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_)
        )
    });
    let has_light = has_node_type(&graph, |d| {
        matches!(
            d,
            NodeData::DirectionalLight(_)
                | NodeData::PointLight(_)
                | NodeData::SpotLight(_)
        )
    });

    let root = graph.roots().first().copied().unwrap_or_else(|| {
        graph.add_root(NodeData::Separator(SeparatorNode))
    });

    if !has_camera {
        let cam = PerspectiveCameraNode::look_at(
            Vec3::new(5.0, 4.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        );
        graph.insert_child(root, 0, NodeData::PerspectiveCamera(cam));
    }

    if !has_light {
        graph.insert_child(
            root,
            1,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -0.8, -0.6).normalize(),
                color: Vec3::new(1.0, 0.95, 0.9),
                intensity: 1.2,
                light_group: None,
            }),
        );
        graph.insert_child(
            root,
            2,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(0.8, -0.4, 1.0).normalize(),
                color: Vec3::new(0.7, 0.8, 1.0),
                intensity: 0.5,
                light_group: None,
            }),
        );
        graph.insert_child(
            root,
            3,
            NodeData::PointLight(PointLightNode {
                location: Vec3::new(2.0, 3.0, 2.0),
                color: Vec3::new(1.0, 0.6, 0.3),
                intensity: 10.0,
                light_group: None,
            }),
        );
    }

    if !has_node_type(&graph, |d| matches!(d, NodeData::Material(_))) {
        graph.insert_child(
            root,
            0,
            NodeData::Material(MaterialNode {
                diffuse_color: Vec3::new(0.9, 0.9, 0.9),
                ambient_color: Vec3::new(0.25, 0.25, 0.25),
                specular_color: Vec3::new(0.04, 0.04, 0.04),
                shininess: 64.0,
                base_color: Vec3::new(0.94, 0.94, 0.94),
                metallic: 0.0,
                roughness: 0.45,
                opacity: 1.0,
                ..Default::default()
            }),
        );
    }

    graph
}

fn collect_material_nodes(graph: &SceneGraph) -> Vec<NodeId> {
    fn walk(graph: &SceneGraph, node: NodeId, out: &mut Vec<NodeId>) {
        let Some(entry) = graph.get(node) else {
            return;
        };
        if matches!(entry.data, NodeData::Material(_)) {
            out.push(node);
        }
        for &c in &entry.children {
            walk(graph, c, out);
        }
    }

    let mut out = Vec::new();
    for &root in graph.roots() {
        walk(graph, root, &mut out);
    }
    out
}

fn attach_panel_demo_lights(
    graph: &mut SceneGraph,
) -> ([NodeId; 3], [f32; 3]) {
    let root = graph.roots().first().copied().unwrap_or_else(|| {
        graph.add_root(NodeData::Separator(SeparatorNode))
    });
    let base = [8.0, 12.0, 18.0];
    let l0 = graph.add_child(
        root,
        NodeData::PointLight(PointLightNode {
            location: Vec3::new(-2.0, 1.5, 2.0),
            color: Vec3::new(0.8, 0.9, 1.0),
            intensity: base[0],
            light_group: None,
        }),
    );
    let l1 = graph.add_child(
        root,
        NodeData::PointLight(PointLightNode {
            location: Vec3::new(0.0, 2.0, 2.0),
            color: Vec3::new(1.0, 0.9, 0.6),
            intensity: base[1],
            light_group: None,
        }),
    );
    let l2 = graph.add_child(
        root,
        NodeData::PointLight(PointLightNode {
            location: Vec3::new(2.0, 2.5, 2.0),
            color: Vec3::new(1.0, 0.6, 0.5),
            intensity: base[2],
            light_group: None,
        }),
    );
    ([l0, l1, l2], base)
}

fn attach_render_debug_overlay(graph: &mut SceneGraph) -> NodeId {
    let root = graph.roots().first().copied().unwrap_or_else(|| {
        graph.add_root(NodeData::Separator(SeparatorNode))
    });
    let overlay = graph.add_child(
        root,
        NodeData::Transform(TransformNode::default()),
    );
    let grid_size = 6;
    let spacing = 1.0f32;

    for i in -grid_size..=grid_size {
        let z = i as f32 * spacing;
        let line =
            graph.add_child(overlay, NodeData::Separator(SeparatorNode));
        graph.add_child(
            line,
            NodeData::Transform(TransformNode {
                translation: Vec3::new(0.0, 0.01, z),
                scale: Vec3::new(
                    grid_size as f32 * spacing * 2.0,
                    0.01,
                    0.01,
                ),
                ..Default::default()
            }),
        );
        graph.add_child(
            line,
            NodeData::Material(MaterialNode {
                diffuse_color: Vec3::new(0.2, 0.7, 1.0),
                ambient_color: Vec3::new(0.05, 0.15, 0.2),
                specular_color: Vec3::ZERO,
                shininess: 2.0,
                base_color: Vec3::new(0.2, 0.7, 1.0),
                metallic: 0.0,
                roughness: 0.95,
                opacity: 0.22,
                ..Default::default()
            }),
        );
        graph.add_child(line, NodeData::Cube(CubeNode::default()));
    }
    for i in -grid_size..=grid_size {
        let x = i as f32 * spacing;
        let line =
            graph.add_child(overlay, NodeData::Separator(SeparatorNode));
        graph.add_child(
            line,
            NodeData::Transform(TransformNode {
                translation: Vec3::new(x, 0.01, 0.0),
                scale: Vec3::new(
                    0.01,
                    0.01,
                    grid_size as f32 * spacing * 2.0,
                ),
                ..Default::default()
            }),
        );
        graph.add_child(
            line,
            NodeData::Material(MaterialNode {
                diffuse_color: Vec3::new(0.2, 0.7, 1.0),
                ambient_color: Vec3::new(0.05, 0.15, 0.2),
                specular_color: Vec3::ZERO,
                shininess: 2.0,
                base_color: Vec3::new(0.2, 0.7, 1.0),
                metallic: 0.0,
                roughness: 0.95,
                opacity: 0.22,
                ..Default::default()
            }),
        );
        graph.add_child(line, NodeData::Cube(CubeNode::default()));
    }

    let axis_sep =
        graph.add_child(overlay, NodeData::Separator(SeparatorNode));
    graph.add_child(
        axis_sep,
        NodeData::Transform(TransformNode {
            translation: Vec3::new(0.0, 0.15, 0.0),
            scale: Vec3::new(0.08, 0.3, 0.08),
            ..Default::default()
        }),
    );
    graph.add_child(
        axis_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(1.0, 0.3, 0.3),
            ambient_color: Vec3::new(0.2, 0.06, 0.06),
            specular_color: Vec3::ZERO,
            shininess: 2.0,
            base_color: Vec3::new(1.0, 0.3, 0.3),
            metallic: 0.0,
            roughness: 0.95,
            opacity: 0.8,
            ..Default::default()
        }),
    );
    graph.add_child(axis_sep, NodeData::Cube(CubeNode::default()));

    overlay
}

fn has_node_type(
    graph: &SceneGraph,
    pred: impl Fn(&NodeData) -> bool,
) -> bool {
    for &root in graph.roots() {
        if has_node_type_recursive(graph, root, &pred) {
            return true;
        }
    }
    false
}

fn has_node_type_recursive(
    graph: &SceneGraph,
    node: NodeId,
    pred: &impl Fn(&NodeData) -> bool,
) -> bool {
    let Some(entry) = graph.get(node) else {
        return false;
    };
    if pred(&entry.data) {
        return true;
    }
    for &child in &entry.children {
        if has_node_type_recursive(graph, child, pred) {
            return true;
        }
    }
    false
}
