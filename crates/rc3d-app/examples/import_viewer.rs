use std::env;
use std::path::Path;
use std::sync::{Arc, Mutex};

use rc3d_actions::{fit_camera_to_scene, CameraFitConfig};
use rc3d_app::camera_controller::CameraController;
use rc3d_app::control_panel::{preset_for_import_viewer_panel, RenderFeaturePanelState};
use rc3d_app::{AdaptiveQualityMode, App};
use rc3d_core::NodeId;
use rc3d_core::{
    math::{Mat4, Vec3},
    DisplayMode,
};
use rc3d_engine::{Engine, EngineRegistry};
use rc3d_scene::node_data::*;
use winit::keyboard::KeyCode;

#[derive(Debug)]
struct ModelAnimationEngine {
    target_transform: NodeId,
    debug_overlay_root: NodeId,
    panel: Arc<Mutex<RenderFeaturePanelState>>,
    playback_time: f32,
    last_time: Option<f64>,
}

impl ModelAnimationEngine {
    fn new(
        target_transform: NodeId,
        debug_overlay_root: NodeId,
        panel: Arc<Mutex<RenderFeaturePanelState>>,
    ) -> Self {
        Self {
            target_transform,
            debug_overlay_root,
            panel,
            playback_time: 0.0,
            last_time: None,
        }
    }
}

impl Engine for ModelAnimationEngine {
    fn evaluate(&mut self, graph: &mut rc3d_scene::SceneGraph, time: f64) {
        let mut s = self.panel.lock().expect("panel state lock");
        let dt = if let Some(prev) = self.last_time {
            (time - prev).max(0.0) as f32
        } else {
            0.0
        };
        self.last_time = Some(time);
        let step_dt = s.advance_time(dt);
        self.playback_time += step_dt;

        let t = self.playback_time;
        let channel_a_wave = (t * 1.2).sin() * 0.02;
        let channel_b_wave = (t * 3.0).sin();
        let channel_c_wave = (t * 6.0).sin();
        let tx =
            s.channel_b_weight * channel_b_wave * 0.15 + s.channel_c_weight * channel_c_wave * 0.32;
        let ty = s.channel_a_weight * channel_a_wave
            + s.channel_b_weight * channel_b_wave.abs() * 0.04
            + s.channel_c_weight * channel_c_wave.abs() * 0.08;
        let yaw = s.channel_b_weight * (t * 1.2).sin() * 0.2
            + s.channel_c_weight * (t * 2.4).sin() * 0.45;
        let scale = if s.show_model {
            Vec3::ONE
        } else {
            Vec3::splat(0.0001)
        };
        let debug_scale = if s.show_debug_overlay {
            Vec3::ONE
        } else {
            Vec3::splat(0.0001)
        };
        drop(s);

        if let Some(entry) = graph.get_mut(self.target_transform) {
            if let NodeData::Transform(tr) = &mut entry.data {
                tr.translation = Vec3::new(tx, ty, 0.0);
                tr.rotation = Mat4::from_rotation_y(yaw);
                tr.scale = scale;
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
    print_import_viewer_help();

    let args: Vec<String> = env::args().collect();
    let no_panel = args.iter().skip(1).any(|arg| arg == "--no-panel");
    let adaptive_mode = parse_adaptive_quality_mode(&args);
    let Some(path_arg) = args.iter().skip(1).find(|arg| !arg.starts_with("--")) else {
        eprintln!("Usage: import_viewer <file.stl|file.obj|file.iv> [--high-contrast=on|off]");
        return;
    };

    let path = Path::new(path_arg);
    let high_contrast = parse_high_contrast(&args, path);
    let graph = match rc3d_io::import_file(path) {
        Ok(g) => {
            println!("Loaded: {}", path.display());
            g
        }
        Err(e) => {
            eprintln!("Import error: {e}");
            return;
        }
    };

    let mut graph = ensure_camera_and_light(graph, high_contrast);
    let anim_root = insert_animation_root(&mut graph);
    let (target, orbit_radius) = fit_camera_to_scene(&mut graph, CameraFitConfig::default());
    let debug_overlay_root =
        attach_import_debug_overlay(&mut graph, anim_root, target, orbit_radius);
    let _controller_root = find_first_camera_node(&graph).unwrap_or(graph.roots()[0]);
    let ctrl = CameraController::new(target, orbit_radius);
    let event_loop =
        winit::event_loop::EventLoop::new().expect("failed to create import_viewer event loop");
    let panel_preset = preset_for_import_viewer_panel();
    let panel_title = panel_preset.config.title.clone();
    let panel_state = Arc::new(Mutex::new(panel_preset.state));
    if no_panel {
        println!("Panel: disabled | title: {panel_title}");
    } else {
        println!("Panel: enabled (embedded) | title: {panel_title}");
    }
    println!("Adaptive quality mode: {:?}", adaptive_mode);
    let mut engines = EngineRegistry::new();
    engines.add(ModelAnimationEngine::new(
        anim_root,
        debug_overlay_root,
        panel_state.clone(),
    ));
    let mut app = App::new(graph)
        .with_camera_controller(ctrl)
        .with_engines(engines)
        .with_initial_display_mode(DisplayMode::ShadedWithEdges)
        .with_hdr_post_processing(true)
        .with_adaptive_quality_mode(adaptive_mode);
    if !no_panel {
        let panel_for_text = panel_state.clone();
        let panel_for_key = panel_state.clone();
        let selected = Arc::new(Mutex::new(0usize));
        let selected_for_text = selected.clone();
        let selected_for_key = selected.clone();
        let selected_for_mouse = selected.clone();
        let panel_for_mouse = panel_for_key.clone();
        app = app
            .with_panel_overlay_text_hook(move || {
                let sel = *selected_for_text.lock().expect("panel selected lock");
                build_import_viewer_panel_overlay_with_selection(&panel_for_text, sel)
            })
            .with_panel_overlay_key_hook(move |key| {
                apply_import_viewer_panel_key(&panel_for_key, &selected_for_key, key)
            })
            .with_panel_overlay_mouse_hook(move |x, y, w, h| {
                apply_import_viewer_panel_mouse(&panel_for_mouse, &selected_for_mouse, x, y, w, h)
            });
    }
    event_loop.run_app(&mut app).expect("event loop error");
}

fn build_import_viewer_panel_overlay_with_selection(
    state: &Arc<Mutex<RenderFeaturePanelState>>,
    selected: usize,
) -> String {
    let s = state.lock().expect("panel state lock");
    let mark = |idx: usize| if idx == selected { ">" } else { " " };
    let check = |v: bool| if v { "[x]" } else { "[ ]" };
    let slider = make_slider(s.time_scale, 0.0, 3.0, 16);
    format!(
        "+-- Embedded Control Panel ------------------+\n\
{} {} Show Model\n\
{} {} Show Debug Overlay\n\
{} {} Channel A\n\
{} {} Channel B\n\
{} {} Channel C\n\
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
        slider,
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

fn apply_import_viewer_panel_key(
    state: &Arc<Mutex<RenderFeaturePanelState>>,
    selected: &Arc<Mutex<usize>>,
    key: KeyCode,
) {
    let mut sel = selected.lock().expect("panel selected lock");
    let item_count = 6usize;
    match key {
        KeyCode::ArrowUp => {
            *sel = (*sel + item_count - 1) % item_count;
            return;
        }
        KeyCode::ArrowDown => {
            *sel = (*sel + 1) % item_count;
            return;
        }
        _ => {}
    }
    let mut s = state.lock().expect("panel state lock");
    match key {
        KeyCode::F5 => s.show_model = !s.show_model,
        KeyCode::F6 => s.show_debug_overlay = !s.show_debug_overlay,
        KeyCode::F7 => s.channel_a_active = !s.channel_a_active,
        KeyCode::F8 => s.channel_b_active = !s.channel_b_active,
        KeyCode::F9 => s.channel_c_active = !s.channel_c_active,
        KeyCode::BracketLeft => s.time_scale = (s.time_scale - 0.1).max(0.0),
        KeyCode::BracketRight => s.time_scale = (s.time_scale + 0.1).min(3.0),
        KeyCode::Enter | KeyCode::Space => match *sel {
            0 => s.show_model = !s.show_model,
            1 => s.show_debug_overlay = !s.show_debug_overlay,
            2 => s.channel_a_active = !s.channel_a_active,
            3 => s.channel_b_active = !s.channel_b_active,
            4 => s.channel_c_active = !s.channel_c_active,
            _ => {}
        },
        KeyCode::ArrowLeft => {
            if *sel == 5 {
                s.time_scale = (s.time_scale - 0.1).max(0.0);
            }
        }
        KeyCode::ArrowRight => {
            if *sel == 5 {
                s.time_scale = (s.time_scale + 0.1).min(3.0);
            }
        }
        _ => {}
    }
}

fn apply_import_viewer_panel_mouse(
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
    const ITEM_LINES: [i32; 6] = [1, 2, 3, 4, 5, 6];
    let line = ((y - HUD_TOP) / LINE_H).floor() as i32;
    let rel = line - PANEL_START_LINE;
    let Some(idx) = ITEM_LINES.iter().position(|v| *v == rel) else {
        return false;
    };
    *selected.lock().expect("panel selected lock") = idx;
    let mut s = state.lock().expect("panel state lock");
    match idx {
        0 => s.show_model = !s.show_model,
        1 => s.show_debug_overlay = !s.show_debug_overlay,
        2 => s.channel_a_active = !s.channel_a_active,
        3 => s.channel_b_active = !s.channel_b_active,
        4 => s.channel_c_active = !s.channel_c_active,
        5 => {
            let slider_left = HUD_LEFT + 240.0;
            let slider_right = slider_left + 120.0;
            if x >= slider_left && x <= slider_right {
                let t = rc3d_core::utils::math::remap(x, slider_left, slider_right);
                s.time_scale = t * 3.0;
            } else {
                s.time_scale = (s.time_scale + 0.1).min(3.0);
            }
        }
        _ => {}
    }
    true
}

fn insert_animation_root(graph: &mut rc3d_scene::SceneGraph) -> NodeId {
    let scene_root = graph.add_root(NodeData::Separator(SeparatorNode));
    graph.add_child(scene_root, NodeData::Transform(TransformNode::default()));
    let old_roots: Vec<NodeId> = graph
        .roots()
        .iter()
        .copied()
        .filter(|r| *r != scene_root)
        .collect();

    fn clone_subtree(
        src: &rc3d_scene::SceneGraph,
        dst: &mut rc3d_scene::SceneGraph,
        src_id: NodeId,
        parent: NodeId,
    ) {
        let Some(src_entry) = src.get(src_id) else {
            return;
        };
        let new_id = dst.add_child(parent, src_entry.data.clone());
        if let Some(dst_entry) = dst.get_mut(new_id) {
            dst_entry.name = src_entry.name.clone();
            dst_entry.display_mode = src_entry.display_mode;
        }
        for &child in &src_entry.children {
            clone_subtree(src, dst, child, new_id);
        }
    }

    let snapshot = std::mem::take(graph);
    let mut rebuilt = rc3d_scene::SceneGraph::new();
    let rebuilt_scene_root = rebuilt.add_root(NodeData::Separator(SeparatorNode));
    let rebuilt_anim_root = rebuilt.add_child(
        rebuilt_scene_root,
        NodeData::Transform(TransformNode::default()),
    );
    for root in old_roots {
        clone_subtree(&snapshot, &mut rebuilt, root, rebuilt_anim_root);
    }
    *graph = rebuilt;
    rebuilt_anim_root
}

fn attach_import_debug_overlay(
    graph: &mut rc3d_scene::SceneGraph,
    parent: NodeId,
    center: Vec3,
    orbit_radius: f32,
) -> NodeId {
    let root = graph.add_child(parent, NodeData::Separator(SeparatorNode));
    let overlay_root = graph.add_child(
        root,
        NodeData::Transform(TransformNode::from_translation(center)),
    );
    let axis_len = orbit_radius.max(1.0) * 0.45;
    let axis_thickness = (axis_len * 0.02).max(0.01);
    let box_extent = axis_len * 0.7;

    // X axis helper
    let x_sep = graph.add_child(overlay_root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        x_sep,
        NodeData::Transform(TransformNode {
            translation: Vec3::new(axis_len * 0.5, 0.0, 0.0),
            scale: Vec3::new(axis_len, axis_thickness, axis_thickness),
            ..Default::default()
        }),
    );
    graph.add_child(
        x_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(1.0, 0.25, 0.25),
            ambient_color: Vec3::new(0.2, 0.05, 0.05),
            specular_color: Vec3::ZERO,
            shininess: 2.0,
            base_color: Vec3::new(1.0, 0.25, 0.25),
            metallic: 0.0,
            roughness: 0.9,
            opacity: 0.65,
            ..Default::default()
        }),
    );
    graph.add_child(x_sep, NodeData::Cube(CubeNode::default()));

    // Y axis helper
    let y_sep = graph.add_child(overlay_root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        y_sep,
        NodeData::Transform(TransformNode {
            translation: Vec3::new(0.0, axis_len * 0.5, 0.0),
            scale: Vec3::new(axis_thickness, axis_len, axis_thickness),
            ..Default::default()
        }),
    );
    graph.add_child(
        y_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.25, 1.0, 0.25),
            ambient_color: Vec3::new(0.05, 0.2, 0.05),
            specular_color: Vec3::ZERO,
            shininess: 2.0,
            base_color: Vec3::new(0.25, 1.0, 0.25),
            metallic: 0.0,
            roughness: 0.9,
            opacity: 0.65,
            ..Default::default()
        }),
    );
    graph.add_child(y_sep, NodeData::Cube(CubeNode::default()));

    // Z axis helper
    let z_sep = graph.add_child(overlay_root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        z_sep,
        NodeData::Transform(TransformNode {
            translation: Vec3::new(0.0, 0.0, axis_len * 0.5),
            scale: Vec3::new(axis_thickness, axis_thickness, axis_len),
            ..Default::default()
        }),
    );
    graph.add_child(
        z_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.25, 0.45, 1.0),
            ambient_color: Vec3::new(0.05, 0.1, 0.2),
            specular_color: Vec3::ZERO,
            shininess: 2.0,
            base_color: Vec3::new(0.25, 0.45, 1.0),
            metallic: 0.0,
            roughness: 0.9,
            opacity: 0.65,
            ..Default::default()
        }),
    );
    graph.add_child(z_sep, NodeData::Cube(CubeNode::default()));

    // Bounding helper cube
    let b_sep = graph.add_child(overlay_root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        b_sep,
        NodeData::Transform(TransformNode {
            translation: Vec3::ZERO,
            scale: Vec3::splat(box_extent),
            ..Default::default()
        }),
    );
    graph.add_child(
        b_sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(1.0, 0.9, 0.3),
            ambient_color: Vec3::new(0.2, 0.18, 0.06),
            specular_color: Vec3::ZERO,
            shininess: 2.0,
            base_color: Vec3::new(1.0, 0.9, 0.3),
            metallic: 0.0,
            roughness: 0.95,
            opacity: 0.18,
            ..Default::default()
        }),
    );
    graph.add_child(b_sep, NodeData::Cube(CubeNode::default()));

    overlay_root
}

fn print_import_viewer_help() {
    println!("Import viewer example");
    println!("Usage: cargo run -p rc3d-app --example import_viewer -- <file.stl|file.obj|file.iv> [--high-contrast=on|off] [--no-panel] [--adaptive-quality=off|on|auto-idle-lock]");
    println!("Controls:");
    println!("  Mouse drag: orbit camera");
    println!("  Embedded panel HUD: clickable checkbox/slider + F5/F6/F7/F8/F9 and [ / ]");
    println!("  Escape: clear selection | Close window to quit");
    println!("Feature switches:");
    println!("  --high-contrast=on|off");
    println!("  --no-panel");
    println!("  --adaptive-quality=off|on|auto-idle-lock");
    println!("  Auto camera-fit, auto light/material completion");
    println!("  Panel: visibility (model/debug overlay), channel enable, update control, channel transition/mix, global rate");
}

fn parse_high_contrast(args: &[String], path: &Path) -> bool {
    for arg in args.iter().skip(2) {
        if let Some(value) = arg.strip_prefix("--high-contrast=") {
            return matches!(value, "on" | "true" | "1");
        }
    }
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("stl"))
        .unwrap_or(false)
}

fn parse_adaptive_quality_mode(args: &[String]) -> AdaptiveQualityMode {
    for arg in args.iter().skip(1) {
        if let Some(v) = arg.strip_prefix("--adaptive-quality=") {
            return match v {
                "off" => AdaptiveQualityMode::Off,
                "on" => AdaptiveQualityMode::On,
                "auto-idle-lock" => AdaptiveQualityMode::AutoIdleLock,
                _ => AdaptiveQualityMode::AutoIdleLock,
            };
        }
    }
    AdaptiveQualityMode::AutoIdleLock
}

/// Find the first Separator root that contains geometry, or the first root.
fn find_geometry_root(graph: &rc3d_scene::SceneGraph) -> NodeId {
    let roots = graph.roots();
    for &root in roots {
        if let Some(entry) = graph.get(root) {
            if matches!(entry.data, NodeData::Separator(_)) && has_geometry_recursive(graph, root) {
                return root;
            }
        }
    }
    roots[0]
}

fn has_geometry_recursive(graph: &rc3d_scene::SceneGraph, node: NodeId) -> bool {
    let Some(entry) = graph.get(node) else {
        return false;
    };
    match &entry.data {
        NodeData::Coordinate3(_)
        | NodeData::TextureCoordinate2(_)
        | NodeData::IndexedFaceSet(_)
        | NodeData::Cube(_)
        | NodeData::Sphere(_)
        | NodeData::Cone(_)
        | NodeData::Cylinder(_)
        | NodeData::Triangle(_) => return true,
        _ => {}
    }
    for &child in &entry.children {
        if has_geometry_recursive(graph, child) {
            return true;
        }
    }
    false
}

fn ensure_camera_and_light(
    mut graph: rc3d_scene::SceneGraph,
    high_contrast: bool,
) -> rc3d_scene::SceneGraph {
    let has_camera = graph
        .roots()
        .iter()
        .any(|&root| has_camera_recursive(&graph, root));

    if !has_camera {
        // Insert camera + light at index 0 so they are visited BEFORE geometry.
        // RenderCollector processes children in order; VP matrix must be set first.
        let target_root = find_geometry_root(&graph);
        graph.insert_child(
            target_root,
            0,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(5.0, 5.0, 8.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );
        // Main key light: upper-left-front
        graph.insert_child(
            target_root,
            1,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
                color: Vec3::ONE,
                intensity: 1.0,
                light_group: None,
            }),
        );
        // Fill light: upper-right, softer intensity to brighten shadows
        graph.insert_child(
            target_root,
            2,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(1.0, -0.6, 0.8).normalize(),
                color: Vec3::new(0.9, 0.92, 1.0),
                intensity: 0.5,
                light_group: None,
            }),
        );
        // Rim/back-fill light: illuminates rear and underside faces
        graph.insert_child(
            target_root,
            3,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(0.3, 0.8, 1.0).normalize(),
                color: Vec3::new(0.85, 0.88, 0.95),
                intensity: 0.4,
                light_group: None,
            }),
        );
    }

    let has_material = graph
        .roots()
        .iter()
        .any(|&root| has_material_recursive(&graph, root));
    if !has_material {
        let target_root = find_geometry_root(&graph);
        let cam_count = if has_camera { 0 } else { 4 };
        graph.insert_child(
            target_root,
            cam_count,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.7, 0.7, 0.7))),
        );
    }

    if high_contrast {
        apply_high_contrast_mode(&mut graph);
        log::info!("High-contrast import mode enabled");
    }

    graph
}

fn apply_high_contrast_mode(graph: &mut rc3d_scene::SceneGraph) {
    if !has_directional_light(graph) {
        let target_root = find_geometry_root(graph);
        graph.insert_child(
            target_root,
            0,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -0.8, -0.6).normalize(),
                color: Vec3::ONE,
                intensity: 2.4,
                light_group: None,
            }),
        );
        graph.insert_child(
            target_root,
            1,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(1.0, -0.5, 0.7).normalize(),
                color: Vec3::new(0.9, 0.92, 1.0),
                intensity: 1.0,
                light_group: None,
            }),
        );
    }
    for &root in graph.roots().to_vec().iter() {
        boost_contrast_recursive(graph, root);
    }
}

fn has_directional_light(graph: &rc3d_scene::SceneGraph) -> bool {
    for &root in graph.roots() {
        if has_directional_light_recursive(graph, root) {
            return true;
        }
    }
    false
}

fn has_directional_light_recursive(graph: &rc3d_scene::SceneGraph, node: NodeId) -> bool {
    let Some(entry) = graph.get(node) else {
        return false;
    };
    if matches!(entry.data, NodeData::DirectionalLight(_)) {
        return true;
    }
    for &child in &entry.children {
        if has_directional_light_recursive(graph, child) {
            return true;
        }
    }
    false
}

fn boost_contrast_recursive(graph: &mut rc3d_scene::SceneGraph, node: NodeId) {
    let children = graph.children(node).unwrap_or(&[]).to_vec();
    if let Some(entry) = graph.get_mut(node) {
        match &mut entry.data {
            NodeData::DirectionalLight(light) => {
                light.intensity = light.intensity.max(2.2);
                light.color = Vec3::ONE;
            }
            NodeData::Material(mat) => {
                mat.diffuse_color = mat.diffuse_color.max(Vec3::splat(0.75));
                mat.base_color = mat.base_color.max(Vec3::splat(0.75));
                mat.ambient_color = mat.ambient_color.max(Vec3::splat(0.4));
                mat.specular_color = mat.specular_color.max(Vec3::splat(0.6));
                mat.shininess = mat.shininess.max(48.0);
                mat.roughness = mat.roughness.min(0.65);
            }
            _ => {}
        }
    }
    for child in children {
        boost_contrast_recursive(graph, child);
    }
}

fn find_first_camera_node(graph: &rc3d_scene::SceneGraph) -> Option<NodeId> {
    for &root in graph.roots() {
        if let Some(id) = find_camera_recursive(graph, root) {
            return Some(id);
        }
    }
    None
}

fn find_camera_recursive(graph: &rc3d_scene::SceneGraph, node: NodeId) -> Option<NodeId> {
    let entry = graph.get(node)?;
    if matches!(
        entry.data,
        NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_)
    ) {
        return Some(node);
    }
    for &child in &entry.children {
        if let Some(id) = find_camera_recursive(graph, child) {
            return Some(id);
        }
    }
    None
}

fn has_camera_recursive(graph: &rc3d_scene::SceneGraph, node: rc3d_core::NodeId) -> bool {
    let Some(entry) = graph.get(node) else {
        return false;
    };
    match &entry.data {
        NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_) => return true,
        _ => {}
    }
    for &child in &entry.children {
        if has_camera_recursive(graph, child) {
            return true;
        }
    }
    false
}

fn has_material_recursive(graph: &rc3d_scene::SceneGraph, node: rc3d_core::NodeId) -> bool {
    let Some(entry) = graph.get(node) else {
        return false;
    };
    if matches!(entry.data, NodeData::Material(_)) {
        return true;
    }
    for &child in &entry.children {
        if has_material_recursive(graph, child) {
            return true;
        }
    }
    false
}
