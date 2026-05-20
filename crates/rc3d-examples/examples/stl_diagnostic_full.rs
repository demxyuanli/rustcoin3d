//! Full-effects rendering diagnostic — toggle every post-processing pass at runtime.
//!
//! Usage: stl_diagnostic_full <file.stl>
//!
//! Keys:
//!   F1  — HDR Post (master switch; disables dependent effects when off)
//!   F2  — SSR  (screen-space reflections)
//!   F3  — TAA  (temporal anti-aliasing)
//!   F4  — Motion Blur
//!   F5  — DOF  (depth of field)
//!   F6  — Color Grading
//!   F7  — Volumetric Fog
//!   F8  — Wireframe Overlay (full-mesh edges)
//!   F9  — Cluster Lights
//!   F10 — Omni Shadows
//!   F11 — X-Ray Mode
//!   ,/. — DOF focus distance -/+
//!   ;/' — DOF aperture -/+
//!   =/- — Bloom strength +/-
//!   [/] — Vignette +/-
//!
//! HUD shows the status of every toggle. Effects that require HDR
//! (SSR/TAA/MB/DOF/CG/Fog) are automatically disabled when HDR is off.

use std::env;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rc3d_actions::{fit_camera_to_scene, CameraFitConfig};
use rc3d_engine_api::{CameraController, Engine};
use rc3d_core::math::Vec3;
use rc3d_core::{DisplayMode, NodeId};
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowAttributes;

// Toggle atomics — one per controllable effect
struct EffectToggles {
    hdr: AtomicBool,
    ssr: AtomicBool,
    taa: AtomicBool,
    motion_blur: AtomicBool,
    dof: AtomicBool,
    color_grading: AtomicBool,
    volumetric_fog: AtomicBool,
    wireframe_overlay: AtomicBool,
    cluster_lights: AtomicBool,
    omni_shadows: AtomicBool,
    xray: AtomicBool,
}

impl Default for EffectToggles {
    fn default() -> Self {
        Self {
            hdr: AtomicBool::new(false),
            ssr: AtomicBool::new(false),
            taa: AtomicBool::new(false),
            motion_blur: AtomicBool::new(false),
            dof: AtomicBool::new(false),
            color_grading: AtomicBool::new(false),
            volumetric_fog: AtomicBool::new(false),
            wireframe_overlay: AtomicBool::new(false),
            cluster_lights: AtomicBool::new(true),
            omni_shadows: AtomicBool::new(false),
            xray: AtomicBool::new(false),
        }
    }
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    let args: Vec<String> = env::args().collect();
    let Some(path_arg) = args.iter().skip(1).find(|a| !a.starts_with("--")) else {
        eprintln!("Usage: stl_diagnostic_full <file.stl>");
        return;
    };
    let path = Path::new(path_arg);

    let t0 = Instant::now();
    let graph = match rc3d_io::import_file(path) {
        Ok(g) => {
            let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let node_count = count_nodes(&g);
            println!(
                "[FULL] Loaded {} in {:.0}ms ({} scene nodes)",
                path.display(),
                load_ms,
                node_count
            );
            g
        }
        Err(e) => {
            eprintln!("Import error: {e}");
            return;
        }
    };

    let mut graph = setup_scene(graph);
    let (target, orbit_radius) = fit_camera_to_scene(&mut graph, CameraFitConfig::default());
    let ctrl = CameraController::new(target, orbit_radius);

    // Shared state for cross-thread communication (hooks run in event loop)
    let toggles = Arc::new(EffectToggles::default());
    let tg_hook = toggles.clone();
    let tg_hud = toggles.clone();
    let tg_key = toggles.clone();

    // Bloom and vignette params (preserved across frames via atomics)
    let bloom_val = Arc::new(AtomicU32::new(8)); // 0.8 * 10
    let vignette_val = Arc::new(AtomicU32::new(3)); // 0.3 * 10
    let bloom_hook = bloom_val.clone();
    let vignette_hook = vignette_val.clone();
    let bloom_hud = bloom_val.clone();
    let vignette_hud = vignette_val.clone();
    let bloom_key = bloom_val;
    let vignette_key = vignette_val;

    // DOF params (stored as tenths: focus*10, aperture*10)
    let dof_focus_val = Arc::new(AtomicU32::new(50)); // 5.0 * 10
    let dof_aperture_val = Arc::new(AtomicU32::new(20)); // 2.0 * 10
    let dof_focus_hook = dof_focus_val.clone();
    let dof_aperture_hook = dof_aperture_val.clone();
    let dof_focus_hud = dof_focus_val.clone();
    let dof_aperture_hud = dof_aperture_val.clone();
    let dof_focus_key = dof_focus_val;
    let dof_aperture_key = dof_aperture_val;

    // GPU timing storage (written by pre_render_hook, read by HUD hook)
    let gpu_timings: Arc<Mutex<Vec<(String, f64)>>> = Arc::new(Mutex::new(Vec::new()));
    let gpu_timings_hook = gpu_timings.clone();
    let gpu_timings_hud = gpu_timings.clone();

    println!("[FULL] Keys: F1=HDR F2=SSR F3=TAA F4=MBlur F5=DOF F6=ColorGrading F7=VolFog");
    println!("[FULL]       F8=WireOverlay F9=ClusterLights F10=OmniShadow F11=XRay");
    println!("[FULL]       ,/.=DOFfocus ;/'=DOFaperture =/-=bloom [/]=vignette");

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = event_loop
        .create_window(WindowAttributes::default().with_title("Full Effects Diagnostic"))
        .expect("failed to create window");

    let mut engine = Engine::new(&window);
    engine.load_scene(graph);
    // Add point light as a root node so traversal visits it directly
    engine.scene_mut().add_root(NodeData::PointLight(PointLightNode {
        location: Vec3::new(3.0, 4.0, 5.0),
        color: Vec3::new(1.0, 0.3, 0.1),
        intensity: 5000.0,
        light_group: None,
    }));
    engine.controller = ctrl;
    engine.set_display_mode(DisplayMode::ShadedWithEdges);
    engine.continuous_redraw = true;

    // Pre-render hook: apply all effect toggles.
    // Runs after reapply_cad_tier_constraints (which sets tier_wants_shadow/edges from
    // the default Visualization tier) but before GPU submit. Since we never call
    // set_display_tier, cad_tier_authoritative stays false and the inner
    // render_draw_calls_core path won't re-apply tier config after this hook.
    engine.pre_render_hook = Some(Box::new(move |renderer| {
        let hdr = tg_hook.hdr.load(Ordering::Relaxed);

        // Master HDR switch (set_hdr_post_processing creates post_fx targets on demand)
        renderer.set_hdr_post_processing(hdr);

        // Effects that require HDR — auto-disable when HDR is off
        renderer.set_ssr(hdr && tg_hook.ssr.load(Ordering::Relaxed));
        renderer.set_taa(hdr && tg_hook.taa.load(Ordering::Relaxed));
        renderer.set_motion_blur(hdr && tg_hook.motion_blur.load(Ordering::Relaxed));
        renderer.set_dof(hdr && tg_hook.dof.load(Ordering::Relaxed));
        renderer.set_color_grading(hdr && tg_hook.color_grading.load(Ordering::Relaxed));
        renderer.set_volumetric_fog(hdr && tg_hook.volumetric_fog.load(Ordering::Relaxed));

        // Independent effects (pub fields / pub setters)
        renderer.wireframe_overlay = tg_hook.wireframe_overlay.load(Ordering::Relaxed);
        renderer.set_cluster_lights(tg_hook.cluster_lights.load(Ordering::Relaxed));
        renderer.set_omni_shadows(tg_hook.omni_shadows.load(Ordering::Relaxed));
        renderer.set_xray_mode(tg_hook.xray.load(Ordering::Relaxed));

        // Sync display mode: wireframe overlay forces FlatWithEdge
        if renderer.wireframe_overlay {
            renderer.set_display_mode(DisplayMode::FlatWithEdge);
        } else {
            renderer.set_display_mode(DisplayMode::ShadedWithEdges);
        }

        // Bloom & vignette params (via post_processor uniform)
        let bloom = bloom_hook.load(Ordering::Relaxed) as f32 / 10.0;
        let vig = vignette_hook.load(Ordering::Relaxed) as f32 / 10.0;
        renderer.set_post_effect_params(vig, 0.0, bloom, 0.0);

        // DOF params
        renderer.dof_focus_distance = dof_focus_hook.load(Ordering::Relaxed) as f32 / 10.0;
        renderer.dof_aperture = dof_aperture_hook.load(Ordering::Relaxed) as f32 / 10.0;

        // Collect GPU timings from previous frame for HUD display
        let ts = &renderer.gpu_timer.last_timestamps;
        let labels = &renderer.gpu_timer.labels;
        let period_ns = renderer.gpu_timer.timestamp_period_ns as f64;
        let mut timings = Vec::with_capacity(labels.len());
        for (i, label) in labels.iter().enumerate() {
            let bi = i * 2;
            if bi + 1 < ts.len() {
                let dur_ticks = ts[bi + 1].saturating_sub(ts[bi]);
                let dur_us = dur_ticks as f64 * period_ns / 1000.0;
                timings.push((label.to_string(), dur_us));
            }
        }
        if let Ok(mut g) = gpu_timings_hook.lock() {
            *g = timings;
        }
    }));

    // HUD overlay: live status of every toggle
    engine.hud_text_hook = Some(Box::new(move || {
        let hdr = tg_hud.hdr.load(Ordering::Relaxed);
        let on = |b: bool| if b { "ON " } else { "OFF" };
        // When HDR is off, dependent effects show "-" regardless of stored value
        let dep = |b: bool| {
            if !hdr {
                " - "
            } else if b {
                "ON "
            } else {
                "OFF"
            }
        };
        let bloom = bloom_hud.load(Ordering::Relaxed) as f32 / 10.0;
        let vig = vignette_hud.load(Ordering::Relaxed) as f32 / 10.0;
        let dof_focus = dof_focus_hud.load(Ordering::Relaxed) as f32 / 10.0;
        let dof_ap = dof_aperture_hud.load(Ordering::Relaxed) as f32 / 10.0;

        // Build GPU timing lines
        let gpu_lines: String = gpu_timings_hud
            .lock()
            .map(|timings| {
                let mut out = String::from(
                    "+--- GPU Timings (us) -------+\n",
                );
                for (label, us) in timings.iter() {
                    out.push_str(&format!(
                        "| {:<24} {:>5.0} |\n",
                        label, us
                    ));
                }
                out.push_str("+---------------------------+\n");
                out
            })
            .unwrap_or_default();

        format!(
            "+--- Full Effects Toggle ---+\n\
             | F1 HDR Post:       {}   |\n\
             | F2 SSR:            {}   |\n\
             | F3 TAA:            {}   |\n\
             | F4 Motion Blur:    {}   |\n\
             | F5 DOF:            {}   |\n\
             | F6 Color Grading:  {}   |\n\
             | F7 Volumetric Fog: {}   |\n\
             | F8 Wire Overlay:   {}   |\n\
             | F9 Cluster Lights: {}   |\n\
             |F10 Omni Shadows:   {}   |\n\
             |F11 X-Ray:          {}   |\n\
             | Bloom: {:.1}  Vig: {:.1}    |\n\
             | DOF fcs: {:.1}  DOF ap: {:.1} |\n\
             +---------------------------+\n\
             {gpu_lines}",
            on(hdr),
            dep(tg_hud.ssr.load(Ordering::Relaxed)),
            dep(tg_hud.taa.load(Ordering::Relaxed)),
            dep(tg_hud.motion_blur.load(Ordering::Relaxed)),
            dep(tg_hud.dof.load(Ordering::Relaxed)),
            dep(tg_hud.color_grading.load(Ordering::Relaxed)),
            dep(tg_hud.volumetric_fog.load(Ordering::Relaxed)),
            on(tg_hud.wireframe_overlay.load(Ordering::Relaxed)),
            on(tg_hud.cluster_lights.load(Ordering::Relaxed)),
            on(tg_hud.omni_shadows.load(Ordering::Relaxed)),
            on(tg_hud.xray.load(Ordering::Relaxed)),
            bloom,
            vig,
            dof_focus,
            dof_ap,
        )
    }));

    // Keyboard hook: F1-F11 toggle effects, =/- adjust bloom, [/] adjust vignette
    engine.panel_overlay_key_hook = Some(Box::new(move |key: PhysicalKey| -> bool {
        let code = match key {
            PhysicalKey::Code(c) => c,
            _ => return false,
        };
        match code {
            KeyCode::F1 => {
                let prev = tg_key.hdr.load(Ordering::Relaxed);
                tg_key.hdr.store(!prev, Ordering::Relaxed);
                println!("[FULL] HDR Post: {}", if !prev { "ON" } else { "OFF" });
                true
            }
            KeyCode::F2 => {
                let prev = tg_key.ssr.load(Ordering::Relaxed);
                tg_key.ssr.store(!prev, Ordering::Relaxed);
                println!("[FULL] SSR: {}", if !prev { "ON" } else { "OFF" });
                true
            }
            KeyCode::F3 => {
                let prev = tg_key.taa.load(Ordering::Relaxed);
                tg_key.taa.store(!prev, Ordering::Relaxed);
                println!("[FULL] TAA: {}", if !prev { "ON" } else { "OFF" });
                true
            }
            KeyCode::F4 => {
                let prev = tg_key.motion_blur.load(Ordering::Relaxed);
                tg_key.motion_blur.store(!prev, Ordering::Relaxed);
                println!("[FULL] Motion Blur: {}", if !prev { "ON" } else { "OFF" });
                true
            }
            KeyCode::F5 => {
                let prev = tg_key.dof.load(Ordering::Relaxed);
                tg_key.dof.store(!prev, Ordering::Relaxed);
                println!("[FULL] DOF: {}", if !prev { "ON" } else { "OFF" });
                true
            }
            KeyCode::F6 => {
                let prev = tg_key.color_grading.load(Ordering::Relaxed);
                tg_key.color_grading.store(!prev, Ordering::Relaxed);
                println!("[FULL] Color Grading: {}", if !prev { "ON" } else { "OFF" });
                true
            }
            KeyCode::F7 => {
                let prev = tg_key.volumetric_fog.load(Ordering::Relaxed);
                tg_key.volumetric_fog.store(!prev, Ordering::Relaxed);
                println!("[FULL] Volumetric Fog: {}", if !prev { "ON" } else { "OFF" });
                true
            }
            KeyCode::F8 => {
                let prev = tg_key.wireframe_overlay.load(Ordering::Relaxed);
                tg_key.wireframe_overlay.store(!prev, Ordering::Relaxed);
                println!(
                    "[FULL] Wireframe Overlay: {}",
                    if !prev { "ON" } else { "OFF" }
                );
                true
            }
            KeyCode::F9 => {
                let prev = tg_key.cluster_lights.load(Ordering::Relaxed);
                tg_key.cluster_lights.store(!prev, Ordering::Relaxed);
                println!(
                    "[FULL] Cluster Lights: {}",
                    if !prev { "ON" } else { "OFF" }
                );
                true
            }
            KeyCode::F10 => {
                let prev = tg_key.omni_shadows.load(Ordering::Relaxed);
                tg_key.omni_shadows.store(!prev, Ordering::Relaxed);
                println!(
                    "[FULL] Omni Shadows: {}",
                    if !prev { "ON" } else { "OFF" }
                );
                true
            }
            KeyCode::F11 => {
                let prev = tg_key.xray.load(Ordering::Relaxed);
                tg_key.xray.store(!prev, Ordering::Relaxed);
                println!("[FULL] X-Ray: {}", if !prev { "ON" } else { "OFF" });
                true
            }
            KeyCode::Equal => {
                let v = bloom_key.load(Ordering::Relaxed);
                if v < 20 {
                    bloom_key.store(v + 1, Ordering::Relaxed);
                    println!("[FULL] Bloom: {:.1}", (v + 1) as f32 / 10.0);
                }
                true
            }
            KeyCode::Minus => {
                let v = bloom_key.load(Ordering::Relaxed);
                if v > 0 {
                    bloom_key.store(v - 1, Ordering::Relaxed);
                    println!("[FULL] Bloom: {:.1}", (v.saturating_sub(1)) as f32 / 10.0);
                }
                true
            }
            KeyCode::BracketLeft => {
                let v = vignette_key.load(Ordering::Relaxed);
                if v > 0 {
                    vignette_key.store(v - 1, Ordering::Relaxed);
                    println!(
                        "[FULL] Vignette: {:.1}",
                        (v.saturating_sub(1)) as f32 / 10.0
                    );
                }
                true
            }
            KeyCode::BracketRight => {
                let v = vignette_key.load(Ordering::Relaxed);
                if v < 20 {
                    vignette_key.store(v + 1, Ordering::Relaxed);
                    println!("[FULL] Vignette: {:.1}", (v + 1) as f32 / 10.0);
                }
                true
            }
            // DOF focus: Comma/Period (decrease/increase)
            KeyCode::Comma => {
                let v = dof_focus_key.load(Ordering::Relaxed);
                if v > 10 {
                    dof_focus_key.store(v - 1, Ordering::Relaxed);
                    println!("[FULL] DOF focus: {:.1}", (v.saturating_sub(1)) as f32 / 10.0);
                }
                true
            }
            KeyCode::Period => {
                let v = dof_focus_key.load(Ordering::Relaxed);
                if v < 500 {
                    dof_focus_key.store(v + 1, Ordering::Relaxed);
                    println!("[FULL] DOF focus: {:.1}", (v + 1) as f32 / 10.0);
                }
                true
            }
            // DOF aperture: Semicolon/Quote (decrease/increase)
            KeyCode::Semicolon => {
                let v = dof_aperture_key.load(Ordering::Relaxed);
                if v > 5 {
                    dof_aperture_key.store(v - 1, Ordering::Relaxed);
                    println!("[FULL] DOF aperture: {:.1}", (v.saturating_sub(1)) as f32 / 10.0);
                }
                true
            }
            KeyCode::Quote => {
                let v = dof_aperture_key.load(Ordering::Relaxed);
                if v < 100 {
                    dof_aperture_key.store(v + 1, Ordering::Relaxed);
                    println!("[FULL] DOF aperture: {:.1}", (v + 1) as f32 / 10.0);
                }
                true
            }
            _ => false,
        }
    }));

    let mut cursor_prev: (f64, f64) = (0.0, 0.0);

    let _ = event_loop.run(move |event, elwt| {
        match &event {
            Event::WindowEvent { event: win_event, .. } => match win_event {
                WindowEvent::RedrawRequested => {
                    engine.render();
                    window.request_redraw();
                }
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(size) => {
                    engine.resize(size.width, size.height);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let left_orbit = engine.on_pick.is_none();
                    engine.controller.dispatch_window_event(
                        win_event,
                        cursor_prev,
                        left_orbit,
                    );
                    cursor_prev = (position.x, position.y);
                }
                WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                    let left_orbit = engine.on_pick.is_none();
                    engine.controller.dispatch_window_event(
                        win_event,
                        cursor_prev,
                        left_orbit,
                    );
                    if let WindowEvent::MouseWheel { .. } = win_event {
                        window.request_redraw();
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed {
                        if let Some(ref mut hook) = engine.panel_overlay_key_hook {
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
        }
    });
}

// ── Scene setup ──

fn setup_scene(mut graph: SceneGraph) -> SceneGraph {
    let target_root = find_geometry_root(&graph);

    let has_camera = has_node_type_recursive(&graph, target_root, |d| {
        matches!(
            d,
            NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_)
        )
    });
    if !has_camera {
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
        graph.insert_child(
            target_root,
            1,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
                color: Vec3::ONE,
                intensity: 2.2,
                light_group: None,
            }),
        );
        graph.insert_child(
            target_root,
            2,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(1.0, -0.6, 0.8).normalize(),
                color: Vec3::new(0.9, 0.92, 1.0),
                intensity: 1.2,
                light_group: None,
            }),
        );
        graph.insert_child(
            target_root,
            3,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(0.3, 0.8, 1.0).normalize(),
                color: Vec3::new(0.85, 0.88, 0.95),
                intensity: 0.8,
                light_group: None,
            }),
        );
    }

    let has_material = has_node_type_recursive(&graph, target_root, |d| {
        matches!(d, NodeData::Material(_))
    });
    if !has_material {
        let insert_idx = if has_camera { 0 } else { 4 };
        graph.insert_child(
            target_root,
            insert_idx,
            NodeData::Material(MaterialNode {
                diffuse_color: Vec3::splat(0.9),
                ambient_color: Vec3::splat(0.4),
                specular_color: Vec3::splat(0.5),
                shininess: 48.0,
                base_color: Vec3::splat(0.92),
                metallic: 0.0,
                roughness: 0.4,
                opacity: 1.0,
                ..Default::default()
            }),
        );
    }

    for &root in graph.roots().to_vec().iter() {
        boost_materials(&mut graph, root);
    }
    graph
}

fn boost_materials(graph: &mut SceneGraph, node: NodeId) {
    let children = graph.children(node).unwrap_or(&[]).to_vec();
    if let Some(entry) = graph.get_mut(node) {
        if let NodeData::Material(mat) = &mut entry.data {
            mat.base_color = mat.base_color.max(Vec3::splat(0.75));
            mat.ambient_color = mat.ambient_color.max(Vec3::splat(0.35));
            mat.roughness = mat.roughness.min(0.65);
        }
        if let NodeData::DirectionalLight(light) = &mut entry.data {
            light.intensity = light.intensity.max(2.0);
        }
    }
    for child in children {
        boost_materials(graph, child);
    }
}

fn find_geometry_root(graph: &SceneGraph) -> NodeId {
    for &root in graph.roots() {
        if let Some(entry) = graph.get(root) {
            if matches!(entry.data, NodeData::Separator(_)) {
                return root;
            }
        }
    }
    graph.roots()[0]
}

fn has_node_type_recursive(
    graph: &SceneGraph,
    node: NodeId,
    pred: impl Fn(&NodeData) -> bool + Copy,
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

fn count_nodes(graph: &SceneGraph) -> usize {
    let mut count = 0;
    for &root in graph.roots() {
        count += count_nodes_recursive(graph, root);
    }
    count
}

fn count_nodes_recursive(graph: &SceneGraph, node: NodeId) -> usize {
    let Some(entry) = graph.get(node) else {
        return 0;
    };
    let mut n = 1;
    for &child in &entry.children {
        n += count_nodes_recursive(graph, child);
    }
    n
}
