//! Automated render-effect toggle test harness (uses pre_render_hook path).
//!
//! This test mirrors the exact same code path as the interactive
//! stl_diagnostic_full: it applies effect toggles via a pre_render_hook
//! that runs after the tier system's reapply_cad_tier_constraints but
//! before GPU submission. Uses panic::catch_unwind to detect GPU crashes.
//!
//! Usage: cargo run -p rc3d-examples --example stl_diagnostic_full_test [file.stl]

use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_engine_api::Engine;
use rc3d_scene::node_data::*;

static TEST_LOG: Mutex<Vec<(String, bool)>> = Mutex::new(Vec::new());

fn log_result(name: &str, passed: bool) {
    let status = if passed { "PASS" } else { "FAIL" };
    println!("  [{status}] {name}");
    TEST_LOG.lock().unwrap().push((name.to_string(), passed));
}

/// Render N frames through Engine::render (which triggers pre_render_hook),
/// catching any GPU validation panics.
fn render_frames_safe(engine: &mut Engine, count: usize) -> bool {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        for _ in 0..count {
            engine.render();
        }
    }));
    if result.is_err() {
        return false;
    }
    // One more frame to catch deferred backend validation errors
    panic::catch_unwind(AssertUnwindSafe(|| {
        engine.render();
    }))
    .is_ok()
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let args: Vec<String> = std::env::args().collect();
    let stl_path = args.get(1).map(|s| s.as_str()).unwrap_or("test_data/car engine.stl");

    println!("=== stl_diagnostic_full Effect Toggle Test ===");
    println!("Scene: {stl_path}\n");

    // ── Create engine, load scene ──
    let event_loop = winit::event_loop::EventLoop::new().expect("event loop");
    let window = event_loop
        .create_window(winit::window::WindowAttributes::default().with_title("Effect Toggle Test"))
        .expect("window");

    let mut engine = Engine::new(&window);
    match engine.import(stl_path) {
        Ok(root_id) => {
            println!("Imported: root={root_id:?}");
            let g = engine.scene_mut();
            let has_cam = g.children(root_id).map(|c| c.iter().any(|&cid| {
                g.get(cid).map_or(false, |e| matches!(e.data,
                    NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_)))
            })).unwrap_or(false);
            if !has_cam {
                g.insert_child(root_id, 0, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                    Vec3::new(5.0, 5.0, 8.0), Vec3::ZERO, Vec3::Y,
                    std::f32::consts::FRAC_PI_4, 800.0 / 600.0,
                )));
                g.insert_child(root_id, 1, NodeData::DirectionalLight(DirectionalLightNode {
                    direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
                    color: Vec3::ONE, intensity: 1.5, light_group: None,
                }));
            }
            g.insert_child(root_id, 2, NodeData::Material(MaterialNode {
                base_color: Vec3::splat(0.9), roughness: 0.5, metallic: 0.0,
                ..Default::default()
            }));
        }
        Err(e) => { eprintln!("Import error: {e}"); std::process::exit(1); }
    }
    engine.set_display_mode(DisplayMode::ShadedWithEdges);
    engine.continuous_redraw = false;

    // ── Shared state: mirrors EffectToggles from stl_diagnostic_full ──
    let hdr = Arc::new(AtomicBool::new(false));
    let ssr = Arc::new(AtomicBool::new(false));
    let taa = Arc::new(AtomicBool::new(false));
    let motion_blur = Arc::new(AtomicBool::new(false));
    let dof = Arc::new(AtomicBool::new(false));
    let color_grading = Arc::new(AtomicBool::new(false));
    let volumetric_fog = Arc::new(AtomicBool::new(false));
    let wireframe_overlay = Arc::new(AtomicBool::new(false));
    let cluster_lights = Arc::new(AtomicBool::new(true));
    let omni_shadows = Arc::new(AtomicBool::new(false));
    let xray = Arc::new(AtomicBool::new(false));

    // Pre-render hook — same logic as stl_diagnostic_full
    {
        let hdr_h = hdr.clone(); let ssr_h = ssr.clone(); let taa_h = taa.clone();
        let mb_h = motion_blur.clone(); let dof_h = dof.clone(); let cg_h = color_grading.clone();
        let fog_h = volumetric_fog.clone(); let wf_h = wireframe_overlay.clone();
        let cl_h = cluster_lights.clone(); let os_h = omni_shadows.clone();
        let xr_h = xray.clone();

        engine.pre_render_hook = Some(Box::new(move |renderer| {
            let h = hdr_h.load(Ordering::Relaxed);
            renderer.set_hdr_post_processing(h);
            renderer.set_ssr(h && ssr_h.load(Ordering::Relaxed));
            renderer.set_taa(h && taa_h.load(Ordering::Relaxed));
            renderer.set_motion_blur(h && mb_h.load(Ordering::Relaxed));
            renderer.set_dof(h && dof_h.load(Ordering::Relaxed));
            renderer.set_color_grading(h && cg_h.load(Ordering::Relaxed));
            renderer.set_volumetric_fog(h && fog_h.load(Ordering::Relaxed));
            renderer.wireframe_overlay = wf_h.load(Ordering::Relaxed);
            renderer.set_cluster_lights(cl_h.load(Ordering::Relaxed));
            renderer.set_omni_shadows(os_h.load(Ordering::Relaxed));
            renderer.set_xray_mode(xr_h.load(Ordering::Relaxed));
            if renderer.wireframe_overlay {
                renderer.set_display_mode(DisplayMode::FlatWithEdge);
            } else {
                renderer.set_display_mode(DisplayMode::ShadedWithEdges);
            }
        }));
    }

    // Warm-up
    render_frames_safe(&mut engine, 2);

    // ── Helper: apply toggle set and test ──
    let set_hdr = |v: bool| hdr.store(v, Ordering::Relaxed);
    let set_ssr = |v: bool| ssr.store(v, Ordering::Relaxed);
    let set_taa = |v: bool| taa.store(v, Ordering::Relaxed);
    let set_mb = |v: bool| motion_blur.store(v, Ordering::Relaxed);
    let set_dof = |v: bool| dof.store(v, Ordering::Relaxed);
    let set_cg = |v: bool| color_grading.store(v, Ordering::Relaxed);
    let set_fog = |v: bool| volumetric_fog.store(v, Ordering::Relaxed);
    let set_wf = |v: bool| wireframe_overlay.store(v, Ordering::Relaxed);
    let set_cl = |v: bool| cluster_lights.store(v, Ordering::Relaxed);
    let set_os = |v: bool| omni_shadows.store(v, Ordering::Relaxed);
    let set_xr = |v: bool| xray.store(v, Ordering::Relaxed);

    fn reset_all(
        h: impl Fn(bool), ssr: impl Fn(bool), taa: impl Fn(bool),
        mb: impl Fn(bool), dof: impl Fn(bool), cg: impl Fn(bool),
        fog: impl Fn(bool), wf: impl Fn(bool), cl: impl Fn(bool),
        os: impl Fn(bool), xr: impl Fn(bool),
    ) {
        h(false); ssr(false); taa(false); mb(false); dof(false); cg(false);
        fog(false); wf(false); cl(true); os(false); xr(false);
    }
    macro_rules! reset {
        () => { reset_all(&set_hdr, &set_ssr, &set_taa, &set_mb, &set_dof, &set_cg, &set_fog, &set_wf, &set_cl, &set_os, &set_xr); };
    }

    // ══════════════════════════════════════════
    // Phase 1: Independent effects
    // ══════════════════════════════════════════
    println!("--- Phase 1: Independent Toggles ---");

    reset!(); set_hdr(true);
    log_result("HDR ON", render_frames_safe(&mut engine, 3));
    reset!(); // hdr=false
    log_result("HDR OFF", render_frames_safe(&mut engine, 3));

    reset!(); set_wf(true);
    log_result("WireframeOverlay ON", render_frames_safe(&mut engine, 3));
    reset!(); // wf=false
    log_result("WireframeOverlay OFF", render_frames_safe(&mut engine, 3));

    reset!(); set_cl(false);
    log_result("ClusterLights OFF", render_frames_safe(&mut engine, 3));
    reset!(); // cl=true
    log_result("ClusterLights ON", render_frames_safe(&mut engine, 3));

    reset!(); set_os(true);
    log_result("OmniShadows ON", render_frames_safe(&mut engine, 3));
    reset!();
    log_result("OmniShadows OFF", render_frames_safe(&mut engine, 3));

    reset!(); set_xr(true);
    log_result("XRay ON", render_frames_safe(&mut engine, 3));
    reset!();
    log_result("XRay OFF", render_frames_safe(&mut engine, 3));

    // ══════════════════════════════════════════
    // Phase 2: HDR-dependent effects
    // ══════════════════════════════════════════
    println!("--- Phase 2: HDR-Dependent Effects ---");

    // SSR
    reset!(); set_hdr(true); set_ssr(true);
    log_result("SSR ON (HDR=ON)", render_frames_safe(&mut engine, 3));
    set_ssr(false);
    log_result("SSR OFF (HDR=ON)", render_frames_safe(&mut engine, 2));
    reset!(); set_ssr(true); // HDR=OFF, pre_render_hook will disable
    log_result("SSR ON (HDR=OFF)", render_frames_safe(&mut engine, 3));

    // TAA
    reset!(); set_hdr(true); set_taa(true);
    log_result("TAA ON (HDR=ON)", render_frames_safe(&mut engine, 3));
    set_taa(false);
    log_result("TAA OFF (HDR=ON)", render_frames_safe(&mut engine, 2));
    reset!(); set_taa(true);
    log_result("TAA ON (HDR=OFF)", render_frames_safe(&mut engine, 3));

    // Motion Blur
    reset!(); set_hdr(true); set_mb(true);
    log_result("MotionBlur ON (HDR=ON)", render_frames_safe(&mut engine, 3));
    set_mb(false);
    log_result("MotionBlur OFF (HDR=ON)", render_frames_safe(&mut engine, 2));
    reset!(); set_mb(true);
    log_result("MotionBlur ON (HDR=OFF)", render_frames_safe(&mut engine, 3));

    // DOF
    reset!(); set_hdr(true); set_dof(true);
    log_result("DOF ON (HDR=ON)", render_frames_safe(&mut engine, 3));
    set_dof(false);
    log_result("DOF OFF (HDR=ON)", render_frames_safe(&mut engine, 2));
    reset!(); set_dof(true);
    log_result("DOF ON (HDR=OFF)", render_frames_safe(&mut engine, 3));

    // Color Grading
    reset!(); set_hdr(true); set_cg(true);
    log_result("ColorGrading ON (HDR=ON)", render_frames_safe(&mut engine, 3));
    set_cg(false);
    log_result("ColorGrading OFF (HDR=ON)", render_frames_safe(&mut engine, 2));
    reset!(); set_cg(true);
    log_result("ColorGrading ON (HDR=OFF)", render_frames_safe(&mut engine, 3));

    // Volumetric Fog
    reset!(); set_hdr(true); set_fog(true);
    log_result("VolumetricFog ON (HDR=ON)", render_frames_safe(&mut engine, 3));
    set_fog(false);
    log_result("VolumetricFog OFF (HDR=ON)", render_frames_safe(&mut engine, 2));
    reset!(); set_fog(true);
    log_result("VolumetricFog ON (HDR=OFF)", render_frames_safe(&mut engine, 3));

    // ══════════════════════════════════════════
    // Phase 3: Combinations
    // ══════════════════════════════════════════
    println!("--- Phase 3: Combinations ---");

    reset!(); set_hdr(true);
    set_ssr(true); set_taa(true); set_mb(true); set_dof(true);
    set_cg(true); set_fog(true); set_cl(false); set_os(true);
    log_result("All HDR effects ON + OmniShadow", render_frames_safe(&mut engine, 3));

    reset!(); set_hdr(true);
    set_ssr(true); set_taa(true); set_fog(true); set_wf(true);
    log_result("HDR+SSR+TAA+Fog+Wireframe", render_frames_safe(&mut engine, 3));

    reset!();
    log_result("All OFF (baseline)", render_frames_safe(&mut engine, 3));

    // ══════════════════════════════════════════
    // Summary
    // ══════════════════════════════════════════
    println!("\n=== Test Results Summary ===");
    let log = TEST_LOG.lock().unwrap();
    let total = log.len();
    let passed = log.iter().filter(|(_, p)| *p).count();
    let failed: Vec<&(String, bool)> = log.iter().filter(|(_, p)| !*p).collect();

    for (name, ok) in log.iter() {
        println!("  [{}] {name}", if *ok { "PASS" } else { "FAIL" });
    }
    println!("---\n{passed}/{total} passed");

    if !failed.is_empty() {
        println!("\nFAILED:");
        for (name, _) in &failed { println!("  - {name}"); }
        std::process::exit(1);
    } else {
        println!("All tests passed.");
    }

    let _ = event_loop.run(move |_event, elwt| { elwt.exit(); });
}
