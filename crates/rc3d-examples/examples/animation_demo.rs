//! Animation demo — engine-driven transform animation.
//!
//! Keys: B — cycle background mode (Solid/Gradient/Horizontal/Image)
//!
//! Usage: cargo run -p rc3d-examples --example animation_demo

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use rc3d_core::math::Vec3;
use rc3d_engine::{ElapsedTimeEngine, EngineRegistry, InterpolateVec3Engine, SineField, SineOscillatorEngine};
use rc3d_engine_api::background::BackgroundSettings;
use rc3d_examples::common::run_example_with_hooks;
use rc3d_render::background::{BgMode, ImageFit};
use rc3d_scene::node_data::*;
use winit::keyboard::{KeyCode, PhysicalKey};

fn main() {
    let bg_mode = Arc::new(AtomicU32::new(1)); // start at VerticalGradient (1)
    let bg_key = bg_mode.clone();
    let bg_hook = bg_mode.clone();

    run_example_with_hooks("Animation Demo", |engine| {
        engine.set_background(BackgroundSettings {
            mode: BgMode::VerticalGradient,
            top_color: [0.02, 0.10, 0.35, 1.0],
            bot_color: [0.02, 0.02, 0.08, 1.0],
            ..Default::default()
        });

        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        // Camera
        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(4.0, 3.0, 6.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        // Directional light
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 0.8,
            light_group: None,
        }));

        // Orbiting point lights
        let light1_id = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(3.0, 2.0, 0.0))),
        );
        graph.add_child(light1_id, NodeData::PointLight(PointLightNode {
            location: Vec3::ZERO,
            color: Vec3::new(1.0, 0.3, 0.2),
            intensity: 15.0,
            light_group: None,
        }));
        let light2_id = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(-3.0, 2.0, 0.0))),
        );
        graph.add_child(light2_id, NodeData::PointLight(PointLightNode {
            location: Vec3::ZERO,
            color: Vec3::new(0.2, 0.5, 1.0),
            intensity: 15.0,
            light_group: None,
        }));

        // Ground plane
        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.3, 0.3, 0.35),
            roughness: 0.8,
            ..Default::default()
        }));
        let ground = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, -0.55, 0.0))),
        );
        graph.add_child(ground, NodeData::Cube(CubeNode { width: 8.0, height: 0.1, depth: 8.0 }));

        // Rotating cube (blue)
        let cube_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        let cube_tf = graph.add_child(cube_sep, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(-1.5, 1.0, 0.0),
        )));
        graph.add_child(cube_sep, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.2, 0.4, 0.9),
            metallic: 0.3,
            roughness: 0.3,
            ..Default::default()
        }));
        graph.add_child(cube_sep, NodeData::Cube(CubeNode { width: 0.8, height: 0.8, depth: 0.8 }));

        // Oscillating sphere (orange)
        let sphere_tf = graph.add_child(root, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(0.0, 1.5, 0.0),
        )));
        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.9, 0.5, 0.1),
            metallic: 0.1,
            roughness: 0.4,
            ..Default::default()
        }));
        graph.add_child(sphere_tf, NodeData::Sphere(SphereNode { radius: 0.6 }));

        // Sliding cube (green)
        let slide_tf = graph.add_child(root, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(1.5, 0.4, 0.0),
        )));
        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.1, 0.8, 0.3),
            metallic: 0.2,
            roughness: 0.5,
            ..Default::default()
        }));
        graph.add_child(slide_tf, NodeData::Cube(CubeNode { width: 0.6, height: 0.6, depth: 0.6 }));

        // Engines
        let mut engines = EngineRegistry::new();
        engines.add(ElapsedTimeEngine::new(cube_tf, 1.2, Vec3::new(0.0, 1.0, 0.3).normalize()));
        engines.add(SineOscillatorEngine::new(sphere_tf, 0.8, 0.8, SineField::TranslationY));
        engines.add(InterpolateVec3Engine {
            transform_node: slide_tf,
            from: Vec3::new(1.5, 0.4, -1.5),
            to: Vec3::new(1.5, 0.4, 1.5),
            period_secs: 3.0,
        });
        engines.add(ElapsedTimeEngine::new(light1_id, 0.6, Vec3::new(0.0, 0.0, 1.0)));
        engines.add(ElapsedTimeEngine::new(light2_id, -0.6, Vec3::new(0.0, 0.0, 1.0)));

        engine.world_mut().engines = Some(engines);

        // Pre-render hook: apply background mode from shared state
        let bg_hook_clone = bg_hook.clone();
        engine.pre_render_hook = Some(Box::new(move |renderer| {
            let m = bg_hook_clone.load(Ordering::Relaxed);
            let mode = match m % 7 {
                0 => BgMode::Solid,
                1 => BgMode::VerticalGradient,
                2 => BgMode::HorizontalGradient,
                3 => BgMode::CenterGradient,
                4 => BgMode::DiagonalGradient,
                5 => BgMode::Image,
                _ => BgMode::SkyGround,
            };
            let bg = rc3d_render::background::BgSettings {
                mode,
                image_fit: ImageFit::Stretch,
                top_color: [0.02, 0.10, 0.35, 1.0],
                bot_color: [0.02, 0.02, 0.08, 1.0],
                image_path: Some("bg.png".into()),
                cube_faces: Default::default(),
            };
            renderer.set_background(bg);
        }));

        // HUD overlay: show current background mode
        let bg_hud = bg_mode.clone();
        engine.hud_text_hook = Some(Box::new(move || {
            let m = bg_hud.load(Ordering::Relaxed);
            let name = match m % 7 {
                0 => "Solid",
                1 => "VerticalGradient",
                2 => "HorizontalGradient",
                3 => "CenterGradient",
                4 => "DiagonalGradient",
                5 => "Image (bg.png)",
                _ => "SkyGround",
            };
            format!("+--- Animation Demo ------+\n| B: cycle background    |\n| Mode: {:<17} |\n+------------------------+", name)
        }));

        // B key: cycle background mode
        engine.panel_overlay_key_hook = Some(Box::new(move |key: PhysicalKey| -> bool {
            if let PhysicalKey::Code(KeyCode::KeyB) = key {
                let prev = bg_key.load(Ordering::Relaxed);
                bg_key.store(prev + 1, Ordering::Relaxed);
                let mode = match (prev + 1) % 7 {
                    0 => "Solid",
                    1 => "VerticalGradient",
                    2 => "HorizontalGradient",
                    3 => "CenterGradient",
                    4 => "DiagonalGradient",
                    5 => "Image",
                    _ => "SkyGround",
                };
                println!("[ANIM] Background: {mode}");
                true
            } else {
                false
            }
        }));
    });
}
