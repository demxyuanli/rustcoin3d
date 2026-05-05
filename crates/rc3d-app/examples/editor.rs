//! Editor — Unity-style UI (menu, toolbar, Hierarchy, Inspector, status) plus gizmo, viewports, undo, clipping, measurement.
//!
//! Camera: middle drag = orbit | right drag = pan | wheel = zoom — in multi-viewport layouts, wheel zoom targets the viewport under the cursor; middle/right drag locks orbit/pan to the viewport where the button was pressed until release.  
//! (Left click is reserved for selection / gizmo — not orbit.)
//!
//! Gizmo: toolbar Move / Rotate / Scale, or T / R / G, then drag handles in the view.  
//! Ctrl+Z / Ctrl+Y: undo / redo (also under Edit in the menu bar).  
//! C: cycle layout (Single -> Quad -> Left/Right -> Top/Bottom)  
//! Tab: cycle active viewport (when multi-viewport)  
//! P: section edit on/off, [ / ]: nudge planes  
//! X / Y / Z: axis clip toggles (without Ctrl)  
//! F: fit camera to selection  
//! Ctrl + left drag: box select  
//! M: measurement mode, click two points  
//! Escape: clear selection and measurements  
//! W / S / E / H: wireframe / shaded / shaded+edges / hidden-line (also View menu)  
//! I: cycle IBL preset

use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();

    println!("rustcoin3d Editor (egui in-viewport, Unity-style layout)");
    println!("  Menu bar, tools row, Hierarchy / Inspector, bottom status.");
    println!("  Open Help from the Help menu for shortcuts.");

    println!("  Debug: RUST_LOG=rc3d_app=trace logs HandleEventAction / EventCallback discovery on wheel or pointer.");

    let graph = build_demo_scene();

    let ctrl = CameraController::new(Vec3::ZERO, 10.0);

    let mut app = App::new(graph)
        .with_window_title("rustcoin3d Editor")
        .with_editor_ui(true)
        .with_camera_controller(ctrl)
        .with_initial_display_mode(DisplayMode::ShadedWithEdges);

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
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
        }),
    );
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(0.6, -0.4, 0.5).normalize(),
            color: Vec3::new(0.6, 0.7, 1.0),
            intensity: 0.5,
        }),
    );

    // Floor
    let floor_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
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
        }),
    );

    // PBR material grid: rows = increasing roughness (0.0 → 1.0), cols = increasing metallic (0.0 → 1.0)
    let grid_size = 5;
    let spacing = 2.5;
    for row in 0..grid_size {
        let roughness = row as f32 / (grid_size - 1) as f32;
        for col in 0..grid_size {
            let metallic = col as f32 / (grid_size - 1) as f32;
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            let x = (col as f32 - 2.0) * spacing;
            let z = (row as f32 - 2.0) * spacing;
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(x, 1.2, z))),
            );
            // Base color: warm gold hue with metallic/roughness variation
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
            graph.add_child(sep, NodeData::Sphere(SphereNode { radius: 0.8 }));
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
