//! PBR material showcase — grid of spheres with metallic (rows) × roughness (cols) variation.
//!
//! Demonstrates:
//!   - PBR metallic/roughness workflow
//!   - Multiple point lights for specular highlight testing
//!   - IBL environment map integration
//!   - HDR post-processing chain
//!
//! Keys:
//!   Mouse drag: Orbit camera around the grid
//!   L: Cycle IBL preset (neutral/studio/warm)
//!   1-4: Change shadow cascade count
//!   F: Cycle display mode
//!   Escape: clear selection | close window to quit

use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::{math::Vec3, DisplayMode};
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

const GRID_SIZE: usize = 7;
const SPACING: f32 = 2.5;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();
    print_pbr_materials_help();

    let graph = build_pbr_scene();

    let ctrl = CameraController::new(
        Vec3::new(
            GRID_SIZE as f32 * SPACING * 0.5,
            2.0,
            GRID_SIZE as f32 * SPACING * 0.8,
        ),
        GRID_SIZE as f32 * SPACING * 1.2,
    );

    let mut app = App::new(graph)
        .with_camera_controller(ctrl)
        .with_initial_display_mode(DisplayMode::Shaded)
        .with_hdr_post_processing(true);

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn print_pbr_materials_help() {
    println!("PBR material showcase");
    println!("Usage: cargo run -p rc3d-app --example pbr_materials");
    println!("Scene:");
    println!("  {}x{} sphere grid", GRID_SIZE, GRID_SIZE);
    println!("  Rows: roughness 0.0 -> 1.0");
    println!("  Cols: metallic 0.0 -> 1.0");
    println!("Controls:");
    println!("  Mouse drag: orbit camera");
    println!("  L: Cycle IBL preset");
    println!("  F: Cycle display mode");
    println!("  Escape: clear selection | Close window to quit");
    println!("Feature switches:");
    println!("  HDR post-processing enabled by default");
}

fn build_pbr_scene() -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera
    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(-2.0, 8.0, 14.0),
            Vec3::new(
                GRID_SIZE as f32 * SPACING * 0.5,
                0.5,
                GRID_SIZE as f32 * SPACING * 0.5,
            ),
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // Key light (directional, casts shadow)
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity: 1.5,
            light_group: None,
        }),
    );

    // Fill light (directional, softer)
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(0.7, -0.3, 0.5).normalize(),
            color: Vec3::new(0.6, 0.7, 1.0),
            intensity: 0.6,
            light_group: None,
        }),
    );

    // Point lights for dynamic highlights
    graph.add_child(
        root,
        NodeData::PointLight(PointLightNode {
            location: Vec3::new(
                GRID_SIZE as f32 * SPACING * 0.3,
                3.0,
                GRID_SIZE as f32 * SPACING * 0.3,
            ),
            color: Vec3::new(1.0, 0.9, 0.8),
            intensity: 20.0,
            light_group: None,
        }),
    );

    // Floor plane
    let floor = add_floor(&mut graph, root);

    // Sphere grid: columns = metallic, rows = roughness
    for row in 0..GRID_SIZE {
        let roughness = row as f32 / (GRID_SIZE - 1) as f32;
        for col in 0..GRID_SIZE {
            let metallic = col as f32 / (GRID_SIZE - 1) as f32;
            add_pbr_sphere(
                &mut graph,
                root,
                Vec3::new(col as f32 * SPACING, 1.2, row as f32 * SPACING),
                metallic,
                roughness,
            );
        }
    }

    let _ = floor; // Keep floor alive
    graph
}

fn add_pbr_sphere(
    graph: &mut SceneGraph,
    parent: rc3d_core::NodeId,
    position: Vec3,
    metallic: f32,
    roughness: f32,
) {
    let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));

    // Transform
    graph.add_child(
        sep,
        NodeData::Transform(TransformNode::from_translation(position)),
    );

    // Base color: warm gold hue with slight variation
    let hue = 0.12 + metallic * 0.05;
    let saturation = 0.3 + roughness * 0.4;
    let base = hsl_to_rgb(hue, saturation, 0.7);

    // Material
    graph.add_child(
        sep,
        NodeData::Material(MaterialNode {
            diffuse_color: base,
            ambient_color: base * 0.15,
            specular_color: Vec3::new(0.04, 0.04, 0.04),
            shininess: ((1.0 - roughness).max(0.01) * 128.0),
            base_color: base,
            metallic,
            roughness,
            opacity: 1.0,
            ..Default::default()
        }),
    );

    // Sphere geometry
    graph.add_child(sep, NodeData::Sphere(SphereNode { radius: 1.0 }));

    // Label as a thin column underneath each sphere
    let label_sep = graph.add_child(sep, NodeData::Separator(SeparatorNode));
    graph.add_child(
        label_sep,
        NodeData::Transform(TransformNode {
            translation: Vec3::new(0.0, -1.4, 0.0),
            rotation: rc3d_core::math::Mat4::IDENTITY,
            scale: Vec3::new(0.3, 0.3, 0.3),
            center: Vec3::ZERO,
        }),
    );
    let label_color = Vec3::new(0.3, 0.3, 0.3);
    graph.add_child(
        label_sep,
        NodeData::Material(MaterialNode::from_diffuse(label_color)),
    );
    graph.add_child(
        label_sep,
        NodeData::Cube(CubeNode {
            width: 1.0,
            height: 0.05,
            depth: 1.0,
        }),
    );
}

fn add_floor(graph: &mut SceneGraph, parent: rc3d_core::NodeId) -> rc3d_core::NodeId {
    let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));

    let floor_size = GRID_SIZE as f32 * SPACING * 1.5;
    let y_offset = -0.5;

    graph.add_child(
        sep,
        NodeData::Transform(TransformNode {
            translation: Vec3::new(floor_size * 0.5, y_offset, floor_size * 0.5),
            rotation: rc3d_core::math::Mat4::IDENTITY,
            scale: Vec3::new(floor_size * 0.5, 0.1, floor_size * 0.5),
            center: Vec3::ZERO,
        }),
    );

    graph.add_child(
        sep,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.5, 0.5, 0.5),
            ambient_color: Vec3::new(0.05, 0.05, 0.05),
            specular_color: Vec3::new(0.04, 0.04, 0.04),
            shininess: 4.0,
            base_color: Vec3::new(0.5, 0.5, 0.5),
            metallic: 0.0,
            roughness: 0.9,
            opacity: 1.0,
            ..Default::default()
        }),
    );

    graph.add_child(
        sep,
        NodeData::Cube(CubeNode {
            width: 2.0,
            height: 1.0,
            depth: 2.0,
        }),
    );

    sep
}

/// Simple HSL to RGB conversion.
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
