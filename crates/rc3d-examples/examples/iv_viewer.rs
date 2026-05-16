//! IV Import Viewer — loads and displays OpenInventor (.iv) files.
//!
//! Usage: cargo run -p rc3d-examples --example iv_viewer [file.iv]

use std::env;
use std::fs;
use std::path::Path;

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    let path_arg = args.iter().skip(1).next();

    // Capture args before the closure consumes them
    let iv_content = if let Some(path_arg) = path_arg {
        let path = Path::new(path_arg);
        match fs::read_to_string(path) {
            Ok(content) => Some(content),
            Err(e) => {
                eprintln!("Failed to read {}: {e}", path.display());
                return;
            }
        }
    } else {
        None
    };

    run_example("IV Viewer", |engine| {
        let graph = if let Some(ref content) = iv_content {
            match rc3d_io::parse_iv(content) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("Parse error: {e}");
                    build_fallback(engine)
                }
            }
        } else {
            build_fallback(engine)
        };

        engine.load_scene(graph);
    });
}

fn build_fallback(engine: &mut rc3d_engine_api::Engine) -> rc3d_scene::SceneGraph {
    let graph = engine.scene_mut();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(3.0, 2.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
        direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
        color: Vec3::ONE,
        intensity: 1.0,
        light_group: None,
    }));

    graph.add_child(root, NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.5, 0.8))));
    graph.add_child(root, NodeData::Cube(CubeNode::default()));

    // Return a fresh graph — we used engine.scene_mut() which borrows, so copy
    // the root structure. Since we're already mutating the engine's graph,
    // just return an empty graph and the current one has our fallback.
    rc3d_scene::SceneGraph::new()
}
