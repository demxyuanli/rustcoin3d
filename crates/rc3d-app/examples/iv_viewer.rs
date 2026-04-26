use std::env;
use std::fs;
use std::path::Path;

use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    let iv_content = if args.len() > 1 {
        let path = Path::new(&args[1]);
        match fs::read_to_string(path) {
            Ok(content) => {
                println!("Loaded: {}", path.display());
                content
            }
            Err(e) => {
                eprintln!("Failed to read {}: {e}", path.display());
                return;
            }
        }
    } else {
        // No argument: use builtin demo scene
        println!("Usage: iv_viewer <file.iv>");
        println!("No file specified, loading builtin demo scene.\n");
        builtin_scene()
    };

    let graph = match rc3d_io::parse_iv(&iv_content) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Parse error: {e}");
            eprintln!("Falling back to builtin scene.");
            build_fallback()
        }
    };

    // Print the round-trip .iv output
    let iv_out = rc3d_io::write_iv(&graph);
    println!("--- Serialized .iv ---\n{iv_out}");

    let mut app = App::new(graph);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn builtin_scene() -> String {
    r#"#Inventor V2.1 ascii
Separator {
    PerspectiveCamera {
        position 3 3 5
        nearDistance 0.1
        farDistance 100
        heightAngle 0.785
    }
    DirectionalLight {
        direction -1 -1 -1
    }
    Material { diffuseColor 0.9 0.2 0.2 }
    Cube { }
}"#
    .to_string()
}

fn build_fallback() -> rc3d_scene::SceneGraph {
    let mut g = rc3d_scene::SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(3.0, 3.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );
    g.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
        }),
    );
    g.add_child(
        root,
        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.8, 0.4, 0.2))),
    );
    g.add_child(root, NodeData::Cube(CubeNode::default()));
    g
}
