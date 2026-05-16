//! STL rendering diagnostic — import and display an STL file.
//!
//! Usage: cargo run -p rc3d-examples --example stl_diagnostic <file.stl>

use std::env;
use std::path::Path;

use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_examples::common::run_example;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path_arg = args.iter().skip(1).next().cloned();

    let Some(ref path_str) = path_arg else {
        eprintln!("Usage: stl_diagnostic <file.stl>");
        return;
    };

    let path = Path::new(path_str);
    let path_buf = path.to_path_buf();

    run_example("STL Diagnostic", |engine| {
        engine.set_display_mode(DisplayMode::Shaded);

        match engine.import(&path_buf) {
            Ok(root_id) => {
                log::info!("Imported STL at root node {:?}", root_id);

                // Add a directional light if none present
                let graph = engine.scene_mut();
                graph.add_child(
                    root_id,
                    rc3d_scene::node_data::NodeData::DirectionalLight(
                        rc3d_scene::node_data::DirectionalLightNode {
                            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
                            color: Vec3::ONE,
                            intensity: 1.2,
                            light_group: None,
                        },
                    ),
                );
            }
            Err(e) => {
                eprintln!("Import error: {e}");
            }
        }
    });
}
