use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

use super::command::CliCommand;
use crate::engine::state::{EngineEvent, EngineState, TestRun};

pub fn execute(cmd: CliCommand, state: &mut EngineState) {
    match cmd {
        CliCommand::SceneLoad(path) => {
            state.add_log("info", &format!("Loading scene: {path}"));
            build_demo_scene(&mut state.scene);
            state.push_event(EngineEvent::SceneLoaded);
            state.add_log("info", "Scene loaded (demo)");
        }
        CliCommand::SceneReset => {
            state.scene = rc3d_scene::SceneGraph::new();
            state.selection.clear();
            state.push_event(EngineEvent::SceneReset);
            state.add_log("info", "Scene reset");
        }
        CliCommand::TestRun(suite) => {
            let name = suite.unwrap_or_else(|| "default".into());
            state.add_log("info", &format!("Test run started: {name}"));
            state.active_test = Some(TestRun {
                name: name.clone(),
                total: 0,
                passed: 0,
                failed: 0,
                running: true,
            });
            state.push_event(EngineEvent::TestStarted(name));
        }
        CliCommand::TestStop => {
            state.active_test = None;
            state.push_event(EngineEvent::TestStopped);
            state.add_log("info", "Test run stopped");
        }
        CliCommand::CameraOrbit { dx: _, dy: _ }
        | CliCommand::CameraPan { dx: _, dy: _ }
        | CliCommand::CameraZoom(_)
        | CliCommand::CameraFit => {
            state.add_log("info", "Camera command received (use mouse for direct viewport control)");
        }
        CliCommand::Select(id) => {
            state.selection.clear();
            state.selection.insert(id);
            state.push_event(EngineEvent::SelectionChanged);
            state.add_log("info", &format!("Selected node {id:?}"));
        }
        CliCommand::SelectClear => {
            state.selection.clear();
            state.push_event(EngineEvent::SelectionChanged);
            state.add_log("info", "Selection cleared");
        }
        CliCommand::PropSet { node, field, value } => {
            state.add_log("info", &format!("Set {field}={value} on {node:?}"));
            state.push_event(EngineEvent::PropertyChanged { node, field });
        }
        CliCommand::DisplayMode(mode) => {
            state.display_mode = mode.clone();
            state.push_event(EngineEvent::DisplayModeChanged(mode));
            state.add_log("info", "Display mode changed");
        }
        CliCommand::LogFilter(level) => {
            state.add_log("info", &format!("Log filter set to: {level}"));
        }
        CliCommand::LogClear => {
            state.log_entries.clear();
            state.push_event(EngineEvent::LogCleared);
        }
        CliCommand::Help => {
            state.add_log("info", "Commands: scene load/reset | test run/stop | select <id>/clear | prop set <n> <f> <v> | display <mode> | log filter/clear | help | quit");
        }
        CliCommand::Quit => {
            state.push_event(EngineEvent::Quit);
        }
    }
}

fn build_demo_scene(scene: &mut rc3d_scene::SceneGraph) {
    let root = scene.add_root(NodeData::Separator(SeparatorNode));

    // Camera
    scene.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
        Vec3::new(5.0, 4.0, 8.0),
        Vec3::ZERO,
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        1.33,
    )));

    // Light
    scene.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
        direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
        color: Vec3::new(1.0, 0.95, 0.9),
        intensity: 1.2,
        light_group: None,
    }));

    // Floor
    let floor_sep = scene.add_child(root, NodeData::Separator(SeparatorNode));
    scene.add_child(floor_sep, NodeData::Material(MaterialNode {
        diffuse_color: Vec3::new(0.4, 0.4, 0.45),
        base_color: Vec3::new(0.4, 0.4, 0.45),
        roughness: 0.8,
        ..Default::default()
    }));
    scene.add_child(floor_sep, NodeData::Cube(CubeNode { width: 10.0, height: 0.2, depth: 10.0 }));

    // A few spheres
    for i in 0..3 {
        let sep = scene.add_child(root, NodeData::Separator(SeparatorNode));
        scene.add_child(sep, NodeData::Transform(TransformNode::from_translation(
            Vec3::new(-2.0 + 2.0 * i as f32, 1.2, 0.0),
        )));
        scene.add_child(sep, NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.2 + 0.3 * i as f32, 0.5, 0.7),
            base_color: Vec3::new(0.2 + 0.3 * i as f32, 0.5, 0.7),
            roughness: 0.4,
            metallic: 0.1 * i as f32,
            ..Default::default()
        }));
        scene.add_child(sep, NodeData::Sphere(SphereNode { radius: 0.8 }));
    }
}
