# CLI Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new standalone crate `rc3d-cli-editor` with a 4-panel egui layout driven by a CLI command engine, rendering a 3D viewport via rc3d-render.

**Architecture:** Single winit window; egui panels overlaid on a full-window 3D render (same overlay pattern as existing editor). A hand-rolled CLI parser feeds commands into a central `EngineState`; panels react to state changes via events. New crate, zero modification to existing crates.

**Tech Stack:** Rust, winit 0.30, wgpu 24, egui 0.31, rc3d-render, rc3d-scene, rc3d-core

---

### Task 1: Crate skeleton and workspace membership

**Files:**
- Create: `crates/rc3d-cli-editor/Cargo.toml`
- Create: `crates/rc3d-cli-editor/src/main.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Add crate to workspace members**

In root `Cargo.toml`, add `"crates/rc3d-cli-editor"` to the `members` list after `"crates/rc3d-script"`:

```toml
[workspace]
members = [
    "crates/rc3d-core",
    "crates/rc3d-fields",
    "crates/rc3d-scene",
    "crates/rc3d-nodes",
    "crates/rc3d-actions",
    "crates/rc3d-mesh",
    "crates/rc3d-nurbs",
    "crates/rc3d-render",
    "crates/rc3d-engine",
    "crates/rc3d-io",
    "crates/rc3d-app",
    "crates/rc3d-gizmo",
    "crates/rc3d-script",
    "crates/rc3d-cli-editor",
]
```

- [ ] **Step 2: Create Cargo.toml**

```toml
[package]
name = "rc3d-cli-editor"
version.workspace = true
edition.workspace = true
rust-version.workspace = true

[[bin]]
name = "cli-editor"
path = "src/main.rs"

[dependencies]
rc3d-core = { workspace = true }
rc3d-scene = { workspace = true }
rc3d-actions = { workspace = true }
rc3d-render = { workspace = true }
slotmap = { workspace = true }
wgpu = { workspace = true }
winit = { workspace = true }
pollster = { workspace = true }
log = { workspace = true }
env_logger = { workspace = true }
egui = "0.31.1"
egui-wgpu = "0.31.1"
egui-winit = "0.31.1"
```

- [ ] **Step 3: Create minimal main.rs**

```rust
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d_cli_editor=info"))
        .init();
    println!("rc3d CLI Editor starting...");
}
```

- [ ] **Step 4: Verify it compiles**

Run: `rtk cargo check -p rc3d-cli-editor`
Expected: compile success (main prints startup message)

- [ ] **Step 5: Commit**

```bash
rtk git add Cargo.toml crates/rc3d-cli-editor/
rtk git commit -m "feat: add rc3d-cli-editor crate skeleton"
```

---

### Task 2: CLI command types and parser

**Files:**
- Create: `crates/rc3d-cli-editor/src/cli/mod.rs`
- Create: `crates/rc3d-cli-editor/src/cli/command.rs`
- Create: `crates/rc3d-cli-editor/src/cli/parser.rs`

- [ ] **Step 1: Define CliCommand enum**

Create `crates/rc3d-cli-editor/src/cli/command.rs`:

```rust
use rc3d_core::NodeId;

#[derive(Debug, Clone)]
pub enum CliCommand {
    // Scene lifecycle
    SceneLoad(String),
    SceneReset,

    // Test execution
    TestRun(Option<String>),
    TestStop,

    // Camera
    CameraOrbit { dx: f32, dy: f32 },
    CameraPan { dx: f32, dy: f32 },
    CameraZoom(f32),
    CameraFit,

    // Selection
    Select(NodeId),
    SelectClear,

    // Property mutation
    PropSet { node: NodeId, field: String, value: String },

    // Display
    DisplayMode(String),

    // Log control
    LogFilter(String),
    LogClear,

    // Help
    Help,
    Quit,
}
```

- [ ] **Step 2: Implement parser**

Create `crates/rc3d-cli-editor/src/cli/parser.rs`:

```rust
use rc3d_core::NodeId;
use super::command::CliCommand;

pub fn parse(input: &str) -> Result<CliCommand, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("empty command".into());
    }

    let mut parts: Vec<&str> = trimmed.split_whitespace().collect();
    let cmd = parts.remove(0).to_lowercase();

    match cmd.as_str() {
        "scene" => parse_scene(&parts),
        "test" => parse_test(&parts),
        "camera" => parse_camera(&parts),
        "select" => parse_select(&parts),
        "prop" => parse_prop(&parts),
        "display" => parse_display(&parts),
        "log" => parse_log(&parts),
        "help" => Ok(CliCommand::Help),
        "quit" | "exit" => Ok(CliCommand::Quit),
        _ => Err(format!("unknown command: {cmd}")),
    }
}

fn parse_scene(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("load") => {
            let path = parts.get(1).ok_or("usage: scene load <path>")?;
            Ok(CliCommand::SceneLoad(path.to_string()))
        }
        Some("reset") => Ok(CliCommand::SceneReset),
        _ => Err("usage: scene load <path> | scene reset".into()),
    }
}

fn parse_test(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("run") => Ok(CliCommand::TestRun(parts.get(1).map(|s| s.to_string()))),
        Some("stop") => Ok(CliCommand::TestStop),
        _ => Err("usage: test run [suite] | test stop".into()),
    }
}

fn parse_camera(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("orbit") => {
            let dx: f32 = parts.get(1).unwrap_or(&"0").parse().map_err(|_| "invalid dx")?;
            let dy: f32 = parts.get(2).unwrap_or(&"0").parse().map_err(|_| "invalid dy")?;
            Ok(CliCommand::CameraOrbit { dx, dy })
        }
        Some("pan") => {
            let dx: f32 = parts.get(1).unwrap_or(&"0").parse().map_err(|_| "invalid dx")?;
            let dy: f32 = parts.get(2).unwrap_or(&"0").parse().map_err(|_| "invalid dy")?;
            Ok(CliCommand::CameraPan { dx, dy })
        }
        Some("zoom") => {
            let amount: f32 = parts.get(1).unwrap_or(&"1").parse().map_err(|_| "invalid zoom")?;
            Ok(CliCommand::CameraZoom(amount))
        }
        Some("fit") => Ok(CliCommand::CameraFit),
        _ => Err("usage: camera orbit|pan|zoom|fit <params>".into()),
    }
}

fn parse_select(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("clear") => Ok(CliCommand::SelectClear),
        Some(id_str) => {
            let id: u64 = id_str.parse().map_err(|_| "invalid node id")?;
            Ok(CliCommand::Select(NodeId::from(slotmap::KeyData::from_ffi(id))))
        }
        None => Err("usage: select <id> | select clear".into()),
    }
}

fn parse_prop(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("set") => {
            let node_str = parts.get(1).ok_or("usage: prop set <node> <field> <value>")?;
            let field = parts.get(2).ok_or("missing field")?;
            let value = parts.get(3).ok_or("missing value")?;
            let node_id: u64 = node_str.parse().map_err(|_| "invalid node id")?;
            Ok(CliCommand::PropSet {
                node: NodeId::from(slotmap::KeyData::from_ffi(node_id)),
                field: field.to_string(),
                value: value.to_string(),
            })
        }
        _ => Err("usage: prop set <node> <field> <value>".into()),
    }
}

fn parse_display(parts: &[&str]) -> Result<CliCommand, String> {
    let mode = parts.first().ok_or("usage: display <wireframe|shaded|edges|hidden>")?;
    Ok(CliCommand::DisplayMode(mode.to_string()))
}

fn parse_log(parts: &[&str]) -> Result<CliCommand, String> {
    match parts.first().copied() {
        Some("filter") => Ok(CliCommand::LogFilter(parts.get(1).unwrap_or(&"info").to_string())),
        Some("clear") => Ok(CliCommand::LogClear),
        _ => Err("usage: log filter <level> | log clear".into()),
    }
}
```

- [ ] **Step 3: Create CLI module index**

Create `crates/rc3d-cli-editor/src/cli/mod.rs`:

```rust
mod command;
mod parser;

pub use command::CliCommand;
pub use parser::parse;
```

- [ ] **Step 4: Verify it compiles**

Run: `rtk cargo check -p rc3d-cli-editor`
Expected: compile success

- [ ] **Step 5: Commit**

```bash
rtk git add crates/rc3d-cli-editor/
rtk git commit -m "feat: add CLI command types and parser"
```

---

### Task 3: Engine state and executor

**Files:**
- Create: `crates/rc3d-cli-editor/src/engine/mod.rs`
- Create: `crates/rc3d-cli-editor/src/engine/state.rs`
- Create: `crates/rc3d-cli-editor/src/cli/executor.rs`

- [ ] **Step 1: Define EngineState and EngineEvent**

Create `crates/rc3d-cli-editor/src/engine/state.rs`:

```rust
use std::collections::{HashSet, VecDeque};
use rc3d_core::NodeId;
use rc3d_scene::SceneGraph;

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct TestRun {
    pub name: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub running: bool,
}

#[derive(Debug, Clone)]
pub enum EngineEvent {
    SceneLoaded,
    SceneReset,
    TestStarted(String),
    TestPassed(String),
    TestFailed(String),
    TestStopped,
    SelectionChanged,
    PropertyChanged { node: NodeId, field: String },
    LogAppended,
    LogCleared,
    DisplayModeChanged(String),
    Quit,
}

pub struct EngineState {
    pub scene: SceneGraph,
    pub selection: HashSet<NodeId>,
    pub active_test: Option<TestRun>,
    pub log_entries: VecDeque<LogEntry>,
    pub display_mode: String,
    pub events: VecDeque<EngineEvent>,
}

impl EngineState {
    pub fn new() -> Self {
        Self {
            scene: SceneGraph::new(),
            selection: HashSet::new(),
            active_test: None,
            log_entries: VecDeque::with_capacity(256),
            display_mode: "shaded".into(),
            events: VecDeque::new(),
        }
    }

    pub fn push_event(&mut self, event: EngineEvent) {
        self.events.push_back(event);
    }

    pub fn take_events(&mut self) -> Vec<EngineEvent> {
        std::mem::take(&mut self.events).into_iter().collect()
    }

    pub fn add_log(&mut self, level: &str, message: &str) {
        if self.log_entries.len() >= 256 {
            self.log_entries.pop_front();
        }
        self.log_entries.push_back(LogEntry {
            level: level.to_string(),
            message: message.to_string(),
        });
        self.push_event(EngineEvent::LogAppended);
    }
}
```

- [ ] **Step 2: Implement executor**

Create `crates/rc3d-cli-editor/src/cli/executor.rs`:

```rust
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

use super::command::CliCommand;
use crate::engine::state::{EngineEvent, EngineState};

pub fn execute(cmd: CliCommand, state: &mut EngineState) {
    match cmd {
        CliCommand::SceneLoad(path) => {
            state.add_log("info", &format!("Loading scene: {path}"));
            // Scene loading happens async via io crate — for now, build a minimal demo scene
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
            state.active_test = Some(crate::engine::state::TestRun {
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
            // Camera handled in the viewport via mouse; CLI camera commands are recorded for future use
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
```

- [ ] **Step 3: Create engine module index**

Create `crates/rc3d-cli-editor/src/engine/mod.rs`:

```rust
pub mod state;
```

- [ ] **Step 4: Verify it compiles**

Run: `rtk cargo check -p rc3d-cli-editor`
Expected: compile success

- [ ] **Step 5: Commit**

```bash
rtk git add crates/rc3d-cli-editor/
rtk git commit -m "feat: add engine state and CLI executor"
```

---

### Task 4: Window setup and main loop

**Files:**
- Create: `crates/rc3d-cli-editor/src/render.rs`
- Modify: `crates/rc3d-cli-editor/src/main.rs`

- [ ] **Step 1: Create wgpu + egui renderer wrapper**

Create `crates/rc3d-cli-editor/src/render.rs`:

```rust
use egui_wgpu::ScreenDescriptor;
use rc3d_render::Renderer;
use std::collections::VecDeque;
use wgpu;

pub struct RenderContext {
    pub renderer: Renderer,
    pub egui_ctx: egui::Context,
    pub egui_winit: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,
    pub screen_descriptor: ScreenDescriptor,
    pub clipped_primitives: Vec<egui::ClippedPrimitive>,
    pub pixels_per_point: f32,
    pub textures_to_free: Vec<egui::TextureId>,
}

impl RenderContext {
    pub fn new(window: &winit::window::Window, renderer: Renderer) -> Self {
        let egui_ctx = egui::Context::default();
        let max_side = renderer.device.limits().max_texture_dimension_2d as usize;
        let winit_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(max_side),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &renderer.device,
            renderer.config.format,
            None,
            1,
            false,
        );
        let size = window.inner_size();
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [size.width.max(1), size.height.max(1)],
            pixels_per_point: window.scale_factor() as f32,
        };
        Self {
            renderer,
            egui_ctx,
            egui_winit: winit_state,
            egui_renderer,
            screen_descriptor,
            clipped_primitives: Vec::new(),
            pixels_per_point: window.scale_factor() as f32,
            textures_to_free: Vec::new(),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f32) {
        self.renderer.resize(width, height);
        self.screen_descriptor.size_in_pixels = [width.max(1), height.max(1)];
        self.screen_descriptor.pixels_per_point = scale_factor;
        self.pixels_per_point = scale_factor;
    }

    pub fn on_window_event(&mut self, window: &winit::window::Window, event: &winit::event::WindowEvent) -> bool {
        self.egui_winit.on_window_event(window, event).consumed
    }
}
```

- [ ] **Step 2: Rewrite main.rs with winit event loop**

Replace `crates/rc3d-cli-editor/src/main.rs`:

```rust
mod cli;
mod engine;
mod render;

use std::collections::VecDeque;
use std::time::Instant;

use rc3d_actions::RenderCollector;
use rc3d_core::math::Mat4;
use rc3d_core::NodeId;
use rc3d_render::Renderer;
use rc3d_scene::SceneGraph;
use render::RenderContext;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::WindowAttributes;

use cli::CliCommand;
use engine::state::{EngineEvent, EngineState};

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d_cli_editor=info"),
    )
    .init();

    let event_loop = EventLoop::new().unwrap();
    let mut app = CliEditorApp::new();
    event_loop.run_app(&mut app).expect("event loop");
}

struct CliEditorApp {
    state: EngineState,
    render_ctx: Option<RenderContext>,
    window: Option<winit::window::Window>,
    last_frame: Instant,
    frame_count: u64,
    cli_input: String,
    cli_history: VecDeque<String>,
}

impl CliEditorApp {
    fn new() -> Self {
        Self {
            state: EngineState::new(),
            render_ctx: None,
            window: None,
            last_frame: Instant::now(),
            frame_count: 0,
            cli_input: String::new(),
            cli_history: VecDeque::with_capacity(256),
        }
    }

    fn execute_command(&mut self, input: &str) {
        self.cli_history.push_back(format!("> {input}"));
        match cli::parse(input) {
            Ok(cmd) => {
                let is_quit = matches!(cmd, CliCommand::Quit);
                cli::executor::execute(cmd, &mut self.state);
                if is_quit {
                    std::process::exit(0);
                }
            }
            Err(e) => {
                self.state.add_log("error", &e);
            }
        }
    }
}

impl ApplicationHandler for CliEditorApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("rc3d CLI Editor")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
                )
                .expect("create window");

            let renderer = pollster::block_on(Renderer::new(&window));

            let mut render_ctx = RenderContext::new(&window, renderer);
            render_ctx.renderer.set_hud_enabled(false);

            self.render_ctx = Some(render_ctx);
            self.window = Some(window);

            // Load initial demo scene
            self.execute_command("scene load demo");
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let ui_consumed = if let (Some(ctx), Some(window)) = (&mut self.render_ctx, &self.window) {
            ctx.on_window_event(window, &event)
        } else {
            false
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(ctx) = &mut self.render_ctx {
                    ctx.resize(size.width, size.height, 1.0);
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                self.frame_count += 1;
                let now = Instant::now();
                let _dt = now.duration_since(self.last_frame);
                self.last_frame = now;

                if let (Some(ctx), Some(window)) = (&mut self.render_ctx, &self.window) {
                    // Build draw calls from scene
                    let mut collector = rc3d_actions::RenderCollector::new();
                    // Set render state before traverse
                    let roots: Vec<rc3d_core::NodeId> = self.state.scene.roots().to_vec();
                    if !roots.is_empty() {
                        for &root in &roots {
                            collector.traverse(&self.state.scene, root);
                        }
                    }
                    let draw_calls: Vec<rc3d_render::DrawCall> =
                        collector.draw_calls.iter().cloned().collect();

                    // Run egui
                    let raw_input = ctx.egui_winit.take_egui_input(window);
                    let full_output = ctx.egui_ctx.run(raw_input, |egui_ctx| {
                        build_egui_ui(egui_ctx, &mut self.state, &mut self.cli_input, &mut self.cli_history);
                    });
                    ctx.egui_winit.handle_platform_output(window, full_output.platform_output);

                    for (id, delta) in &full_output.textures_delta.set {
                        ctx.egui_renderer.update_texture(&ctx.renderer.device, &ctx.renderer.queue, *id, delta);
                    }
                    ctx.textures_to_free.extend(full_output.textures_delta.free.iter().copied());

                    ctx.pixels_per_point = full_output.pixels_per_point;
                    ctx.clipped_primitives = ctx.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

                    // Render 3D scene with egui overlay
                    let stats = ctx.renderer.render_draw_calls_with_overlay(
                        &draw_calls,
                        &self.state.scene,
                        Some(&mut |encoder, view| {
                            // Free old egui textures
                            for id in ctx.textures_to_free.drain(..) {
                                ctx.egui_renderer.free_texture(&id);
                            }
                            // Paint egui
                            let screen = ScreenDescriptor {
                                size_in_pixels: ctx.screen_descriptor.size_in_pixels,
                                pixels_per_point: ctx.pixels_per_point,
                            };
                            let bufs = ctx.egui_renderer.update_buffers(
                                &ctx.renderer.device,
                                &ctx.renderer.queue,
                                encoder,
                                &ctx.clipped_primitives,
                                &screen,
                            );
                            if !bufs.is_empty() {
                                ctx.renderer.queue.submit(bufs);
                            }
                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("egui_overlay"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            ctx.egui_renderer.render(&mut pass.forget_lifetime(), &ctx.clipped_primitives, &screen);
                        }),
                    );
                    let _ = stats;

                    // Process engine events (clear the queue)
                    let _events = self.state.take_events();
                }

                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if ui_consumed {
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                    return;
                }
                if key_event.state == ElementState::Pressed && !key_event.repeat {
                    if key_event.logical_key == winit::keyboard::Key::named(winit::keyboard::NamedKey::Enter) {
                        let input = std::mem::take(&mut self.cli_input);
                        if !input.trim().is_empty() {
                            self.execute_command(&input);
                        }
                    }
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
                let _ = (ui_consumed, delta);
            }
            _ => {}
        }
    }
}

fn build_egui_ui(
    ctx: &egui::Context,
    state: &EngineState,
    cli_input: &mut String,
    cli_history: &VecDeque<String>,
) {
    // Left panel — CLI
    egui::SidePanel::left("cli_panel")
        .default_width(320.0)
        .show(ctx, |ui| {
            ui.heading("CLI Console");
            ui.separator();
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in cli_history.iter().rev().take(50) {
                        ui.monospace(line);
                    }
                    for entry in state.log_entries.iter().rev().take(20) {
                        let color = match entry.level.as_str() {
                            "error" => egui::Color32::RED,
                            "warn" => egui::Color32::YELLOW,
                            _ => egui::Color32::WHITE,
                        };
                        ui.colored_label(color, format!("[{}] {}", entry.level, entry.message));
                    }
                });
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(">");
                ui.add(
                    egui::TextEdit::singleline(cli_input)
                        .desired_width(f32::INFINITY)
                        .hint_text("type command..."),
                );
            });
        });

    // Right panel — Properties
    egui::SidePanel::right("props_panel")
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.heading("Properties");
            ui.separator();
            if let Some(id) = state.selection.iter().next().copied() {
                ui.label(format!("Node: {id:?}"));
                if let Some(entry) = state.scene.get(id) {
                    ui.label(format!("Name: {}", entry.name.as_deref().unwrap_or("(unnamed)")));
                }
            } else {
                ui.label("No selection");
            }
            ui.separator();
            ui.label(format!("Display: {}", state.display_mode));
            if let Some(test) = &state.active_test {
                ui.separator();
                ui.label(format!("Test: {} [{}/{}]", test.name, test.passed, test.total));
            }
        });

    // Bottom panel — Logs / Diagnostics
    egui::TopBottomPanel::bottom("log_panel")
        .default_height(150.0)
        .show(ctx, |ui| {
            ui.heading("Logs & Diagnostics");
            ui.separator();
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for entry in state.log_entries.iter().rev().take(30) {
                        ui.monospace(format!("[{}] {}", entry.level, entry.message));
                    }
                });
        });

    // Central area is transparent — 3D scene shows through
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ctx, |_ui| {});
}
```

- [ ] **Step 2: Verify it compiles**

Run: `rtk cargo check -p rc3d-cli-editor`
Expected: must compile without errors. Fix any issues.

Note: `RenderCollector::new()` may not exist — use `rc3d_actions::RenderCollector` default constructor. If `traverse` signature differs, consult `rc3d_actions` docs via grep.

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-cli-editor/
rtk git commit -m "feat: add window setup, main loop, and 4-panel egui layout"
```

---

### Task 5: Wire mouse camera control into viewport

**Files:**
- Modify: `crates/rc3d-cli-editor/src/main.rs`

- [ ] **Step 1: Add camera state and mouse handling**

Add these fields to `CliEditorApp`:

```rust
camera_yaw: f32,
camera_pitch: f32,
camera_distance: f32,
camera_target: rc3d_core::math::Vec3,
mouse_pressed: Option<(f64, f64, MouseButton)>,
```

Initialize in `new()`:
```rust
camera_yaw: 0.0,
camera_pitch: -0.4,
camera_distance: 8.0,
camera_target: rc3d_core::math::Vec3::ZERO,
mouse_pressed: None,
```

- [ ] **Step 2: Handle mouse in window_event**

Add to `WindowEvent::MouseInput` match arm (before other arms):

```rust
WindowEvent::MouseInput { state: button_state, button, .. } => {
    if !ui_consumed && button_state == ElementState::Pressed {
        self.mouse_pressed = Some((self.mouse_pressed.map(|m| m.0).unwrap_or(0.0), self.mouse_pressed.map(|m| m.1).unwrap_or(0.0), button));
    } else if button_state == ElementState::Released {
        self.mouse_pressed = None;
    }
    if let Some(w) = &self.window { w.request_redraw(); }
}
```

Add to the `WindowEvent::CursorMoved` handling (before it checks `ui_consumed`):

```rust
WindowEvent::CursorMoved { position, .. } => {
    if let Some((prev_x, prev_y, button)) = self.mouse_pressed {
        let dx = position.x - prev_x;
        let dy = position.y - prev_y;
        match button {
            MouseButton::Middle => {
                self.camera_yaw += dx as f32 * 0.005;
                self.camera_pitch -= dy as f32 * 0.005;
            }
            MouseButton::Right => {
                let right = rc3d_core::math::Vec3::new(
                    self.camera_yaw.cos(), 0.0, -self.camera_yaw.sin()
                );
                let up = rc3d_core::math::Vec3::Y;
                self.camera_target += right * (-dx as f32 * 0.01) + up * (dy as f32 * 0.01);
            }
            _ => {}
        }
    }
    self.mouse_pressed = Some((position.x, position.y, self.mouse_pressed.map(|m| m.2).unwrap_or(MouseButton::Left)));
}
```

- [ ] **Step 3: Use camera in draw call collection**

Before building draw calls in `RedrawRequested`, compute the camera position and build view/projection:

```rust
let eye = rc3d_core::math::Vec3::new(
    self.camera_yaw.cos() * self.camera_pitch.cos() * self.camera_distance + self.camera_target.x,
    self.camera_pitch.sin() * self.camera_distance + self.camera_target.y,
    self.camera_yaw.sin() * self.camera_pitch.cos() * self.camera_distance + self.camera_target.z,
);
let view = Mat4::look_at_rh(eye, self.camera_target, rc3d_core::math::Vec3::Y);
let proj = Mat4::perspective_rh(800.0 / 600.0, std::f32::consts::FRAC_PI_4, 0.1, 1000.0);

// Pass these to collector
collector.view_matrix = view;
collector.projection_matrix = proj;
collector.camera_pos = eye;
```

- [ ] **Step 4: Add wheel zoom**

Add to mouse wheel handler:
```rust
WindowEvent::MouseWheel { delta, .. } => {
    let dy = match delta {
        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
        winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
    };
    self.camera_distance = (self.camera_distance - dy * 0.5).clamp(0.5, 50.0);
    if let Some(w) = &self.window { w.request_redraw(); }
}
```

- [ ] **Step 5: Verify it compiles**

Run: `rtk cargo check -p rc3d-cli-editor`
Expected: compile success

- [ ] **Step 6: Commit**

```bash
rtk git add crates/rc3d-cli-editor/src/main.rs
rtk git commit -m "feat: add mouse-driven camera control for 3D viewport"
```

---

### Task 6: CLI input focus management and Enter key handling

**Files:**
- Modify: `crates/rc3d-cli-editor/src/main.rs`

- [ ] **Step 1: Request egui keyboard focus on CLI input**

In `build_egui_ui`, after creating the `TextEdit`, call `.request_focus()` the first time:

Add a field to `CliEditorApp`:
```rust
cli_focus_requested: bool,
```

In `new()`: `cli_focus_requested: false,`

In `build_egui_ui`, after the TextEdit:
```rust
let resp = ui.add(
    egui::TextEdit::singleline(cli_input)
        .desired_width(f32::INFINITY)
        .hint_text("type command..."),
);
if !*cli_focus_requested {
    resp.request_focus();
    *cli_focus_requested = true;
}
```

- [ ] **Step 2: Handle Enter key via egui (remove global Enter capture)**

In `window_event`, remove the global Enter key handler that was `execute_command`. Instead, add an egui event in `build_egui_ui`:

In `build_egui_ui`, after the TextEdit:
```rust
if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
    // Command will be executed via the cli_history
}
```

Actually, simpler approach: use `enter_pressed` on the response. Remove the global keyboard handler for Enter, and instead check in `RedrawRequested`:

After running egui:
```rust
// Check if Enter was pressed in the CLI input — if so, execute command
if full_output.platform_output.events.iter().any(|e| matches!(e, egui::PlatformOutput { .. })) {
    // If the text contains a newline (from Enter), parse and execute
}
```

Simplest approach: use `ui.input(|i| i.key_pressed(egui::Key::Enter))` and check if the CLI input field has focus. If so, execute the command and clear input.

Add to `build_egui_ui` after the TextEdit:
```rust
if resp.has_focus() {
    if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
        // Pass execution signal via a mutable bool
    }
}
```

Better: pass a callback. Modify `build_egui_ui` signature:

```rust
fn build_egui_ui(
    ctx: &egui::Context,
    state: &EngineState,
    cli_input: &mut String,
    cli_history: &VecDeque<String>,
    on_command: &mut impl FnMut(String),
) {
```

Then in the Enter check:
```rust
if resp.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
    let cmd = cli_input.trim().to_string();
    if !cmd.is_empty() {
        on_command(std::mem::take(cli_input));
    }
}
```

Update the call site to pass the callback. Remove the global keyboard handler for Enter from `window_event`.

- [ ] **Step 3: Verify it compiles**

Run: `rtk cargo check -p rc3d-cli-editor`
Expected: compile success

- [ ] **Step 4: Commit**

```bash
rtk git add crates/rc3d-cli-editor/src/main.rs
rtk git commit -m "feat: wire CLI input focus and Enter-to-execute"
```

---

### Task 7: Build and run verification

**Files:**
- None (verification only)

- [ ] **Step 1: Build the binary**

Run: `rtk cargo build -p rc3d-cli-editor`
Expected: build success, binary at `target/debug/cli-editor.exe`

- [ ] **Step 2: Launch and smoke test**

Run: `RUST_LOG=rc3d_cli_editor=info cargo run -p rc3d-cli-editor`
Expected:
- Window opens with title "rc3d CLI Editor"
- Left panel shows CLI console with "> " prompt
- Center shows 3D scene (floor + 3 spheres)
- Right panel shows Properties
- Bottom panel shows Logs
- Typing `help` + Enter shows command list in console
- Middle-drag orbits camera, right-drag pans, wheel zooms

- [ ] **Step 3: Commit (if any fixes needed)**

```bash
rtk git add crates/rc3d-cli-editor/
rtk git commit -m "fix: address issues found during smoke test"
```
