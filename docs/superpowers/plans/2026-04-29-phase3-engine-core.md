# Phase 3: Engine-Only Core Features — Implementation Plan

**Goal:** Add measurement/annotation tools, offscreen rendering, undo/redo, and markup/redlining.

**Architecture:** All features implemented within existing crates. Measurement uses scene graph nodes + overlay rendering. Offscreen rendering wraps headless wgpu. Undo/redo is a command pattern. Markup uses 2D overlay geometry.

**Order:** T3-3 (Undo/Redo) → T1-3 (Measurement) → T1-6 (Offscreen) → T3-4 (Markup)

---

### Task 1: Undo/Redo System (T3-3)

**Files:**
- Create: `crates/rc3d-actions/src/undo.rs`
- Modify: `crates/rc3d-actions/src/lib.rs` (add module + re-export)

Implement `Command` trait and `CommandHistory`:

```rust
// crates/rc3d-actions/src/undo.rs

use rc3d_core::NodeId;
use rc3d_scene::SceneGraph;

/// A reversible operation on the scene graph.
pub trait Command: std::fmt::Debug + Send + Sync {
    fn execute(&mut self, graph: &mut SceneGraph);
    fn undo(&mut self, graph: &mut SceneGraph);
    fn description(&self) -> &str;
}

/// Bounded undo/redo history.
pub struct CommandHistory {
    undo_stack: Vec<Box<dyn Command>>,
    redo_stack: Vec<Box<dyn Command>>,
    max_depth: usize,
}

impl CommandHistory {
    pub fn new(max_depth: usize) -> Self {
        Self { undo_stack: Vec::new(), redo_stack: Vec::new(), max_depth: max_depth.clamp(1, 1024) }
    }

    pub fn push(&mut self, cmd: Box<dyn Command>) {
        self.redo_stack.clear();
        self.undo_stack.push(cmd);
        if self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
    }

    pub fn undo(&mut self, graph: &mut SceneGraph) -> bool {
        let mut cmd = match self.undo_stack.pop() {
            Some(c) => c,
            None => return false,
        };
        cmd.undo(graph);
        self.redo_stack.push(cmd);
        true
    }

    pub fn redo(&mut self, graph: &mut SceneGraph) -> bool {
        let mut cmd = match self.redo_stack.pop() {
            Some(c) => c,
            None => return false,
        };
        cmd.execute(graph);
        self.undo_stack.push(cmd);
        true
    }

    pub fn can_undo(&self) -> bool { !self.undo_stack.is_empty() }
    pub fn can_redo(&self) -> bool { !self.redo_stack.is_empty() }
}

// ── Built-in commands ──

/// Change a Transform node's translation.
#[derive(Debug)]
pub struct SetTranslationCommand {
    pub node: NodeId,
    pub old_value: glam::Vec3,
    pub new_value: glam::Vec3,
}

impl Command for SetTranslationCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        apply_translation(graph, self.node, self.new_value);
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        apply_translation(graph, self.node, self.old_value);
    }
    fn description(&self) -> &str { "SetTranslation" }
}

fn apply_translation(graph: &mut SceneGraph, node: NodeId, value: glam::Vec3) {
    if let Some(e) = graph.get_mut(node) {
        if let rc3d_scene::NodeData::Transform(t) = &mut e.data {
            t.translation = value;
        }
    }
}
```

Add to `lib.rs`:
```rust
pub mod undo;
pub use undo::{Command, CommandHistory, SetTranslationCommand};
```

Verify: `rtk cargo check` — 0 errors. Commit.

---

### Task 2: Measurement Tool (T1-3)

**Files:**
- Modify: `crates/rc3d-scene/src/node_data.rs` (MeasurementNode)
- Modify: `crates/rc3d-actions/src/lib.rs` (MeasurementAction)
- Create: `crates/rc3d-actions/src/measurement.rs`
- Create: `crates/rc3d-render/src/render_passes/pass_measurement.rs`

Add `MeasurementNode` to node_data:
```rust
#[derive(Clone, Debug)]
pub struct MeasurementNode {
    pub points: Vec<rc3d_core::math::Vec3>,
    pub measurement_type: MeasurementType,
    pub label: String,
    pub color: [f32; 4],
    pub value: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementType {
    Distance,
    Angle,
    Radius,
    Diameter,
}

impl Default for MeasurementNode {
    fn default() -> Self {
        Self { points: Vec::new(), measurement_type: MeasurementType::Distance, label: String::new(), color: [1.0, 1.0, 0.0, 1.0], value: 0.0 }
    }
}
```

Add `Measurement(MeasurementNode)` variant + `"Measurement"` type_name + no-op match arms in all Actions.

Create `crates/rc3d-actions/src/measurement.rs`:
```rust
use glam::Vec3;
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

pub struct MeasurementAction {
    pub mode: MeasurementMode,
    pub points: Vec<Vec3>,
    pub result_node: Option<NodeId>,
}

#[derive(Clone, Copy, Debug)]
pub enum MeasurementMode { Distance, Angle, Radius }

impl MeasurementAction {
    pub fn new(mode: MeasurementMode) -> Self {
        Self { mode, points: Vec::new(), result_node: None }
    }

    pub fn add_point(&mut self, point: Vec3) -> Option<(f32, String)> {
        self.points.push(point);
        let (count_needed, value, label) = match self.mode {
            MeasurementMode::Distance if self.points.len() >= 2 => {
                let d = self.points[0].distance(self.points[1]);
                (2, d, format!("{:.2}", d))
            }
            MeasurementMode::Angle if self.points.len() >= 3 => {
                let a = self.points[0] - self.points[1];
                let b = self.points[2] - self.points[1];
                let angle = a.angle_between(b).to_degrees();
                (3, angle, format!("{:.1}°", angle))
            }
            MeasurementMode::Radius if self.points.len() >= 2 => {
                let r = self.points[0].distance(self.points[1]);
                (2, r, format!("R={:.2}", r))
            }
            _ => return None,
        };
        Some((value, label))
    }

    pub fn create_node(&self, graph: &mut SceneGraph, parent: NodeId) -> NodeId {
        use rc3d_scene::node_data::*;
        let node = MeasurementNode {
            points: self.points.clone(),
            measurement_type: match self.mode {
                MeasurementMode::Distance => MeasurementType::Distance,
                MeasurementMode::Angle => MeasurementType::Angle,
                MeasurementMode::Radius => MeasurementType::Radius,
            },
            label: String::new(),
            color: [1.0, 1.0, 0.0, 1.0],
            value: 0.0,
        };
        graph.add_child(parent, NodeData::Measurement(node))
    }
}
```

Create `pass_measurement.rs` to render measurement lines and labels. Wire into `render_passes.rs`.

Verify: `rtk cargo check` — 0 errors. Commit.

---

### Task 3: Offscreen Rendering (T1-6)

**Files:**
- Create: `crates/rc3d-render/src/offscreen.rs`
- Modify: `crates/rc3d-render/src/lib.rs`

```rust
// crates/rc3d-render/src/offscreen.rs

use wgpu::{TextureFormat, Extent3d, TextureDescriptor, TextureUsages, TextureViewDescriptor};

/// Render-to-texture for screenshots, thumbnails, and print output.
pub struct OffscreenRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl OffscreenRenderer {
    pub async fn new(width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.expect("offscreen adapter");

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.expect("offscreen device");

        let texture = device.create_texture(&TextureDescriptor {
            label: Some("Offscreen target"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());

        Self { device, queue, texture, view, width, height }
    }

    /// Copy the rendered texture to a CPU-readable buffer.
    pub fn read_pixels(&self) -> Vec<u8> {
        let size = (self.width * self.height * 4) as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Offscreen readback"),
            size,
            usage: TextureUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.width * 4),
                    rows_per_image: Some(self.height),
                },
            },
            Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = buffer.slice(..).get_mapped_range().to_vec();
        buffer.unmap();
        data
    }

    pub fn save_screenshot(&self, path: &str) {
        let pixels = self.read_pixels();
        // Save via image crate: image::save_buffer(path, &pixels, width, height, ColorType::Rgba8)
    }
}
```

Verify: `rtk cargo check` — 0 errors. Commit.

---

### Task 4: Markup/Redlining (T3-4)

**Files:**
- Modify: `crates/rc3d-scene/src/node_data.rs` (MarkupNode)
- Create: `crates/rc3d-render/src/render_passes/pass_markup.rs`

Add MarkupNode to node_data:
```rust
#[derive(Clone, Debug)]
pub struct MarkupNode {
    pub elements: Vec<MarkupElement>,
    pub layer_name: String,
    pub visible: bool,
}

#[derive(Clone, Debug)]
pub enum MarkupElement {
    Line { start: [f32; 2], end: [f32; 2], color: [f32; 4], width: f32 },
    Rect { origin: [f32; 2], size: [f32; 2], color: [f32; 4], filled: bool },
    Circle { center: [f32; 2], radius: f32, color: [f32; 4] },
    Freehand { points: Vec<[f32; 2]>, color: [f32; 4], width: f32 },
    Text { position: [f32; 2], string: String, size: f32, color: [f32; 4] },
}

impl Default for MarkupNode {
    fn default() -> Self {
        Self { elements: Vec::new(), layer_name: String::new(), visible: true }
    }
}
```

Add `Markup(MarkupNode)` variant + `"Markup"` type_name + no-op traversal in all Actions.

Create `pass_markup.rs` for rendering 2D overlay markup lines/shapes using the edge_overlay pipeline.

Verify: `rtk cargo check` — 0 errors. Commit.
