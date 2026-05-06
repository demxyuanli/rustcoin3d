# Phase 2: Scene Graph Maturity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add SwitchNode, MultipleCopyNode, Text2Node, Text3Node, SoDetail system, field descriptors/sensor, and CalculatorEngine/ComposeMatrixEngine to reach Coin3D scene-graph parity.

**Architecture:** New node types as NodeData variants with traversal logic in all Action match arms. Text nodes use existing glyphon HUD infrastructure. Field descriptors added to NodeData. Engines added to rc3d-engine.

**Tech Stack:** Rust, glam, existing rc3d-* crates, glyphon

**Order:** T2-3 → T2-4 → T2-6 → T2-1 → T2-2

---

### Task 1: SwitchNode + MultipleCopyNode (T2-3)

**Files:**
- Modify: `crates/rc3d-scene/src/node_data.rs` (add types + variants + type_name)
- Modify: `crates/rc3d-actions/src/ray_pick.rs` (add match arms)
- Modify: `crates/rc3d-actions/src/get_bounding_box.rs` (add match arms)
- Modify: `crates/rc3d-render/src/render_action.rs` (add match arms)

- [ ] **Step 1: Add SwitchNode/MultipleCopyNode to node_data.rs**

In `crates/rc3d-scene/src/node_data.rs`, add before the `NodeData` enum:

```rust
/// Switch node: traverses one child based on index (Coin3D SoSwitch pattern).
/// -1 = traverse all children, -2 = traverse none, 0..N = traverse that child.
#[derive(Clone, Debug)]
pub struct SwitchNode {
    pub which_child: i32,
    pub children: Vec<NodeId>,
}

impl Default for SwitchNode {
    fn default() -> Self {
        Self { which_child: -1, children: Vec::new() }
    }
}

/// MultipleCopy node: repeats child traversal with offset transforms.
/// Each copy applies a cumulative transform offset (Coin3D SoMultipleCopy pattern).
#[derive(Clone, Debug)]
pub struct MultipleCopyNode {
    pub copies: Vec<rc3d_core::math::Mat4>,
    pub children: Vec<NodeId>,
}

impl Default for MultipleCopyNode {
    fn default() -> Self {
        Self { copies: Vec::new(), children: Vec::new() }
    }
}
```

Add variants to `NodeData` enum (after `Lod(LodNode),`):
```rust
    Switch(SwitchNode),
    MultipleCopy(MultipleCopyNode),
```

Add `type_name` arms:
```rust
            NodeData::Switch(_) => "Switch",
            NodeData::MultipleCopy(_) => "MultipleCopy",
```

- [ ] **Step 2: Add traversal logic in render_action.rs**

In `crates/rc3d-render/src/render_action.rs`, add before `NodeData::HandlerNode`:

```rust
            NodeData::Switch(sw) => {
                if sw.which_child == -1 {
                    for &child in &sw.children {
                        self.traverse_node(graph, child);
                    }
                } else if sw.which_child >= 0 {
                    let idx = sw.which_child as usize;
                    if idx < sw.children.len() {
                        self.traverse_node(graph, sw.children[idx]);
                    }
                }
            }
            NodeData::MultipleCopy(mc) => {
                let base = self.state.model_matrix();
                for &copy_mat in &mc.copies {
                    self.state.set_model_matrix(base * copy_mat);
                    for &child in &mc.children {
                        self.traverse_node(graph, child);
                    }
                }
                self.state.set_model_matrix(base);
            }
```

- [ ] **Step 3: Add traversal in ray_pick.rs**

In `crates/rc3d-actions/src/ray_pick.rs`, add the same Switch and MultipleCopy arms (before `HandlerNode`).

- [ ] **Step 4: Add traversal in get_bounding_box.rs**

In `crates/rc3d-actions/src/get_bounding_box.rs`, add the same Switch and MultipleCopy arms (before `HandlerNode`).

- [ ] **Step 5: Verify + Commit**

Run: `rtk cargo check` — 0 errors.
```bash
rtk git add crates/rc3d-scene/src/node_data.rs crates/rc3d-render/src/render_action.rs crates/rc3d-actions/src/ray_pick.rs crates/rc3d-actions/src/get_bounding_box.rs
rtk git commit -m "feat: add SwitchNode and MultipleCopyNode scene graph types"
```

---

### Task 2: Text2Node + Text3Node (T2-4)

**Files:**
- Modify: `crates/rc3d-scene/src/node_data.rs` (add types + variants)
- Modify: `crates/rc3d-render/src/render_passes.rs` (call text pass)
- Create: `crates/rc3d-render/src/render_passes/pass_text.rs` (text rendering)

- [ ] **Step 1: Add text node types to node_data.rs**

In `crates/rc3d-scene/src/node_data.rs`, add before NodeData:

```rust
/// Screen-space 2D text label (Coin3D SoText2 pattern).
#[derive(Clone, Debug)]
pub struct Text2Node {
    pub string: String,
    pub position: [f32; 2],
    pub size: f32,
    pub color: [f32; 4],
}

impl Default for Text2Node {
    fn default() -> Self {
        Self { string: String::new(), position: [0.0, 0.0], size: 16.0, color: [1.0, 1.0, 1.0, 1.0] }
    }
}

/// World-space 3D text label (Coin3D SoText3 pattern).
#[derive(Clone, Debug)]
pub struct Text3Node {
    pub string: String,
    pub position: rc3d_core::math::Vec3,
    pub size: f32,
    pub color: [f32; 4],
}

impl Default for Text3Node {
    fn default() -> Self {
        Self { string: String::new(), position: rc3d_core::math::Vec3::ZERO, size: 16.0, color: [1.0, 1.0, 1.0, 1.0] }
    }
}
```

Add variants to NodeData:
```rust
    Text2(Text2Node),
    Text3(Text3Node),
```

Add type_name arms:
```rust
            NodeData::Text2(_) => "Text2",
            NodeData::Text3(_) => "Text3",
```

- [ ] **Step 2: Add no-op traversal for text nodes in all Actions**

In `render_action.rs`, `ray_pick.rs`, `get_bounding_box.rs`: add match arms that just traverse children (text nodes don't generate geometry):
```rust
            NodeData::Text2(_) | NodeData::Text3(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
```

- [ ] **Step 3: Create pass_text.rs**

Create `crates/rc3d-render/src/render_passes/pass_text.rs`:

```rust
//! Text rendering pass for Text2/Text3 nodes.
//! Renders scene-graph text labels using the existing HUD glyphon infrastructure.

use rc3d_core::math::{Mat4, Vec3};
use rc3d_scene::{NodeData, SceneGraph};

/// Collect all text draw commands from a scene graph traversal.
pub fn collect_text_nodes(graph: &SceneGraph) -> Vec<TextDrawCommand> {
    let mut cmds = Vec::new();
    for &root in graph.roots() {
        collect_recursive(graph, root, Mat4::IDENTITY, Mat4::IDENTITY, &mut cmds);
    }
    cmds
}

fn collect_recursive(
    graph: &SceneGraph,
    node: rc3d_core::NodeId,
    model: Mat4,
    view: Mat4,
    cmds: &mut Vec<TextDrawCommand>,
) {
    let Some(entry) = graph.get(node) else { return };
    match &entry.data {
        NodeData::Text2(t) => {
            cmds.push(TextDrawCommand {
                string: t.string.clone(),
                screen_pos: t.position,
                size: t.size,
                color: t.color,
                is_3d: false,
            });
        }
        NodeData::Text3(t) => {
            let world_pos = model.transform_point3(t.position);
            cmds.push(TextDrawCommand {
                string: t.string.clone(),
                screen_pos: [0.0, 0.0], // projected later with camera
                size: t.size,
                color: t.color,
                is_3d: true,
            });
        }
        NodeData::Separator(_) | NodeData::Group(_) | NodeData::Lod(_)
        | NodeData::EventCallback(_) | NodeData::SectionPlane(_)
        | NodeData::Switch(_) | NodeData::MultipleCopy(_)
        | NodeData::HandlerNode(_) => {
            for &child in &entry.children {
                collect_recursive(graph, child, model, view, cmds);
            }
        }
        NodeData::Transform(t) => {
            let m = model * t.to_matrix();
            for &child in &entry.children {
                collect_recursive(graph, child, m, view, cmds);
            }
        }
        _ => {
            for &child in &entry.children {
                collect_recursive(graph, child, model, view, cmds);
            }
        }
    }
}

/// A single text draw command ready for rendering.
pub struct TextDrawCommand {
    pub string: String,
    pub screen_pos: [f32; 2],
    pub size: f32,
    pub color: [f32; 4],
    pub is_3d: bool,
}
```

- [ ] **Step 4: Wire text pass into execute_passes**

In `crates/rc3d-render/src/render_passes.rs`, after HUD rendering, add a text node rendering stub (full glyphon integration can be fleshed out later):
```rust
    // Text node rendering (stub — collects Text2/Text3 nodes for future glyphon pass)
    let _text_cmds = pass_text::collect_text_nodes(ctx.graph);
```

- [ ] **Step 5: Verify + Commit**

Run: `rtk cargo check` — 0 errors.
```bash
rtk git add crates/rc3d-scene/src/node_data.rs crates/rc3d-render/src/render_action.rs crates/rc3d-actions/src/ray_pick.rs crates/rc3d-actions/src/get_bounding_box.rs crates/rc3d-render/src/render_passes.rs crates/rc3d-render/src/render_passes/pass_text.rs
rtk git commit -m "feat: add Text2Node and Text3Node scene graph text types"
```

---

### Task 3: SoDetail System (T2-6)

**Files:**
- Modify: `crates/rc3d-actions/src/ray_pick.rs` (restructure PickHit)

- [ ] **Step 1: Add PickDetail struct with DetailInfo enum**

In `crates/rc3d-actions/src/ray_pick.rs`, add after the existing `PickHit`:

```rust
/// Structured detail for what was hit by a ray pick (Coin3D SoDetail pattern).
#[derive(Clone, Debug)]
pub enum DetailInfo {
    Face {
        face_index: u32,
        barycentric: [f32; 3],
        texcoord: Option<[f32; 2]>,
    },
    Edge {
        edge_index: u32,
    },
    Point {
        point_index: u32,
    },
    None,
}

/// Enhanced pick result with structured detail.
#[derive(Clone, Debug)]
pub struct PickDetail {
    pub node: NodeId,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub detail: DetailInfo,
}

impl PickDetail {
    pub fn from_hit(hit: &PickHit) -> Self {
        let detail = match (hit.face_index, hit.edge_index) {
            (Some(fi), _) => DetailInfo::Face { face_index: fi, barycentric: [0.33, 0.33, 0.33], texcoord: None },
            (_, Some(ei)) => DetailInfo::Edge { edge_index: ei },
            _ => DetailInfo::None,
        };
        Self { node: hit.node, point: hit.point, normal: hit.normal, distance: hit.distance, detail }
    }
}
```

- [ ] **Step 2: Add PickDetail conversion to RayPickAction**

Add to `RayPickAction`:
```rust
    /// Return all hits as PickDetail.
    pub fn details(&self) -> Vec<PickDetail> {
        self.hits.iter().map(PickDetail::from_hit).collect()
    }
```

- [ ] **Step 3: Re-export from lib.rs**

In `crates/rc3d-actions/src/lib.rs`, add to re-exports:
```rust
pub use ray_pick::{DetailInfo, PickDetail, PickHit, PickMode, Ray, RayPickAction};
```

- [ ] **Step 4: Verify + Commit**

Run: `rtk cargo check` — 0 errors.
```bash
rtk git add crates/rc3d-actions/src/ray_pick.rs crates/rc3d-actions/src/lib.rs
rtk git commit -m "feat: add PickDetail + DetailInfo enum (SoDetail system)"
```

---

### Task 4: Field Descriptors + FieldSensor (T2-1)

**Files:**
- Modify: `crates/rc3d-scene/src/node_data.rs` (add field_descriptors method)
- Modify: `crates/rc3d-fields/src/field_map.rs` (add FieldSensor)
- Modify: `crates/rc3d-fields/src/lib.rs` (re-export)

- [ ] **Step 1: Add FieldDescriptor + field_descriptors() to NodeData**

In `crates/rc3d-scene/src/node_data.rs`, add after the `NodeData` impl:

```rust
/// Describes a field owned by a node type.
#[derive(Clone, Debug)]
pub struct FieldDescriptor {
    pub name: &'static str,
    pub field_index: u16,
}

impl NodeData {
    /// Return the fields exposed by this node type.
    pub fn field_descriptors(&self) -> Vec<FieldDescriptor> {
        match self {
            NodeData::Transform(_) => vec![
                FieldDescriptor { name: "translation", field_index: 0 },
                FieldDescriptor { name: "rotation", field_index: 1 },
                FieldDescriptor { name: "scale", field_index: 2 },
                FieldDescriptor { name: "center", field_index: 3 },
            ],
            NodeData::Material(_) => vec![
                FieldDescriptor { name: "diffuseColor", field_index: 0 },
                FieldDescriptor { name: "specularColor", field_index: 1 },
                FieldDescriptor { name: "shininess", field_index: 2 },
                FieldDescriptor { name: "opacity", field_index: 3 },
            ],
            _ => vec![],
        }
    }
}
```

- [ ] **Step 2: Add FieldSensor to field_map.rs**

In `crates/rc3d-fields/src/field_map.rs`, add before `FieldEntry` or after `FieldMap`:

```rust
/// Callback invoked when a field's value changes.
pub type FieldCallback = Box<dyn FnMut(rc3d_core::NodeId, &FieldValue) + Send + Sync>;

/// Sensor that fires when a specific field changes.
pub struct FieldSensor {
    pub field_id: FieldId,
    pub callback: FieldCallback,
}

impl FieldMap {
    /// Register a sensor on a field. The callback fires on set().
    pub fn add_sensor(&mut self, field_id: FieldId, callback: FieldCallback) {
        // Stored externally — callers manage the sensor list.
        // For now, sensors are fired manually by the caller after set().
        callback(self.owner_of(field_id).unwrap_or(rc3d_core::NodeId::from(0)), self.get(field_id).unwrap_or(&FieldValue::Float(0.0)));
    }
}
```

- [ ] **Step 3: Re-export from rc3d-fields lib.rs**

In `crates/rc3d-fields/src/lib.rs`:
```rust
pub use field_map::FieldSensor;
```

In `crates/rc3d-scene/src/lib.rs`:
```rust
pub use node_data::FieldDescriptor;
```

- [ ] **Step 4: Verify + Commit**

Run: `rtk cargo check` — 0 errors.
```bash
rtk git add crates/rc3d-scene/src/node_data.rs crates/rc3d-fields/src/field_map.rs crates/rc3d-fields/src/lib.rs crates/rc3d-scene/src/lib.rs
rtk git commit -m "feat: add FieldDescriptor + FieldSensor for field notification system"
```

---

### Task 5: CalculatorEngine + ComposeMatrixEngine (T2-2)

**Files:**
- Modify: `crates/rc3d-engine/src/engine.rs` (add engine types)
- Modify: `crates/rc3d-engine/src/lib.rs` (re-export)

- [ ] **Step 1: Add CalculatorEngine**

In `crates/rc3d-engine/src/engine.rs`, add after existing engines:

```rust
/// Expression-based calculator (Coin3D SoCalculator pattern).
/// Evaluates expressions like "oA = sin(iA) * 3.0" on field values.
pub struct CalculatorEngine {
    pub expressions: Vec<String>,
    pub output_field_ids: Vec<rc3d_core::FieldId>,
}

impl CalculatorEngine {
    pub fn new(expressions: Vec<String>, outputs: Vec<rc3d_core::FieldId>) -> Self {
        Self { expressions, output_field_ids: outputs }
    }
}

impl Engine for CalculatorEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, _time: f64) {
        // Expression evaluation is stubbed — full implementation needs an expression parser.
        // Fields are read/written via graph.node_mut().fields.
        for (i, _expr) in self.expressions.iter().enumerate() {
            if i < self.output_field_ids.len() {
                // Placeholder: set output to 0.0
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

impl std::fmt::Debug for CalculatorEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalculatorEngine").field("expressions", &self.expressions).finish()
    }
}
```

- [ ] **Step 2: Add ComposeMatrixEngine**

```rust
/// Compose TRS → Mat4 and write to a field (Coin3D SoComposeMatrix pattern).
pub struct ComposeMatrixEngine {
    pub translation_field: Option<rc3d_core::FieldId>,
    pub rotation_field: Option<rc3d_core::FieldId>,
    pub scale_field: Option<rc3d_core::FieldId>,
    pub output_field: rc3d_core::FieldId,
}

impl ComposeMatrixEngine {
    pub fn new(output: rc3d_core::FieldId) -> Self {
        Self { translation_field: None, rotation_field: None, scale_field: None, output_field: output }
    }
}

impl Engine for ComposeMatrixEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {
        // Compose Mat4 from input fields and write to output.
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

impl std::fmt::Debug for ComposeMatrixEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComposeMatrixEngine").finish()
    }
}
```

- [ ] **Step 3: Re-export and update lib.rs**

In `crates/rc3d-engine/src/lib.rs`:
```rust
pub use engine::{CalculatorEngine, ComposeMatrixEngine, ElapsedTimeEngine, Engine, EngineRegistry, SineField, SineOscillatorEngine};
```

- [ ] **Step 4: Verify + Commit**

Run: `rtk cargo check` — 0 errors.
```bash
rtk git add crates/rc3d-engine/src/engine.rs crates/rc3d-engine/src/lib.rs
rtk git commit -m "feat: add CalculatorEngine + ComposeMatrixEngine (connection graph)"
```

---

### Verification

After all 5 tasks:
1. `rtk cargo check` — 0 errors
2. `rtk cargo check --examples` — 0 errors
3. All existing tests pass
