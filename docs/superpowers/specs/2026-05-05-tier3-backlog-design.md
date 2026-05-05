# Tier 3 Backlog — Implementation Design

Date: 2026-05-05
Status: design-approved
Scope: T3-2 (NURBS), T3-3 (Undo), T3-4 (Markup), T3-8 (Attributes)

## Deferred
T3-1 (Mesh boolean), T3-5 (STEP/IGES/IFC), T3-6 (VR/AR), T3-7 (Mesh repair) — require separate project initiation.

## Execution Order

Phase 1 (parallel): T3-2 + T3-8 — no mutual dependencies
Phase 2 (parallel): T3-3 + T3-4 — share scene graph mutation patterns, T3-3 depends on T3-8 FieldValue types

---

## T3-2: NURBS Surfaces

### Crate: `crates/rc3d-nurbs/`

Dependencies: `rc3d-core` only (Vec3/Vec4/Mat4).

### Modules

| Module | Purpose |
|--------|---------|
| `knot.rs` | Knot vector utilities: uniform, open-uniform, custom; knot insertion and refinement |
| `basis.rs` | B-spline and Bernstein basis function evaluation (Cox-de Boor recurrence) |
| `curve.rs` | `NurbsCurve`: evaluate, tangent, arc-length, adaptive tessellation |
| `surface.rs` | `NurbsSurface`: evaluate, normal, partial derivatives |
| `tessellate.rs` | Adaptive subdivision driven by curvature → `rc3d_mesh::TriangleMesh` |

### Core Types

```rust
// curve.rs
struct NurbsCurve {
    control_points: Vec<Vec4>,  // homogeneous (x,y,z,w)
    knots: Vec<f32>,
    degree: usize,
}
impl NurbsCurve {
    fn evaluate(&self, t: f32) -> Vec3;
    fn tangent(&self, t: f32) -> Vec3;
    fn arc_length(&self, n_samples: usize) -> f32;
    fn tessellate(&self, tolerance: f32) -> Vec<Vec3>;
    fn insert_knot(&mut self, t: f32);
}

// surface.rs
struct NurbsSurface {
    control_points: Vec<Vec<Vec4>>,  // u-count × v-count grid
    u_knots: Vec<f32>,
    v_knots: Vec<f32>,
    u_degree: usize,
    v_degree: usize,
}
impl NurbsSurface {
    fn evaluate(&self, u: f32, v: f32) -> Vec3;
    fn normal(&self, u: f32, v: f32) -> Vec3;
    fn tessellate_adaptive(&self, tolerance: f32) -> rc3d_mesh::TriangleMesh;
    fn tessellate_uniform(&self, u_samples: usize, v_samples: usize) -> rc3d_mesh::TriangleMesh;
}
```

### Integration

- `tessellate()` returns `TriangleMesh` → direct feed into GPU buffer pipeline
- No `NurbsSurfaceNode` needed in scene graph; tessellated result is `IndexedFaceSet`
- `rc3d-io` may add IGES entity-128 parser later

### Tests
- Circle as NURBS → evaluate points on unit circle, error < 1e-6
- Derivative convergence: finite difference approximates analytical tangent
- Knot insertion invariance (Cox-de Boor property)
- Adaptive tessellation: fewer triangles in flat regions, more in high-curvature

### Out of scope
- Trimmed NURBS, surface-surface intersection, skinning/lofting

---

## T3-8: Attribute System

### FieldValue Extension (`crates/rc3d-fields/src/field_value.rs`)

Add 3 variants:
```rust
String(String),   // names, paths, labels
Binary(Vec<u8>),  // embedded textures, serialized data
Float64(f64),     // high-precision scalars
```

### field_descriptors() Completion (`crates/rc3d-scene/src/node_data.rs`)

Currently only 3 types have descriptors (Transform, Material, DirectionalLight).
Add descriptors for all ~30 NodeData variants. Examples:

```rust
NodeData::PointLight(pl) => vec![
    FieldDescriptor { name: "location",        kind: FieldKind::Vec3f },
    FieldDescriptor { name: "color",           kind: FieldKind::Vec3f },
    FieldDescriptor { name: "intensity",       kind: FieldKind::Float },
    FieldDescriptor { name: "cutoff_distance", kind: FieldKind::Float },
],
NodeData::SectionPlane(sp) => vec![
    FieldDescriptor { name: "plane",   kind: FieldKind::Vec4f },
    FieldDescriptor { name: "enabled", kind: FieldKind::Bool },
],
```

Types with no runtime fields (Cube, Sphere, Cone, Cylinder, Separator, Group, Switch, LOD, MultipleCopy) return empty vec.

### Custom Attributes (`crates/rc3d-scene/src/node_entry.rs`)

Add to `NodeEntry`:
```rust
pub attributes: HashMap<String, String>,  // user-defined key-value pairs
```

Editor Inspector gets a "Custom Attributes" section: table of key-value rows, add/delete.

### Files Changed

| File | Change |
|------|--------|
| `crates/rc3d-fields/src/field_value.rs` | +3 variants |
| `crates/rc3d-scene/src/node_data.rs` | +~60 lines field_descriptors |
| `crates/rc3d-scene/src/node_entry.rs` | +1 field |
| `crates/rc3d-scene/src/scene_graph.rs` | init `HashMap::new()` in add_node |
| `crates/rc3d-app/src/editor_ui.rs` | Inspector attributes section |
| `crates/rc3d-io/src/iv.rs` | Info node → name mapping |

### Out of scope
- Attribute inheritance, expression evaluation, JSON serialization
- IV round-trip for custom attributes (only name mapping)

---

## T3-3: Undo Full Coverage

### Strategy: 4 reusable Command types

| Command | Covers |
|---------|--------|
| `SetFieldCommand<T>` | All material/light/camera/section-plane mutations |
| `AddChildCommand` | CreateNode, DuplicateNode |
| `RemoveChildCommand` | DeleteNode (stores deep copy of removed subtree) |
| `CompoundCommand` | Atomic transactions (gizmo drag = 1 undo) |

Existing `SetTranslationCommand`/`SetRotationCommand`/`SetScaleCommand` stay as-is (they were the first 3, still valid).

### SetFieldCommand (`crates/rc3d-actions/src/undo.rs`)

```rust
pub struct SetFieldCommand<T: Clone + Debug + Send + Sync + 'static> {
    node: NodeId,
    old_value: T,
    new_value: T,
    apply: Box<dyn Fn(&mut NodeEntry, T) + Send + Sync>,
    desc: String,
}
impl<T: Clone + Debug + Send + Sync + 'static> Command for SetFieldCommand<T> {
    fn execute(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.node) {
            (self.apply)(entry, self.new_value.clone());
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.node) {
            (self.apply)(entry, self.old_value.clone());
        }
    }
    fn description(&self) -> &str { &self.desc }
}
```

### AddChildCommand / RemoveChildCommand

```rust
pub struct AddChildCommand {
    parent: NodeId,
    child: NodeId,
}
pub struct RemoveChildCommand {
    parent: NodeId,
    child: NodeId,
    removed_subtree: Option<Box<NodeEntry>>,  // deep-clone for restore
    child_index: usize,
}
```

### CompoundCommand

```rust
pub struct CompoundCommand {
    commands: Vec<Box<dyn Command>>,
    desc: String,
}
impl Command for CompoundCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        for cmd in &mut self.commands { cmd.execute(graph); }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        for cmd in self.commands.iter_mut().rev() { cmd.undo(graph); }
    }
    fn description(&self) -> &str { &self.desc }
}
```

### EditorCommand Wiring

~20 EditorCommand variants in `apply_editor_commands()` change from inline mutation to command execution:
```rust
EditorCommand::SetBaseColor { node, color } => {
    let old = /* extract current value from graph */;
    let cmd = SetFieldCommand::new(node, old, color, "SetBaseColor",
        |e, v| if let NodeData::Material(m) = &mut e.data { m.base_color = v });
    app.command_history.execute(Box::new(cmd), &mut app.world.graph);
}
```

Gizmo drag uses `CompoundCommand` to bundle translate+rotate+scale into one atomic undo step.

### Out of scope
- Undo history UI (list of past actions)
- Undo persistence across sessions
- Graph snapshot/diff approach for large subtree removal

---

## T3-4: Markup Redline Editing

### Architecture

```
winit mouse events
    ↓
MarkupTool state machine (crates/rc3d-actions/src/markup_tool.rs)
    ├─ Mode: Line | Rect | Circle | Freehand | Select
    ├─ Temp state: accumulated click points, current mouse position
    └─ On completion → push MarkupElement into MarkupNode in scene graph
    ↓
pass_markup (crates/rc3d-render/src/render_passes/pass_markup.rs)
    ├─ Collect all MarkupNodes from scene graph
    ├─ Generate screen-space wireframe vertices per element type
    └─ Render with depth_compare: Always, fixed color/width, no lighting
```

### MarkupTool (`crates/rc3d-actions/src/markup_tool.rs`)

```rust
pub enum MarkupTool { Select, Line, Rect, Circle, Freehand }

pub struct MarkupAction {
    pub tool: MarkupTool,
    pub target_node: Option<NodeId>,
    pub click_points: Vec<Vec2>,
    pub preview_element: Option<MarkupElement>,
}

impl MarkupAction {
    fn set_tool(&mut self, tool: MarkupTool);
    fn on_mouse_down(&mut self, screen_pos: Vec2, graph: &mut SceneGraph) -> bool;
    fn on_mouse_move(&mut self, screen_pos: Vec2);
    fn on_mouse_up(&mut self, screen_pos: Vec2, graph: &mut SceneGraph) -> Option<MarkupElement>;
    fn cancel(&mut self);  // Esc/right-click → clear temp state
}
```

On `mouse_up` returning `Some(element)`, caller pushes it into `target_node.elements`.

### Render Pass (`crates/rc3d-render/src/render_passes/pass_markup.rs`)

Pipeline: `depth_compare: Always`, `blend: ALPHA_BLENDING`, no lighting.

Vertex generation per element type:
- **Line**: 2 vertices (start→end)
- **Rect**: 4 vertices (origin→diagonal, filled or outline)
- **Circle**: 64-segment polyline approximation
- **Freehand**: polyline from point list
- **Dimension**: extension lines + arrows + label line
- **Text**: deferred to HUD/glyphon layer, not in this pass

### Editor Integration

Toolbar: `Markup: [Select] [Line] [Rect] [Circle] [Freehand]`

New EditorCommand variants:
```rust
SetMarkupTool(MarkupTool),
MarkupMouseDown { viewport_id, screen_pos },
MarkupMouseMove { screen_pos },
MarkupMouseUp { screen_pos },
ClearAllMarkup { node },
```

`MarkupAction` instance lives in `App` alongside `MeasurementAction`.

### Out of scope
- GD&T tolerance annotations
- Automatic dimension value computation (manual label string input)
- Markup undo (covered by T3-3's SetFieldCommand on element list)
- 3D text element rendering (Text variant shown via egui panel)

---

## Files Changed Summary

| Crate | File | Change |
|-------|------|--------|
| rc3d-nurbs | `src/{lib,knot,basis,curve,surface,tessellate}.rs` | New crate (~400 lines) |
| rc3d-fields | `src/field_value.rs` | +3 variants (~10 lines) |
| rc3d-scene | `src/node_data.rs` | +~60 lines field_descriptors |
| rc3d-scene | `src/node_entry.rs` | +1 field attributes |
| rc3d-scene | `src/scene_graph.rs` | init HashMap in add_node |
| rc3d-actions | `src/undo.rs` | +4 Command types (~100 lines) |
| rc3d-actions | `src/markup_tool.rs` | New file (~120 lines) |
| rc3d-actions | `src/lib.rs` | Export new types |
| rc3d-render | `src/render_passes/pass_markup.rs` | New file (~80 lines) |
| rc3d-render | `src/render_passes.rs` | Add mod + pass call |
| rc3d-render | `src/pipelines.rs` | Add markup_lines pipeline |
| rc3d-app | `src/app/editor_commands.rs` | Wire ~20 commands to undo (~150 lines) |
| rc3d-app | `src/app/mod.rs` | Add MarkupAction field + dispatch |
| rc3d-app | `src/editor_ui.rs` | Attributes inspector + markup toolbar (~80 lines) |
| rc3d-io | `src/iv.rs` | Info → name mapping (~5 lines) |

Estimated total: ~1,200 lines across 16 files + 1 new crate.
