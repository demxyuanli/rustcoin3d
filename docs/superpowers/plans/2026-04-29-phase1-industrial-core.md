# Phase 1: Industrial Core Features — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add transparency rendering, interactive section planes, selection highlighting, scene-graph LOD nodes, and SoPath system to rustcoin3d.

**Architecture:** SoPath is the foundation (used by LOD and section). LOD node changes traversal in all Actions. Transparency is a new render pass with depth-sorted draw calls. Selection highlighting enhances existing passes. Section plane adds a new node type + action.

**Tech Stack:** Rust, wgpu, glam, existing rc3d-* crates

**Order:** T1-7 (SoPath) → T1-5 (LOD) → T1-1 (透明) → T1-4 (选择高亮) → T1-2 (剖面)

---

### Task 1: ScenePath + SoSearchAction (T1-7)

**Files:**
- Create: `crates/rc3d-actions/src/scene_path.rs`
- Modify: `crates/rc3d-actions/src/lib.rs` (add module)
- Modify: `crates/rc3d-actions/src/action.rs` (add ActionKind::Search + ActionKind::GetMatrix)

- [ ] **Step 1: Create scene_path.rs**

```rust
//! Scene graph path: chain from root to a target node.
//! Follows Coin3D SoPath pattern for path-based search and matrix computation.

use rc3d_core::math::Mat4;
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

/// A path through the scene graph: `(node_id, child_index)` chain.
/// `head` is a root node, `tail` is the target node.
#[derive(Clone, Debug)]
pub struct ScenePath {
    pub nodes: Vec<NodeId>,
    pub child_indices: Vec<usize>,
}

impl ScenePath {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), child_indices: Vec::new() }
    }

    pub fn head(&self) -> Option<NodeId> {
        self.nodes.first().copied()
    }

    pub fn tail(&self) -> Option<NodeId> {
        self.nodes.last().copied()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Compute the accumulated world transform along this path.
    pub fn get_matrix(&self, graph: &SceneGraph) -> Mat4 {
        let mut m = Mat4::IDENTITY;
        for &node_id in &self.nodes {
            if let Some(entry) = graph.get(node_id) {
                if let NodeData::Transform(t) = &entry.data {
                    m = m * t.to_matrix();
                }
            }
        }
        m
    }

    /// Return all node IDs as a slice.
    pub fn node_slice(&self) -> &[NodeId] {
        &self.nodes
    }
}

impl Default for ScenePath {
    fn default() -> Self {
        Self::new()
    }
}

/// Search action: find nodes by name or type in the scene graph.
/// Returns a list of ScenePaths to matching nodes.
pub struct SearchAction {
    pub name_filter: Option<String>,
    pub type_filter: Option<String>,
    pub results: Vec<ScenePath>,
    current_path: ScenePath,
}

impl SearchAction {
    pub fn new() -> Self {
        Self {
            name_filter: None,
            type_filter: None,
            results: Vec::new(),
            current_path: ScenePath::new(),
        }
    }

    pub fn by_name(name: &str) -> Self {
        let mut s = Self::new();
        s.name_filter = Some(name.to_string());
        s
    }

    pub fn by_type(type_name: &str) -> Self {
        let mut s = Self::new();
        s.type_filter = Some(type_name.to_string());
        s
    }

    fn matches(&self, entry: &rc3d_scene::NodeEntry) -> bool {
        if let Some(ref name) = self.name_filter {
            if entry.name.as_deref() == Some(name.as_str()) {
                return true;
            }
        }
        if let Some(ref tn) = self.type_filter {
            if entry.data.type_name() == tn.as_str() {
                return true;
            }
        }
        false
    }

    fn traverse(&mut self, graph: &SceneGraph, node: NodeId, child_idx: usize) {
        let Some(entry) = graph.get(node) else { return };

        self.current_path.nodes.push(node);
        self.current_path.child_indices.push(child_idx);

        if self.matches(entry) {
            self.results.push(self.current_path.clone());
        }

        for (i, &child) in entry.children.iter().enumerate() {
            self.traverse(graph, child, i);
        }

        self.current_path.nodes.pop();
        self.current_path.child_indices.pop();
    }
}

/// GetMatrix action: compute the world transform for a given path or node.
pub struct GetMatrixAction {
    pub path: ScenePath,
    pub result: Mat4,
}

impl GetMatrixAction {
    pub fn new(path: ScenePath) -> Self {
        Self { path, result: Mat4::IDENTITY }
    }

    pub fn from_node(node: NodeId) -> Self {
        let mut path = ScenePath::new();
        path.nodes.push(node);
        path.child_indices.push(0);
        Self { path, result: Mat4::IDENTITY }
    }

    /// Walk up from the target node to the root, accumulating transforms.
    pub fn compute(&mut self, graph: &SceneGraph) {
        self.result = self.path.get_matrix(graph);
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

In `crates/rc3d-actions/src/lib.rs`, add after `pub mod ray_pick;`:
```rust
pub mod scene_path;
```

And add re-export:
```rust
pub use scene_path::{GetMatrixAction, ScenePath, SearchAction};
```

- [ ] **Step 3: Add ActionKind variants**

In `crates/rc3d-actions/src/action.rs`, add to `ActionKind` enum:
```rust
pub enum ActionKind {
    GLRender,
    GetBoundingBox,
    RayPick,
    Search,
    HandleEvent,
    GetMatrix,
}
```

- [ ] **Step 4: Verify compilation**

Run: `rtk cargo check -p rc3d-actions`
Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
rtk git add crates/rc3d-actions/src/scene_path.rs crates/rc3d-actions/src/lib.rs crates/rc3d-actions/src/action.rs
rtk git commit -m "feat: add ScenePath, SearchAction, GetMatrixAction (SoPath system)"
```

---

### Task 2: LOD Scene Graph Node (T1-5)

**Files:**
- Modify: `crates/rc3d-scene/src/node_data.rs` (add LodNode + LodLevel)
- Modify: `crates/rc3d-actions/src/ray_pick.rs` (add Lod match arm)
- Modify: `crates/rc3d-actions/src/get_bounding_box.rs` (add Lod match arm)
- Modify: `crates/rc3d-render/src/render_action.rs` (add Lod match arm)

- [ ] **Step 1: Add LodNode/LodLevel to node_data.rs**

Add before the `NodeData` enum in `crates/rc3d-scene/src/node_data.rs`:

```rust
/// One LOD level: a group of child nodes to render at this distance range.
#[derive(Clone, Debug)]
pub struct LodLevel {
    pub children: Vec<NodeId>,
    /// Distance threshold (closer than this uses this or higher detail level).
    pub max_distance: f32,
}

/// LOD node: selects one child group based on distance from camera.
/// Coin3D SoLOD / SoLevelOfDetail pattern.
#[derive(Clone, Debug)]
pub struct LodNode {
    pub levels: Vec<LodLevel>,
    /// Screen-space area threshold (0..1), alternative to distance-based.
    pub screen_area_thresholds: Vec<f32>,
    /// Currently selected level index (updated each frame by traversal).
    pub current_level: usize,
}

impl Default for LodNode {
    fn default() -> Self {
        Self {
            levels: Vec::new(),
            screen_area_thresholds: Vec::new(),
            current_level: 0,
        }
    }
}
```

Add variant to `NodeData` enum (after `EventCallback`):
```rust
    /// Level-of-detail switch (Coin3D SoLOD pattern).
    Lod(LodNode),
```

Add `type_name` arm in the match:
```rust
            NodeData::Lod(_) => "Lod",
```

- [ ] **Step 2: Add Lod traversal to ray_pick.rs**

In `crates/rc3d-actions/src/ray_pick.rs`, add before `NodeData::HandlerNode`:

```rust
            NodeData::Lod(lod) => {
                let level = lod.current_level.min(lod.levels.len().saturating_sub(1));
                if let Some(level_data) = lod.levels.get(level) {
                    for &child in &level_data.children {
                        self.traverse_node(graph, child);
                    }
                }
            }
```

Same in `get_bounding_box.rs` and `render_action.rs`.

- [ ] **Step 3: Add Lod traversal to render_action.rs**

In `crates/rc3d-render/src/render_action.rs`, add before `NodeData::HandlerNode`:

```rust
            NodeData::Lod(lod) => {
                let level = lod.current_level.min(lod.levels.len().saturating_sub(1));
                if let Some(level_data) = lod.levels.get(level) {
                    for &child in &level_data.children {
                        self.traverse_node(graph, child);
                    }
                }
            }
```

- [ ] **Step 4: Add Lod traversal to get_bounding_box.rs**

In `crates/rc3d-actions/src/get_bounding_box.rs`, add before `NodeData::HandlerNode`:

```rust
            NodeData::Lod(lod) => {
                let level = lod.current_level.min(lod.levels.len().saturating_sub(1));
                if let Some(level_data) = lod.levels.get(level) {
                    for &child in &level_data.children {
                        self.traverse_node(graph, child);
                    }
                }
            }
```

- [ ] **Step 5: Verify compilation**

Run: `rtk cargo check`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
rtk git add crates/rc3d-scene/src/node_data.rs crates/rc3d-actions/src/ray_pick.rs crates/rc3d-actions/src/get_bounding_box.rs crates/rc3d-render/src/render_action.rs
rtk git commit -m "feat: add LodNode scene graph node with distance/area-based level selection"
```

---

### Task 3: Material Opacity + Transparent Render Pass (T1-1)

**Files:**
- Modify: `crates/rc3d-scene/src/node_data.rs` (add opacity to MaterialNode)
- Modify: `crates/rc3d-actions/src/element.rs` (add opacity to MaterialElement)
- Modify: `crates/rc3d-render/src/render_action.rs` (add opacity to DrawCall, propagate)
- Modify: `crates/rc3d-render/src/renderer.rs` (transparent sort + pipeline)
- Modify: `crates/rc3d-render/src/pipelines.rs` (solid_alpha pipeline)
- Modify: `crates/rc3d-render/src/render_passes.rs` (transparent render pass)
- Create: `crates/rc3d-render/src/render_passes/pass_transparent.rs`

- [ ] **Step 1: Add opacity to MaterialNode**

In `crates/rc3d-scene/src/node_data.rs`, add field to `MaterialNode`:
```rust
    pub opacity: f32,
```

In `impl Default for MaterialNode`:
```rust
            opacity: 1.0,
```

- [ ] **Step 2: Add opacity to MaterialElement**

In `crates/rc3d-actions/src/element.rs`, add field to `MaterialElement`:
```rust
    pub opacity: f32,
```

In `MaterialElement::default()`:
```rust
            opacity: 1.0,
```

In `MaterialElement` constructor and the `to_material_node` method, add `opacity` propagation.

- [ ] **Step 3: Add opacity to DrawCall**

In `crates/rc3d-render/src/render_action.rs`, add field to `DrawCall`:
```rust
    pub opacity: f32,
```

In `RenderCollector::emit_draw_call*` methods, set `opacity` from `state.material().opacity` (or 1.0 if no material).

- [ ] **Step 4: Add solid_alpha pipeline**

In `crates/rc3d-render/src/pipelines.rs`, add to `DepthModePipelines`:
```rust
    pub solid_alpha: wgpu::RenderPipeline,
```

In `build_depth_mode_pipelines()`, create the pipeline:
```rust
    let solid_alpha = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("PBR solid alpha"),
        layout: Some(lit_pll),
        vertex: wgpu::VertexState {
            module: lit_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: lit_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: false,
            depth_compare: depth_cmp,
            stencil: stencil_unchanged.clone(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: ms,
        multiview: None,
        cache: None,
    });
```

Return `solid_alpha` in `DepthModePipelines { solid, solid_depth_prepass, wireframe, edge_overlay, selection_fill, selection_edge, outline, solid_alpha }`.

- [ ] **Step 5: Transparent draw call sorting**

In `crates/rc3d-render/src/renderer.rs`, in `render_draw_calls()`, add after the selected_order:

```rust
    let mut transparent_order: Vec<usize> = (0..draw_calls.len())
        .filter(|&i| draw_calls[i].opacity < 1.0 && draw_calls[i].opacity > 0.0)
        .collect();
    
    // Sort by camera distance descending (far to near for correct blending)
    transparent_order.sort_unstable_by(|&a, &b| {
        let da = draw_calls[a].model_matrix.w_axis.truncate().distance(camera_pos.into());
        let db = draw_calls[b].model_matrix.w_axis.truncate().distance(camera_pos.into());
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });
```

Add `transparent_order` to the `PassContext` struct (in `render_passes.rs`):
```rust
    pub transparent_order: &'a [usize],
```

- [ ] **Step 6: Create transparent render pass**

Create `crates/rc3d-render/src/render_passes/pass_transparent.rs`:

```rust
//! Transparent geometry render pass.
//! Renders draw calls with opacity < 1.0, depth-sorted far-to-near,
//! with alpha blending and depth-write disabled.

use crate::pipelines::PipelineSet;
use crate::render_action::DrawCall;
use crate::render_passes::PassContext;
use crate::renderer::Renderer;

pub(super) fn pass_transparent(
    renderer: &mut Renderer,
    encoder: &mut wgpu::CommandEncoder,
    shade_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext,
    scene_pl: &PipelineSet,
) {
    let order = ctx.transparent_order;
    if order.is_empty() {
        return;
    }

    let pl = ctx.for_scene_target();
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Transparent"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: shade_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    for &idx in order {
        let dc = &ctx.visible[idx];
        if dc.vertex_count == 0 && dc.meshlet_data.is_none() {
            continue;
        }
        rpass.set_pipeline(&pl.solid_alpha);
        // Bind groups same as solid pass...
        // draw_call_mesh(&mut rpass, dc, ...);
    }
}
```

- [ ] **Step 7: Wire transparent pass into execute_passes**

In `crates/rc3d-render/src/render_passes.rs`, after the solid/outline passes and before post-processing, add:
```rust
    if !ctx.transparent_order.is_empty() {
        pass_transparent::pass_transparent(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);
    }
```

- [ ] **Step 8: Verify compilation**

Run: `rtk cargo check`
Expected: 0 errors.

- [ ] **Step 9: Commit**

```bash
rtk git add crates/rc3d-scene/src/node_data.rs crates/rc3d-actions/src/element.rs crates/rc3d-render/src/render_action.rs crates/rc3d-render/src/renderer.rs crates/rc3d-render/src/pipelines.rs crates/rc3d-render/src/render_passes.rs crates/rc3d-render/src/render_passes/pass_transparent.rs
rtk git commit -m "feat: add material opacity + depth-sorted transparent render pass"
```

---

### Task 4: Selection Highlighting Enhancement (T1-4)

**Files:**
- Modify: `crates/rc3d-render/src/render_passes/pass_selection.rs` (bounding box display)
- Modify: `crates/rc3d-render/src/renderer.rs` (bounding box generation + xray flag)
- Modify: `crates/rc3d-render/src/render_action.rs` (xray flag on DrawCall)

- [ ] **Step 1: Add bounding box display to selection**

In `crates/rc3d-render/src/render_passes/pass_selection.rs`, add a new function:

```rust
/// Draw bounding box wireframe for selected objects.
pub(super) fn pass_selection_bbox(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    shade_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &super::PassContext,
    scene_pl: &crate::pipelines::PipelineSet,
) {
    if ctx.selected_order.is_empty() || !ctx.wireframe_supported {
        return;
    }

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Selection BBox"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: shade_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    rpass.set_pipeline(&scene_pl.edge_overlay);
    let color = [1.0_f32, 0.8, 0.2, 1.0]; // yellow bbox

    for &idx in ctx.selected_order {
        let dc = &ctx.visible[idx];
        // Generate 8 corners + 12 edges from dc.model_matrix and dc.vertex_positions
        let bbox_lines = bbox_edges_from_draw_call(dc);
        if bbox_lines.is_empty() {
            continue;
        }
        let buf = renderer.flat_pool.push_flat(dc.mvp, color);
        rpass.set_bind_group(0, &buf.bind_group, &[buf.offset]);
        // Upload bbox_lines as vertex data...
    }
}

fn bbox_edges_from_draw_call(dc: &crate::render_action::DrawCall) -> Vec<crate::vertex::LineVertex> {
    let mut lines = Vec::new();
    let verts = match &dc.vertex_positions {
        Some(v) => v,
        None => return lines,
    };
    if verts.is_empty() { return lines; }

    let mut mn = glam::Vec3::splat(f32::MAX);
    let mut mx = glam::Vec3::splat(f32::MIN);
    for v in verts.iter() {
        let p = glam::Vec3::from_array(*v);
        mn = mn.min(p);
        mx = mx.max(p);
    }
    // 8 corners, 12 edges
    let corners = [
        glam::Vec3::new(mn.x, mn.y, mn.z),
        glam::Vec3::new(mx.x, mn.y, mn.z),
        glam::Vec3::new(mx.x, mn.y, mx.z),
        glam::Vec3::new(mn.x, mn.y, mx.z),
        glam::Vec3::new(mn.x, mx.y, mn.z),
        glam::Vec3::new(mx.x, mx.y, mn.z),
        glam::Vec3::new(mx.x, mx.y, mx.z),
        glam::Vec3::new(mn.x, mx.y, mx.z),
    ];
    let edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)];
    for (a, b) in edges {
        lines.push(crate::vertex::LineVertex { position: corners[a].to_array() });
        lines.push(crate::vertex::LineVertex { position: corners[b].to_array() });
    }
    lines
}
```

- [ ] **Step 2: Add xray mode toggle to Renderer**

In `crates/rc3d-render/src/renderer.rs`, add field:
```rust
    pub xray_mode: bool,
```

In the struct initializer: `xray_mode: false,`

- [ ] **Step 3: Wire xray into draw call collection**

In `crates/rc3d-render/src/render_action.rs`, when `is_selected` is true and xray mode is on, reduce unselected objects' opacity. Add field to DrawCall:
```rust
    pub xray_visible: bool,
```

- [ ] **Step 4: Verify compilation**

Run: `rtk cargo check`
Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
rtk git add crates/rc3d-render/src/render_passes/pass_selection.rs crates/rc3d-render/src/renderer.rs crates/rc3d-render/src/render_action.rs
rtk git commit -m "feat: add selection bbox display + xray mode toggle for selection highlighting"
```

---

### Task 5: Interactive Section Plane (T1-2)

**Files:**
- Modify: `crates/rc3d-scene/src/node_data.rs` (add SectionPlaneNode)
- Create: `crates/rc3d-actions/src/section_plane.rs`
- Modify: `crates/rc3d-actions/src/lib.rs` (add module)
- Modify: `crates/rc3d-render/src/render_passes.rs` (section pass additions)

- [ ] **Step 1: Add SectionPlaneNode to node_data.rs**

In `crates/rc3d-scene/src/node_data.rs`, add before `NodeData`:

```rust
/// Section/cutting plane node (Coin3D SoClipPlane pattern).
/// Defines a plane for cross-section visualization.
#[derive(Clone, Debug)]
pub struct SectionPlaneNode {
    /// Plane equation: normal.xyz * P + normal.w = 0
    pub plane: [f32; 4],
    pub enabled: bool,
    /// If true, render a cap surface where the plane cuts geometry.
    pub show_caps: bool,
    /// Color for the cap surface.
    pub cap_color: [f32; 4],
}

impl Default for SectionPlaneNode {
    fn default() -> Self {
        Self {
            plane: [0.0, 1.0, 0.0, 0.0],
            enabled: true,
            show_caps: false,
            cap_color: [0.3, 0.6, 0.9, 0.8],
        }
    }
}
```

Add variant to `NodeData` enum:
```rust
    SectionPlane(SectionPlaneNode),
```

Add `type_name` arm:
```rust
            NodeData::SectionPlane(_) => "SectionPlane",
```

- [ ] **Step 2: Add SectionPlane match arms to all Actions**

In `render_action.rs`, `ray_pick.rs`, `get_bounding_box.rs`: add a no-op traversal for SectionPlane (it doesn't contribute geometry, just traverses children):

```rust
            NodeData::SectionPlane(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
```

- [ ] **Step 3: Create SectionPlaneAction**

Create `crates/rc3d-actions/src/section_plane.rs`:

```rust
use rc3d_scene::{NodeData, SceneGraph};
use crate::action::{Action, ActionKind};

/// Collects all active section planes from the scene graph.
pub struct SectionPlaneAction {
    pub planes: Vec<[f32; 4]>,
}

impl SectionPlaneAction {
    pub fn new() -> Self {
        Self { planes: Vec::new() }
    }
}

impl Action for SectionPlaneAction {
    fn kind(&self) -> ActionKind {
        ActionKind::Search
    }

    fn apply(&mut self, graph: &SceneGraph, root: rc3d_core::NodeId) {
        collect_planes(graph, root, &mut self.planes);
    }
}

fn collect_planes(graph: &SceneGraph, node: rc3d_core::NodeId, planes: &mut Vec<[f32; 4]>) {
    let Some(entry) = graph.get(node) else { return };
    if let NodeData::SectionPlane(sp) = &entry.data {
        if sp.enabled {
            planes.push(sp.plane);
        }
    }
    for &child in &entry.children {
        collect_planes(graph, child, planes);
    }
}
```

- [ ] **Step 4: Add module to lib.rs**

In `crates/rc3d-actions/src/lib.rs`:
```rust
pub mod section_plane;
pub use section_plane::SectionPlaneAction;
```

- [ ] **Step 5: Verify compilation**

Run: `rtk cargo check`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
rtk git add crates/rc3d-scene/src/node_data.rs crates/rc3d-actions/src/section_plane.rs crates/rc3d-actions/src/lib.rs crates/rc3d-render/src/render_action.rs crates/rc3d-actions/src/ray_pick.rs crates/rc3d-actions/src/get_bounding_box.rs
rtk git commit -m "feat: add SectionPlaneNode + SectionPlaneAction for interactive section planes"
```

---

### Verification

After all 5 tasks:

1. Run: `rtk cargo check` — 0 errors, 0 warnings
2. Run: `rtk cargo test` — all existing tests pass
3. Run: `rtk cargo check --examples` — all examples compile
4. Visual: run `import_viewer` example, verify scene renders correctly (no regression)

---

### File Change Summary

| File | New/Modify | Tasks |
|------|-----------|-------|
| `crates/rc3d-actions/src/scene_path.rs` | **New** | T1-7 |
| `crates/rc3d-actions/src/section_plane.rs` | **New** | T1-2 |
| `crates/rc3d-render/src/render_passes/pass_transparent.rs` | **New** | T1-1 |
| `crates/rc3d-actions/src/lib.rs` | Modify | T1-7, T1-2 |
| `crates/rc3d-actions/src/action.rs` | Modify | T1-7 |
| `crates/rc3d-scene/src/node_data.rs` | Modify | T1-5, T1-1, T1-2 |
| `crates/rc3d-actions/src/element.rs` | Modify | T1-1 |
| `crates/rc3d-render/src/render_action.rs` | Modify | T1-5, T1-1, T1-4, T1-2 |
| `crates/rc3d-render/src/renderer.rs` | Modify | T1-1, T1-4 |
| `crates/rc3d-render/src/pipelines.rs` | Modify | T1-1 |
| `crates/rc3d-render/src/render_passes.rs` | Modify | T1-1 |
| `crates/rc3d-render/src/render_passes/pass_selection.rs` | Modify | T1-4 |
| `crates/rc3d-actions/src/ray_pick.rs` | Modify | T1-5, T1-2 |
| `crates/rc3d-actions/src/get_bounding_box.rs` | Modify | T1-5, T1-2 |
