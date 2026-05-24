# Annotation Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Annotations follow animated models and disappear when they face away from the camera or leave the viewport.

**Architecture:** Add `AnnotationVisibility` field to `ProjectedAnnotation`, compute back-face + NDC culling in `project_annotation_elements`. Cache projected markup data in `FrameState` and skip CPU projection work on static frames.

**Tech Stack:** Rust, glam, wgpu, rc3d-scene annotation subsystem

---

### Task 1: Add `AnnotationVisibility` and plane normal helper

**Files:**
- Modify: `crates/rc3d-render/src/render_passes/pass_effects/collect.rs`
- Modify: `crates/rc3d-render/src/render_passes/pass_markup/projection.rs`

- [ ] **Step 1: Add `AnnotationVisibility` to `ProjectedAnnotation`**

In `collect.rs`, add before `ProjectedAnnotation`:

```rust
/// Visibility flags computed per-annotation during projection.
#[derive(Clone, Debug, Default)]
pub struct AnnotationVisibility {
    pub back_facing: bool,
    pub outside_ndc: bool,
    /// Deferred: occlusion check (Phase 1.5)
    pub occluded: bool,
}
```

Add a `visibility` field to `ProjectedAnnotation`:

```rust
#[derive(Clone, Debug)]
pub struct ProjectedAnnotation {
    pub element: AnnotationElement,
    pub model_matrix: Mat4,
    pub style: AnnotationStyle,
    pub visibility: AnnotationVisibility,
}
```

In `collect.rs`, the push site initializes with `Default::default()`:

```rust
// Line ~155 in collect.rs, add visibility field:
commands.annotation_elements.push(ProjectedAnnotation {
    element,
    model_matrix: el_model,
    style: ann.style.clone(),
    visibility: AnnotationVisibility::default(),
});
```

- [ ] **Step 2: Add `annotation_plane_normal` to projection.rs**

At the end of `projection.rs`, add:

```rust
/// Compute the annotation-plane normal in model-local space.
/// Returns `None` for types without a well-defined plane (Leader, Callout).
use rc3d_scene::node_data::AnnotationElement;

pub(crate) fn annotation_plane_normal(element: &AnnotationElement) -> Option<glam::Vec3> {
    match element {
        AnnotationElement::Dimension { start, end, offset_dir, .. } => {
            let s = glam::Vec3::from(start.coords());
            let e = glam::Vec3::from(end.coords());
            let off = glam::Vec3::from(*offset_dir);
            let n = (e - s).cross(off);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::AngleDimension { center, arm1, arm2, .. } => {
            let c = glam::Vec3::from(center.coords());
            let a1 = glam::Vec3::from(arm1.coords());
            let a2 = glam::Vec3::from(arm2.coords());
            let n = (a1 - c).cross(a2 - c);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::RadialDimension { center, perimeter, .. } => {
            let d = glam::Vec3::from(perimeter.coords()) - glam::Vec3::from(center.coords());
            if d.length_squared() > 1e-8 { Some(d.normalize()) } else { None }
        }
        AnnotationElement::DiameterDimension { center, p1, p2, .. } => {
            let c = glam::Vec3::from(center.coords());
            let a = glam::Vec3::from(p1.coords());
            let b = glam::Vec3::from(p2.coords());
            let n = (a - c).cross(b - c);
            if n.length_squared() > 1e-8 { Some(n.normalize()) } else { None }
        }
        AnnotationElement::Datum { .. } => Some(glam::Vec3::Z),
        AnnotationElement::Leader { .. } | AnnotationElement::Callout { .. } => None,
    }
}
```

- [ ] **Step 3: Build to verify compilation**

```bash
cargo check -p rc3d-render 2>&1
```

Expected: 0 errors

- [ ] **Step 4: Commit**

```bash
git add crates/rc3d-render/src/render_passes/pass_effects/collect.rs crates/rc3d-render/src/render_passes/pass_markup/projection.rs
git commit -m "feat: add AnnotationVisibility and plane normal helper for annotation culling"
```

---

### Task 2: Add visibility culling to `project_annotation_elements`

**Files:**
- Modify: `crates/rc3d-render/src/render_passes/pass_markup/primitives.rs`
- Modify: `crates/rc3d-render/src/render_passes/pass_markup/mod.rs`

- [ ] **Step 1: Add `camera_pos` parameter to `project_annotation_elements`**

In `primitives.rs`, change the function signature:

```rust
pub(super) fn project_annotation_elements(
    elements: &[ProjectedAnnotation],
    scene_vp: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
    camera_pos: glam::Vec3,
    labels: &mut Vec<WorldLabelCommand>,
) -> Vec<MarkupVertex> {
```

- [ ] **Step 2: Add culling helpers just before `project_annotation_elements`**

```rust
fn element_midpoint_world(element: &AnnotationElement, model: glam::Mat4) -> glam::Vec3 {
    let pts: &[[f32; 3]] = match element {
        AnnotationElement::Dimension { start, end, .. } => &[start.coords(), end.coords()][..],
        AnnotationElement::AngleDimension { center, .. } => &[center.coords()][..],
        AnnotationElement::RadialDimension { center, .. } => &[center.coords()][..],
        AnnotationElement::DiameterDimension { center, .. } => &[center.coords()][..],
        AnnotationElement::Leader { anchor, .. } => &[anchor.coords()][..],
        AnnotationElement::Callout { anchor, .. } => &[anchor.coords()][..],
        AnnotationElement::Datum { position, .. } => &[position.coords()][..],
    };
    let mut sum = glam::Vec3::ZERO;
    for p in pts { sum += model.transform_point3(glam::Vec3::from(*p)); }
    sum / pts.len() as f32
}

fn element_anchor_ndc(
    element: &AnnotationElement,
    model: glam::Mat4,
    scene_vp: glam::Mat4,
    depth_reversed_z: bool,
) -> Option<[f32; 3]> {
    let anchor = match element {
        AnnotationElement::Dimension { start, .. } => start.coords(),
        AnnotationElement::AngleDimension { center, .. } => center.coords(),
        AnnotationElement::RadialDimension { center, .. } => center.coords(),
        AnnotationElement::DiameterDimension { center, .. } => center.coords(),
        AnnotationElement::Leader { anchor, .. } => anchor.coords(),
        AnnotationElement::Callout { anchor, .. } => anchor.coords(),
        AnnotationElement::Datum { position, .. } => position.coords(),
    };
    project_point_ndc(
        glam::Vec3::from(anchor),
        model,
        scene_vp,
        depth_reversed_z,
    )
}
```

- [ ] **Step 3: Insert culling logic into the loop body**

In `project_annotation_elements`, after `for pa in elements {` and before `let proj_ndc = ...`, insert:

```rust
for pa in elements {
    let model = pa.model_matrix;
    let style = &pa.style;

    // --- visibility culling ---
    let mut visibility = AnnotationVisibility::default();
    if let Some(plane_n) = projection::annotation_plane_normal(&pa.element) {
        let world_n = model.transform_vector3(plane_n);
        let midpoint = element_midpoint_world(&pa.element, model);
        let to_camera = camera_pos - midpoint;
        visibility.back_facing = world_n.dot(to_camera) < 0.0;
    }
    visibility.outside_ndc = element_anchor_ndc(
        &pa.element, model, scene_vp, depth_reversed_z,
    ).is_none();
    if visibility.back_facing || visibility.outside_ndc {
        continue;
    }
    // --- end culling ---

    let proj_ndc = |p: &[f32; 3]| { ... original code ... };
```

Note: `projection::annotation_plane_normal` needs to be either imported or use the full path `super::projection::annotation_plane_normal`.

- [ ] **Step 4: Update call site in `pass_markup` (mod.rs)**

Change the call at line ~236:

```rust
let projected = project_annotation_elements(
    &effect_commands.annotation_elements,
    scene_vp,
    surface_w as f32,
    surface_h as f32,
    depth_reversed_z,
    renderer.frame.scene_camera_pos,   // was: &mut world_labels
    &mut world_labels,
);
```

- [ ] **Step 5: Build and run tests**

```bash
cargo check -p rc3d-render 2>&1
cargo test -p rc3d-render pass_effects 2>&1
cargo test -p rc3d-render pass_text 2>&1
cargo test -p rc3d-scene annotation 2>&1
```

Expected: 0 errors, all tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/rc3d-render/src/render_passes/pass_markup/primitives.rs crates/rc3d-render/src/render_passes/pass_markup/mod.rs
git commit -m "feat: add back-face and NDC visibility culling for annotations"
```

---

### Task 3: Static frame caching for annotation projection

**Files:**
- Modify: `crates/rc3d-render/src/renderer_internals.rs`
- Modify: `crates/rc3d-render/src/render_passes.rs`

- [ ] **Step 1: Add cached fields to `FrameState`**

In `renderer_internals.rs`, after `annotation_world_labels` (line ~128), add:

```rust
/// Cached projected markup vertices (for static-frame fast path).
pub cached_projected_markup: Vec<crate::vertex::MarkupVertex>,
/// Cached world labels (for static-frame fast path).
pub cached_projected_labels: Vec<crate::world_label::WorldLabelCommand>,
```

- [ ] **Step 2: Initialize the cached fields**

In `renderer.rs` around line 914 where `FrameState` is constructed, add the two new fields after `annotation_world_labels`:

```rust
cached_projected_markup: Vec::new(),
cached_projected_labels: Vec::new(),
```

- [ ] **Step 3: Cache projected output after pass_markup**

In `render_passes.rs`, after the `pass_markup` call (line ~613-623), capture the projected data:

The issue is that `pass_markup` currently takes `renderer.frame.annotation_world_labels` via `std::mem::take` and consumes them. The projected vertices are local variables within `pass_markup`. We need to refactor to return the projected data for caching.

Refactor: Extract the projection logic from `pass_markup` into a separate function that returns both the projected vertices and labels, then pass them into the GPU pass.

In `pass_markup/mod.rs`, add a new function:

```rust
/// Compute projected render data for annotations (CPU side).
/// Returns None when there's nothing to draw.
pub fn compute_projected_markup(
    renderer: &mut crate::renderer::Renderer,
    effect_commands: &EffectCommands,
    scene_vp: glam::Mat4,
    surface_w: f32,
    surface_h: f32,
    depth_reversed_z: bool,
) -> (Vec<MarkupVertex>, Vec<crate::world_label::WorldLabelCommand>) {
    let mut world_labels = std::mem::take(&mut renderer.frame.annotation_world_labels);
    let projected = project_annotation_elements(
        &effect_commands.annotation_elements,
        scene_vp,
        surface_w,
        surface_h,
        depth_reversed_z,
        renderer.frame.scene_camera_pos,
        &mut world_labels,
    );
    (projected, world_labels)
}
```

Then refactor `pass_markup` to accept the pre-computed data:

```rust
pub fn pass_markup(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    surface_w: u32,
    surface_h: u32,
    scene_vp: glam::Mat4,
    depth_reversed_z: bool,
    projected: &[MarkupVertex],
    world_labels: &[crate::world_label::WorldLabelCommand],
) {
    let legacy = &renderer.frame.markup_vertices;
    if projected.is_empty() && legacy.is_empty() && world_labels.is_empty() {
        return;
    }

    // 3D annotations: ... same as before but using the passed-in projected + world_labels ...
    if !projected.is_empty() || !world_labels.is_empty() {
        // ... same rendering code ...
        if !projected.is_empty() {
            // draw projected MarkupVertex lines ...
        }
        if !world_labels.is_empty() {
            // draw world labels ...
        }
    }
    // Legacy 2D MarkupNode overlay: same as before ...
}
```

- [ ] **Step 4: Wire static frame detection in `render_passes.rs`**

In `execute_passes` (and `render_overlay_only_frame`), replace the single `pass_markup` call with:

```rust
// Static frame: reuse cached projection output.
let is_static = renderer.frame.bvh_fully_static && renderer.frame.static_frame_count >= 2;
if !is_static {
    let (projected, wl) = pass_markup::compute_projected_markup(
        renderer,
        ctx.effect_commands,
        ctx.scene_vp,
        ew as f32,
        eh as f32,
        ctx.depth_reversed_z,
    );
    renderer.frame.cached_projected_markup = projected;
    renderer.frame.cached_projected_labels = wl;
}
pass_markup::pass_markup(
    renderer,
    &mut encoder,
    view,
    &depth_view,
    ew,
    eh,
    ctx.scene_vp,
    ctx.depth_reversed_z,
    &renderer.frame.cached_projected_markup,
    &renderer.frame.cached_projected_labels,
);
```

- [ ] **Step 5: Build and run full test suite**

```bash
cargo check --workspace 2>&1
cargo test -p rc3d-render 2>&1
```

Expected: 0 errors, all tests pass (including pass_markup tests if any)

- [ ] **Step 6: Commit**

```bash
git add crates/rc3d-render/src/renderer_internals.rs crates/rc3d-render/src/renderer.rs crates/rc3d-render/src/render_passes/pass_markup/mod.rs crates/rc3d-render/src/render_passes.rs
git commit -m "perf: cache annotation projection on static frames"
```

---

### Task 4: Add node-bound annotation to markup_dimensions example

**Files:**
- Modify: `crates/rc3d-examples/examples/markup_dimensions.rs`

- [ ] **Step 1: Add a node-bound Dimension to the animated cube**

After the existing AnnotationSet at line ~74 (after the Datum element), add a second AnnotationSet bound to the cube Transform:

```rust
// Node-bound annotation: follows the animated cube
graph.add_child(
    cube_tf,
    NodeData::AnnotationSet(AnnotationSetNode {
        style: style.clone(),
        elements: vec![
            AnnotationElement::Dimension {
                start: AnnotationPoint::on_node(cube_tf, [hw, 0.0, hw]),
                end: AnnotationPoint::on_node(cube_tf, [hw, 0.0, -hw]),
                offset_dir: [1.0, 0.0, 0.0],
                extension_len: 0.2,
                arrow_size: 0.12,
                label: String::new(),
                label_mode: AnnotationLabelMode::Auto,
                color: [1.0, 0.8, 0.0, 1.0],
            },
        ],
        visible: true,
    }),
);
```

The `AnnotationPoint::on_node(cube_tf, ...)` binds the points to the animated cube Transform. Use the existing `use rc3d_scene::annotation::AnnotationPoint;` import.

- [ ] **Step 2: Build the example**

```bash
cargo check -p rc3d-examples --example markup_dimensions 2>&1
```

Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
git add crates/rc3d-examples/examples/markup_dimensions.rs
git commit -m "example: add node-bound annotation to animated cube in markup_dimensions"
```

---

### Task 5: Visual validation and final verification

- [ ] **Step 1: Run the example**

```bash
cargo run -p rc3d-examples --example markup_dimensions
```

Verify:
- The yellow Dimension follows the bobbing cube (not the orange static annotations)
- Orbit camera behind the cube — the annotation disappears (back-face culling)
- Orbit to make the cube leave the viewport — annotation disappears (NDC culling)

- [ ] **Step 2: Run full test suite**

```bash
cargo test --workspace 2>&1
```

All tests pass.

- [ ] **Step 3: Commit any final adjustments**

```bash
git commit -am "chore: final verification of annotation Phase 1"
```
