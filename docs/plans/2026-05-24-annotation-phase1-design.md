# Annotation Phase 1: Node Binding + Visibility

**Date**: 2026-05-24  
**Status**: design  
**Scope**: `rc3d-scene` / `rc3d-render` / `rc3d-examples`

## Goal

Annotations follow animated models and disappear when they face away from the camera or leave the viewport. Foundation for interactive editing (Phase 2).

## Current State

### Working
- 7 annotation element types with geometry + label rendering
- `AnnotationPoint { node: Option<NodeId>, local: [f32; 3] }` — node binding via `node_world_matrix()`
- `effective_annotation_model()` combines set_matrix with node_world
- `prepare_annotation_for_render()` produces `(localized_element, model_matrix)` for the renderer
- Unit tests for node binding pass

### Gaps
- `markup_dimensions` example uses only bare local coords — no node-bound annotations exercised
- No visibility culling: annotations behind the camera, outside the viewport, or occluded by geometry still render
- No caching: annotation positions are re-resolved every frame even on static scenes

## Design

### 1. Audit and unify render path

Verify that the full pipeline (`collect.rs` → `project_annotation_elements`) uses `prepare_annotation_for_render` exclusively. Remove or downgrade the legacy `resolve_element` if it is unused.

Add a node-bound annotation to `markup_dimensions` example: attach a Dimension to the animated cube's Transform node so the label follows the geometry.

### 2. Visibility culling

Add `AnnotationVisibility` to `ProjectedAnnotation`:

```rust
struct AnnotationVisibility {
    back_facing: bool,
    outside_ndc: bool,
    occluded: bool,
}
```

Culling stages (executed in `project_annotation_elements`):

| Stage | Check | Cost |
|-------|-------|------|
| Back-face | annotation-plane normal dot to_camera < 0 | cheap (dot product) |
| NDC bounds | label anchor NDC outside [-1, 1] or z outside [0, 1] | cheap |
| Occlusion | sample depth buffer at anchor NDC; compare to computed depth | expensive (GPU readback or deferred check) |

Back-face and NDC bounds run every frame. Occlusion is deferred to Phase 1.5 (needs depth buffer readback plumbing) — annotations behind geometry are an edge case for static scenes.

### 3. Skip reprojection on static frames

When the renderer detects a static frame (camera + scene unchanged), skip `project_annotation_elements` and reuse the previous frame's projected vertices and world labels. The existing `StaticFrameTracker` or equivalent mechanism in `render_passes.rs` already tracks this state.

A `bool` flag on the frame struct: `annotation_dirty` — set on scene mutation or camera change, cleared after projection.

### 4. Example validation

Modify `markup_dimensions` to add at least one Dimension bound to the animated cube Transform node. Run the example and verify:
- Label follows the bobbing cube
- Label disappears when camera faces away from the annotation plane
- Label disappears when the cube moves off screen

## Files

| File | Change |
|------|--------|
| `render_passes/pass_effects/collect.rs` | Audit resolve path |
| `render_passes/pass_markup/primitives.rs` | Add visibility checks |
| `render_passes/pass_markup/mod.rs` | Wire static-frame skip |
| `renderer_internals.rs` | Add `annotation_dirty` flag |
| `examples/markup_dimensions.rs` | Add node-bound annotation |

## Acceptance

1. `cargo run -p rc3d-examples --example markup_dimensions` shows annotation following the animated cube
2. Annotation hidden when camera looks at cube from behind (annotation plane back-facing)
3. `cargo test -p rc3d-render pass_effects` — existing tests pass
4. `cargo test -p rc3d-scene annotation` — existing tests pass
5. Static-frame fast path: annotations don't re-project on consecutive identical frames
