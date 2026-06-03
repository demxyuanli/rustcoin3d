# A5: Extrusion Face Mesh Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route extrusion surfaces through the ruled-surface mesh path instead of generic CDT, improving mesh quality for extruded shapes.

**Architecture:** Add `SurfaceGeom::Extrusion` to the `prefers_native_uv_ruled()` check and the ruled-surface dispatch. Extrusion is naturally a ruled surface (u = generatrix parameter, v = linear extrusion direction), so the existing two-wire ruled mesh algorithm applies directly.

**Tech Stack:** Rust, rc3d-shape crate

---

### Task 1: Add Extrusion to ruled-surface preference

**Files:**
- Modify: `crates/rc3d-shape/src/mesh/face_fill.rs:429-437` (`surface_is_revolution_like`)
- Modify: `crates/rc3d-shape/src/mesh/face_dispatch.rs:70-81` (`prefers_native_uv_trim`)

- [ ] **Step 1: Add Extrusion to `surface_is_revolution_like`**

In `crates/rc3d-shape/src/mesh/face_fill.rs`, update the `surface_is_revolution_like` function to include Extrusion:

```rust
pub(crate) fn surface_is_revolution_like(surface: &SurfaceGeom) -> bool {
    match surface {
        SurfaceGeom::Revolution { .. } | SurfaceGeom::Extrusion { .. } => true,
        SurfaceGeom::Offset { basis, .. } => {
            matches!(basis.as_ref(), SurfaceGeom::Revolution { .. })
        }
        _ => false,
    }
}
```

- [ ] **Step 2: Add Extrusion to `prefers_native_uv_trim`**

In `crates/rc3d-shape/src/mesh/face_dispatch.rs`, add Extrusion to the matches! list:

```rust
pub fn prefers_native_uv_trim(surface: &SurfaceGeom) -> bool {
    matches!(
        surface,
        SurfaceGeom::Revolution { .. }
            | SurfaceGeom::BSpline(_)
            | SurfaceGeom::Offset { .. }
            | SurfaceGeom::Cylinder { .. }
            | SurfaceGeom::Cone { .. }
            | SurfaceGeom::Torus { .. }
            | SurfaceGeom::Extrusion { .. }
    )
}
```

- [ ] **Step 3: Run tests to verify no regressions**

Run: `cargo test -p rc3d-shape`
Expected: All existing tests pass (237 tests). The new match arms only expand the set of surfaces routed to ruled/trim paths — no behavioral change for existing surface types.

- [ ] **Step 4: Commit**

```bash
git add crates/rc3d-shape/src/mesh/face_fill.rs crates/rc3d-shape/src/mesh/face_dispatch.rs
git commit -m "feat(mesh): route Extrusion surfaces through ruled-surface mesh path"
```
