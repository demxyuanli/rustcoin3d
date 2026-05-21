# Code Audit Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all critical and high-risk issues identified in the code audit of rc3d-render

**Architecture:** Three-phase approach:
1. Phase 1: Critical memory safety fixes (dangling pointer, unsafe Send)
2. Phase 2: Resource management fixes (buffer pooling, pool overflow handling)
3. Phase 3: Error handling improvements (replace expect/unwrap with proper error types)

**Tech Stack:** Rust, WGPU, bytemuck, thiserror

---

## Phase 1: Critical Memory Safety Fixes

### Task 1: Fix Dangling Pointer in render_passes.rs

**Files:**
- Modify: `crates/rc3d-render/src/render_passes.rs:170-191`
- Modify: `crates/rc3d-render/src/render_passes.rs:555-577`
- Modify: `crates/rc3d-render/src/selection_outline.rs:763`

**Step 1: Analyze current pointer usage**

Read the current implementation to understand how `scene_tex_raw` is used:
```
crates/rc3d-render/src/render_passes.rs lines 555-577
crates/rc3d-render/src/selection_outline.rs lines 755-780
```

**Step 2: Replace pointer with TextureView clone**

Instead of passing `*const wgpu::Texture`, pass a cloned `wgpu::TextureView` that has proper lifetime management.

In `render_passes.rs`, change:
```rust
let (eff_width, eff_height, scene_tex_raw): (u32, u32, *const wgpu::Texture) = match &presentation {
```

To:
```rust
let (eff_width, eff_height, scene_tex): (u32, u32, wgpu::TextureView) = match &presentation {
```

**Step 3: Update selection_outline function signature**

Modify `selection_outline::encode_selection_outline_pass` to accept `&wgpu::TextureView` instead of `*const wgpu::Texture`.

**Step 4: Update all call sites**

Search for all usages of `scene_tex_raw` and update accordingly.

**Step 5: Test compilation**

Run: `cargo check --package rc3d-render`
Expected: No errors related to pointer changes

---

### Task 2: Fix unsafe impl Send in render_action.rs

**Files:**
- Modify: `crates/rc3d-render/src/render_action.rs:270-285`

**Step 1: Understand the CacheTarget usage**

Search for where `RenderCollector` is used and whether it's ever accessed from multiple threads:
```
grep -n "RenderCollector" crates/rc3d-render/src/*.rs
```

**Step 2: Remove unsafe impl Send**

If `RenderCollector` is strictly single-threaded (which the comment claims), wrap it in a `PhantomData<&mut ()>` marker instead of manual `Send` impl:

```rust
struct CacheTarget {
    ptr: *mut crate::flat_draw_cache::FlatDrawCache,
    _marker: std::marker::PhantomData<&mut ()>,
}

unsafe impl Send for CacheTarget {}  // REMOVE THIS LINE
```

Replace with:
```rust
struct CacheTarget {
    ptr: *mut crate::flat_draw_cache::FlatDrawCache,
    _marker: std::marker::PhantomData<&mut ()>,  // Ensures !Send
}
```

**Step 3: Update get_mut to use unsafe block with contract**

```rust
unsafe fn get_mut(&self) -> &mut crate::flat_draw_cache::FlatDrawCache {
    &mut *self.0
}
```

Add contract comment explaining the safety invariant.

**Step 4: Verify no thread-safety issues**

Run: `cargo check --package rc3d-render --features profiler`
Expected: No new warnings

---

### Task 3: Fix Omni Shadow Buffer Allocation (shadow_omni.rs)

**Files:**
- Modify: `crates/rc3d-render/src/shadow_omni.rs:135-250` (OmniShadowRenderer)
- Modify: `crates/rc3d-render/src/shadow_omni.rs:252-348` (render_omni_shadow_pass)

**Step 1: Add pre-allocated uniform buffer to OmniShadowRenderer**

In `OmniShadowRenderer::new`, a `uniform_buffer` is already created (line 139). We need to use it.

**Step 2: Add per-face bind group storage**

Add a cached array of bind groups:
```rust
pub struct OmniShadowRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub shadow_bgl: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    omni_resource_bgl: wgpu::BindGroupLayout,
    // NEW: Cache bind groups for each face
    face_bind_groups: [wgpu::BindGroup; 6],
}
```

**Step 3: Create bind groups once in constructor**

In `OmniShadowRenderer::new()`, after creating `uniform_buffer`, create all 6 face bind groups.

**Step 4: Modify render_omni_shadow_pass to reuse**

Replace the per-face buffer/bindgroup creation with:
```rust
for face in 0..6u32 {
    // Update uniform buffer with current face's VP matrix
    renderer.queue.write_buffer(
        &omni.uniform_buffer,
        0,
        bytemuck::bytes_of(&OmniShadowUniforms { ... }),
    );

    // Use cached bind group
    let ubg = &omni.face_bind_groups[face as usize];

    // ... rest of render pass using ubg
}
```

**Step 5: Test compilation**

Run: `cargo check --package rc3d-render`
Expected: Compiles without errors

---

## Phase 2: Resource Management Fixes

### Task 4: Fix Pool Overflow Handling

**Files:**
- Modify: `crates/rc3d-render/src/selection_outline.rs:824`
- Modify: `crates/rc3d-render/src/gpu_resource.rs` (GpuUniformPool)

**Step 1: Add overflow logging to flat_pool.push_flat**

When pool is full, log a warning instead of silently dropping:
```rust
pub fn push_flat(&mut self, uniform: &FlatUniforms) -> Option<NonZeroU32> {
    // ... existing code ...
    if offset.is_none() {
        log::warn!("flat_pool overflow: {} entries, cursor={}, capacity={}",
            self.stride, self.cursor, self.capacity);
    }
    offset
}
```

**Step 2: Search for other pool push sites**

```bash
grep -n "pool.push" crates/rc3d-render/src/*.rs
```

Add similar overflow warnings to all pool push sites.

**Step 3: Consider increasing default pool sizes**

Check current pool sizes in `renderer.rs`:
```bash
grep -n "pool" crates/rc3d-render/src/renderer.rs | head -20
```

If overflows are frequent, increase pool capacity.

**Step 4: Test with existing tests**

Run: `cargo test --package rc3d-render`
Expected: All tests pass

---

### Task 5: Fix Velocity Buffer Per-Frame Allocation

**Files:**
- Modify: `crates/rc3d-render/src/render_passes.rs:764-797`

**Step 1: Add velocity_params_buffer to GpuInternals**

In `renderer.rs`, add a persistent buffer:
```rust
pub struct GpuInternals {
    // ... existing fields ...
    pub velocity_params_buffer: Option<wgpu::Buffer>,
}
```

**Step 2: Initialize buffer in renderer setup**

```rust
if self.enable_motion_blur || self.enable_taa {
    self.gpu.velocity_params_buffer = Some(self.device.create_buffer(
        &wgpu::BufferDescriptor {
            label: Some("Velocity Params"),
            size: 144,  // mat4x4 + mat4x4 + vec2 + vec2
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }
    ));
}
```

**Step 3: Replace buffer creation with write_buffer**

In `render_passes.rs`, replace:
```rust
let vel_uniform = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { ... });
```

With:
```rust
renderer.queue.write_buffer(
    renderer.gpu.velocity_params_buffer.as_ref().unwrap(),
    0,
    &vel_data,
);
```

**Step 4: Test**

Run: `cargo check --package rc3d-render`

---

## Phase 3: Error Handling Improvements

### Task 6: Create RenderError Type

**Files:**
- Create: `crates/rc3d-render/src/error.rs`
- Modify: `crates/rc3d-render/src/lib.rs`

**Step 1: Define RenderError enum**

```rust
#[derive(thiserror::Error, Debug)]
pub enum RenderError {
    #[error("Texture cache error: {0}")]
    TextureCache(String),

    #[error("Pool overflow: {0}")]
    PoolOverflow(&'static str),

    #[error("Device error: {0}")]
    Device(String),

    #[error("Surface error: {0}")]
    Surface(String),
}
```

**Step 2: Export in lib.rs**

```rust
pub mod error;
pub use error::RenderError;
```

**Step 3: Verify compilation**

Run: `cargo check --package rc3d-render`

---

### Task 7: Replace Critical expect/unwrap with Proper Error Handling

**Files:**
- Modify: `crates/rc3d-render/src/render_passes.rs:236, 561-563, 593`
- Modify: `crates/rc3d-render/src/texture_cache.rs:348-352`

**Step 1: Identify critical expect sites**

Replace `expect()` with proper error handling in these categories:
1. Initialization failures (device creation, surface creation)
2. Resource lookups that should always succeed
3. Critical path operations

**Step 2: Replace with unwrap_or_else or expect**

For texture_cache handle lookups, use:
```rust
let albedo_view = self.textures.get(albedo_handle)
    .map(|t| &t.view)
    .ok_or_else(|| RenderError::TextureCache(
        format!("Missing albedo handle: {:?}", albedo_handle)
    ))?;
```

**Step 3: Batch test**

Run: `cargo check --package rc3d-render`
Expected: No new errors

---

### Task 8: Fix Atomic Ordering in render_action.rs

**Files:**
- Modify: `crates/rc3d-render/src/render_action.rs:87-97`

**Step 1: Fix ordering**

Change `Relaxed` to `SeqCst`:
```rust
pub fn set_feature_crease_angle(deg: f32) {
    FEATURE_CREASE_ANGLE_BITS.store(deg.to_bits(), std::sync::atomic::Ordering::SeqCst);
}
```

**Step 2: Check other atomic usages**

```bash
grep -n "atomic::Ordering" crates/rc3d-render/src/*.rs
```

Verify all other usages are correct.

---

## Phase 4: Low Priority Cleanups

### Task 9: Extract Hardcoded Constants

**Files:**
- Modify: `crates/rc3d-render/src/render_passes.rs`
- Create: `crates/rc3d-render/src/constants.rs`

**Step 1: Create constants module**

```rust
// crates/rc3d-render/src/constants.rs

/// Maximum number of point/spot lights per cluster
pub const MAX_CLUSTER_LIGHTS: usize = 256;

/// Shadow map bias values
pub const SHADOW_BIAS_CONSTANT: f32 = 2.0;
pub const SHADOW_BIAS_SLOPE_SCALE: f32 = 2.0;

/// Volumetric fog parameters
pub const VOL_FOG_SAMPLE_COUNT: u32 = 32;

/// Light type constants (matches shader values)
pub const LIGHT_TYPE_POINT: f32 = 1.0;
pub const LIGHT_TYPE_SPOT: f32 = 3.0;
```

**Step 2: Replace magic numbers**

Replace all hardcoded values in render_passes.rs and shadow_omni.rs with constants.

**Step 3: Test**

Run: `cargo check --package rc3d-render`

---

### Task 10: Remove dead_code Attributes

**Files:**
- Modify: `crates/rc3d-render/src/render_passes.rs:38-58`

**Step 1: Verify field usage**

Search for each `#[allow(dead_code)]` field usage:
```bash
grep -n "csm_split_depths\|shadow_params\|transparent_order" crates/rc3d-render/src/render_passes.rs
```

**Step 2: Remove or document**

If truly unused, remove the field. If intended for future use, document why.

---

## Verification

### Final Test Commands

```bash
# Full workspace check
cargo check --workspace

# Run all tests
cargo test --workspace

# Run clippy
cargo clippy --workspace -- -D warnings

# Run doc tests
cargo doc --workspace --no-deps
```

---

## Notes

- **Task ordering**: Execute Phase 1 tasks first as they fix critical memory safety issues
- **Risk assessment**: Phase 1 changes are invasive; test thoroughly after each task
- **Rollback plan**: Use `git stash` to preserve original code before making changes
