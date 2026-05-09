# Optimization Guide: Memory & Performance Features

Guide to the optimization infrastructure added in May 2026. Covers GPU-driven
culling, mesh pooling, static frame fast path, and shared utilities.

## Enabling Features

### Performance Settings

All features are controlled via `RenderSettings::performance`:

```rust
use rc3d_render::settings::{RenderSettings, PerformanceSettings};

let settings = RenderSettings {
    performance: PerformanceSettings {
        parallel_traversal: true,       // rayon-based scene traversal
        gpu_culling: true,              // compute shader frustum cull
        gpu_culling_threshold: 4096,    // min objects for GPU path
        mesh_pool_capacity: 4096,       // streaming mesh pool size
        mesh_pool_max_mb: 512,          // GPU memory budget for meshes
    },
    ..Default::default()
};
renderer.apply_settings(settings);

// Allocate GPU buffers for culling (required after settings)
renderer.enable_gpu_culling(65536);
```

### Toggle at Runtime

```rust
// GPU culling on/off
renderer.enable_gpu_culling(65536);
// Parallel traversal
renderer.set_parallel_traversal(true);
// Light set table (auto-populated during traversal)
renderer.set_light_sets(collector.light_sets);
```

## GPU Compute Culling

### Architecture

```
CPU: upload transforms → transform_buffer (128B/object, STORAGE)
GPU: object_cull compute → reads transform_buffer + frustum uniform
                        → writes instance_indices + instance_count
CPU: staging readback → replaces BVH culling results
```

### How It Works

1. **Frame N**: CPU uploads all object transforms to `transform_buffer`
2. **Frame N**: GPU cull compute runs, writing visible indices to `instance_indices_buffer`
3. **Frame N**: Results copied to staging buffer
4. **Frame N+1**: CPU reads staging → rebuilds visible set from GPU indices
5. **Frame N+1**: Normal rendering pipeline uses GPU-derived visibility

### Requirements

- `gpu_culling_threshold`: Only activates for scenes with more than this many objects
- `gpu_cull_ready` flag ensures frame-delayed synchronization
- Falls back to CPU BVH culling when GPU results are stale

### Limitations

- 1-frame latency (results from frame N used on frame N+1)
- Not suitable for scenes with rapid camera movement
- Indirect draw integration pending (currently uses readback)

## Streaming Mesh Pool

### Architecture

```
GpuMeshPool: fixed-size array of GpuMeshSlot
  ├── hash_to_slot: LruCache<u64, usize>  (mesh_hash → slot index)
  ├── free_slots: Vec<usize>              (available slots)
  ├── total_bytes: u64                    (GPU memory used)
  └── max_bytes: u64                      (budget limit)

Upload flow:
  1. Check mesh_hash in pool → hit: promote to MRU, return slot
  2. Miss: compute new_bytes for buffers
  3. Evict LRU entries until total_bytes + new_bytes ≤ max_bytes
  4. Create GPU buffers, insert slot, update total_bytes
```

### Budget Management

```rust
// Set 256 MB GPU budget for meshes
renderer.gpu.assets.mesh_pool.as_mut()
    .map(|p| p.set_max_bytes(256 * 1024 * 1024));
```

The pool automatically evicts least-recently-used meshes when:
- Uploading a new mesh would exceed `max_bytes`
- Pool slot count exceeds `capacity`

### Throttling

Uploads are capped at 16 per frame to avoid pipeline stalls.
If a mesh can't be uploaded this frame, it will be retried next frame.

## Static Frame Fast Path

### How It Works

1. During BVH incremental update, check if ALL AABBs are unchanged from the previous frame
2. Compare current view-projection matrix with `last_vp`
3. If both static for 2+ consecutive frames: skip BVH query entirely
4. Reuse cached `static_visible_indices` from the previous static frame

### When It Activates

- Scene objects haven't moved (AABBs unchanged)
- Camera hasn't moved (VP matrix unchanged)
- 2+ consecutive static frames (1-frame warmup to validate)

### Disabling

Automatically disabled when:
- Any AABB changes → `bvh_fully_static = false`, `static_frame_count = 0`
- Camera moves → `static_frame_count = 0`
- Object count changes → BVH rebuilt from scratch

## LightSetTable

### Usage

Automatically populated during scene traversal. Each unique light configuration
(direction, color, type, position, spot params) is stored once and referenced
by index.

```rust
// During traversal (automatic):
let light_key = hash_light_params(&dirs, &colors, &types, &positions, &spots, count);
let light_set_id = self.light_sets.intern(light_key, packed_lights);

// During rendering (lookup):
let lights = ctx.light_sets.get(dc.light_set_id);
// lights.0 = light_dirs, lights.1 = light_colors, etc.
```

### Benefits

- Each DrawCall: 1280B (5 × 16 × 4 × 4) → 4B (u32 index)
- For 1M objects with 5 unique light configs: ~1.2GB CPU memory saved
- Cluster lighting pass deduplicates by light_set_id

## Shared Utils (rc3d-core)

### Graph Algorithms

```rust
use rc3d_core::utils::graph;

// Topological sort into parallel layers
let groups = graph::toposort_layered(&edges, node_count);

// Single linear order (returns Err on cycle)
let order = graph::toposort_linear(&edges, node_count)?;

// BFS traversal with visited set
graph::bfs_visit(start_node, |id| adjacency(id), |id| visit(id));
```

### Math Helpers

```rust
use rc3d_core::utils::math;

let t = math::remap(value, 0.0, 100.0);           // → [0, 1]
let v = math::safe_normalize(near_zero_vec, Vec3::Y);
let tris = math::triangle_count(&indices);
```

### Float Hashing

```rust
use rc3d_core::utils::hash;

let key = hash::f32x3_to_bits(position);
let color_key = hash::f32x4_to_bits(rgba);
```

### Sort Helpers

```rust
use rc3d_core::utils::sort;

// Sort (K, count) pairs descending
sort::sort_by_count_desc(&mut type_counts);

// Sort by derived count
sort::sort_by_key_count_desc(&mut items, |it| it.draw_calls);
```

### Ring Buffer

```rust
use rc3d_core::utils::ring::RingBuffer;

let mut buf = RingBuffer::new(120); // capacity 120
buf.push(frame_time_ms);            // auto-evicts oldest when full
```

## Frame Allocation Reuse Pattern

When adding new per-frame Vecs, use the take → fill → restore pattern:

```rust
// FrameState holds the reusable buffer:
pub struct FrameState {
    pub my_buffer: Vec<MyData>,
}

// At use site:
let mut my_data = std::mem::take(&mut self.frame.my_buffer);
my_data.clear();
my_data.extend(compute_data());
// ... use my_data through rendering pipeline ...
// After use, restore:
self.frame.my_buffer = std::mem::take(&mut my_data);
```

This avoids per-frame allocation when the buffer size is stable.
