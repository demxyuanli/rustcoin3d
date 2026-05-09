# rustcoin3d — Industrial 3D Engine

Coin3D/HOOPS-aligned 3D visualization engine in Rust + wgpu. Designed for
large-scale industrial visualization: CAD import, real-time rendering,
and interactive scene editing.

## Quick Start

```bash
# Build all examples
cargo build -p rc3d-app --examples

# Import and view a 3D file
cargo run -p rc3d-app --example import_viewer -- model.stl

# Editor with scene graph inspector
cargo run -p rc3d-app --example editor

# Stress test with adaptive quality
cargo run -p rc3d-app --example adaptive_stress_test

# CLI editor (terminal)
cargo run -p rc3d-cli-editor
```

## Architecture

```
crates/
├── rc3d-core/      — Math, AABB, BVH, ID types, shared utils
├── rc3d-fields/     — Field/connection system (Coin3D-style)
├── rc3d-scene/      — Scene graph (SlotMap<NodeId, NodeEntry>), 47 node types
├── rc3d-actions/    — Traversal actions (ray pick, bounding box, undo, measurement)
├── rc3d-engine/     — Simulation engines, time management
├── rc3d-io/         — File import (STL, OBJ, glTF, FBX, Inventor)
├── rc3d-mesh/       — Triangle mesh, meshlet generation, LOD, tessellation
├── rc3d-render/     — wgpu renderer (see below)
├── rc3d-gizmo/      — 3D manipulator (translate, rotate, scale)
├── rc3d-app/        — Application framework, examples
├── rc3d-cli-editor/ — Terminal-based editor
├── rc3d-nurbs/      — NURBS curves and surfaces
├── rc3d-pointcloud/ — Point cloud octree
├── rc3d-pdf/        — 3D PDF export
└── rc3d-script/     — Scripting engine
```

## Renderer

wgpu-based cluster-deferred PBR renderer:

| Feature | Description |
|---------|-------------|
| **PBR** | Metallic-roughness with IBL (HDR environment maps) |
| **Shadows** | CSM (4 cascades, blend zones), omni-directional |
| **Lighting** | Cluster-based forward lighting, up to 256 point/spot lights |
| **Post FX** | TAA, SSR, SSAO, motion blur, DoF, bloom, color grading, volumetric fog |
| **Meshlet** | GPU-driven cluster culling with HZB occlusion |
| **Selection** | Screen-space outline, edge overlay, x-ray |
| **Display** | Shaded, wireframe, hidden-line, flat, shaded-with-edges |
| **Adaptive** | 5-level quality controller with EMA+hysteresis |

### Rendering Pipeline (per frame)

```
Frame Start
├── Static frame fast path? [enabled, scene+camera unchanged 2+ frames]
│     Yes → reuse cached visible indices → skip culling
│     No  → continue
├── GPU culling? [enabled, objects > threshold]
│     Yes → readback staging → GPU indices replace CPU culling
│     No  → CPU BVH incremental culling (reused allocations)
├── Mesh upload [LRU cache, active GPU cleanup on eviction]
├── Sort [by light key → material key → distance]
├── Shadow depth pass [CSM 4 cascades]
├── HZB build [depth downsampling]
├── Solid + outline pass [instanced draw batching]
├── Effect passes [decal, volume, point cloud]
├── Post FX [SSAO → SSR → DoF → bloom → TAA → tonemap]
└── HUD overlay
```

## New Modules (2026-05)

### GPU-Driven Rendering

| Module | File | Purpose |
|--------|------|---------|
| GPU Culling | `gpu_culling.rs` | Compute shader frustum cull → staging readback |
| Object Cull Shader | `shaders/object_cull.wgsl` | Per-object AABB vs 6-plane frustum |
| Mesh Pool | `mesh_pool.rs` | Streaming GPU mesh pool, LRU + budget eviction |
| Cluster Tree | `cluster_tree.rs` | Hierarchical LOD cluster culling pipeline |
| Tree Cull Shader | `shaders/cluster_tree_cull.wgsl` | Multi-level cluster frustum + HZB cull |
| Light Set | `light_set.rs` | Deduplicated light parameters (1280B → 4B/draw) |

### Shared Utils (`rc3d-core/src/utils/`)

| Module | Functions |
|--------|-----------|
| `graph.rs` | `toposort_layered`, `toposort_linear`, `bfs_visit` |
| `math.rs` | `remap`, `lerp`, `safe_normalize`, `triangle_count` |
| `hash.rs` | `f32x3_to_bits`, `f32x4_to_bits`, `f32_total_key` |
| `sort.rs` | `sort_by_count_desc`, `sort_by_key_count_desc` |
| `ring.rs` | `RingBuffer<T>` |

### Configuration

```rust
use rc3d_render::settings::PerformanceSettings;

let settings = RenderSettings {
    performance: PerformanceSettings {
        gpu_culling: true,
        gpu_culling_threshold: 4096,
        parallel_traversal: false,
        mesh_pool_capacity: 4096,
        mesh_pool_max_mb: 512,
    },
    ..Default::default()
};
renderer.apply_settings(settings);
renderer.enable_gpu_culling(65536); // allocate GPU buffers
```

## Performance Characteristics

| Scene | Objects | Draw Calls | Pipeline | Frame Time |
|-------|---------|------------|----------|------------|
| Stress test | 10K | ~10K | Instanced batching | ~5ms CPU |
| Stress test (static) | 10K | ~10K | Static fast path | ~1ms CPU |
| Import viewer | 1-100K | varies | Streaming mesh | ~8-16ms |
| Target (GPU) | 1M+ | indirect | GPU-driven | TBD |

Key optimizations applied:
- **Light dedup**: 1280B → 4B per draw call (LightSetTable)
- **Static fast path**: Zero culling work when scene is idle
- **Frame allocation reuse**: 8 Vecs reused across frames (~60MB savings @ 1M objects)
- **BVH incremental**: Only dirty AABBs trigger BVH updates
- **Direct cache emit**: FlatDrawCache populated during traversal (no conversion pass)
- **Pool expansion**: phong 64K, flat 32K, mesh cache 4K → no silent draw drops

## Development

```bash
cargo check --workspace          # fast compile check (222 tests)
cargo test                        # run all tests
cargo build -p rc3d-app --examples
cargo clippy --workspace          # lint check
```

## License

Proprietary — internal use. Contact the repository owner for details.
