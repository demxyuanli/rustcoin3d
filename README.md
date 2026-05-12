# rustcoin3d — Industrial 3D Visualization Engine

Coin3D/HOOPS-aligned 3D visualization engine in Rust + wgpu. Designed for
large-scale industrial visualization: CAD import, real-time rendering,
and interactive scene editing.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Crate dependency graph, core design principles, key data structures, NodeData reference |
| [Rendering Pipeline](docs/rendering-pipeline.md) | Full frame pipeline, culling, lighting, PBR shading, post-processing, draw call batching |
| [Scene Graph](docs/scene-graph.md) | SceneGraph API, all 47 node types, traversal model, dirty flags, animation, serialization |
| [Engine System](docs/engine-system.md) | 12 simulation engines, time management, physics, sensors, field connections |
| [Shaders](docs/shaders.md) | Complete catalog of 45 WGSL shaders with data structures and performance notes |
| [Gap Analysis](docs/industrial-viz-gap-analysis.md) | Coin3D/HOOPS comparison, roadmap, TODO checklist |
| [Optimization Guide](docs/optimization-guide.md) | GPU culling, mesh pool, static frame fast path, LightSetTable, shared utils |

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
├── rc3d-core/       — Math, AABB, BVH, ID types, shared utils (graph, hash, ring, sort)
├── rc3d-fields/     — Field/connection system (Coin3D-style)
├── rc3d-scene/      — Scene graph (SlotMap<NodeId, NodeEntry>), 47 node types, animation
├── rc3d-nodes/      — Re-exports (convenience crate)
├── rc3d-mesh/       — Triangle mesh, meshlet generation, LOD, tessellation
├── rc3d-nurbs/      — NURBS curves and surfaces
├── rc3d-actions/    — Traversal actions (ray pick, bounding box, undo, events, intersection)
├── rc3d-engine/     — Simulation engines, time management, physics, scheduler
├── rc3d-io/         — File import (STL, OBJ, glTF, FBX, Inventor)
├── rc3d-render/     — wgpu renderer (PBR, shadows, culling, post-fx, 45 shaders)
├── rc3d-gizmo/      — 3D manipulator (translate, rotate, scale)
├── rc3d-script/     — Rhai scripting engine
├── rc3d-pointcloud/ — Large-scale point cloud octree (OOC)
├── rc3d-pdf/        — 3D PDF export
├── rc3d-app/        — Application framework, 42 examples, editor UI, camera control
└── rc3d-cli-editor/ — Terminal-based editor
```

## Renderer

wgpu-based cluster-deferred PBR renderer:

| Feature | Description |
|---------|-------------|
| **PBR** | Metallic-roughness (GGX/Smith) with IBL (HDR environment maps, BRDF LUT) |
| **Shadows** | CSM (4 cascades, 8% blend zones), omni-directional point light shadows |
| **Lighting** | Cluster-based forward (16×8×24 grid), LightSetTable dedup (1280B→4B/draw) |
| **Post FX** | TAA (YCoCg), SSR (HIZ-accelerated), SSAO, motion blur, DoF, bloom, color grading, volumetric fog, auto-exposure |
| **GPU Culling** | Dual-path: CPU BVH + GPU compute (frustum + HZB occlusion), meshlet cluster tree |
| **Selection** | Screen-space outline, edge overlay, bounding box, x-ray mode |
| **Display** | Shaded, wireframe, hidden-line, flat, shaded-with-edges |
| **Adaptive** | 5-level quality controller with EMA+hysteresis, interaction-aware reduction |

### Rendering Pipeline (per frame)

```
Frame Start
├── Static frame fast path? [scene+camera unchanged ≥2 frames]
│     Yes → reuse cached visible indices → skip culling
│     No  → continue
├── GPU culling? [objects > threshold]
│     Yes → readback staging → GPU indices replace CPU culling
│     No  → CPU BVH incremental culling
├── Mesh upload [LRU cache, 16 uploads/frame max, 512MB default budget]
├── Sort [by light key → material key → distance]
├── CSM shadow depth [4 cascades]
├── HZB build [depth downsampling]
├── Solid + outline pass [instanced draw batching]
├── Effect passes [decal, volume, point cloud]
├── Post FX [SSAO → SSR → DoF → bloom → TAA → tonemap]
└── HUD overlay [text, grid, viewport dividers]
```

## Scene Graph (minimal example)

```rust
use rc3d_app::{App, CameraController};
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

fn main() {
    let mut g = rc3d_scene::SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(
        PerspectiveCameraNode::look_at(
            Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y,
            std::f32::consts::FRAC_PI_4, 800.0 / 600.0,
        ),
    ));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
        direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
        color: Vec3::ONE, intensity: 1.0, light_group: None,
    }));
    g.add_child(root, NodeData::Material(MaterialNode {
        base_color: Vec3::new(0.8, 0.2, 0.2),
        roughness: 0.4, metallic: 0.0, ..Default::default()
    }));
    g.add_child(root, NodeData::Cube(CubeNode::default()));

    let orbit = CameraController::new(Vec3::ZERO, 10.0);
    winit::event_loop::EventLoop::new().unwrap()
        .run_app(&mut App::new(g).with_camera_controller(orbit))
        .expect("event loop");
}
```

## Examples (42 demos)

| Category | Examples |
|----------|----------|
| Getting Started | `triangle`, `cube`, `rotating_cube` |
| Scene | `scene_graph`, `annotation`, `billboard`, `environment_node`, `exploded_view` |
| Rendering | `pbr_materials`, `pbr_variant_viewer`, `render_features`, `material_variants`, `instancing` |
| Lighting | `area_light`, `light_linking`, `shadow_demo`, `reflection` |
| Camera | `stereo_camera`, `walk_camera` |
| Import | `import_viewer`, `import_viewer_async`, `iv_viewer` |
| Animation | `animation_demo`, `animation_control_panel`, `blend_animation` |
| Editor | `editor`, `selection_set`, `picking`, `markup_dimensions` |
| Engines | `engines_demo`, `scripted_scene` |
| Effects | `post_effects`, `volumetric_demo`, `decal_viewer` |
| Specialized | `point_cloud_viewer`, `nurbs_viewer`, `profile_viewer`, `section_caps` |
| Diagnostics | `adaptive_stress_test`, `large_scene_stress` |

## Node Types (47 variants)

| Category | Variants |
|----------|----------|
| **Grouping** | Separator, Group, Billboard, Transform, Coordinate3, TextureCoordinate2, Normal, ShapeHints, MaterialBinding, ResetTransform, Texture2Transform, File |
| **Shapes** | Triangle, Cube, Sphere, Cone, Cylinder, IndexedFaceSet, IndexedLineSet, SkinnedMesh, MorphTarget |
| **Cameras** | PerspectiveCamera, OrthographicCamera, StereoCamera |
| **Lights** | DirectionalLight, PointLight, SpotLight, AreaLight |
| **Traversal** | Lod, Switch, MultipleCopy, SectionPlane, PickStyle, EventCallback |
| **Annotations** | Text2, Text3, Measurement, Markup, Annotation |
| **Specialized** | ExplodedView, ReflectionPlane, Decal, RayTracing, Volume, PointCloud, Environment, Material |
| **Extensibility** | HandlerNode(Arc\<dyn NodeHandler\>), Custom(u16, Box\<dyn CustomNodeData\>) |

## Performance Characteristics

| Scene | Objects | Draw Calls | Pipeline | Frame Time |
|-------|---------|------------|----------|------------|
| Stress test | 10K | ~10K | Instanced batching | ~5ms CPU |
| Stress test (static) | 10K | ~10K | Static fast path | ~1ms CPU |
| Import viewer | 1-100K | varies | Streaming mesh | ~8-16ms |
| Target (GPU) | 1M+ | indirect | GPU-driven | TBD |

Key optimizations:
- **Light dedup**: 1280B → 4B per draw call (LightSetTable)
- **Static fast path**: Zero culling work when scene is idle
- **Frame allocation reuse**: 8 Vecs reused across frames (~60MB savings @ 1M objects)
- **BVH incremental**: Only dirty AABBs trigger BVH updates
- **Direct cache emit**: FlatDrawCache populated during traversal (no conversion pass)
- **Pool expansion**: phong 64K, flat 32K, mesh cache 4K

## Development

```bash
cargo check --workspace          # fast compile check
cargo test                        # 222 tests
cargo build -p rc3d-app --examples
cargo clippy --workspace          # lint check
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| wgpu 24 | GPU abstraction (Vulkan/Metal/DX12) |
| winit 0.30 | Window creation and event loop |
| glam 0.29 | Linear algebra (Vec3, Mat4, Quat) |
| slotmap | Stable-ID arena storage for scene graph |
| glyphon | GPU text rendering (HUD) |
| rayon | Parallel traversal |
| rhai | Embedded scripting |
| meshopt | Mesh optimization (meshlets, LOD) |
| serde/serde_json | Serialization |
| image | Texture loading |
| tracy-client | GPU/CPU profiling |

## License

BSD-3-Clause
