# rustcoin3d Architecture Reference

## 1. Crate Dependency Graph

```
rc3d-core (foundation: AABB, BVH, NodeId, math, utils)
    ^
rc3d-fields (FieldValue enum, FieldMap with dirty tracking)
    ^
rc3d-scene (SceneGraph, NodeData enum, animation, sensors, traversal)
    ^
    +---- rc3d-nodes (re-export convenience layer)
    +---- rc3d-mesh (tessellation, meshlets, LOD, topology)
    +---- rc3d-nurbs (NURBS curves & surfaces)
    |
rc3d-actions (Action system, ray pick, bounding box, undo, events)
rc3d-engine (Simulation engines, time manager, physics, scheduler)
rc3d-io (File import: glTF, FBX, OBJ, STL, Inventor)
    |
rc3d-render (wgpu renderer: PBR, shadows, culling, post-processing, HUD)
rc3d-gizmo (3D transform manipulators)
rc3d-script (Rhai scripting integration)
rc3d-pointcloud (Large-scale point cloud octree)
rc3d-pdf (3D PDF export)
    |
rc3d-app (Application framework, editor UI, camera control, examples)
rc3d-cli-editor (Terminal-based editor binary)
```

## 2. Core Design Principles

### 2.1 Coin3D/Open Inventor Heritage

The engine follows the Coin3D/Open Inventor scene graph paradigm:

- **Scene Graph**: Directed acyclic graph of typed nodes (not generic graph nodes)
- **Separator/Group**: Separator saves/restores traversal state; Group is pass-through
- **State Stack**: 8 typed element stacks during traversal (model matrix, material, lights, etc.)
- **Action/Visitor**: Extensible traversal actions applied to subtrees
- **Field System**: Named, typed, dynamically connectable fields per node
- **Engine System**: Simulation nodes that evaluate per-frame, connected via fields

### 2.2 SlotMap Storage

All node storage uses `slotmap::SlotMap<NodeId, NodeEntry>` providing:
- O(1) random access by `NodeId`
- Stable IDs across insertions and deletions (no dangling pointers)
- Compact memory layout with generational index validation

### 2.3 Data-Oriented Rendering

The renderer separates hot and cold data:
- **Hot path** (per-frame upload): 64-byte `GpuDrawData` struct
- **Cold path** (dirty-only update): 72-byte `CachedDrawMetadata`
- **Frame reuse**: 8 Vecs pre-allocated and reused each frame

## 3. Key Data Structures

### 3.1 SceneGraph

```rust
pub struct SceneGraph {
    nodes: SlotMap<NodeId, NodeEntry>,   // O(1) access
    roots: Vec<NodeId>,                   // Top-level nodes
    selected: HashSet<NodeId>,           // Current selection
    selection_sets: HashMap<String, HashSet<NodeId>>,  // Named sets
}
```

### 3.2 NodeEntry

```rust
pub struct NodeEntry {
    pub data: NodeData,                   // One of 47+ variants
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub dirty_flags: u8,                  // Bit flags for change tracking
    pub name: Option<String>,
    pub display_mode: Option<DisplayMode>,
    pub fields: FieldMap,                 // Dynamic typed fields
    pub attributes: HashMap<String, String>,
}
```

Dirty flag bits: `TRANSFORM(0)`, `MATERIAL(1)`, `GEOMETRY(2)`, `CHILDREN(3)`, `REMOVED(4)`, `FROZEN(7)`

### 3.3 State Elements (Traversal Stack)

| Element ID | Type | Description |
|-----------|------|-------------|
| 0 | ModelMatrixElement | Current world-space transform |
| 1 | ViewMatrixElement | View matrix |
| 2 | ProjectionMatrixElement | Projection matrix |
| 3 | CoordinateElement | Per-vertex positions |
| 4 | NormalElement | Per-vertex normals |
| 5 | MaterialElement | PBR material parameters |
| 6 | LightElement | Accumulated light list |
| 7 | TextureCoordinate2Element | UV coordinates |

## 4. NodeData Categories (47 variants)

| Category | Variants | Purpose |
|----------|----------|---------|
| **Grouping** | Separator, Group, Billboard, Transform, Coordinate3, TextureCoordinate2, Normal, ShapeHints, MaterialBinding, ResetTransform, Texture2Transform, File | Scene structure, state management, attribute binding |
| **Shapes** | Triangle, Cube, Sphere, Cone, Cylinder, IndexedFaceSet, IndexedLineSet, SkinnedMesh, MorphTarget | Geometric primitives and meshes |
| **Cameras** | PerspectiveCamera, OrthographicCamera, StereoCamera | Viewpoint definition |
| **Lights** | DirectionalLight, PointLight, SpotLight, AreaLight | Illumination sources |
| **Materials** | Material | PBR shading parameters |
| **Traversal** | Lod, Switch, MultipleCopy, SectionPlane, ResetTransform, PickStyle, EventCallback | Flow control and rendering modifiers |
| **Annotations** | Text2, Text3, Measurement, Markup, Annotation | On-screen text, dimensions, markup |
| **Advanced** | ExplodedView, ReflectionPlane, Decal, RayTracing, Volume, PointCloud, Environment | Specialized rendering effects |
| **Extensibility** | HandlerNode(Arc\<dyn NodeHandler\>), Custom(u16, Box\<dyn CustomNodeData\>) | User-defined node behavior |

## 5. Action System

### 5.1 Core Trait

```rust
pub trait Action: Send {
    fn kind(&self) -> ActionKind;
    fn apply(&mut self, graph: &SceneGraph, root: NodeId);
}
```

### 5.2 Action Implementations

| Action | File | Function |
|--------|------|----------|
| `RenderAction` | `rc3d-render::render_action.rs` | Scene traversal → DrawCall collection (79KB) |
| `GetBoundingBoxAction` | `rc3d-actions::get_bounding_box.rs` | World-space AABB computation |
| `RayPickAction` | `rc3d-actions::ray_pick.rs` | Ray-triangle/sphere intersection |
| `HandleEventAction` | `rc3d-actions::handle_event.rs` | Mouse/keyboard/touch event routing |
| `SectionPlaneAction` | `rc3d-actions::section_plane.rs` | Clipping plane collection |
| `IntersectionDetectionAction` | `rc3d-actions::intersection_detection.rs` | Pairwise geometry intersection |
| `SearchAction` | `rc3d-actions::scene_path.rs` | Node lookup by name/type |

### 5.3 Parallel Traversal

Root-level parallelism via rayon:
```rust
par_apply_to_all_roots(action, graph)  // Each root subtree runs on a rayon thread
```

## 6. File Format Support

| Format | Module | Read | Write | Notes |
|--------|--------|------|-------|-------|
| glTF | `gltf.rs` (16KB) | Yes | - | Binary (.glb) and text (.gltf) |
| FBX | `fbx/` directory | Yes | - | Binary FBX parser |
| OBJ | `obj.rs` (8KB) | Yes | - | Wavefront OBJ with MTL |
| STL | `stl.rs` (10KB) | Yes | - | Binary and ASCII |
| Inventor | `iv.rs` (23KB) | Yes | Yes | Coin3D native format |

## 7. Memory Efficiency

| Technique | Saving | Mechanism |
|-----------|--------|-----------|
| LightSetTable | 1280B → 4B/draw | Dedup light parameters by key |
| FlatDrawCache | ~200B/draw | Separate hot (64B) from cold (72B) |
| Frame allocation reuse | ~60MB @ 1M objects | 8 Vecs reused each frame |
| Static frame fast path | ~5ms CPU skipped | Cache visible set when scene+camera static |
| BVH incremental update | Avoid full rebuild | Only re-insert dirty AABBs |
| GPU indirect draw | CPU avoids per-draw dispatch | GPU compute writes draw args buffer |
| LOD cluster tree | Sub-linear scaling | Hierarchical culling for large meshes |
