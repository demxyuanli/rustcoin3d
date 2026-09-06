# rustcoin3d Architecture Reference

**Generated:** 2026-06-22 | **Crates:** 21 | **Lines:** ~120K Rust | **Tests:** ~500

Industrial 3D visualization engine in Rust + wgpu, aligned with Coin3D/HOOPS paradigms.

---

## 1. Crate Architecture & Dependency Graph

### 1.1 Topological Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 3 — Entry Points                                           │
│  rc3d-studio (binary)  rc3d-cli-editor  rc3d-examples            │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 2 — Application & Integration                              │
│  rc3d-editor ─── rc3d-engine-api ─── rc3d-scene-api              │
│  rc3d-gizmo     rc3d-effects                                     │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 1 — Domain Engines                                         │
│  rc3d-render   rc3d-io     rc3d-shape   rc3d-engine   rc3d-actions│
│  rc3d-nurbs    rc3d-script  rc3d-pointcloud  rc3d-pdf  rc3d-nodes│
├─────────────────────────────────────────────────────────────────┤
│ LAYER 0 — Foundation                                             │
│  rc3d-core ────── rc3d-fields ────── rc3d-scene ───── rc3d-mesh │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Full Dependency Matrix

| Crate | Depends On (internal) | Role |
|-------|----------------------|------|
| `rc3d-core` | — | `NodeId`, math (`glam`), AABB, BVH, errors, utils |
| `rc3d-mesh` | core | Mesh data, meshlets, LOD, tessellation |
| `rc3d-fields` | core | Dynamic field/value map (`FieldMap`) |
| `rc3d-scene` | core, fields | Scene graph (SlotMap DAG), `NodeData` (62 variants) |
| `rc3d-nurbs` | core, mesh | NURBS curves/surfaces for GPU tessellation |
| `rc3d-engine` | core, fields, scene | Simulation engines (28 types), `EngineRegistry` + field graph |
| `rc3d-shape` | core, mesh, nurbs | B-Rep kernel: topology, geometry, bool, heal, mesh |
| `rc3d-actions` | core, fields, scene, mesh | Visitor/action pattern (render, pick, LOD, measurement) |
| `rc3d-nodes` | core, fields, scene | Re-export shim for node types |
| `rc3d-pointcloud` | core, scene | Point cloud with out-of-core tile caching |
| `rc3d-pdf` | core, scene | PDF/U3D document generation |
| `rc3d-script` | core, scene | Rhai scripting bindings |
| `rc3d-io` | core, scene, shape | STEP/STL/OBJ/glTF/FBX/VRML/IV/BREP import/export |
| `rc3d-render` | core, fields, scene, mesh, actions | wgpu GPU renderer (~60 modules) |
| `rc3d-gizmo` | core, scene, actions, render | 3D transform manipulator |
| `rc3d-effects` | core, render | Post-processing effect graph, render config |
| `rc3d-scene-api` | core, scene, mesh, engine | High-level declarative scene DSL |
| `rc3d-engine-api` | core, scene, render, engine, io, actions, scene-api, effects, nurbs | Primary integration facade (`Engine`, `World`) |
| `rc3d-editor` | core, scene, actions, render, engine-api, gizmo | egui editor (panels, nav cube, compositor graph via `egui-snarl` on the bottom dock; GPU exec stays in `rc3d-render`) |
| `rc3d-studio` | engine-api, editor | Desktop editor host (winit + Fluent chrome) |
| `rc3d-cli-editor` | core, scene, actions, render | CLI+TUI binary (ratatui + egui) |
| `rc3d-examples` | engine-api, scene-api, editor, ... | Demo applications |

### 1.3 Public-Facing vs Internal Crates

| Visibility | Crates |
|-----------|--------|
| **Public API** | `rc3d-engine-api`, `rc3d-scene-api`, `rc3d-editor` |
| **Internal** | All others (libraries + `rc3d-studio` / `rc3d-cli-editor` / `rc3d-examples` entry points) |

---

## 2. Core Layer — Foundation Types

### 2.1 `rc3d-core` — The Sole Root

Every crate transitively depends on `rc3d-core`. It has zero internal dependencies.

| Module | Key Types | Purpose |
|--------|-----------|---------|
| `id` | `NodeId`, `FieldId`, `GeometryId` | SlotMap newtypes for O(1) lookup |
| `math` | `Real` (=f64), `PVec3` (=DVec3), `Vec3`, `Mat4` | Dual-precision math via `glam` |
| `aabb` | `Aabb` | Axis-aligned bounding box with union/intersect/ray |
| `bvh` | `Bvh<T>` | Bounding volume hierarchy for spatial queries |
| `color` | `Color4f` | Linear/sRGB color |
| `display` | `DisplayMode`, `FillStyle`, `EdgeStyle`, `Appearance`, `VisualStyle` / `VisualStyleLibrary`, `ClipPlane`, `ClipCapsStyle` | Presets, orthogonal fill x edges, named style catalog |
| `projection` | perspective/orthographic builders | Depth-reversed Z projection matrices |
| `error` | `EngineError`, `EngineResult<T>` | Unified error type (Io, Parse, Scene, Render, etc.) |
| `utils` | graph (toposort, BFS), hash, math, ring, sort | Shared algorithmic utilities |

### 2.2 `rc3d-fields` — Dynamic Field System

```
FieldMap {
    entries: SlotMap<FieldId, FieldEntry>
}

FieldEntry {
    value: FieldValue,         // Bool|Int32|Float|Float64|Vec2f|Vec3f|Vec4f|Mat4f|
                               // FloatArray|Vec3fArray|Int32Array|String|Binary
    dirty: bool,
    owner: NodeId,
    field_index: u16,
    connections: Vec<FieldId>,  // field→field edges for engine propagation
}
```

Key operations: `insert`, `get`, `set` (propagates dirty via BFS along connections), `connect(from, to)`.
Cross-node typed fields use `SceneGraph::connect_fields` / `propagate_fields` (`FieldRef` + reverse lookup).

### 2.3 `rc3d-mesh` — Mesh Data Structures

| Type | Purpose |
|------|---------|
| `TriangleMesh` | Indexed triangle list with normals |
| `MeshletData` | Meshlet clusters for GPU culling |
| `EdgeKey` | Packed u64 edge identifier |
| `meshopt` integration | Vertex cache optimization, overdraw reduction |

---

## 3. Scene Graph Layer

### 3.1 `SceneGraph` — The Central Data Model

```rust
pub struct SceneGraph {
    nodes: SlotMap<NodeId, NodeEntry>,   // O(1) access, stable keys
    roots: Vec<NodeId>,                  // Top-level nodes (no parent)
    selected: HashSet<NodeId>,           // Current selection
    selection_sets: HashMap<String, HashSet<NodeId>>,
    face_tints / edge_tints,             // Sub-entity colors (runtime)
}

pub struct NodeEntry {
    pub data: NodeData,                  // Typed payload (enum)
    pub parent: Option<NodeId>,          // Parent link
    pub children: Vec<NodeId>,           // Child links
    pub dirty_flags: u8,                 // TRANSFORM|MATERIAL|GEOMETRY|CHILDREN|REMOVED
    pub name: Option<String>,
    pub display_mode: Option<DisplayMode>,
    pub fill_style: Option<FillStyle>,
    pub edge_style: Option<EdgeStyle>,
    pub fields: FieldMap,                // Dynamic fields
    pub attributes: HashMap<String, String>,
}
```

### 3.2 `NodeData` Enum — 56 Variants

| Category | Variants |
|----------|----------|
| **Grouping** | `Separator`, `Group`, `Billboard`, `Sprite`, `Annotation`, `Switch`, `Lod`, `MultipleCopy` |
| **Transform** | `Transform`, `ResetTransform`, `TransformManip`, `Dragger` |
| **Geometry** | `Coordinate3`, `Normal`, `TextureCoordinate2`, `IndexedFaceSet`, `IndexedLineSet`, `InstancedMesh`, `BatchedMesh` |
| **Primitives** | `Triangle`, `Cube`, `Sphere`, `Cone`, `Cylinder`, `Torus` |
| **Material** | `Material`, `MaterialBinding`, `Texture2Transform`, `ShapeHints` |
| **Camera** | `PerspectiveCamera`, `OrthographicCamera`, `StereoCamera` (dual-eye SBS/TB/anaglyph), `CubeCamera` |
| **Light** | `DirectionalLight`, `PointLight`, `SpotLight`, `AreaLight`, `HemisphereLight`, `LightProbe` |
| **Environment** | `Environment` (ambient + fog) |
| **Effects** | `Decal`, `ExplodedView`, `ReflectionPlane`, `RayTracing`, `Volume`, `PointCloud` |
| **Annotation** | `Measurement`, `Markup`, `AnnotationSet` (`elements` visual + `pmi` semantic records) |
| **Event/Pick** | `EventCallback`, `PickStyle` |
| **Animation** | `SkinnedMesh`, `MorphTarget` |
| **Extension** | `HandlerNode(Arc<dyn NodeHandler>)`, `Custom(u16, Box<dyn CustomNodeData>)` |
| **File** | `File` (external reference, inlined during traversal) |

### 3.3 Traversal

```rust
pub struct DfsPreOrder<'a> {
    graph: &'a SceneGraph,
    stack: Vec<NodeId>,
}
// Usage: graph.traverse_dfs(root) or graph.traverse_all()
// Parallel: graph.traverse_parallel_all(action_factory) using rayon
```

### 3.4 `NodeHandler` — Extension Without Modifying Core Enum

```rust
pub trait NodeHandler: Debug + Send + Sync {
    fn handler_name(&self) -> &'static str;
    fn traverse(&self, graph: &SceneGraph, node: NodeId,
                children: &[NodeId], recur: &mut dyn FnMut(NodeId));
}
```

`NodeData::HandlerNode(Arc<dyn NodeHandler>)` allows custom traversal logic without growing the core enum.

---

## 4. Engine Layer — Simulation & Time

### 4.1 `Engine` Trait

```rust
pub trait Engine: Any + Debug {
    fn evaluate(&mut self, graph: &mut SceneGraph, time: f64);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
```

Engines receive `&mut SceneGraph` — they locate nodes by `NodeId` and mutate them directly. They must set `dirty_flags` to trigger render cache invalidation.

### 4.2 Engine Implementations (28 types)

| Engine | Coin3D Analog | Behavior |
|--------|---------------|----------|
| `ElapsedTimeEngine` | `SoElapsedTime` | Continuous rotation around axis at `speed` rad/s |
| `SineOscillatorEngine` | — | Sine wave on transform translation/scale |
| `CalculatorEngine` | `SoCalculator` | Expression evaluation with `sin()`, `cos()`, `iA` ports |
| `ComposeMatrixEngine` | `SoComposeMatrix` | TRS components → Mat4 |
| `DecomposeMatrixEngine` | `SoDecomposeMatrix` | Mat4 → translation / rotation / scale |
| `OneShotEngine` | `SoOneShot` | Trigger once for specified duration |
| `CounterEngine` | `SoCounter` | Integer counter cycling in [min, max] |
| `TimeCounterEngine` | `SoTimeCounter` | Frequency-driven integer cycle |
| `InterpolateVec3Engine` | `SoInterpolateVec3f` | Linear translation interpolation |
| `InterpolateFloatEngine` | `SoInterpolateFloat` | Scalar field interpolation |
| `InterpolateRotationEngine` | `SoInterpolateRotation` | Quaternion SLERP |
| `ComposeVec2fEngine` | `SoComposeVec2f` | Two floats → Vec2 |
| `ComposeVec3fEngine` | `SoComposeVec3f` | Three floats → Vec3 |
| `ComposeVec4fEngine` | `SoComposeVec4f` | Four floats → Vec4 |
| `DecomposeVec2fEngine` | `SoDecomposeVec2f` | Vec2 → x/y |
| `DecomposeVec3fEngine` | `SoDecomposeVec3f` | Vec3 → x/y/z |
| `DecomposeVec4fEngine` | `SoDecomposeVec4f` | Vec4 → x/y/z/w |
| `ComposeRotationEngine` | `SoComposeRotation` | Axis-angle → Mat4 |
| `DecomposeRotationEngine` | `SoDecomposeRotation` | Mat4 → axis-angle |
| `GateEngine` | `SoGate` | Enable/trigger pass-through |
| `ConcatenateEngine` | `SoConcatenate` | Up to 8 inputs → array |
| `SelectOneEngine` | `SoSelectOne` | Index into array |
| `BoolOperationEngine` | `SoBoolOperation` | AND/OR/XOR/NOT/NAND/NOR |
| `TransformVec3fEngine` | `SoTransformVec3f` | Matrix × vector |
| `OnOffEngine` | `SoOnOff` | Boolean toggle |
| `TriggerAnyEngine` | `SoTriggerAny` | Edge-trigger on any input change |
| `AnimationMixerEngine` | — | Object/joint clip mixer |
| `ParticleEngine` | — | PointCloud emitter tick |

### 4.3 `EngineRegistry`

```rust
pub struct EngineRegistry {
    pub engines: Vec<Box<dyn Engine>>,
    pub connections: Vec<EngineConnection>,
}
// evaluate_all(graph, time)
//   empty connections → insertion order
//   else → toposort engine→engine edges, set_input / evaluate / write node fields
// SceneGraph.field_graph: cross-node FieldRef edges, propagate_fields after engines
```

### 4.4 Supporting Types

| Type | File | Purpose |
|------|------|---------|
| `SimulationScheduler` | `scheduler.rs` | `before_engines` / `after_engines` callback hooks |
| `TimeManager` | `time_manager.rs` | Monotonic clock via `Instant::now()`, `time_scale` (0=pause) |

---

## 5. Action/Visitor Layer

### 5.1 `Action` Trait

```rust
pub trait Action: Send {
    fn kind(&self) -> ActionKind;
    fn apply(&mut self, graph: &SceneGraph, root: NodeId);
}
```

### 5.2 Typed Element Stack (State Machine)

During traversal, `State` maintains 8 typed element stacks. `Separator` nodes push/pop all 8 for state isolation:

| Index | Type | Content |
|-------|------|---------|
| 0 | `ModelMatrixElement` | Accumulated world transform |
| 1 | `ViewMatrixElement` | Camera view matrix |
| 2 | `ProjectionMatrixElement` | Camera projection matrix |
| 3 | `CoordinateElement` | Current vertex positions |
| 4 | `NormalElement` | Current normals |
| 5 | `MaterialElement` | Current PBR material |
| 6 | `LightElement` | Accumulated scene lights |
| 7 | `TextureCoordinate2Element` | Current UV coordinates |

### 5.3 Key Actions

| Action | Purpose |
|--------|---------|
| `RenderCollector` | Traverses scene graph → produces `Vec<DrawCall>` |
| `GetBoundingBoxAction` | Computes world-space AABB for subtree |
| `RayPickAction` | Ray-scene intersection test |
| `SearchAction` | Find nodes by name |
| `GetMatrixAction` | Sample transform matrix at node |
| `SectionPlaneAction` | Collect active clip planes + cap/hatch style |
| `HandleEventAction` | Route events to `EventCallback` nodes; `Engine::handle_window_event` is the shared winit entry |
| `update_all_lod_nodes` | LOD selection by camera distance |
| `MeasurementAction` | Compute distance/angle/radius measurements |
| `MarkupAction` / `AnnotationTool` | 2D/3D annotation processing |

---

## 6. B-Rep Geometry Kernel (`rc3d-shape`)

### 6.1 Topology Storage — `BRepStore`

```
BRepStore {
    vertices: SlotMap<VertexKey, BRepVertex>,     // 3D points
    edges:    SlotMap<EdgeKey, BRepEdge>,         // bounded curves
    wires:    SlotMap<WireKey, BRepWire>,         // ordered edge loops
    faces:    SlotMap<FaceKey, BRepFace>,         // bounded surfaces
    shells:   SlotMap<ShellKey, BRepShell>,       // face collections
    solids:   SlotMap<SolidKey, BRepSolid>,       // shell + void shells
    compounds: SlotMap<CompoundKey, BRepCompound>, // solid collections

    // Indices
    vertex_spatial_index: SpatialIndexF64<VertexKey>,     // spatial dedup
    edge_hash_index: HashMap<(VK,VK), Vec<EdgeKey>>,      // endpoint→edge lookup
    edge_to_faces: HashMap<EdgeKey, Vec<FaceKey>>,         // edge→face reverse index
    vertex_to_edges: HashMap<VertexKey, Vec<EdgeKey>>,     // vertex→edge reverse index
    trim_ranges: HashMap<FaceKey, (u_min,u_max,v_min,v_max)>,
    tolerance: ToleranceContext,
}
```

### 6.2 Topology Types

| Type | Fields |
|------|--------|
| `BRepVertex` | `position: PVec3`, `tolerance: Real` |
| `BRepEdge` | `curve: CurveGeom`, `tolerance`, `v_low`, `v_high`, `t_min`, `t_max`, `pcurves: HashMap<FK, Curve2d>`, `cached_deflection` |
| `BRepWire` | `edges: Vec<(EdgeKey, Orientation)>` |
| `BRepFace` | `surface: SurfaceGeom`, `outer_wire`, `inner_wires`, `same_sense`, `tolerance`, `seam_edges`, `degenerated_edges`, `color` |
| `BRepShell` | `faces: Vec<(FaceKey, Orientation)>`, `closed: bool`, `step_id` |
| `BRepSolid` | `outer_shell: ShellKey`, `void_shells: Vec<ShellKey>` |
| `BRepCompound` | `solids: Vec<SolidKey>` |

### 6.3 Geometry Types

**11 Curve Types** (`CurveGeom`):

| Variant | Parameters | Evaluation |
|---------|-----------|------------|
| `Line` | origin + direction | Linear |
| `Circle` | center, axis, radius, ortho axes | Trigonometric |
| `Ellipse` | center, axis, semi_major/minor, ortho axes | Trigonometric |
| `Hyperbola` | center, axis, semi_major/minor, ortho axes | Hyperbolic |
| `Parabola` | center, axis, focal_dist, ortho axes | Quadratic |
| `BSpline` | degree, control points, knots, weights | Cox-de Boor |
| `BezierCurve` | degree, control points, weights | De Casteljau (Richardson for rational) |
| `Trimmed` | basis + t_min/t_max | Delegates to basis |
| `Composite` | segments + cached lengths | Segment lookup |
| `Polyline` | points | Linear interpolation |
| `Offset` | basis + offset_dir + distance | Delegates + offset |

**9 Surface Types** (`SurfaceGeom`):

| Variant | Parameters |
|---------|-----------|
| `Plane` | origin, normal, u_dir |
| `Cylinder` | origin, axis, radius, ortho axes |
| `Cone` | apex, axis, semi_angle, radius_at_apex, ortho axes |
| `Sphere` | center, radius |
| `Torus` | center, axis, major_r, minor_r, ortho axes |
| `BSpline` | `NurbsSurface` (degree_u/v, control point grid, knots, weights) |
| `Extrusion` | generatrix curve + direction vector |
| `Revolution` | generatrix curve + axis origin/direction |
| `Offset` | basis surface + signed distance |

**7 Curve2d Types** (UV-space pcurves): Line, Circle, Ellipse, BSpline, Trimmed, Polyline, Composite

### 6.4 NURBS Surface (`NurbsSurface`)

```rust
pub struct NurbsSurface {
    pub degree_u: usize,
    pub degree_v: usize,
    pub control_points: Vec<Vec<PVec3>>,   // [u_count][v_count]
    pub weights: Vec<Vec<Real>>,
    pub knots_u: Vec<Real>,
    pub knots_v: Vec<Real>,
}
```

Key methods: `evaluate(u,v)`, `derivative(u,v)`, `normal(u,v)`, `evaluate_with_hessian(u,v)`, knot insertion/removal (Boehm), degree elevation/reduction, `from_points_grid()`, static constructors for plane/cylinder/cone/sphere/torus as NURBS.

### 6.5 Boolean Operations

**Entry point:** `boolean_brep(shells_a, shells_b, reg, op, options) -> BRepBoolResult`

**Operations:** `Union`, `Intersection`, `Difference`

**5-Stage Pipeline:**

```
Stage 1: PaveFiller ── face-face intersection (marching + Newton, analytic fallback)
    │                   VV→VE→EE→VF→EF→FF interference phases
    v
Stage 2: Split ──────── split faces along intersection curves (BOPDS PaveBlock or UV marching)
    │
    v
Stage 3: Classify ───── ray-casting point-in-solid (4 off-axis rays when cardinal hits boundary)
    │
    v
Stage 4: Select ─────── select faces by BoolOp (Union/Intersection/Difference)
    │
    v
Stage 5: Stitch ─────── sew selected faces into result shell, build solids
```

**Special cases:**
- Coplanar faces: 2D polygon clipping in UV space (Greiner-Hormann)
- No intersection: containment-based trivial resolution

### 6.6 Healing Pipeline

**Entry points:** `auto_heal_shell()`, `run_heal_pipeline()`

**Heal Levels:** `Basic`, `Standard`, `Advanced` (with `HealPolicy` for pass selection)

**Passes (18 total):**

| Pass | Category | What it fixes |
|------|----------|---------------|
| `unify_same_domain` | Face merge | Merge coplanar/coaxial adjacent faces |
| `fix_small_faces` | Geometry | Remove faces below area threshold |
| `fix_edge_tolerances` | Tolerance | Adjust edge tolerances to span 3D-2D gap |
| `fix_wire_order` | Topology | Reorder edges into continuous loops |
| `same_param_reparam` | PCurve | Adjust pcurves so 3D edge matches face UV |
| `pcurve_fix` | PCurve | Repair broken pcurve mappings |
| `seam` | Topology | Handle periodic surface seam edges |
| `degenerated` | Topology | Handle sphere/cone pole degeneracies |
| `self_intersect` | UV | Detect and fix UV wire self-intersections |
| `face_self_intersect` | Surface | Detect normal-flip regions on BSpline surfaces |
| `intersecting_wires` | Topology | Merge overlapping inner wires |
| `free_bounds` | Topology | Close open wire boundaries |
| `compose_shell` | Topology | Rebuild shell from connected faces |
| `edge_connect` | Topology | Merge coincident vertices, fix edge refs |
| `face_fold` | Surface | Detect folded surface regions |
| `continuity` | Geometry | Check C0/C1/G1 continuity at edges |
| `canonical` | Geometry | Fit canonical curves to near-circular edges |
| `solid_fix` | Topology | Fix solid-level issues (empty shells) |

### 6.7 Meshing Pipeline

**Entry point:** `mesh_brep_shell()` / `mesh_brep_shell_with_report()`

```
BRepShell
    │
    v
Edge Discretization ── incremental sampling with cached deflection
    │
    v
Face Meshing ── per-face algorithm dispatch:
    ├── Plane: center-fan triangulation
    ├── Revolution: uniform U sweep + seam handling
    ├── Extrusion/BSpline: UV Delaunay (CDT with Steiner points)
    └── Fallback: parametric grid → 3D fill
    │
    v
Shell Assembly ── merge face meshes at shared edges
    │
    v
Post-Process ── gap welding, T-junction repair, free-edge closing
    │
    v
MeshResult { vertices: Vec<[f64;3]>, normals: Vec<[f64;3]>, indices: Vec<u32> }
```

**Config:** `BRepMeshConfig { edge: EdgeDiscConfig, face: FaceFillConfig, refine, optimize, relative_deflection, weld_tolerance, ... }`

### 6.8 BREP Export

**Entry point:** `write_brep(store: &BRepStore, output: &mut impl Write) -> io::Result<()>`

Outputs OCC-compatible `DBRep_DrawableShape` ASCII format:
- **Sections:** DBRep_DrawableShape → CASCADE Topology V1 → Locations → Curve2ds → Curves → Polygon3D → Surfaces → Triangulations → TShapes
- **Curve coverage:** Types 1-7 (Line, Circle, Ellipse, Hyperbola, Parabola, Bezier, BSpline)
- **Surface coverage:** Types 1-8 (Plane through BSpline, including rational NURBS)
- **TShapes:** Full topology with orientations, tolerances, flags in dependency order (Ve→Ed→Wi→Fa→Sh→So→Co)

---

## 7. I/O Layer (`rc3d-io`)

### 7.1 Format Support Matrix

| Format | Direction | Extensions | Internal Target |
|--------|-----------|------------|-----------------|
| **STEP** | Import + Export | `.step`, `.stp` | `ShapeDocument` → `SceneGraph` |
| **STL** | Import + Export | `.stl` | `SceneGraph` (direct) |
| **OBJ** | Import | `.obj` | `SceneGraph` (direct) |
| **glTF** | Import | `.gltf`, `.glb` | `SceneGraph` (v2.0, PBR materials) |
| **FBX** | Import | `.fbx` | `SceneGraph` (v7400 binary, skeletons+animation) |
| **VRML** | Import | `.wrl` | `SceneGraph` (VRML97) |
| **IV** | Import + Export | `.iv` | `SceneGraph` (OpenInventor 1.0) |
| **IGES** | Import + Export | `.igs`, `.iges` | `BRepStore` → `SceneGraph` |
| **BREP ASCII** | Export | `.brep` | From `BRepStore` (OCC format) |
| **BREP Binary** | Import + Export | `.brepbin` | From/To `BRepStore` (custom binary) |

### 7.2 Unified Entry Point

```rust
pub fn import_file(path: &Path) -> Result<SceneGraph, ImportError>
```

Dispatches by file extension. IGES goes through `BRepStore → ShapeDocument → SceneEmitPlan → SceneGraph` bridge.

### 7.3 STEP Import Pipeline (Deepest Path)

```
Raw STEP bytes
    │
    ▼
[1] decode_step_bytes()           UTF-8 or Latin-1 decode
    │
    ▼
[2] Part21 Parser                 Lexer → Token → Entity → model::Exchange
    │  read_exchange_with_recovery() — streaming support for large files
    ▼
[3] Adapter                       model::Exchange → EntityIndex (HashMap<u64, EntityRecord>)
    │  exchange_from_model() — CompatMerge or StrictFidelity mode
    ▼
[4] StepCafTransfer::transfer()   EntityIndex → BRepStore + ShapeDocument
    │  ├── 4-pass B-Rep build (surfaces → edges+pcurves → wires+faces+shells → roots)
    │  ├── DedupIndex (CDSR/MAPPED_ITEM instancing)
    │  ├── AssemblyContext → XDE label hierarchy
    │  └── PMI extraction + provenance binding
    ▼
[5] Post-processing               same_parameter, heal pipeline, continuity checks
    │
    ▼
[6] SceneEmitPlan → SceneGraph    Per-instance: Separator → Transform → Material → IndexedFaceSet
    │                              f64→f32 conversion at render boundary
    ▼
[7] StepImportResult { document, graph, report, entities }
```

### 7.4 Export Paths

| Format | Function | Source |
|--------|----------|--------|
| STL ASCII | `write_ascii_stl()` | `&[Vec3]` + `&[i32]` |
| STL Binary | `write_binary_stl()` | `&[Vec3]` + `&[i32]` |
| STEP (graph) | `write_step_from_graph()` | `SceneGraph` (mesh reverse-engineering) |
| STEP (entities) | `write_step_from_entities()` | `EntityIndex` (clean pass-through) |
| STEP (parametric) | `write_step_parametric()` | `BRepStore` (exact B-Rep topology) |
| IGES | `write_iges()` / `write_iges_string()` | `BRepStore` |
| IV | `write_iv()` | `SceneGraph` |
| BREP Binary | `write_brep_binary()` | `BRepStore` |
| BREP ASCII | `write_brep()` | `BRepStore` (rc3d-shape) |

---

## 8. Render Layer (`rc3d-render`)

### 8.1 `Renderer` — Main Struct

Owns all GPU state: device, queue, surface, pipelines, shadow resources, post-processing passes, HUD, viewport layout, mesh pool, shape cache, light set table.

### 8.2 Per-Frame Render Flow

```
render_draw_calls(draw_calls, scene)
    │
    ├── 1. FlatDrawCache update (persistent per-draw metadata, 64B+72B per draw)
    ├── 2. BVH build/incremental update from draw call AABBs
    ├── 3. Frustum cull (CPU BVH → visible indices)
    ├── 4. GPU compute cull (optional, async readback replaces CPU results)
    ├── 5. Mesh upload (GpuMeshPool, LRU+budget, max 16 uploads/frame)
    ├── 6. Sort draw calls (opaque/edge/selected/transparent groups)
    ├── 7. CSM setup (light view-proj matrices, split distances)
    ├── 8. Upload global frame uniforms (lights, IBL, CSM)
    ├── 9. Build PassContext { visible draws, sort orders, matrices, settings }
    │
    ▼
execute_passes() — `render_passes/{execute,film,overlay}.rs`, single command encoder
    │
    ├── Background (gradient/image/solid)
    ├── GPU Cull Dispatch (compute shader, if enabled)
    ├── CSM Shadow Depth (render to depth array texture)
    ├── Omni Shadow (cube-array, up to 4 point lights x 6 faces; transparent casters)
    ├── Meshlet HZB Prepass (optional: cull → depth → HZB build → fine cull)
    ├── Cluster Light Cull (compute shader)
    ├── Solid + Outline (PBR deferred, Cook-Torrance + Lambertian)
    ├── Section Caps (back-face render with clip planes)
    ├── Transparent (WBOIT MRT on LDR/HDR shade, or painter fallback)
    ├── Effects (Decal, Volume, PointCloud)
    ├── Wireframe Overlay (on film, or deferred after FXAA blit)
    ├── Selection Fill + Edge + BBox
    ├── Feature Edge Overlay (depth-tested + anti-aliased)
    ├── LDR FXAA / LDR film blit / HDR Post-Processing
    ├── Compositor (Viewer-reachable CAD look flags, then Mix/Blur GPU ops on the film)
    ├── Screen-space edges + deferred line overlays
    ├── Ground Grid
    ├── Gizmo Overlay (Engine.gizmo generate_lines)
    ├── Viewport Borders
    ├── Markup Lines (annotation overlay with occlusion downsampling)
    ├── Overlay composite (nav cube tiles)
    ├── HUD Overlay (FPS, mode name; top-left of the 3D film)
    └── post_swapchain callback (egui)
```

### 8.3 PBR Shader Binding Model

| Bind Group | Content |
|------------|---------|
| Group 0 | Per-draw uniforms (`SceneUniforms`/`FlatUniforms`) via `GpuUniformPool` with dynamic offset |
| Group 1 | PBR material textures (albedo, sampler, normal, metallic-roughness, emissive, occlusion) |
| Group 2 | Shadow + Global Frame (CSM depth array, omni cube-array, comparison samplers, global uniform buffer) |
| Group 3 | IBL + Instance SSBO (environment map, BRDF LUT, env sampler, instance data SSBO, morph target buffer) |

### 8.4 Shader Permutation System

- 7 feature flags: `HAS_NORMAL_MAP`, `HAS_SHADOW`, `HAS_ALBEDO_TEX`, `HAS_IBL`, `HAS_MR_TEX`, `HAS_EMISSIVE_TEX`, `HAS_OCCLUSION_TEX`
- `preprocess_wgsl()` — `#ifdef`/`#ifndef`/`#else`/`#endif` → pure WGSL (supports nesting)
- `ShaderVariantCache` — LRU cache keyed by `(PermutationKey, source_kind)`
- `ShaderReload` — poll-based file watcher (30-frame interval), invalidates cache on change

### 8.5 GPU Resource Management

| Resource | Type | Strategy |
|----------|------|----------|
| `GpuMeshPool` | Vertex+Index buffers | LRU eviction, 512MB budget, 4096 slots |
| `ShapeCache` | CPU-side mesh cache | `ShapeKey` hash (Cube/Sphere/Cone/...) |
| `FlatDrawCache` | Per-draw metadata | Persistent hot (64B) + cold (72B) data |
| `LightSetTable` | Light parameter dedup | 1280B→4B per draw call |
| `GpuUniformPool` | Uniform buffer sub-allocation | Ring-buffer per pool type |

### 8.6 Display Modes

| Mode | Solid Pipeline | Edges | Shadows | Post-FX |
|------|---------------|-------|---------|---------|
| `Shaded` | PBR + IBL | None | Yes | Full |
| `ShadedWithEdges` | PBR + IBL | Feature edges | Yes | Full |
| `HiddenLine` | Solid dark gray | Visible crease + dashed occluded (Fast HLR) | Yes | Full |
| `Flat` | Solid color | None | No | Reduced |
| `FlatWithEdge` | Solid color | Feature edges | No | Reduced |
| `Wireframe` | Wireframe only | Full topology | No | Reduced |

Presets above still seed both axes. Per-node `fill_style` / `edge_style` compose independently. `EdgeStyle` classification: `Crease` = hard+perimeter (preset default), `Hard` / `Perimeter` / `Adjacent` / `Silhouette` use a per-draw line list (not the shared GPU crease buffer); `Full` is all unique mesh edges. Named catalog: `VisualStyleLibrary::builtin()` (`Shaded`, `Wireframe`, `HardEdges`, `Silhouette`, `HiddenLine`, …) applied via `SceneGraph::apply_visual_style` / `Engine::apply_visual_style`. HiddenLine uses GPU Fast HLR (`edge_overlay_hidden`: inverted depth + screen-space dash). Vector hardcopy: `Engine::export_hidden_line_svg` (`vector_hlr.rs`) classifies the same crease/wire edges against filled triangles. Not HOOPS HIO analytic HLR. PMI: `AnnotationSetNode.pmi` (`PmiRecord` / `PmiBinding`) resolved by `SceneGraph::bind_pmi` / `Engine::bind_scene_pmi`; JSON sidecar via `PmiDocument` (not a STEP AP242 parser).

### 8.7 Quality Tiers (`CadDisplayTier`)

| Tier | PBR | IBL | CSM | SSAO | TAA | HDR | SSR | VolFog | DoF |
|------|-----|-----|-----|------|-----|-----|-----|--------|-----|
| DesignCreation (0) | Flat | — | — | — | — | — | — | — | — |
| Visualization (1) | ✓ | ✓ | ✓ | — | — | — | — | — | — |
| IndustrialDisplay (2) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — |
| ProductRendering (3) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Identity compositor (`Beauty → Viewer`) derives these flags from the effective tier each frame. Viewer-reachable CAD compositor nodes override them as a test matrix (absent node = off). Bloom is compositor-only. HDR path uses TAA; LDR uses FXAA (not both). Color grading follows Industrial/Product. Visualization keeps motion blur off (orbit smear). Compositor presets: Identity, Industrial (SSAO+TAA+grade+shadows), Product (+SSR/fog/DOF/shadows), Hidden Line, X-Ray Edges, SSAO+TAA, Edges only. HUD `CAD:` line reports effective/requested tier, `tier` vs `comp` source, and active flags. Studio verification: `rtk cargo run -p rc3d-studio -- --cad-matrix` (writes `target/cad-matrix-report.txt`).

Interaction on requested tier 2/3 degrades `effective_tier` to Visualization until cooldown recovery. Identity graphs follow that drop; Viewer-reachable CAD compositor nodes are re-applied after `update_tier` so the test matrix still wins during orbit.

### 8.8 GPU Culling Pipeline

```
Frame N-1: GPU Cull Dispatch (compute shader)
    │  ├── Upload object transforms + AABBs to storage buffer
    │  ├── Upload frustum planes to uniform buffer
    │  ├── Dispatch: per-object frustum test → atomic counter → instance index buffer
    │  └── Copy visible count+indices to staging buffer
    │
Frame N: Map staging buffer (non-blocking)
    │  ├── Read visible instance indices
    │  └── Replace CPU BVH culling results
```

---

## 9. Application Layer

### 9.1 `rc3d-engine-api::Engine` — The Primary Facade

```rust
pub struct Engine {
    pub world: World,                          // SceneGraph + engines + collector
    pub renderer: Option<Renderer>,            // wgpu GPU renderer
    pub controller: CameraController,          // Orbit/pan/zoom controller
    pub viewport_cameras: ViewportCameraSet,   // Quad pack: vp.Top/Front/Right/Persp
    pub fps: FpsTracker,
    pub adaptive_control: AdaptiveControl,
    pub hidden_nodes: HashSet<NodeId>,
    pub visual_styles: VisualStyleLibrary,     // Named HOOPS-style catalog
    pub gizmo: Gizmo,                          // Transform manipulator overlay
    pub hud_text_hook: Option<Box<dyn Fn() -> String>>,
    pub pre_render_hook: Option<Box<dyn FnMut(&mut Renderer)>>,
    pub on_pick: Option<Box<dyn FnMut(&mut SceneGraph, NodeId, Vec3)>>,
    // ...
}
```

**Key methods:**
- `new(window)` — async wgpu init via pollster
- `render()` — single-frame main loop (see flow below)
- `load_scene(graph)` — replace scene, invalidate caches
- `bind_scene_pmi()` — resolve `AnnotationSet.pmi` names and stamp unbound annotation points
- `export_hidden_line_svg(path)` — Fast HLR edges as SVG (visible solid, hidden dashed)
- `set_gizmo_mode` / `set_lod_range_scale` — manipulator mode and per-separator LOD scale
- `import(path)` — import file into scene graph (delegates to `rc3d-io`)
- `camera_mut()` — access orbit/pan/zoom controller

### 9.2 `Engine::render()` — Complete Frame Flow

```
render()
  ├── 1. Reapply CAD tier constraints to renderer
  ├── 2. world.evaluate_engines()            Run simulation engines on scene graph
  ├── 3. renderer.set_materials()            Sync material library
  ├── 4. world.reset_collector(dm)           Clear draw calls, reset state stacks
  ├── 5. Apply hidden_nodes set
  ├── 6. Update camera nodes                 Single: main controller; Quad pack: per-viewport controllers
  ├── 7. world.traverse_all_roots()          RenderCollector → Vec<DrawCall>
  ├── 8. renderer.collect_markup_vertices()  Annotation overlay extraction
  ├── 9. Transfer light sets + clip planes   From collector to renderer
  ├──10. Fallback camera                     Default perspective if none found
  ├──11. apply_world_camera()                View/projection → all draw calls
  ├──11b.apply_ghost_unselected()            HOOPS Ghost: unselected fill → Blend
  ├──12. Cache draw calls                    For static-frame fast path
  ├──13. clear_all_dirty_flags()             Reset after render
  ├──14. Transfer effect commands            Decals, volumes, point clouds
  ├──15. Call pre_render_hook
  ├──16. render_draw_calls / render_standard_quad_views  Single pass or 4-tile blit
  ├──17. Update HUD overlay                  FPS, mode, markup text
  ├──18. Report frame time                   For adaptive quality control
  └──19. fps.push()                          Track metrics
```

### 9.3 Editor Architecture

```
Editor                                // Public API
  └── EditorUi                        // egui context + renderer
        ├── egui::Context              // Immediate-mode UI state
        ├── egui-wgpu renderer        // GPU overlay pass
        ├── EditorCommand queue        // VecDeque<EditorCommand>
        └── UI panels:
              ├── Menu bar (File, Edit, View, Tools, Bookmarks, Create, Settings)
              ├── Shared Keymap (menus + Studio host + Settings rebind)
              ├── Floating left tool strip (Photoshop-style flyouts, display/overlay)
              ├── Right side dock (icon tabs: Hierarchy / Inspector / Render / History / Assets)
              ├── Bottom dock (icon tabs: Document / Compositor / Console)
              └── Workspace presets (Model / LookDev / Compositor)
```

**EditorCommand** (~80 variants): undo, redo, transform commit, camera change, material edit, visibility toggle, display mode, create/delete node, import file, bookmark, markup mouse, measurement pick, hide/isolate/lock, history jump, keymap bind, **LoadCase / SetCaseParam / RunCaseStep** (Studio case library), etc. `apply_command` in `rc3d-editor/src/apply/` routes by domain (`io`, `render`, `scene`, `tools`). Studio is a thin host (winit/DWM); editor chrome stays in `rc3d-editor`. The eframe `ui/panel.rs` window is examples-only.

**Studio case library** (`rc3d-studio/src/cases/`): catalog of param/process demos built from shared scene builders (not forked example `main`s). Floating **Case Library** window plus Assets side tab; param sliders emit `SetCaseParam`, process buttons emit `RunCaseStep`. Stress/diagnostic examples remain CLI-only under `rc3d-examples`.

### 9.4 Gizmo System

**Modes:** `Translate`, `Rotate`, `Scale`

**Handles:**
| Handle | Shape | Ray Test |
|--------|-------|----------|
| `TranslateArrow(axis)` | Cylinder shaft + cone tip | `ray_intersect_cylinder` + `ray_intersect_cone` |
| `RotateRing(axis)` | Torus (ring of spheres) | `ray_intersect_torus` |
| `ScaleHandle(axis)` | Line + cube tip | `ray_intersect_cube` |
| `TranslatePlane(axis)` | XY/YZ/ZX plane square | `ray_intersect_plane_rect` |

**Drag Delta:** axis drag = projection to line (1 DoF), plane drag = 2D delta in plane.

### 9.5 CLI Editor — Dual-Thread Architecture

```
┌──────────────────────┐     ┌──────────────────────┐
│   GPU Thread          │     │   TUI Thread          │
│   (winit event loop)  │     │   (ratatui terminal)  │
│                       │     │                       │
│  Renderer + egui      │◄────│  CLI Command Parser   │
│  Viewport + Panels    │ cmds│  Command Dispatch     │
│                       │     │  EngineState (shared) │
│  EngineState          │────►│  Scrollback + History │
│  (Arc<RwLock<..>>)    │     │                       │
└──────────────────────┘     └──────────────────────┘
```

Commands: `scene load`, `camera orbit/pan/zoom/fit`, `select`, `prop set`, `display`, `test run`, `log filter`, `help`, `quit`.

### 9.6 `rc3d-scene-api` — Declarative Scene DSL

```rust
let scene = Scene::new()
    .add(Cube::default().at(0.0, 0.0, 0.0).material(my_material))
    .add(Sphere::default().at(2.0, 0.0, 0.0).radius(0.5))
    .set_camera(PerspectiveCamera::default().at(5.0, 5.0, 5.0))
    .add_light(DirectionalLight::default().direction(-1, -1, -1))
    .build();
// Returns SceneGraph ready for engine.load_scene()
```

### 9.7 Examples Architecture

Three tiers of example complexity:

```
Tier 1: run_example("Title", |engine| { ... })
        Minimal: build scene via NodeData, camera auto-driven by controller
        Used by: cube, hello_scene, materials, primitives, etc.

Tier 2: run_example_with_hooks("Title", |engine| { ... })
        Adds: on_pick, panel_overlay hooks, continuous_redraw
        Used by: render_features, annotation_edit, markup_dimensions

Tier 3: run_app(MyApp { ... })
        Custom ApplicationHandler, full event control
        Used by: editor, multi-window scenarios
```

---

## 10. Complete Data Flow — File to Pixels

```
FILE (.step/.stl/.obj/.gltf/.fbx/.iv/.wrl)
    │
    ▼
rc3d-io parser ── format-specific parser → internal representation
    │
    ├── Mesh formats (STL/OBJ/glTF/FBX/VRML):
    │   direct → SceneGraph (Separator → Transform → Material → IndexedFaceSet)
    │
    └── B-Rep formats (STEP/IGES):
        EntityIndex → BRepStore → ShapeDocument
            │
            ├── HealPipeline (auto_heal_shell, tolerance fix, wire order, etc.)
            │
            ├── MeshEmitPlan (BRepMeshConfig → edge discretization → face CDT → shell assembly)
            │   → MeshResult { vertices: [f64;3], normals, indices }
            │
            └── SceneEmitPlan → SceneGraph (Separator → Transform → Material → Coordinate3+Normal+IndexedFaceSet)
                [f64→f32 conversion at boundary]

    ▼
SceneGraph
    │
    ▼
PER FRAME:
    │
    world.evaluate_engines()          ← ElapsedTime, Calculator, Interpolate, etc.
    world.traverse_all_roots()        ← RenderCollector (state machine traversal)
    │
    ▼
Vec<DrawCall>                         ← geometry + transforms + materials + lights
    │
    ▼
FlatDrawCache update                  ← persistent per-draw metadata (dirty-aware)
BVH frustum cull                      ← CPU or GPU compute (async readback)
Mesh upload (GpuMeshPool)             ← LRU + budget, max 16/frame
    │
    ▼
execute_passes() ── `render_passes/{execute,film,overlay}.rs`, single command encoder:
    Background → CSM Shadows → Omni Shadows → HZB Prepass → Cluster Cull →
    Solid → Section Caps → Transparent (WBOIT) → Effects →
    Wireframe → Selection → Edge Overlay → LDR/HDR present → Compositor (CAD look + film ops) →
    SS edges / deferred lines → Grid/Gizmo/Markup → Overlay composite → HUD → Swapchain
    │
    ▼
GPU presents to window
```

---

## 11. External Interfaces Summary

### 11.1 Public C API

None. All interfaces are Rust-native via crate public APIs.

### 11.2 Public Rust API Surface

| Crate | Primary Public Types | Consumer Use Case |
|-------|---------------------|-------------------|
| `rc3d-engine-api` | `Engine`, `World`, `CameraController`, `ViewportCameraSet` | Main integration point for any application |
| `rc3d-scene-api` | `Scene`, `NodeHandle`, `Shape` trait, geometry/materials/lights constructors | Declarative scene construction |
| `rc3d-editor` | `Editor`, `EditorCommand` | Embeddable 3D editor widget |
| `rc3d-shape` | `BRepStore`, geometry types, `MeshResult`, `write_brep`, boolean/heal/mesh functions | B-Rep processing (advanced use) |
| `rc3d-io` | `import_file()`, per-format parse/write functions | File format conversion |

### 11.3 Extension Points

| Mechanism | Location | Purpose |
|-----------|----------|---------|
| `Engine` trait | `rc3d-engine` | Custom simulation engines |
| `NodeHandler` trait | `rc3d-scene` | Custom node types without modifying `NodeData` enum |
| `Custom(u16, Box<dyn CustomNodeData>)` | `rc3d-scene` + `NodeTypeRegistry` | User-registered custom node variants |
| `pre_render_hook` | `rc3d-engine-api::Engine` | Pre-render callback |
| `hud_text_hook` | `rc3d-engine-api::Engine` | Custom HUD overlay text |
| `on_pick` / `on_pick_hit` | `rc3d-engine-api::Engine` | Node pick + face/edge `PickHit` |
| `set_face_tint` / `set_edge_tint` | `SceneGraph` | HOOPS sub-entity color (cube `tri/2`, IFS `face_ids`) |
| `Action` trait | `rc3d-actions` | Custom scene graph traversals |
| `FieldMap::connect()` | `rc3d-fields` | Field propagation graphs for engines |

### 11.4 File-Based Interfaces

| Interface | Direction | Format |
|-----------|-----------|--------|
| STEP (.step/.stp) | In/Out | ISO 10303-214 Part21 ASCII |
| STL (.stl) | In/Out | ASCII + Binary |
| OBJ (.obj) | In | Wavefront ASCII |
| glTF (.gltf/.glb) | In | glTF 2.0 JSON + Binary |
| FBX (.fbx) | In | FBX 7400 Binary |
| VRML (.wrl) | In | VRML97 ASCII |
| IV (.iv) | In/Out | OpenInventor 1.0 ASCII |
| IGES (.igs/.iges) | In/Out | IGES 5.3 Fixed-Width ASCII |
| BREP (.brep) | Out | OCC CASCADE Topology V1 ASCII |
| BREP Binary (.brepbin) | In/Out | Custom binary (magic: `RUSTBREP`) |
| Pipeline Cache | In/Out | wgpu binary cache (`pipeline_cache.bin`) |

### 11.5 Runtime Configuration

| Mechanism | Type |
|-----------|------|
| `RC3D_STEP_DIR` | Environment variable for STEP test data path |
| `DisplayMode` | Per-node or global rendering mode |
| `ghost_unselected` | HOOPS Isolate/Ghost: unselected filled draws become translucent |
| `CadDisplayTier` | Quality tier (DesignCreation → ProductRendering) |
| `AdaptiveQuality` | Auto-downgrade based on frame time EMA |
| `HealLevel` | Healing aggressiveness (Basic/Standard/Advanced) |
| `BRepMeshConfig` | Mesh quality parameters (deflection, tolerance, fallback) |
| `TessellationTier` | Mesh detail level (Preview/Standard/Precision) |
| `CameraController` bookmarks | 9 slots (Ctrl+digit save, digit recall) |

---

## 12. Key Architectural Patterns

### 12.1 SlotMap Everywhere
Both `SceneGraph` and `BRepStore` use `slotmap::SlotMap<K, V>` for O(1) access with stable, generational keys. This avoids dangling references and enables concurrent iteration patterns.

### 12.2 Visitor Pattern for Traversal
The `Action` trait separates traversal logic from the data structure. Multiple actions (`RenderCollector`, `RayPickAction`, `GetBoundingBoxAction`) traverse the same scene graph independently. `Separator` nodes provide state push/pop isolation.

### 12.3 Engine = Direct Mutation
Unlike message-passing architectures, engines receive `&mut SceneGraph` and mutate nodes directly via `NodeId`. They set `dirty_flags` to trigger render cache invalidation — the renderer never needs to diff the graph.

### 12.4 Dual-Precision Boundary
The geometry kernel (`rc3d-shape`) uses `f64` (`Real`/`PVec3`) throughout. Conversion to `f32` (`Vec3`/`Mat4`) happens at the `SceneEmitPlan → SceneGraph` boundary for GPU consumption. This preserves CAD precision while respecting GPU hardware.

### 12.5 B-Rep → Mesh Late Binding
STEP/IGES import builds a full `BRepStore` with exact geometry. Meshing (CDT, grid, fill) happens only when needed for rendering or STL export. Heal passes run on the B-Rep representation before meshing.

### 12.6 No Common Import Trait
Each file format has its own module with independent functions. The unified `import_file()` dispatches by extension. This avoids forcing dissimilar formats (mesh vs B-Rep) into a common abstraction.

### 12.7 Static Frame Fast Path
When scene and camera are unchanged for 2+ frames, BVH culling is skipped entirely. `FlatDrawCache` persists draw metadata across frames; only dirty nodes trigger rebuild.

---

*Document version: 1.0 — Generated from codebase analysis at commit 5bc1653*
