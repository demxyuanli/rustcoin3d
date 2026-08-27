# Scene Graph System

## 1. Design Overview

The scene graph is a directed acyclic graph of typed nodes. Each node has children (for hierarchy) and typed data (`NodeData` enum). The traversal model follows the Coin3D/Open Inventor pattern: a Separator node saves traversal state before entering children and restores it on exit, providing state isolation between sibling subtrees.

## 2. SceneGraph API

### 2.1 Construction

```rust
let mut graph = SceneGraph::new();

// Add a root node (no parent)
let root = graph.add_root(NodeData::Separator(SeparatorNode));

// Add a child
let child = graph.add_child(root, NodeData::Cube(CubeNode::default()));

// Insert at specific index
graph.insert_child(root, 0, NodeData::Sphere(SphereNode::default()));

// Remove (recursively deletes entire subtree)
graph.remove(child);
```

### 2.2 Access

```rust
// O(1) lookup by NodeId
if let Some(entry) = graph.get(node_id) {
    println!("Node: {:?}", entry.data.type_name());
}

// Mutable access
if let Some(entry) = graph.get_mut(node_id) {
    entry.name = Some("MyCube".to_string());
}

// Query relationships
let children = graph.children(node_id);  // Option<&[NodeId]>
let roots = graph.roots();                // &[NodeId]
let count = graph.node_count();
```

### 2.3 Traversal

```rust
// DFS pre-order from a specific root
for (node_id, node_entry) in graph.traverse_dfs(root) {
    // process node
}

// From all roots at once
for (node_id, node_entry) in graph.traverse_all() {
    // process node
}

// Get all nodes in a subtree pre-order (IDs only)
let ids: Vec<NodeId> = graph.subtree_preorder_ids(root);
```

## 3. NodeData Enum (Complete Reference)

### 3.1 Grouping Nodes

| Variant | Struct | Coin3D Equivalent | Behavior |
|---------|--------|-------------------|----------|
| `Separator` | `SeparatorNode` | SoSeparator | Push/pop all 8 state elements before/after children |
| `Group` | `GroupNode` | SoGroup | Ordered children container, no state save/restore |
| `Billboard` | `BillboardNode` | SoBillboard | Rotates children to face camera (axis-aligned or spherical) |
| `Transform` | `TransformNode` | SoTransform | Translation, rotation (quaternion), scale, center, scaleOrientation |
| `Rotation` | `RotationNode` | SoRotation | Axis-angle rotation; multiplies the current model matrix |
| `RotationXYZ` | `RotationXYZNode` | SoRotationXYZ | Rotation about X, Y, or Z (radians) |
| `ResetTransform` | `ResetTransformNode` | SoResetTransform | Resets model matrix to identity |
| `File` | `FileNode` | SoFile/SoWWWInline | External file reference for composition |

### 3.2 Attribute Nodes

| Variant | Struct | Purpose |
|---------|--------|---------|
| `Coordinate3` | `Coordinate3Node` | Vertex positions (Vec\<Vec3\>) |
| `TextureCoordinate2` | `TextureCoordinate2Node` | UV coordinates per vertex |
| `Normal` | `NormalNode` | Per-vertex normals |
| `Material` | `MaterialNode` | PBR parameters, `toon_steps`, `visualize_normals`, `visualize_depth` |
| `ShapeHints` | `ShapeHintsNode` | Vertex ordering, shape type, face type, crease angle |
| `MaterialBinding` | `MaterialBindingNode` | Material binding mode (per-vertex, per-face, etc.) |
| `Texture2Transform` | `Texture2TransformNode` | 2D UV transformation (translation, rotation, scale, center) |
| `Environment` | `EnvironmentNode` | Ambient intensity/color, attenuation, fog parameters |

### 3.3 Shape Nodes

| Variant | Struct | Vertices | Description |
|---------|--------|----------|-------------|
| `Triangle` | `TriangleNode` | 3 | Single triangle from first 3 coordinates |
| `Cube` | `CubeNode` | 24 (6 faces) | Axis-aligned box (width, height, depth) |
| `Sphere` | `SphereNode` | Variable | UV sphere (radius, segments) |
| `Cone` | `ConeNode` | Variable | Cone (bottom radius, height, segments) |
| `Cylinder` | `CylinderNode` | Variable | Cylinder (radius, height, segments) |
| `IndexedFaceSet` | `IndexedFaceSetNode` | Arbitrary | Triangle mesh; optional `face_ids` + `SceneGraph` face/edge tints |
| `IndexedLineSet` | `IndexedLineSetNode` | Arbitrary | Line segments (line width, color) |
| `SkinnedMesh` | `SkinnedMeshNode` | Arbitrary | Skeleton + vertex skinning data + animation clips |
| `MorphTarget` | `MorphTargetNode` | Arbitrary | Blend shape targets with position/normal/tangent deltas |

### 3.4 Camera Nodes

| Variant | Struct | Parameters |
|---------|--------|------------|
| `PerspectiveCamera` | `PerspectiveCameraNode` | position, orientation, fov, near, far, aspect, reverse_depth |
| `OrthographicCamera` | `OrthographicCameraNode` | position, orientation, height, near, far, aspect, reverse_depth |
| `StereoCamera` | `StereoCameraNode` | base camera, IPD, convergence; `Engine` renders L/R eyes (SBS / top-bottom / anaglyph). IPD~0 = mono |
| `CubeCamera` | `CubeCameraNode` | position, near, far, resolution, update_period, enabled — 6-face capture to local IBL |

Helper method:
```rust
PerspectiveCameraNode::look_at(eye, target, up, fov, aspect)
```

### 3.5 Light Nodes

| Variant | Struct | Parameters | Shadow |
|---------|--------|------------|--------|
| `DirectionalLight` | `DirectionalLightNode` | direction, color, intensity, light_group | CSM (4 cascades) |
| `PointLight` | `PointLightNode` | position, color, intensity, light_group | Omni cubemap |
| `SpotLight` | `SpotLightNode` | position, direction, color, intensity, cutoff, falloff, light_group | - |
| `HemisphereLight` | `HemisphereLightNode` | sky_color, ground_color, intensity, direction | - |
| `LightProbe` | `LightProbeNode` | sh (9 RGB L2 coeffs), intensity | - |
| `Sprite` | `SpriteNode` | texture_path, color, size, size_attenuation, center, opacity | - |
| `AreaLight` | `AreaLightNode` | position, direction, color, intensity, width, height, shape | - |

### 3.6 Traversal Control

| Variant | Struct | Description |
|---------|--------|-------------|
| `Lod` | `LodNode` | Level-of-detail based on camera distance (levels: Vec\<LodLevel\>) |
| `Switch` | `SwitchNode` | Conditional child visibility (which_child: -1=all, -2=none, 0..N=specific) |
| `MultipleCopy` | `MultipleCopyNode` | Repeat children N times with N transform matrices |
| `SectionPlane` | `SectionPlaneNode` | Clipping plane (equation, cap color, optional hatch) |
| `PickStyle` | `PickStyleNode` | Controls raypick-ability of children |
| `EventCallback` | `EventCallbackNode` | Event routing marker for HandleEventAction (`Engine::handle_window_event`) |
| `TransformManip` | `TransformManipNode` | Scene-graph transform manipulator (mode, space, size, target). `target = None` binds the preceding sibling Transform. Child `Dragger` nodes filter gizmo handles |
| `Dragger` | `DraggerNode` | Composable axis/plane/rotate/scale part under a `TransformManip` |

### 3.7 Annotation Nodes

| Variant | Struct | Description |
|---------|--------|-------------|
| `Text2` | `Text2Node` | Screen-space 2D text (position, size, color, string) |
| `Text3` | `Text3Node` | World-space 3D text (position, size, color, string) |
| `Font` | `FontNode` | Coin3D SoFont: `name`, `size` (0 keeps text size), `FontStyle` (Sans/Serif/Typewriter). Applies to subsequent Text2/Text3 siblings. World labels rasterize to SDF. |
| `Measurement` | `MeasurementNode` | Distance/angle/radius/diameter measurement |
| `Markup` | `MarkupNode` | Rich markup (lines, rects, circles, dimensions, leaders, callouts) |
| `Annotation` | `AnnotationNode` | Overlay rendering group — children rendered without depth test (Coin3D SoAnnotation) |
| `AnnotationSet` | `AnnotationSetNode` | 3D annotation elements placed under an `Annotation` node. Contains `Vec<AnnotationElement>` |

**AnnotationElement variants** (3D world-space, projected to screen each frame):
| Variant | Purpose |
|---------|---------|
| `Dimension` | Linear dimension (extension lines + arrows); `label_mode` + auto format |
| `AngleDimension` | Arc between two rays from `center` |
| `RadialDimension` | Center to perimeter with arrow |
| `DiameterDimension` | Line through center between `p1` and `p2` |
| `Leader` | Anchor to label (pixel `label_offset`) |
| `Callout` | Leader + circular callout at label |
| `Datum` | Datum cross at a point |

**AnnotationSetNode** also carries `AnnotationStyle` (extension/arrow defaults, `label_height_factor`, `font_size` as target screen height in px, `decimals`, `unit_suffix`, `arc_segments`).

**Label modes** (`AnnotationLabelMode`): `Auto` (format from geometry when `label` empty), `Fixed` (use `label`), `Prefix` (`label` + formatted value).

**Logic** lives in `rc3d-scene/src/annotation/` (geometry, format, label resolve). Rendering: `pass_markup::project_annotation_elements`.

**AnnotationPoint** — spatial fields (`start`, `end`, `center`, `anchor`, `position`, …) use `AnnotationPoint` instead of raw `[f32; 3]`:
- `AnnotationPoint::local([x,y,z])` — coordinates in `AnnotationSet` local space (default).
- `AnnotationPoint::on_node(node_id, local)` — point in that node's local space; renderer uses `prepare_annotation_for_render` (`model_matrix = set_matrix * node_world`).
- Serde: plain `[f32;3]` array or `{ "node": "...", "local": [...] }`.

**From interactive measurement** — `rc3d_scene::annotation::{world_distance_annotation, world_angle_annotation, world_radius_annotation, world_diameter_annotation}` build an `AnnotationSetNode` from world picks. `rc3d_actions::MeasurementAction::build_annotation_set()` / `create_annotation_node()` wrap this for tools.

**Design rules**:
1. **Single-plane**: Each annotation on one axis-aligned plane (∥XY/XZ/YZ).
2. **All 3D**: Geometry in world units, one VP projection per frame.
3. **Fixed world axes**: Leader/Callout label offsets map along world +X/+Y from anchor.
4. **World-space text**: Labels are textured quads (`world_label` pass), camera-facing with per-frame scaling so on-screen height matches `font_size` (pixels), independent of camera distance.
5. **Label font**: On startup loads the first available font from `RC3D_FONT_PATH`, `RC3D_FONT_DIR`, or OS defaults (`arial.ttf` / `segoeui.ttf` on Windows). Set `RC3D_FONT_PATH` to a `.ttf` before launching the app to use a custom font.

**Scene graph example**:
```rust
let ann = graph.add_child(root, NodeData::Annotation(AnnotationNode));
let obj_ann = graph.add_child(ann, NodeData::Separator(SeparatorNode));
graph.add_child(obj_ann, NodeData::Transform(/* object-local translation */));
graph.add_child(obj_ann, NodeData::AnnotationSet(AnnotationSetNode {
    elements: vec![
        AnnotationElement::Dimension { start, end, offset_dir, ... },
        AnnotationElement::Leader { anchor, label_offset, ... },
        AnnotationElement::Datum { position, size, ... },
    ],
    visible: true,
    pmi: vec![], // PmiRecord: id / kind / bindings (node_name, face, edge) / tolerances
}));
```

`AnnotationSet.pmi` is the semantic layer (STEP-style id, face/edge refs, ±tol). `SceneGraph::set_name` + `bind_pmi` resolve `node_name` to `NodeId` and stamp unbound `AnnotationPoint`s. JSON interchange: `PmiDocument` (`Engine::apply_pmi_json`). Visual geometry is unchanged.

`AnnotationSet` captures `model_matrix` from its position in the scene graph (including accumulated Transforms within the same `Separator`). The renderer projects all 3D annotation points to screen-space `MarkupVertex` in `pass_markup::project_annotation_elements`.

### 3.8 Specialized Nodes

| Variant | Struct | Description |
|---------|--------|-------------|
| `ExplodedView` | `ExplodedViewNode` | Part movement for exploded view rendering |
| `ReflectionPlane` | `ReflectionPlaneNode` | Planar reflection rendering |
| `Decal` | `DecalNode` | Screen-space projected texture decal |
| `RayTracing` | `RayTracingNode` | GPU ray tracing mode toggle |
| `Volume` | `VolumeNode` | Volumetric data metadata |
| `PointCloud` | `PointCloudNode` | Out-of-core point cloud reference |
| `MorphTarget` | `MorphTargetNode` | Blend shape deltas |
| `InstancedMesh` | `InstancedMeshNode` | GPU-instanced mesh (per-instance transforms) |
| `BatchedMesh` | `BatchedMeshNode` | Packed multi-geometry batch; instances reference geometry ranges + local transforms |

### 3.9 Extensibility Nodes

| Variant | Storage | Purpose |
|---------|---------|---------|
| `HandlerNode` | `Arc<dyn NodeHandler>` | Custom traversal behavior (trait-based) |
| `Custom(u16, Box<dyn CustomNodeData>)` | Type ID + trait object | Fully user-defined node types via `NodeTypeRegistry` |

## 4. Traversal (State Stack Model)

### 4.1 State Stack Operations

During traversal, an `rc3d-actions::State` carries 8 `Element` stacks:

```
ModelMatrix     ─┐
ViewMatrix       │
ProjectionMatrix │
Coordinate3      ├── Each stack is Vec<Box<dyn Element>>
Normal           │     Push: Separator entry
Material         │     Pop:  Separator exit
Light            │     Modify: attribute nodes
TextureCoord2   ─┘
```

### 4.2 Separator Behavior

```
Separator.enter(state):
    save by pushing current values onto all stacks

Separator.exit(state):
    restore by popping all stacks
```

### 4.3 Transform Accumulation

```rust
// render_action.rs
NodeData::Transform(t) => {
    let m = current_model_matrix * t.to_matrix();  // accumulate
    // state.set_model(m);
}
```

Important: Transforms accumulate *without automatic restore*. Without a Separator boundary, sibling transform nodes will leak into each other. Always wrap separate transform groups in Separators.

### 4.4 Billboard Facing

```rust
fn compute_billboard_facing(axis_aligned: bool, pos: Vec3, camera: Vec3) -> Mat4 {
    let dir = (camera - pos).normalize();
    if axis_aligned {
        // Spherical: constrain Y rotation, face camera in XZ plane
        let fwd = Vec3::new(dir.x, 0.0, dir.z).normalize();
        Mat4::look_at_rh(Vec3::ZERO, fwd, Vec3::Y)
    } else {
        // Spherical: full 3-axis facing
        let right = Vec3::Y.cross(dir).normalize();
        let up = dir.cross(right).normalize();
        Mat4::from_cols(right.extend(0), up.extend(0), (-dir).extend(0), Vec3::ZERO.extend(1))
    }
}
```

## 5. Dirty Flag System

### 5.1 Flag Bits

```rust
pub const TRANSFORM: u8  = 1 << 0;  // 1   - Model matrix changed
pub const MATERIAL: u8   = 1 << 1;  // 2   - Material properties changed
pub const GEOMETRY: u8   = 1 << 2;  // 4   - Geometry data changed
pub const CHILDREN: u8   = 1 << 3;  // 8   - Children added/removed/reordered
pub const REMOVED: u8    = 1 << 4;  // 16  - Node removed from graph
pub const FROZEN: u8     = 1 << 7;  // 128 - Static subtree, skip traversal
```

### 5.2 Render Cache Invalidation

- `TRANFORM` dirty → re-upload model matrix for affected draw calls
- `MATERIAL` dirty → re-upload material uniform
- `GEOMETRY` dirty → re-tessellate and upload GPU buffers
- `CHILDREN` dirty → rebuild draw call list for subtree
- `FROZEN` set → skip entire subtree during render traversal
- `FROZEN` cleared → force full subtree rebuild

### 5.3 Methods

```rust
graph.clear_all_dirty_flags();        // After full frame evaluation
graph.has_any_dirty();               // Check if cache update needed
graph.mark_fields_dirty_subtree(id); // Propagate field changes downward
```

## 6. Animation System

### 6.1 Skeleton

```rust
pub struct Skeleton {
    pub joints: Vec<Joint>,
    pub global_bind_pose: Vec<Mat4>,  // Pre-computed for GPU
}

pub struct Joint {
    pub name: String,
    pub parent: Option<usize>,          // Index into joints array
    pub local_bind_transform: Mat4,     // T-pose local transform
    pub inverse_bind_matrix: Mat4,      // For skinning
}
```

### 6.2 Animation Clips

```rust
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub tracks: Vec<JointTrack>,           // skeleton joints
    pub object_tracks: Vec<ObjectTrack>,   // any Transform / MorphTarget node
}

pub struct ObjectTrack {
    pub binding: PropertyBinding,          // NodeId + Translation/Rotation/Scale/MorphWeight
    pub keyframes: Vec<ObjectKeyframe>,    // Vec3 / Quat / Scalar
}
```

`AnimationMixer` (driven by `AnimationMixerEngine` each frame) samples object tracks and writes `TransformNode` TRS or `MorphTargetNode.weights`. Bindings use live `NodeId` values, so clips are built against a concrete scene (CAD assembly setup). Joint sampling is unchanged (`sample_all` / GPU skinning). `LoopMode::{Repeat, Once, PingPong}` wraps player time.

### 6.3 GPU Skinning

- Compute shader: `shaders/gpu_skinning.wgsl`
- Runs per-frame before main rendering passes
- Outputs skinned vertex buffer for use by PBR shader
- Supports blend trees: `BlendTree`, `BlendNode`, `BlendClip`, `BlendMask`

## 7. Serialization

### 7.1 Manual Serialize/Deserialize

NodeData uses stable `u16` discriminants for forward compatibility:

```rust
#[derive(Serialize, Deserialize)]
enum NodeDataTag {
    Separator = 0,
    Group = 1,
    Transform = 2,
    Cube = 3,
    // ... 47+ variants
}
```

### 7.2 HandlerNode Serialization

```rust
// Serialize: stores handler_name() as string
// Deserialize: looks up name in NodeTypeRegistry, falls back to DummyHandler
impl Serialize for NodeData {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            NodeData::HandlerNode(h) => {
                // Tag + handler_name string
            }
            // ...
        }
    }
}
```

### 7.3 Custom Node Registration

```rust
use rc3d_scene::node_type_registry::global_registry;

global_registry().lock().unwrap().register(
    100,                                      // Type ID (u16)
    "MyCustomNode",                           // Type name
    MyCustomNode::deserialize_custom,         // Deserializer function
);
```

## 8. Parallel Traversal

### 8.1 Usage

```rust
use rc3d_scene::traversal::traverse_parallel;

// Process each root subtree on a separate rayon thread
traverse_parallel(graph.roots(), graph, |node_id, entry, processor| {
    processor.process(node_id, entry);
});
```

### 8.2 Constraints

- Each root subtree is fully independent (no cross-subtree state)
- Action must implement `Clone + Sync`
- State isolation via Separator boundaries ensures correctness
- Default: disabled (opt-in via `PerformanceSettings::parallel_traversal`)

## 9. Field System (rc3d-fields)

### 9.1 FieldValue Enum

```rust
pub enum FieldValue {
    Bool(bool), Int32(i32), Float(f32), Float64(f64),
    Vec2f(Vec2), Vec3f(Vec3), Vec4f(Vec4), Mat4f(Mat4),
    FloatArray(Vec<f32>), Vec3fArray(Vec<Vec3>), Int32Array(Vec<i32>),
    String(String), Binary(Vec<u8>),
}
```

### 9.2 Field Connections

Fields can be connected for value propagation (Coin3D engine pattern):

```rust
// Intra-node FieldMap (FieldId, same node)
field_map.connect(source_field_id, dest_field_id);
field_map.propagate(field_id);

// Cross-node typed fields (Coin3D SoField::connectFrom)
graph.connect_fields(FieldRef::new(src, 0), FieldRef::new(dst, 0));
graph.field_sources(FieldRef::new(dst, 0)); // reverse lookup
graph.field_targets(FieldRef::new(src, 0));
// World::evaluate_engines calls graph.propagate_fields() after engines
```

### 9.3 Field Descriptors

Each node type describes its fields for editor introspection:

```rust
NodeData::field_descriptors() → Vec<FieldDescriptor>
// FieldDescriptor: { name, field_index, value_type, min, max, enum_values }
```
