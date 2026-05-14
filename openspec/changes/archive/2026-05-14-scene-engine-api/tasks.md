## 1. Workspace Setup

- [x] 1.1 Create `crates/rc3d-scene-api/` crate with Cargo.toml (depends on rc3d-scene, rc3d-core, rc3d-mesh)
- [x] 1.2 Create `crates/rc3d-effects/` crate with Cargo.toml (depends on rc3d-render, rc3d-core)
- [x] 1.3 Add both crates to workspace `Cargo.toml` members
- [x] 1.4 Verify `cargo check --workspace` passes with new crates

## 2. Shape Trait + Built-in Shapes

- [x] 2.1 Define `Shape` trait (`compile()`, `aabb()`, `as_any()`) in rc3d-scene-api
- [x] 2.2 Implement `Cube` shape with builder methods (`width()`, `height()`, `depth()`, `at()`, `scale()`, `rotate()`, `material()`)
- [x] 2.3 Implement `Sphere` shape with builder methods
- [x] 2.4 Implement `Cone` and `Cylinder` shapes with builder methods
- [x] 2.5 Implement `Mesh` shape with `from_raw(positions, indices)` constructor (IndexedFaceSet wrapper)
- [x] 2.6 Implement `LineSet` shape with `from_lines(positions, indices)` constructor (IndexedLineSet wrapper)
- [x] 2.7 Add unit tests for each shape's `compile()` output

## 3. Material + Light + Camera Types

- [x] 3.1 Implement `Material::pbr()` builder with all MaterialNode field mapping
- [x] 3.2 Implement `Material::diffuse()` convenience constructor
- [x] 3.3 Implement `DirectionalLight` and `PointLight` types
- [x] 3.4 Implement `PerspectiveCamera` with `look_at()` constructor
- [x] 3.5 Implement `OrthographicCamera`

## 4. Scene Container

- [x] 4.1 Implement `Scene` struct with internal `SceneGraph` + root Separator
- [x] 4.2 Implement `scene.add(shape)` — auto-wrap in Separator → [Transform] → [Material] → ShapeNode
- [x] 4.3 Implement `scene.set_camera()` and `scene.add_light()`
- [x] 4.4 Implement `scene.build()` — finalize SceneGraph for consumption
- [x] 4.5 Implement `NodeHandle` as opaque reference to scene graph nodes

## 5. Group + Query

- [x] 5.1 Implement `Group::new(name)` with `add(child)` builder — internal Separator mapping
- [x] 5.2 Implement nested Group support
- [x] 5.3 Implement `Query` struct with `any::<T>()` type-based search
- [x] 5.4 Implement `Query::collect_all::<T>()` and `Query::bounds_of(handle)`
- [x] 5.5 Add tests for query correctness on multi-level scenes

## 6. Geometry Compile Layer

- [x] 6.1 Implement `Geometry::compile(shape)` — Shape → TriangleMesh via rc3d-mesh tessellators
- [x] 6.2 Implement `Geometry::validate()` — degenerate triangle / non-manifold detection
- [x] 6.3 Implement `Geometry::aabb()`, `face_count()`, `boundary_edges()` query methods
- [x] 6.4 Implement `Geometry::triangle_buffers()` and `Geometry::phong_buffers()` GPU buffer extraction
- [x] 6.5 Add tests for each shape's compiled geometry properties

## 7. RenderConfig + EffectGraph

- [x] 7.1 Define `Shadow::CSM` config struct (cascade_count, resolution, soft)
- [x] 7.2 Define `PostEffect` enum (SSAO, SSR, Bloom, TAA, Tonemap, DOF, MotionBlur, VolumetricFog)
- [x] 7.3 Implement `RenderConfig` builder with `enable()`, `display_mode()`, `build()`
- [x] 7.4 Implement `RenderConfig::default()` and `RenderConfig::high_quality()` presets
- [x] 7.5 Implement `EffectGraph` — DAG of ordered passes with dependency resolution
- [x] 7.6 Implement `EffectGraph::apply_to(pass_context)` — map effects to PassContext flags

## 8. App Bridge

- [x] 8.1 Implement `App::with_effects(render_config)` in rc3d-app
- [x] 8.2 Wire EffectGraph → Renderer pass context on initialization
- [x] 8.3 Implement Scene → SceneGraph compilation inside `App::new(scene)`
- [x] 8.4 Verify backward compatibility: existing `App::new(graph)` API unchanged

## 9. Integration Tests + Examples

- [x] 9.1 Write integration test: cube scene with directional light + PBR material → build → verify SceneGraph structure
- [x] 9.2 Write integration test: scene with groups and queries
- [x] 9.3 Write integration test: RenderConfig with CSM + SSAO → build → verify EffectGraph
- [x] 9.4 Implement `examples/hello_scene.rs` — minimal scene with new API (Cube + light + camera)
- [x] 9.5 Implement `examples/pbr_scene.rs` — PBR material showcase with new API
- [x] 9.6 Implement `examples/render_effects.rs` — effect toggling with new API
