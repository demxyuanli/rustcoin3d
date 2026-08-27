# Coin3D / three.js Alignment Evaluation — 2026-07-08

## Current Architecture

```
rc3d-scene (Coin3D-aligned)
├── NodeData: 62 variants (Separator, Transform, Rotation, RotationXYZ, Font, Material, Camera, Light, Shape primitives, BatchedMesh, HemisphereLight, Sprite, LightProbe, …)
├── SceneGraph: SlotMap<NodeId, NodeEntry> — Coin3D SoNode tree
├── Action/Visitor: traversal pattern — Coin3D SoAction pattern
├── animation: Skeleton, AnimationClip (JointTrack + ObjectTrack), AnimationMixer — three.js SkinnedMesh / mixer equivalent

rc3d-render (wgpu deferred PBR)
├── PBR: Cook-Torrance GGX, image-based lighting, prefiltered envmap
├── Shadow: CSM (cascaded shadow maps), omni shadows
├── Post: SSAO, SSR, TAA, bloom, DoF, auto-exposure, FXAA, color grading
├── GPU culling: compute frustum cull → indirect draw
├── Cluster deferred: light culling in view-space clusters

rc3d-mesh (geometry topology)
├── TriangleMesh: vertices + indices + face/edge topology
├── MeshletData: GPU meshlet culling support
├── BVH: ray-pick acceleration
├── tessellate: cube, sphere, cone primitives

rc3d-io (mesh I/O)
├── STL (binary/ASCII read/write)
├── OBJ (read)
└── glTF 2.0 (read)
```

---

## 1. Coin3D Alignment — Current Status

### Full Match ✓

| Coin3D Concept | rustcoin3d | Notes |
|---------------|------------|-------|
| SoSeparator | `Separator` | Scene sub-tree grouping |
| SoGroup | `Group` | Non-separating group |
| SoTransform / SoResetTransform | `Transform` / `ResetTransform` | Matrix stack |
| SoRotation / SoRotationXYZ | `Rotation` / `RotationXYZ` | Axis-angle and X/Y/Z property nodes; `NodeData::local_matrix()` |
| SoClipPlane | `SectionPlane` | Clip plane + caps |
| SoEngineOutput | `EngineConnection` | Engine ports + Kahn toposort (`rotating_cube`) |
| SoGate / SoConcatenate / SoDecomposeVec3f | `GateEngine` / `ConcatenateEngine` / `DecomposeVec3fEngine` | Plus SelectOne, BoolOp, Compose/Decompose Vec2/4, matrix, rotation |
| SoField::connectFrom | `SceneGraph::connect_fields` | `FieldRef` + reverse lookup; `propagate_fields` after engines |
| SoMaterial / SoMaterialBinding | `MaterialNode` / `MaterialBinding` | PBR materials only |
| SoCoordinate3 / SoNormal | `Coordinate3` / `Normal` | Vertex attributes |
| SoTextureCoordinate2 | `TextureCoordinate2` | UV coords |
| SoTexture2Transform | `Texture2Transform` | Texture transform |
| SoPickStyle | `PickStyle` | Picking control |
| SoSwitch | `Switch` | Conditional traversal |
| SoLevelOfDetail | `LodNode` | LOD selection + `range_scale` per-separator distance scale |
| SoMultipleCopy | `MultipleCopyNode` | Instancing |
| SoPerspectiveCamera | `PerspectiveCamera` | Projection + `Engine::apply_standard_quad_views` (Persp/Top/Front/Right pack) |
| SoOrthographicCamera | `OrthographicCamera` | Ortho projection |
| SoDirectionalLight | `DirectionalLight` | Sun/directional |
| SoPointLight | `PointLight` | Omni light |
| SoSpotLight | `SpotLight` | Cone light |
| HemisphereLight (three.js) | `HemisphereLight` | Sky/ground indirect diffuse |
| LightProbe (three.js) | `LightProbe` | L2 SH irradiance (9 RGB coeffs, last probe wins) |
| SoCube / SoSphere / SoCone / SoCylinder | `Cube` / `Sphere` / `Cone` / `Cylinder` | Shape primitives |
| SoTorus | `Torus` | Torus primitive |
| SoIndexedFaceSet | `IndexedFaceSet` | Generic mesh |
| SoIndexedLineSet | `IndexedLineSet` | Wireframe mesh |
| SoText2 / SoText3 | `Text2` / `Text3` | Screen/3D text |
| SoFont | `Font` | Name / size / style for subsequent Text2/Text3; world labels are SDF |
| SoAnnotation | `Annotation` / `AnnotationSet` | 3D annotation (dimensions, leaders) + `AnnotationSet.pmi` semantic records (named node/face/edge, JSON sidecar). No STEP AP242 file reader. |
| SoTransformManip / SoDragger | `TransformManip` / `Dragger` | Scene-graph manipulator; child draggers filter gizmo handles |
| SoShapeHints | `ShapeHints` | Normals/winding hints |
| SoAction / SoCallbackAction | `Action` / `Visitor` | Traversal pattern |
| SoCamera | `StereoCamera` | Dual-eye pass: off-axis frustum, IPD, SideBySide / TopBottom / Anaglyph |
| SoSectionPlane | `SectionPlane` | Cross-section + cap fill + ANSI/ISO hatch (`hatch_enabled`) |
| SoExplodedView | `ExplodedView` | Assembly explode |
| SoDrawStyle | `NodeEntry.display_mode` + `fill_style` / `edge_style` | Presets still map FILLED / LINES / FILLED+LINES. Fill (`Shaded`/`Flat`/`HiddenLine`/`None`) and edges (`None`/`Crease`/`Silhouette`/`Full`/`Perimeter`/`Hard`/`Adjacent`) are orthogonal; Separator isolates like Coin3D. Mixed frames Load line overlays so filled siblings survive; FXAA postpones those lines until after the swapchain blit Clear. CSM shadows follow per-draw fill (`any` lit). Object outline is selection-only (`selection_outline`). HOOPS Isolate/Ghost: `Engine::set_ghost_unselected` fades unselected fill through the transparent pass (empty selection is a no-op). Named catalog: `VisualStyleLibrary` applied with `apply_visual_style`. HiddenLine is Fast HLR: dark fill + crease overlay + dashed occluded edges (`Greater`/`Less` depth, screen-space dash). Vector hardcopy: `Engine::export_hidden_line_svg`. Transform manipulator: `TransformManip` + `Dragger` nodes drive `Engine.gizmo` (`generate_lines` overlay). Omni shadows: cube-array up to 4 lights; transparent casters write CSM/omni depth. LOD: `LodNode.range_scale`. Sub-entity color: `SceneGraph::set_face_tint` / `set_edge_tint` (cube faces = `tri/2`, IFS `face_ids`). Example: `triangle` Wireframe / HardEdges / Perimeter / Adjacent / Silhouette / HiddenLine; `picking` tints cube faces; `selection_set` ghosts unfocused spheres. |

### Partial Match △

| Coin3D Concept | Status | Gap |
|---------------|--------|-----|
| SoBaseColor / SoDiffuseColor | Via `MaterialNode` PBR | No legacy Phong color model |
| SoComplexity | `ShapeHints` | Limited tessellation control (mesh engine removed) |
| SoRayTracing | `RayTracing` node exists | No DXR full pipeline; inline ray queries available (wgpu PR #6291) |
| SoShaderProgram | N/A | No custom shader node |

### Missing ✗

| Coin3D Concept | three.js Equivalent | Priority |
|---------------|---------------------|----------|
| SoFile | `FileNode` + `Engine::render` / `load_scene` resolve | ✓ `resolve_file_nodes` inlines glTF/OBJ/STL/FBX |
| SoRotation / SoRotationXYZ | `RotationNode` / `RotationXYZNode` | ✓ axis-angle + X/Y/Z; Separator flatten via `local_matrix()` |
| SoFont | `FontNode` + SDF world labels | ✓ name/size/`FontStyle`; coverage-to-SDF + `fwidth` AA |
| SoScale | `scale` property | LOW (Transform covers it) |
| SoTranslation | `position` property | LOW (Transform covers it) |
| SoBumpMap / SoNormalMap | Via `MaterialNode` PBR | Covered |

---

## 2. three.js Advanced Rendering Alignment

### Material System

| three.js | rustcoin3d | Gap |
|----------|------------|-----|
| MeshStandardMaterial | `MaterialNode` (PBR metallic-roughness) | ✓ Covered |
| MeshPhysicalMaterial | `MaterialNode` + clearcoat/specular/transmission/sheen/anisotropy/iridescence | ✓ Thin-film `KHR_materials_iridescence` |
| MeshPhongMaterial | `phong.wgsl` | ✓ Legacy path exists |
| MeshToonMaterial | `MaterialNode.toon_steps` | ✓ Cel bands |
| MeshNormalMaterial | `MaterialNode.visualize_normals` | ✓ World-normal color |
| MeshBasicMaterial | `flat_color.wgsl` shader | ✓ Unlit |
| MeshDepthMaterial | `MaterialNode.visualize_depth` | ✓ Camera-distance grayscale |
| PointsMaterial | Via `PointCloud` node | △ Limited point size/attenuation |
| LineBasicMaterial | Via `IndexedLineSet` | △ Limited line width |
| ShaderMaterial | `MaterialNode.custom_wgsl` | ✓ Custom WGSL injection (`material_fs` snippet or full `@vertex`/`@fragment`) |
| RawShaderMaterial | `MaterialNode.custom_wgsl` | ✓ Full WGSL when source contains `@fragment` |

### Post-Processing

| three.js EffectComposer | rustcoin3d | Gap |
|--------------------------|------------|-----|
| BloomPass | ✓ `bloom_prefilter` | |
| SSAOPass | ✓ `ssao` / `ssao_blur` | |
| SSRPass | ✓ `ssr` | |
| TAARenderPass | ✓ `taa_resolve` | |
| BokehPass | ✓ `dof` | |
| AfterimagePass | ✓ `motion_blur` | |
| FXAAShader | ✓ `fxaa_ldr` | |
| LUTPass | ✓ `color_grading` | |
| AdaptiveToneMappingPass | ✓ `auto_exposure` | |
| UnrealBloomPass | ✓ `bloom_prefilter` | |
| SMAAPass | ✗ | **DONE** — 2-pass simplified SMAA (edge-detect + blend) |
| OutlinePass | ✓ `selection_outline` | Screen-space nearest mask + half-res Sobel + separable blur; pixel thickness via `outline_width`. Example: `picking` (left-click selects). |
| FilmPass | ✓ `PostEffectParams.grain` | |
| GlitchPass | ✓ `set_post_stylize` glitch | |
| HalftonePass | ✓ `set_post_stylize` halftone | |

### Geometry / Topology

| three.js | rustcoin3d | Gap |
|----------|------------|-----|
| BufferGeometry | `TriangleMesh` (rc3d-mesh) | ✓ Positions + normals + UVs |
| InstancedMesh | `InstancedMeshNode` + `MultipleCopyNode` | ✓ GPU instancing |
| BatchedMesh | `BatchedMeshNode` | ✓ Packed multi-geometry VB/IB; per-instance ranges share one GPU mesh |
| TransformControls | `TransformManipNode` + `DraggerNode` | ✓ Scene-graph manipulator; `Engine.gizmo` overlay |
| SkinnedMesh | `SkinnedMesh` + `animation.rs` | ✓ GPU skinning via compute |
| MorphTarget | `MorphTarget` node + `ObjectTrack::morph_weight` | ✓ Weights keyframed through `AnimationMixer` |
| AnimationMixer | `AnimationMixer` + `AnimationMixerEngine` | ✓ Object TRS + morph tracks on any `NodeId`; joint tracks unchanged |
| BufferAttribute | Via GPU buffer uploads | ✓ |
| InterleavedBuffer | `Vertex` interleaved layout | ✓ |
| Geometry groups (materialIndex) | `IndexedFaceSetNode.material_groups` | ✓ Shared index buffer + per-group `index_first`/`index_draw_count`; OBJ `usemtl` |
| EdgesGeometry | `edge_detect.wgsl` + `ShadedWithEdges` | ✓ Screen-space + 12 deg crease overlay; no CPU `EdgesGeometry` (duplicates mesh crease filter; CAD edges need B-Rep) |
| WireframeGeometry | `pass_wireframe` | ✓ |
| LineSegments | `IndexedLineSet` | ✓ |
| Sprite | `SpriteNode` | ✓ Camera-facing textured quad; `size_attenuation` |

### Rendering Pipeline

| three.js WebGLRenderer | rustcoin3d | Gap |
|------------------------|------------|-----|
| Forward rendering | N/A | ✗ No forward path (deferred only) |
| Deferred rendering | `pass_solid` (G-buffer) | ✓ Cluster-deferred |
| Shadow maps (PCF) | CSM + omni shadows | ✓ Cascade slices interpolate along the camera projection near/far (not the tightened split far), so small casters stay on-map |
| Shadow maps (PCSS) | N/A | ✗ Soft shadows |
| Environment maps | `Environment` node + IBL | ✓ |
| CubeCamera / local probe | `CubeCameraNode` + cube capture pass | ✓ Six 90-degree faces → equirect IBL (one probe) |
| StereoCamera / StereoEffect | `StereoCameraNode` + dual-eye tiles | ✓ Off-axis IPD; SideBySide / TopBottom / red-cyan anaglyph |
| Light probes | `LightProbeNode` + `GlobalFrameUniforms.sh_l2` | ✓ L2 SH irradiance (three.js `shGetIrradianceAt`) |
| Render-to-texture | Via `reflection_plane` | △ Limited |
| WebGPURenderer | wgpu backend | ✓ |

### Loader / I/O

| three.js | rustcoin3d | Gap |
|----------|------------|-----|
| GLTFLoader | `parse_gltf_file()` | ✓ glTF 2.0 |
| OBJLoader | `parse_obj_file()` | ✓ |
| STLLoader | `parse_stl_file()` | ✓ |
| FBXLoader | `parse_fbx_file()` via `import_file` | ✓ Binary FBX 7.4 (V7400); ASCII / 7.5+ not supported |
| DRACOLoader | `parse_gltf_file()` + `draco.rs` | ✓ `KHR_draco_mesh_compression` |
| KTX2Loader | `parse_gltf_file()` + `ktx2.rs` | ✓ `KHR_texture_basisu` → RGBA8 |
| EXRLoader | `image` crate HDR/EXR | ✓ HDR envmap loading |
| TextureLoader | `RgbaImageData::from_path` | ✓ PNG/JPG via `image` crate |

---

## 3. Geometry Topology Layer — Post B-Rep Removal

### What Was Removed

| Layer | Reason |
|-------|--------|
| `rc3d-shape` (BRepStore, EdgeKey, FaceKey, BRepVertex) | No STEP import → no B-Rep data source |
| `BREP writer` (OCC ASCII format) | Visualization doesn't export B-Rep |
| `brep_binary` (BRepStore serialization) | No B-Rep data to persist |

### What Remains (mesh topology)

| Type | Crate | Description |
|------|-------|-------------|
| `TriangleMesh` | rc3d-mesh | Vertex positions + indices + face/edge adjacency |
| `EdgeKey` / `FaceId` / `EdgeId` | rc3d-mesh | Mesh-local edge/face indexing |
| `BVH` / `BvhTriangle` | rc3d-mesh | Ray-pick acceleration |
| `MeshletData` | rc3d-mesh | GPU meshlet culling |
| `Vertex` | rc3d-render | Interleaved vertex format (position, normal, UV, tangent) |

### Gap: No Higher-Level Topology

The mesh topology is flat — no concept of:
- Connected components
- Boundary edges
- Face groups / material groups
- Adjacency queries beyond ray-pick BVH
- Edge highlighting / selection (done via screen-space `edge_detect.wgsl`)

**Assessment**: For a visualization engine, flat mesh topology is adequate. three.js `EdgesGeometry` is a mesh dihedral-angle filter, not CAD B-Rep edges. Visualization already has screen-space `edge_detect.wgsl`, `ShadedWithEdges` crease overlay, and `pass_wireframe` for all triangle edges. True CAD feature edges require B-Rep/STEP, which this engine does not import.

---

## 4. Priority Matrix

### HIGH — Direct visualization gaps

| # | Feature | Effort | Impact | Status |
|---|---------|--------|--------|--------|
| 1 | File node → glTF loader wiring | Low | High — models loadable via scene graph | **DONE** |
| 2 | MeshPhysicalMaterial clearcoat + transmission | Medium | High — automotive & glass PBR | **DONE** — clearcoat + IBL refraction (`HAS_TRANSMISSION`) |
| 3 | Forward rendering path | High | High — transparent sorting | **DEFERRED** — see analysis |
| 4 | SMAA anti-aliasing | Low | Medium — TAA complement | **DONE** |

### MEDIUM — Feature completeness

| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 5 | HDR envmap loading (EXR/HDR) | Low | Medium — IBL requires pre-processed envmaps | **DONE** |
| 6 | Per-face material groups (multi-material meshes) | Medium | Medium | **DONE** — `IndexedFaceSetNode.material_groups` + OBJ `usemtl` |
| 7 | MeshPhysicalMaterial sheen extension | Medium | Medium — fabric/furniture surfaces | **DONE** |
| 8-17 | All remaining MEDIUM+LOW | — | — | **DEFERRED** — outside CAD visualization scope |

---

## 5. Iteration Plan (Ralph Loop)

Ralph loop: iterate through remaining MEDIUM+LOW items, one per iteration.

### Iteration 1 — HDR envmap loading (EXR)

**Effort**: Low (~50 lines)
**Impact**: Medium — removes pre-processing step for IBL cubemaps
**Done condition**: `parse_exr_bytes()` → HDR texture; glTF EXR extension loaded

### Iteration 2 — Sheen extension

**Effort**: Medium (~100 lines shader + fields)
**Impact**: Medium — fabric/furniture PBR
**Done condition**: `MaterialNode.sheen_color`, `MaterialNode.sheen_roughness`; `pbr.wgsl` HAS_SHEEN path

### Iteration 3 — HDR envmap loading (EXR)

Skip to remaining items after sheen...

Remaining order: SoScale / SoTranslation **LOW** (Transform covers) → CPU EdgesGeometry **WONTFIX** (duplicates crease overlay; CAD edges need B-Rep)

1. **Immediate**: Wire `FileNode` to glTF/OBJ/STL loaders — enables scene-graph-driven model loading
2. **This sprint**: MeshPhysicalMaterial clearcoat + transmission (highest visual PBR gap per audit)
3. **This sprint**: SMAA anti-aliasing (1-2 day, three.js `SMAAPass` reference, complements TAA)
4. **Next sprint**: HDR envmap loading (EXR) for improved IBL quality
5. **Architecture decision**: Forward rendering path for transparent objects (or continue with deferred-only)
6. **Topology**: Flat mesh topology sufficient — three.js `BufferGeometry` same model. Screen-space / crease overlay / wireframe for visualization. CAD feature edges stay with B-Rep (not restored from triangle meshes).
