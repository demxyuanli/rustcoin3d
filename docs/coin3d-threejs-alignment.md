# Coin3D / three.js Alignment Evaluation — 2026-07-08

## Current Architecture

```
rc3d-scene (Coin3D-aligned)
├── NodeData: 47 variants (Separator, Transform, Material, Camera, Light, Shape primitives, Annotation, …)
├── SceneGraph: SlotMap<NodeId, NodeEntry> — Coin3D SoNode tree
├── Action/Visitor: traversal pattern — Coin3D SoAction pattern
├── animation: Skeleton, AnimationClip, VertexSkinData — three.js SkinnedMesh equivalent

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
| SoMaterial / SoMaterialBinding | `MaterialNode` / `MaterialBinding` | PBR materials only |
| SoCoordinate3 / SoNormal | `Coordinate3` / `Normal` | Vertex attributes |
| SoTextureCoordinate2 | `TextureCoordinate2` | UV coords |
| SoTexture2Transform | `Texture2Transform` | Texture transform |
| SoPickStyle | `PickStyle` | Picking control |
| SoSwitch | `Switch` | Conditional traversal |
| SoLevelOfDetail | `LodNode` | LOD selection |
| SoMultipleCopy | `MultipleCopyNode` | Instancing |
| SoPerspectiveCamera | `PerspectiveCamera` | Projection |
| SoOrthographicCamera | `OrthographicCamera` | Ortho projection |
| SoDirectionalLight | `DirectionalLight` | Sun/directional |
| SoPointLight | `PointLight` | Omni light |
| SoSpotLight | `SpotLight` | Cone light |
| SoCube / SoSphere / SoCone / SoCylinder | `Cube` / `Sphere` / `Cone` / `Cylinder` | Shape primitives |
| SoTorus | `Torus` | Torus primitive |
| SoIndexedFaceSet | `IndexedFaceSet` | Generic mesh |
| SoIndexedLineSet | `IndexedLineSet` | Wireframe mesh |
| SoText2 / SoText3 | `Text2` / `Text3` | Screen/3D text |
| SoAnnotation | `Annotation` / `AnnotationSet` | 3D annotation (dimensions, leaders) |
| SoEnvironment | `Environment` | Sky/IBL |
| SoShapeHints | `ShapeHints` | Normals/winding hints |
| SoAction / SoCallbackAction | `Action` / `Visitor` | Traversal pattern |
| SoCamera | `StereoCamera` | Stereo rendering |
| SoSectionPlane | `SectionPlane` | Cross-section |
| SoExplodedView | `ExplodedView` | Assembly explode |

### Partial Match △

| Coin3D Concept | Status | Gap |
|---------------|--------|-----|
| SoBaseColor / SoDiffuseColor | Via `MaterialNode` PBR | No legacy Phong color model |
| SoComplexity | `ShapeHints` | Limited tessellation control (mesh engine removed) |
| SoDrawStyle | `DisplayMode` (Flat/Wireframe/Points) | Missing FILLED+WIREFRAME overlay |
| SoRayTracing | `RayTracing` node exists | No DXR full pipeline; inline ray queries available (wgpu PR #6291) |
| SoShaderProgram | N/A | No custom shader node |

### Missing ✗

| Coin3D Concept | three.js Equivalent | Priority |
|---------------|---------------------|----------|
| SoFile | GLTFLoader / `FileNode` exists but no loader wired | HIGH |
| SoRotation / SoRotationXYZ | `Euler` / `Quaternion` rotation nodes | MEDIUM |
| SoScale | `scale` property | LOW (Transform covers it) |
| SoTranslation | `position` property | LOW (Transform covers it) |
| SoFont | Font3D / `Text3` exists but no SDF font rendering | MEDIUM |
| SoClipPlane | `Material.clippingPlanes` | LOW |
| SoBumpMap / SoNormalMap | Via `MaterialNode` PBR | Covered |

---

## 2. three.js Advanced Rendering Alignment

### Material System

| three.js | rustcoin3d | Gap |
|----------|------------|-----|
| MeshStandardMaterial | `MaterialNode` (PBR metallic-roughness) | ✓ Covered |
| MeshPhysicalMaterial | N/A | ✗ Clearcoat (HIGH), sheen (MEDIUM), transmission (MEDIUM), anisotropy (LOW), iridescence (LOW) |
| MeshPhongMaterial | N/A | ✗ Legacy Phong (low priority for PBR engine) |
| MeshToonMaterial | N/A | ✗ Cel shading |
| MeshNormalMaterial | N/A | ✗ Debug normals |
| MeshBasicMaterial | `flat_color.wgsl` shader | ✓ Unlit |
| MeshDepthMaterial | N/A | ✗ Shadow/depth-only material |
| PointsMaterial | Via `PointCloud` node | △ Limited point size/attenuation |
| LineBasicMaterial | Via `IndexedLineSet` | △ Limited line width |
| ShaderMaterial | N/A | ✗ Custom shader injection |
| RawShaderMaterial | N/A | ✗ Raw WGSL injection |

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
| FilmPass | ✗ | Film grain |
| GlitchPass | ✗ | Glitch effect |
| HalftonePass | ✗ | Halftone |

### Geometry / Topology

| three.js | rustcoin3d | Gap |
|----------|------------|-----|
| BufferGeometry | `TriangleMesh` (rc3d-mesh) | ✓ Positions + normals + UVs |
| InstancedMesh | `MultipleCopyNode` | ✓ Basic instancing |
| SkinnedMesh | `SkinnedMesh` + `animation.rs` | ✓ GPU skinning via compute |
| MorphTarget | `MorphTarget` node | △ Vertex morph (blend shapes) |
| BufferAttribute | Via GPU buffer uploads | ✓ |
| InterleavedBuffer | `Vertex` interleaved layout | ✓ |
| Geometry groups (materialIndex) | `FaceMaterialGroup` (removed with emit_plan) | ✗ Per-face materials |
| EdgesGeometry | `edge_detect.wgsl` | ✓ Screen-space edges; add CPU-side extraction for CAD wireframe export if needed |
| WireframeGeometry | `pass_wireframe` | ✓ |
| LineSegments | `IndexedLineSet` | ✓ |

### Rendering Pipeline

| three.js WebGLRenderer | rustcoin3d | Gap |
|------------------------|------------|-----|
| Forward rendering | N/A | ✗ No forward path (deferred only) |
| Deferred rendering | `pass_solid` (G-buffer) | ✓ Cluster-deferred |
| Shadow maps (PCF) | CSM + omni shadows | ✓ |
| Shadow maps (PCSS) | N/A | ✗ Soft shadows |
| Environment maps | `Environment` node + IBL | ✓ |
| Light probes | N/A | ✗ |
| Render-to-texture | Via `reflection_plane` | △ Limited |
| WebGPURenderer | wgpu backend | ✓ |

### Loader / I/O

| three.js | rustcoin3d | Gap |
|----------|------------|-----|
| GLTFLoader | `parse_gltf_file()` | ✓ glTF 2.0 |
| OBJLoader | `parse_obj_file()` | ✓ |
| STLLoader | `parse_stl_file()` | ✓ |
| FBXLoader | N/A | ✗ |
| DRACOLoader | N/A | ✗ |
| KTX2Loader | N/A | ✗ |
| EXRLoader | N/A | ✗ HDR envmap loading |
| TextureLoader | N/A | ✗ PNG/JPG loading |

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

**Assessment**: For a visualization engine, flat mesh topology is adequate. three.js operates the same way (BufferGeometry has no adjacency — edges are extracted from index buffer via `EdgesGeometry` on CPU side, or `WireframeGeometry` for all edges). The current screen-space edge detection (`edge_detect.wgsl`) is a valid visualization approach. For CAD-precision wireframe output, CPU-side index-buffer-based edge extraction (like three.js `EdgesGeometry`) should be available as an alternative.

---

## 4. Priority Matrix

### HIGH — Direct visualization gaps

| # | Feature | Effort | Impact | Status |
|---|---------|--------|--------|--------|
| 1 | File node → glTF loader wiring | Low | High — models loadable via scene graph | **DONE** |
| 2 | MeshPhysicalMaterial clearcoat + transmission | Medium | High — automotive & glass PBR | **DONE** — clearcoat shader + fields, transmission deferred |
| 3 | Forward rendering path | High | High — transparent sorting | **DEFERRED** — see analysis |
| 4 | SMAA anti-aliasing | Low | Medium — TAA complement | **DONE** |

### MEDIUM — Feature completeness

| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 5 | HDR envmap loading (EXR/HDR) | Low | Medium — IBL requires pre-processed envmaps | **DONE** |
| 6 | Per-face material groups (multi-material meshes) | Medium | Medium | **DEFERRED** — single-material-per-mesh covers 90% of CAD viz |
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

Remaining order: sheen → per-face materials → SDF fonts → anisotropy → CPU edges → cel shading → film grain → Draco → KTX2

1. **Immediate**: Wire `FileNode` to glTF/OBJ/STL loaders — enables scene-graph-driven model loading
2. **This sprint**: MeshPhysicalMaterial clearcoat + transmission (highest visual PBR gap per audit)
3. **This sprint**: SMAA anti-aliasing (1-2 day, three.js `SMAAPass` reference, complements TAA)
4. **Next sprint**: HDR envmap loading (EXR) for improved IBL quality
5. **Architecture decision**: Forward rendering path for transparent objects (or continue with deferred-only)
6. **Topology**: Flat mesh topology sufficient — three.js `BufferGeometry` same model. Screen-space edges for visualization; CPU-side index extraction added only if CAD wireframe export needed.
