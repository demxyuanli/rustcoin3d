# Rendering Pipeline Design

## 1. Architecture Overview

```
Scene Graph → RenderAction → DrawCall[] ──┐
                                          ├──> Sorting → GPU Buffers
       Camera → Frustum → BVH Culling ────┘
                                          │
   GPU Compute Culling (parallel) ────────┘
                                          │
          Shadow Maps (CSM 4 Cascades) ───┘
                                          │
     HZB Build (Depth Pyramid) ───────────┘
                                          │
  Solid Pass → Decal → Volume → Wireframe → Edge ─┘
                                          │
        Post FX (SSAO→SSR→DOF→Bloom→TAA) ─┘
                                          │
                    HUD → Swapchain ──────┘
```

## 2. Frame Pipeline (execute_passes order)

### Stage 1: Preparation
1. **Surface acquisition** — Get swapchain texture or offscreen target
2. **Skinning compute** — GPU compute dispatch for skeletal animation
3. **Background pass** — Gradient / image / cubemap background

### Stage 2: Culling
4. **GPU compute culling dispatch** — Compute shader frustum cull (runs alongside CPU)
5. **CSM shadow depth** — 4 cascades, each with CPU frustum cull, depth-only draw
6. **HZB pre-pass + meshlet culling** — For large meshes:
   - First meshlet cull (coarse, no HZB)
   - Depth pre-pass
   - HZB pyramid build (downsample max/min)
   - Second meshlet cull (HZB-accelerated occlusion)

### Stage 3: Lighting
7. **Cluster light culling** — Assign point/spot lights to 3D cluster grid (16×8×24)

### Stage 4: Main Passes
8. **Solid pass** — PBR shading + outline + section caps
9. **Decal pass** — Projected texture decals
10. **Volume pass** — Raymarch volume rendering
11. **Point cloud pass** — Large-scale point rendering
12. **Wireframe pass** — Wireframe display mode only
13. **Edge overlay** — Topological edge lines with depth test
14. **Selection pass** — Selection fill + screen-space outline
15. **Transparent pass** — Alpha-blended, distance-sorted

### Stage 5: Post-Processing (HDR path)
16. **SSAO** — Screen space ambient occlusion
17. **SSR** — Screen space reflections (HIZ-accelerated)
18. **Volumetric fog** — Raymarch atmospheric scattering
19. **Motion blur** — Velocity-buffer post blur
20. **DOF** — Circle-based depth of field
21. **Color grading** — LUT-based color transformation
22. **Bloom** — Prefilter → HDR composite
23. **TAA resolve** — Temporal anti-aliasing (YCoCg, AABB clip)
24. **Tonemap + FXAA** — HDR→LDR with optional FXAA
25. **Blit to swapchain** — Final output

### Stage 6: Overlays
26. **Viewport grid** — Ground plane reference grid
27. **Viewport borders** — Multi-viewport split dividers
28. **Markup overlay** — Measurement/markup lines + 3D annotation projection:
    - 2D `MarkupNode` elements (screen-space lines, rects, circles) collected via `collect_markup_lines`
    - 3D `AnnotationElement` (Dimension/Leader/Datum) collected via `collect_effect_nodes` → `ProjectedAnnotation`
    - Dimension: 12 3D points computed on annotation plane, projected to screen. Filled arrowheads (4-line V-shape).
    - Leader: 2 3D points (anchor + world-axis offset label position), fixed world axes prevent camera drift.
    - Datum: 5 3D points (center + 4 diagonal cross arms), fixed world XZ-plane diagonals.
    - All projected via `screen_space_ortho` for correct pixel-to-NDC mapping.
29. **HUD text** — glyphon text rendering (FPS stats + scene text)
30. **App callbacks** — Custom post-swapchain overlays

## 3. Culling Pipeline (Dual-Path)

### 3.1 CPU BVH Culling

```
BVH incremental update (dirty AABBs only)
    ↓
Frustum-AABB test per BVH node
    ↓
Visible draw call indices collected
    ↓
Static frame fast path: reuse cached indices if scene+camera unchanged ≥2 frames
```

### 3.2 GPU Compute Culling

```
Frame N: Upload transforms → transform_buffer (128B/object)
         Dispatch object_cull compute shader
         Copy visible indices → staging buffer
Frame N+1: Read staging → replace CPU visibility
         Render with GPU-derived indices
```

- 1-frame latency (GPU results used next frame)
- Activation threshold: `gpu_culling_threshold` objects (default 4096)
- Falls back to CPU culling when GPU results stale
- Shader: `shaders/object_cull.wgsl` — 1 thread/object, 6-plane frustum test

### 3.3 Meshlet Culling (Hierarchical)

For meshes >500K triangles:
```
Level 0 (coarse) → frustum cull
    ↓
Depth pre-pass (visible clusters only)
    ↓
HZB pyramid build (downsample_max)
    ↓
Level 1 (fine) → frustum + HZB occlusion cull
    ↓
Indirect draw with visible cluster counts
```

Shader: `shaders/cluster_tree_cull.wgsl`

## 4. Lighting System

### 4.1 Light Types

| Type | Shadow | Culling | Parameters |
|------|--------|---------|------------|
| DirectionalLight | CSM (4 cascades) | Frustum | direction, color, intensity |
| PointLight | Omni (cubemap) | Cluster | position, color, intensity |
| SpotLight | - | Cluster | position, direction, cutoff, falloff |
| AreaLight | - | - | position, direction, size, shape |

### 4.2 Cluster-Based Light Culling

- Grid: 16×8×24 (X×Y×Z) in clip space
- Each cell stores: light indices, light count
- Compute shader assigns lights to grid cells
- Fragment shader reads per-cluster light list

### 4.3 LightSetTable (Deduplication)

Each unique combination of light directions/colors/types/positions is stored once:
- `DrawCall.light_set_id: u32` (4 bytes)
- Lookup: `light_sets.get(id) → PackedLights`
- Saving: ~1280B → 4B per draw call (~1.2GB saved @ 1M objects)

### 4.4 Image-Based Lighting (IBL)

- HDR environment map → irradiance convolution + prefiltered envmap
- BRDF LUT (split-sum approximation)
- Precomputed at startup or on environment change

## 5. PBR Shading Model

### 5.1 BRDF Components

- **Diffuse**: Lambertian with Disney diffuse for roughness
- **Specular**: GGX normal distribution, Smith geometry attenuation, Schlick Fresnel
- **IBL**: Split-sum approximation with prefiltered envmap + BRDF LUT

### 5.2 Per-Instance Data (96 bytes)

```hlsl
struct InstanceData {
    model: mat4x4<f32>,     // 64 bytes
    mvp: mat4x4<f32>,       // 64 bytes
    diffuse_color: vec4<f32>,  // 16 bytes
    metallic_roughness: vec4<f32>,
    emissive: vec4<f32>,
    morph_weights: array<f32, 8>,
}
```

### 5.3 Material Pipeline

- `MaterialNode` in scene → `CachedDrawMetadata` (72B, dirty-updated)
- `MaterialUniform` (48B, uploaded with draw data)
- Material binding groups cached by texture path hash
- Alpha modes: Opaque, Mask (threshold discard), Blend

## 6. Shadow Mapping

### 6.1 Cascaded Shadow Maps (CSM)

- 4 cascades with log-uniform split scheme
- Lambda parameter controls split distribution
- Each cascade: orthographic projection from light perspective
- 8% blend zone between cascades
- PCF filtering with configurable kernel radius
- Shadow factor computed in PBR fragment shader

### 6.2 Omni-Directional Shadows

- Point light → cubemap rendering (6 faces)
- Shadow pool texture: 512×512 per face
- Used by cluster light system

## 7. Post-Processing Effects

### 7.1 Temporal Anti-Aliasing (TAA)

- YCoCg color space for better quality
- AABB clipping (3×3 neighborhood) to prevent ghosting
- Motion vector buffer for reprojection
- Blend factor: 0.05–0.1 (configurable)

### 7.2 Screen Space Reflections (SSR)

- HIZ-accelerated ray marching
- Parameters: step count, max distance, thickness, stride, roughness cutoff
- Output: HDR reflection texture

### 7.3 Screen Space Ambient Occlusion (SSAO)

- Hemisphere sampling with 4×4 noise rotation texture
- Parameters: radius (0.8), bias (0.02), power (1.0)
- Separable blur pass after sampling

### 7.4 Depth of Field (DOF)

- Circle-of-confusion based blur
- Parameters: focus distance, aperture, max blur radius

### 7.5 Motion Blur

- Velocity+depth buffer post-process
- Per-pixel velocity from camera + object motion

### 7.6 Volumetric Fog

- Raymarch through volume in HDR space
- Atmospheric scattering model

### 7.7 Color Grading

- LUT-based (lookup table) color transformation
- Applied before tonemapping

### 7.8 Bloom

- Prefilter compute shader → half-resolution bloom target
- Multiple mip levels for wide bloom
- Composite with HDR before tonemap

### 7.9 Auto Exposure

- HDR luminance-based eye adaptation
- Histogram approach with adaptation speed

## 8. Sorting Strategy

| Pass | Sort Key | Purpose |
|------|----------|---------|
| Solid | light_hash → material_key | Group same lights/materials |
| Edge | display_mode → overlay_color → mesh_handle | Reduce state changes |
| Selection | display_mode → mesh_handle | Batching efficiency |
| Transparent | camera_distance (far→near) | Correct alpha blending |

## 9. Draw Call Batching

### 9.1 Material Batching

```
draw_opaque_triangle_batches():
  for each material group:
    set_bind_group(material)
    for each mesh in group:
      skip set_vertex_buffer if same as last
      draw_indexed(instances)
```

### 9.2 Instance Batching

Multiple instances of the same mesh share:
- Vertex buffer (bound once)
- Material bind group (bound once)
- Different per-instance data (model matrix, color) via storage buffer

### 9.3 Indirect Draw (GPU-Driven)

```
GPU compute writes:
  → instance_indices_buffer (visible instance IDs)
  → indirect_args_buffer (DrawIndirectArgs per mesh)

CPU render pass:
  draw_indexed_indirect(indirect_args_buffer)
```

## 10. Performance Modes

### 10.1 Adaptive Quality

5 levels controlled by `AdaptiveQualityMode`:
- Level 0 (idle): Full quality — all effects enabled
- Level 1: Reduced shadow cascade count, larger bloom step
- Level 2: Disable DOF, SSR step reduction
- Level 3: Disable SSR, reduce SSAO samples
- Level 4 (heavy load): Disable most post-effects, force FXAA

### 10.2 Triggers

- Interaction active (camera orbiting/panning/zooming) → immediate reduction
- Triangle count > threshold → gradual reduction
- EMA (exponential moving average) with 30-frame cooldown for prevention of rapid toggling

## 11. Display Modes

| Mode | Description | Implementation |
|------|-------------|----------------|
| Shaded | Full PBR shading | Default solid pass |
| ShadedWithEdges | PBR + edge overlay | Solid pass + edge pass (depth-tested) |
| Wireframe | Wireframe only | Separate wireframe pass |
| HiddenLine | Hidden line removal | Wireframe with depth pre-pass |
| Flat | Flat (no lighting) | Flat color vertex shader |
