# Shader Catalog

Complete reference of 45 WGSL shaders in `crates/rc3d-render/src/shaders/`.

## 1. PBR & Shading

| Shader | Size | Description |
|--------|------|-------------|
| `pbr.wgsl` | 14 KB | **Main PBR shader**. GGX/Smith BRDF, IBL split-sum, CSM shadows (4 cascades, 8% blend), morph targets (up to 8 weights), alpha modes (opaque/mask/blend), section plane clipping (up to 6 planes), normal mapping with bitangent flip |
| `phong.wgsl` | - | Phong shading model (fallback/performance mode) |
| `flat_color.wgsl` | - | Flat/unlit vertex color pass |
| `triangle.wgsl` | - | Simple triangle rendering |

### PBR Shader Data Structures

```rust
struct SceneUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    num_lights: u32,
    lights: array<LightParams, 16>,
    ibl_params: IblParams,
    csm_matrices: array<mat4x4<f32>, 4>,
    csm_splits: array<f32, 4>,
    clip_planes: array<vec4<f32>, 6>,
    num_clip_planes: u32,
}

struct MaterialUniform {  // 48 bytes
    base_color: vec4<f32>,
    emissive_color: vec4<f32>,
    metallic_roughness_anisotropic: vec4<f32>,
}
```

## 2. Shadows

| Shader | Description |
|--------|-------------|
| `shadow_depth.wgsl` | CSM depth rendering — writes linear depth to 2D texture array (4 layers) |
| `shadow_omni.wgsl` | Omni-directional shadow for point lights — renders to cube map faces |

## 3. GPU Culling

| Shader | Description |
|--------|-------------|
| `object_cull.wgsl` | Per-object frustum cull. 1 thread/object. Tests AABB against 6 frustum planes. Atomic append visible instance IDs + instance counts per mesh. |
| `cluster_cull.wgsl` | Cluster-based partial cull (transition shader) |
| `cluster_tree_cull.wgsl` | Hierarchical meshlet culling. Top-down: frustum test → HZB occlusion test per level. Indirect dispatch for visible clusters only. |
| `cluster_light_cull.wgsl` | Assigns point/spot lights to 3D cluster grid (16×8×24). Per-cluster light count + light index list. |
| `compact_finalize.wgsl` | Finalizes compacted instance index buffer after culling |
| `cluster_compact.wgsl` | Compacts cluster light index lists |

## 4. HZB (Hierarchical Z-Buffer)

| Shader | Description |
|--------|-------------|
| `hzb_depth_to_mip0.wgsl` | Converts depth texture to HZB mip 0 format |
| `hzb_downsample_max.wgsl` | HZB pyramid downsample (max reduction) — for occlusion queries |
| `hzb_downsample_min.wgsl` | HZB pyramid downsample (min reduction) — for near-plane queries |

## 5. Post-Processing

| Shader | Description |
|--------|-------------|
| `post_tonemap_fxaa.wgsl` | **Composite pass**: HDR tonemapping + FXAA + bloom + SSAO composition. Samples HDR input, bloom texture, SSAO texture. Outputs LDR for swapchain/offscreen. |
| `taa_resolve.wgsl` | Temporal anti-aliasing. YCoCg color space, AABB clip (3×3 neighborhood), motion vector reprojection. Blend factor 0.05–0.1. |
| `ssao.wgsl` | Screen space ambient occlusion. Hemisphere sampling, 4×4 noise rotation, radius 0.8, bias 0.02, power 1.0. |
| `ssao_blur.wgsl` | Separable blur for SSAO output (horizontal + vertical passes) |
| `ssr.wgsl` | Screen space reflections. HIZ-accelerated ray marching. Parameters: steps, max distance, thickness, stride, roughness cutoff. |
| `motion_blur.wgsl` | Velocity+depth based post-process motion blur |
| `dof.wgsl` | Circle-of-confusion depth of field. Parameters: focus distance, aperture, max radius. |
| `color_grading.wgsl` | LUT-based color grading (3D lookup texture) |
| `bloom_prefilter.wgsl` | HDR bloom prefilter compute. Downsamples to half-res with brightness threshold. |
| `fxaa_ldr.wgsl` | LDR FXAA (used when HDR pipeline disabled) |
| `auto_exposure.wgsl` | HDR luminance-based eye adaptation |
| `mip_downsample.wgsl` | General mip level downsample compute |

## 6. IBL (Image-Based Lighting)

| Shader | Description |
|--------|-------------|
| `irradiance_convolution.wgsl` | Convolves HDR environment map to diffuse irradiance cubemap |
| `prefilter_envmap.wgsl` | Pre-filters environment map at multiple roughness levels for specular IBL |
| `brdf_lut.wgsl` | Pre-computes BRDF integration LUT (split-sum approximation) |
| `background_cubemap.wgsl` | Renders environment cubemap as background |

## 7. Effects

| Shader | Description |
|--------|-------------|
| `decal_project.wgsl` | Screen-space projected texture decal. Uses depth buffer for projection, supports angle cutoff and fade. |
| `volume_raymarch.wgsl` | 3D volume raymarching. Samples 3D texture with transfer function. |
| `point_cloud.wgsl` | Large-scale point cloud vertex shader. Reads from octree tile buffer. |
| `reflection_plane.wgsl` | Planar reflection rendering. Renders mirrored geometry to reflection texture. |
| `section_cap.wgsl` | Section plane cap rendering. Renders colored triangle caps where geometry is clipped. |

## 8. Selection & Outlines

| Shader | Description |
|--------|-------------|
| `selection_outline_depth.wgsl` | Non-selected occluder depth for hidden/visible split |
| `selection_outline_mask.wgsl` | Selected-object mask (R=inside, G=visible) |
| `selection_outline_edge.wgsl` | Half-res Sobel on the downsampled mask |
| `selection_outline_blur.wgsl` | Separable Gaussian blur for outline thickness |
| `selection_outline_composite.wgsl` | Additive overlay; `mask.r` keeps the stroke outside the object |

## 9. Skinning & Geometry

| Shader | Description |
|--------|-------------|
| `gpu_skinning.wgsl` | Compute shader for skeletal animation. Reads bind pose + joint transforms, writes skinned vertices. Supports up to 128 joints. |
| `background.wgsl` | Simple background fill (gradient or solid color) |
| `blit_tex.wgsl` | Texture blit pass (HDR→LDR for non-post-processed path) |

## 10. Shader Variant System

### 10.1 Preprocessor Macros

The PBR shader uses WGSL `#define` preprocessing for variant selection:

```wgsl
// Variant defines (set at pipeline creation):
// #define HAS_NORMALS       — normal mapping enabled
// #define HAS_TANGENTS      — tangent space normals
// #define HAS_TEXCOORDS     — UV coordinates present
// #define HAS_COLORS        — vertex colors present
// #define ALPHA_MODE_MASK   — alpha-test discard
// #define ALPHA_MODE_BLEND  — alpha blending
// #define USE_IBL           — image-based lighting
// #define USE_CSM           — cascaded shadow maps
// #define MORPH_TARGETS     — blend shape support
```

### 10.2 Pipeline Caching

- `PipelineCacheManager` — caches compiled pipelines by key
- `ShaderVariantCache` — caches preprocessed shader sources
- Each permutation of defines creates a unique pipeline
- Hot reload: `ShaderHotReload` watches source files in development

### 10.3 Depth Modes

```
DepthModePipelines (8 variants):
  ├── Forward Z  → 4 PBR variants (HDR/LDR, standard/edge)
  └── Reverse Z  → 4 PBR variants (HDR/LDR, standard/edge)
```

## 11. Performance Characteristics

| Shader | Threads/Group | Total Dispatches | Bottleneck |
|--------|--------------|------------------|------------|
| `object_cull` | 256 | ceil(objects/256) | Memory bandwidth |
| `cluster_tree_cull` | 64 | indirect (visible clusters) | Divergent control flow |
| `cluster_light_cull` | 256 | ceil(lights/256) | Atomic contention |
| `hzb_downsample_max` | 256 | log2(resolution) | Memory access pattern |
| `gpu_skinning` | 256 | ceil(vertices/256) | ALU (matrix multiply) |
| `ssao` | 256 | ceil(pixels/256) | Texture sampling |
| `taa_resolve` | 256 | ceil(pixels/256) | Texture sampling |
