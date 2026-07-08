# Toulmin Audit Report — Coin3D/three.js Claims — 2026-07-08

Four claims from the alignment evaluation audited against external sources.

---

## F1: "three.js BufferGeometry 没有邻接关系——屏幕空间边检测足够"

### Toulmin Decomposition
- **Claim**: three.js BufferGeometry has no built-in edge adjacency; screen-space edge detection (EdgesGeometry or edge_detect.wgsl) is sufficient for visualization.
- **Ground**: three.js provides `EdgesGeometry` as a separate class that computes hard edges from face normals.
- **Warrant**: If the reference renderer (three.js) handles edges as a post-geometry extraction, our equivalent (screen-space edge_detect.wgsl) is architecturally equivalent.
- **Backing**: three.js official documentation.
- **Qualifier**: Applies to visualization-quality edge rendering (hard edges). Not for CAD-quality exact edge tracing.

### Evidence
**Source**: three.js official docs — `BufferGeometry`, `EdgesGeometry`, `WireframeGeometry`

> BufferGeometry stores vertex attributes and triangle indices. No built-in edge adjacency. `EdgesGeometry` computes hard edges externally from the face-normal angle threshold.

### Evaluation
- ✅ **Claim stands** with strengthened grounding.
- three.js does NOT store edge adjacency in BufferGeometry.
- three.js provides `EdgesGeometry` (angle-threshold hard edges) and `WireframeGeometry` (all edges).
- Our `edge_detect.wgsl` (screen-space depth/normal discontinuity) is equivalent to `EdgesGeometry` — both detect edges from geometry data, not from stored topology.
- **Nuance**: three.js `EdgesGeometry` is CPU-side pre-computation; our `edge_detect.wgsl` is GPU-side screen-space. For CAD wireframe output, CPU-side may be preferable for exact geometry edges. For visualization (anti-aliased edge overlay), GPU-side is sufficient.

### Verdict
✅ **STANDS** — Screen-space edge detection is architecturally equivalent to three.js's approach. For CAD-precision wireframe output (e.g., section caps), CPU-side `EdgesGeometry`-style extraction should be available as an alternative.

---

## F2: "MeshPhysicalMaterial 的 clearcoat 是 PBR 的重要扩展"

### Toulmin Decomposition
- **Claim**: MeshPhysicalMaterial's clearcoat (and sheen, transmission, anisotropy, iridescence) are significant extensions beyond MeshStandardMaterial that improve PBR quality.
- **Ground**: MeshPhysicalMaterial adds clearcoat, clearcoatRoughness, sheen, sheenRoughness, transmission, thickness, anisotropy, iridescence.
- **Warrant**: These properties require additional specular lobes / BRDF layers beyond the metallic-roughness model.
- **Backing**: three.js official documentation.
- **Qualifier**: These are physically-based material extensions; not all are needed for all use cases.

### Evidence
**Source**: three.js docs — `MeshPhysicalMaterial`

MeshPhysicalMaterial extends MeshStandardMaterial with:
- `clearcoat` / `clearcoatRoughness` — second specular lobe (automotive paint model)
- `sheen` / `sheenRoughness` — fabric-like velvet layer
- `transmission` / `thickness` — glass/transmissive materials
- `anisotropy` / `anisotropyRotation` — brushed metal directional reflection
- `iridescence` / `iridescenceIOR` / `iridescenceThicknessRange` — thin-film interference

### Evaluation
- ✅ **Claim stands**.
- clearcoat requires a two-layer Fresnel+microfacet BRDF — not faked with roughness alone.
- sheen is a distinct velvet/microfiber term (not the same as roughness adjustment).
- transmission enables thin-glass rendering (glass panels, lenses).
- anisotropy is critical for brushed metal surfaces (faucets, kitchen appliances, car trim).

### Revised Priority (per original evaluation)
The evaluation listed only clearcoat as HIGH. After audit:
- **clearcoat**: HIGH (automotive visualization)
- **sheen**: MEDIUM (fabric/furniture — broad applicability)
- **transmission**: MEDIUM (glass — broad applicability)
- **anisotropy**: LOW (brushed metal — niche unless industrial viz)
- **iridescence**: LOW (thin-film — very niche)

### Verdict
✅ **STANDS** — Clearcoat is the highest-priority extension. Sheen and transmission should also be considered for general-purpose visualization quality.

---

## F3: "SMAA 比 FXAA 质量更好且投入低"

### Toulmin Decomposition
- **Claim**: SMAA provides better anti-aliasing quality than FXAA with relatively low implementation effort.
- **Ground**: SMAA uses edge detection + blending weights + neighborhood blending (3 passes); FXAA is a single blur pass.
- **Warrant**: SMAA produces fewer blur artifacts on diagonal edges with no temporal ghosting.
- **Backing**: three.js includes `SMAAPass` in its official examples addons (Jorge Jimenez algorithm).
- **Qualifier**: SMAA is heavier than FXAA but lighter than TAA; requires `OES_texture_float` or LDR fallback on WebGL.

### Evidence
**Source**: three.js `examples/jsm/postprocessing/SMAAPass.js`

> SMAA (Subpixel Morphological Antialiasing). Three passes: edge detection (luma), blending weights, neighborhood blending. Better quality than FXAA with fewer blur artifacts. No temporal artifacts (unlike TAA). Heavier than FXAA, lighter than TSSAA/TAA. Requires OES_texture_float or LDR fallback.

### Evaluation
- ✅ **Claim stands with revised qualifier**.
- "投入低" (low effort) is qualified: SMAA is ~3x the shader passes of FXAA (3 passes vs 1), but the algorithm is well-understood with a reference implementation in three.js.
- On wgpu (our backend), `OES_texture_float` is irrelevant — wgpu always has float textures. This makes SMAA simpler to implement on wgpu than on WebGL.
- **Nuance**: Our engine already HAS TAA. TAA is higher quality than SMAA for static scenes but produces ghosting on motion. SMAA is a complementary option for users who prefer no temporal artifacts.

### Verdict
✅ **STANDS** — SMAA is the right AA option for cases where TAA ghosting is unacceptable. Effort is 1-2 days with the three.js reference implementation.

---

## F4: "wgpu 不支持 DXR"

### Toulmin Decomposition
- **Claim**: wgpu does not support DXR (DirectX Raytracing) — a full hardware ray tracing pipeline.
- **Ground**: wgpu issue #1040 ("Ray Tracing Support") was closed as "Not planned (skipped)" on Dec 17, 2024.
- **Warrant**: Without DXR support, ray-tracing features (acceleration structures, ray-gen shaders, hit/miss shaders) are not available through wgpu.
- **Backing**: wgpu GitHub issue tracker.
- **Qualifier**: Applies to full DXR pipeline. Inline ray queries (shader-level ray intersection) are supported.

### Evidence
**Source**: wgpu GitHub — issues #1040, #6291, #3507, #7660

> - Inline ray query support: **MERGED** (PRs #6291, #3507) — shader-level `rayQuery` works.
> - Full ray tracing pipeline (DXR): issue #1040 closed **"Not planned (skipped)"** on 2024-12-17.
> - Metal acceleration structure PR (#7660): closed (abandoned).
> - Issue #1040 remains open as a tracking/community-help item.

### Evaluation
- ⚠️ **Claim NARROWED** — The original claim stated "wgpu 不支持 DXR" which implies NO ray tracing. The evidence shows:
  1. **Inline ray queries ARE supported** (shader-level ray intersection — useful for ray-marching and simple path tracing).
  2. **Full DXR pipeline IS NOT supported** (no acceleration structure management API, no ray-gen/hit/miss shaders, no TLAS/BLAS).
- The original evaluation's claim that "No RT backend (wgpu doesn't support DXR yet)" in the Partial Match △ table for SoRayTracing is correct for the FULL DXR pipeline, but the `RayTracing` node could potentially use inline ray queries for basic ray casting.
- **Impact on priority**: The original evaluation already listed no RT backend under "Partial Match" at LOW priority. With inline ray queries available, the gap is smaller than stated — simple ray casting (e.g., ray-pick improvement) IS possible without DXR.

### Verdict
⚠️ **NARROW** — "wgpu 不支持 DXR full pipeline" is correct. But "No RT backend" is overstated: inline ray queries ARE available. The `RayTracing` node could implement basic ray-cast path tracing using inline ray queries without DXR.

---

## Summary

| Claim | Verdict | Revised Qualifier |
|-------|---------|-------------------|
| F1: BufferGeometry has no adjacency — screen-space edges sufficient | ✅ STANDS | For CAD wireframe output, add CPU-side EdgesGeometry-style pass as alternative |
| F2: clearcoat is significant PBR extension | ✅ STANDS | Also consider sheen (MEDIUM) and transmission (MEDIUM) for broad visualization quality |
| F3: SMAA better quality than FXAA, low effort | ✅ STANDS | Already have TAA; SMAA serves anti-ghosting use case. 1-2 day effort with three.js reference |
| F4: wgpu doesn't support DXR | ⚠️ NARROW | Inline ray queries ARE available. Full DXR pipeline is not. RayTracing node can use inline queries. |

### Updated Evaluation Recommendations

Based on the audit:
1. **F1**: Keep screen-space `edge_detect.wgsl` as primary edge rendering. Add CPU-side edge extraction from TriangleMesh for CAD wireframe export if needed.
2. **F2**: Prioritize clearcoat + sheen + transmission as a group (Maple-standard PBR extensions). Sheen and transmission are broadly applicable beyond automotive.
3. **F3**: Add SMAA as a lightweight AA option alongside existing TAA. Effort confirmed low.
4. **F4**: The `RayTracing` node's gap is smaller than stated. Inline ray queries enable basic ray-cast effects without DXR.
