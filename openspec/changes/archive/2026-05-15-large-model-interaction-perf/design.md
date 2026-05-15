## Context

The rustcoin3d renderer targets industrial 3D visualization (Coin3D/HOOPS-aligned) using wgpu (primarily D3D12 on Windows). The renderer already has a tier-based quality system (`CadDisplayTier`), interaction-aware degradation (`update_tier()`), and performance mode for large scenes.

Current rendering pipeline for a large mesh model (2.7M triangles, meshlet-enabled):
- Depth prepass (if shadows enabled): ~0.5ms GPU  
- CSM shadow pass: ~0.02ms GPU (meshlet-accelerated)
- Solid pass (PBR): ~29ms GPU ← main bottleneck
- Edge overlay (geometry-based): variable, disabled in perf mode
- Post-processing (HDR tonemap): ~4ms GPU
- Total GPU: ~33ms → 30 FPS cap

During interaction (camera orbit/pan/zoom), the `interaction_active` flag is set, which triggers:
- `update_tier()` degradation (tier 2→1)
- Performance mode (skips edges, SSAO, effects)
- Interaction quality override in event_handler (forces Flat display + disables HDR)

All of these help but don't address the fundamental GPU pixel fillrate bottleneck.

## Goals / Non-Goals

**Goals:**
- Reduce GPU frame time by 50-70% during interaction, targeting 30+ FPS for 2.7M-triangle models
- Make interaction feel responsive (no freeze-then-jump behavior caused by event pileup during long render)
- Maintain visual quality acceptable for navigation (not final-quality rendering)
- Automatically switch between quality/performance configs based on interaction state
- No breaking changes to existing API

**Non-Goals:**
- Changing static-frame rendering quality or performance
- GPU-driven LOD or mesh simplification (separate project)
- Async/threaded rendering pipeline (too invasive)
- Real-time mesh decimation (complex, error-prone)

## Decisions

### Decision 1: Dynamic resolution via intermediate render target

**Chosen**: Create a smaller intermediate HDR texture during interaction (e.g. 50% of viewport), render solid/shadow/edge passes to it, then upscale in the post-processing pass to the swapchain view.

**Alternatives considered**:
- *wgpu `set_viewport` + scissor*: Reduces pixel area but doesn't actually reduce the render target size. The GPU still processes the full swapchain texture. Rejected because the pixel shader still runs at full resolution for all geometry.
- *Hardware Variable Rate Shading (VRS)*: Requires DX12 Ultimate / Vulkan 1.2, not available on all GPUs. Added complexity.
- *Foveated rendering*: Overkill for CAD; uniform reduction is sufficient.

**Rationale**: An intermediate render target is the standard approach (Unreal/Unity dynamic resolution, console games' "checkerboard rendering"). It directly reduces pixel shader invocations proportionally to the scale factor (50% scale → 25% pixels → ~7ms solid pass from 29ms).

**Scale factor**: 50% (0.5x each axis) during interaction. This is the sweet spot: aggressive enough to make a real difference (4x fewer pixels), but still produces acceptable upscale quality with bilinear filtering.

### Decision 2: Skip depth prepass during interaction

**Chosen**: Gate depth prepass on `!interaction_active` in addition to existing conditions.

**Rationale**: The depth prepass reduces solid-pass overdraw by populating the depth buffer first. During interaction, the prepass itself consumes ~3-5ms GPU time. Skipping it and relying on reverse-Z depth testing in the solid pass saves that time. Overdraw in the solid pass may increase slightly, but the net GPU time is lower for large models where vertex processing (not overdraw) dominates.

### Decision 3: Screen-space edge detection from depth buffer

**Chosen**: A full-screen compute or fragment shader that reads the depth texture, computes depth discontinuities via Sobel filter (3x3 kernel), and outputs edge color to an overlay. Implementation in two phases:
- Phase A (this change): Depth-only edge detection via existing `depth_texture`. Single pass, ~0.5ms GPU.
- Phase B (future): Add normal buffer for better feature edge detection (silhouette + crease).

**Alternatives considered**:
- *Keep geometry-based edges during interaction*: Too expensive for large models (millions of edges).
- *No edges during interaction*: Loses important visual feedback for navigation.
- *Temporal edge accumulation*: Reduces edge flickering but adds implementation complexity. Defer to future.

**Rationale**: Screen-space edge detection is the industry standard for game engines (Source 2, many Unity/Unreal titles). It decouples edge rendering cost from triangle count — the cost is fixed per-pixel regardless of model complexity. Depth discontinuities naturally highlight geometric edges, creases, and silhouettes, providing good navigation feedback.

### Decision 4: Integration with existing interaction flow

**Chosen**: Extend the existing `interaction_quality_override` block in `event_handler.rs` to also set `renderer.interaction_render_scale` and `renderer.skip_prepass`. No new event types or hooks needed.

**Rationale**: The `interaction_active` flag is already correctly set and the quality override path already handles HDR/display mode switching. Adding resolution scaling and prepass skip here keeps all interaction optimizations in one place.

## Risks / Trade-offs

- [Upscale quality at 50%] → Use bilinear filtering (not nearest-neighbor). At 50% scale, bilinear upscale is visually acceptable for navigation. Slight softness is tolerable during camera movement.
- [Depth-only edge detection misses internal creases] → Acceptable trade-off for Phase A. Silhouette and depth-discontinuity edges provide sufficient spatial reference during navigation. Phase B adds normal-buffer edge detection for creases.
- [Intermediate texture allocation on first interaction frame] → Pre-allocate the downscale texture at renderer initialization (sized to 50% of initial viewport). Resize on window resize. Avoids allocation stutter during first interaction frame.
- [Edge artifacts from upscaling] → The upscale pass happens AFTER post-processing (including any edge overlay), so edges are rendered at reduced resolution. This may cause slight edge aliasing at upscale boundaries. Mitigated by bilinear interpolation; if problematic, add an FXAA pass after upscale.
