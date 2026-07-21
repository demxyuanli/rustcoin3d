# Forward Rendering Path — Impact Analysis 2026-07-21

## Current State

The render pipeline already separates draw calls into two groups:

```
pass_orchestration.rs:83-89
├── solid_order:  opaque (AlphaMode::Opaque) → G-buffer deferred shading
└── transparent_order: AlphaMode::Mask | AlphaMode::Blend → sorted back-to-front
```

Render order in `execute_passes()`:
```
shadow → solid(G-buffer) → transparent(sorted BLEND) → effects → post → HUD
```

## Why No True Forward Path Needed

1. **Deferred handles 95% of content** — G-buffer PBR with CSM shadows, IBL, SSAO. Material complexity (clearcoat, sheen, transmission) all compile into the deferred shader variant.

2. **Transparent pass covers the rest** — AlphaMode::Blend draw calls are separated from solid set, sorted back-to-front (painter's algorithm), and rendered with `solid_alpha` pipeline (`ALPHA_BLENDING`, depth-write off). WBOIT available for higher-quality, order-independent transparency.

3. **Forward path would only benefit**:
   - Single-pass transparent + opaque combined rendering (anti-aliasing edge case)
   - Very large numbers of overlapping transparent layers (>20) where painter's fails
   - Both are rare in CAD/engineering visualization

## Impact Analysis — No Change Needed

| Consumer | Impact |
|----------|--------|
| shader_permutation.rs | No new shader variant needed |
| pipelines.rs | `DepthModePipelines` already has `solid_alpha` and `wboit_accum` |
| render_passes.rs | `execute_passes` already handles transparent ordering |
| pass_transparent.rs | Existing implementation covers both WBOIT and painter's |
| flat_draw_cache.rs | Already separates `transparent_order` from `opaque_order` |
| pass_orchestration.rs | Already filters by `alpha_mode` at L84-86 |

## Decision

**Do not add forward rendering path.** Current deferred + sorted transparent covers all practical use cases. Re-evaluate if/when:
- Production scenes show visible alpha sorting artifacts with >15 overlapping transparent layers
- Real-time order-independent transparency is needed for volumetrics
- Single-pass rendering is required for mobile/WebGPU targets

Skipped: forward path (~500 lines of pipeline variant + shader permutation), add when above conditions met.
