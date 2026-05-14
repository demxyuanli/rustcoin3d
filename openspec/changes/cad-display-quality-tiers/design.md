## Context

The renderer currently has ~15 individual feature toggles (`enable_taa`, `enable_ssr`, `enable_ssao`, etc.) plus `GpuTier` for capability detection and `AdaptiveQuality` for triangle-count-based degradation. These interact in unclear ways. The project needs a unified tier system that maps directly to CAD workflow stages.

## Goals / Non-Goals

**Goals:**
- Single `CadDisplayTier` enum controls all pass enablement
- Per-tier interaction degradation rules
- Hysteresis on tier changes to prevent oscillation
- Backward compatible — existing feature toggles derived from tier

**Non-Goals:**
- Per-pass quality parameters (MSAA sample count, shadow map resolution) — out of scope
- Resolution scaling — different concern, can be layered on top later
- Dynamic tier selection based on frame time — manual tier + interaction override only for now

## Decisions

**Decision 1: `CadDisplayTier` enum with `TierConfig` struct**

```rust
pub enum CadDisplayTier {
    DesignCreation = 0,
    Visualization = 1,
    IndustrialDisplay = 2,
    ProductRendering = 3,
}

struct TierConfig {
    shaded: bool,
    ibl: bool,
    shadows: bool,
    edges: bool,
    motion_blur: bool,
    ssao: bool,
    taa: bool,
    fxaa: bool,
    color_grading: bool,
    ssr: bool,
    volumetric_fog: bool,
    dof: bool,
    hdr_post: bool,
    target_fps: u32,
}
```

Rationale: Single struct per tier makes it easy to see what each tier enables. No boolean matrix to maintain.

**Decision 2: Interaction degradation via clamp**

When `interaction_active` is true:
- `effective_tier = min(requested_tier, degrade_tier)` where `degrade_tier` = max(0, requested_tier - degrade_steps)
- Tier 3: degrade_steps = 2 (→ tier 1 during interaction)
- Tier 2: degrade_steps = 1 (→ tier 1 during interaction)
- Tier 0, 1: degrade_steps = 0 (no degradation)

**Decision 3: Recovery with cooldown**

After interaction stops, tier ramps back one step per cooldown period (e.g., 500ms):
1→2 after 500ms idle, then 2→3 after another 500ms. Prevents oscillation during rapid view changes.

**Decision 4: Replace `AdaptiveQuality` and derivation of feature flags**

`CadDisplayTier` becomes the single source of truth. Feature toggles (`enable_taa`, etc.) are set once from the tier config when tier changes. Keep the existing toggle fields for backward compat but derive them from tier.

## Risks / Trade-offs

- **Manual tier selection**: User must explicitly choose tier 2 or 3. No automatic quality adjustment based on GPU capability. → Mitigation: GpuTier still prevents enabling passes the hardware can't handle.
- **Interaction degrade might feel aggressive**: Tier 3→1 drops shadows, SSAO, SSR, DOF all at once. → Mitigation: Cooldown ensures these only drop during sustained interaction.
