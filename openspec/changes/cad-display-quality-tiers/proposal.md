## Why

CAD model visualization requires different rendering quality levels for different stages of the workflow — from fast model editing to final product rendering. The current renderer has feature toggles but no unified quality-tier system that maps CAD workflow stages to specific pass combinations, resolution settings, and interaction degradation behavior.

## What Changes

- Replace ad-hoc feature toggles with a `CadDisplayTier` enum (DesignCreation=0, Visualization=1, IndustrialDisplay=2, ProductRendering=3)
- Each tier defines exactly which passes run and at what target resolution/framerate
- Interaction degradation: tier 3→2 or 3→1, tier 2→1; tiers 0 and 1 never degrade
- Auto-recovery: after interaction stops, tier ramps back up with cooldown to prevent oscillation
- Replace existing `GpuTier`/AdaptiveQuality logic with the new tier system as the single source of truth for render configuration

## Capabilities

### New Capabilities
- `cad-display-tier`: Define four CAD workflow display tiers with per-tier pass configuration, resolution targets, and degradation rules

### Modified Capabilities
- None (new capability only; existing behavior preserved via tier defaults)

## Impact

- `crates/rc3d-render/src/renderer.rs` — new `CadDisplayTier` enum, tier config struct, interaction degrade/recover logic
- `crates/rc3d-render/src/render_passes.rs` — gate passes on tier instead of individual feature flags
- `crates/rc3d-render/src/renderer_internals.rs` — store tier in GpuInternals
- `crates/rc3d-studio/src/app.rs` — wire interaction state to tier degrade
- No shader changes required
- Backward compatible: default tier maps to current behavior
