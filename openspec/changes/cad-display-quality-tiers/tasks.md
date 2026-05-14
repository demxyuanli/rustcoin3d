## 1. Define CadDisplayTier and TierConfig

- [x] 1.1 Add `CadDisplayTier` enum and `TierConfig` struct to `renderer_internals.rs`
- [x] 1.2 Define `TierConfig::for_tier(tier)` factory that returns pass configuration for each tier
- [x] 1.3 Add `requested_tier`, `effective_tier`, `interaction_active`, `tier_cooldown_frames` to `GpuInternals`

## 2. Implement degradation and recovery logic

- [x] 2.1 Add `update_tier()` method on Renderer that computes `effective_tier` from `requested_tier` and `interaction_active`
- [x] 2.2 Implement cooldown: after interaction stops, step effective tier up one level per cooldown period
- [x] 2.3 Set `requested_tier` default to `Visualization` (tier 1) for backward compatibility

## 3. Derive feature toggles from tier

- [x] 3.1 Add `apply_tier_config()` that sets all feature toggle fields from TierConfig
- [x] 3.2 Call `apply_tier_config()` in Renderer whenever effective tier changes
- [x] 3.3 Feature toggles already gate render passes (existing code)

## 4. Wire interaction state from app layer

- [x] 4.1 `self.interaction_active` already set by event_handler (existing behavior)
- [x] 4.2 `interaction_active = true` already set on camera orbit/pan/zoom start
- [x] 4.3 `interaction_active = false` already set on camera idle

## 5. Wire GpuTier constraint

- [x] 5.1 `set_display_tier()` clamps to `Visualization` max on Basic GPU tier
- [x] 5.2 `log::warn!` when user requests tier above GPU capability

## 6. Verification

- [x] 6.1 `cargo check --workspace` — 0 errors
- [x] 6.2 `cargo test` — 250 passed, 4 ignored
- [ ] 6.3 Manual test: switch between tiers via `set_display_tier()`
- [ ] 6.4 Manual test: orbit camera (trigger degrade), stop (verify recovery)
- [ ] 6.5 Manual test: regression check with existing examples
