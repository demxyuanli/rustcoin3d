## ADDED Requirements

### Requirement: CadDisplayTier enum defines four CAD workflow tiers

The system SHALL provide a `CadDisplayTier` enum with four variants: `DesignCreation=0`, `Visualization=1`, `IndustrialDisplay=2`, `ProductRendering=3`. Each variant SHALL have an associated `TierConfig` that specifies which rendering passes are enabled.

#### Scenario: Each tier maps to correct pass configuration

- **WHEN** `CadDisplayTier::DesignCreation` is active
- **THEN** Flat shading with edges SHALL be enabled and all other passes SHALL be disabled
- **WHEN** `CadDisplayTier::Visualization` is active
- **THEN** PBR shading, IBL, CSM shadows, edges, and motion blur SHALL be enabled
- **AND** SSAO, TAA, color grading, SSR, volumetric fog, DOF SHALL be disabled
- **WHEN** `CadDisplayTier::IndustrialDisplay` is active
- **THEN** all Visualization passes plus SSAO, TAA/FXAA, color grading SHALL be enabled
- **AND** SSR, volumetric fog, DOF SHALL be disabled
- **WHEN** `CadDisplayTier::ProductRendering` is active
- **THEN** all IndustrialDisplay passes plus SSR, volumetric fog, DOF SHALL be enabled

### Requirement: Interaction degradation lowers effective tier

The system SHALL compute an `effective_tier` that is lower than the requested tier when `interaction_active` is true.

#### Scenario: Tier 3 degrades to tier 1 during interaction

- **WHEN** `requested_tier` is `ProductRendering` AND `interaction_active` is true
- **THEN** `effective_tier` SHALL be `Visualization` (degrade by 2 steps)

#### Scenario: Tier 2 degrades to tier 1 during interaction

- **WHEN** `requested_tier` is `IndustrialDisplay` AND `interaction_active` is true
- **THEN** `effective_tier` SHALL be `Visualization` (degrade by 1 step)

#### Scenario: Tiers 0 and 1 never degrade

- **WHEN** `requested_tier` is `DesignCreation` or `Visualization` AND `interaction_active` is true
- **THEN** `effective_tier` SHALL equal `requested_tier` (degrade steps = 0)

### Requirement: Tier recovery after interaction stops

After `interaction_active` transitions from true to false, the system SHALL ramp the effective tier back toward the requested tier one step at a time, with a cooldown period between each step to prevent oscillation.

#### Scenario: Tier recovers with cooldown

- **WHEN** `interaction_active` becomes false AND `requested_tier` is `ProductRendering` AND `effective_tier` is `Visualization`
- **THEN** after cooldown (500ms) of no interaction, `effective_tier` SHALL step to `IndustrialDisplay`
- **AND** after another cooldown (500ms) of no interaction, `effective_tier` SHALL step to `ProductRendering`

#### Scenario: Interaction restarts cooldown

- **WHEN** cooldown timer is counting toward next tier AND `interaction_active` becomes true again
- **THEN** cooldown SHALL reset AND `effective_tier` SHALL immediately degrade again

### Requirement: Feature toggles derived from effective tier

The system SHALL derive all existing feature toggle flags (`enable_taa`, `enable_ssr`, `enable_ssao`, `enable_motion_blur`, `enable_color_grading`, `enable_dof`, `enable_volumetric_fog`, `hdr_post_processing`, `hud_enabled`, `grid_enabled`) from the `effective_tier`'s `TierConfig`.

#### Scenario: Feature flags reflect tier config

- **WHEN** tier changes from `IndustrialDisplay` to `ProductRendering`
- **THEN** `enable_ssr` SHALL become true
- **AND** `enable_volumetric_fog` SHALL become true
- **AND** `enable_dof` SHALL become true
