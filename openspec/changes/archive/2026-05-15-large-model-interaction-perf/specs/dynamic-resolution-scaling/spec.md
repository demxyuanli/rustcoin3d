## ADDED Requirements

### Requirement: Render at reduced resolution during interaction
When the renderer's `interaction_active` flag is true, the system SHALL render solid, shadow, and edge passes to an intermediate HDR texture scaled to a configurable fraction of the viewport, then upscale the result to the swapchain view during post-processing.

#### Scenario: Interaction starts
- **WHEN** `interaction_active` transitions from false to true AND `interaction_render_scale < 1.0`
- **THEN** the renderer creates (or reuses) an intermediate HDR render target at scale × viewport dimensions AND all solid-depth passes target this intermediate texture

#### Scenario: Interaction ends
- **WHEN** `interaction_active` transitions from true to false
- **THEN** the intermediate render target is released after a cooldown period (1 second) AND rendering resumes at native viewport resolution

#### Scenario: Scale factor configuration
- **WHEN** the user calls `renderer.set_interaction_render_scale(0.5)`
- **THEN** subsequent interaction frames render at 50% resolution in each axis

#### Scenario: Window resize during interaction
- **WHEN** the window is resized while `interaction_active` is true
- **THEN** the intermediate render target is recreated at the new scale × viewport dimensions on the next frame

### Requirement: Upscale quality
The upscale from intermediate resolution to swapchain SHALL use bilinear filtering (not nearest-neighbor).

#### Scenario: Visual quality
- **WHEN** rendering is upscaled from 50% resolution
- **THEN** the output has soft but acceptable visual quality, sufficient for camera navigation
