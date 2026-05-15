## ADDED Requirements

### Requirement: Automatic quality/performance path switching
The renderer SHALL switch between a "quality" render configuration (static camera) and a "performance" render configuration (interacting camera) automatically based on `interaction_active`.

#### Scenario: Enter performance path
- **WHEN** `interaction_active` is set to true
- **THEN** the renderer enables dynamic resolution scaling, skips the depth prepass, and uses screen-space edge detection (if available) for the current frame

#### Scenario: Exit performance path
- **WHEN** `interaction_active` is set to false
- **THEN** the renderer restores native resolution, re-enables the depth prepass, and restores geometry-based edge rendering after a cooldown

### Requirement: Skip depth prepass during interaction
The depth prepass SHALL be skipped when `interaction_active` is true, regardless of shadow or display mode settings.

#### Scenario: Prepass skipped during orbit
- **WHEN** the user is orbiting the camera AND shadows are enabled
- **THEN** the solid pass renders with reverse-Z depth testing without a preceding depth prepass

#### Scenario: Prepass restored after interaction
- **WHEN** the user stops camera interaction
- **THEN** the depth prepass is re-enabled on the next frame

### Requirement: Interaction render scale configuration
A public API method `Renderer::set_interaction_render_scale(scale: f32)` SHALL be available to set the resolution scale factor for interaction frames, clamped to [0.25, 1.0].

#### Scenario: Default scale
- **WHEN** `set_interaction_render_scale` has not been called
- **THEN** the interaction scale defaults to 0.5 (50% resolution per axis)

#### Scenario: Clamp out-of-range values
- **WHEN** `set_interaction_render_scale` is called with a value < 0.25 or > 1.0
- **THEN** the value is clamped to 0.25 or 1.0 respectively
