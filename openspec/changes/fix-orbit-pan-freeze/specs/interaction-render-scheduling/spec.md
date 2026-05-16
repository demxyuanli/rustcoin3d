## ADDED Requirements

### Requirement: Continuous rendering during camera interaction

The system SHALL render frames continuously during camera orbit, pan, and zoom interactions, producing smooth visual updates that are not delayed or blocked by the Windows input event queue.

#### Scenario: Orbit updates every frame during mouse drag

- **WHEN** the user holds middle mouse button and drags to orbit the camera
- **THEN** the viewport SHALL render at least one frame per 50ms (20+ fps) throughout the drag
- **AND** the displayed image SHALL update to reflect intermediate camera positions, not only the final position on release

#### Scenario: Pan updates every frame during mouse drag

- **WHEN** the user holds right mouse button and drags to pan the camera
- **THEN** the viewport SHALL render at least one frame per 50ms (20+ fps) throughout the drag

#### Scenario: Zoom updates every wheel event

- **WHEN** the user scrolls the mouse wheel to zoom
- **THEN** the viewport SHALL render within one frame after the scroll event

### Requirement: No regression for static scene rendering

The system SHALL maintain the existing rendering behavior and power efficiency when the camera is static (no user interaction).

#### Scenario: Static scene uses existing render scheduling

- **WHEN** the camera is static and no user interaction is active
- **THEN** the render loop SHALL use the existing `request_redraw()` mechanism
- **AND** the frame rate SHALL be no higher than necessary to maintain vsync

### Requirement: Interaction detection covers all camera modes

The interaction state SHALL correctly detect all forms of camera interaction including legacy single-camera orbit/pan/zoom and multi-viewport camera orbit/pan/zoom.

#### Scenario: Legacy camera orbit detected

- **WHEN** `app.state.camera_controller` is `Some` and `middle_orbit_held` or `left_orbit_held` or `panning` is true
- **THEN** the system SHALL classify this as active interaction

#### Scenario: Viewport camera orbit detected

- **WHEN** any `ViewportCamera` in `app.state.viewport_cameras` has an active orbit or pan state
- **THEN** the system SHALL classify this as active interaction
