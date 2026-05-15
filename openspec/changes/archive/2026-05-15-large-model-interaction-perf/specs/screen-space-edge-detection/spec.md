## ADDED Requirements

### Requirement: Depth-based edge detection shader
The system SHALL provide a full-screen shader that detects edges by computing depth discontinuities from the depth texture using a Sobel (3x3) filter kernel.

#### Scenario: Edge detection from depth
- **WHEN** the edge detection pass executes with a valid depth texture
- **THEN** pixels where the depth gradient magnitude exceeds a configurable threshold are colored with the configured edge color, and all other pixels are transparent

#### Scenario: Edge detection during interaction
- **WHEN** `interaction_active` is true AND `renderer.screen_space_edges` is true
- **THEN** the screen-space edge detection pass replaces geometry-based edge rendering for that frame

#### Scenario: Threshold configuration
- **WHEN** the user calls `renderer.set_ss_edge_threshold(0.02)`
- **THEN** subsequent edge detection frames use a Sobel gradient threshold of 0.02 (in depth units)

### Requirement: SS edge overlay compositing
The edge detection output SHALL be composited over the scene color as a post-processing overlay, either in the LDR post-fx stage or as a separate overlay pass.

#### Scenario: Edge overlay in LDR path
- **WHEN** HDR post-processing is disabled (LDR path) AND screen-space edges are active
- **THEN** the edge overlay is applied after the FXAA pass but before swapchain present

#### Scenario: Edge overlay in HDR path
- **WHEN** HDR post-processing is enabled AND screen-space edges are active
- **THEN** the edge overlay is applied after tonemapping, on the final LDR output before swapchain present
