## ADDED Requirements

### Requirement: Basic meshlet draw without GPU culling

The system SHALL provide a meshlet rendering path that draws all meshlet triangles using the full (uncompacted) vertex and index buffers via direct `draw_indexed` calls, without any GPU compute-based culling passes.

#### Scenario: Meshlet geometry renders correctly with PBR pipeline

- **WHEN** `draw_meshlets` is enabled AND `meshlet_gpu_cull_enabled` is false AND a meshlet-indexed draw call exists
- **THEN** the system SHALL draw all meshlet triangles using `draw_indexed(0..total_indices, 0, first_instance..first_instance+1)` with the full meshlet index buffer
- **AND** the rendered output SHALL be visually identical to the standard instanced draw path for the same geometry

#### Scenario: Basic path works on integrated GPU

- **WHEN** GPU tier is Basic (integrated GPU) AND `draw_meshlets` is enabled
- **THEN** the system SHALL use the basic draw path (no cull compute passes)
- **AND** no rendering artifacts (broken faces, flickering, wrong lines) SHALL appear

### Requirement: InstanceData pre-insert for meshlet draws

The system SHALL pre-insert meshlet `InstanceData` (model, mvp, material parameters) as the first element of the `all_instances` vector before Phase 1 standard instance collection, so the Phase 2 batch `queue.write_buffer` places it at slot 0 of the instance SSBO.

#### Scenario: Meshlet InstanceData at slot 0

- **WHEN** a meshlet draw is processed AND standard draws also exist
- **THEN** the meshlet InstanceData SHALL occupy slot 0 of the instance SSBO
- **AND** standard draw InstanceData SHALL start at slot 1
- **AND** standard draw `first_instance` values in `draw_batches` SHALL reflect the +1 offset

#### Scenario: Only meshlet draws, no standard draws

- **WHEN** a scene has only meshlet-indexed geometry (no standard draws)
- **THEN** `all_instances` SHALL contain exactly one entry (the meshlet InstanceData at slot 0)
- **AND** Phase 2 SHALL write this single entry to the instance SSBO
- **AND** the meshlet draw SHALL read instances[0] correctly

### Requirement: Vertex buffer binding isolation

The system SHALL reset `last_bound_mesh` to `None` after each meshlet draw call to prevent subsequent standard draws within the same render pass from reusing meshlet cluster vertex/index buffer bindings.

#### Scenario: Standard draw after meshlet draw re-binds correct buffers

- **WHEN** a meshlet draw changes vertex/index buffer bindings to meshlet cluster buffers
- **THEN** the next standard draw SHALL re-bind its own mesh vertex/index buffers regardless of mesh ID match
- **AND** the standard draw SHALL NOT use meshlet cluster buffers as vertex/index data
