## ADDED Requirements

### Requirement: Optional GPU meshlet culling

The system SHALL provide an optional compute-based meshlet culling pipeline (frustum cull + compact + finalize indirect args + `draw_indexed_indirect`) that is enabled only when `meshlet_gpu_cull_enabled` is true in `GpuCapability`.

#### Scenario: GPU cull enabled on capable hardware

- **WHEN** `meshlet_gpu_cull_enabled` is true AND meshlet-indexed draws exist
- **THEN** the system SHALL run the cull compute passes (cull → compact → finalize) in the main command encoder before the render pass
- **AND** `draw_clustered` SHALL use `draw_indexed_indirect` with the compacted index buffer and indirect draw args

#### Scenario: GPU cull disabled falls back to basic path

- **WHEN** `meshlet_gpu_cull_enabled` is false AND meshlet-indexed draws exist
- **THEN** the system SHALL skip all cull compute passes
- **AND** meshlet draws SHALL use the basic path (full index buffer, direct `draw_indexed`)

### Requirement: GpuCapability tier controls cull enablement

The `GpuCapability` struct SHALL include `meshlet_gpu_cull_enabled: bool`. At renderer init, this SHALL be set to `true` for Standard and Enhanced GPU tiers, and `false` for Basic tier (integrated GPUs).

#### Scenario: Integrated GPU gets basic path

- **WHEN** GPU device type is IntegratedGpu or Cpu
- **THEN** `GpuTier::Basic` SHALL be assigned
- **AND** `meshlet_gpu_cull_enabled` SHALL be false

#### Scenario: Discrete GPU gets cull path

- **WHEN** GPU device type is DiscreteGpu
- **THEN** `GpuTier::Standard` SHALL be assigned
- **AND** `meshlet_gpu_cull_enabled` SHALL be true

### Requirement: Cull reset uses encoder.clear_buffer

The cull pipeline SHALL reset the `visible_buffer` atomic counter and `indirect_buffer` draw args using `encoder.clear_buffer` (not `queue.write_buffer`) to ensure the clears are ordered on the encoder timeline with subsequent compute dispatches.

#### Scenario: Clear ordered before compute

- **WHEN** `cull_and_compact` is called with an encoder
- **THEN** `encoder.clear_buffer` SHALL be called on `visible_buffer` (4 bytes) and `indirect_buffer` (24 bytes) before any compute dispatch
- **AND** no `queue.write_buffer` SHALL be used for these resets
