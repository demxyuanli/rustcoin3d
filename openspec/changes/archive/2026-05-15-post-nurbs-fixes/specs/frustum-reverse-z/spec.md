## ADDED Requirements

### Requirement: Reverse-Z frustum near-plane extraction
The `Frustum::from_view_projection` function SHALL accept a `depth_reversed_z: bool` parameter. When false (forward-Z), the near plane SHALL be extracted as `r2`. When true (reverse-Z), the near plane SHALL be extracted as `r2 - r3`. The far plane SHALL remain `r3 - r2` in both modes.

#### Scenario: Forward-Z projection
- **WHEN** `depth_reversed_z` is false and the VP matrix uses a wgpu forward-Z projection
- **THEN** the near frustum plane corresponds to `z_clip = 0` in clip space

#### Scenario: Reverse-Z projection
- **WHEN** `depth_reversed_z` is true and the VP matrix uses a reverse-Z projection (ndc_z near→1, far→0)
- **THEN** the near frustum plane corresponds to `z_clip = w` in clip space

#### Scenario: AABB correctly passes near-plane test in reverse-Z
- **WHEN** an AABB is entirely in front of the reverse-Z near plane
- **THEN** `intersects_aabb` returns true

### Requirement: Backward compatibility
Existing callers of `Frustum::from_view_projection` that do not pass `depth_reversed_z` SHALL continue to compile and behave identically to before the change. `depth_reversed_z` SHALL default to `false`.

#### Scenario: No parameter passed
- **WHEN** `Frustum::from_view_projection(vp)` is called without a depth flag
- **THEN** forward-Z near-plane extraction (`r2`) is used, matching the current behavior
