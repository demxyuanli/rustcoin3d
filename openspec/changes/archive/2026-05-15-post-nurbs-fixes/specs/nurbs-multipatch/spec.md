## ADDED Requirements

### Requirement: Stitch two patches along a shared boundary
The system SHALL provide a function to stitch two `NurbsSurface` patches along a specified edge (u=0, u=1, v=0, or v=1). The shared boundary curve control points SHALL match exactly in 3D position. Stitching SHALL produce a single `TessellatedSurface` with deduplicated boundary vertices, ensuring no gaps at the seam.

#### Scenario: Position-continuous patch pair
- **WHEN** two patches with matching boundary control points are stitched along patch A's u=1 edge and patch B's u=0 edge
- **THEN** the output `TessellatedSurface` has no duplicate vertices along the shared boundary

#### Scenario: Mismatched boundaries are rejected
- **WHEN** two patches are stitched but their boundary control points differ by more than a tolerance
- **THEN** the stitch returns an error describing the mismatch

### Requirement: G¹ normal continuity at boundaries
The stitching function SHALL support an optional G¹ continuity mode. When enabled, vertex normals along the shared boundary SHALL be computed as the average of the surface normals from both patches at each shared vertex.

#### Scenario: G¹ stitching with normal averaging
- **WHEN** two patches are stitched with G¹ continuity enabled
- **THEN** normal vectors at shared boundary vertices equal the normalized average of the two patch normals

### Requirement: Multi-patch assembly
The system SHALL support stitching an arbitrary number of patches by repeated application of the pairwise stitch function. The input SHALL be a list of `NurbsSurface` patches with adjacency information specifying which edge connects to which.

#### Scenario: Four-patch assembly
- **WHEN** four patches are assembled in a 2×2 grid
- **THEN** the output is a single `TessellatedSurface` covering all four patches with no visible seams

### Requirement: Boundary curve extraction
The `NurbsSurface` type SHALL expose a method to extract a `NurbsCurve` representing an isoparametric edge (u=0, u=1, v=0, or v=1) for use in boundary matching and validation.

#### Scenario: Extract u=0 boundary curve
- **WHEN** `surface.boundary_curve(BoundaryEdge::UMin)` is called
- **THEN** the returned `NurbsCurve` has control points matching the surface's control points along the u=0 edge
