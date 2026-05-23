## ADDED Requirements

### Requirement: Tessellate planar faces from edge loop vertices
The system SHALL extract polygon vertices from planar face edge loops and triangulate via fan triangulation.

#### Scenario: Rectangular face
- **WHEN** a planar face has an outer loop with 4 LINE edges forming a rectangle
- **THEN** the system extracts 4 vertices and produces 2 triangles via fan triangulation

#### Scenario: Face with mixed LINE and CIRCLE edges
- **WHEN** a planar face has 2 LINE edges and 2 CIRCLE edges
- **THEN** the system samples CIRCLE edges and produces a closed polygon for triangulation

### Requirement: Tessellate curved surfaces via UV sampling
The system SHALL sample curved surfaces in UV parameter space and triangulate the resulting grid.

#### Scenario: Cylindrical surface
- **WHEN** a `CYLINDRICAL_SURFACE` face is sampled with 32×32 UV grid
- **THEN** the system produces a quad grid of 32×32 vertices and triangulates each quad

### Requirement: Handle face orientation
The system SHALL use the `FaceSurface.same_sense` flag to determine face normal orientation and invert triangle winding when false.

#### Scenario: Reversed face
- **WHEN** a `FaceSurface` has `same_sense=false`
- **THEN** the system inverts triangle winding order for that face

### Requirement: Produce closed IndexedFaceSet format
The system SHALL output vertex coordinates and triangle indices in the engine's `IndexedFaceSetNode` format (triangle index triples with -1 sentinel).

#### Scenario: Single triangle
- **WHEN** a face produces one triangle with vertices at indices 0, 1, 2
- **THEN** the output is `[0, 1, 2, -1]`

### Requirement: Merge duplicate vertices across faces
The system SHALL reuse vertex indices for identical positions to reduce GPU buffer size.

#### Scenario: Shared edge between adjacent faces
- **WHEN** two adjacent faces share an edge with the same endpoint coordinates
- **THEN** the shared vertex is stored once and referenced by both faces' triangles
