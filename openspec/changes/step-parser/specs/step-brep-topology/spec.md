## ADDED Requirements

### Requirement: Traverse Shell to FaceSurface entities
The system SHALL iterate all Shell entities in the entity map and extract FaceSurface entities from their face lists.

#### Scenario: ClosedShell with FaceSurface faces
- **WHEN** a `CLOSED_SHELL` entity references `(#10, #11, #12)` as faces and `#10` is a `FACE` or `FACE_SURFACE`
- **THEN** the system resolves each face reference and collects `FaceSurface` data

#### Scenario: Shell with mixed face types
- **WHEN** a `SHELL` entity contains both `FACE_SURFACE` and `ORIENTED_FACE` entries
- **THEN** the system unwraps `ORIENTED_FACE` to access the underlying `FACE_SURFACE`

### Requirement: Extract face boundary loops
The system SHALL extract outer and inner boundary loops (FACE_OUTER_BOUND, FACE_BOUND) from each FaceSurface.

#### Scenario: Single outer bound
- **WHEN** a `FACE_SURFACE` references `(#5)` as a `FACE_OUTER_BOUND`
- **THEN** the system resolves `#5` → `EDGE_LOOP` and collects its edge list

#### Scenario: Face with hole
- **WHEN** a `FACE_SURFACE` has one `FACE_OUTER_BOUND` and one `FACE_BOUND` (hole)
- **THEN** both boundary loops are extracted for later tessellation

### Requirement: Walk edge loops to edge curves
The system SHALL iterate `EDGE_LOOP` edge lists and resolve each edge to its `EDGE_CURVE` geometry.

#### Scenario: Loop with OrientedEdges
- **WHEN** an `EDGE_LOOP` contains `ORIENTED_EDGE` entities
- **THEN** the system unwraps each `ORIENTED_EDGE` to access the underlying `EDGE_CURVE` and its orientation flag

#### Scenario: Loop with direct EdgeCurves
- **WHEN** an `EDGE_LOOP` contains bare `EDGE_CURVE` entities (no orientation wrapper)
- **THEN** the system extracts `EDGE_CURVE` directly with default forward orientation

### Requirement: Handle missing or malformed topology
The system SHALL skip individual faces, bounds, or edges that fail resolution without aborting the entire import.

#### Scenario: Unresolvable edge reference
- **WHEN** an `EDGE_LOOP` references `#999` which is not a valid edge entity
- **THEN** that single face is skipped and processing continues with remaining faces
