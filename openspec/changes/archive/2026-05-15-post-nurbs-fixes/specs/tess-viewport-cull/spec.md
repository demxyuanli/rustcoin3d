## ADDED Requirements

### Requirement: Off-screen quads skip subdivision
During screen-space adaptive tessellation, after projecting a quad's 4 corners to screen coordinates, the system SHALL compute the axis-aligned bounding rect `(x_min, y_min, x_max, y_max)` of the 4 projected points. If the entire bounding rect lies outside the viewport on any side (`x_max < 0`, `x_min > viewport_w`, `y_max < 0`, or `y_min > viewport_h`), the quad SHALL be emitted immediately via `emit_quad` without further subdivision and without consuming vertex budget.

#### Scenario: Quad fully to the left of viewport
- **WHEN** all 4 projected corners have `x_max < 0`
- **THEN** the quad is emitted without subdivision and `vertices_used` is unchanged

#### Scenario: Quad fully above viewport
- **WHEN** all 4 projected corners have `y_max < 0`
- **THEN** the quad is emitted without subdivision and `vertices_used` is unchanged

#### Scenario: Quad partially visible
- **WHEN** a quad's bounding rect overlaps the viewport
- **THEN** the quad proceeds through normal subdivision heuristics

### Requirement: Recovered budget benefits visible quads
When off-screen quads are skipped without consuming their budget allocation, the saved vertex budget SHALL remain available for visible quads that are processed later. The per-quad budget tracking via `vertices_used` SHALL not be incremented for skipped quads.

#### Scenario: Budget not consumed by off-screen quad
- **WHEN** an off-screen quad with a 2000-vertex budget is skipped
- **THEN** the 2000 vertices remain available for subsequent visible quads in the same pool
