## ADDED Requirements

### Requirement: Evaluate LINE curves
The system SHALL extract start point, direction vector, and end point from `LINE` entities.

#### Scenario: Simple line segment
- **WHEN** a `LINE` references a `CARTESIAN_POINT` and a `VECTOR`
- **THEN** the system returns the start point coordinates and computes the end point as `start + vector`

### Requirement: Evaluate CIRCLE curves
The system SHALL extract center, radius, and axis placement from `CIRCLE` entities and sample points along the arc.

#### Scenario: Full circle in XY plane
- **WHEN** a `CIRCLE` has `AXIS2_PLACEMENT_3D` with location at origin and axis (0,0,1), radius 5.0
- **THEN** sampling with 16 segments produces 16 points at radius 5.0 around the origin in the XY plane

#### Scenario: Circle with offset center
- **WHEN** a `CIRCLE` has `AXIS2_PLACEMENT_3D` with location (10, 20, 30)
- **THEN** all sampled points are offset by (10, 20, 30)

### Requirement: Evaluate ELLIPSE curves
The system SHALL extract semi-axis lengths and sample points along elliptical arcs.

#### Scenario: Ellipse with semi-axes 3.0 and 1.5
- **WHEN** an `ELLIPSE` has `semi_axis_1=3.0` and `semi_axis_2=1.5`
- **THEN** sampled points form an ellipse with major radius 3.0 and minor radius 1.5

### Requirement: Evaluate B_SPLINE_CURVE_WITH_KNOTS
The system SHALL evaluate B-spline curves using `rc3d-nurbs` or direct de Boor algorithm.

#### Scenario: Degree-3 NURBS curve
- **WHEN** a `B_SPLINE_CURVE_WITH_KNOTS` has degree=3, control points, and knot vector
- **THEN** the system evaluates points along the curve using de Boor's algorithm

### Requirement: Sample curves based on edge usage
The system SHALL sample curves with segment count proportional to their arc length and the face tessellation tolerance.

#### Scenario: Short line segment
- **WHEN** a LINE edge spans 0.1 units
- **THEN** no intermediate samples are needed (start and end points suffice)

#### Scenario: Long circular arc
- **WHEN** a CIRCLE edge spans 180 degrees
- **THEN** at least 8 intermediate samples are generated for smooth rendering
