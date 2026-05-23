## ADDED Requirements

### Requirement: Evaluate PLANE surfaces
The system SHALL extract plane placement from `PLANE` entities and provide a 3D point-to-2D parameterization.

#### Scenario: PLANE at origin
- **WHEN** a `PLANE` references `AXIS2_PLACEMENT_3D` at origin with Z axis
- **THEN** the surface maps 3D point (x, y, 0) to UV coordinates (x, y)

### Requirement: Evaluate CYLINDRICAL_SURFACE
The system SHALL extract radius and axis from `CYLINDRICAL_SURFACE` and provide UV parameterization (angle, height).

#### Scenario: Cylinder of radius 5
- **WHEN** a `CYLINDRICAL_SURFACE` has radius 5.0 and Z-axis
- **THEN** UV sampling with 32×32 grid produces points at radius 5.0 around the Z axis

### Requirement: Evaluate CONICAL_SURFACE
The system SHALL extract radius, semi-angle, and axis from `CONICAL_SURFACE`.

#### Scenario: Cone with semi-angle 15 degrees
- **WHEN** a `CONICAL_SURFACE` has semi_angle=0.2618 (15°) and base radius 10
- **THEN** the surface expands from apex at the given angle

### Requirement: Evaluate SPHERICAL_SURFACE
The system SHALL extract center and radius from `SPHERICAL_SURFACE` and provide spherical parameterization.

#### Scenario: Sphere of radius 3
- **WHEN** a `SPHERICAL_SURFACE` has radius 3.0
- **THEN** UV sampling produces points at distance 3.0 from center

### Requirement: Evaluate TOROIDAL_SURFACE
The system SHALL extract major and minor radii from `TOROIDAL_SURFACE`.

#### Scenario: Torus with major=10, minor=2
- **WHEN** a `TOROIDAL_SURFACE` has major_radius=10.0 and minor_radius=2.0
- **THEN** UV sampling produces the expected toroidal surface

### Requirement: Evaluate B_SPLINE_SURFACE_WITH_KNOTS
The system SHALL construct `rc3d_nurbs::NurbsSurface` from `B_SPLINE_SURFACE_WITH_KNOTS` entity data.

#### Scenario: Degree (3,3) B-spline surface
- **WHEN** a `B_SPLINE_SURFACE_WITH_KNOTS` has u_degree=3, v_degree=3 with control points grid
- **THEN** the system creates a `NurbsSurface` and calls `tessellate_adaptive(tolerance)`

### Requirement: Handle unsupported surface types
The system SHALL skip faces with unsupported surface types without aborting the import.

#### Scenario: Unsupported surface
- **WHEN** a face references a surface type not in the supported list
- **THEN** that face is skipped and processing continues
