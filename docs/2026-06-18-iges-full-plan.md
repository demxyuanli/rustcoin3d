# Phase 7: Full IGES Entity Support

Date: 2026-06-18.

## Target entities (8 new)

| Type | Name | Priority | Status |
|------|------|----------|--------|
| 110 | Line | P0 | ✅ Done |
| 100 | Circular Arc | P0 | ✅ Done |
| **102** | Composite Curve | P1 | 🔧 |
| **108** | Plane (bounded) | P1 | 🔧 |
| **120** | Surface of Revolution | P1 | 🔧 |
| **122** | Tabulated Cylinder | P1 | 🔧 |
| **128** | Rational BSpline Surface | P1 | 🔧 |
| **144** | Trimmed Surface | P1 | 🔧 |
| **142** | Curve on Parametric Surface | P2 | 🔧 |
| **140** | Offset Surface | P2 | 🔧 |

## Implementation plan

**File**: `crates/rc3d-io/src/iges.rs` (extend ~800 lines)

Each entity type:
1. Parse parameter data from DE (Directory Entry) pointer
2. Convert to rustcoin3d geometry type (CurveGeom / SurfaceGeom / Curve2d)
3. Add to BRepStore with proper topology

## Algorithm per type

- **102 Composite Curve**: collect child curve entities → `CurveGeom::Composite`
- **108 Plane**: bounded plane → `SurfaceGeom::Plane` + wire from boundary curves
- **120 Revolution**: axis + generatrix curve + angles → `SurfaceGeom::Revolution`
- **122 Tabulated Cylinder**: directrix curve + generatrix direction → `SurfaceGeom::Extrusion`
- **128 BSpline**: control points + knots + weights → `SurfaceGeom::BSpline`
- **144 Trimmed Surface**: basis surface + boundary loops → face with inner/outer wires
- **142 CurveOnSurface**: 3D curve + surface ref → `CurveGeom` + `Curve2d` pair
- **140 Offset Surface**: basis surface + offset distance → `SurfaceGeom::Offset`

## Tests

- Composite curve roundtrip
- BSpline surface import
- Revolution surface import
- Full IGES file with multiple entity types

## Verification

- `cargo check` + `cargo test` pass
- Import sample IGES file → valid BRepStore
