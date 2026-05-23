## Context

The engine currently uses `truck-stepio` (0.3) for STEP entity parsing and `truck-meshalgo` (0.4) for B-rep tessellation. These crates from ricosjp/truck are experimental (pre-1.0), lack assembly entity types, and create a dependency risk for future API changes. The STEP import pipeline must be brought in-house with zero external dependencies for long-term maintainability.

Current state: `crates/rc3d-io/src/step.rs` (112 lines) delegates to truck for all parsing and tessellation. Assembly transforms are not extracted, causing all components to render at origin.

## Goals / Non-Goals

**Goals:**
- Zero external STEP dependencies (remove `ruststep`, `truck-stepio`, `truck-meshalgo`)
- Self-developed ISO 10303-21 exchange structure parser
- B-rep topology traversal: Shell → FaceSurface → EdgeLoop → EdgeCurve
- Curve sampling for LINE, CIRCLE, ELLIPSE, B_SPLINE_CURVE_WITH_KNOTS
- Surface evaluation for PLANE, CYLINDER, CONE, SPHERE, TORUS, B_SPLINE_SURFACE
- Face tessellation into triangle meshes (polygon extraction for planar, UV sampling for curved)
- Assembly hierarchy reconstruction with AXIS2_PLACEMENT_3D transforms
- Proper SceneGraph output with Separator/Transform/Coordinate3/IndexedFaceSet per component

**Non-Goals:**
- Boolean operations on solids
- Exact trimmed surface tessellation (constraint Delaunay) — fan triangulation is acceptable
- STEP serialization (write)
- AP validation (accept any valid exchange structure)
- XML-encoded STEP (Part 28) — ASCII only

## Decisions

### 1. Parser: Recursive descent over nom

**Decision**: Hand-written recursive descent parser with `StepValue` enum.

**Rationale**: STEP syntax is trivial (`#ID = NAME(p1, p2);` with nested lists). A recursive descent parser is ~200 lines, has no dependencies, produces clear error messages, and is easier to debug than `nom` combinator chains. The `nom` crate itself has future-compatibility warnings (nom 3.x rejected by future Rust).

**Alternative considered**: Use `nom` for parser combinators. Rejected due to dependency weight, future-compat warnings, and overkill for the grammar complexity.

### 2. Entity model: Flat HashMap vs typed tables

**Decision**: Single `HashMap<u64, EntityRecord>` (ID → name + parameter list).

**Rationale**: truck-stepio uses a typed `Table` struct with 43 separate HashMaps (one per entity type). This requires schema knowledge at compile time. A single flat map is schema-agnostic — works for any AP (203/214/242) without recompilation. Entity type checking happens at access time via string comparison.

**Alternative considered**: Typed entity tables like truck-stepio. Rejected due to coupling with specific AP schemas.

### 3. Face tessellation: Fan triangulation vs Delaunay

**Decision**: Fan triangulation for all faces. No constraint Delaunay.

**Rationale**: Fan triangulation is correct for convex faces (~90% of mechanical CAD faces). Non-convex faces with holes will have visual artifacts, but this is an acceptable initial tradeoff. Constraint Delaunay (via `spade` or self-implemented) would add ~400 lines and significant debugging effort.

**Alternative considered**: Use `spade` crate for constraint Delaunay. Rejected to minimize dependencies. Can be added later if needed.

### 4. Surface tessellation: Uniform UV sampling vs adaptive

**Decision**: Uniform UV sampling (32×32 grid) for elementary surfaces (CYLINDER, CONE, SPHERE, TORUS). Use `rc3d_nurbs::NurbsSurface::tessellate_adaptive` for B_SPLINE surfaces.

**Rationale**: Elementary surfaces have simple parameterizations where uniform sampling produces good results. B_SPLINE surfaces benefit from the existing adaptive tessellator in `rc3d-nurbs`.

### 5. Module structure: directory vs single file

**Decision**: `crates/rc3d-io/src/step/` directory with 7 focused files.

**Rationale**: The module is ~1200 lines across 7 concerns (parsing, values, topology, geometry, tessellation, assembly, scene). A single file would be unwieldy. Each file has one clear responsibility and can be tested independently.

## Risks / Trade-offs

**[Risk] Fan triangulation produces artifacts on non-convex faces with holes**
→ Mitigation: Acceptable initial tradeoff. Can upgrade to constraint Delaunay later without changing the module API.

**[Risk] B_SPLINE surface evaluation may be slow for large models**
→ Mitigation: `rc3d-nurbs` already has adaptive tessellation with tolerance control. Tune tolerance based on model scale.

**[Risk] Assembly hierarchy may differ between CAD exporters**
→ Mitigation: Use defensive matching on entity names. Handle missing transforms gracefully (identity fallback). Test against FreeCAD, SolidWorks, and Fusion 360 exports.

**[Risk] STEP file encoding (UTF-8 vs Latin-1) may cause parse failures**
→ Mitigation: Read file as bytes, detect encoding from header, convert to string. Most modern STEP files are ASCII/UTF-8.
