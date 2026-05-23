## 1. Module setup + dependencies

- [x] 1.1 Create `crates/rc3d-io/src/step/` directory and `mod.rs` with StepError + parse entry points
- [x] 1.2 Remove `truck-stepio`, `truck-meshalgo`, `ruststep` from `rc3d-io/Cargo.toml`
- [x] 1.3 Verify build passes with zero external STEP dependencies

## 2. Parser (step-exchange-parser)

- [x] 2.1 Create `value.rs` with `StepValue` enum (Integer, Real, String, Enum, Ref, List, Typed, Omitted)
- [x] 2.2 Implement recursive-descent parser in `parser.rs` — tokenize and parse HEADER + DATA sections
- [x] 2.3 Build `EntityIndex = HashMap<u64, EntityRecord { name, params }>` from parsed data sections
- [x] 2.4 Add parser unit tests: simple entity, nested list, typed param, omitted param, empty data section
- [x] 2.5 Test against real STEP files (faceted cube, B-rep assembly)

## 3. Topology traversal (step-brep-topology)

- [x] 3.1 Implement Shell traversal — iterate `CLOSED_SHELL`/`SHELL` entities, resolve face references
- [x] 3.2 Implement FaceSurface extraction — resolve bounds (FACE_OUTER_BOUND, FACE_BOUND)
- [x] 3.3 Implement EdgeLoop traversal — resolve edge list, unwrap ORIENTED_EDGE
- [x] 3.4 Implement EdgeCurve extraction — resolve edge_start/edge_end + curve geometry reference
- [x] 3.5 Add error resilience — skip individual faces/bounds/edges that fail resolution

## 4. Curve geometry (step-curve-geometry)

- [x] 4.1 Implement LINE evaluation — extract pnt/dir, compute endpoints
- [x] 4.2 Implement CIRCLE evaluation — extract center/radius, sample points in 3D using AXIS2_PLACEMENT_3D
- [x] 4.3 Implement ELLIPSE evaluation — extract semi-axes, sample elliptical arc points
- [x] 4.4 Implement B_SPLINE_CURVE evaluation — construct NurbsCurve from control points + knots, call tessellate()
- [x] 4.5 Add adaptive sampling — segment count proportional to arc length

## 5. Surface geometry (step-surface-geometry)

- [x] 5.1 Implement PLANE surface evaluation — AXIS2_PLACEMENT_3D to 3D parameterization
- [x] 5.2 Implement CYLINDER/CONE/SPHERE/TORUS parametric evaluation with UV-to-3D mapping
- [x] 5.3 Implement B_SPLINE_SURFACE → NurbsSurface construction from control points grid + knot vectors
- [x] 5.4 Handle unsupported surface types gracefully (skip face, continue)

## 6. Face tessellation (step-face-tessellation)

- [x] 6.1 Implement planar face tessellation — extract polygon from edge loop vertices, fan triangulate
- [x] 6.2 Implement curved surface tessellation — UV grid sampling + quad triangulation
- [x] 6.3 Handle face orientation — invert winding when `same_sense=false`
- [x] 6.4 Implement vertex deduplication — hash-based position merging across faces
- [x] 6.5 Output IndexedFaceSet format (triangle triples with -1 sentinel)

## 7. Assembly hierarchy (step-assembly-hierarchy)

- [x] 7.1 Parse NEXT_ASSEMBLY_USAGE_OCCURRENCE → extract product references + transform references
- [x] 7.2 Parse ITEM_DEFINED_TRANSFORMATION → AXIS2_PLACEMENT_3D → compute 4×4 matrix
- [x] 7.3 Build product-to-shape mapping via PRODUCT_DEFINITION_SHAPE chain
- [x] 7.4 Build multi-level assembly tree with nested transform accumulation
- [x] 7.5 Apply transforms to tessellated vertices before SceneGraph insertion
- [x] 7.6 Handle missing transforms with identity fallback

## 8. Scene output + integration

- [x] 8.1 Build SceneGraph with per-component Separator/Transform/Coordinate3/IFS hierarchy
- [x] 8.2 Update `lib.rs` re-exports (no API change)
- [x] 8.3 Build and verify with faceted cube STEP test
- [x] 8.4 Build and verify with B-rep assembly STEP test
- [x] 8.5 Run full workspace tests — all 223+ passing

## 9. Polish

- [x] 9.1 Remove unused code from old step.rs
- [x] 9.2 Run clippy and fix warnings
- [x] 9.3 Smoke test: import_viewer with real assembly STEP file
