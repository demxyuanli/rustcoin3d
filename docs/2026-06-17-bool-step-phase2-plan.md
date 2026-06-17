# Phase 2: Boolean + STEP 覆盖

Based on: OCCT audit deferred items. Date: 2026-06-17.

## Group A: Boolean completion

### A1: VV (Vertex-Vertex) interference
**File**: `bool/pave_filler.rs` (+ `bopds.rs`)
**Goal**: Detect coincident vertices, build SameDomain (SD) map.
**Algorithm**:
1. Collect all vertices from both shells into spatial index (SpatialIndexF64)
2. For each pair within tolerance → record as SameDomain pair
3. Store in `BopDS.sd_vertices: HashMap<VertexKey, VertexKey>` (canonical mapping)
4. Use SD map in downstream EF/VF passes to unify vertex references
**OCC ref**: `BOPAlgo_PaveFiller_1.cxx:PerformVV()`
**Verify**: Two cubes touching at a corner → 1 SD pair detected

### A2: EE (Edge-Edge) interference
**File**: `bool/pave_filler.rs`
**Goal**: Wire existing `intersect_edge_edge()` into pave_filler.
**Algorithm**:
1. For each edge in shell A vs each edge in shell B
2. AABB quick reject → `intersect_edge_edge()` → EdgeEdgeHit list
3. Each hit → create InterfPoint → FaceFaceInterf with both faces
4. Already have `intersect_edge_edge()` at `intersect_edge.rs:167`
**OCC ref**: `BOPAlgo_PaveFiller_3.cxx:PerformEE()`
**Verify**: Two edges crossing at a point → 1 intersection detected

### A3: VE (Vertex-Edge) interference
**File**: `bool/pave_filler.rs`
**Goal**: Detect vertices of shell A lying on edges of shell B.
**Algorithm**:
1. For each vertex in shell A (not in SD map already)
2. For each edge in shell B
3. Project vertex onto edge curve → if distance < tol → hit
4. Create split vertex on edge → new PaveBlock
**OCC ref**: `BOPAlgo_PaveFiller_2.cxx:PerformVE()`
**Verify**: Vertex at (5,0,0) on edge from (0,0,0) to (10,0,0) → 1 hit

### A4: VF (Vertex-Face) interference
**File**: `bool/pave_filler.rs`
**Goal**: Detect vertices of shell A lying on faces of shell B.
**Algorithm**:
1. For each vertex in shell A (not in SD map already)
2. For each face in shell B
3. Project vertex onto face surface → if distance < tol AND UV inside face bounds → hit
4. Record IN/ON state for FaceInfo
**OCC ref**: `BOPAlgo_PaveFiller_4.cxx:PerformVF()`
**Verify**: Vertex at (3,4,0) on plane z=0 → 1 hit

### A5: FaceInfo
**File**: `bool/bopds.rs`
**Goal**: Track per-face split-vertex and state data.
**Add to BopDS**:
```rust
pub struct FaceInfo {
    pub split_vertices: Vec<VertexKey>,  // vertices from VF/EF on this face
    pub pave_blocks_on: Vec<PaveBlock>,  // pave blocks IN/ON this face
    pub face_state: FaceState,           // In/Out/On classification
}
pub enum FaceState { In, Out, On }
pub face_infos: HashMap<FaceKey, FaceInfo>,
```
**OCC ref**: `BOPDS_FaceInfo`

### A6: BuilderSolid
**File**: `bool/builder_solid.rs` (new)
**Goal**: Reconstruct solid topology from split faces.
**Algorithm**:
1. Collect all split faces from BuilderFace
2. Group faces into closed shells (edge adjacency)
3. For each shell: test if closed (each edge shared by exactly 2 faces)
4. Select shell with largest volume → outer shell; rest → void shells
5. Assemble into BRepSolid
**OCC ref**: `BOPAlgo_BuilderSolid.cxx`
**Verify**: Boolean union of two cubes → 1 solid with correct face count

## Group B: STEP coverage + parametric writer

### B1: CSG entity types
**File**: `step/entity_types.rs`
**Goal**: Add missing CSG/swept/faceted entity type variants.
**Add to EntityType enum**:
```
MANIFOLD_SOLID_BREP (already exists)
BREP_WITH_VOIDS (already exists)
FACETED_BREP, FACETED_BREP_AND_BREP_WITH_VOIDS
BLOCK, RIGHT_CIRCULAR_CYLINDER, RIGHT_CIRCULAR_CONE, SPHERE, TORUS, RIGHT_ANGULAR_WEDGE
HALF_SPACE_SOLID, BOXED_HALF_SPACE
BOOLEAN_RESULT, CSG_SOLID
EXTRUDED_AREA_SOLID, REVOLVED_AREA_SOLID
EXTRUDED_FACE_SOLID, REVOLVED_FACE_SOLID
SHELL_BASED_SURFACE_MODEL
CONNECTED_FACE_SET, CONNECTED_EDGE_SET
GEOMETRIC_CURVE_SET, GEOMETRIC_SET
```
**Verify**: Parse STEP with Block/Cylinder → entity type recognized

### B2: CSG builder
**File**: `step/brep/build/csg.rs` (new)
**Goal**: Convert CSG primitives to B-Rep faces.
**Algorithm**:
1. BLOCK → 6 rectangular faces (Plane surfaces)
2. CYLINDER → 1 cylindrical face + 2 planar end caps
3. CONE → 1 conical face + 1-2 planar end caps
4. SPHERE → 1 spherical face
5. TORUS → 1 toroidal face
6. WEDGE → 5 planar faces
7. EXTRUDED_AREA_SOLID → extrude profile face along vector
8. BOOLEAN_RESULT → recursive: build operands, then apply boolean op
**OCC ref**: `BRepPrimAPI_Make*` classes
**Verify**: Import STEP with Block entity → 6 faces in BRepStore

### B3: Parametric STEP writer
**File**: `step/write/parametric.rs` (new)
**Goal**: Write BRepStore topology as exact parametric STEP entities.
**Algorithm**:
1. Walk BRepStore shells/faces/edges/vertices
2. For each SurfaceGeom variant → emit corresponding STEP surface entity:
   - Plane → PLANE (AXIS2_PLACEMENT_3D)
   - Cylinder → CYLINDRICAL_SURFACE
   - Cone → CONICAL_SURFACE
   - Sphere → SPHERICAL_SURFACE
   - Torus → TOROIDAL_SURFACE
   - BSpline → B_SPLINE_SURFACE_WITH_KNOTS
   - Extrusion → SURFACE_OF_LINEAR_EXTRUSION
   - Revolution → SURFACE_OF_REVOLUTION
3. For each CurveGeom variant → emit corresponding STEP curve entity
4. Assembly: MANIFOLD_SOLID_BREP → CLOSED_SHELL → ADVANCED_FACE → FACE_OUTER_BOUND → EDGE_LOOP
5. Write ISO 10303-21 header + DATA section
**OCC ref**: `STEPControl_Writer`, `TopoDSToStep`
**Verify**: Roundtrip: import Cube.step → BRepStore → write → re-import → same geometry

## Execution order (dependency chain)

```
A1 VV ─┐
        ├─→ A3 VE ─→ A4 VF ─→ A5 FaceInfo ─→ A6 BuilderSolid
A2 EE ─┘
(independent)
        B1 entity types ─→ B2 CSG builder
B3 parametric writer (independent)
```

## Success criteria
- [ ] `cargo check --workspace` — 0 errors
- [ ] `cargo test` — all tests pass (≥404 pass)
- [ ] VV: coincident vertex pairs detected
- [ ] EE: edge-edge intersections found by pave_filler
- [ ] VE: vertex-on-edge hits detected
- [ ] VF: vertex-on-face hits detected
- [ ] FaceInfo populated with split vertices
- [ ] BuilderSolid: solid reconstruction from split faces
- [ ] CSG entities recognized in STEP parser
- [ ] CSG builder: Block/Cylinder/Sphere → B-Rep faces
- [ ] Parametric writer: roundtrip preserves geometry identity
