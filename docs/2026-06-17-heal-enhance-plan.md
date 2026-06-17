# Phase 3: Heal Enhancements

Based on: OCCT audit — ShapeFix/ShapeUpgrade/ShapeAnalysis gaps.
Date: 2026-06-17.

## C1: UnifySameDomain
**OCC**: `ShapeUpgrade_UnifySameDomain`
**Goal**: Merge adjacent faces that share the same geometry into single faces.
**Algorithm**:
1. Build face adjacency graph (shared edges)
2. For each adjacent pair:
   - Check if surfaces are geometrically identical (same type, same parameters)
   - Check if edge between them is not a sharp feature (G1 continuity)
3. Merge qualifying face pairs: combine wires, remove shared edge, rebuild larger face
4. Update shell: replace merged faces with new unified face
**File**: `heal/unify_same_domain.rs` (new)
**Verify**: Two coplanar adjacent faces → 1 merged face

## C2: FixSmallFace
**OCC**: `ShapeFix_FixSmallFace`
**Goal**: Detect faces with area below threshold and merge into neighbors.
**Algorithm**:
1. Compute face area via existing `face_area()`
2. If area < threshold: find best neighbor (largest shared edge)
3. Merge small face into neighbor: absorb wire, remove shared edge
4. Remove small face from shell
**File**: `heal/face_fix.rs` (extend existing)
**Verify**: Face of area < 0.01 → merged into adjacent face

## C3: FixSmallSolid
**OCC**: `ShapeFix_FixSmallSolid`
**Goal**: Remove solids with volume below threshold.
**Algorithm**:
1. Compute solid volume via existing `solid_volume()`
2. If volume < threshold: remove solid from document
3. Optionally merge void shells into adjacent solids
**File**: `heal/shell_fix.rs` (extend) or new `heal/solid_fix.rs`
**Verify**: Solid with volume < 0.001 → removed

## C4: ModelHealer
**OCC**: `BRepMesh_ModelHealer`
**Goal**: Repair mesh gaps and overlaps between adjacent faces.
**Algorithm**:
1. After meshing, detect boundary edges that are not shared
2. For each gap: find closest boundary vertex on adjacent face
3. Weld vertices within tolerance
4. Remove T-junctions by splitting edges at junction points
**File**: `mesh/post_process.rs` (extend existing stub)
**Verify**: Two meshes with 0.001 gap → gap closed after heal

## Execution order

C1 (UnifySameDomain) → C2 (FixSmallFace) → C3 (FixSmallSolid) → C4 (ModelHealer)

C1 and C4 are independent. C2 depends on C1 (unified faces don't need merge).
C3 depends on C2 (merged faces change solid volume).

## Success criteria
- [ ] `cargo check --workspace` — 0 errors
- [ ] `cargo test` — all tests pass (≥430)
- [ ] C1: adjacent coplanar faces merged
- [ ] C2: small faces removed
- [ ] C3: tiny solids removed
- [ ] C4: mesh gaps closed after heal
