# Phase 6: Performance Analysis + Documentation

Date: 2026-06-17.

## F1: Benchmark harness
**File**: `crates/rc3d-shape/benches/` (new directory)
**Goal**: Add criterion benchmarks for key hot paths.
**Benchmarks**:
- BSpline evaluation (d0/d1/d2) at varying degrees
- Face area computation (plane vs sphere vs BSpline)
- Boolean intersection (cube vs cube)
- Mesh generation (plane face, cylinder face)
- Spatial index find_near (1000 points)
**Verify**: All benchmarks compile and run

## F2: Hot-path profiling
**Goal**: Identify and optimize top 5 bottlenecks.
**Method**: Use the benchmarks from F1 to identify slow paths.
**Common fixes**:
- Replace O(n²) with HashMap lookups where applicable
- Reduce allocations in hot loops (pre-allocate Vecs)
- Cache frequently computed values
**Verify**: Before/after timing comparison

## F3: Crate documentation
**Goal**: Add comprehensive module-level docs to all public modules.
**Files to document**:
- `rc3d-shape/src/geom/` -- curve/surface evaluation
- `rc3d-shape/src/heal/` -- healing pipeline
- `rc3d-shape/src/bool/` -- boolean operations
- `rc3d-io/src/step/` -- STEP I/O
**Verify**: `cargo doc --no-deps --document-private-items` produces no warnings

## F4: Architecture doc
**File**: `docs/architecture.md`
**Goal**: High-level architecture overview of the geometry kernel.
**Sections**: Type system, data flow, module dependencies, OCC alignment table.
