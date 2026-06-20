# Phase D: Real CAD File Integration Testing

Date: 2026-06-18.

## Goals

1. Import pipeline end-to-end test (STEP → BRepStore → heal → mesh → export)
2. Roundtrip fidelity (import → parametric write → re-import → compare)
3. Performance benchmarks on production-sized files
4. Quality metrics (mesh deflection, face count, heal pass diagnostics)

## Tasks

### D1: Corpus test framework
**File**: `crates/rc3d-io/tests/cad_corpus.rs` (new)
**Goal**: Parameterized test runner for a directory of STEP files.
```rust
fn corpus_roundtrip(file: &str) {
    // 1. Import STEP
    // 2. Heal all shells (Standard tier)
    // 3. Mesh (Standard tier)
    // 4. Parametric write
    // 5. Re-import
    // 6. Compare face/edge/vertex counts
}
```

### D2: Roundtrip fidelity metrics
**Goal**: Verify parametric STEP writer produces valid output.
Metrics: entity count match, surface type preservation, curve type preservation.

### D3: Performance profiling
**Goal**: Measure import + heal + mesh time for representative files.
Output: per-phase timing, memory usage.

### D4: Mesh quality validation
**Goal**: Verify mesh quality meets production thresholds.
Metrics: chordal deflection < configured tolerance, no degenerate tris, watertight boundaries.

## Success criteria

- [ ] At least 5 STEP files processed with 0 pipeline errors
- [ ] Roundtrip: face/edge count preserved within 5%
- [ ] Heal: all files heal with ≤ 2 iterations
- [ ] Mesh: all faces produce ≥ 1 triangle
- [ ] No panics, no unwraps triggered

## Test files

Use existing `test_output/brep/` and `steps/` directories:
- Cube.step, cs.step, Shape.step, Shape-1.step, Shape-2.step
- Rev.step, OffsetPlaneHoleEdge.step
- Any other STEP files found in test data
