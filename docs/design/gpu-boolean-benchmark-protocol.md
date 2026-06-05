# GPU Boolean Acceleration — Benchmark Protocol

> 2026-06-05 | Companion to `gpu-boolean-acceleration.md`

## Test Models

| Name | Faces | Source | Notes |
|------|-------|--------|-------|
| Cube-Cube | 12 | `make_square_face()` | Baseline trivial case |
| Two Planes | 2 | `make_plane_shell()` | Minimal intersection |
| Gear | ~500 | STEP import | Common mechanical part |
| Engine Block | ~50k | STEP import | Medium assembly |
| Car Body | ~200k | STEP import | Stress test |

## Metrics

Per boolean pipeline phase, measure:

| Phase | Metric | Unit |
|-------|--------|------|
| PaveFiller | Wall time | ms |
| PaveFiller | Face pairs tested | count |
| PaveFiller | Intersections found | count |
| Split | Wall time | ms |
| Split | Faces split | count |
| Classify | Wall time | ms |
| Select | Wall time | ms |
| Stitch | Wall time | ms |
| **Total** | **Wall time** | **ms** |

GPU-specific metrics (when GPU path is active):

| Metric | Unit |
|--------|------|
| GPU buffer upload time | µs |
| GPU dispatch time | µs |
| GPU readback time | µs |
| Candidate pair count | count |
| Candidate pair precision (vs ground truth) | % |
| GPU memory usage | MB |

## Success Criteria

GPU candidate detection must:

1. **Recall**: Find 100% of actual face-face intersections (zero false negatives)
2. **Precision**: Reduce CPU face-pair testing by > 90% vs brute force
3. **Latency**: Total wall time < 50% of CPU-only pipeline for models > 1k faces
4. **Memory**: GPU buffer allocation < 256 MB for 100k-face models

## Benchmark Command

```bash
# CPU baseline
cargo bench --bench boolean_pipeline -- \
    --model <path> \
    --op union \
    --iterations 10

# GPU accelerated (future)
cargo bench --bench boolean_pipeline -- \
    --model <path> \
    --op union \
    --gpu \
    --iterations 10
```

## Current Baseline (2026-06-05)

Measured on existing test suite (cargo test -p rc3d-shape --lib -- bool):

| Test | Faces | Time |
|------|-------|------|
| `test_bool_intersection_two_planes` | 2 | < 1ms |
| `test_boolean_with_proper_edges` | 2 (with edges) | < 1ms |
| `test_boolean_no_intersection_disjoint_union` | 2 | < 1ms |

Full boolean test suite (56 tests): ~0.3s total
