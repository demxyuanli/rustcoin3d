# BREP Export Regression Tests

Last updated: 2026-06-30

## Quick Reference

```bash
# All regression tests (recommended before merge)
cargo test -p rc3d-shape -- seam locations shell_close && cargo test -p rc3d-io -- brep_compare

# Fast feedback loop (< 5s)
cargo test -p rc3d-shape -- seam

# Structural validation (no OCC needed, <1s)
rustc compare/validate_brep.rs -o target/debug/validate_brep.exe
target/debug/validate_brep.exe compare/out/step-*.brep

# OCC topology check (requires Open CASCADE SDK)
# Build instructions in compare/validate_brep_occ.cpp header
# validate_brep_occ.exe compare/out/step-*.brep

# Full comparison (11 files, ~4s)
cargo test -p rc3d-io -- brep_compare_all_files_export_and_check -- --nocapture

# Simple primitives only (6 files, < 1s)
cargo test -p rc3d-io -- brep_compare_simple_primitives -- --nocapture
```

---

## Tier 1 — Critical (must pass)

Run before any BREP-related commit.

### seam::tests (7 tests)
```
cargo test -p rc3d-shape -- seam
```
- `sphere_vertex_loop_gets_seam_edge` — sphere seam creation + idempotency
- `cylinder_trimmed_face_gets_periodic_seam` — seam skipped when outside trim hull
- `cylinder_full_wrap_gets_periodic_seam` — seam added for wrap-around trim
- `(4 others from crate-internal test modules)`

**Covers**: Duplicate BSpline seam prevention (cylinder V=4→2, torus V=5→1).

### locations::tests (1 test)
```
cargo test -p rc3d-shape -- locations
```
- `assign_locations_for_circles_at_different_z` — Z-translation Location for two circles at different heights on different cylindrical faces. Verifies location assigned, vertex_locations populated.

**Covers**: Location transform detection, edges_share_face guard, rigid transform computation.

### shell_close::tests (7 tests)
```
cargo test -p rc3d-shape -- shell_close
```
- `close_open_cylinder_shell_adds_closure_face` — open cylinder gets planar closure
- `torus_like_internal_seam_skipped` — F+R self-loop edges on torus/fully-periodic faces skipped

**Covers**: Shell closure detection, internal-seam filtering, fully-periodic face skip.

### store::tests (12 tests)
```
cargo test -p rc3d-shape -- store
```
- `test_find_or_add_vertex_dedup` / `near_duplicate` / `distinct_beyond_tolerance`
- `test_add_edge_with_pcurve_dedup`
- `test_find_shared_edges`
- `test_set_pcurve_replace` / `test_pcurve_mut` / `test_set_pcurve_new_face`
- `test_vertex_to_edges_index` / `test_dual_index_consistency`
- `test_seam_edge_indices`

**Covers**: Vertex dedup, edge dedup, PCURVE management, index consistency.

### brep_compare_all_files_export_and_check (1 test)
```
cargo test -p rc3d-io -- brep_compare_all_files_export_and_check -- --nocapture
```
11 STEP files → BREP export → structural comparison vs OCC reference.

**Covers**: Full pipeline integration — STEP import, B-Rep build, seam fix, vertex welding, location assignment, shell closure, BREP export.

---

## Tier 2 — Fast Feedback (pre-commit)

### brep_compare_simple_primitives (1 test)
```
cargo test -p rc3d-io -- brep_compare_simple_primitives -- --nocapture
```
6 simple primitives: cube, cylinder, cone, torus, sphere, revolution.

### brep_compare_discover_files (1 test)
```
cargo test -p rc3d-io -- brep_compare_discover_files
```
Verifies STEP file discovery + OCC reference matching in compare/ directory.

---

## Topology Alignment Status

| File | V | E | Fa | vs OCC |
|------|---|---|-----|--------|
| cube | 8 | 12 | 6 | ✓ full match |
| cylinder | 2 | 3 | 3 | ✓ full match |
| holedPlate | 60 | 90 | 32 | ✓ full match |
| OffsetPlaneHoleEdge | 16 | 24 | 10 | ✓ full match |
| Shape | 10 | 18 | 11 | ✓ full match |
| Shape-2 | 54 | 82 | 31 | ✓ full match |
| Sphere | 2 | 3 | 1 | ✓ full match |
| torus | 1 | 2 | 1 | ✓ full match |
| Shape-1 | **61✓** | 127/132 | 60✓ | V matches |
| cone | 2✓ | 2/3✗ | 2/3✗ | OCC reparameterization |
| rev | 8/4✗ | 12/8✗ | 6/5✗ | OCC Locations |

**Acceptable differences** (geometrically correct, watertight):
- cone: STEP has radius_at_apex=0; OCC reparameterizes to 5, adding top plane. Our 2-face cone is watertight (apex is singular point).
- rev: OCC uses 2 Location transforms to share 4 edges across faces. Our 6-face shell is watertight; counts differ due to Location-based edge merging.
- Shape-1 E=127 vs 132: OCC splits more edges.

---

## Key Architectural Decisions

1. **Seam detection** (`seam.rs`): `is_full_circle_3d` prevents duplicate BSpline seams. `wire_has_parametric_seam` treats closed circles as parametric seams.
2. **Vertex welding** (`store.rs`): `weld_vertices(tol)` merges vertices within tolerance. First pass at 1e-4, final pass at 2e-4 after all processing.
3. **Location assignment** (`locations.rs`): `edges_share_face` guard prevents canonicalizing same-face edges (protects cylinder). Only canonicalizes vertices when edges belong to different faces.
4. **Shell closure** (`shell_close.rs`): Fully-periodic surfaces (Torus, Sphere) excluded. Internal seams (F+R in same wire) excluded.
5. **No vertex canonicalization for same-face edges**: Top/bottom circles on a cylinder share the cylindrical face — must stay at their original positions.
