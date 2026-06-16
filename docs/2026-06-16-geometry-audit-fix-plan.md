# Geometry Audit Fix Plan

Based on: rustcoin3d vs OCCT 7.8.0 cross-audit (5 probes, 30+ findings).
Date: 2026-06-16

## Strategy

**Path A**: `type Real = f64` alias in geometry kernel, `DVec3`/`DMat4` for geom math,
`Vec3`/`Mat4` (f32) retained for GPU rendering. Cast at render boundary.

## Phases (execution order matters — each phase depends on prior)

### Phase 0: f32→f64 Migration (Foundation)

**Scope**: `rc3d-core` math types + `rc3d-shape` geometry kernel (~1,700 f32 occurrences, ~145 public fns)

1. **rc3d-core**: Introduce `type Real = f64`, `type PVec3 = glam::DVec3`, `type PMat4 = glam::DMat4`
2. **rc3d-core**: Keep `Vec3`/`Mat4` (f32) for render crate use
3. **rc3d-shape**: Replace all `f32` → `Real`, `Vec3` → `PVec3`, `Mat4` → `PMat4`
4. **rc3d-shape**: Update all public API signatures (`pub fn.*f32` → `pub fn.*Real`)
5. **rc3d-shape**: Update tolerance constants in `tolerance.rs`
6. **rc3d-io/step**: Update STEP I/O to use `Real`/`PVec3`
7. **rc3d-render**: Add `as_vec3()`/`as_mat4()` cast helpers at GPU upload boundary
8. **All crates**: Update tests — adjust numerical tolerances for f64 precision
9. **Cargo.toml**: Add `glam` with `features=["f64"]` as `glam64` package alias

**Verify**: `cargo check --workspace` + `cargo test` pass (expect tolerance tuning)

### Phase 1: Boolean Critical Fixes

**P0-1: CommonBlock consumption in builder_face.rs**
- Read `bopds.common_blocks` in `detect_edge_split_points`
- Wire CommonBlock edge groups into face wire loop construction
- Shared edges across adjacent faces → single edge in result topology
- **Verify**: Multi-face boolean test produces watertight result

**P0-2: Plane-cylinder general case intersection**
- Implement ellipse intersection for oblique plane-cylinder cuts
- Replace `None` return with computed ellipse curve
- **Verify**: Plane-cylinder at 45° angle produces correct intersection curve

### Phase 2: Geometry/Heal Correctness

**P1-3: Shell orientation multi-point sampling**
- Replace single midpoint sample with N≥5 samples along shared edge
- Use topological propagation as primary, normal check as tiebreaker
- **Verify**: Sphere/torus shells orient correctly

**P1-4: B-spline→Bezier decomposition fix**
- Insert internal knots to full multiplicity before extracting Bezier segments
- Implement proper `insert_knot` loop per internal knot value
- **Verify**: Multi-span cubic B-spline decomposes to correct Bezier curves

**P1-5: Volume integral orient parameter**
- Replace `face.same_sense` with shell-level `_orient` parameter
- Apply shell face orientation to flux sign
- **Verify**: Mixed-orientation shell volume matches analytical

### Phase 3: Secondary Fixes

**P2-6: Wire reorder — replace XOR hash with robust vertex matching**
- Use quantized position comparison with tolerance instead of XOR hash
- Add collision detection fallback (distance check when hash matches)

**P2-7: BSpline fit — improve numerical stability**
- Add iterative refinement step after Gaussian elimination
- Raise pivot threshold or switch to SVD for ill-conditioned systems

**P2-8: NaN handling — replace silent zero with error**
- Return `Option<Vec3>` or `Result` from B-spline eval instead of NaN→zero
- Propagate errors to callers

### Phase 4: Validation

- Run full test suite (target: 400+ tests, 0 failures)
- Run BREP roundtrip test
- Run boolean regression tests
- Run mesh pipeline on test STEP files

## Excluded (deferred to future cycles)

- CSG STEP entity support
- Parametric STEP writer
- IGES/VRML format support
- Binary persistence format
- Incremental mesh reuse
- ModelHealer (gap/overlap repair)
- FaceInfo state tracking in BOPDS
- VV/VE/VF/EE interference detection
- BOPAlgo_BuilderSolid (solid reconstruction)
- UnifySameDomain / DivideContinuity
- FixSmallFace / FixSmallSolid

## Success Criteria

- [ ] `cargo check --workspace` — 0 errors
- [ ] `cargo test` — all tests pass (tolerance-adjusted)
- [ ] f32→f64 migration complete with clean render boundary
- [ ] CommonBlock consumed in boolean face building
- [ ] Plane-cylinder general case produces correct ellipse
- [ ] Shell orientation works on curved surfaces
- [ ] B-spline→Bezier decomposition geometrically correct
- [ ] Volume integral uses proper shell orientation
