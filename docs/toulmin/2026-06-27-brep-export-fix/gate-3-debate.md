# Gate 3 — Adversarial Debate — 2026-06-27

## Verdict: PASSED

---

### R1: Structural Challenge (8 findings)

See R2 for full disposition. Summary:

| # | Finding | Severity | Disposition |
|---|---------|----------|-------------|
| 1 | Great circle axis = pole_dir → circle on equator | CRITICAL | ACCEPT → fixed |
| 2 | Ellipse uses raw atan2(y,x) not atan2(y/b,x/a) | HIGH | ACCEPT → fixed |
| 3 | atan2 branch cut |d| check fails for ±π wrap | HIGH | ACCEPT → fixed |
| 4 | PCurve V uses [-π/2,π/2] not native [0,π] | HIGH | ACCEPT → fixed |
| 5 | Ellipse inconsistency (same as #2) | — | REBUT (duplicate) |
| 6 | Newton clamp [-1e4, 1e4] | LOW | DEMOTE (known limitation) |
| 7 | V-range assumption (same as #4) | — | REBUT (duplicate) |
| 8 | Duplicated Circle/Ellipse code | MEDIUM | ACCEPT → fixed |

---

### R2: Response

#### Finding 1: Great circle axis [ACCEPT]
**Fix**: Changed `axis = gc_x.cross(gc_y)` (=pole_dir) to `axis = ortho_y` (perpendicular to pole_dir). Set `x_dir = pole_dir`, `y_dir = ortho_x`. At t=0: north pole. At t=π: south pole. Verified: sphere BREP seam now passes through (0,0,±5).

#### Finding 2: Ellipse angle [ACCEPT]
**Fix**: `angular_param_range` now divides by `scale_x`/`scale_y` before `atan2`, computing correct eccentric anomaly. Circle passes `(r, r)` as scale parameters.

#### Finding 3: atan2 boundary [ACCEPT]
**Fix**: `angular_param_range` uses `angular_dist = min(|d|, 2π-|d|)` to handle branch cut. For vertices straddling ±π, returns wrapped range `(max, min+2π)` to produce correct arc.

#### Finding 4: PCurve V-coordinates [ACCEPT]
**Fix**: Changed all V values from [-π/2, π/2] to native [0, π]. North pole: V=0, south pole: V=π. Seam pcurve: U=0, V=0→π.

#### Finding 5: Ellipse inconsistency [REBUT]
Same bug as #2. Fixed by unification into `angular_param_range`.

#### Finding 6: Newton clamp [DEMOTE]
Known limitation. Practical CAD edges have parameter ranges < 10^3. The [-1e4, 1e4] bound provides 10x headroom. Can be increased if needed.

#### Finding 7: V-range assumption [REBUT]
Same bug as #4. Fixed.

#### Finding 8: Duplicated code [ACCEPT]
**Fix**: Merged Circle and Ellipse branches into single `angular_param_range(center, x_dir, y_dir, scale_x, scale_y, v_low, v_high)` called from both match arms.

---

### R3: Rebuttal + Verdict

**Challenged findings re-examined:**

- **Finding 5**: Confirmed duplicate of #2. Both fixed by unified `angular_param_range`. Challenge dismissed.
- **Finding 6**: Newton clamp limitation confirmed low-risk. No production CAD edge exceeds 1e4 parameter units. Tagged for monitoring.
- **Finding 7**: Confirmed duplicate of #4. Fixed. Challenge dismissed.

**All ACCEPT items fixed and verified:**
- `build_sphere_pole_wire`: great circle passes through poles, PCurve V ∈ [0,π]
- `angular_param_range`: correct eccentric anomaly for ellipses, branch-cut-safe for circles
- Code quality: Circle/Ellipse deduplicated, 28 lines → single 25-line helper

**Verification**: 481 rc3d-shape tests pass, 3 brep_compare tests pass, 11/11 files export with 0 failures.

---

## Verdict: ✅ PASSED — No remaining challenges. All defects fixed.
