# Subsuper Adapter + BSpline 参数扫描 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Shape-2 BSpline surfaces being rendered as flat planes by correcting the subsuper adapter's primary-type selection and replacing hardcoded NURBS parameter offsets with scanning.

**Architecture:** 3 files — `subsuper.rs` (priority-based record selection), `nurbs_build.rs` (adaptive param scanning), `adapter/mod.rs` (updated test). No new files.

**Tech Stack:** Rust, STEP Part 21 parsing, B-spline (NURBS) construction

---

### Task 1: Priority-Based Record Selection in Subsuper Adapter

**Files:**
- Modify: `crates/rc3d-io/src/step/adapter/subsuper.rs:102-123` (`select_primary_record_structured`)
- Modify: `crates/rc3d-io/src/step/adapter/mod.rs:116-138` (updated test expectation)
- Test: existing: `adapter_picks_bspline_with_knots`, `test_parse_subsuper_*`

- [ ] **Step 1: Write a failing test for the priority-based selection**

Add a new test to `crates/rc3d-io/src/step/adapter/mod.rs` tests module:

```rust
#[test]
fn adapter_picks_bspline_surface_from_subsurface_wrapper() {
    // Simulates Shape-2 pattern: BOUNDED_SURFACE wrapping B_SPLINE_SURFACE
    let input = "ISO-10303-21;\nHEADER;ENDSEC;\nDATA;\n#1 = (\n\
BOUNDED_SURFACE()\nB_SPLINE_SURFACE(6,10,(#10,#11))\n\
);
ENDSEC;
END-ISO-10303-21;
";
    let exchange = read_exchange(input).expect("part21 read");
    let idx = to_entity_index(&exchange, &AdapterOptions::compat_merge()).expect("adapter");
    let e = idx.get(&1).expect("#1");
    // Should pick B_SPLINE_SURFACE (concrete type), not BOUNDED_SURFACE (wrapper)
    assert_eq!(e.name, "B_SPLINE_SURFACE");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk cargo test -p rc3d-io --lib adapter_picks_bspline_surface -- --nocapture`
Expected: FAIL — currently returns "BOUNDED_SURFACE" or "REPRESENTATION_ITEM" from leaf_index

- [ ] **Step 3: Implement priority-based selection**

In `crates/rc3d-io/src/step/adapter/subsuper.rs`, replace `select_primary_record_structured`:

```rust
use crate::step::primary_keyword::PRIORITY_TYPES;

/// Select the primary record from structured (keyword, StepValue) pairs.
///
/// 1. Priority match: find the highest-priority PRIORITY_TYPES record whose
///    first param is Integer or Ref (structural types like B_SPLINE_SURFACE,
///    PLANE etc. — excludes W_KNOTS which has only knot data as first param).
/// 2. Fallback: use the parser-assigned leaf_index.
fn select_primary_record_structured(
    pairs: &[(String, &StepValue)],
    leaf_index: usize,
) -> Result<(usize, String), String> {
    if pairs.is_empty() {
        return Err("no records".to_string());
    }
    if pairs.len() == 1 {
        return Ok((0, pairs[0].0.clone()));
    }

    // Rule 1-2: priority-based with structural check
    for prio_type in PRIORITY_TYPES {
        for (i, (name, params)) in pairs.iter().enumerate() {
            if name.as_str() != *prio_type {
                continue;
            }
            // Structural check: first param must be Integer (degree) or Ref (placement)
            let has_struct = params
                .as_list()
                .and_then(|l| l.first())
                .map_or(false, |v| {
                    matches!(v, StepValue::Integer(_) | StepValue::Ref(_))
                });
            if has_struct {
                return Ok((i, name.clone()));
            }
        }
    }

    // Rule 3: fallback to leaf_index
    let idx = leaf_index.min(pairs.len() - 1);
    Ok((idx, pairs[idx].0.clone()))
}
```

- [ ] **Step 4: Update existing test expectations**

In `adapter/mod.rs`, the test `adapter_picks_bspline_with_knots` currently expects `REPRESENTATION_ITEM` (line 130). With priority-based selection, B_SPLINE_CURVE is the first structural PRIORITY_TYPE match. Update:

```rust
// Before:
assert_eq!(e.name, "REPRESENTATION_ITEM");
// After:
assert_eq!(e.name, "B_SPLINE_CURVE");
```

In `parser.rs`, the test `test_parse_subsuper_multiline_ap242` also expects `REPRESENTATION_ITEM`:

```rust
// Before:
assert_eq!(e.name, "REPRESENTATION_ITEM");
// After:
assert_eq!(e.name, "B_SPLINE_CURVE");
```

- [ ] **Step 5: Check `instance.rs` mapping consistency**

Verify that `parse_subsuper_external_prefix` still sorts alphabetically and produces `External` mapping. If the user/linter has changed this to `Internal`, ensure tests match. Read the current state first:

Run: `grep -n "sort_records\|External\|Internal.*leaf" crates/rc3d-io/src/step/part21/instance.rs`
If `External` → keep alphabetical sort. If `Internal` → records in original STEP order.
Priority-based selection works correctly with EITHER ordering.

- [ ] **Step 6: Run all lib tests**

Run: `rtk cargo test -p rc3d-io --lib`
Expected: All tests pass (including the new test and updated expectations)

- [ ] **Step 7: Commit**

```bash
git add crates/rc3d-io/src/step/adapter/subsuper.rs \
        crates/rc3d-io/src/step/adapter/mod.rs \
        crates/rc3d-io/src/step/parser.rs
git commit -m "fix(step): priority-based subsuper record selection

Use PRIORITY_TYPES with structural check (Integer/Ref first param)
to identify concrete geometric types in subsuper entities.
Fixes BOUNDED_SURFACE() wrapping B_SPLINE_SURFACE being named
after the wrapper instead of the concrete surface type."
```

---

### Task 2: Adaptive Parameter Scanning in NURBS Builder

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/geom/nurbs_build.rs:44-56` (knot extraction)
- Test: existing tests cover this path

- [ ] **Step 1: Add a test that verifies merged-param BSpline surface construction**

Add to `nurbs_build.rs` tests (or to `surface.rs` tests):

```rust
#[test]
fn test_nurbs_from_merged_subsurface_params() {
    use crate::step::parser::EntityRecord;
    use crate::step::value::StepValue;

    // Simulate CompatMerge output for Shape-2 entity #40:
    // B_SPLINE_SURFACE(6,10,CPs) + B_SPLINE_SURFACE_WITH_KNOTS(mults,knots)
    // where the closed flags are omitted (common in STEP files)
    let params = StepValue::List(vec![
        StepValue::Integer(6),   // degree_u
        StepValue::Integer(10),  // degree_v
        // control_points: 7x11 grid of refs
        StepValue::List(vec![
            StepValue::List(vec![StepValue::Ref(100); 11]);
            7
        ]),
        // u_multiplicities
        StepValue::List(vec![StepValue::Integer(4), StepValue::Integer(4)]),
        // v_multiplicities
        StepValue::List(vec![StepValue::Integer(4), StepValue::Integer(4)]),
        // u_knots
        StepValue::List(vec![
            StepValue::Real(0.0), StepValue::Real(0.0),
            StepValue::Real(0.33), StepValue::Real(0.67),
            StepValue::Real(1.0), StepValue::Real(1.0),
        ]),
        // v_knots
        StepValue::List(vec![
            StepValue::Real(0.0), StepValue::Real(0.0),
            StepValue::Real(0.33), StepValue::Real(0.67),
            StepValue::Real(1.0), StepValue::Real(1.0),
        ]),
    ]);

    // Build entity from mock data — need cartesian points in entity index
    // This test requires a real EntityIndex with the referenced points.
    // Skip for now — the existing export_step_stl test covers this path.
}
```

Since a proper unit test requires a real EntityIndex with cartesian points, use the **existing integration test** (`export_step_stl`) as the validation gate.

- [ ] **Step 2: Implement `scan_knot_data` helper**

Add to `nurbs_build.rs`, before `build_nurbs_surface`:

```rust
/// Scan merged params for knot multiplicity and value lists.
/// Used when the hardcoded position (off+7) fails because CompatMerge
/// produces a different layout than standalone B_SPLINE_SURFACE_WITH_KNOTS.
fn scan_knot_data(
    params: &StepValue,
    cp_u: usize,
    cp_v: usize,
) -> Option<(Vec<i64>, Vec<i64>, Vec<StepValue>, Vec<StepValue>)> {
    let list = params.as_list()?;
    let mut past_cps = false;
    let mut int_lists: Vec<&[StepValue]> = Vec::new();
    let mut real_lists: Vec<&[StepValue]> = Vec::new();

    for val in list {
        match val {
            StepValue::List(inner) if !inner.is_empty() => {
                // Detect the control point list: List of Lists of Refs
                if inner.iter().all(|v| matches!(v, StepValue::List(_))) {
                    past_cps = true;
                    continue;
                }
                if !past_cps {
                    continue;
                }
                if inner.iter().all(|v| matches!(v, StepValue::Integer(_))) {
                    int_lists.push(inner.as_slice());
                } else if inner.iter().all(|v| matches!(v, StepValue::Real(_))) {
                    real_lists.push(inner.as_slice());
                }
            }
            _ => {}
        }
    }

    // Expect at least 2 int lists (u_mults, v_mults) and 2 real lists (u_knots, v_knots)
    if int_lists.len() < 2 || real_lists.len() < 2 {
        return None;
    }

    let u_mults: Vec<i64> = int_lists[0].iter()
        .filter_map(|v| v.as_int()).collect();
    let v_mults: Vec<i64> = int_lists[1].iter()
        .filter_map(|v| v.as_int()).collect();
    let u_knot_vals: Vec<StepValue> = real_lists[0].to_vec();
    let v_knot_vals: Vec<StepValue> = real_lists[1].to_vec();

    Some((u_mults, v_mults, u_knot_vals, v_knot_vals))
}
```

Add `as_int()` helper if not available on `StepValue`:
```rust
impl StepValue {
    fn as_int(&self) -> Option<i64> {
        match self {
            StepValue::Integer(n) => Some(*n),
            _ => None,
        }
    }
}
```

Check if `as_int()` already exists on StepValue — search the codebase. If not, add it inline in nurbs_build.rs or use pattern matching directly.

- [ ] **Step 3: Modify `build_nurbs_surface` to use scanning fallback**

Replace the knot extraction section (lines ~44-56):

```rust
// Extract knot vectors — try hardcoded position first, scan as fallback
let mult_base = off + 7;
let u_mults = geom::nth_list_ints(params, mult_base);
let v_mults = geom::nth_list_ints(params, mult_base + 1);
let u_knot_vals = geom::nth_list_reals(params, mult_base + 2);
let v_knot_vals = geom::nth_list_reals(params, mult_base + 3);

let (u_mults, v_mults, u_knot_vals, v_knot_vals) =
    if u_mults.is_empty() || v_mults.is_empty() {
        // Hardcoded position failed (CompatMerge layout differs from standalone).
        // Scan the parameter list for knot data.
        if let Some((um, vm, uk, vk)) = scan_knot_data(params, rows, cols) {
            (um, vm, uk, vk)
        } else {
            (u_mults, v_mults, u_knot_vals, v_knot_vals) // use defaults
        }
    } else {
        (u_mults, v_mults, u_knot_vals, v_knot_vals)
    };
```

- [ ] **Step 4: Add defensive validation**

After knot and weight construction, validate dimensions:

```rust
// Guard: knot vectors must have correct length for valid B-spline
let needed_u = rows + degree_u + 1;
let needed_v = cols + degree_v + 1;
if knots_u.len() < needed_u || knots_v.len() < needed_v {
    return None;
}

// Guard: weights must match control point dimensions
if weights.len() != rows || weights.first().map(|r| r.len()).unwrap_or(0) != cols {
    return None;
}
```

- [ ] **Step 5: Run lib tests + STL export**

Run: `rtk cargo test -p rc3d-io --lib`
Expected: All tests pass

Run: `rtk cargo test -p rc3d-io --test export_step_stl --release -- --nocapture`
Expected: Shape-2 faces now show BSpline surface type, NOT Plane fallback. No crash.

- [ ] **Step 6: Commit**

```bash
git add crates/rc3d-io/src/step/brep/geom/nurbs_build.rs
git commit -m "fix(step): adaptive knot/weight scanning for merged subsuper BSpline params

CompatMerge produces different param layouts depending on omitted
flags in the STEP file. Hardcoded off+7 fails for subsuper entities.
Add scan_knot_data() that finds knot multiplicities and values by
inspecting param types past the control point list, regardless of
their absolute positions."
```

---

### Task 3: Integration Validation

- [ ] **Step 1: Run full test suite**

```bash
rtk cargo test -p rc3d-io --release
```
Expected: All tests pass (378+)

- [ ] **Step 2: Verify Shape-2 STL quality**

```bash
rtk cargo test -p rc3d-io --test export_step_stl --release -- --nocapture
```

Check output for:
- No "build_surface failed" warnings for Shape-2 entities
- Shape-2 faces using BSpline surface (not Plane)
- Triangle count may differ from before (now using curved surface triangulation instead of flat plane)

- [ ] **Step 3: Verify no regressions on Shape, Shape-1, cs**

Same test covers all models. Verify:
- Shape.step: ~38K tris, no grid fallback change
- Shape-1.step: same surface types, no new warnings
- cs.step: sphere + planes, unchanged

- [ ] **Step 4: Clean up any remaining diagnostic code**

Remove any `eprintln!` or `[diag]` debug prints that may have been left from investigation:

```bash
grep -rn "\[diag\]" crates/rc3d-io/src/
```
Expected: no matches

- [ ] **Step 5: Commit final state**

```bash
git add -A crates/rc3d-io/src/
git commit -m "chore: clean up diagnostics after subsuper adapter fix"
```
