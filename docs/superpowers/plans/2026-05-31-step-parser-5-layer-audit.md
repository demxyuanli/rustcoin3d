# STEP Parser 5-Layer Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit and fix the 5-layer STEP Part21 parser against ISO 10303-21 and OCC reference behavior, with structural output tests added at each layer.

**Architecture:** 5 independent tasks — L0 (lexer) → L1 (syntax) → L2 (subsuper) → L3 (type mapping) → L4 (param merge). Each layer is self-contained: add tests, fix gaps, verify, commit.

**Tech Stack:** Rust, ISO 10303-21 Part21, hand-written recursive descent parser

---

### Task 1: L0 — Lexer Tests

**Files:**
- Modify: `crates/rc3d-io/src/step/part21/lexer.rs` (test module)
- Modify: `crates/rc3d-io/src/step/part21/token.rs` (add Binary variant if needed)

- [ ] **Step 1: Add comprehensive lexer tests**

Add to the `mod tests` block at the bottom of `lexer.rs`:

```rust
#[test]
fn lex_all_token_types_roundtrip() {
    let input = "#12 = CARTESIAN_POINT('label', (0.0, 1.5e-3, -2.0), .T., $);";
    let tokens = lex(input).unwrap();
    // Verify each token type
    assert!(matches!(tokens[0].0, Token::Ref(12)));
    assert!(matches!(tokens[1].0, Token::Eq));
    assert!(matches!(&tokens[2].0, Token::Keyword(k) if k == "CARTESIAN_POINT"));
    assert!(matches!(tokens[3].0, Token::LParen));
    assert!(matches!(&tokens[4].0, Token::String(s) if s == "label"));
    assert!(matches!(tokens[5].0, Token::Comma));
    assert!(matches!(tokens[6].0, Token::LParen));
    assert!(matches!(tokens[7].0, Token::Real(v) if (v - 0.0).abs() < 1e-10));
    assert!(matches!(tokens[8].0, Token::Comma));
    assert!(matches!(tokens[9].0, Token::Real(v) if (v - 1.5e-3).abs() < 1e-10));
    assert!(matches!(tokens[13].0, Token::Enum(e) if e == ".T."));
    assert!(matches!(tokens[15].0, Token::Omitted));
    assert!(matches!(tokens[17].0, Token::Semi));
}
```

```rust
#[test]
fn lex_escaped_apostrophe() {
    let tokens = lex("#1 = STRING('it''s');").unwrap();
    assert!(matches!(&tokens[2].0, Token::String(s) if s == "it's"));
}
```

```rust
#[test]
fn lex_negative_real_with_exponent() {
    let tokens = lex("#1 = REAL_VAL(-1.5e+3);").unwrap();
    assert!(matches!(tokens[2].0, Token::Real(v) if (v + 1500.0).abs() < 1e-10));
}
```

```rust
#[test]
fn lex_block_comment_skipped() {
    let tokens = lex("/* this is a comment */\n#1 = POINT(0.0);").unwrap();
    // Comment should be completely skipped, parsing should start at #1
    assert!(matches!(tokens[0].0, Token::Ref(1)));
    assert!(matches!(&tokens[1].0, Token::Keyword(k) if k == "POINT"));
}
```

```rust
#[test]
fn lex_keyword_with_underscores() {
    let tokens = lex("#1 = B_SPLINE_SURFACE_WITH_KNOTS();").unwrap();
    assert!(matches!(&tokens[2].0, Token::Keyword(k) if k == "B_SPLINE_SURFACE_WITH_KNOTS"));
}
```

```rust
#[test]
fn lex_empty_input() {
    let tokens = lex("").unwrap();
    assert!(tokens.is_empty());
}
```

- [ ] **Step 2: Run lexer tests**

Run: `rtk cargo test -p rc3d-io --lib lex_`
Expected: All 6 new tests PASS

- [ ] **Step 3: Commit**

```bash
git add crates/rc3d-io/src/step/part21/lexer.rs
git commit -m "test(step): comprehensive lexer token coverage tests"
```

---

### Task 2: L1 — Syntax (Parameter Parser) Tests

**Files:**
- Modify: `crates/rc3d-io/src/step/part21/params.rs` (test module)
- Modify: `crates/rc3d-io/src/step/part21/instance.rs` (test module)

- [ ] **Step 1: Add parameter parsing edge case tests**

Add to the `mod tests` block in `params.rs`:

```rust
#[test]
fn parse_entity_with_trailing_comma_in_nested() {
    // ISO allows trailing comma: (1.0,2.0,)
    let input = "#1 = PT(#2,(1.0,2.0,));";
    let (inst, _) = crate::step::part21::instance::parse_instance(input).unwrap();
    assert_eq!(inst.id, 1);
    assert_eq!(inst.records[0].keyword, "PT");
}
```

```rust
#[test]
fn parse_deeply_nested_param_list() {
    // 5 levels of nesting
    let input = "#1 = NEST((((((42))))));";
    let (inst, _) = crate::step::part21::instance::parse_instance(input).unwrap();
    assert_eq!(inst.id, 1);
    assert_eq!(inst.records[0].keyword, "NEST");
}
```

```rust
#[test]
fn parse_typed_param_nested_value() {
    let (val, rest) = parse_param("LENGTH_MEASURE(0.001)").unwrap();
    assert!(rest.trim().is_empty());
    assert!(matches!(val, StepValue::Typed(name, _) if name == "LENGTH_MEASURE"));
}

#[test]
fn parse_mixed_types_in_list() {
    let (val, rest) = parse_param_list("#1, 2.0, 'str', .T., $").unwrap();
    assert!(rest.trim().is_empty());
    let list = val.as_list().unwrap();
    assert_eq!(list.len(), 5);
    assert!(matches!(list[0], StepValue::Ref(1)));
    assert!(matches!(list[1], StepValue::Real(v) if (v - 2.0).abs() < 1e-10));
    assert!(matches!(&list[2], StepValue::String(s) if s == "str"));
    assert!(matches!(&list[3], StepValue::Enum(e) if e == ".T."));
    assert!(matches!(list[4], StepValue::Omitted));
}

#[test]
fn parse_enum_variants() {
    for (input, expected) in &[
        (".T.", ".T."),
        (".F.", ".F."),
        (".UNSPECIFIED.", ".UNSPECIFIED."),
        (".MILLI.", ".MILLI."),
        (".METRE.", ".METRE."),
    ] {
        let (val, rest) = parse_param(input).unwrap();
        assert!(rest.trim().is_empty(), "rest not empty for {}", input);
        assert!(matches!(&val, StepValue::Enum(e) if e == expected));
    }
}
```

Add to `instance.rs` tests:

```rust
#[test]
fn parse_simple_entity_with_string() {
    let (inst, rest) = parse_instance("#1 = LABEL('hello');").unwrap();
    assert_eq!(inst.id, 1);
    assert!(rest.trim().is_empty());
    assert_eq!(inst.records[0].keyword, "LABEL");
}

#[test]
fn parse_two_entities() {
    let (inst1, rest) = parse_instance("#1 = FIRST(1.0);#2 = SECOND(2.0);").unwrap();
    assert_eq!(inst1.id, 1);
    let (inst2, rest) = parse_instance(rest).unwrap();
    assert_eq!(inst2.id, 2);
    assert!(rest.trim().is_empty());
}
```

- [ ] **Step 2: Run syntax tests**

Run: `rtk cargo test -p rc3d-io --lib "parse_"`
Expected: All new tests PASS

- [ ] **Step 3: Commit**

```bash
git add crates/rc3d-io/src/step/part21/params.rs crates/rc3d-io/src/step/part21/instance.rs
git commit -m "test(step): edge case tests for parameter parser and entity syntax"
```

---

### Task 3: L2 — Subsuper Edge Cases

**Files:**
- Modify: `crates/rc3d-io/src/step/part21/instance.rs` (test module)
- Modify: `crates/rc3d-io/src/step/adapter/subsuper.rs` (potential fix)
- Modify: `crates/rc3d-io/src/step/adapter/mod.rs` (test module)

- [ ] **Step 1: Add subsuper edge case tests, fix `find_outer_paren` bug**

Working subsuper test (add to `instance.rs` tests):

```rust
#[test]
fn subsuper_internal_three_level() {
    // Simulates AP242: (GRANDCHILD() CHILD() PARENT())
    let input = "#1 = (CHILD(1.0) PARENT());";
    let (inst, _) = parse_instance(input).unwrap();
    assert_eq!(inst.id, 1);
    // Internal: leaf is first record (CHILD) in original STEP order
    assert!(inst.records.len() >= 2);
    // Verify records are present
    let keywords: Vec<&str> = inst.records.iter().map(|r| r.keyword.as_str()).collect();
    assert!(keywords.contains(&"CHILD"));
    assert!(keywords.contains(&"PARENT"));
}

#[test]
fn subsuper_internal_multiline() {
    let input = "#1 = (\nBOUNDED_SURFACE()\nB_SPLINE_SURFACE(2,3,(#10))\n);";
    let (inst, _) = parse_instance(input).unwrap();
    assert_eq!(inst.id, 1);
    assert!(inst.records.len() >= 2);
    let keywords: Vec<&str> = inst.records.iter().map(|r| r.keyword.as_str()).collect();
    assert!(keywords.contains(&"B_SPLINE_SURFACE"));
    assert!(keywords.contains(&"BOUNDED_SURFACE"));
}

#[test]
fn subsuper_external_with_keyword_after_paren() {
    // OCC format: (SUPER1()SUPER2())ENTITY(1.0,2.0);
    let input = "#1 = (WRAPPER())REAL_ENTITY(1.0, 2.0);";
    let (inst, _) = parse_instance(input).unwrap();
    assert_eq!(inst.id, 1);
    // External mapping: REAL_ENTITY is outside the parens
    let keywords: Vec<&str> = inst.records.iter().map(|r| r.keyword.as_str()).collect();
    assert!(keywords.contains(&"REAL_ENTITY"));
}
```

- [ ] **Step 2: Fix `find_outer_paren` — verify it handles edge cases**

The current implementation at `instance.rs:174-191` counts parenthesis depth. Verify with a test:

```rust
#[test]
fn find_outer_paren_balanced_nested() {
    // find_outer_paren handles: (...(...)...)
    let input = "(OUTER (INNER)) rest";
    let (close_pos, inner) = super::find_outer_paren(input);
    assert_eq!(inner, "OUTER (INNER)");
    assert_eq!(&input[close_pos..].trim(), "rest");
}
```

If the test passes, no fix needed. If it fails, fix `find_outer_paren`.

- [ ] **Step 3: Add adapter-level subsuper integration test**

Add to `adapter/mod.rs` tests:

```rust
#[test]
fn adapter_subsurface_bspline_multi_level() {
    let input = "ISO-10303-21;\nHEADER;ENDSEC;\nDATA;\n#1 = (\n\
BOUNDED_SURFACE()\nB_SPLINE_SURFACE(2,3,(#10,#11))\n\
B_SPLINE_SURFACE_WITH_KNOTS((2,2),(2,2),(0.0,1.0),(0.0,1.0),.UNSPECIFIED.)\n\
SURFACE()\nGEOMETRIC_REPRESENTATION_ITEM()\nREPRESENTATION_ITEM('')\n\
);
ENDSEC;
END-ISO-10303-21;
";
    let exchange = read_exchange(input).expect("part21 read");
    let idx = to_entity_index(&exchange, &AdapterOptions::compat_merge()).expect("adapter");
    let e = idx.get(&1).expect("#1");
    // Priority-based selection picks B_SPLINE_SURFACE (first structural type)
    assert_eq!(e.name, "B_SPLINE_SURFACE");
    // Verify params contain the degree (Integer(2))
    if let StepValue::List(params) = &e.params {
        assert!(params.iter().any(|v| matches!(v, StepValue::Integer(2))));
    } else {
        panic!("expected List params");
    }
}
```

- [ ] **Step 4: Run all subsuper tests**

Run: `rtk cargo test -p rc3d-io --lib subsuper`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add crates/rc3d-io/src/step/part21/instance.rs
git add crates/rc3d-io/src/step/adapter/mod.rs
git commit -m "test(step): subsuper edge case tests for multi-level and multiline entities"
```

---

### Task 4: L3 — Type Mapping Completeness

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/build/surface.rs` (add supertype handlers if missing)
- Modify: `crates/rc3d-io/src/step/brep/build/curve.rs` (add supertype handlers if missing)
- Modify: `crates/rc3d-io/src/step/entity_types.rs` (add missing mappings if any)

- [ ] **Step 1: Audit and add missing supertype handlers in build_surface**

Check `build_surface` for these supertype names that need unwrap handling:

```rust
// Add "GEOMETRIC_REPRESENTATION_ITEM" alongside existing "SURFACE" handler
// These are AP242 supertype wrappers that reference a basis surface.

// Read current state and check which supertypes are missing.
// Currently handled:
//   - BOUNDED_SURFACE, CURVE_BOUNDED_SURFACE → unwrap via nth_ref(1)
//   - RECTANGULAR_TRIMMED_SURFACE → unwrap via nth_ref(1)
//   - ELEMENTARY_SURFACE, SWEPT_SURFACE → unwrap via nth_ref(1) or warn
//   - SURFACE → handled (priority-based adapter fix)
//
// Verify: GEOMETRIC_REPRESENTATION_ITEM should also be handled if it
// appears as a face's surface reference.
```

Read `build/surface.rs` to confirm all supertypes are covered. If "GEOMETRIC_REPRESENTATION_ITEM" is missing, add:

```rust
"GEOMETRIC_REPRESENTATION_ITEM" => {
    // Unwrap to underlying surface when referenced directly as face surface
    if let Some(basis_id) = geom::nth_ref(&record.params, 1) {
        build_surface(basis_id, entities)
    } else {
        log::warn!(
            "[BRep] build_surface: GEOMETRIC_REPRESENTATION_ITEM for #{} has no basis",
            surface_id
        );
        None
    }
}
```

- [ ] **Step 2: Audit build_curve for supertype unwrap**

Read `build/curve.rs` and confirm these supertypes are handled:
- BOUNDED_CURVE (line 89-93) ✓
- SURFACE_CURVE, SEAM_CURVE, INTERSECTION_CURVE (line 81-84) ✓
- OFFSET_CURVE_3D (line 85-88) ✓
- COMPOSITE_CURVE (line 63-78) ✓

If any are missing, add them.

- [ ] **Step 3: Add test for surface type detection**

Add to `adapter/mod.rs` tests:

```rust
#[test]
fn entity_type_mapping_coverage() {
    // Verify all common surface/curve types are in EntityType::from_name
    let surface_types = [
        "PLANE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE",
        "SPHERICAL_SURFACE", "TOROIDAL_SURFACE",
        "B_SPLINE_SURFACE", "B_SPLINE_SURFACE_WITH_KNOTS",
        "RATIONAL_B_SPLINE_SURFACE",
        "SURFACE_OF_LINEAR_EXTRUSION", "SURFACE_OF_REVOLUTION",
        "OFFSET_SURFACE", "BOUNDED_SURFACE", "RECTANGULAR_TRIMMED_SURFACE",
        "CURVE_BOUNDED_SURFACE",
    ];
    for name in &surface_types {
        let ty = EntityType::from_name(name);
        assert_ne!(ty, EntityType::Unknown, "Missing mapping: {}", name);
    }
}
```

- [ ] **Step 4: Run tests**

Run: `rtk cargo test -p rc3d-io --lib`
Expected: All PASS, including new tests

- [ ] **Step 5: Commit**

```bash
git add crates/rc3d-io/src/step/brep/build/surface.rs
git add crates/rc3d-io/src/step/brep/build/curve.rs
git add crates/rc3d-io/src/step/adapter/mod.rs
git commit -m "fix(step): audit type mapping completeness for surface/curve supertypes"
```

---

### Task 5: L4 — Parameter Merge Improvement

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/geom/nurbs_build.rs` (add scan_knot_data fallback)
- Modify: `crates/rc3d-io/src/step/adapter/subsuper.rs` (verify merge order)

- [ ] **Step 1: Add scan_knot_data fallback to nurbs_build**

Read `nurbs_build.rs` to check if `scan_knot_data` is implemented. If not, implement as described in the spec:

After the hardcoded `mult_base = off + 7` read, add a fallback:

```rust
let mult_base = off + 7;
let mut u_mults = geom::nth_list_ints(params, mult_base);
let mut v_mults = geom::nth_list_ints(params, mult_base + 1);
let mut u_knot_vals = geom::nth_list_reals(params, mult_base + 2);
let mut v_knot_vals = geom::nth_list_reals(params, mult_base + 3);

// If hardcoded position fails (CompatMerge layout differs from standalone),
// scan the parameter list for knot data by type inspection.
if u_mults.is_empty() || v_mults.is_empty() {
    if let Some((um, vm, uk, vk)) = scan_knot_data(params, rows, cols) {
        u_mults = um;
        v_mults = vm;
        u_knot_vals = uk;
        v_knot_vals = vk;
    }
}
```

Implement `scan_knot_data`:

```rust
/// Scan merged params for knot multiplicity and value lists
/// by inspecting parameter types past the control point list.
fn scan_knot_data(
    params: &StepValue,
    _cp_u: usize,
    _cp_v: usize,
) -> Option<(Vec<i64>, Vec<i64>, Vec<StepValue>, Vec<StepValue>)> {
    let list = params.as_list()?;
    let mut past_cps = false;
    let mut int_lists: Vec<&[StepValue]> = Vec::new();
    let mut real_lists: Vec<&[StepValue]> = Vec::new();

    for val in list.iter() {
        match val {
            StepValue::List(inner) if !inner.is_empty() => {
                // Detect the CP list: List of Lists of Refs
                if inner.iter().any(|v| matches!(v, StepValue::List(_))) {
                    past_cps = true;
                    continue;
                }
                if !past_cps { continue; }
                if inner.iter().all(|v| matches!(v, StepValue::Integer(_))) {
                    int_lists.push(inner.as_slice());
                } else if inner.iter().all(|v| matches!(v, StepValue::Real(_))) {
                    real_lists.push(inner.as_slice());
                }
            }
            _ => {}
        }
    }

    if int_lists.len() < 2 || real_lists.len() < 2 {
        return None;
    }

    Some((
        int_lists[0].iter().filter_map(|v| {
            if let StepValue::Integer(n) = v { Some(*n) } else { None }
        }).collect(),
        int_lists[1].iter().filter_map(|v| {
            if let StepValue::Integer(n) = v { Some(*n) } else { None }
        }).collect(),
        real_lists[0].to_vec(),
        real_lists[1].to_vec(),
    ))
}
```

- [ ] **Step 2: Verify merge order consistency**

Verify that `merge_all_params_structured` preserves the record order from the External/Internal mapping. The order should be:
- For External (alphabetically sorted): BOUNDED_SURFACE, B_SPLINE_SURFACE, B_SPLINE_SURFACE_WITH_KNOTS, ...
- For Internal (original STEP order): records in file order

Already verified in Task 1's adapter fix. No change needed.

- [ ] **Step 3: Add test for knot data scanning**

```rust
#[test]
fn test_scan_knot_data_from_merged_params() {
    // Simulate CompatMerge output for B_SPLINE_SURFACE + B_SPLINE_SURFACE_WITH_KNOTS
    let params = StepValue::List(vec![
        StepValue::Integer(2),  // degree_u
        StepValue::Integer(3),  // degree_v
        StepValue::List(vec![  // control_points
            StepValue::List(vec![StepValue::Ref(10), StepValue::Ref(11)]),
            StepValue::List(vec![StepValue::Ref(12), StepValue::Ref(13)]),
        ]),
        StepValue::Enum(".UNSPECIFIED.".to_string()), // surface_form (omitted in some files)
        StepValue::Enum(".F.".to_string()),           // u_closed
        StepValue::Enum(".F.".to_string()),           // v_closed  
        StepValue::Enum(".F.".to_string()),           // self_intersect
        StepValue::List(vec![StepValue::Integer(2), StepValue::Integer(2)]), // u_mults
        StepValue::List(vec![StepValue::Integer(2), StepValue::Integer(2)]), // v_mults
        StepValue::List(vec![StepValue::Real(0.0), StepValue::Real(1.0)]),   // u_knots
        StepValue::List(vec![StepValue::Real(0.0), StepValue::Real(1.0)]),   // v_knots
    ]);
    let result = scan_knot_data(&params, 2, 2);
    assert!(result.is_some());
    let (um, vm, _uk, _vk) = result.unwrap();
    assert_eq!(um, vec![2, 2]);
    assert_eq!(vm, vec![2, 2]);
}
```

- [ ] **Step 4: Run all tests + STL export**

Run: `rtk cargo test -p rc3d-io --lib`
Expected: All PASS

Run: `rtk cargo test -p rc3d-io --test export_step_stl --release`
Expected: PASS, Shape-2 still correct (62K tris, BSpline surfaces)

- [ ] **Step 5: Commit**

```bash
git add crates/rc3d-io/src/step/brep/geom/nurbs_build.rs
git commit -m "feat(step): scan_knot_data fallback for merged subsuper BSpline params"
```

---

### Final Integration

- [ ] **Run complete test suite**

```bash
rtk cargo test -p rc3d-io --release
```
Expected: All pass (380+ tests, 4 ignored)
