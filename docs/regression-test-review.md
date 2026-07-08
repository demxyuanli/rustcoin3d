# Regression Test Review — 2026-07-08

Post geometry-engine reduction (B-Rep entirely removed). Workspace: **333 passed, 6 ignored** (49 suites).

---

## 1. Test Inventory by Crate

| Crate | Tests | Status |
|-------|-------|--------|
| rc3d-shape | 124 | ✅ |
| rc3d-io | 0 | ❌ GAP |
| rc3d-core | ~40 | ✅ (math, bvh, spatial) |
| rc3d-render | ~100 | ✅ (shaders, passes) |
| rc3d-scene | ~80 | ✅ (scene graph) |
| rc3d-engine | ~40 | ✅ (engine) |
| Others (small crates) | ~88 | ✅ |

---

## 2. rc3d-shape — REMOVED

The entire rc3d-shape crate (124 tests) was removed. Visualization engine doesn't
use B-Rep topology, and there's no STEP import to generate B-Rep data.

## 3. rc3d-io Test Breakdown (0 tests) — GAP

rc3d-io is now a pure mesh I/O crate (STL, OBJ, glTF). Previous tests (22)
were BREP/STEP-related and removed with those modules.

**Current modules**: `stl.rs`, `obj.rs`, `gltf.rs` — all untested.

---

## 4. Gap Analysis — Mesh I/O (rc3d-io)

### STL Export — UNTESTED

| Function | Risk |
|----------|------|
| `write_binary_stl()` | Binary STL format correctness. No roundtrip test. |
| `write_ascii_stl()` | ASCII STL format. No test. |

### STL Import — UNTESTED

| Function | Risk |
|----------|------|
| `parse_stl()` | Binary/ASCII STL parsing. No test. |
| `parse_stl_file()` | File I/O + parse. No test. |

### OBJ — UNTESTED

| Function | Risk |
|----------|------|
| `parse_obj()` / `parse_obj_file()` | OBJ parsing. No test. |

### glTF — UNTESTED

| Function | Risk |
|----------|------|
| `parse_gltf_file()` | glTF 2.0 import. No test. |

## 5. Recommended Actions

1. **STL roundtrip**: Create triangle mesh → write_binary_stl → parse_stl → assert same triangles.
2. **OBJ roundtrip**: Same pattern.
3. **glTF minimal**: Test with a minimal embedded glTF buffer (single triangle).

```rust
// Quick STL roundtrip sketch
#[test]
fn stl_roundtrip() {
    let tris = vec![
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    let binary = write_binary_stl("test", &tris).unwrap();
    let parsed = parse_stl(&binary).unwrap();
    assert_eq!(parsed.len(), 1);
}
```
