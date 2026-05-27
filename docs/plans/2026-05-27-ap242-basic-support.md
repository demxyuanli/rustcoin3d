# AP242 Basic Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add AP242 mode detection, void shell mesh subtraction, and SURFACE_CURVE entity type registration.

**Architecture:** Three independent additions: (1) parse FILE_SCHEMA header to detect AP mode and expose it in `StepImportReport`, (2) during mesh phase, subtract void shell meshes from outer shell mesh using centroid-in-mesh classification, (3) register `SURFACE_CURVE` in `EntityType` to eliminate Unknown classification.

**Tech Stack:** Rust, existing `rc3d-io/step/` modules

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `crates/rc3d-io/src/step/entity_types.rs` | Modify | Add `SurfaceCurve` variant |
| `crates/rc3d-io/src/step/header.rs` | Modify | Parse AP schema identifier from FILE_SCHEMA |
| `crates/rc3d-io/src/step/import_options.rs` | Modify | Add `ap_schema` field to `StepImportReport` |
| `crates/rc3d-io/src/step/mod.rs` | Modify | Wire AP detection into import path; wire void mesh subtraction |
| `crates/rc3d-io/src/step/brep/mesh/void_subtract.rs` | Create | Mesh-level void subtraction logic |
| `crates/rc3d-io/src/step/brep/mesh/mod.rs` | Modify | Add `void_subtract` module + re-export |

---

### Task 1: Add SURFACE_CURVE to EntityType enum

**Files:**
- Modify: `crates/rc3d-io/src/step/entity_types.rs:19-20`

**Why:** SURFACE_CURVE is handled by string matching in `geom.rs`, `pcurve.rs`, and `brep/build.rs`, but classified as `Unknown` in the parser. Adding it to the enum makes type detection accurate and eliminates false "unknown entity" warnings.

- [ ] **Step 1: Add variant to enum**

In `entity_types.rs`, after line 19 (`SeamCurve,`), add:

```rust
    SurfaceCurve,
```

- [ ] **Step 2: Add string mapping**

In `entity_types.rs`, after line 114 (`"SEAM_CURVE" => Self::SeamCurve,`), add:

```rust
            "SURFACE_CURVE" => Self::SurfaceCurve,
```

- [ ] **Step 3: Build check**

```bash
rtk cargo check -p rc3d-io
```

Expected: compiles cleanly.

- [ ] **Step 4: Commit**

```bash
rtk git add crates/rc3d-io/src/step/entity_types.rs
rtk git commit -m "feat(step): add SurfaceCurve to EntityType enum"
```

---

### Task 2: AP242 Mode Detection from FILE_SCHEMA

**Files:**
- Modify: `crates/rc3d-io/src/step/header.rs`
- Modify: `crates/rc3d-io/src/step/import_options.rs`
- Modify: `crates/rc3d-io/src/step/mod.rs`

**Why:** The FILE_SCHEMA header contains the AP identifier (e.g. `AP242_MANAGED_MODEL_BASED_3D_ENGINEERING`). Parsing it enables reporting and future schema-aware dispatch.

- [ ] **Step 1: Add AP schema detection to HeaderInfo**

In `header.rs`, add after the `extra` field in `HeaderInfo`:

```rust
    /// Detected AP schema (e.g. "AP242", "AP203", "AP214") from FILE_SCHEMA.
    pub ap_schema: Option<String>,
```

- [ ] **Step 2: Add schema parsing logic**

In `header.rs`, add this function:

```rust
/// Extract the AP identifier from a FILE_SCHEMA identifier string.
/// FILE_SCHEMA strings use the format: 'SCHEMA_NAME {{ version }}'
pub(crate) fn extract_ap_schema(schema_str: &str) -> Option<String> {
    let name = schema_str.split("{{").next()?.trim().trim_matches('\'');
    let upper = name.to_uppercase();
    for ap in &["AP242", "AP214", "AP203", "AP209", "AP210", "AP238"] {
        if upper.contains(ap) {
            return Some(ap.to_string());
        }
    }
    // Fallback: check for well-known schema names
    if upper.contains("AUTOMOTIVE_DESIGN") {
        return Some("AP214".to_string());
    }
    if upper.contains("CONFIG_CONTROL_DESIGN") {
        return Some("AP203".to_string());
    }
    if upper.contains("MANAGED_MODEL_BASED") {
        return Some("AP242".to_string());
    }
    None
}
```

- [ ] **Step 3: Call detection in parse_header**

In `header.rs`, inside `parse_header()`, after the `"FILE_SCHEMA"` match arm (~line 67), update to:

```rust
            "FILE_SCHEMA" => {
                info.file_schema = parse_string_list(&args_str);
                // Detect AP schema from first schema identifier
                info.ap_schema = info.file_schema.first()
                    .and_then(|s| extract_ap_schema(s));
            }
```

- [ ] **Step 4: Add test for AP schema detection**

In `header.rs` tests module, add:

```rust
    #[test]
    fn test_ap_schema_detection() {
        let input = "HEADER;
FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING {{ 1 0 10303 242 3 1 1 }}'));
ENDSEC;
DATA;";
        let (header, _) = parse_header(input).unwrap();
        assert_eq!(header.ap_schema.as_deref(), Some("AP242"));
    }

    #[test]
    fn test_ap_schema_ap203() {
        let input = "HEADER;
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;";
        let (header, _) = parse_header(input).unwrap();
        assert_eq!(header.ap_schema.as_deref(), Some("AP203"));
    }

    #[test]
    fn test_ap_schema_ap214() {
        let input = "HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN {{ 1 0 10303 214 3 1 1 }}'));
ENDSEC;
DATA;";
        let (header, _) = parse_header(input).unwrap();
        assert_eq!(header.ap_schema.as_deref(), Some("AP214"));
    }
```

- [ ] **Step 5: Run header tests**

```bash
rtk cargo test -p rc3d-io -- header
```

Expected: 6 tests pass (3 existing + 3 new).

- [ ] **Step 6: Add ap_schema to StepImportReport**

In `import_options.rs`, add field to `StepImportReport`:

```rust
    /// Detected AP schema (e.g. "AP242", "AP203", "AP214", or None if unrecognized).
    pub ap_schema: Option<String>,
```

Update the `Default` impl — the field auto-defaults to `None` via `..Default::default()` so no change needed there.

- [ ] **Step 7: Wire into import path**

In `mod.rs`, inside `exchange_to_scene_graph()`, after `import_report.unknown_entity_count = exchange.diagnostics.unknown_entity_count;` (~line 124), add:

```rust
    if let Some(ref header) = exchange.header {
        import_report.ap_schema = header.ap_schema.clone();
        if let Some(ref ap) = import_report.ap_schema {
            log::info!("[STEP] detected schema: {}", ap);
        }
    }
```

- [ ] **Step 8: Run full check + tests**

```bash
rtk cargo check -p rc3d-io
rtk cargo test -p rc3d-io
```

Expected: compiles, all 222+ tests pass.

- [ ] **Step 9: Commit**

```bash
rtk git add crates/rc3d-io/src/step/header.rs crates/rc3d-io/src/step/import_options.rs crates/rc3d-io/src/step/mod.rs
rtk git commit -m "feat(step): detect AP schema from FILE_SCHEMA header"
```

---

### Task 3: Void Shell Mesh Subtraction

**Files:**
- Create: `crates/rc3d-io/src/step/brep/mesh/void_subtract.rs`
- Modify: `crates/rc3d-io/src/step/brep/mesh/mod.rs`
- Modify: `crates/rc3d-io/src/step/mod.rs`

**Why:** BREP_WITH_VOIDS solids have void shells that are collected but not subtracted from the outer shell mesh. We need mesh-level subtraction so that holes are visible in the rendered mesh.

**Approach:** For each outer mesh triangle, test if its centroid lies inside any void mesh via ray casting. Remove triangles whose centroids are inside a void. Use a spatial grid for efficient candidate filtering.

- [ ] **Step 1: Create void_subtract.rs with failing test**

Create `crates/rc3d-io/src/step/brep/mesh/void_subtract.rs`:

```rust
//! Mesh-level void shell subtraction for BREP_WITH_VOIDS.
//!
//! Classifies each outer-mesh triangle against void meshes using
//! centroid ray-casting. Triangles whose centroids fall inside any
//! void are removed.

use rc3d_core::math::Vec3;
use crate::step::mesh_result::MeshResult;

/// Result of void subtraction.
#[derive(Debug)]
pub struct VoidSubtractResult {
    /// Mesh with void triangles removed.
    pub mesh: MeshResult,
    /// Number of triangles removed.
    pub removed_tris: usize,
}

/// Subtract void shell meshes from the outer shell mesh.
///
/// Keeps triangles whose centroid is NOT inside any void.
pub fn subtract_void_meshes(
    outer_mesh: &MeshResult,
    void_meshes: &[MeshResult],
) -> VoidSubtractResult {
    if void_meshes.is_empty() || outer_mesh.indices.is_empty() {
        return VoidSubtractResult {
            mesh: outer_mesh.clone(),
            removed_tris: 0,
        };
    }

    // Collect void triangles as (v0, v1, v2) for ray-cast tests
    let void_tris: Vec<[Vec3; 3]> = void_meshes
        .iter()
        .flat_map(|m| {
            m.indices.chunks(4).filter_map(|chunk| {
                if chunk.len() < 3 { return None; }
                let i0 = chunk[0] as usize;
                let i1 = chunk[1] as usize;
                let i2 = chunk[2] as usize;
                Some([
                    m.vertices[i0],
                    m.vertices[i1],
                    m.vertices[i2],
                ])
            })
        })
        .collect();

    if void_tris.is_empty() {
        return VoidSubtractResult {
            mesh: outer_mesh.clone(),
            removed_tris: 0,
        };
    }

    // Spatial grid for void triangles: cell size = avg void edge length * 2
    let cell_size = void_tris.iter()
        .flat_map(|t| {
            let e01 = (t[1] - t[0]).length();
            let e12 = (t[2] - t[1]).length();
            let e20 = (t[0] - t[2]).length();
            vec![e01, e12, e20]
        })
        .fold(1e-6f32, f32::max)
        .max(1e-3) * 2.0;

    let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::new();

    for (ti, tri) in void_tris.iter().enumerate() {
        let cx = (tri[0].x + tri[1].x + tri[2].x) / 3.0;
        let cy = (tri[0].y + tri[1].y + tri[2].y) / 3.0;
        let cz = (tri[0].z + tri[1].z + tri[2].z) / 3.0;
        // Expand to bounding box cells
        let min_x = tri[0].x.min(tri[1].x).min(tri[2].x);
        let max_x = tri[0].x.max(tri[1].x).max(tri[2].x);
        let min_y = tri[0].y.min(tri[1].y).min(tri[2].y);
        let max_y = tri[0].y.max(tri[1].y).max(tri[2].y);
        let min_z = tri[0].z.min(tri[1].z).min(tri[2].z);
        let max_z = tri[0].z.max(tri[1].z).max(tri[2].z);
        let ci0 = (min_x / cell_size).floor() as i32;
        let ci1 = (max_x / cell_size).ceil() as i32;
        let cj0 = (min_y / cell_size).floor() as i32;
        let cj1 = (max_y / cell_size).ceil() as i32;
        let ck0 = (min_z / cell_size).floor() as i32;
        let ck1 = (max_z / cell_size).ceil() as i32;
        for ci in ci0..=ci1 {
            for cj in cj0..=cj1 {
                for ck in ck0..=ck1 {
                    grid.entry((ci, cj, ck)).or_default().push(ti);
                }
            }
        }
    }

    let cell_size_inv = 1.0 / cell_size;
    let mut kept_vertices = Vec::new();
    let mut kept_indices = Vec::new();
    let mut removed = 0usize;

    for chunk in outer_mesh.indices.chunks(4) {
        if chunk.len() < 3 { continue; }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        let v0 = outer_mesh.vertices[i0];
        let v1 = outer_mesh.vertices[i1];
        let v2 = outer_mesh.vertices[i2];
        let centroid = Vec3::new(
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0,
        );

        let inside = point_inside_void_mesh(&centroid, &void_tris, &grid, cell_size_inv);
        if inside {
            removed += 1;
        } else {
            let base = kept_vertices.len() as i32;
            kept_vertices.extend_from_slice(&[v0, v1, v2]);
            kept_indices.extend_from_slice(&[
                base, base + 1, base + 2, -1,
            ]);
        }
    }

    VoidSubtractResult {
        mesh: MeshResult {
            vertices: kept_vertices,
            indices: kept_indices,
            normals: outer_mesh.normals.clone(),
        },
        removed_tris: removed,
    }
}

/// Ray-cast test: is `point` inside the void mesh?
///
/// Casts a ray along +X and counts triangle intersections.
/// Odd count → inside, even count → outside.
fn point_inside_void_mesh(
    point: &Vec3,
    void_tris: &[[Vec3; 3]],
    grid: &std::collections::HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size_inv: f32,
) -> bool {
    let cx = (point.x * cell_size_inv).floor() as i32;
    let cy = (point.y * cell_size_inv).floor() as i32;
    let cz = (point.z * cell_size_inv).floor() as i32;

    // Collect candidate triangles from spatial grid
    let mut seen = std::collections::HashSet::new();
    for &(ci, cj, ck) in &[
        (cx, cy, cz),
        (cx - 1, cy, cz),
        (cx + 1, cy, cz),
        (cx, cy - 1, cz),
        (cx, cy + 1, cz),
        (cx, cy, cz - 1),
        (cx, cy, cz + 1),
    ] {
        if let Some(candidates) = grid.get(&(ci, cj, ck)) {
            seen.extend(candidates);
        }
    }

    // Möller–Trumbore ray-triangle intersection along +X
    let ray_origin = *point;
    let ray_dir = Vec3::X;
    let mut count = 0u32;

    for &ti in &seen {
        let tri = &void_tris[ti];
        if ray_triangle_intersect(&ray_origin, &ray_dir, tri) {
            count += 1;
        }
    }

    count % 2 == 1
}

/// Möller–Trumbore ray-triangle intersection.
fn ray_triangle_intersect(origin: &Vec3, dir: &Vec3, tri: &[Vec3; 3]) -> bool {
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);

    if det.abs() < 1e-12 {
        return false;
    }

    let inv_det = 1.0 / det;
    let tvec = *origin - tri[0];
    let u = tvec.dot(pvec) * inv_det;
    if u < 0.0 || u > 1.0 {
        return false;
    }

    let qvec = tvec.cross(e1);
    let v = dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = e2.dot(qvec) * inv_det;
    t > 1e-8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_voids_returns_unchanged() {
        let outer = MeshResult {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            indices: vec![0, 1, 2, -1],
            normals: vec![],
        };
        let result = subtract_void_meshes(&outer, &[]);
        assert_eq!(result.mesh.indices.len(), 4);
        assert_eq!(result.removed_tris, 0);
    }

    #[test]
    fn test_void_subtraction_removes_inner_tris() {
        // Outer: large box (two tris making a square at z=0)
        let outer = MeshResult {
            vertices: vec![
                Vec3::new(-2.0, -2.0, 0.0),
                Vec3::new(2.0, -2.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(-2.0, 2.0, 0.0),
            ],
            indices: vec![
                0, 1, 2, -1,
                0, 2, 3, -1,
            ],
            normals: vec![],
        };
        // Void: small box inside outer
        let void = MeshResult {
            vertices: vec![
                Vec3::new(-1.0, -1.0, 0.1),
                Vec3::new(1.0, -1.0, 0.1),
                Vec3::new(1.0, 1.0, 0.1),
                Vec3::new(-1.0, 1.0, 0.1),
                Vec3::new(-1.0, -1.0, -0.1),
                Vec3::new(1.0, -1.0, -0.1),
                Vec3::new(1.0, 1.0, -0.1),
                Vec3::new(-1.0, 1.0, -0.1),
            ],
            indices: vec![
                // Top face
                0, 1, 2, -1,
                0, 2, 3, -1,
                // Bottom face
                7, 6, 5, -1,
                7, 5, 4, -1,
                // Side faces
                0, 4, 5, -1,
                0, 5, 1, -1,
                1, 5, 6, -1,
                1, 6, 2, -1,
                2, 6, 7, -1,
                2, 7, 3, -1,
                3, 7, 4, -1,
                3, 4, 0, -1,
            ],
            normals: vec![],
        };
        let result = subtract_void_meshes(&outer, &[void]);
        // The center of the outer square should be inside the void box
        assert!(result.removed_tris > 0);
        assert!(!result.mesh.indices.is_empty());
    }

    #[test]
    fn test_ray_triangle_hit() {
        let tri = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(ray_triangle_intersect(
            &Vec3::new(-1.0, 0.3, 0.3),
            &Vec3::X,
            &tri
        ));
    }

    #[test]
    fn test_ray_triangle_miss() {
        let tri = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(!ray_triangle_intersect(
            &Vec3::new(-1.0, 2.0, 2.0),
            &Vec3::X,
            &tri
        ));
    }
}
```

- [ ] **Step 2: Run the new tests (should compile + pass)**

```bash
rtk cargo test -p rc3d-io -- void_subtract
```

Expected: 4 tests pass.

- [ ] **Step 3: Register module in mesh/mod.rs**

In `crates/rc3d-io/src/step/brep/mesh/mod.rs`, add after the existing `pub mod` lines (~line 11):

```rust
pub mod void_subtract;
```

- [ ] **Step 4: Wire void subtraction into import path**

In `crates/rc3d-io/src/step/mod.rs`, inside `exchange_to_scene_graph()`, in the loop over `root_solids` (~line 245), after `let base_mesh = brep::mesh::mesh_brep_shell(...)` and the empty check, add void subtraction:

After the existing block:
```rust
            let base_mesh = brep::mesh::mesh_brep_shell(
                solid.outer_shell,
                &reg,
                &mesh_config,
                &total_heal.skip_face_keys,
            );
            if base_mesh.vertices.is_empty() || base_mesh.indices.is_empty() {
                continue;
            }
```

Insert the void subtraction logic before `log::info!([STEP] mesh: ...)`:

```rust
            // Subtract void shells from mesh
            let void_meshes: Vec<_> = solid.void_shells.iter().map(|&vk| {
                brep::mesh::mesh_brep_shell(
                    vk,
                    &reg,
                    &mesh_config,
                    &total_heal.skip_face_keys,
                )
            }).collect();
            let void_result = brep::mesh::void_subtract::subtract_void_meshes(
                &base_mesh, &void_meshes,
            );
            if void_result.removed_tris > 0 {
                log::info!(
                    "[STEP] void subtraction: removed {} tris, kept {}",
                    void_result.removed_tris,
                    void_result.mesh.indices.len() / 4,
                );
            }
            let final_mesh = void_result.mesh;
```

Then change all subsequent references from `base_mesh` to `final_mesh`.

- [ ] **Step 5: Run full tests**

```bash
rtk cargo test -p rc3d-io
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
rtk git add crates/rc3d-io/src/step/brep/mesh/void_subtract.rs crates/rc3d-io/src/step/brep/mesh/mod.rs crates/rc3d-io/src/step/mod.rs
rtk git commit -m "feat(step): subtract void shell meshes from outer shell mesh"
```

---

### Task 4: Integration — add ap_schema to test coverage

**Files:**
- Modify: `crates/rc3d-io/tests/step_files.rs`

- [ ] **Step 1: Add AP schema assertion to an existing test**

In `step_files.rs`, find a test that parses a STEP file and add:

```rust
assert!(report.ap_schema.is_some(), "AP schema should be detected");
```

Do this inside one of the existing test functions after `import_report` is available.

- [ ] **Step 2: Run tests**

```bash
rtk cargo test -p rc3d-io
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-io/tests/step_files.rs
rtk git commit -m "test(step): verify AP schema detection in integration tests"
```

