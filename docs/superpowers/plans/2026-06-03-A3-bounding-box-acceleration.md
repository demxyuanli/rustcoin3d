# A3: Bounding Box Acceleration for Boolean Operations

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace O(F₁×F₂) brute-force face-face intersection loop with AABB overlap pre-filter, reducing intersection tests for models with many faces.

**Architecture:** Compute per-face AABB from vertex positions, then use sort-based sweep-and-prune to find overlapping pairs. Only call the expensive `intersect_surfaces_brep` for overlapping pairs. The AABB computation is O(V) per face (iterate edge vertices), and the sweep is O(F·logF). No new data structures stored on BRepStore — AABBs are computed on-the-fly per intersection call.

**Tech Stack:** Rust, rc3d-shape crate, Vec3 from rc3d_core

---

### Task 1: Add `face_vertex_bbox` function and AABB overlap test

**Files:**
- Create: `crates/rc3d-shape/src/bool/aabb.rs`
- Modify: `crates/rc3d-shape/src/bool/mod.rs` — add `mod aabb;`

- [ ] **Step 1: Write the failing test**

Create `crates/rc3d-shape/src/bool/aabb.rs` with a test module:

```rust
//! Axis-aligned bounding box utilities for boolean operation acceleration.

use rc3d_core::math::Vec3;
use crate::store::BRepStore;
use crate::topo::FaceKey;

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    /// Empty AABB (inverted bounds).
    pub fn empty() -> Self {
        AABB {
            min: Vec3::splat(f32::MAX),
            max: Vec3::splat(f32::MIN),
        }
    }

    /// Expand to include a point.
    pub fn expand(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    /// Check if two AABBs overlap.
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

/// Compute the AABB of a face from its edge vertex positions.
pub fn face_vertex_bbox(face: FaceKey, reg: &BRepStore) -> Option<AABB> {
    let mut bbox = AABB::empty();
    let mut any = false;

    for wire_key in crate::topo_iter::iter_wires_of_face(face, reg) {
        let wire = reg.wires.get(wire_key)?;
        for &(ek, _) in &wire.edges {
            let edge = reg.edges.get(ek)?;
            for vk in [edge.v_low, edge.v_high] {
                if let Some(v) = reg.vertices.get(vk) {
                    bbox.expand(v.position);
                    any = true;
                }
            }
        }
    }

    if any { Some(bbox) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_overlap_touching() {
        let a = AABB { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(1.0, 1.0, 1.0) };
        let b = AABB { min: Vec3::new(1.0, 0.0, 0.0), max: Vec3::new(2.0, 1.0, 1.0) };
        assert!(a.overlaps(&b), "touching faces should overlap");
    }

    #[test]
    fn test_aabb_no_overlap() {
        let a = AABB { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(1.0, 1.0, 1.0) };
        let b = AABB { min: Vec3::new(2.0, 2.0, 2.0), max: Vec3::new(3.0, 3.0, 3.0) };
        assert!(!a.overlaps(&b), "separated faces should not overlap");
    }

    #[test]
    fn test_aabb_overlap_partial() {
        let a = AABB { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(2.0, 2.0, 2.0) };
        let b = AABB { min: Vec3::new(1.0, 1.0, 1.0), max: Vec3::new(3.0, 3.0, 3.0) };
        assert!(a.overlaps(&b), "partially overlapping faces should overlap");
    }

    #[test]
    fn test_aabb_overlap_contained() {
        let a = AABB { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(4.0, 4.0, 4.0) };
        let b = AABB { min: Vec3::new(1.0, 1.0, 1.0), max: Vec3::new(2.0, 2.0, 2.0) };
        assert!(a.overlaps(&b), "contained face should overlap");
    }
}
```

- [ ] **Step 2: Register module in `bool/mod.rs`**

Read `crates/rc3d-shape/src/bool/mod.rs` and add `mod aabb;` alongside the existing module declarations.

- [ ] **Step 3: Run tests**

Run: `cargo test -p rc3d-shape -- aabb`
Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/rc3d-shape/src/bool/aabb.rs crates/rc3d-shape/src/bool/mod.rs
git commit -m "feat(bool): add AABB type and face vertex bounding box computation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Integrate AABB pre-filter into `compute_intersections_brep`

**Files:**
- Modify: `crates/rc3d-shape/src/bool/intersect.rs:29-54` — replace brute-force loop with AABB-filtered loop

- [ ] **Step 1: Write the test**

Add a test to `crates/rc3d-shape/src/bool/intersect.rs` that verifies the AABB filter doesn't miss valid intersections. The test creates two shells with far-apart faces and one pair that actually intersects — it should still find the intersection:

Append to the existing test module (or create one if it doesn't exist) in `intersect.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use rc3d_core::math::Vec3;
    use std::collections::HashMap;

    /// Two perpendicular planes that intersect → should produce an intersection curve.
    #[test]
    fn test_intersect_finds_crossing_planes() {
        let mut reg = BRepStore::new();

        // Plane A: z=0, small rectangle near origin
        let v0 = reg.find_or_add_vertex(Vec3::new(-1.0, -1.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, -1.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(-1.0, 1.0, 0.0), 1e-4);

        let line_x = CurveGeom::Line { origin: Vec3::new(-1.0, -1.0, 0.0), direction: Vec3::new(2.0, 0.0, 0.0) };
        let line_y = CurveGeom::Line { origin: Vec3::new(1.0, -1.0, 0.0), direction: Vec3::new(0.0, 2.0, 0.0) };
        let line_x2 = CurveGeom::Line { origin: Vec3::new(1.0, 1.0, 0.0), direction: Vec3::new(-2.0, 0.0, 0.0) };
        let line_y2 = CurveGeom::Line { origin: Vec3::new(-1.0, 1.0, 0.0), direction: Vec3::new(0.0, -2.0, 0.0) };

        let (lo01, hi01) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        let (lo12, hi12) = if v1 < v2 { (v1, v2) } else { (v2, v1) };
        let (lo23, hi23) = if v2 < v3 { (v2, v3) } else { (v3, v2) };
        let (lo30, hi30) = if v3 < v0 { (v3, v0) } else { (v0, v3) };

        let e01 = reg.edges.insert(BRepEdge { curve: line_x, tolerance: 1e-4, v_low: lo01, v_high: hi01, pcurves: HashMap::new() });
        let e12 = reg.edges.insert(BRepEdge { curve: line_y, tolerance: 1e-4, v_low: lo12, v_high: hi12, pcurves: HashMap::new() });
        let e23 = reg.edges.insert(BRepEdge { curve: line_x2, tolerance: 1e-4, v_low: lo23, v_high: hi23, pcurves: HashMap::new() });
        let e30 = reg.edges.insert(BRepEdge { curve: line_y2, tolerance: 1e-4, v_low: lo30, v_high: hi30, pcurves: HashMap::new() });

        let w_a = reg.wires.insert(BRepWire { edges: vec![(e01, Orientation::Forward), (e12, Orientation::Forward), (e23, Orientation::Forward), (e30, Orientation::Forward)] });
        let f_a = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X },
            outer_wire: w_a, inner_wires: vec![], same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });

        // Plane B: y=0, rectangle that crosses Plane A
        let v4 = reg.find_or_add_vertex(Vec3::new(-1.0, 0.0, -1.0), 1e-4);
        let v5 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, -1.0), 1e-4);
        let v6 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 1.0), 1e-4);
        let v7 = reg.find_or_add_vertex(Vec3::new(-1.0, 0.0, 1.0), 1e-4);

        let (lo45, hi45) = if v4 < v5 { (v4, v5) } else { (v5, v4) };
        let (lo56, hi56) = if v5 < v6 { (v5, v6) } else { (v6, v5) };
        let (lo67, hi67) = if v6 < v7 { (v6, v7) } else { (v7, v6) };
        let (lo74, hi74) = if v7 < v4 { (v7, v4) } else { (v4, v7) };

        let e45 = reg.edges.insert(BRepEdge { curve: CurveGeom::Line { origin: Vec3::new(-1.0, 0.0, -1.0), direction: Vec3::new(2.0, 0.0, 0.0) }, tolerance: 1e-4, v_low: lo45, v_high: hi45, pcurves: HashMap::new() });
        let e56 = reg.edges.insert(BRepEdge { curve: CurveGeom::Line { origin: Vec3::new(1.0, 0.0, -1.0), direction: Vec3::new(0.0, 0.0, 2.0) }, tolerance: 1e-4, v_low: lo56, v_high: hi56, pcurves: HashMap::new() });
        let e67 = reg.edges.insert(BRepEdge { curve: CurveGeom::Line { origin: Vec3::new(1.0, 0.0, 1.0), direction: Vec3::new(-2.0, 0.0, 0.0) }, tolerance: 1e-4, v_low: lo67, v_high: hi67, pcurves: HashMap::new() });
        let e74 = reg.edges.insert(BRepEdge { curve: CurveGeom::Line { origin: Vec3::new(-1.0, 0.0, 1.0), direction: Vec3::new(0.0, 0.0, -2.0) }, tolerance: 1e-4, v_low: lo74, v_high: hi74, pcurves: HashMap::new() });

        let w_b = reg.wires.insert(BRepWire { edges: vec![(e45, Orientation::Forward), (e56, Orientation::Forward), (e67, Orientation::Forward), (e74, Orientation::Forward)] });
        let f_b = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Y, u_dir: Vec3::X },
            outer_wire: w_b, inner_wires: vec![], same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });

        let sk_a = reg.shells.insert(BRepShell { faces: vec![(f_a, Orientation::Forward)], closed: false, step_id: None });
        let sk_b = reg.shells.insert(BRepShell { faces: vec![(f_b, Orientation::Forward)], closed: false, step_id: None });

        let results = compute_intersections_brep(&[sk_a], &[sk_b], &reg);
        assert_eq!(results.len(), 1, "Two crossing planes should produce 1 intersection");
        assert_eq!(results[0].curves_3d.len(), 1, "Should produce 1 curve");
    }
}
```

- [ ] **Step 2: Run test to verify it fails (or passes with current brute-force)**

Run: `cargo test -p rc3d-shape -- test_intersect_finds_crossing_planes`
Expected: PASS (the brute-force loop already works, this is a regression guard)

- [ ] **Step 3: Modify `compute_intersections_brep` to use AABB pre-filter**

Replace the function body in `crates/rc3d-shape/src/bool/intersect.rs` (lines 29-54) with:

```rust
/// Compute intersections between two sets of B-Rep shells.
pub fn compute_intersections_brep(
    shells_a: &[ShellKey],
    shells_b: &[ShellKey],
    reg: &BRepStore,
) -> Vec<FaceIntersectionResult> {
    use crate::bool::aabb::{self, AABB};

    // Collect all faces with their AABBs
    let mut faces_a: Vec<(FaceKey, AABB)> = Vec::new();
    for &sk in shells_a {
        let shell = match reg.shells.get(sk) { Some(s) => s, None => continue };
        for &(fk, _) in &shell.faces {
            if let Some(bbox) = aabb::face_vertex_bbox(fk, reg) {
                faces_a.push((fk, bbox));
            }
        }
    }

    let mut faces_b: Vec<(FaceKey, AABB)> = Vec::new();
    for &sk in shells_b {
        let shell = match reg.shells.get(sk) { Some(s) => s, None => continue };
        for &(fk, _) in &shell.faces {
            if let Some(bbox) = aabb::face_vertex_bbox(fk, reg) {
                faces_b.push((fk, bbox));
            }
        }
    }

    // Sweep-and-prune along X axis for overlapping pairs
    let mut results = Vec::new();
    let mut sorted_a: Vec<_> = faces_a.iter().collect();
    let mut sorted_b: Vec<_> = faces_b.iter().collect();
    sorted_a.sort_by(|a, b| a.1.min.x.partial_cmp(&b.1.min.x).unwrap_or(std::cmp::Ordering::Equal));
    sorted_b.sort_by(|a, b| a.1.min.x.partial_cmp(&b.1.min.x).unwrap_or(std::cmp::Ordering::Equal));

    for &(fka, ref bbox_a) in &sorted_a {
        for &(fkb, ref bbox_b) in &sorted_b {
            if bbox_b.min.x > bbox_a.max.x {
                break; // remaining B faces are past A in X
            }
            if !bbox_a.overlaps(bbox_b) {
                continue;
            }
            let face_a = match reg.faces.get(fka) { Some(f) => f, None => continue };
            let face_b = match reg.faces.get(fkb) { Some(f) => f, None => continue };
            if let Some(curves) = intersect_surfaces_brep(face_a, face_b, reg) {
                results.push(FaceIntersectionResult {
                    face_a: fka, face_b: fkb,
                    curves_3d: curves, pcurves_on_a: vec![], pcurves_on_b: vec![],
                });
            }
        }
    }

    results
}
```

- [ ] **Step 4: Run all tests**

Run: `cargo test -p rc3d-shape`
Expected: All tests pass, including `test_intersect_finds_crossing_planes`.

- [ ] **Step 5: Commit**

```bash
git add crates/rc3d-shape/src/bool/intersect.rs
git commit -m "perf(bool): add AABB pre-filter to face-face intersection loop

Replaces O(F₁×F₂) brute-force with sweep-and-prune along X axis.
Only face pairs with overlapping AABBs are tested for intersection.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
