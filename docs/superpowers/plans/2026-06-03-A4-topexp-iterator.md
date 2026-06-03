# A4: TopExp Topology Iterator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a unified topology traversal module that replaces 34+ inline wire iterations and 14+ inline shell iterations with named, reusable functions.

**Architecture:** Add `topo_iter.rs` alongside `topo.rs` in rc3d-shape. Provide free functions that take `(key, &BRepStore)` and return iterators or collections. Functions are thin wrappers over the same SlotMap lookups — no new data structures, no caching. Existing code continues to work; new code uses the iterators.

**Tech Stack:** Rust, rc3d-shape crate, SlotMap

---

### Task 1: Create the `topo_iter` module with core functions

**Files:**
- Create: `crates/rc3d-shape/src/topo_iter.rs`
- Modify: `crates/rc3d-shape/src/lib.rs` — add `pub mod topo_iter;`

- [ ] **Step 1: Write the module with all iterator functions**

Create `crates/rc3d-shape/src/topo_iter.rs`:

```rust
//! Topology traversal utilities (OCC TopExp equivalent).
//!
//! Provides named functions for common topology iteration patterns.
//! Use these instead of inline `for &(ek, _) in &wire.edges` loops.

use crate::store::BRepStore;
use crate::topo::*;
use std::collections::HashSet;

/// Iterate all WireKeys of a face (outer_wire + inner_wires).
pub fn iter_wires_of_face(face: FaceKey, reg: &BRepStore) -> Vec<WireKey> {
    let Some(face) = reg.faces.get(face) else {
        return vec![];
    };
    let mut wires = Vec::with_capacity(1 + face.inner_wires.len());
    wires.push(face.outer_wire);
    wires.extend_from_slice(&face.inner_wires);
    wires
}

/// Iterate all (EdgeKey, Orientation) pairs in a face's wires.
pub fn iter_edge_orientations_of_face(
    face: FaceKey,
    reg: &BRepStore,
) -> Vec<(EdgeKey, Orientation)> {
    iter_wires_of_face(face, reg)
        .iter()
        .filter_map(|wk| reg.wires.get(*wk))
        .flat_map(|w| w.edges.iter().copied())
        .collect()
}

/// Iterate all EdgeKeys in a face's wires (deduplicated).
pub fn iter_edges_of_face(face: FaceKey, reg: &BRepStore) -> Vec<EdgeKey> {
    iter_edge_orientations_of_face(face, reg)
        .iter()
        .map(|(ek, _)| *ek)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

/// Iterate all unique EdgeKeys in a shell (deduplicated).
pub fn iter_edges_of_shell(shell: ShellKey, reg: &BRepStore) -> Vec<EdgeKey> {
    let Some(shell) = reg.shells.get(shell) else {
        return vec![];
    };
    let mut set = HashSet::new();
    for &(fk, _) in &shell.faces {
        for ek in iter_edges_of_face(fk, reg) {
            set.insert(ek);
        }
    }
    set.into_iter().collect()
}

/// Iterate all (FaceKey, Orientation) pairs in a shell.
pub fn iter_face_orientations_of_shell(
    shell: ShellKey,
    reg: &BRepStore,
) -> Vec<(FaceKey, Orientation)> {
    let Some(shell) = reg.shells.get(shell) else {
        return vec![];
    };
    shell.faces.clone()
}

/// Iterate all FaceKeys in a shell (without orientation).
pub fn iter_faces_of_shell(shell: ShellKey, reg: &BRepStore) -> Vec<FaceKey> {
    iter_face_orientations_of_shell(shell, reg)
        .iter()
        .map(|(fk, _)| *fk)
        .collect()
}

/// Get the two endpoint vertices of an edge.
pub fn iter_vertices_of_edge(edge: EdgeKey, reg: &BRepStore) -> Option<(VertexKey, VertexKey)> {
    reg.edges.get(edge).map(|e| (e.v_low, e.v_high))
}

/// Iterate all unique VertexKeys in a face (via its edges).
pub fn iter_vertices_of_face(face: FaceKey, reg: &BRepStore) -> Vec<VertexKey> {
    let mut set = HashSet::new();
    for ek in iter_edges_of_face(face, reg) {
        if let Some((vl, vh)) = iter_vertices_of_edge(ek, reg) {
            set.insert(vl);
            set.insert(vh);
        }
    }
    set.into_iter().collect()
}

/// Collect all unique EdgeKeys from a shell into a Vec (convenience wrapper).
pub fn shell_edge_keys(shell: ShellKey, reg: &BRepStore) -> Vec<EdgeKey> {
    iter_edges_of_shell(shell, reg)
}

/// Collect all unique FaceKeys from a solid (outer + void shells).
pub fn iter_faces_of_solid(solid: SolidKey, reg: &BRepStore) -> Vec<FaceKey> {
    let Some(solid) = reg.solids.get(solid) else {
        return vec![];
    };
    let mut faces = iter_faces_of_shell(solid.outer_shell, reg);
    for &void_shell in &solid.void_shells {
        faces.extend(iter_faces_of_shell(void_shell, reg));
    }
    faces
}
```

- [ ] **Step 2: Register the module**

In `crates/rc3d-shape/src/lib.rs`, add after `pub mod topo;`:

```rust
pub mod topo_iter;
```

- [ ] **Step 3: Run cargo check**

Run: `cargo check -p rc3d-shape`
Expected: Compiles with no errors.

- [ ] **Step 4: Commit**

```bash
git add crates/rc3d-shape/src/topo_iter.rs crates/rc3d-shape/src/lib.rs
git commit -m "feat(topo): add TopExp-style topology traversal module"
```

---

### Task 2: Write tests for all iterator functions

**Files:**
- Modify: `crates/rc3d-shape/src/topo_iter.rs` — add `#[cfg(test)] mod tests`

- [ ] **Step 1: Add test module to `topo_iter.rs`**

Append to the end of `crates/rc3d-shape/src/topo_iter.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use rc3d_core::math::Vec3;
    use std::collections::HashMap;

    /// Build a minimal shell: 2 faces sharing 1 edge, each face has 1 wire with edges.
    fn setup_two_face_shell(reg: &mut BRepStore) -> (ShellKey, FaceKey, FaceKey, EdgeKey) {
        // 4 vertices forming a rectangle
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::Y, 1e-4);

        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let (lo01, hi01) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        let (lo12, hi12) = if v1 < v2 { (v1, v2) } else { (v2, v1) };
        let (lo30, hi30) = if v3 < v0 { (v3, v0) } else { (v0, v3) };
        let (lo23, hi23) = if v2 < v3 { (v2, v3) } else { (v3, v2) };

        let e01 = reg.edges.insert(BRepEdge {
            curve: line.clone(), tolerance: 1e-4,
            v_low: lo01, v_high: hi01, pcurves: HashMap::new(),
        });
        let e12 = reg.edges.insert(BRepEdge {
            curve: CurveGeom::Line { origin: Vec3::X, direction: Vec3::Y },
            tolerance: 1e-4, v_low: lo12, v_high: hi12, pcurves: HashMap::new(),
        });
        let e23 = reg.edges.insert(BRepEdge {
            curve: CurveGeom::Line { origin: Vec3::Y, direction: Vec3::X },
            tolerance: 1e-4, v_low: lo23, v_high: hi23, pcurves: HashMap::new(),
        });
        let e30 = reg.edges.insert(BRepEdge {
            curve: CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::Y },
            tolerance: 1e-4, v_low: lo30, v_high: hi30, pcurves: HashMap::new(),
        });

        let w_a = reg.wires.insert(BRepWire {
            edges: vec![(e01, Orientation::Forward), (e12, Orientation::Forward),
                        (e23, Orientation::Forward), (e30, Orientation::Forward)],
        });
        let f_a = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X },
            outer_wire: w_a, inner_wires: vec![], same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });

        // Second face sharing edge e01
        let w_b = reg.wires.insert(BRepWire {
            edges: vec![(e01, Orientation::Reversed)],
        });
        let f_b = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Y, u_dir: Vec3::X },
            outer_wire: w_b, inner_wires: vec![], same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });

        let shell = reg.shells.insert(BRepShell {
            faces: vec![(f_a, Orientation::Forward), (f_b, Orientation::Forward)],
            closed: false, step_id: None,
        });

        (shell, f_a, f_b, e01)
    }

    #[test]
    fn test_iter_wires_of_face() {
        let mut reg = BRepStore::new();
        let (_, f_a, _, _) = setup_two_face_shell(&mut reg);
        let wires = iter_wires_of_face(f_a, &reg);
        assert_eq!(wires.len(), 1, "face A should have 1 wire (outer only)");
    }

    #[test]
    fn test_iter_edges_of_face() {
        let mut reg = BRepStore::new();
        let (_, f_a, _, _) = setup_two_face_shell(&mut reg);
        let edges = iter_edges_of_face(f_a, &reg);
        assert_eq!(edges.len(), 4, "face A should have 4 unique edges");
    }

    #[test]
    fn test_iter_edge_orientations_of_face() {
        let mut reg = BRepStore::new();
        let (_, f_a, _, _) = setup_two_face_shell(&mut reg);
        let eo = iter_edge_orientations_of_face(f_a, &reg);
        assert_eq!(eo.len(), 4, "face A has 4 oriented edges");
        assert!(eo.iter().all(|(_, o)| *o == Orientation::Forward));
    }

    #[test]
    fn test_iter_edges_of_shell_dedup() {
        let mut reg = BRepStore::new();
        let (shell, _, _, e_shared) = setup_two_face_shell(&mut reg);
        let edges = iter_edges_of_shell(shell, &reg);
        // face A has 4 edges, face B has 1 edge (e_shared), total unique = 4
        assert!(edges.contains(&e_shared));
        assert_eq!(edges.len(), 4, "shell should have 4 unique edges (e01 shared)");
    }

    #[test]
    fn test_iter_faces_of_shell() {
        let mut reg = BRepStore::new();
        let (shell, f_a, f_b, _) = setup_two_face_shell(&mut reg);
        let faces = iter_faces_of_shell(shell, &reg);
        assert_eq!(faces.len(), 2);
        assert!(faces.contains(&f_a));
        assert!(faces.contains(&f_b));
    }

    #[test]
    fn test_iter_vertices_of_edge() {
        let mut reg = BRepStore::new();
        let (_, _, _, e01) = setup_two_face_shell(&mut reg);
        let (vl, vh) = iter_vertices_of_edge(e01, &reg).unwrap();
        assert!(vl < vh, "v_low should be < v_high");
    }

    #[test]
    fn test_iter_vertices_of_face() {
        let mut reg = BRepStore::new();
        let (_, f_a, _, _) = setup_two_face_shell(&mut reg);
        let verts = iter_vertices_of_face(f_a, &reg);
        assert_eq!(verts.len(), 4, "rectangle face should have 4 vertices");
    }

    #[test]
    fn test_missing_key_returns_empty() {
        let reg = BRepStore::new();
        let fake_face = FaceKey::from(slotmap::KeyData::from_ffi(1));
        assert!(iter_edges_of_face(fake_face, &reg).is_empty());
        assert!(iter_wires_of_face(fake_face, &reg).is_empty());
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p rc3d-shape -- topo_iter`
Expected: All 8 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/rc3d-shape/src/topo_iter.rs
git commit -m "test(topo): add tests for TopExp-style iterator functions"
```

---

### Task 3: Replace inline traversals in heal module (10 call sites)

**Files:**
- Modify: `crates/rc3d-shape/src/heal/check.rs` — replace 8 inline iterations
- Modify: `crates/rc3d-shape/src/heal/degenerated.rs` — replace 2 inline iterations
- Modify: `crates/rc3d-shape/src/heal/seam.rs` — replace 2 inline iterations

- [ ] **Step 1: Replace inline iterations in `heal/check.rs`**

Add import at top of file:
```rust
use crate::topo_iter;
```

For each occurrence of the pattern:
```rust
for wire_key in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
    let Some(wire) = reg.wires.get(*wire_key) else { continue };
    for &(ek, _) in &wire.edges {
```

Replace with:
```rust
for ek in topo_iter::iter_edges_of_face(fk, reg) {
```

Where `fk` is the FaceKey variable used in the surrounding context. If Orientation is needed, use `topo_iter::iter_edge_orientations_of_face` instead.

Do this for all 8 occurrences in check.rs.

- [ ] **Step 2: Replace inline iterations in `heal/degenerated.rs`**

Add import:
```rust
use crate::topo_iter;
```

Replace the 2 occurrences of the wire iteration pattern with `topo_iter::iter_edges_of_face`.

- [ ] **Step 3: Replace inline iterations in `heal/seam.rs`**

Add import:
```rust
use crate::topo_iter;
```

Replace the 2 occurrences with appropriate topo_iter calls.

- [ ] **Step 4: Run full test suite**

Run: `cargo test -p rc3d-shape`
Expected: All 237 tests pass (same count as before). The replacements are purely structural — same loops, same results.

- [ ] **Step 5: Commit**

```bash
git add crates/rc3d-shape/src/heal/check.rs crates/rc3d-shape/src/heal/degenerated.rs crates/rc3d-shape/src/heal/seam.rs
git commit -m "refactor(heal): replace inline topology iterations with topo_iter functions"
```
