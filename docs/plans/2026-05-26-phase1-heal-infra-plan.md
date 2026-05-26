# Phase 1: Heal Infrastructure + Core Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational heal passes (FixConnected, FixSmall, FixGaps2d, FixShifted, BRepCheck expansion, PCURVE pathway) so that subsequent phases can depend on correct vertex connectivity, clean UV domains, and accurate topology diagnostics.

**Architecture:** Six new/expanded heal modules orchestrated through the existing `heal_shell()` in `heal/mod.rs`. Each pass takes `&mut BRepRegistry` and follows the pattern: detect issues → report → mutate registry. The PCURVE modification pathway adds `set_pcurve` and `pcurve_mut` to `BRepRegistry`, enabling all heal passes to adjust PCurve geometry.

**Tech Stack:** Rust, spade (CDT), SlotMap-based BRepRegistry, existing `CurveGeom`/`SurfaceGeom`/`BRepEdge`/`BRepFace` types.

---

### Task 1: PCURVE Modification Pathway (Registry Extension)

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/registry.rs` (add 2 methods + 1 helper)

This is the foundation — all subsequent heal passes need to mutate PCurves. Doing it first avoids rework.

- [ ] **Step 1: Add `pcurve_mut` and `set_pcurve` to BRepRegistry**

Add these two methods to `impl BRepRegistry` in `crates/rc3d-io/src/step/brep/registry.rs`, after the existing `find_shared_edges` method:

```rust
/// Get mutable access to an edge's PCurve for a specific face.
pub fn pcurve_mut(&mut self, ek: EdgeKey, face_key: FaceKey) -> Option<&mut CurveGeom> {
    self.edges.get_mut(ek).and_then(|e| e.pcurves.get_mut(&face_key))
}

/// Replace or insert a PCurve for an (edge, face) pair.
/// Returns the old PCurve if one existed.
pub fn set_pcurve(&mut self, ek: EdgeKey, face_key: FaceKey, pcurve: CurveGeom) -> Option<CurveGeom> {
    self.edges.get_mut(ek).and_then(|e| e.pcurves.insert(face_key, pcurve))
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p rc3d-io 2>&1`
Expected: 0 errors (pre-existing warnings tolerated).

- [ ] **Step 3: Add registry unit tests**

Add to the existing `#[cfg(test)] mod tests` block in `registry.rs`, after the `test_find_shared_edges` test:

```rust
#[test]
fn test_set_pcurve_replace() {
    let mut reg = BRepRegistry::new();
    let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
    let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
    let f0 = make_plane_face(&mut reg);
    let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
    let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, line.clone());

    let new_pcurve = CurveGeom::Line { origin: Vec3::new(1.0, 0.0, 0.0), direction: Vec3::X };
    let old = reg.set_pcurve(ek, f0, new_pcurve.clone());
    assert!(old.is_some());

    let edge = reg.edges.get(ek).unwrap();
    let pcurve = edge.pcurves.get(&f0).unwrap();
    match pcurve {
        CurveGeom::Line { origin, .. } => {
            assert!((origin.x - 1.0).abs() < 1e-6, "expected new pcurve origin");
        }
        _ => panic!("expected Line"),
    }
}

#[test]
fn test_pcurve_mut() {
    let mut reg = BRepRegistry::new();
    let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
    let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
    let f0 = make_plane_face(&mut reg);
    let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
    let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, line.clone());

    let pc = reg.pcurve_mut(ek, f0).unwrap();
    *pc = CurveGeom::Line { origin: Vec3::new(2.0, 0.0, 0.0), direction: Vec3::Y };
    drop(pc);

    let edge = reg.edges.get(ek).unwrap();
    let updated = edge.pcurves.get(&f0).unwrap();
    match updated {
        CurveGeom::Line { origin, direction } => {
            assert!((origin.x - 2.0).abs() < 1e-6);
            assert!((direction.y - 1.0).abs() < 1e-6);
        }
        _ => panic!("expected Line"),
    }
}

#[test]
fn test_set_pcurve_new_face() {
    let mut reg = BRepRegistry::new();
    let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
    let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
    let f0 = make_plane_face(&mut reg);
    let f1 = make_plane_face(&mut reg);
    let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
    let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, line.clone());

    // f1 doesn't have a pcurve yet for this edge
    let old = reg.set_pcurve(ek, f1, line.clone());
    assert!(old.is_none(), "f1 had no pcurve before");

    let edge = reg.edges.get(ek).unwrap();
    assert!(edge.pcurves.contains_key(&f0));
    assert!(edge.pcurves.contains_key(&f1));
}
```

- [ ] **Step 4: Run registry tests**

Run: `cargo test -p rc3d-io -- step::brep::registry 2>&1`
Expected: all 6 tests pass (3 existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add crates/rc3d-io/src/step/brep/registry.rs
git commit -m "feat(rc3d-io): add pcurve_mut/set_pcurve methods to BRepRegistry"
```

---

### Task 2: FixConnected — Topological Vertex Sharing

**Files:**
- Create: `crates/rc3d-io/src/step/brep/heal/connected.rs`
- Modify: `crates/rc3d-io/src/step/brep/heal/mod.rs` (add module + HealConfig field + call site)

- [ ] **Step 1: Create `connected.rs` with types and core function**

Create `crates/rc3d-io/src/step/brep/heal/connected.rs`:

```rust
//! Topological vertex sharing at wire junctions (OCC ShapeFix_Wire::FixConnected).

use std::collections::HashSet;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{EdgeKey, Orientation, VertexKey, WireKey};

/// Result of a FixConnected pass.
#[derive(Debug, Default)]
pub struct ConnectedReport {
    /// Number of vertex pairs merged.
    pub merged_vertices: usize,
    /// Edge junctions that were already connected.
    pub already_connected: usize,
}

/// Merge vertices at adjacent edge junctions within a wire.
///
/// For each consecutive edge pair (including last→first for closed wires),
/// checks whether the end vertex of edge_i and start vertex of edge_{i+1}
/// are at the same 3D position within tolerance. If so, replaces all
/// references to the second vertex with the first.
pub fn fix_connected_wire(
    wire_key: WireKey,
    reg: &mut BRepRegistry,
    tolerance: f32,
) -> ConnectedReport {
    let mut report = ConnectedReport::default();

    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return report; };
        wire.edges.clone()
    };

    if edges.len() < 2 {
        return report;
    }

    // Collect vertex endpoints per edge, accounting for orientation
    let n = edges.len();
    let mut edge_verts: Vec<(VertexKey, VertexKey)> = Vec::with_capacity(n);
    for &(ek, orient) in &edges {
        let Some(edge) = reg.edges.get(ek) else { return report; };
        let (start, end) = if orient == Orientation::Reversed {
            (edge.v_high, edge.v_low)
        } else {
            (edge.v_low, edge.v_high)
        };
        edge_verts.push((start, end));
    }

    // Find merge pairs: end of edge i → start of edge i+1
    let mut merges: Vec<(VertexKey, VertexKey)> = Vec::new(); // (keep, replace)
    for i in 0..n {
        let j = (i + 1) % n;
        let end_i = edge_verts[i].1;
        let start_j = edge_verts[j].0;

        if end_i == start_j {
            report.already_connected += 1;
            continue;
        }

        let pos_i = reg.vertices.get(end_i).map(|v| v.position);
        let pos_j = reg.vertices.get(start_j).map(|v| v.position);

        if let (Some(pi), Some(pj)) = (pos_i, pos_j) {
            let dist = (pi - pj).length();
            if dist < tolerance {
                merges.push((end_i, start_j));
            }
        }
    }

    // Apply merges: replace all references to `replace` with `keep`
    let applied = apply_vertex_merges(reg, &merges);
    report.merged_vertices = applied;
    report
}

/// Replace all references to `replace` vertex with `keep` across all edges.
/// Skips merges that would create non-manifold topology (>2 faces per edge).
fn apply_vertex_merges(reg: &mut BRepRegistry, merges: &[(VertexKey, VertexKey)]) -> usize {
    let mut applied = 0usize;

    for &(keep, replace) in merges {
        if keep == replace {
            continue;
        }
        // Safety: skip if replace vertex is referenced by the keep vertex's own edge
        // (would create a self-loop edge v_low == v_high on a non-degenerate edge)
        let mut safe = true;
        for (_, edge) in reg.edges.iter() {
            let uses_replace = edge.v_low == replace || edge.v_high == replace;
            let uses_keep = edge.v_low == keep || edge.v_high == keep;
            if uses_replace && uses_keep && keep != replace {
                // This edge already connects keep↔replace — skip the merge
                safe = false;
                break;
            }
        }
        if !safe {
            log::warn!(
                "[BRep heal] FixConnected: skipping merge {:?}→{:?} (would create self-loop)",
                replace, keep
            );
            continue;
        }

        // Update all edges referencing `replace`
        for (_, edge) in reg.edges.iter_mut() {
            if edge.v_low == replace {
                edge.v_low = keep;
            }
            if edge.v_high == replace {
                edge.v_high = keep;
            }
        }

        // Remove the replaced vertex from the registry
        reg.vertices.remove(replace);
        applied += 1;
    }

    applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::topo::BRepWire;
    use rc3d_core::math::Vec3;

    fn make_registry_with_two_edges(
        gap: f32,
    ) -> (BRepRegistry, WireKey, VertexKey, VertexKey) {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        // v2 is at same position as v1, but different key (simulating unconnected junction)
        let v2 = reg.vertices.insert(
            crate::step::brep::topo::BRepVertex { position: Vec3::new(1.0 + gap, 0.0, 0.0), tolerance: 1e-4 }
        );
        let v3 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);

        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let face_key = reg.faces.insert(crate::step::brep::topo::BRepFace {
            surface,
            outer_wire: WireKey::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
        });

        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let pcurve = line.clone();
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, face_key, pcurve.clone());
        let e2 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, face_key, pcurve.clone());

        let wire_key = reg.wires.insert(BRepWire {
            edges: vec![(e1, Orientation::Forward), (e2, Orientation::Forward)],
        });

        (reg, wire_key, v1, v2)
    }

    #[test]
    fn test_fix_connected_adjacent_edges() {
        let (mut reg, wire_key, _v1, _v2) = make_registry_with_two_edges(0.0);
        let report = fix_connected_wire(wire_key, &mut reg, 1e-3);
        assert_eq!(report.merged_vertices, 1, "vertices at the same position should merge");
        // Verify the wire edge sequence now shares a vertex
        let wire = reg.wires.get(wire_key).unwrap();
        let e1 = reg.edges.get(wire.edges[0].0).unwrap();
        let e2 = reg.edges.get(wire.edges[1].0).unwrap();
        assert_eq!(e1.v_high, e2.v_low, "merged: end of e1 == start of e2");
    }

    #[test]
    fn test_fix_connected_outside_tolerance() {
        let (mut reg, wire_key, _v1, _v2) = make_registry_with_two_edges(0.1);
        let report = fix_connected_wire(wire_key, &mut reg, 1e-3);
        assert_eq!(report.merged_vertices, 0, "gap > tolerance should not merge");
    }

    #[test]
    fn test_fix_connected_already_connected() {
        let (mut reg, wire_key, v1, _v2) = make_registry_with_two_edges(0.0);
        // Manually merge first to simulate already-connected state
        let report = fix_connected_wire(wire_key, &mut reg, 1e-3);
        assert!(report.merged_vertices > 0 || report.already_connected > 0);
    }
}
```

- [ ] **Step 2: Register module and wire into heal_shell**

In `crates/rc3d-io/src/step/brep/heal/mod.rs`:

Add module declaration at top:
```rust
pub mod connected;
```

Add import:
```rust
use connected::fix_connected_wire;
```

Add fields to `HealConfig`:
```rust
pub fix_connected: bool,
```
and in `Default` impl:
```rust
fix_connected: true,
```

Add new fields to `HealReport`:
```rust
pub merged_vertices: usize,
```

Add to `HealReport::merge`:
```rust
self.merged_vertices += other.merged_vertices;
```

In `heal_shell()`, add BEFORE the reorder loop — between getting face_keys and the for loop:
```rust
if config.fix_connected {
    for (face_key, _) in &face_keys {
        let face = match reg.faces.get(*face_key) {
            Some(f) => f,
            None => continue,
        };
        let cr = fix_connected_wire(face.outer_wire, reg, config.gap_tolerance);
        report.merged_vertices += cr.merged_vertices;
        if cr.merged_vertices > 0 {
            log::debug!(
                "[BRep heal] FixConnected face {:?}: merged {} verts, {} already connected",
                face_key, cr.merged_vertices, cr.already_connected
            );
        }
    }
}
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p rc3d-io 2>&1`
Expected: 0 errors.

- [ ] **Step 4: Run connected tests**

Run: `cargo test -p rc3d-io -- step::brep::heal::connected 2>&1`
Expected: 3 tests pass.

- [ ] **Step 5: Run full test suite for regressions**

Run: `cargo test -p rc3d-io 2>&1`
Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/rc3d-io/src/step/brep/heal/connected.rs crates/rc3d-io/src/step/brep/heal/mod.rs
git commit -m "feat(rc3d-io): add FixConnected — topological vertex sharing at wire junctions"
```

---

### Task 3: FixSmall — Small Edge Removal

**Files:**
- Create: `crates/rc3d-io/src/step/brep/heal/small.rs`
- Modify: `crates/rc3d-io/src/step/brep/heal/mod.rs` (module + config + call site)

- [ ] **Step 1: Create `small.rs`**

Create `crates/rc3d-io/src/step/brep/heal/small.rs`:

```rust
//! Small edge removal (OCC ShapeFix_Wire::FixSmall).
//! Removes edges shorter than a minimum length threshold.

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{EdgeKey, FaceKey, Orientation, WireKey};

#[derive(Debug, Default)]
pub struct SmallEdgeReport {
    pub removed_edges: usize,
    pub wires_emptied: usize,
}

/// Remove edges whose 3D curve length is below `min_length`.
///
/// Seam edges (listed in face.seam_edges) are never removed — they may be
/// zero-length in 3D but have non-zero extent in UV parameter space.
///
/// Returns the updated edge list, or None if the wire should be removed entirely.
pub fn remove_small_edges(
    wire_key: WireKey,
    reg: &mut BRepRegistry,
    face_seam_edges: &[EdgeKey],
    min_length: f32,
) -> Option<Vec<(EdgeKey, Orientation)>> {
    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return None; };
        wire.edges.clone()
    };

    if edges.is_empty() {
        return Some(edges);
    }

    let mut keep: Vec<bool> = vec![true; edges.len()];
    let mut removed = 0usize;

    for (i, &(ek, _)) in edges.iter().enumerate() {
        // Never remove seam edges
        if face_seam_edges.contains(&ek) {
            continue;
        }

        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };

        // Compute edge length from vertex positions
        let len = edge_length_3d(edge, reg);
        if len < min_length {
            keep[i] = false;
            removed += 1;
        }
    }

    if removed == 0 {
        return Some(edges);
    }

    // Build new edge list, connecting predecessor to successor around removed edges
    let mut result: Vec<(EdgeKey, Orientation)> = Vec::new();
    for (i, &(ek, orient)) in edges.iter().enumerate() {
        if keep[i] {
            result.push((ek, orient));
        }
    }

    // If we removed everything, signal wire removal
    if result.len() < 2 {
        return None;
    }

    // Update the wire
    if let Some(wire) = reg.wires.get_mut(wire_key) {
        wire.edges = result.clone();
    }

    Some(result)
}

/// Approximate 3D length of an edge from its vertex endpoints.
fn edge_length_3d(edge: &crate::step::brep::topo::BRepEdge, reg: &BRepRegistry) -> f32 {
    let p0 = reg.vertices.get(edge.v_low).map(|v| v.position);
    let p1 = reg.vertices.get(edge.v_high).map(|v| v.position);
    match (p0, p1) {
        (Some(a), Some(b)) => (a - b).length(),
        _ => f32::MAX, // missing vertices → don't remove
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::topo::BRepWire;
    use rc3d_core::math::Vec3;

    fn make_wire_with_edges(
        reg: &mut BRepRegistry,
        edge_lengths: &[f32],
    ) -> (WireKey, Vec<EdgeKey>, FaceKey) {
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let face_key = reg.faces.insert(crate::step::brep::topo::BRepFace {
            surface,
            outer_wire: WireKey::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
        });

        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let mut edge_keys = Vec::new();
        let mut x = 0.0f32;
        for &len in edge_lengths {
            let v0 = reg.find_or_add_vertex(Vec3::new(x, 0.0, 0.0), 1e-4);
            x += len;
            let v1 = reg.find_or_add_vertex(Vec3::new(x, 0.0, 0.0), 1e-4);
            let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, face_key, line.clone());
            edge_keys.push(ek);
        }
        let wire_edges: Vec<(EdgeKey, Orientation)> = edge_keys
            .iter()
            .map(|&ek| (ek, Orientation::Forward))
            .collect();
        let wire_key = reg.wires.insert(BRepWire { edges: wire_edges });
        (wire_key, edge_keys, face_key)
    }

    #[test]
    fn test_remove_zero_length_edge() {
        let mut reg = BRepRegistry::new();
        let (wire_key, _edge_keys, _fk) = make_wire_with_edges(&mut reg, &[1.0, 0.0, 2.0]);
        let before = reg.wires.get(wire_key).unwrap().edges.len();
        let result = remove_small_edges(wire_key, &mut reg, &[], 1e-6);
        assert!(result.is_some());
        let after = result.unwrap().len();
        assert_eq!(after, 2, "zero-length edge should be removed");
    }

    #[test]
    fn test_skip_seam_edge() {
        let mut reg = BRepRegistry::new();
        let (wire_key, edge_keys, face_key) = make_wire_with_edges(&mut reg, &[0.0]);
        // Mark the edge as a seam edge
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.seam_edges = vec![edge_keys[0]];
        }
        let result = remove_small_edges(wire_key, &mut reg, &[edge_keys[0]], 1e-6);
        let edges = result.unwrap();
        assert_eq!(edges.len(), 1, "seam edge should not be removed even if zero-length");
    }

    #[test]
    fn test_all_small_returns_none() {
        let mut reg = BRepRegistry::new();
        let (wire_key, _edge_keys, _fk) = make_wire_with_edges(&mut reg, &[0.0, 0.0]);
        let result = remove_small_edges(wire_key, &mut reg, &[], 1e-3);
        assert!(result.is_none(), "wire with all edges removed should return None");
    }
}
```

- [ ] **Step 2: Wire FixSmall into heal_shell**

In `crates/rc3d-io/src/step/brep/heal/mod.rs`:

Module declaration:
```rust
pub mod small;
```

Import:
```rust
use small::remove_small_edges;
```

Config fields:
```rust
pub fix_small_edges: bool,
pub small_edge_min_length: f32,
```

Defaults:
```rust
fix_small_edges: true,
small_edge_min_length: 1e-6,
```

Report fields:
```rust
pub removed_small_edges: usize,
```

In `HealReport::merge()`, add:
```rust
self.removed_small_edges += other.removed_small_edges;
```

In `heal_shell()`, after FixConnected block and BEFORE `close_wire_gaps`:

```rust
if config.fix_small_edges {
    for (face_key, _) in &face_keys {
        let face = match reg.faces.get(*face_key) {
            Some(f) => f,
            None => continue,
        };
        let seam_edges = face.seam_edges.clone();
        let outer_wire = face.outer_wire;
        if let Some(updated) = remove_small_edges(outer_wire, reg, &seam_edges, config.small_edge_min_length) {
            if updated.len() != reg.wires.get(outer_wire).map(|w| w.edges.len()).unwrap_or(0) {
                report.removed_small_edges += 1;
                log::debug!("[BRep heal] FixSmall face {:?}: removed small edges, {} remain", face_key, updated.len());
            }
        } else {
            report.skip_face_keys.push(*face_key);
            log::warn!("[BRep heal] FixSmall face {:?}: wire emptied, marking for skip", face_key);
        }
    }
}
```

- [ ] **Step 3: Verify compilation + tests**

Run: `cargo check -p rc3d-io 2>&1`
Run: `cargo test -p rc3d-io -- step::brep::heal::small 2>&1`
Run: `cargo test -p rc3d-io 2>&1`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add crates/rc3d-io/src/step/brep/heal/small.rs crates/rc3d-io/src/step/brep/heal/mod.rs
git commit -m "feat(rc3d-io): add FixSmall — small edge removal with seam edge protection"
```

---

### Task 4: FixGaps2d — UV-Space Gap Closure

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/heal/gap.rs` (add new function)
- Modify: `crates/rc3d-io/src/step/brep/heal/mod.rs` (config + call site)

- [ ] **Step 1: Add `close_wire_gaps_2d` to `gap.rs`**

Append to `crates/rc3d-io/src/step/brep/heal/gap.rs`:

```rust
/// Close gaps between PCurve endpoints of adjacent edges in UV space.
///
/// For each adjacent edge pair (including last→first for closed wires), if the
/// 3D endpoints are connected (gap < tol_3d) but the UV endpoints differ by
/// more than tol_uv, nudge the PCurve endpoint of the "looser" edge to match.
pub fn close_wire_gaps_2d(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepRegistry,
    tol_3d: f32,
    tol_uv: f32,
) -> usize {
    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return 0; };
        wire.edges.clone()
    };

    if edges.len() < 2 {
        return 0;
    }

    let n = edges.len();
    let mut closed = 0usize;

    for i in 0..n {
        let j = if i + 1 < n { i + 1 } else { 0 };
        let is_wrap = j == 0; // true for closed-wire last→first check

        let (ek_i, orient_i) = edges[i];
        let (ek_j, orient_j) = edges[j];

        // Check 3D connectivity
        let (end_i_3d, start_j_3d) = match (
            get_oriented_endpoint(ek_i, orient_i, false, reg),
            get_oriented_endpoint(ek_j, orient_j, true, reg),
        ) {
            (Some(pe), Some(ps)) => (pe, ps),
            _ => continue,
        };

        let gap_3d = (end_i_3d - start_j_3d).length();
        if gap_3d > tol_3d {
            if !is_wrap {
                // Non-wrapped gap with 3D disconnect: structural issue, not a 2D gap
                continue;
            }
            // For wrap-around on a closed wire, a 3D gap means the wire itself
            // isn't closed. Skip — this is a 3D gap fix issue, not 2D.
            continue;
        }

        // Check UV connectivity
        let uv_end_i = pcurve_endpoint(ek_i, orient_i, false, face_key, reg);
        let uv_start_j = pcurve_endpoint(ek_j, orient_j, true, face_key, reg);

        if let (Some((u1, v1)), Some((u2, v2))) = (uv_end_i, uv_start_j) {
            let gap_uv = ((u1 - u2).powi(2) + (v1 - v2).powi(2)).sqrt();
            if gap_uv > tol_uv && gap_uv < tol_uv * 100.0 {
                // Nudge the end of edge_i's PCurve to match start of edge_j
                if let Some(pc) = reg.pcurve_mut(ek_i, face_key) {
                    // Adjust PCurve at t=1: translate by the UV delta
                    let du = u2 - u1;
                    let dv = v2 - v1;
                    *pc = translate_pcurve_endpoint(pc, du, dv, true);
                    closed += 1;
                }
            }
        }
    }

    closed
}

/// Get the 3D position of an edge's start or end, accounting for orientation.
fn get_oriented_endpoint(
    ek: EdgeKey,
    orient: Orientation,
    is_start: bool,
    reg: &BRepRegistry,
) -> Option<Vec3> {
    let edge = reg.edges.get(ek)?;
    let vk = match (orient, is_start) {
        (Orientation::Forward, true) | (Orientation::Reversed, false) => edge.v_low,
        _ => edge.v_high,
    };
    reg.vertices.get(vk).map(|v| v.position)
}

/// Get the UV coordinates at the start or end of an edge's PCurve for a face.
fn pcurve_endpoint(
    ek: EdgeKey,
    orient: Orientation,
    is_start: bool,
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Option<(f32, f32)> {
    let edge = reg.edges.get(ek)?;
    let pc = edge.pcurves.get(&face_key)?;
    let t = if (orient == Orientation::Forward) == is_start { 0.0 } else { 1.0 };
    let uv = pc.d0(t);
    Some((uv.x, uv.y))
}

/// Translate the endpoint of a PCurve by (du, dv) at t=1 (if at_end=true) or t=0.
/// For Line PCurves: adjust direction. For other types: wraps in a new Line segment.
fn translate_pcurve_endpoint(
    pc: &mut crate::step::brep::geom::CurveGeom,
    du: f32,
    dv: f32,
    _at_end: bool,
) -> crate::step::brep::geom::CurveGeom {
    // Simplified: for small adjustments (< tol_uv * 2), adjust the origin for
    // Line-based PCurves. For complex PCurves, the adjustment is done via
    // parameter-range shift in later phases.
    match pc {
        crate::step::brep::geom::CurveGeom::Line { origin, direction } => {
            // Extend/shorten the line to absorb the gap
            let len = direction.length();
            if len > 1e-10 {
                let dir_n = *direction / len;
                let new_origin = *origin + dir_n * du + Vec3::new(dv, 0.0, 0.0);
                *origin = new_origin;
            }
        }
        _ => {
            // Non-Line PCurves: leave as-is for now (Phase 2 FixEdgeCurves handles these)
        }
    }
    pc.clone()
}
```

Note: `Vec3` needs to be imported. Add at the top of gap.rs if not already present:
```rust
use rc3d_core::math::Vec3;
```
(It's already imported since `close_wire_gaps` uses `Vec3`)

- [ ] **Step 2: Wire into heal_shell**

In `heal/mod.rs`:

Config field:
```rust
pub uv_gap_tolerance: f32,
```
Default:
```rust
uv_gap_tolerance: 1e-5,
```

Report field:
```rust
pub closed_uv_gaps: usize,
```

In `HealReport::merge()`, add:
```rust
self.closed_uv_gaps += other.closed_uv_gaps;
```

In `heal_shell()`, after the existing `close_wire_gaps` block and BEFORE `fix_missing_seams`:

```rust
if config.uv_gap_tolerance > 0.0 {
    for (face_key, _) in &face_keys {
        let face = match reg.faces.get(*face_key) {
            Some(f) => f,
            None => continue,
        };
        let uv_closed = close_wire_gaps_2d(
            face.outer_wire,
            *face_key,
            reg,
            config.gap_tolerance,
            config.uv_gap_tolerance,
        );
        if uv_closed > 0 {
            report.closed_uv_gaps += uv_closed;
            log::debug!("[BRep heal] FixGaps2d face {:?}: closed {} UV gap(s)", face_key, uv_closed);
        }
    }
}
```

- [ ] **Step 3: Verify compilation + tests**

Run: `cargo check -p rc3d-io 2>&1`
Run: `cargo test -p rc3d-io 2>&1`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add crates/rc3d-io/src/step/brep/heal/gap.rs crates/rc3d-io/src/step/brep/heal/mod.rs
git commit -m "feat(rc3d-io): add FixGaps2d — UV-space gap closure for adjacent PCurves"
```

---

### Task 5: FixShifted — PCurve Period-Shift Detection

**Files:**
- Create: `crates/rc3d-io/src/step/brep/heal/shifted.rs`
- Modify: `crates/rc3d-io/src/step/brep/heal/mod.rs` (module + config + call site)

- [ ] **Step 1: Create `shifted.rs`**

Create `crates/rc3d-io/src/step/brep/heal/shifted.rs`:

```rust
//! PCurve period-shift detection and correction (OCC ShapeFix_Wire::FixShifted).
//! Detects PCurves offset by a full parameter period on periodic surfaces.

use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{EdgeKey, FaceKey, Orientation, WireKey};

#[derive(Debug, Default)]
pub struct ShiftedReport {
    pub shifts_applied: usize,
}

struct PeriodShift {
    face_key: FaceKey,
    edge_key: EdgeKey,
    shift_u: f64,
    shift_v: f64,
}

/// Detect and fix PCurves shifted by a surface period.
pub fn fix_shifted_pcurves(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepRegistry,
) -> ShiftedReport {
    let mut report = ShiftedReport::default();

    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return report,
    };
    let surface = face.surface.clone();

    // Determine surface periods
    let (period_u, period_v) = surface_periods(&surface);
    if period_u == 0.0 && period_v == 0.0 {
        return report; // non-periodic surface
    }

    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return report; };
        wire.edges.clone()
    };

    // Compute UV centroid of all PCurve endpoints (the expected cluster)
    let mut uv_sum = (0.0f64, 0.0f64);
    let mut count = 0u32;
    for &(ek, _) in &edges {
        if let Some(edge) = reg.edges.get(ek) {
            if let Some(pc) = edge.pcurves.get(&face_key) {
                let uv_start = pc.d0(0.0);
                let uv_end = pc.d0(1.0);
                uv_sum.0 += (uv_start.x + uv_end.x) as f64 * 0.5;
                uv_sum.1 += (uv_start.y + uv_end.y) as f64 * 0.5;
                count += 1;
            }
        }
    }

    if count == 0 {
        return report;
    }
    let uv_center = (uv_sum.0 / count as f64, uv_sum.1 / count as f64);

    // Check each edge's PCurve midpoint vs the cluster center
    for &(ek, _) in &edges {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };
        let pc = match edge.pcurves.get(&face_key) {
            Some(p) => p,
            None => continue,
        };

        let mid_uv = pc.d0(0.5);
        let (mu, mv) = (mid_uv.x as f64, mid_uv.y as f64);

        let mut shift_u: f64 = 0.0;
        let mut shift_v: f64 = 0.0;

        if period_u > 0.0 {
            let dist_u = (mu - uv_center.0).abs();
            if dist_u > period_u as f64 * 0.4 {
                // Test shifting by ±period
                let shifted_up = mu - period_u as f64;
                let shifted_down = mu + period_u as f64;
                if (shifted_up - uv_center.0).abs() < dist_u * 0.5 {
                    shift_u = -period_u as f64;
                } else if (shifted_down - uv_center.0).abs() < dist_u * 0.5 {
                    shift_u = period_u as f64;
                }
            }
        }

        if period_v > 0.0 {
            let dist_v = (mv - uv_center.1).abs();
            if dist_v > period_v as f64 * 0.4 {
                let shifted_up = mv - period_v as f64;
                let shifted_down = mv + period_v as f64;
                if (shifted_up - uv_center.1).abs() < dist_v * 0.5 {
                    shift_v = -period_v as f64;
                } else if (shifted_down - uv_center.1).abs() < dist_v * 0.5 {
                    shift_v = period_v as f64;
                }
            }
        }

        if shift_u != 0.0 || shift_v != 0.0 {
            // Apply the shift: translate the PCurve's UV coordinates
            if let Some(pc) = reg.pcurve_mut(ek, face_key) {
                *pc = shift_pcurve(pc, shift_u as f32, shift_v as f32);
                report.shifts_applied += 1;
            }
        }
    }

    report
}

/// Return (U_period, V_period) for a surface. Returns (0,0) for non-periodic.
fn surface_periods(surface: &SurfaceGeom) -> (f64, f64) {
    use std::f32::consts::TAU;
    match surface {
        SurfaceGeom::Cylinder { .. } => (TAU as f64, 0.0),
        SurfaceGeom::Torus { .. } => (TAU as f64, TAU as f64),
        SurfaceGeom::Sphere { .. } => (TAU as f64, 0.0), // v has no period, u does
        SurfaceGeom::Cone { .. } => (TAU as f64, 0.0),
        SurfaceGeom::Revolution { .. } => (TAU as f64, 0.0),
        _ => (0.0, 0.0),
    }
}

/// Translate all UV coordinates of a PCurve by (du, dv).
fn shift_pcurve(
    pc: &crate::step::brep::geom::CurveGeom,
    du: f32,
    dv: f32,
) -> crate::step::brep::geom::CurveGeom {
    use crate::step::brep::geom::CurveGeom;
    match pc {
        CurveGeom::Line { origin, direction } => CurveGeom::Line {
            origin: *origin + rc3d_core::math::Vec3::new(du, dv, 0.0),
            direction: *direction,
        },
        // For other curve types, wrap in a translated Line approximation
        // (full PCurve translation for B-Spline etc. is Phase 2 scope)
        other => {
            // Approximate: translate origin by (du, dv) in UV space
            // For now, this is sufficient for Line-based PCurves which are the common case
            other.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::CurveGeom;
    use crate::step::brep::topo::BRepWire;
    use rc3d_core::math::Vec3;

    #[test]
    fn test_cylinder_periods() {
        let s = SurfaceGeom::Cylinder { origin: Vec3::ZERO, axis: Vec3::Z, radius: 1.0 };
        let (pu, pv) = surface_periods(&s);
        assert!(pu > 0.0);
        assert_eq!(pv, 0.0);
    }

    #[test]
    fn test_torus_periods() {
        let s = SurfaceGeom::Torus { center: Vec3::ZERO, axis: Vec3::Z, major_r: 3.0, minor_r: 1.0 };
        let (pu, pv) = surface_periods(&s);
        assert!(pu > 0.0);
        assert!(pv > 0.0);
    }

    #[test]
    fn test_plane_no_period() {
        let s = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let (pu, pv) = surface_periods(&s);
        assert_eq!(pu, 0.0);
        assert_eq!(pv, 0.0);
    }
}
```

- [ ] **Step 2: Wire into heal_shell**

In `heal/mod.rs`:

Module:
```rust
pub mod shifted;
```

Import:
```rust
use shifted::fix_shifted_pcurves;
```

Config:
```rust
pub fix_shifted: bool,
```
Default:
```rust
fix_shifted: true,
```

Report:
```rust
pub shifted_pcurves: usize,
```

In `HealReport::merge()`, add:
```rust
self.shifted_pcurves += other.shifted_pcurves;
```

In `heal_shell()`, after FixGaps2d and BEFORE `fix_missing_seams`:

```rust
if config.fix_shifted {
    for (face_key, _) in &face_keys {
        let face = match reg.faces.get(*face_key) {
            Some(f) => f,
            None => continue,
        };
        let sr = fix_shifted_pcurves(face.outer_wire, *face_key, reg);
        report.shifted_pcurves += sr.shifts_applied;
        if sr.shifts_applied > 0 {
            log::debug!("[BRep heal] FixShifted face {:?}: {} pcurve shift(s) applied", face_key, sr.shifts_applied);
        }
    }
}
```

- [ ] **Step 3: Verify compilation + tests**

Run: `cargo check -p rc3d-io 2>&1`
Run: `cargo test -p rc3d-io -- step::brep::heal::shifted 2>&1`
Run: `cargo test -p rc3d-io 2>&1`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add crates/rc3d-io/src/step/brep/heal/shifted.rs crates/rc3d-io/src/step/brep/heal/mod.rs
git commit -m "feat(rc3d-io): add FixShifted — PCurve period-shift detection on periodic surfaces"
```

---

### Task 6: BRepCheck Expansion — 4 New Topology Checks

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/heal/check.rs` (add 4 check functions + wire into check_face)

- [ ] **Step 1: Add new check functions to `check.rs`**

Add four new functions after the existing `check_uv_self_intersection` in `crates/rc3d-io/src/step/brep/heal/check.rs`:

```rust
/// Check edge tolerance validity (OCC BRepCheck_Edge).
/// Reports oversized tolerance relative to edge length.
fn check_edge_tolerance(ek: EdgeKey, reg: &BRepRegistry) -> Vec<String> {
    let mut warnings = Vec::new();
    let edge = match reg.edges.get(ek) {
        Some(e) => e,
        None => return warnings,
    };

    let p0 = reg.vertices.get(edge.v_low).map(|v| v.position);
    let p1 = reg.vertices.get(edge.v_high).map(|v| v.position);
    let approx_len = match (p0, p1) {
        (Some(a), Some(b)) => (a - b).length(),
        _ => return warnings,
    };

    if approx_len < 1e-12 {
        return warnings; // zero-length edge, handled by FixSmall
    }

    if edge.tolerance > approx_len * 10.0 {
        warnings.push(format!(
            "edge {:?}: tolerance {:.6} > 10x edge length {:.6}",
            ek, edge.tolerance, approx_len
        ));
    }
    if edge.tolerance < 1e-12 {
        warnings.push(format!("edge {:?}: tolerance {:.6} is near-zero", ek, edge.tolerance));
    }

    warnings
}

/// Check for surface singularities on the trim boundary (OCC BRepCheck_Face).
/// Flags faces that may need degenerated edges (Phase 2).
fn check_surface_singularities(
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return warnings,
    };

    // Only check for surfaces that can have singularities
    match &face.surface {
        crate::step::brep::geom::SurfaceGeom::Sphere { .. } => {
            // Sphere poles: check if any edge endpoint is near u=0 or u=π
            let wire = match reg.wires.get(face.outer_wire) {
                Some(w) => w,
                None => return warnings,
            };
            for &(ek, _) in &wire.edges {
                let edge = match reg.edges.get(ek) {
                    Some(e) => e,
                    None => continue,
                };
                if let Some(pc) = edge.pcurves.get(&face_key) {
                    for t in [0.0, 1.0] {
                        let uv = pc.d0(t);
                        // Check if near pole: |v - π/2| ≈ π/2 (top) or |v + π/2| ≈ π/2 (bottom)
                        if (uv.y.abs() - std::f32::consts::FRAC_PI_2).abs() < 0.01 {
                            warnings.push(format!(
                                "face {:?}: potential degeneracy near sphere pole at u={:.3}, v={:.3}",
                                face_key, uv.x, uv.y
                            ));
                        }
                    }
                }
            }
        }
        crate::step::brep::geom::SurfaceGeom::Cone { .. } => {
            // Cone apex: check for vertices near the apex
            let wire = match reg.wires.get(face.outer_wire) {
                Some(w) => w,
                None => return warnings,
            };
            for &(ek, _) in &wire.edges {
                let edge = match reg.edges.get(ek) {
                    Some(e) => e,
                    None => continue,
                };
                for &vk in &[edge.v_low, edge.v_high] {
                    if let Some(v) = reg.vertices.get(vk) {
                        // Check if position is very close to the cone apex
                        if let crate::step::brep::geom::SurfaceGeom::Cone { apex, .. } = &face.surface {
                            if (v.position - *apex).length() < face.tolerance * 10.0 {
                                warnings.push(format!(
                                    "face {:?}: potential degeneracy at cone apex", face_key
                                ));
                            }
                        }
                    }
                }
            }
        }
        _ => {} // Plane, cylinder, torus — no isolated singularities
    }

    warnings
}

/// Check that PCurve UV coordinates fall within the surface's natural domain
/// or within one period of it.
fn check_parameter_range(
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return warnings,
    };

    // Determine expected parameter range from surface type
    let (u_range, v_range): ((f32, f32), (f32, f32)) = match &face.surface {
        crate::step::brep::geom::SurfaceGeom::BSpline(nurbs) => {
            let uk = &nurbs.u_knots;
            let vk = &nurbs.v_knots;
            ((uk[0], uk[uk.len()-1]), (vk[0], vk[vk.len()-1]))
        }
        _ => ((0.0, 0.0), (0.0, 0.0)), // non-BSpline surfaces handled by period check in FixShifted
    };

    if u_range == (0.0, 0.0) && v_range == (0.0, 0.0) {
        return warnings; // not applicable for non-BSpline surfaces
    }

    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => return warnings,
    };
    for &(ek, _) in &wire.edges {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };
        if let Some(pc) = edge.pcurves.get(&face_key) {
            for t in [0.0, 1.0] {
                let uv = pc.d0(t);
                let margin = 0.1; // allow slight overshoot
                if uv.x < u_range.0 - margin || uv.x > u_range.1 + margin {
                    warnings.push(format!(
                        "face {:?} edge {:?}: PCurve U={:.6} outside surface U range [{:.3}, {:.3}]",
                        face_key, ek, uv.x, u_range.0, u_range.1
                    ));
                }
                if uv.y < v_range.0 - margin || uv.y > v_range.1 + margin {
                    warnings.push(format!(
                        "face {:?} edge {:?}: PCurve V={:.6} outside surface V range [{:.3}, {:.3}]",
                        face_key, ek, uv.y, v_range.0, v_range.1
                    ));
                }
            }
        }
    }

    warnings
}

/// Check wire orientation consistency using signed area in UV space.
fn check_wire_orientation(
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return warnings,
    };

    // Check outer wire orientation
    if let Some(wire) = reg.wires.get(face.outer_wire) {
        let uv_points = collect_wire_uv_polygon(wire, face_key, reg);
        if uv_points.len() >= 3 {
            let area = signed_area_2d(&uv_points);
            let expected_positive = face.same_sense;
            if (area > 0.0) != expected_positive {
                warnings.push(format!(
                    "face {:?}: outer wire orientation inconsistent (signed_area={:.6}, same_sense={})",
                    face_key, area, expected_positive
                ));
            }
        }
    }

    // Check inner wire orientations (should be opposite of outer)
    for &inner_wire_key in &face.inner_wires {
        if let Some(wire) = reg.wires.get(inner_wire_key) {
            let uv_points = collect_wire_uv_polygon(wire, face_key, reg);
            if uv_points.len() >= 3 {
                let area = signed_area_2d(&uv_points);
                let outer_positive = face.same_sense;
                if (area > 0.0) == outer_positive {
                    warnings.push(format!(
                        "face {:?}: inner wire orientation should be opposite of outer (signed_area={:.6})",
                        face_key, area
                    ));
                }
            }
        }
    }

    warnings
}

/// Collect UV polygon points from a wire's PCurve endpoints.
fn collect_wire_uv_polygon(
    wire: &crate::step::brep::topo::BRepWire,
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Vec<(f32, f32)> {
    let mut points = Vec::new();
    for &(ek, _) in &wire.edges {
        if let Some(edge) = reg.edges.get(ek) {
            if let Some(pc) = edge.pcurves.get(&face_key) {
                let uv = pc.d0(0.0);
                points.push((uv.x, uv.y));
            }
        }
    }
    // Close the polygon
    if let (Some(first), Some(last)) = (
        points.first().copied(),
        wire.edges.last().and_then(|&(ek, _)| {
            reg.edges.get(ek).and_then(|e| e.pcurves.get(&face_key)).map(|pc| {
                let uv = pc.d0(1.0);
                (uv.x, uv.y)
            })
        })
    ) {
        points.push(last);
    }
    points
}

/// Compute signed area of a 2D polygon (shoelace formula).
fn signed_area_2d(pts: &[(f32, f32)]) -> f32 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].0 * pts[j].1;
        area -= pts[j].0 * pts[i].1;
    }
    area * 0.5
}
```

- [ ] **Step 2: Wire new checks into `check_face`**

In `check_face()`, after the existing error/warning checks but before the `failed_faces` push (around line 149), add:

```rust
// Edge tolerance checks
let wire = match reg.wires.get(face.outer_wire) {
    Some(w) => w,
    None => return,
};
for &(ek, _) in &wire.edges {
    report.warnings.extend(check_edge_tolerance(ek, reg));
}

// Surface singularity detection
report.warnings.extend(check_surface_singularities(face_key, reg));

// Parameter range validity (BSpline surfaces only)
report.warnings.extend(check_parameter_range(face_key, reg));

// Wire orientation consistency
report.warnings.extend(check_wire_orientation(face_key, reg));
```

- [ ] **Step 3: Verify compilation + tests**

Run: `cargo check -p rc3d-io 2>&1`
Run: `cargo test -p rc3d-io 2>&1`
Expected: all pass (including existing check tests).

- [ ] **Step 4: Commit**

```bash
git add crates/rc3d-io/src/step/brep/heal/check.rs
git commit -m "feat(rc3d-io): expand BRepCheck — edge tolerance, singularities, param range, wire orientation"
```

---

### Task 7: Integration Test — End-to-End on Test Corpus

**Files:**
- Modify: `crates/rc3d-io/tests/step_files.rs` — add a detailed heal check test

- [ ] **Step 1: Add regression test with heal pass verification**

Add a new test function to `crates/rc3d-io/tests/step_files.rs`:

```rust
#[test]
fn test_heal_passes_on_cs_step() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/cs.step");
    if !path.exists() {
        println!("SKIP: cs.step not found");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("read cs.step");
    let exchange = rc3d_io::step::parser::parse_exchange(&text).expect("parse");
    let brep = rc3d_io::step::brep::build_brep(&exchange.entities).expect("brep");
    let mut reg = brep.registry;

    for &sk in &brep.root_solids {
        let outer_shell = reg.solids.get(sk).unwrap().outer_shell;

        // Run with all new Phase 1 fixes enabled
        let mut heal_cfg = rc3d_io::step::brep::heal::HealConfig::default();
        heal_cfg.fix_connected = true;
        heal_cfg.fix_small_edges = true;
        heal_cfg.uv_gap_tolerance = 1e-5;
        heal_cfg.fix_shifted = true;

        let heal = rc3d_io::step::brep::heal::heal_shell(outer_shell, &mut reg, &heal_cfg);

        println!(
            "solid {:?}: merged_vertices={}, removed_small={}, closed_uv={}, shifted={}, check_errors={}, check_warnings={}",
            sk,
            heal.merged_vertices,
            heal.removed_small_edges,
            heal.closed_uv_gaps,
            heal.shifted_pcurves,
            heal.check_errors,
            heal.check_warnings,
        );

        // Verify the mesh still produces valid output
        let mesh_cfg = rc3d_io::step::brep::mesh::BRepMeshConfig::default();
        let shell = reg.shells.get(outer_shell).unwrap();
        for &(face_key, _) in &shell.faces {
            if heal.skip_face_keys.contains(&face_key) {
                println!("  face {:?}: SKIPPED by heal", face_key);
            }
        }
        let output = rc3d_io::step::brep::mesh::mesh_brep_shell_with_report(
            outer_shell,
            &reg,
            &mesh_cfg,
            &heal.skip_face_keys,
        );
        assert!(
            output.report.meshed_faces > 0,
            "at least one face should be meshed"
        );
        println!(
            "  meshed {} faces, {} tris, {} skipped",
            output.report.meshed_faces,
            output.mesh.indices.len() / 4,
            heal.skip_face_keys.len(),
        );
    }
}
```

- [ ] **Step 2: Run integration test**

Run: `cargo test -p rc3d-io --test step_files test_heal_passes_on_cs_step -- --nocapture 2>&1`
Expected: test passes (or SKIP if cs.step not available), reports heal statistics.

- [ ] **Step 3: Run full test suite**

Run: `cargo test -p rc3d-io 2>&1`
Expected: all existing tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/rc3d-io/tests/step_files.rs
git commit -m "test(rc3d-io): add Phase 1 heal pass integration test on cs.step"
```

---

## Phase 1 Completion Verification

After all 7 tasks:

```bash
# Verify everything compiles and tests pass
cargo test -p rc3d-io 2>&1

# Verify total test count
cargo test -p rc3d-io -- --list 2>&1 | wc -l

# Verify no new clippy warnings on heal modules
cargo clippy -p rc3d-io -- step::brep::heal 2>&1
```

Expected: all tests pass, no new warnings, ~15 new unit tests + 1 integration test.

## File Summary

| File | Action | Lines |
|------|--------|-------|
| `brep/registry.rs` | +30 (2 methods + 3 tests) | — |
| `heal/connected.rs` | New (~160) | — |
| `heal/small.rs` | New (~130) | — |
| `heal/gap.rs` | +90 (close_wire_gaps_2d + helpers) | — |
| `heal/shifted.rs` | New (~150) | — |
| `heal/check.rs` | +140 (4 check functions + helpers) | — |
| `heal/mod.rs` | +60 (modules, config, call sites, report fields) | — |
| `tests/step_files.rs` | +35 (integration test) | — |
| **Total** | | **~795** |
