# Phase 3: Mesh Quality — Implementation Plan

> **For agentic workers:** Use subagent-driven-development

**Goal:** CDT handles degenerated edges correctly. Steiner uses edge midpoints. Intersecting wires and periodic degeneracies are fixed.

**Dependencies:** Phase 2 (degenerated edge entities exist, self-intersection fixed).

---

### Task 1: Degenerated Edge CDT Integration + Steiner Midpoint

**Files:** Modify `mesh/face_cdt.rs`

- Degenerated edges added as CDT constraints (UV extent is real, 3D collapses to point)
- Triangle extraction skips zero-area-in-3D triangles on degenerated edges
- Steiner refinement: edge midpoint insertion when single edge fails deflection
- Sort splits by chord error before truncation

### Task 2: FixIntersectingWires

**Files:** Create `heal/intersecting_wires.rs`, modify `heal/mod.rs`

- Detect outer-inner and inner-inner wire intersections in UV space
- Trim/split intersecting wires, remove inner wires entirely outside outer
- Merge intersecting inner wires

### Task 3: FixPeriodicDegenerated

**Files:** Create `heal/periodic.rs`, modify `heal/mod.rs`

- Detect single wire wrapping full parameter period on periodic surfaces
- Reconstruct degenerated edges at parameter poles
- Must run BEFORE FixMissingSeam

### Task 4: Integration + regression

**Files:** Extend tests

---

**Total:** 1 modified + 2 new + mod.rs, ~470 lines
