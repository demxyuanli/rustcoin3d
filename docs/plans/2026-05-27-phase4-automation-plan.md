# Phase 4: Automation + Advanced Features — Implementation Plan

> **For agentic workers:** Use subagent-driven-development

**Goal:** Iterative auto-heal pipeline, vertex position correction, relative deflection, global properties, continuity check.

**Dependencies:** Phases 1-3 (heal pipeline is feature-complete).

---

### Task 1: HealPipeline — Iterative Auto-Heal

**Files:** Create `heal/pipeline.rs`, modify `heal/mod.rs`

- Iterative heal loop: run checks, apply fixes, re-check, repeat
- Three levels: Basic, Standard, Advanced
- Fix selection based on detected issues
- Max 5 iterations, stop when converged

### Task 2: FixVertexPosition — Vertex Position Correction

**Files:** Create `heal/vertex_position.rs`, modify `heal/mod.rs`

- Project vertices onto associated surfaces/curves
- Skip if no valid projection exists

### Task 3: Relative Deflection Mode

**Files:** Modify `mesh/edge_disc.rs`

- OCC isRelative: deflection = edge_length * factor

### Task 4: Global Properties (BRepGProp)

**Files:** Create `brep/properties.rs`

- Volume, surface area, center of mass from mesh

### Task 5: Continuity Check (G0/G1)

**Files:** Create `heal/continuity.rs`

- Detect G0 (positional gap) and G1 (tangential mismatch) defects

### Task 6: Pipeline Wiring (Gap Closure 2026-05-27)

**Files:** `heal/pipeline.rs`, `heal/mod.rs`, `mod.rs`, `import_options.rs`

- [x] `select_fixes`: enable `fix_vertex_position` (Standard+ iter0), `fix_split_face` (iter>0 + pcurve issues)
- [x] `detect_intersecting_wires` in check → adaptive `fix_intersecting_wires`
- [x] `check_shell_continuity` + `compute_mesh_properties` logged in import path
- [x] `StepImportOptions.mesh_relative_deflection` (default 0.001 for strict/preview import)

---

**Total:** 4 new files + 3 modified, ~400 lines
