# Reachability Fix Plan — "Make it Loopy"

Based on: 5-system critical reachability review. Date: 2026-06-18.

## Phase R1: Critical Bug Fixes (geometry correctness)
**3 bugs, ~20 lines total**

1. `curve_eval.rs:793` — Circle/Ellipse d012() missing center → bind `center`, add to d0
2. `curve_eval.rs:288` — de_casteljau_d1 ignores rational weights
3. `project.rs:975` — find_param_on_curve dead params → remove or use

## Phase R2: API Wiring (public visibility)
**5 re-exports, 1 auto-detect**

1. `lib.rs`: add `pub use` for IGES, IGES writer, binary persistence, parametric writer
2. `lib.rs`: add `"iges"` / `"igs"` to `import_file()` auto-detect
3. `step/write/mod.rs`: re-export parametric writer

## Phase R3: Pipeline Connection (dead code → alive)
**4 connections**

1. `bool/mod.rs`: gate BuilderSolid behind `BRepBoolOptions.build_solids` flag
2. `mesh/finalize.rs`: call ModelHealer (heal_mesh_gaps + fix_t_junctions) from finalize_shell_mesh
3. `mesh/init.rs`: call discretize_edges_incremental from init_shell_mesh
4. `heal/pipeline.rs`: call heal_solid() from run_heal_pipeline

## Phase R4: Config Enabling (passes reachable)
**4 config changes**

1. FixNotchedEdges → Basic tier in select_fixes() (iter=0)
2. FixTails → Basic tier in select_fixes() (iter=0)
3. UnifySameDomain → Precision tier in for_tier()
4. FixSmallSolid → Standard tier in for_tier()

## Execution order

R1 (bugs) → R2 (API) → R3 (pipeline) → R4 (config)

R1: parallel (3 independent bugs)
R2: serial (single file, 5 changes)
R3: parallel (4 independent connections)
R4: serial (single file, 4 config changes)
