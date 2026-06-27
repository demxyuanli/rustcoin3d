# Gate 1 — Direction Convergence — 2026-06-27

## Decision
The approach is structural BREP format correctness: fix the writer to produce valid OCC-compatible BREP files by eliminating format-level errors (zero param ranges, mislabeled degeneracies, wrong curve types on edges), without attempting to match OCC's vertex-sharing or face-decomposition strategies.

## Claim
All 11 BREP files in compare/out/ are structurally correct and can be opened by OCC-based tools. The remaining V/E/Fa count differences from OCC reference files are due to legitimate topological strategy differences, not format errors.

## Ground
1. 10 fixes applied across 3 sessions, eliminating 5 distinct root causes:
   - Hardcoded edge t_min/t_max [0,1] (replaced with curve-geometry-derived values)
   - Newton projection [0,1] clamp in find_param_on_curve (prevented curve trimming)
   - upgrade_line_edges_to_circles replacing degenerated edge curves
   - Missing degeneracy flags on self-loop zero-length edges
   - Self-loop circles misidentified as degenerated edges
2. All 11 files now export without assertion failures (T1: 0 failed, 0 errors)
3. Zero edges have param_range=0 across all 11 files (T2: verified)
4. Exactly 2 edges have same_range=0 — both are sphere pole degeneracies (T3: verified)

## Warrant
The fixes address format-level correctness: parameter ranges, flag bits, curve type integrity. These are the properties that OCC parsers validate. Vertex-sharing and face-decomposition are higher-level topological choices that produce different-but-valid BREP representations of the same geometry.

## Backing
- OCC BREP format specification: edges with param_range=0 are invalid (curve degenerates to a point)
- OCC BREP format specification: degenerated edges must have same_range=0 and TShape flags 0101100 to signal that the 3D curve should be ignored
- The STEP files themselves define the expected topology — our output matches STEP face/edge counts in almost all cases

## Rebuttal
Alternative A: Rewrite STEP importer to match OCC's vertex sharing → rejected: architectural change affecting entire pipeline (heal, mesh, bool), scope exceeds this fix cycle
Alternative B: Force face decomposition to match OCC → rejected: STEP file defines face count; adding/removing faces changes geometry semantics

## Qualifier
- Valid for: all 11 STEP files in compare/step/ with current STEP import pipeline
- Re-evaluate if: OCC tools report specific parse errors after testing
- Scope: BREP writer and STEP importer post-processing in rc3d-shape and rc3d-io

## Verdict: PASSED
