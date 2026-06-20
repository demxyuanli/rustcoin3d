# Critical Reachability Review — June 2026

5-system audit of all code paths against feature reachability.
Key question: "Can a user actually use this?"

## Overall Verdict: PARTIAL

| Subsystem | Verdict | Key Issue |
|-----------|---------|-----------|
| Boolean | PARTIAL | EF missing, BuilderSolid dead, perpendicular crash, HealPassId dead |
| Heal | PARTIAL | 5/16 passes dead, HealPolicy stranded, heal_solid orphaned |
| Mesh | PARTIAL | Incremental reuse dead, ModelHealer dead, ModelPreProcessor discarded |
| I/O | PARTIAL | IGES/Binary/Parametric not exported, 3 import stubs |
| Geometry | PARTIAL | Circle/Ellipse d012 center bug, Bezier intersection dead |

## Critical bugs (affect correct output)

### 1. Circle/Ellipse d012() missing center (GEOM BUG)
**File**: `curve_eval.rs:789-806`
Circle and Ellipse `d012()` arms forget to add `center` to d0.
Off-center circles produce wrong Newton projection convergence.
**Fix**: bind `center` and add to d0.

### 2. BuilderSolid unreachable from public API (BOOL DEAD)
**File**: `builder_solid.rs:30-117`
Fully implemented (adjacency graph, BFS, volume, outer/void) but
never called from `boolean_brep()`. Pipeline stops at shells.
**Fix**: call `build_solids_from_faces()` in boolean_brep result path.

### 3. EF (Edge-Face) missing from pave_filler (BOOL GAP)
**File**: `pave_filler.rs`
6-layer interference has VV/VE/EE/VF/FF but no EF.
Edge piercing face at interior point → no intersection detected.
**Fix**: add EF loop between EE and VF.

## Dead code (implemented, tested, unreachable)

| Feature | Lines | Tests | Why dead |
|---------|-------|-------|----------|
| BuilderSolid | ~120 | 7 | Never called from API |
| BRepBoolHistory | ~50 | 0 | Always None |
| Incremental mesh reuse | ~100 | 1 | Production uses `discretize_edge` |
| ModelHealer (gap/T-junction) | ~250 | 4 | Never called from pipeline |
| Bezier clip intersection | ~400 | 3 | Zero production callers |
| GK15 constants | ~20 | 0 | Dead code |
| 5 HealPassId entries | ~2000 | ~30 | Config never enables |
| FixNotchedEdges | ~170 | 2 | Config never set |
| FixTails | ~70 | 2 | Config never set |
| UnifySameDomain | ~480 | 8 | Config never set |
| heal_solid() | ~30 | 5 | Never called from import |
| IGES import (public API) | ~430 | 5 | Not re-exported |
| IGES writer (public API) | ~680 | 8 | Not re-exported |
| Binary persistence (public API) | ~1300 | 6 | Not re-exported |
| Parametric writer (public API) | ~1200 | 5 | Not re-exported |

## Feature reachability summary

| Feature | User can use? | How to fix |
|---------|:---:|------|
| STEP import | ✅ | — |
| STEP parametric write | ❌ | Add to `lib.rs` re-exports |
| IGES import | ❌ | Add to `lib.rs` + `import_file()` |
| IGES write | ❌ | Add to `lib.rs` re-exports |
| VRML import | ✅ | — |
| Binary persistence | ❌ | Add to `lib.rs` re-exports |
| Boolean Union/Intersection/Difference | ✅ | — |
| BuilderSolid output | ❌ | Wire into `boolean_brep()` |
| Heal full pipeline | ⚠️ | Enable dead passes in `select_fixes()` |
| Incremental mesh | ❌ | Call `discretize_edges_incremental` in init |
| ModelHealer | ❌ | Call from pipeline |
| BRepCheck diagnostics | ⚠️ | Expose statuses in import report |
