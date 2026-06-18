# BRepCheck Complete — 33 OCC status codes

Date: 2026-06-18. Closes the last OCC partial (BRepCheck → ✅).

## Architecture: heal/check/ directory

```
heal/check/
  mod.rs        — BRepCheck_Analyzer: orchestrates all sub-checks
  vertex.rs     — BRepCheck_Vertex: point-on-curve, point-on-surface
  edge.rs       — BRepCheck_Edge: curve presence, SameParameter, tolerance
  wire.rs       — BRepCheck_Wire: closure, self-intersection, ordering
  face.rs       — BRepCheck_Face: wire orientation, intersecting wires
  shell.rs      — BRepCheck_Shell: closure, orientation, multi-connectivity
  solid.rs      — BRepCheck_Solid: shell orientation, region check
```

## Key checks to add (vs current check.rs)

| Check | OCC Status | rustcoin3d |
|-------|-----------|------------|
| Vertex not on curve | InvalidPointOnCurve | NEW |
| Vertex not on surface | InvalidPointOnSurface | NEW |
| Edge no 3D curve | No3DCurve | NEW |
| Edge InvalidSameParameter | InvalidSameParameterFlag | NEW |
| Edge InvalidSameRange | InvalidSameRangeFlag | NEW |
| Face intersecting wires | IntersectingWires | EXISTS |
| Shell not closed | NotClosed | EXISTS |
| Shell unorientable | UnorientableShape | NEW |
| Solid bad orientation | BadOrientation | NEW |
| Wire redundant edge | RedundantEdge | NEW |

## Implementation strategy

1. Create heal/check/ directory
2. Move existing check.rs content into shell.rs + face.rs
3. Add new checker modules
4. Wire analyzer in heal pipeline
5. 33 status codes tracked

## Tests

- Vertex-on-curve: vertex at (0.5,0,0) on line from (0,0,0) to (1,0,0) → OK
- Vertex-off-curve: vertex at (5,0,0) on same line → InvalidPointOnCurve
- Edge no 3D curve: edge with empty curve → No3DCurve
