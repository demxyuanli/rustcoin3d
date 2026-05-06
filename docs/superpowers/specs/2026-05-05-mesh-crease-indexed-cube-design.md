# Design: mesh crease (import) + indexed cube

## Goals

1. **Imported / `IndexedFaceSet` meshes**: preserve author normals when provided; when normals are missing or length-mismatched, recompute vertex normals with optional **crease** (hard-edge) behavior instead of over-smoothing everything.
2. **Procedural cube**: switch `tessellate_cube` to an **indexed** mesh (24 vertices, 36 indices) with the same face winding and per-face UV as today.

## Policy C (normals source of truth)

- If `NormalNode` vectors length **equals** `Coordinate3` point count and is non-empty: **use file normals** (current behavior). No automatic crease pass is applied by default.
- Else: build mesh from indices, then compute normals using **area-weighted** averaging **subject to crease rules** below.

This matches glTF expectations: exporters that duplicate vertices at creases keep hard edges; assets without normals get a predictable geometric fallback.

## Crease semantics (recompute path only)

- **Input**: `TriangleMesh` with `positions` and `tri_indices` (topology built).
- **Parameter**: `crease_angle_degrees: f32` (e.g. 30–60). Edge between two triangles is **sharp** if the angle between face normals exceeds this threshold.
- **Behavior**:
  - Along a **smooth** edge, vertices remain shared; vertex normal is the existing area-weighted average of adjacent face normals.
  - Along a **sharp** edge, **split** the vertex into one copy per adjacent smooth “fan” or per face as needed so each corner gets a normal consistent with flat shading on that side of the edge (standard split: duplicate position with distinct normals per incident face group, or per-face for simplicity where topology allows).
- **Degeneracies**: zero-area faces and colinear edges are skipped; if splitting is ambiguous, fall back to averaged normal for that corner.
- **Tunables**: `crease_angle_degrees <= 0` means treat all edges as smooth (current pure average behavior).

Implementation location: `rc3d-mesh` (`TriangleMesh` method or standalone helper used after `from_indexed_face_set*` and before `compute_tangents` in the render bake path when policy C uses the recompute branch).

**Call site** (`rc3d-render` `IndexedFaceSet` cache build):

- After constructing `TriangleMesh` from coord / index / optional tex:
  - If file normals applied: unchanged.
  - Else: call crease-aware recompute (with a default angle, optionally later wired to scene or import metadata).

Default `crease_angle_degrees` should be chosen once (e.g. 45°) and documented; future work may expose per-graph or per-node settings without changing Policy C.

## Indexed cube

- **Vertices**: 24 (6 faces × 4 corners); **indices**: 36 (12 triangles).
- Preserve each face’s four corner positions, UVs `[0,0]–[1,1]` layout, and **outward face normals** equivalent to current flat shading.
- Replace `from_triangle_list_with_texcoords` with `from_indexed_with_texcoords`.
- `ShapeKey::Cube` fields unchanged so existing mesh cache keys remain valid; vertex/triangle counts stay the same for cache invalidation expectations.

## Non-goals (this iteration)

- Parsing OBJ `s` smoothing groups or arbitrary CAD “hard edge” flags (optional later).
- Changing glTF loader to inject crease metadata when normals are present (Policy C forbids by default).

## Verification

- **Cube**: triangle count 12, vertex count 24, manifold box with correct AABB; `scene_graph` / cube example visually unchanged except possible negligible numerical differences.
- **Crease**: unit test on a **sharp box** built as `IndexedFaceSet` **without** `NormalNode`: normals at cube corners should align with face normals (six-way discontinuity), not a single blended direction.
- **glTF sample with normals**: unchanged appearance vs pre-crease work when normals match coordinate count.

## Approval

- Policy **C** confirmed by stakeholder (2026-05-05).
- Next step: implementation plan (`writing-plans`) then code changes in `rc3d-mesh` and `rc3d-render` (`IndexedFaceSet` path) plus `tessellate_cube`.
