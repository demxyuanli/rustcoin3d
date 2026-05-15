## Context

The NURBS surface rendering pipeline recently received adaptive screen-space tessellation and a frustum near-plane fix for wgpu forward-Z. Three gaps remain:

1. **Reverse-Z depth**: The `frustum.rs` near-plane extraction assumes forward-Z (`z_clip ∈ [0,w]`). Camera nodes with `reverse_depth: true` map near→NDC 1 and far→NDC 0, requiring a different plane (`r2 - r3` instead of `r2`).

2. **Viewport-culled quads**: `subdivide_quad_screen` processes every initial quad through the full subdivision heuristic, even when the quad's projected bounding rect is fully outside the viewport. This wastes CPU time and vertex budget that could go to visible quads.

3. **Multi-patch surfaces**: Current tessellation handles a single `NurbsSurface`. Real CAD models consist of multiple patches joined along boundary curves with position (G⁰) and possibly normal (G¹) continuity.

## Goals / Non-Goals

**Goals:**
- Frustum plane extraction works correctly for both forward-Z and reverse-Z projection matrices
- Quads whose 4 corners all project outside the viewport (same side of any frustum edge) are skipped, recovering their vertex budget
- Two or more `NurbsSurface` patches can be stitched along a shared boundary with G⁰ continuity, producing a single unified `TessellatedSurface`
- G¹ (tangent) continuity is supported as a quality option for smoother seams

**Non-Goals:**
- G² (curvature) continuity — requires second-derivative matching, out of scope
- Trimmed NURBS surfaces
- Dynamic re-stitching during camera movement (stitching happens once at creation time)

## Decisions

### Decision 1: Conditional near-plane extraction

Use `depth_reversed_z` parameter in `Frustum::from_view_projection`:
- Forward-Z: near = `r2` (z_clip=0), far = `r3 - r2` (z_clip=w)
- Reverse-Z: near = `r2 - r3` (z_clip=w), far = `r2` (z_clip=0)

The caller already has `depth_reversed_z` from the draw call or projection matrix. No need to auto-detect from the matrix.

### Decision 2: Viewport culling via AABB bounds check

After projecting the 4 corners to screen space, compute `(x_min, y_min)` and `(x_max, y_max)`. If the bounding rect is fully outside the viewport on any side — `x_max < 0`, `x_min > viewport_w`, `y_max < 0`, or `y_min > viewport_h` — the quad is emitted immediately without subdivision and without consuming budget.

Rationale: Adding a single bounding-rect check is O(1) overhead in an already-existing projection step. The savings (skipping 5 NURBS evaluations + 4 recursive calls per skipped quad) outweigh the cost by ~100x for off-screen quads.

### Decision 3: Boundary-first stitching

Stitching two patches requires:
1. Ensure boundary curves match in parameterization (same knot vector, same control points along the shared edge)
2. Create a single `TessellatedSurface` containing both patches' vertices, with shared boundary vertices deduplicated
3. For G¹: ensure normals along the boundary are averaged between adjacent patches

The stitching API accepts a list of patches and adjacency information (which edge of patch A connects to which edge of patch B).

## Risks / Trade-offs

- **Viewport culling accuracy**: The projected bounding rect of a quad can be conservative — a quad near the viewport edge might be retained even when partially outside. This is acceptable; false negatives would cause visible gaps.
- **Multi-patch memory**: Stitching N patches produces a single `TessellatedSurface` with combined vertex/index buffers. Very large assemblies (100+ patches) may exceed GPU buffer limits. Mitigation: document a practical limit of ~50 patches.
- **G¹ stitching complexity**: Normal averaging at boundaries can produce flat-looking seams where curvature changes discontinuously between patches. This is a cosmetic issue, not a correctness problem.
