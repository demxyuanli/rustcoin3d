## Why

The recent NURBS surface rendering work introduced a frustum near-plane fix for wgpu forward-Z but leaves reverse-Z unsupported. Additionally, screen-space tessellation wastes vertex budget on quads fully outside the viewport, and the system lacks multi-patch NURBS support needed for complex CAD models.

## What Changes

- **Frustum reverse-Z plane extraction**: Add depth-reversed path in `Frustum::from_view_projection` so camera nodes with `reverse_depth: true` use the correct near-plane formula
- **Tessellation viewport culling**: Detect quads whose projected bounding rect is fully outside the viewport and emit them immediately without subdivision, recovering vertex budget for visible regions
- **NURBS multi-patch stitching**: Support joining multiple `NurbsSurface` patches along shared boundaries with position (G⁰) and normal (G¹) continuity, producing a single unified `TessellatedSurface`

## Capabilities

### New Capabilities
- `frustum-reverse-z`: Correct frustum near-plane extraction when depth is reversed (ndc_z near→1, far→0)
- `tess-viewport-cull`: Skip viewport-external quads during adaptive screen-space subdivision
- `nurbs-multipatch`: Stitch adjacent NURBS surface patches with shared boundary curves and continuity constraints

### Modified Capabilities
<!-- No existing specs to modify -->

## Impact

- `crates/rc3d-render/src/frustum.rs` — conditional plane extraction for reverse-Z
- `crates/rc3d-nurbs/src/surface.rs` — viewport culling in `subdivide_quad_screen`, new multipatch tessellation API
- `crates/rc3d-nurbs/src/curve.rs` — shared boundary curve extraction
- `crates/rc3d-nurbs/src/stitch.rs` (NEW) — multi-patch stitching logic
- `crates/rc3d-app/examples/nurbs_viewer.rs` — demonstrate multi-patch surfaces
