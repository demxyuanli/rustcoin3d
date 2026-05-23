## Why

The engine currently relies on `truck-stepio` + `truck-meshalgo` (external crates from ricosjp/truck) for STEP import. Version lock-in creates maintenance risk — any truck API change breaks our import pipeline. Additionally, `truck-stepio` 0.3 lacks assembly entity types (NEXT_ASSEMBLY_USAGE_OCCURRENCE, PRODUCT_DEFINITION_SHAPE), preventing proper assembly hierarchy reconstruction and causing all components to pile up at origin.

## What Changes

- **BREAKING**: Remove `truck-stepio` and `truck-meshalgo` dependencies from `rc3d-io`
- **BREAKING**: Remove `ruststep` dependency from `rc3d-io`
- Add self-developed STEP exchange structure parser (`parser.rs`) — tokenizer + recursive-descent parser producing `HashMap<u64, EntityRecord>`
- Add B-rep topology traversal (`topology.rs`) — walk Shell→FaceSurface→EdgeLoop→EdgeCurve entity graph
- Add curve/surface geometry evaluation (`geom.rs`) — LINE, CIRCLE, ELLIPSE, B_SPLINE_CURVE sampling; PLANE, CYLINDER, CONE, SPHERE, TORUS, B_SPLINE_SURFACE evaluation using `rc3d-nurbs`
- Add face tessellation (`tessellate.rs`) — planar face polygon extraction, curved surface UV sampling + fan triangulation
- Add assembly hierarchy reconstruction (`assembly.rs`) — extract NEXT_ASSEMBLY_USAGE_OCCURRENCE→Transform chain, build multi-level assembly tree with per-component Transforms
- Build SceneGraph with proper Separator/Transform hierarchy per component

## Capabilities

### New Capabilities
- `step-exchange-parser`: Parse ISO 10303-21 exchange structure text into indexed entity records without external STEP libraries
- `step-brep-topology`: Walk STEP B-rep topology graph (Shell, FaceSurface, EdgeLoop, EdgeCurve, OrientedEdge)
- `step-curve-geometry`: Evaluate and sample curve types (LINE, CIRCLE, ELLIPSE, B_SPLINE_CURVE_WITH_KNOTS)
- `step-surface-geometry`: Evaluate surface types (PLANE, CYLINDER, CONE, SPHERE, TORUS, B_SPLINE_SURFACE) using rc3d-nurbs
- `step-face-tessellation`: Convert trimmed faces to triangle meshes (polygon extraction for planar, UV sampling for curved)
- `step-assembly-hierarchy`: Reconstruct assembly tree from PRODUCT, PRODUCT_DEFINITION_SHAPE, NEXT_ASSEMBLY_USAGE_OCCURRENCE entities with AXIS2_PLACEMENT_3D transforms

### Modified Capabilities
<!-- None — this is a new capability replacing the previous truck-based approach -->

## Impact

- `crates/rc3d-io/Cargo.toml`: Remove `truck-stepio`, `truck-meshalgo`, `ruststep` deps
- `crates/rc3d-io/src/step.rs`: Replaced by `step/` module directory
- `crates/rc3d-io/src/lib.rs`: Update re-exports (no signature changes)
- `crates/rc3d-io/src/step/` (new directory): 7 files, ~1200 lines total
- Zero new external dependencies; uses existing `rc3d-nurbs`, `rc3d-core`, `rc3d-scene`
