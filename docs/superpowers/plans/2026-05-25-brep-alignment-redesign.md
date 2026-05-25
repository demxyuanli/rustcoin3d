# B-Rep Alignment Redesign — Implementation Plan

> **Goal:** Replace the mesh-extraction pipeline with a parametric B-Rep kernel,
> bringing rc3d-io's STEP processing substantially closer to OCCT's approach.

**Architecture:** Introduce a `BRepShape` hierarchy (Vertex/Edge/Wire/Face/Shell/Solid)
with retained curve/surface geometry and per-face PCURVEs. **Aggressively replace** the
old mesh-extraction pipeline — old modules are deleted, not deprecated. The new B-Rep
kernel becomes the ONLY code path for STEP import.

**Scope:** ~15 new files, ~10 files removed, ~5 files rewritten, ~3 files modified.
Estimated: 3500-4500 lines new code, ~2500 lines removed.

---

## Architecture Overview

### Before (current):
```
STEP entities → flat StepFace(bounds=Vec<EdgePt>, surface_id) 
  → uniform grid sampling + earcut → IndexedFaceSet
```

### After (target):
```
STEP entities → BRepBuilder → BRepShape(Face(surface+wire+PCURVE))
  → BRepHealer → fixed topology
  → BRepMesher → EdgeDiscretizer(3D+2D adaptive)
              → FaceTriangulator(constrained Delaunay)
              → MeshRefiner(deflection-driven)
              → QualityOptimizer(edge swaps)
  → IndexedFaceSet
```

---

## Phase 1: Parametric B-Rep Core

### 1.1 B-Rep Geometry Primitives

**Create `crates/rc3d-io/src/step/brep/geom.rs`**

Unified geometry enum replacing scattered curve/surface evaluation:

```rust
/// Retained curve geometry (not just a STEP entity reference).
#[derive(Debug, Clone)]
pub enum CurveGeom {
    Line { origin: Vec3, direction: Vec3 },
    Circle { center: Vec3, axis: Vec3, radius: f32, placement: Mat4 },
    Ellipse { center: Vec3, axis: Vec3, semi_major: f32, semi_minor: f32, placement: Mat4 },
    BSpline { degree: usize, control_points: Vec<Vec3>, knots: Vec<f32>, weights: Option<Vec<f32>> },
    Trimmed { basis: Box<CurveGeom>, t_min: f32, t_max: f32 },
    Composite { segments: Vec<(CurveGeom, bool)> }, // (curve, reversed)
    Polyline { points: Vec<Vec3> },
}

impl CurveGeom {
    /// Position at parameter t in [0,1]
    pub fn d0(&self, t: f32) -> Vec3;
    /// First derivative dC/dt
    pub fn d1(&self, t: f32) -> Vec3;
    /// Second derivative d²C/dt²
    pub fn d2(&self, t: f32) -> Vec3;
    /// Approximate arc length
    pub fn arc_length(&self, t0: f32, t1: f32) -> f32;
    /// Curvature at parameter t (|C' × C''| / |C'|³)
    pub fn curvature(&self, t: f32) -> f32;
    /// Adaptive sample points respecting chordal tolerance
    pub fn sample_adaptive(&self, t0: f32, t1: f32, tolerance: f32) -> Vec<(f32, Vec3)>;
}

/// Retained surface geometry.
#[derive(Debug, Clone)]
pub enum SurfaceGeom {
    Plane { origin: Vec3, normal: Vec3, u_dir: Vec3 },
    Cylinder { origin: Vec3, axis: Vec3, radius: f32, placement: Mat4 },
    Cone { apex: Vec3, axis: Vec3, semi_angle: f32, radius_at_apex: f32, placement: Mat4 },
    Sphere { center: Vec3, radius: f32 },
    Torus { center: Vec3, axis: Vec3, major_r: f32, minor_r: f32 },
    BSpline(NurbsSurface),  // reuse existing nurbs.rs
    Extrusion { generatrix: Box<CurveGeom>, direction: Vec3 },
    Revolution { generatrix: Box<CurveGeom>, axis_origin: Vec3, axis_dir: Vec3 },
    Offset { basis: Box<SurfaceGeom>, distance: f32 },
}

impl SurfaceGeom {
    /// Position at (u,v)
    pub fn d0(&self, u: f32, v: f32) -> Vec3;
    /// First partial derivatives (∂S/∂u, ∂S/∂v)
    pub fn d1(&self, u: f32, v: f32) -> (Vec3, Vec3);
    /// Normal at (u,v)
    pub fn normal(&self, u: f32, v: f32) -> Vec3;
    /// Project 3D point to closest (u,v) on surface
    pub fn project(&self, point: Vec3) -> Option<(f32, f32)>;
    /// Evaluate a grid of points
    pub fn evaluate_grid(&self, u_range: (f32,f32), v_range: (f32,f32), n_u: usize, n_v: usize) -> Vec<Vec<Vec3>>;
    /// Estimate UV bounds from 3D bounding box
    pub fn estimate_uv_bounds(&self, points_3d: &[Vec3]) -> (f32, f32, f32, f32);
}
```

### 1.2 B-Rep Topology Primitives

**Create `crates/rc3d-io/src/step/brep/topo.rs`**

Proper OCCT-aligned topology with shared ownership:

```rust
/// Globally unique IDs for topology elements.
slotmap::new_key_type! { pub struct VertexKey; }
slotmap::new_key_type! { pub struct EdgeKey; }
slotmap::new_key_type! { pub struct WireKey; }
slotmap::new_key_type! { pub struct FaceKey; }
slotmap::new_key_type! { pub struct ShellKey; }
slotmap::new_key_type! { pub struct SolidKey; }

/// Orientation relative to geometric definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation { Forward, Reversed, Internal, External }

/// A topological vertex — unique 3D position + tolerance.
#[derive(Debug, Clone)]
pub struct BRepVertex {
    pub position: Vec3,
    pub tolerance: f32,
}

/// A topological edge — owns its 3D curve and per-face PCURVEs.
#[derive(Debug, Clone)]
pub struct BRepEdge {
    pub curve: CurveGeom,
    pub tolerance: f32,
    /// PCURVEs keyed by the face they belong to.
    /// OCCT: BRep_Tool::CurveOnSurface(edge, face)
    pub pcurves: HashMap<FaceKey, CurveGeom>,  // 2D curves in UV space
}

/// A wire = ordered loop of oriented edges.
#[derive(Debug, Clone)]
pub struct BRepWire {
    pub edges: Vec<(EdgeKey, Orientation)>,
}

/// A face = surface + outer wire + hole wires.
#[derive(Debug, Clone)]
pub struct BRepFace {
    pub surface: SurfaceGeom,
    pub outer_wire: WireKey,
    pub inner_wires: Vec<WireKey>,  // holes
    pub same_sense: bool,
    pub tolerance: f32,
}

/// A shell = collection of faces.
#[derive(Debug, Clone)]
pub struct BRepShell {
    pub faces: Vec<(FaceKey, Orientation)>,
    pub closed: bool,
}

/// A solid = outer shell + void shells.
#[derive(Debug, Clone)]
pub struct BRepSolid {
    pub outer_shell: ShellKey,
    pub void_shells: Vec<ShellKey>,
}
```

### 1.3 B-Rep Shape Registry

**Create `crates/rc3d-io/src/step/brep/registry.rs`**

Central SlotMap-based registry for shared ownership:

```rust
pub struct BRepRegistry {
    pub vertices: SlotMap<VertexKey, BRepVertex>,
    pub edges: SlotMap<EdgeKey, BRepEdge>,
    pub wires: SlotMap<WireKey, BRepWire>,
    pub faces: SlotMap<FaceKey, BRepFace>,
    pub shells: SlotMap<ShellKey, BRepShell>,
    pub solids: SlotMap<SolidKey, BRepSolid>,
    /// Reverse index: 3D hash → VertexKey for dedup
    pub vertex_hash_index: HashMap<[u32; 3], VertexKey>,
}

impl BRepRegistry {
    /// Insert or find existing vertex at position (within tolerance).
    pub fn find_or_add_vertex(&mut self, position: Vec3, tolerance: f32) -> VertexKey;
    
    /// Insert edge with PCURVE for a specific face.
    pub fn add_edge_with_pcurve(&mut self, curve: CurveGeom, tolerance: f32,
                                 face: FaceKey, pcurve: CurveGeom) -> EdgeKey;
    
    /// Find edges shared by two faces (for sewing).
    pub fn find_shared_edges(&self, face_a: FaceKey, face_b: FaceKey) -> Vec<EdgeKey>;
    
    /// Iterate over all faces, resolving geometry on the fly.
    pub fn iter_faces(&self) -> impl Iterator<Item = (FaceKey, &BRepFace)>;
}
```

### 1.4 STEP → B-Rep Builder

**Rewrite `crates/rc3d-io/src/step/brep/build.rs`**

The crucial bridge: STEP entities → parametric B-Rep shapes.

```rust
/// Build a full B-Rep from STEP entities.
pub fn build_brep(entities: &EntityIndex) -> Result<BRepBuildResult, StepError> {
    let mut reg = BRepRegistry::default();
    
    // Pass 1: Resolve all surfaces and vertices
    // Pass 2: Build edges (with PCURVEs resolved from SURFACE_CURVE chains)
    // Pass 3: Build wires from edge loops
    // Pass 4: Build faces from surfaces + wires  
    // Pass 5: Build shells from face collections
    // Pass 6: Build solids from shells + voids
    
    // Key difference from current approach:
    // - PCURVEs are attached to edges at BUILD time, not extracted on-the-fly
    // - Surface geometry is evaluated and stored, not just referenced by ID
    // - Orientation is propagated through the hierarchy
    // - SameSense flag is applied during face construction
    
    Ok(BRepBuildResult { registry: reg, root_solids: vec![...] })
}
```

Key construction details:
```rust
// For each ADVANCED_FACE:
//   1. Resolve surface → build SurfaceGeom
//   2. For each FACE_OUTER_BOUND → resolve EDGE_LOOP
//   3. For each ORIENTED_EDGE in loop:
//      a. Build 3D curve from EDGE_CURVE → CurveGeom
//      b. Resolve PCURVE from SURFACE_CURVE chain → 2D CurveGeom
//      c. Register edge with PCURVE for this face
//      d. Track orientation
//   4. Build BRepWire from ordered edges
//   5. Build BRepFace from surface + wire
//   6. Attach face to shell
```

---

## Phase 2: Adaptive Meshing Pipeline

### 2.1 Edge Discretizer

**Create `crates/rc3d-io/src/step/brep/mesh/edge_disc.rs`**

```rust
/// Discretize an edge into a polygon respecting chordal tolerance.
/// Produces BOTH 3D and 2D (via PCURVE) polygons simultaneously.
pub struct EdgeDiscretizer {
    pub deflection: f32,       // max chordal deviation
    pub angle_deflection: f32, // max angular deviation (radians)
    pub min_points: usize,
    pub max_points: usize,
}

impl EdgeDiscretizer {
    /// Discretize a single edge, producing 3D polygon + face-specific 2D polygon.
    pub fn discretize(&self, edge: &BRepEdge, face: FaceKey, 
                      reg: &BRepRegistry) -> EdgePolygon {
        let curve = &edge.curve;
        let pcurve = edge.pcurves.get(&face);
        
        // Adaptive sampling based on curvature:
        // 1. Start with min_points uniformly
        // 2. For each segment, check chordal deviation
        // 3. If deviation > threshold, insert midpoint (binary subdivision)
        // 4. Repeat until max_points or tolerance met
        
        // OCCT: BRepMesh_FastDiscret equivalent
        let mut params_3d = Vec::new();
        let mut params_2d = Vec::new();
        
        // Initial uniform sampling
        // ... curvature-driven refinement ...
        
        EdgePolygon { params_3d, params_2d }
    }
}

pub struct EdgePolygon {
    pub params_3d: Vec<(f32, Vec3)>,  // (t, 3D point) along curve
    pub params_2d: Option<Vec<(f32, (f32, f32))>>, // (t, UV point) along PCURVE
}
```

### 2.2 Constrained Delaunay Triangulator

**Create `crates/rc3d-io/src/step/brep/mesh/face_tri.rs`**

Replace earcut with proper Constrained Delaunay Triangulation (CDT):

```rust
/// Triangulate a face's UV domain with edge polygons as constraints.
pub struct FaceTriangulator {
    pub min_angle: f32, // minimum triangle angle (quality threshold)
}

impl FaceTriangulator {
    pub fn triangulate(&self, face: &BRepFace, edge_polygons: &[EdgePolygon],
                       reg: &BRepRegistry) -> Option<FaceMesh> {
        // 1. Collect all edge polygon vertices as 2D constraints in UV space
        // 2. Build CDT using spade or delaunay2d crate
        //    - Insert all polygon vertices as sites
        //    - Insert edge segments as constraint edges
        // 3. Remove triangles outside the face (point-in-polygon test on UV)
        // 4. Map UV → 3D via surface evaluation
        // 5. Generate normals from surface derivatives
        
        // OCCT: BRepMesh_DelaunayDeflectionControl equivalent
        todo!("CDT-based face triangulation")
    }
}
```

Dependency: add `spade = "2"` or `delaunay2d = "0.1"` to Cargo.toml.

### 2.3 Deflection-Driven Refiner

**Rewrite `crates/rc3d-io/src/step/brep/mesh/refiner.rs`**

Replace uniform refinement with proper deflection-driven refinement:

```rust
/// Refine mesh by inserting nodes where deflection exceeds threshold.
/// OCCT: Wallace-Boissonnat incremental node insertion.
pub struct MeshRefiner {
    pub max_deflection: f32,
    pub max_iterations: usize,
}

impl MeshRefiner {
    pub fn refine(&self, mesh: &mut FaceMesh, surface: &SurfaceGeom) {
        // For each triangle:
        // 1. For each edge midpoint (u_mid, v_mid):
        //    a. Evaluate surface at UV-midpoint → S3
        //    b. Compare with linear midpoint (v0+v1)/2 in 3D
        //    c. If |S3 - linear_midpoint| > max_deflection → mark edge
        // 2. Insert nodes at marked edge midpoints
        // 3. Retriangulate affected triangles
        // 4. Repeat
    }
}
```

Key difference from current refine.rs: uses **actual UV coordinates** from the triangle vertices (obtained from the surface projection during CDT construction), not domain-center guesses.

### 2.4 Quality Optimizer

**Create `crates/rc3d-io/src/step/brep/mesh/optimize.rs`**

```rust
/// Improve triangle quality via edge swaps.
pub struct MeshOptimizer {
    pub min_angle_degrees: f32,
    pub max_iterations: usize,
}

impl MeshOptimizer {
    pub fn optimize(&self, mesh: &mut FaceMesh) {
        // Delaunay edge flip:
        // For each interior edge (shared by two triangles):
        // 1. Compute the quadrilateral formed by the two triangles
        // 2. Check if the edge should be flipped (Delaunay criterion)
        // 3. If flip improves min angle, perform it
        // 4. Iterate until no more flips or max_iterations
    }
}
```

### 2.5 Unified Meshing API

**Create `crates/rc3d-io/src/step/brep/mesh/mod.rs`**

Combine all phases:

```rust
pub struct BRepMeshConfig {
    pub edge_deflection: f32,
    pub edge_angle_deflection: f32,
    pub face_deflection: f32,
    pub min_angle_degrees: f32,
    pub refine_iterations: usize,
    pub optimize_iterations: usize,
}

impl Default for BRepMeshConfig {
    fn default() -> Self {
        Self {
            edge_deflection: 0.1,
            edge_angle_deflection: 0.1,
            face_deflection: 0.05,
            min_angle_degrees: 15.0,
            refine_iterations: 5,
            optimize_iterations: 3,
        }
    }
}

pub fn mesh_brep_shell(shell: &BRepShell, reg: &BRepRegistry, 
                       config: &BRepMeshConfig) -> MeshResult {
    let edge_disc = EdgeDiscretizer { ... };
    let face_tri = FaceTriangulator { ... };
    let refiner = MeshRefiner { ... };
    let optimizer = MeshOptimizer { ... };
    
    // 1. Discretize all edges (shared edges only once)
    let edge_polygons = edge_disc.discretize_all(shell, reg);
    
    // 2. Triangulate each face with CDT
    let mut meshes = Vec::new();
    for (face_key, _) in shell.faces.iter() {
        let face = &reg.faces[*face_key];
        if let Some(mut mesh) = face_tri.triangulate(face, &edge_polygons, reg) {
            // 3. Refine
            refiner.refine(&mut mesh, &face.surface);
            // 4. Optimize
            optimizer.optimize(&mut mesh);
            meshes.push(mesh);
        }
    }
    
    // 5. Merge face meshes into single output
    merge_face_meshes(&meshes)
}
```

---

## Phase 3: Shape Healing

### 3.1 Wire Reordering

**Create `crates/rc3d-io/src/step/brep/heal/reorder.rs`**

Fix edge ordering so that consecutive edges share vertices:

```rust
/// Reorder edges in a wire so they form a connected chain.
/// OCCT: ShapeFix_Wire::FixReorder
pub fn reorder_wire_edges(edges: &[(EdgeKey, Orientation)], 
                           reg: &BRepRegistry) -> Option<Vec<(EdgeKey, Orientation)>> {
    if edges.len() < 2 { return Some(edges.to_vec()); }
    
    // Build adjacency: which edges share a vertex?
    let mut by_start: HashMap<VertexKey, Vec<usize>> = HashMap::new();
    let mut by_end: HashMap<VertexKey, Vec<usize>> = HashMap::new();
    for (i, (eid, orient)) in edges.iter().enumerate() {
        let edge = &reg.edges[*eid];
        let (v_start, v_end) = edge_vertices(edge, reg);
        by_start.entry(v_start).or_default().push(i);
        by_end.entry(v_end).or_default().push(i);
    }
    
    // Greedy chain building: start from first edge, follow adjacency
    // ... (standard graph traversal)
    todo!("wire reordering")
}
```

### 3.2 Gap Closing

**Create `crates/rc3d-io/src/step/brep/heal/gap.rs`**

Merge vertices within tolerance:

```rust
/// Close small gaps between wire edges by merging nearby vertices.
/// OCCT: ShapeFix_Wire::FixGaps
pub fn close_wire_gaps(wire: &mut BRepWire, reg: &mut BRepRegistry, 
                       tolerance: f32) -> usize {
    let mut closed = 0;
    for i in 0..wire.edges.len() {
        let j = (i + 1) % wire.edges.len();
        let (eid_a, _) = wire.edges[i];
        let (eid_b, _) = wire.edges[j];
        
        let end_a = edge_end_vertex(eid_a, reg);
        let start_b = edge_start_vertex(eid_b, reg);
        
        let gap = (reg.vertices[end_a].position - reg.vertices[start_b].position).length();
        if gap > 0.0 && gap < tolerance {
            // Merge start_b into end_a
            reg.merge_vertices(end_a, start_b);
            closed += 1;
        }
    }
    closed
}
```

### 3.3 Orientation Fixer

**Create `crates/rc3d-io/src/step/brep/heal/orient.rs`**

Ensure consistent face orientations within a shell:

```rust
/// Flip face orientations so all normals point consistently 
/// (all outward or all inward) for a closed shell.
/// OCCT: ShapeFix_Shell::FixOrientation
pub fn fix_shell_orientation(shell: &mut BRepShell, reg: &BRepRegistry) -> usize {
    // 1. Pick a seed face — assume its orientation is correct
    // 2. Propagate: for each adjacent face (shares an edge):
    //    - Compare normals at the shared edge
    //    - If normals are opposite, flip the adjacent face
    // 3. BFS propagation through the face adjacency graph
    let mut flipped = 0;
    let mut visited: HashSet<FaceKey> = HashSet::new();
    let mut queue: VecDeque<FaceKey> = VecDeque::new();
    // ... (BFS propagation)
    flipped
}
```

### 3.4 Unified Healing API

**Create `crates/rc3d-io/src/step/brep/heal/mod.rs`**

```rust
pub struct HealConfig {
    pub gap_tolerance: f32,
    pub fix_orientation: bool,
    pub fix_reorder: bool,
    pub remove_small_edges: bool,
    pub min_edge_length: f32,
}

impl Default for HealConfig {
    fn default() -> Self {
        Self {
            gap_tolerance: 1e-4,
            fix_orientation: true,
            fix_reorder: true,
            remove_small_edges: false,
            min_edge_length: 1e-6,
        }
    }
}

pub fn heal_shell(shell: &mut BRepShell, reg: &mut BRepRegistry, 
                  config: &HealConfig) -> HealReport {
    let mut report = HealReport::default();
    
    // Run healing passes
    for (face_key, _) in &shell.faces {
        let face = &reg.faces[*face_key];
        let wire = &reg.wires[face.outer_wire];
        
        if config.fix_reorder {
            if let Some(reordered) = reorder_wire_edges(&wire.edges, reg) {
                // update wire
                report.reordered_wires += 1;
            }
        }
        
        if config.gap_tolerance > 0.0 {
            let mut wire = reg.wires[face.outer_wire].clone();
            let closed = close_wire_gaps(&mut wire, reg, config.gap_tolerance);
            if closed > 0 {
                reg.wires[face.outer_wire] = wire;
                report.closed_gaps += closed;
            }
        }
    }
    
    if config.fix_orientation {
        report.flipped_faces = fix_shell_orientation(shell, reg);
    }
    
    report
}
```

---

## Phase 4: Improved Boolean Foundation

### 4.1 Face-Face Intersection with PCURVE Output

**Rewrite `crates/rc3d-io/src/step/bool/intersect.rs`**

Produce intersection curves with UV on both faces:

```rust
/// Result of face-face intersection — the actual OCCT IntTools_FaceFace output shape.
pub struct FaceIntersectionResult {
    /// Intersection curves in 3D
    pub curves_3d: Vec<CurveGeom>,
    /// PCURVEs of intersection curves on face A
    pub pcurves_on_a: Vec<CurveGeom>,  // 2D curves in face A's UV space
    /// PCURVEs of intersection curves on face B  
    pub pcurves_on_b: Vec<CurveGeom>,  // 2D curves in face B's UV space
}

pub fn intersect_faces(face_a: &BRepFace, face_b: &BRepFace,
                        reg: &BRepRegistry) -> Option<FaceIntersectionResult> {
    // For analytic-analytic pairs: use closed-form solutions (existing code)
    // For general pairs: use marching method with Newton refinement
    // Each intersection curve gets:
    //   - 3D curve geometry
    //   - 2D curve on face A's UV (via projection or closed-form)
    //   - 2D curve on face B's UV
    todo!("face-face intersection with PCURVE output")
}
```

### 4.2 Edge Splitting with PCURVE Insertion

**Rewrite `crates/rc3d-io/src/step/bool/split.rs`**

Split edges at intersection points and add PCURVEs:

```rust
/// Split a face's edges along intersection curves.
/// Returns new faces (not just polygon lists).
pub fn split_face_by_curves(
    face_key: FaceKey,
    intersection_curves: &[CurveGeom],  // 2D curves in this face's UV
    reg: &mut BRepRegistry,
) -> Vec<FaceKey> {
    // 1. For each edge in the face's wire, check if intersection curve crosses it
    // 2. If crossing: split the edge at the intersection point
    //    - Create new vertex at intersection
    //    - Create two new edges from the split
    //    - Add PCURVEs to both new edges
    // 3. Build new wires from the split edges
    // 4. Build new faces from the new wires
    todo!("topology splitting with PCURVE insertion")
}
```

---

## Phase 5: Integration & Removal (hard cutover — no fallback)

### 5.1 Remove Replaced Modules

**Delete these files entirely:**

| File | Replaced by | Reason |
|------|------------|--------|
| `topo/vertex.rs` | `brep/topo.rs` (BRepVertex) | Flat registry → parametric topology |
| `topo/edge.rs` | `brep/topo.rs` (BRepEdge with PCURVEs) | Edges now carry geometry + PCURVEs |
| `topo/shape.rs` | `brep/topo.rs` (BRepFace/Wire/Shell) | Faces now carry surface + wire refs |
| `topo/build.rs` | `brep/build.rs` | Full STEP→B-Rep construction |
| `topo/mod.rs` | `brep/mod.rs` | Module-level replacement |
| `tessellate.rs` | `brep/mesh/` pipeline | CDT replaces earcut+grid |
| `surface_tess.rs` | `brep/mesh/face_tri.rs` | CDT replaces grid sampling |
| `refine.rs` | `brep/mesh/refiner.rs` | Per-edge deviation replaces domain-center probe |
| `curve/derivative.rs` | `brep/geom.rs` (CurveGeom::d1, d2, curvature) | Geometry carries its own derivatives |
| `curve/arc_length.rs` | `brep/geom.rs` (CurveGeom::arc_length) | Geometry carries its own arc length |
| `curve/mod.rs` | `brep/geom.rs` | Module-level replacement |

**Remove `pub mod` declarations from `step/mod.rs`** for: `topo`, `tessellate`, `surface_tess`, `refine`, `curve`.

### 5.2 Rewrite Main Pipeline (single path, no fallback)

**Rewrite `crates/rc3d-io/src/step/mod.rs:parse_step`** — the `parse_step_with_options` function is removed. There is ONE path:

```rust
pub fn parse_step(input: &str) -> Result<SceneGraph, StepError> {
    let exchange = parser::parse_exchange(input)
        .map_err(|e| StepError::Parse(e))?;

    let report = validate::validate(&exchange.entities);
    // ... logging (unchanged) ...

    // Build parametric B-Rep (THE ONLY PATH)
    let brep_result = brep::build::build_brep(&exchange.entities)
        .map_err(|e| StepError::Validation(e))?;
    let mut reg = brep_result.registry;

    // Heal
    let heal_config = brep::heal::HealConfig::default();
    let mut heal_report = brep::heal::HealReport::default();
    for solid_key in &brep_result.root_solids {
        let solid = reg.solids[*solid_key].clone();
        let shell = reg.shells.get_mut(solid.outer_shell).unwrap();
        heal_report.merge(brep::heal::heal_shell(shell, &mut reg, &heal_config));
    }
    log::info!("[STEP] healed: {:?}", heal_report);

    // Extract assembly info
    let transforms = assembly::extract_shell_transforms(&exchange.entities);
    let styles = assembly::extract_shell_styles(&exchange.entities);

    // Build scene graph from B-Rep meshes
    let mesh_config = brep::mesh::BRepMeshConfig::default();
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    // ... default material ...

    for solid_key in &brep_result.root_solids {
        let shell_key = reg.solids[*solid_key].outer_shell;
        let mesh = brep::mesh::mesh_brep_shell(
            &reg.shells[shell_key], &reg, &mesh_config);
        if mesh.vertices.is_empty() { continue; }
        // ... add Coordinate3 + IndexedFaceSet to graph ...
    }

    // Add edge curves overlay
    let edge_count = brep::overlay::build_edge_curves(&mut graph, &reg, &brep_result.root_solids);
    eprintln!("[STEP] {} edge curves rendered", edge_count);

    if let Some(root_entry) = graph.get_mut(root) {
        root_entry.display_mode = Some(DisplayMode::ShadedWithEdges);
    }
    Ok(graph)
}
```

Remove from mod.rs:
- `parse_step_with_shared_topology` (no longer needed — always shared)
- `parse_step_full` (metadata available via brep_result)
- `build_hierarchical_scene` (replaced by brep pipeline)
- `build_hierarchical_scene_from_topo` (replaced by brep pipeline)
- `build_step_edges_overlay` (moved to brep/overlay.rs)
- `StepImportResult` (metadata now available from brep_result directly)

### 5.3 Update lib.rs Exports

**Modify `crates/rc3d-io/src/lib.rs`:**

Remove exports: `parse_step_with_shared_topology`, `parse_step_full`, `StepImportResult`, `topo`, `refine`, `lod`.

New exports:
```rust
pub use step::brep;  // B-Rep kernel public API
pub use step::brep::mesh::BRepMeshConfig;  // allow callers to tune meshing
pub use step::brep::heal::HealConfig;      // allow callers to tune healing
```

`import_file("step")` path: keeps calling `parse_step_file(path)` — which now uses B-Rep internally.

### 5.4 Migrate All Tests

**Modify `crates/rc3d-io/src/step/mod.rs` integration tests:**

Replace old `parse_step_file` → `assert!(mesh_count > 0)` tests with B-Rep pipeline tests:

```rust
#[test]
fn test_assembly_example_loads() {
    // ... load file ...
    let exchange = parser::parse_exchange(&text).unwrap();
    let brep_result = brep::build::build_brep(&exchange.entities).unwrap();
    let reg = &brep_result.registry;
    
    // Verify topology correctness
    assert!(!brep_result.root_solids.is_empty());
    for solid_key in &brep_result.root_solids {
        let solid = &reg.solids[*solid_key];
        let shell = &reg.shells[solid.outer_shell];
        assert!(!shell.faces.is_empty(), "shell should have faces");
        // Verify each face has a surface and wire
        for (face_key, _) in &shell.faces {
            let face = &reg.faces[*face_key];
            let wire = &reg.wires[face.outer_wire];
            assert!(!wire.edges.is_empty(), "face wire should have edges");
            // Verify each edge has the face's PCURVE
            for (edge_key, _) in &wire.edges {
                let edge = &reg.edges[*edge_key];
                assert!(edge.pcurves.contains_key(face_key),
                    "edge should have PCURVE for its face");
            }
        }
    }
    
    // Mesh and verify
    let config = brep::mesh::BRepMeshConfig::default();
    let mesh = brep::mesh::mesh_brep_shell(
        &reg.shells[reg.solids[brep_result.root_solids[0]].outer_shell],
        reg, &config);
    assert!(!mesh.vertices.is_empty());
    assert!(mesh.vertices.len() > 100);
}
```

**Modify `crates/rc3d-io/tests/step_files.rs`:**

Replace old topology collection + tessellation calls with B-Rep pipeline:
- Old: `topology::collect_shells()` → `tessellate::tessellate_faces()`
- New: `brep::build::build_brep()` → `brep::mesh::mesh_brep_shell()`

**Delete tests that test removed modules:**
- `tests/step_files.rs:test_shared_topology_import` — rewrite with B-Rep API
- All tests referencing `topo::`, `tessellate::`, `surface_tess::`, `refine::`, `curve::` — rewrite or remove

---

## File Structure After Migration

```
crates/rc3d-io/src/step/
├── brep/                       # NEW: parametric B-Rep kernel
│   ├── mod.rs
│   ├── geom.rs                 # CurveGeom + SurfaceGeom enums (D0/D1/D2/project/sample_adaptive)
│   ├── topo.rs                 # BRepVertex/Edge/Wire/Face/Shell/Solid + Orientation
│   ├── registry.rs             # SlotMap registry with hash indexes
│   ├── build.rs                # STEP → B-Rep builder (replaces topology.rs + topo/build.rs)
│   ├── overlay.rs              # Edge curve overlay builder (moved from mod.rs)
│   ├── heal/
│   │   ├── mod.rs              # Healing pipeline + HealConfig + HealReport
│   │   ├── reorder.rs          # Wire edge reordering (ShapeFix_Wire equivalent)
│   │   ├── gap.rs              # Gap closing via vertex merging
│   │   └── orient.rs           # Face orientation propagation (BFS)
│   └── mesh/
│       ├── mod.rs              # Unified meshing API + BRepMeshConfig
│       ├── edge_disc.rs        # Adaptive edge discretization (3D + 2D simultaneous)
│       ├── face_tri.rs         # Constrained Delaunay triangulation (replaces earcut)
│       ├── refiner.rs          # Deflection-driven refinement (per-edge midpoint vs surface)
│       └── optimize.rs         # Delaunay edge-flip quality optimization
├── bool/                       # MODIFIED: uses BRep types
│   ├── mod.rs
│   ├── intersect.rs            # face-face with PCURVE output on both faces
│   ├── classify.rs             # (kept)
│   ├── split.rs                # topology splitting (uses BRepEdge/Wire/Face)
│   └── select.rs               # (kept)
│
│   # ── REMOVED (replaced by brep/) ──
│   # topo/vertex.rs, topo/edge.rs, topo/shape.rs, topo/build.rs
│   # tessellate.rs, surface_tess.rs, refine.rs
│   # curve/derivative.rs, curve/arc_length.rs
│
│   # ── KEPT (used by brep/) ──
├── parser.rs                   # STEP entity parser (unchanged)
├── nurbs.rs                    # NURBS surface (used by brep/geom.rs SurfaceGeom::BSpline)
├── pcurve.rs                   # PCURVE extraction (used by brep/build.rs)
├── geom.rs                     # MODIFIED: curve/surface helpers → delegates to brep/geom.rs
├── topology.rs                 # REMOVED (replaced by brep/build.rs)
├── assembly.rs                 # Assembly transforms (used by mod.rs)
├── tree.rs                     # Assembly tree (kept)
├── entity_types.rs             # Entity type enum (used by parser.rs)
├── value.rs                    # StepValue enum (used by parser.rs)
├── validate.rs                 # Validation (kept, enhanced with brep-level checks)
├── header.rs                   # HEADER parsing (kept)
├── write/                      # STEP export (kept, enhanced to use BRep types)
├── xml.rs                      # XML export (kept)
├── pmi/                        # PMI extraction (kept)
├── fillet.rs                   # Fillet stub (kept, updated signature to use BRep types)
├── lod.rs                      # LOD (kept, updated to accept BRep mesh output)
└── mod.rs                      # REWRITTEN: single parse_step path
```

---

## Tasks

### Phase 1: B-Rep Core (12 tasks)

- [ ] **T1.1**: Create `brep/geom.rs` — `CurveGeom` enum with d0/d1/d2/curvature/sample_adaptive. Test: LINE d1 is direction, CIRCLE curvature = 1/r.
- [ ] **T1.2**: Create `brep/geom.rs` — `SurfaceGeom` enum with d0/d1/normal/project/evaluate_grid. Test: plane normal = Z, cylinder project returns correct UV.
- [ ] **T1.3**: Create `brep/topo.rs` — `Orientation` enum, `BRepVertex`, `BRepEdge` (with `pcurves: HashMap<FaceKey, CurveGeom>`).
- [ ] **T1.4**: Create `brep/topo.rs` — `BRepWire`, `BRepFace` (with surface + wire refs), `BRepShell`, `BRepSolid`.
- [ ] **T1.5**: Create `brep/registry.rs` — `BRepRegistry` with SlotMap storage, `find_or_add_vertex()`, `add_edge_with_pcurve()`.
- [ ] **T1.6**: Create `brep/registry.rs` — `find_shared_edges()`, `iter_faces()`, vertex hash index.
- [ ] **T1.7**: Create `brep/build.rs` — resolve surfaces from STEP entities → `SurfaceGeom` (all 10 types).
- [ ] **T1.8**: Create `brep/build.rs` — resolve curves + PCURVEs from EDGE_CURVE/SURFACE_CURVE chains → register edges with per-face PCURVEs.
- [ ] **T1.9**: Create `brep/build.rs` — build wires (EDGE_LOOP→BRepWire), faces (ADVANCED_FACE→BRepFace), shells, solids. Test: build a cube from STEP entities.
- [ ] **T1.10**: Create `brep/overlay.rs` — `build_edge_curves()` moved from mod.rs, adapted for BRep types.
- [ ] **T1.11**: Create `brep/mod.rs` — module declarations, public re-exports.
- [ ] **T1.12**: Add `slotmap = "1"` and `spade = "2"` to Cargo.toml.

### Phase 2: Meshing (9 tasks)

- [ ] **T2.1**: Create `brep/mesh/edge_disc.rs` — `EdgeDiscretizer` with `discretize()` using curvature-driven adaptive sampling. Test: discretize a circle edge — verify max chordal deviation < threshold.
- [ ] **T2.2**: Create `brep/mesh/edge_disc.rs` — `discretize_all()`: discretize shared edges once, cache results per EdgeKey.
- [ ] **T2.3**: Create `brep/mesh/face_tri.rs` — `FaceTriangulator` using spade CDT: insert edge polygon vertices as sites, edge segments as constraints.
- [ ] **T2.4**: Create `brep/mesh/face_tri.rs` — UV→3D mapping via `SurfaceGeom::d0()`, normal from `SurfaceGeom::d1()` cross product. Test: triangulate a cylinder face, verify all normals point outward.
- [ ] **T2.5**: Create `brep/mesh/refiner.rs` — `MeshRefiner` using per-edge midpoint deviation (compare surface evaluation at UV midpoint vs linear 3D midpoint).
- [ ] **T2.6**: Create `brep/mesh/refiner.rs` — iterative refinement with node insertion + local retriangulation. Test: refine a coarse sphere mesh, verify deviation decreases.
- [ ] **T2.7**: Create `brep/mesh/optimize.rs` — Delaunay edge flip: for each interior edge, check Delaunay criterion, flip if it improves min angle.
- [ ] **T2.8**: Create `brep/mesh/mod.rs` — `BRepMeshConfig` + `mesh_brep_shell()` unified API + `mesh_brep_solid()`.
- [ ] **T2.9**: Test: mesh cube → verify watertight (shared edge vertices match), mesh sphere → verify deviation < 1%.

### Phase 3: Healing (5 tasks)

- [ ] **T3.1**: Create `brep/heal/reorder.rs` — build adjacency graph, greedy chain walk, produce reordered edge list.
- [ ] **T3.2**: Create `brep/heal/gap.rs` — detect gaps between consecutive edges (end of edge_i vs start of edge_{i+1}), merge vertices within tolerance.
- [ ] **T3.3**: Create `brep/heal/orient.rs` — BFS from seed face, compare normals at shared edge, flip if inconsistent.
- [ ] **T3.4**: Create `brep/heal/mod.rs` — `HealConfig`, `HealReport`, `heal_shell()`. Test: heal a deliberately misordered wire.
- [ ] **T3.5**: Test: build from a STEP with a known orientation issue, verify healing fixes it.

### Phase 4: Boolean (3 tasks)

- [ ] **T4.1**: Rewrite `bool/intersect.rs` — `FaceIntersectionResult` with 3D curves + pcurves_on_a + pcurves_on_b. Update all existing intersect functions to produce PCURVEs.
- [ ] **T4.2**: Rewrite `bool/split.rs` — accept `BRepFace` + `FaceKey`, split edges at intersection points, produce new `FaceKey`s with proper PCURVEs on split edges.
- [ ] **T4.3**: Wire `bool/mod.rs` pipeline to use BRep types throughout.

### Phase 5: Integration & Removal (6 tasks)

- [ ] **T5.1**: Delete removed modules: `topo/`, `tessellate.rs`, `surface_tess.rs`, `refine.rs`, `curve/`. Remove their `pub mod` declarations from `mod.rs`.
- [ ] **T5.2**: Rewrite `mod.rs` — single `parse_step()` path using B-Rep pipeline (build→heal→mesh→scene). Remove `parse_step_with_options`, `parse_step_with_shared_topology`, `build_hierarchical_scene`, `build_hierarchical_scene_from_topo`, `build_step_edges_overlay`, `StepImportResult`.
- [ ] **T5.3**: Update `lib.rs` — remove old exports, add `pub use step::brep`, `BRepMeshConfig`, `HealConfig`. Keep `import_file` path using `parse_step`.
- [ ] **T5.4**: Migrate `mod.rs` integration tests — replace old topology+tessellation with B-Rep build+mesh. Verify PCURVEs are attached to edges, verify mesh watertightness.
- [ ] **T5.5**: Migrate `tests/step_files.rs` — replace `topology::collect_shells` + `tessellate::tessellate_faces` with `brep::build::build_brep` + `brep::mesh::mesh_brep_shell`. Remove `test_shared_topology_import`.
- [ ] **T5.6**: Full workspace compile + test: `cargo test --workspace`. Verify 0 errors, all tests pass.

---

## Self-Review Notes

- **Spec coverage**: All 10 gaps from analysis addressed:
  1. Face construction with parametric geometry ✓ (BRepFace carries SurfaceGeom + wire refs)
  2. Topology sharing ✓ (edges shared via SlotMap keys, PCURVEs per face)
  3. PCURVE precision ✓ (stored as CurveGeom on edges, not sampled to polygons)
  4. Delaunay triangulation ✓ (spade CDT replaces earcut)
  5. Adaptive edge discretization ✓ (curvature-driven, not uniform)
  6. Deflection-driven refinement ✓ (per-edge midpoint vs surface D0, not domain-center guess)
  7. Edge-swap quality optimization ✓ (new, no equivalent in old code)
  8. Shape healing ✓ (reorder + gap + orient)
  9. Boolean PCURVE output ✓ (intersection curves carry UV on both faces)
  10. Parametric B-Rep core ✓ (replaces flat mesh-extraction)

- **Aggressive replacement**: ~2500 lines removed, ~4000 lines added. Net +1500 lines but all are parametric geometry code.
- **No fallback**: Old code is deleted, not deprecated. Single code path reduces maintenance burden.
- **All tests migrated**: Integration tests verify B-Rep construction correctness (vertex sharing, PCURVE attachment, mesh watertightness) not just mesh statistics.
- **Dependencies**: slotmap (stable keys), spade (CDT) — both are well-maintained Rust crates.
