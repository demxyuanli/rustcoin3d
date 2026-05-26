# STEP B-Rep Redesigned Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all remaining gaps between rc3d-io BRep pipeline and OCC's `STEPControl_Reader → ShapeFix → BRepCheck → BRepMesh_IncrementalMesh`, with CDT-first triangulation, insert-time Steiner refinement, seam-in-wire, and always-on topology checking.

**Architecture:** Three phases. Phase A (8 tasks) fixes core correctness: seam edges go into wires, CDT is primary triangulator with Steiner insert-before-extract, surface-aware fallback, RemoveDegenerated, fixed config defaults. Phase B (4 tasks) adds quality: angular deflection, vertex tolerance fix, small-area detection. Phase C (4 tasks) adds completeness: assembly hierarchy, color transfer, T4 reference gate.

**Tech Stack:** Rust, `rc3d-io`, `spade`, `earcutr`, `rc3d-core::Vec3`

**Spec:** `docs/superpowers/specs/2026-05-26-step-brep-redesign-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/rc3d-io/src/step/brep/heal/seam.rs` | Modify | Insert seam edges into wires instead of face.seam_edges |
| `crates/rc3d-io/src/step/brep/heal/mod.rs` | Modify | Add fix_vertex_tolerance, fix_small_area; remove run_check field |
| `crates/rc3d-io/src/step/brep/heal/check.rs` | Modify | Add face-area, self-intersection, non-manifold checks |
| `crates/rc3d-io/src/step/brep/mesh/face_cdt.rs` | Rewrite | CDT-first + Steiner insertion loop (primary triangulator) |
| `crates/rc3d-io/src/step/brep/mesh/face_fill.rs` | Rewrite | surface_fill_3d replaces grid_fallback_3d; remove adapt_tris_to_deflection |
| `crates/rc3d-io/src/step/brep/mesh/face_uv.rs` | Modify | Rename UvSource::GridFallback → SurfaceFill |
| `crates/rc3d-io/src/step/brep/mesh/refiner.rs` | Modify | Add angular_deflection field |
| `crates/rc3d-io/src/step/brep/mesh/mod.rs` | Modify | Wire RemoveDegenerated; update config defaults; wire surface_fill_3d |
| `crates/rc3d-io/src/step/brep/mesh/report.rs` | Modify | Fix relative_deflection to use min() |
| `crates/rc3d-io/src/step/mod.rs` | Modify | Remove run_check guard; handle skip_face_keys |
| `crates/rc3d-io/src/step/brep/mesh/t4_quality.rs` | Create | Hausdorff comparison with OCC reference meshes |
| `crates/rc3d-io/src/step/topology.rs` | Modify | Parse NAUO assembly chain; parse STYLED_ITEM colors |
| `crates/rc3d-io/src/step/brep/build.rs` | Modify | Pass color data to BRepFace |
| `crates/rc3d-io/src/step/assembly.rs` | Modify | Build assembly tree from NAUO entities |
| `crates/rc3d-io/src/step/brep/topo.rs` | Modify | Add color field to BRepFace; add skip_mesh field |

---

# Phase A — Core Correctness

## Task 1: Insert seam edges into wires

**OCC reference:** `ShapeFix_Face::FixMissingSeam()` + `BRep_Tool::IsClosed()`

```
// OCC ShapeFix_Face::FixMissingSeam algorithm (simplified):
//
// Standard_Boolean ShapeFix_Face::FixMissingSeam(const TopoDS_Face& face)
// {
//   Handle(Geom_Surface) S = BRep_Tool::Surface(face);
//   if (!S->IsUClosed() && !S->IsVClosed()) return Standard_False;
//
//   TopoDS_Wire W = BRep_Tool::Wire(face);  // or VERTEX_LOOP
//   TopTools_IndexedMapOfShape edges;
//   TopExp::MapShapes(W, TopAbs_EDGE, edges);
//
//   if (edges.IsEmpty()) {
//     // VERTEX_LOOP: full closed surface (sphere, torus, etc.)
//     // Insert u=0 and/or v=0 isoparametric seam edges
//     if (S->IsUClosed()) {
//       // Build seam edge along u=UFirst (or ULast — same curve)
//       Standard_Real u_seam = S->UFirst();
//       TopoDS_Edge seam = BRepBuilderAPI_MakeEdge(
//         S->UIso(u_seam),   // isoparametric curve at u_seam
//         S->VFirst(),        // v range start
//         S->VLast()          // v range end
//       );
//       // Insert seam edge into wire
//       BRep_Builder B;
//       B.Add(W, seam);
//       // Also store as PCURVE on face
//       B.UpdateEdge(seam, Geom2d_Line(u_seam, 0, 1), face, Precision::Confusion());
//     }
//     if (S->IsVClosed()) {
//       // Same for v isoparametric seam
//     }
//   } else {
//     // Trimmed face with existing edges
//     // Check if wire already touches the parametric boundary
//     Standard_Real UMin, UMax, VMin, VMax;
//     BRepTools::UVBounds(face, UMin, UMax, VMin, VMax);
//
//     // Collect actual UV extent from edge PCURVEs
//     Standard_Real actUMin = RealLast(), actUMax = RealFirst();
//     for each edge in wire:
//       Handle(Geom2d_Curve) PC = BRep_Tool::CurveOnSurface(edge, face);
//       for t in [0, 0.125, 0.25, ... 1.0]:
//         gp_Pnt2d UV = PC->Value(t);
//         actUMin = Min(actUMin, UV.X());
//         actUMax = Max(actUMax, UV.X());
//
//     // If wire does NOT touch UMin ± tol, insert seam at UMin
//     if (S->IsUClosed() && actUMin > UMin + tolerance) {
//       // Insert seam edge at UMin
//       TopoDS_Edge seam = BuildSeamEdge(S, UMin);
//       B.Add(W, seam);  // Insert into wire, not face.seam_edges
//     }
//     // Similarly for UMax, VMin, VMax gaps...
//   }
//   return Standard_True;
// }
```

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/heal/seam.rs`

### Step 1: Modify `add_vertex_loop_seams` to write into wire

Find the function at ~line 43. Currently pushes to `face.seam_edges` at line 73. Change the block starting at line 72:

```rust
    let count = added.len();
    if count > 0 {
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.seam_edges.extend(added.iter().copied());
        }
    }
```

Replace with:

```rust
    let count = added.len();
    if count > 0 {
        // Insert seam edges directly into the outer wire (OCC style).
        // For VERTEX_LOOP faces the wire is empty, so these become the wire.
        if let Some(face) = reg.faces.get(face_key) {
            if let Some(wire) = reg.wires.get_mut(face.outer_wire) {
                for &ek in &added {
                    wire.edges.push((ek, Orientation::Forward));
                }
            }
        }
        // Also populate seam_edges for backward compatibility (overlay, diagnostics).
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.seam_edges.extend(added.iter().copied());
        }
    }
```

### Step 2: Modify `fix_trimmed_periodic_seam` to write into wire

Find the block at ~line 145 where it pushes to `face.seam_edges`:

```rust
    if let Some(ek) = build_u_isoparam_seam(reg, face_key, surface, u_seam, tol) {
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.seam_edges.push(ek);
        }
        1
    } else {
        0
    }
```

Replace with:

```rust
    if let Some(ek) = build_u_isoparam_seam(reg, face_key, surface, u_seam, tol) {
        if let Some(face) = reg.faces.get(face_key) {
            if let Some(wire) = reg.wires.get_mut(face.outer_wire) {
                wire.edges.push((ek, Orientation::Forward));
            }
            face.seam_edges.push(ek);
        }
        1
    } else {
        0
    }
```

### Step 3: Run tests

```bash
rtk cargo test -p rc3d-io heal::seam --lib
```

Expected: all existing seam tests pass. The seam edge is now in both the wire and face.seam_edges.

### Step 4: Commit

```bash
git add crates/rc3d-io/src/step/brep/heal/seam.rs
git commit -m "fix(rc3d-io): insert seam edges into wire.edges (OCC ShapeFix_Face alignment)"
```

---

## Task 2: Remove seam_edges iteration from collect_face_loops

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/mesh/face_uv.rs`

`collect_face_loops` currently only iterates `wire.edges`, which is now correct — seam edges are already in the wire. No code change needed. But verify: there was never seam_edges iteration code in this function. If any was added between now and implementation, remove it.

### Step 1: Verify no seam_edges iteration exists

```bash
grep -n "seam_edge" crates/rc3d-io/src/step/brep/mesh/face_uv.rs
```

Expected: no matches (or only in unrelated comments/test data).

### Step 2: Run full mesh tests

```bash
rtk cargo test -p rc3d-io mesh --lib
```

Expected: all pass. Seam edges now flow through wire → collect_face_loops automatically.

### Step 3: Commit

```bash
git commit -m "refactor(rc3d-io): seam edges flow through wire.edges — no separate iteration needed"
```

---

## Task 3: check_shell always runs + new checks

**OCC reference:** `BRepCheck_Analyzer::Perform()` + `BRepCheck_Wire` / `BRepCheck_Face` / `BRepCheck_Edge`

```
// OCC BRepCheck_Analyzer algorithm (simplified):
//
// void BRepCheck_Analyzer::Perform(const TopoDS_Shape& S)
// {
//   for each TopoDS_Face face in shape:
//     // --- BRepCheck_Face ---
//     // 1. Face must have a surface
//     if (BRep_Tool::Surface(face).IsNull())
//       AddResult(BRepCheck_InvalidSurface);
//
//     // 2. Face must have non-zero area
//     //    OCC checks via bounding box diagonal, not actual area
//     Bnd_Box B;
//     BRepBndLib::Add(face, B);
//     if (B.IsVoid() || B.SquareExtent() < Precision::SquareConfusion())
//       AddResult(BRepCheck_FaceTooSmall);
//
//     // --- BRepCheck_Wire ---
//     TopoDS_Wire W = BRep_Tool::Wire(face);
//     for each edge in wire:
//       // 3. Wire must be closed: end vertex of edge N = start vertex of edge N+1
//       TopoDS_Vertex v_end = TopExp::LastVertex(edge);
//       TopoDS_Vertex v_start_next = TopExp::FirstVertex(nextEdge);
//       if (!v_end.IsSame(v_start_next))
//         AddResult(BRepCheck_OpenWire);
//
//       // 4. Edge must have a PCURVE on this face
//       Standard_Real f, l;
//       Handle(Geom2d_Curve) PC = BRep_Tool::CurveOnSurface(edge, face, f, l);
//       if (PC.IsNull())
//         AddResult(BRepCheck_NoPCurve, BRepCheck_Warning);
//
//       // 5. Seam edge validity: if v_low == v_high, edge must be on a
//       //    closed parametric boundary (UClosed or VClosed)
//       if (BRep_Tool::IsClosed(edge, face)) {
//         if (!S->IsUClosed() && !S->IsVClosed())
//           AddResult(BRepCheck_InvalidSeamEdge);
//       }
//
//     // --- BRepCheck_SelfIntersection ---
//     // 6. Check UV boundary for self-intersection (O(n²) segment test)
//     //    OCC uses BRepCheck_SelfIntersection with UV tolerance
//     BRepCheck_SelfIntersection checker(face);
//     checker.Perform();
//     if (checker.HasIntersection())
//       AddResult(BRepCheck_SelfIntersectingWire);
//
//   // --- BRepCheck_Edge (non-manifold) ---
//   // 7. Count face references per edge
//   TopTools_IndexedDataMapOfShapeInteger edgeFaceCount;
//   for each face in shape:
//     for each edge in face:
//       edgeFaceCount(edge)++;
//   for each (edge, count) in edgeFaceCount:
//     if (count > 2)
//       AddResult(BRepCheck_NonManifoldEdge, BRepCheck_Warning);
// }
```

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/heal/check.rs`
- Modify: `crates/rc3d-io/src/step/brep/heal/mod.rs`
- Modify: `crates/rc3d-io/src/step/mod.rs`

### Step 1: Write failing test for new checks

Add to bottom of `heal/check.rs` tests:

```rust
#[test]
fn zero_area_face_yields_error() {
    let mut reg = BRepRegistry::new();
    let wire = reg.wires.insert(BRepWire { edges: vec![] });
    let face_key = reg.faces.insert(BRepFace {
        surface: SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        },
        outer_wire: wire,
        inner_wires: vec![],
        same_sense: true,
        tolerance: 1e-4,
        seam_edges: vec![],
    });
    let shell_key = reg.shells.insert(BRepShell {
        faces: vec![(face_key, Orientation::Forward)],
        closed: false,
        step_id: None,
    });
    let report = check_shell(shell_key, &reg);
    assert!(!report.errors.is_empty(),
        "zero-area face (empty wire with no seam edges) should be an error");
}

#[test]
fn non_manifold_edge_yields_warning() {
    let mut reg = BRepRegistry::new();
    let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
    let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
    let curve = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
    let ek = reg.edges.insert(BRepEdge {
        v_low: v0, v_high: v1, curve: curve.clone(),
        tolerance: 1e-4, pcurves: HashMap::new(),
    });
    // Create 3 faces sharing the same edge
    let mut face_keys = Vec::new();
    for _ in 0..3 {
        let wire = reg.wires.insert(BRepWire {
            edges: vec![(ek, Orientation::Forward)],
        });
        face_keys.push(reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
            },
            outer_wire: wire, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![],
        }));
    }
    let faces: Vec<_> = face_keys.iter().map(|&fk| (fk, Orientation::Forward)).collect();
    let shell_key = reg.shells.insert(BRepShell {
        faces, closed: false, step_id: None,
    });
    let report = check_shell(shell_key, &reg);
    assert!(!report.warnings.is_empty(),
        "edge shared by 3 faces should be flagged non-manifold");
}
```

### Step 2: Run — expect fail

```bash
rtk cargo test -p rc3d-io check::tests::zero_area_face --lib
rtk cargo test -p rc3d-io check::tests::non_manifold_edge --lib
```

### Step 3: Add zero-area check

In `check_face` (~line 40), after the wire retrieval, if `wire.edges.is_empty() && face.seam_edges.is_empty()`, the face has no boundary — zero area:

```rust
    if wire.edges.is_empty() && face.seam_edges.is_empty() {
        report.errors.push(format!("face {:?} has zero area (empty wire, no seams)", face_key));
        return;
    }
```

### Step 4: Add non-manifold edge check

Add a new function and call it from `check_shell`:

```rust
fn check_non_manifold(face_keys: &[(FaceKey, Orientation)], reg: &BRepRegistry) -> Vec<String> {
    let mut warnings = Vec::new();
    let mut edge_face_count: HashMap<EdgeKey, usize> = HashMap::new();
    for &(face_key, _) in face_keys {
        let Some(face) = reg.faces.get(face_key) else { continue };
        for wire_key in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
            let Some(wire) = reg.wires.get(*wire_key) else { continue };
            for &(ek, _) in &wire.edges {
                *edge_face_count.entry(ek).or_default() += 1;
            }
        }
    }
    for (ek, count) in edge_face_count {
        if count > 2 {
            warnings.push(format!("edge {:?} is non-manifold (shared by {} faces)", ek, count));
        }
    }
    warnings
}
```

Call it at the end of `check_shell`:

```rust
    let nm_warnings = check_non_manifold(&shell.faces, reg);
    report.warnings.extend(nm_warnings);
```

### Step 5: Add self-intersection check

Add after the wire-closure check in `check_face`:

```rust
fn check_uv_self_intersection(face_key: FaceKey, reg: &BRepRegistry) -> Vec<String> {
    let mut warnings = Vec::new();
    let Some(face) = reg.faces.get(face_key) else { return warnings };
    let Some(wire) = reg.wires.get(face.outer_wire) else { return warnings };
    // Collect UV boundary as line segments
    let mut segments: Vec<((f32,f32),(f32,f32))> = Vec::new();
    for &(ek, _) in &wire.edges {
        let Some(edge) = reg.edges.get(ek) else { continue };
        let Some(pcurve) = edge.pcurves.get(&face_key) else { continue };
        let p0 = pcurve.d0(0.0);
        let p1 = pcurve.d0(1.0);
        segments.push(((p0.x, p0.y), (p1.x, p1.y)));
    }
    // O(n²) check for non-adjacent segment intersection
    let n = segments.len();
    for i in 0..n {
        for j in (i+2)..n {
            if i == 0 && j == n-1 { continue; } // adjacent at loop closure
            if segments_intersect_2d(segments[i], segments[j]) {
                warnings.push(format!(
                    "face {:?}: UV boundary self-intersection between edge {} and edge {}",
                    face_key, i, j
                ));
            }
        }
    }
    warnings
}

fn segments_intersect_2d(a: ((f32,f32),(f32,f32)), b: ((f32,f32),(f32,f32))) -> bool {
    let ((ax0, ay0), (ax1, ay1)) = a;
    let ((bx0, by0), (bx1, by1)) = b;
    let d = (ax1-ax0)*(by1-by0) - (ay1-ay0)*(bx1-bx0);
    if d.abs() < 1e-12 { return false; }
    let t = ((bx0-ax0)*(by1-by0) - (by0-ay0)*(bx1-bx0)) / d;
    let u = ((bx0-ax0)*(ay1-ay0) - (by0-ay0)*(ax1-ax0)) / d;
    t > 1e-6 && t < 1.0 - 1e-6 && u > 1e-6 && u < 1.0 - 1e-6
}
```

Call from `check_shell`:

```rust
    for &(face_key, _) in &shell.faces {
        let si_warnings = check_uv_self_intersection(face_key, reg);
        report.warnings.extend(si_warnings);
    }
```

### Step 6: Remove `run_check` guard from heal_shell and step/mod.rs

In `heal/mod.rs`:
- Remove `run_check: bool` field from `HealConfig` (line 38)
- Remove `run_check: false,` from `Default` impl (line 48)
- At end of `heal_shell` (before line 101 `report`), add:

```rust
    let check_report = check_shell(shell_key, reg);
    for e in &check_report.errors {
        log::warn!("[BRep check] {}", e);
    }
    for w in &check_report.warnings {
        log::debug!("[BRep check] {}", w);
    }
```

In `step/mod.rs` (~line 97), remove the `if heal_config.run_check { ... }` block — `check_shell` now runs inside `heal_shell` automatically. Also remove `use brep::heal::check_shell` import if the caller-side check block is removed.

### Step 7: Run tests

```bash
rtk cargo test -p rc3d-io check::tests --lib
rtk cargo test -p rc3d-io heal --lib
rtk cargo test -p rc3d-io --lib
```

### Step 8: Commit

```bash
git add crates/rc3d-io/src/step/brep/heal/check.rs \
        crates/rc3d-io/src/step/brep/heal/mod.rs \
        crates/rc3d-io/src/step/mod.rs
git commit -m "feat(rc3d-io): check_shell always runs; add area, non-manifold, self-intersection checks"
```

---

## Task 4: CDT-first + Steiner insertion loop

**OCC reference:** `BRepMesh_Face::Update()` + `BRepMesh_Delaun`

```
// OCC BRepMesh_Face algorithm (simplified from BRepMesh_Face.cxx):
//
// void BRepMesh_Face::Update(const TopoDS_Face& theFace, const IMeshTools_Parameters& theParams)
// {
//   // 1. Build wire discretization from shared edge polygons
//   //    → BRepMesh_EdgeDiscret already done; wires carry discretized edges
//
//   // 2. Build face discretization in UV domain
//   BRepMesh_VertexTool aVertexTool;  // UV vertex pool
//   BRepMesh_Delaun aDelaunayTool;    // constrained Delaunay triangulator
//
//   // 2a. Add boundary nodes from edge discretization
//   for each wire loop (outer + inner):
//     for each edge in loop:
//       for each discretized point (U,V) on edge:
//         aVertexTool.Add(Point2D(U, V))
//         aDelaunayTool.AddVertex(Point2D(U, V))
//
//   // 2b. Add constraints (boundary edges)
//   for each wire loop:
//     for each consecutive pair of points on loop boundary:
//       aDelaunayTool.AddConstraint(Point2D(U_i, V_i), Point2D(U_{i+1}, V_{i+1}))
//
//   // 2c. Interior node insertion (deflection-driven Steiner points)
//   Standard_Real aDeflection = theParams.DeflectionInterior;
//   Standard_Real aMinSize    = theParams.MinSize;
//   Standard_Integer aMaxIter = theParams.MaxIterations;
//
//   for (int iter = 0; iter < aMaxIter; ++iter) {
//     Standard_Boolean anySplit = Standard_False;
//     for each triangle T = (P0, P1, P2) in Delaunay triangulation:
//       // Skip triangles with border edges (watertight rule — not OCC default)
//       // but OCC does check deflection on ALL triangles
//
//       Standard_Real maxDev = 0.0;
//       for each edge E = (A, B) of T:
//         // Linear deflection: chord error at edge midpoint
//         Point2D UV_mid = (UV_A + UV_B) / 2.0;
//         gp_Pnt P_mid      = surface->Value(UV_mid);           // surface midpoint
//         gp_Pnt P_linear   = (P_A + P_B) / 2.0;               // linear midpoint
//         Standard_Real dev = P_mid.Distance(P_linear);
//         maxDev = Max(maxDev, dev);
//
//         // Angular deflection: normal variation along edge
//         gp_Vec N_A = surface->DN(UV_A, 1, 0) ^ surface->DN(UV_A, 0, 1);
//         gp_Vec N_B = surface->DN(UV_B, 1, 0) ^ surface->DN(UV_B, 0, 1);
//         Standard_Real angle = N_A.Angle(N_B);  // acos(dot/|NA|*|NB|)
//         maxDev = Max(maxDev, angle * edgeLength); // weighted by edge length
//
//       // Split condition: (linear > deflection) || (angular > angular_deflection)
//       if (maxDev > aDeflection && edgeLength > aMinSize) {
//         // Insert Steiner point at UV centroid projected to surface
//         Point2D UV_centroid = (UV_0 + UV_1 + UV_2) / 3.0;
//         gp_Pnt  P_centroid  = surface->Value(UV_centroid);
//         aDelaunayTool.AddVertex(UV_centroid, P_centroid);
//         anySplit = Standard_True;
//       }
//     }
//     if (!anySplit) break;
//   }
//
//   // 2d. Extract triangles that are inside the trim domain
//   for each triangle in Delaunay:
//     Point2D centroid = (UV_0 + UV_1 + UV_2) / 3.0;
//     if (classifier.IsInside(centroid)):  // point-in-trim test
//       Emit triangle (i0, i1, i2)
//       // Compute normal from surface D1U × D1V
//       normals[i0] = Normalize(surface->DN(UV_0, 1, 0) ^ surface->DN(UV_0, 0, 1))
//       ... same for i1, i2 ...
// }
```

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/mesh/face_cdt.rs` (rewrite core)
- Modify: `crates/rc3d-io/src/step/brep/mesh/face_fill.rs` (remove adapt_tris_to_deflection; use new CDT)
- Modify: `crates/rc3d-io/src/step/brep/mesh/face_uv.rs` (no changes needed)

### Step 1: Write failing test for Steiner insertion

Add to `face_cdt.rs` tests:

```rust
#[test]
fn steiner_split_on_curved_surface() {
    use crate::step::brep::geom::SurfaceGeom;
    use crate::step::brep::topo::BRepFace;
    use super::super::face_uv::{UvLoop, UvSource, UvVertex};

    let surface = SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 10.0 };
    let face = BRepFace {
        surface,
        outer_wire: Default::default(),
        inner_wires: vec![],
        same_sense: true,
        tolerance: 1e-4,
        seam_edges: vec![],
    };
    // A large spherical patch — 4 boundary verts, very curved
    let outer = UvLoop {
        boundary: vec![
            UvVertex { global_idx: 0, uv: (0.0, 0.0) },
            UvVertex { global_idx: 1, uv: (1.57, 0.0) },
            UvVertex { global_idx: 2, uv: (1.57, 1.57) },
            UvVertex { global_idx: 3, uv: (0.0, 1.57) },
        ],
    };
    let loops = FaceUvLoops { outer, inners: vec![], uv_source: UvSource::Pcurve };

    let mut verts = vec![
        surface.d0_native(0.0, 0.0),
        surface.d0_native(1.57, 0.0),
        surface.d0_native(1.57, 1.57),
        surface.d0_native(0.0, 1.57),
    ];
    let mut norms = vec![Vec3::Z; 4];
    let mut indices = Vec::new();
    let mut pos_map = HashMap::new();
    for (i, v) in verts.iter().enumerate() {
        let hash = f32x3_quantized_bits([v.x, v.y, v.z]);
        pos_map.insert(hash, i);
    }

    let config = FaceFillConfig {
        enable_interior: true,
        deflection_interior: 0.5, // loose enough to not split
        min_size: 0.1,
        min_size_relative: 0.0,
        max_adapt_iterations: 4,
    };

    let tris_no_split = triangulate_uv_cdt_with_steiner(
        &loops, &face, &mut verts, &mut norms, &mut indices, &mut pos_map, &config,
    );
    let base_verts = verts.len();

    // Now with tight deflection — should insert Steiner points
    let config_tight = FaceFillConfig {
        deflection_interior: 0.01,
        max_adapt_iterations: 4,
        ..config.clone()
    };
    let tris_with_split = triangulate_uv_cdt_with_steiner(
        &loops, &face, &mut verts, &mut norms, &mut indices, &mut pos_map, &config_tight,
    );
    assert!(verts.len() > base_verts,
        "tight deflection should insert Steiner points, verts: {} -> {}",
        base_verts, verts.len());
    assert!(tris_with_split.len() > tris_no_split.len(),
        "Steiner points should produce more triangles");
}
```

### Step 2: Implement `triangulate_uv_cdt_with_steiner`

Replace the body of `triangulate_uv_cdt` with a new function `triangulate_uv_cdt_with_steiner` that includes Steiner insertion:

```rust
use rc3d_core::utils::hash::f32x3_quantized_bits;

/// CDT-first triangulation with insert-time Steiner refinement.
/// Returns (flat list of global vertex indices per triangle, max_chord_error).
pub fn triangulate_uv_cdt_with_steiner(
    loops: &FaceUvLoops,
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    config: &FaceFillConfig,
) -> (Vec<usize>, f32) {
    // Boundary global indices in order
    let boundary_gis: Vec<usize> = loops.outer.boundary.iter()
        .chain(loops.inners.iter().flat_map(|l| l.boundary.iter()))
        .map(|v| v.global_idx)
        .collect();

    // Build initial CDT
    let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
        ConstrainedDelaunayTriangulation::new();
    let mut handles: Vec<spade::handles::FixedVertexHandle> = Vec::new();
    let mut uv_to_gi: HashMap<(u64, u64), usize> = HashMap::new(); // quantized UV → global_idx

    let insert_uv_point = |cdt: &mut ConstrainedDelaunayTriangulation<Point2<f64>>,
                           uv: (f32, f32), gi: usize,
                           handles: &mut Vec<_>,
                           uv_to_gi: &mut HashMap<_, _>|
     -> Option<spade::handles::FixedVertexHandle> {
        let pt = Point2::new(uv.0 as f64, uv.1 as f64);
        let Ok(h) = cdt.insert(pt) else { return None };
        handles.push(h);
        // Quantize UV for stable lookup
        let key = ((uv.0 * 1e6) as u64, (uv.1 * 1e6) as u64);
        uv_to_gi.insert(key, gi);
        Some(h)
    };

    // Insert outer boundary
    let outer_start = handles.len();
    for v in &loops.outer.boundary {
        insert_uv_point(&mut cdt, v.uv, v.global_idx, &mut handles, &mut uv_to_gi);
    }
    // Add outer constraints
    let n_outer = loops.outer.boundary.len();
    for i in 0..n_outer {
        let a = handles[outer_start + i];
        let b = handles[outer_start + (i + 1) % n_outer];
        let _ = cdt.try_add_constraint(a, b);
    }

    // Insert inner boundaries with constraints
    for inner in &loops.inners {
        let start = handles.len();
        for v in &inner.boundary {
            insert_uv_point(&mut cdt, v.uv, v.global_idx, &mut handles, &mut uv_to_gi);
        }
        let n = inner.boundary.len();
        for i in 0..n {
            let a = handles[start + i];
            let b = handles[start + (i + 1) % n];
            let _ = cdt.try_add_constraint(a, b);
        }
    }

    // Steiner insertion loop
    let mut max_chord = 0.0f32;
    if config.enable_interior && config.deflection_interior > 0.0 {
        for _ in 0..config.max_adapt_iterations {
            let mut splits: Vec<(f64, f64, usize)> = Vec::new(); // (u, v, source_gi)
            for face_handle in cdt.inner_faces() {
                let verts: Vec<_> = face_handle.vertices().iter()
                    .map(|v| v.fix().index()).collect();
                if verts.len() != 3 { continue; }
                let i0 = verts[0]; let i1 = verts[1]; let i2 = verts[2];
                let uv0 = (cdt.vertex(handles[i0]).position().x as f32,
                           cdt.vertex(handles[i0]).position().y as f32);
                let uv1 = (cdt.vertex(handles[i1]).position().x as f32,
                           cdt.vertex(handles[i1]).position().y as f32);
                let uv2 = (cdt.vertex(handles[i2]).position().x as f32,
                           cdt.vertex(handles[i2]).position().y as f32);

                let p0 = face.surface.d0_native(uv0.0, uv0.1);
                let p1 = face.surface.d0_native(uv1.0, uv1.1);
                let p2 = face.surface.d0_native(uv2.0, uv2.1);

                // Check linear deflection at edge midpoints
                let mut tri_max_dev = 0.0f32;
                for (a, b, uva, uvb) in [
                    (&p0, &p1, uv0, uv1),
                    (&p1, &p2, uv1, uv2),
                    (&p2, &p0, uv2, uv0),
                ] {
                    let mid_3d = (*a + *b) * 0.5;
                    let mid_uv = ((uva.0 + uvb.0) * 0.5, (uva.1 + uvb.1) * 0.5);
                    let on_surf = face.surface.d0_native(mid_uv.0, mid_uv.1);
                    let dev = (mid_3d - on_surf).length();
                    tri_max_dev = tri_max_dev.max(dev);
                    // Check MinSize
                    let edge_len = (*a - *b).length();
                    if dev > config.deflection_interior
                        && edge_len > effective_min_size(&boundary_gis.iter().copied().collect(),
                                                         global_vertices, config)
                    {
                        splits.push((mid_uv.0 as f64, mid_uv.1 as f64, 0));
                    }
                }
                max_chord = max_chord.max(tri_max_dev);
            }
            if splits.is_empty() { break; }
            // Dedup splits by UV proximity
            let mut inserted = HashSet::new();
            for (u, v, _) in &splits {
                let key = ((*u * 1e4) as u64, (*v * 1e4) as u64);
                if inserted.contains(&key) { continue; }
                inserted.insert(key);
                let pt_3d = face.surface.d0_native(*u as f32, *v as f32);
                let hash = f32x3_quantized_bits([pt_3d.x, pt_3d.y, pt_3d.z]);
                let gi = *pos_to_idx.entry(hash).or_insert_with(|| {
                    let i = global_vertices.len();
                    global_vertices.push(pt_3d);
                    let mut n = face.surface.normal_native(*u as f32, *v as f32);
                    if !face.same_sense { n = -n; }
                    global_normals.push(n);
                    i
                });
                insert_uv_point(&mut cdt, (*u as f32, *v as f32), gi, &mut handles, &mut uv_to_gi);
            }
        }
    }

    // Extract triangles
    let outer_uv: Vec<(f32, f32)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
    let inner_uv: Vec<Vec<(f32, f32)>> = loops.inners.iter()
        .map(|l| l.boundary.iter().map(|v| v.uv).collect())
        .collect();

    let mut tris = Vec::new();
    for face_handle in cdt.inner_faces() {
        let verts: Vec<_> = face_handle.vertices().iter()
            .map(|v| v.fix().index()).collect();
        if verts.len() != 3 { continue; }
        let i0 = verts[0]; let i1 = verts[1]; let i2 = verts[2];
        let uv0 = (cdt.vertex(handles[i0]).position().x as f32,
                   cdt.vertex(handles[i0]).position().y as f32);
        let uv1 = (cdt.vertex(handles[i1]).position().x as f32,
                   cdt.vertex(handles[i1]).position().y as f32);
        let uv2 = (cdt.vertex(handles[i2]).position().x as f32,
                   cdt.vertex(handles[i2]).position().y as f32);
        let cu = (uv0.0 + uv1.0 + uv2.0) / 3.0;
        let cv = (uv0.1 + uv1.1 + uv2.1) / 3.0;
        if point_in_trim(cu, cv, &outer_uv, &inner_uv) {
            let key0 = ((uv0.0 * 1e6) as u64, (uv0.1 * 1e6) as u64);
            let key1 = ((uv1.0 * 1e6) as u64, (uv1.1 * 1e6) as u64);
            let key2 = ((uv2.0 * 1e6) as u64, (uv2.1 * 1e6) as u64);
            if let (Some(&gi0), Some(&gi1), Some(&gi2)) =
                (uv_to_gi.get(&key0), uv_to_gi.get(&key1), uv_to_gi.get(&key2))
            {
                tris.push(gi0);
                tris.push(gi1);
                tris.push(gi2);
            }
        }
    }

    (tris, max_chord)
}
```

Keep the old `triangulate_uv_cdt` as a thin wrapper for tests:

```rust
pub fn triangulate_uv_cdt(loops: &FaceUvLoops) -> Option<Vec<usize>> {
    let face = BRepFace {
        surface: SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        },
        outer_wire: Default::default(),
        inner_wires: vec![],
        same_sense: true,
        tolerance: 1e-4,
        seam_edges: vec![],
    };
    let mut verts = Vec::new();
    let mut norms = Vec::new();
    let mut idxs = Vec::new();
    let mut pmap = HashMap::new();
    let config = FaceFillConfig { enable_interior: false, ..Default::default() };
    let (tris, _) = triangulate_uv_cdt_with_steiner(
        loops, &face, &mut verts, &mut norms, &mut idxs, &mut pmap, &config,
    );
    if tris.is_empty() { None } else { Some(tris) }
}
```

### Step 3: Remove `adapt_tris_to_deflection` from face_fill.rs

Delete functions: `adapt_tris_to_deflection`, `tri_max_chord_error`, `tri_needs_split`. Remove `insert_surface_point` (replaced by inline code in Steiner loop). Keep `fix_winding`, `accumulate_normals`, `effective_min_size`, `face_boundary_diagonal`.

Remove the `adapt_tris_to_deflection` call from `fill_trimmed` (~line 177). The `fill_trimmed` function now calls `triangulate_uv_cdt_with_steiner` as its primary triangulator, with earcut as fallback only.

### Step 4: Update fill_trimmed to use new CDT function

In `fill_trimmed` (~line 116), replace the current CDT → earcut → adapt logic:

```rust
    let mut max_chord_error = 0.0f32;

    let (tris, chord) = triangulate_uv_cdt_with_steiner(
        loops, face,
        global_vertices, global_normals,
        all_indices, pos_to_idx, config,
    );
    max_chord_error = chord;

    if tris.is_empty() {
        // CDT failed — fall back to earcut with same Steiner loop
        // ... earcut code as before, using surface.project() for Steiner ...
    }
```

### Step 5: Run tests

```bash
rtk cargo test -p rc3d-io face_cdt --lib
rtk cargo test -p rc3d-io face_fill --lib
rtk cargo test -p rc3d-io mesh --lib
rtk cargo test -p rc3d-io --lib
```

### Step 6: Commit

```bash
git add crates/rc3d-io/src/step/brep/mesh/face_cdt.rs \
        crates/rc3d-io/src/step/brep/mesh/face_fill.rs
git commit -m "feat(rc3d-io): CDT-first triangulation with insert-time Steiner refinement"
```

---

## Task 5: surface_fill_3d replaces grid_fallback_3d

**OCC reference:** `BRepMesh_Face::Update()` (surface-only path — no PCURVEs available)

```
// OCC BRepMesh_Face when edge PCURVEs are missing (degraded path):
//
// void BRepMesh_Face::updateFaceFromSurface(const TopoDS_Face& face)
// {
//   Handle(Geom_Surface) S = BRep_Tool::Surface(face);
//
//   // 1. Collect 3D boundary points from edge discretization
//   //    (these come from BRepMesh_EdgeDiscret — always available in 3D)
//   TColgp_SequenceOfPnt boundary3D;
//   for each edge in wire:
//     Handle(Poly_Polygon3D) poly = BRep_Tool::Polygon3D(edge);
//     for each point in poly:
//       boundary3D.Append(point);
//
//   // 2. Project each 3D boundary point to surface → get UV
//   GeomAPI_ProjectPointOnSurf projector;
//   TColgp_SequenceOfPnt2d boundaryUV;
//   Standard_Integer successCount = 0;
//   for each point P in boundary3D:
//     projector.Init(P, S);  // project point to surface
//     if (projector.NbPoints() > 0) {
//       Standard_Real U, V;
//       projector.LowerDistanceParameters(U, V);
//       boundaryUV.Append(gp_Pnt2d(U, V));
//       successCount++;
//     }
//
//   // 3. Build UV triangulation
//   if (successCount > 0.7 * boundary3D.Size()) {
//     // Most points projected successfully → use projected UVs
//     BRepMesh_Delaun del;
//     for each projected (U,V):
//       del.AddVertex(gp_Pnt2d(U, V));
//     AddBoundaryConstraints(del, boundaryUV);
//     // Steiner loop (same as Task 4 algorithm)
//     RunSteinerLoop(del, S, theParams);
//     ExtractTriangles(del);
//   } else {
//     // Fallback: build planar parameterization from 3D points
//     // (OCC does NOT do this — it skips the face. But we have surface
//     //  evaluation available, so we can use it for interior points.)
//     gp_Ax3 planeAx = FitPlaneToPoints(boundary3D);
//
//     // Parameterize boundary in the fitted plane
//     TColgp_SequenceOfPnt2d planeUV;
//     for each P in boundary3D:
//       gp_Vec rel = P.XYZ() - planeAx.Location().XYZ();
//       Standard_Real u = rel.Dot(planeAx.XDirection());
//       Standard_Real v = rel.Dot(planeAx.YDirection());
//       planeUV.Append(gp_Pnt2d(u, v));
//
//     // Build CDT in plane space, then project interior nodes to surface
//     BRepMesh_Delaun del;
//     for each (u,v) in planeUV:
//       del.AddVertex(gp_Pnt2d(u, v));
//     AddBoundaryConstraints(del, planeUV);
//
//     // Steiner loop: for each triangle, project centroid to SURFACE
//     RunSteinerLoop(del, S, theParams, /*useSurfaceProjection=*/true);
//     // ↑ This is the key difference from our old grid_fallback_3d:
//     //   Steiner points go through surface.project() → surface.d0_native()
//     //   instead of staying on the fitted plane.
//     ExtractTriangles(del);
//   }
// }
```

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/mesh/face_fill.rs`
- Modify: `crates/rc3d-io/src/step/brep/mesh/mod.rs` (update call sites)
- Modify: `crates/rc3d-io/src/step/brep/mesh/face_uv.rs` (rename UvSource variant)

### Step 1: Replace grid_fallback_3d with surface_fill_3d

Delete `grid_fallback_3d` (lines 215-325). Implement `surface_fill_3d`:

```rust
pub fn surface_fill_3d(
    face_key: FaceKey,
    boundary_global: &[usize],
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    config: &FaceFillConfig,
) -> FaceMeshRange {
    let first_tri = all_indices.len() / 4;
    let boundary_set: HashSet<usize> = boundary_global.iter().copied().collect();

    if boundary_global.len() < 3 {
        return FaceMeshRange {
            face_key, first_tri, tri_count: 0,
            boundary_global: boundary_set, max_chord_error: 0.0,
        };
    }

    // Project boundary 3D points to surface UV
    let mut projected_uvs: Vec<Option<(f32, f32)>> = Vec::new();
    let mut success_count = 0usize;
    for &gi in boundary_global {
        let pt = global_vertices[gi];
        let uv = face.surface.project(pt)
            .or_else(|| face.surface.inverse_native_uv(pt, 0.5));
        if uv.is_some() { success_count += 1; }
        projected_uvs.push(uv);
    }

    let uv_source: UvSource;
    let mut tris: Vec<usize> = Vec::new();
    let mut max_chord = 0.0f32;

    if success_count as f32 / boundary_global.len() as f32 > 0.7 {
        uv_source = UvSource::SurfaceFill;
        // Build CDT from projected UVs, then Steiner loop
        let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
            ConstrainedDelaunayTriangulation::new();
        let mut handles = Vec::new();
        let mut uv_to_gi: HashMap<(u64, u64), usize> = HashMap::new();

        for (i, &gi) in boundary_global.iter().enumerate() {
            if let Some((u, v)) = projected_uvs[i] {
                let pt = Point2::new(u as f64, v as f64);
                if let Ok(h) = cdt.insert(pt) {
                    handles.push(h);
                    let key = ((u * 1e6) as u64, (v * 1e6) as u64);
                    uv_to_gi.insert(key, gi);
                }
            }
        }
        // Add constraints
        let n = handles.len();
        for i in 0..n {
            let a = handles[i];
            let b = handles[(i + 1) % n];
            let _ = cdt.try_add_constraint(a, b);
        }

        // Steiner loop on surface
        if config.enable_interior {
            for _ in 0..config.max_adapt_iterations {
                let mut splits = Vec::new();
                for face_h in cdt.inner_faces() {
                    let verts: Vec<_> = face_h.vertices().iter()
                        .map(|v| v.fix().index()).collect();
                    if verts.len() != 3 { continue; }
                    // Check deflection at edge midpoints via surface.project()
                    for j in 0..3 {
                        let a = cdt.vertex(handles[verts[j]]).position();
                        let b = cdt.vertex(handles[verts[(j+1)%3]]).position();
                        let mid_uv = ((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
                        let mid_3d = face.surface.d0_native(mid_uv.0 as f32, mid_uv.1 as f32);
                        let pa = face.surface.d0_native(a.x as f32, a.y as f32);
                        let pb = face.surface.d0_native(b.x as f32, b.y as f32);
                        let linear_mid = (pa + pb) * 0.5;
                        let dev = (mid_3d - linear_mid).length();
                        let edge_len = (pa - pb).length();
                        let min_sz = effective_min_size(&boundary_set, global_vertices, config);
                        if dev > config.deflection_interior && edge_len > min_sz {
                            splits.push((mid_uv.0, mid_uv.1));
                            max_chord = max_chord.max(dev);
                        }
                    }
                }
                if splits.is_empty() { break; }
                let mut dedup = HashSet::new();
                for (u, v) in splits {
                    let key = ((u * 1e4) as u64, (v * 1e4) as u64);
                    if dedup.contains(&key) { continue; }
                    dedup.insert(key);
                    let pt_3d = face.surface.d0_native(u as f32, v as f32);
                    let hash = f32x3_quantized_bits([pt_3d.x, pt_3d.y, pt_3d.z]);
                    let gi = *pos_to_idx.entry(hash).or_insert_with(|| {
                        let i = global_vertices.len();
                        global_vertices.push(pt_3d);
                        let mut n = face.surface.normal_native(u as f32, v as f32);
                        if !face.same_sense { n = -n; }
                        global_normals.push(n);
                        i
                    });
                    let pt = Point2::new(u, v);
                    if let Ok(h) = cdt.insert(pt) {
                        handles.push(h);
                        let key = ((u * 1e6) as u64, (v * 1e6) as u64);
                        uv_to_gi.insert(key, gi);
                    }
                }
            }
        }

        // Extract
        for face_h in cdt.inner_faces() {
            let verts: Vec<_> = face_h.vertices().iter()
                .map(|v| v.fix().index()).collect();
            if verts.len() != 3 { continue; }
            for j in 0..3 {
                let pos = cdt.vertex(handles[verts[j]]).position();
                let key = ((pos.x * 1e6) as u64, (pos.y * 1e6) as u64);
                if let Some(&gi) = uv_to_gi.get(&key) {
                    tris.push(gi);
                }
            }
        }
    } else {
        // Fallback: planar parameterization (rare, for badly broken UVs)
        uv_source = UvSource::SurfaceFill;
        // ... existing grid_fallback logic but with surface.project() for
        // interior points. For now: use the same plane-fit approach but
        // always project Steiner points to surface.
    }

    // Emit triangles with winding fix + normal accumulation
    for chunk in tris.chunks(3) {
        if chunk.len() != 3 { continue; }
        let (mut i0, mut i1, mut i2) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if i0 == i1 || i1 == i2 || i2 == i0 { continue; }
        fix_winding(&mut i0, &mut i1, &mut i2, global_vertices, &face.surface, face.same_sense);
        all_indices.extend_from_slice(&[i0, i1, i2, -1]);
        accumulate_normals(i0, i1, i2, global_vertices, global_normals);
    }

    let tri_count = all_indices.len() / 4 - first_tri;
    FaceMeshRange {
        face_key, first_tri, tri_count,
        boundary_global: boundary_set, max_chord_error: max_chord,
    }
}
```

### Step 2: Update call sites in mod.rs

Replace `grid_fallback_3d` with `surface_fill_3d`:

```rust
use face_fill::{FaceFillConfig, FaceMeshRange, fill_trimmed, surface_fill_3d};

// In the fallback path (~line 241):
surface_fill_3d(
    info.face_key,
    &boundary_ordered,
    face,
    &mut global_vertices,
    &mut global_normals,
    &mut all_indices,
    &mut pos_to_idx,
    &scaled_config.face,
)
```

### Step 3: Rename UvSource variant

In `face_uv.rs`:

```rust
pub enum UvSource {
    Pcurve,
    Synthetic,
    SurfaceFill,  // was GridFallback
}
```

Update all match arms and comparisons referencing `UvSource::GridFallback` → `UvSource::SurfaceFill`.

### Step 4: Run tests

```bash
rtk cargo test -p rc3d-io face_fill --lib
rtk cargo test -p rc3d-io mesh --lib
rtk cargo test -p rc3d-io --lib
```

### Step 5: Commit

```bash
git add crates/rc3d-io/src/step/brep/mesh/face_fill.rs \
        crates/rc3d-io/src/step/brep/mesh/mod.rs \
        crates/rc3d-io/src/step/brep/mesh/face_uv.rs
git commit -m "feat(rc3d-io): surface_fill_3d with surface-aware CDT replaces grid_fallback_3d"
```

---

## Task 6: Wire RemoveDegenerated

**OCC reference:** `BRepMesh::RemoveDegenerated()`

```
// OCC BRepMesh::RemoveDegenerated algorithm:
//
// void BRepMesh::RemoveDegenerated(
//   Handle(Poly_Triangulation)& theMesh,
//   const TColgp_Array1OfPnt&    theNodes)
// {
//   // 1. Collect degenerate triangles (area < Precision::SquareConfusion())
//   const TColgp_Array1OfPnt2d& UVNodes = theMesh->UVNodes();
//   const Poly_Array1OfTriangle& triangles = theMesh->Triangles();
//   Standard_Integer nbTri = triangles.Length();
//
//   TColStd_SequenceOfInteger degenerateIdx;
//   for (Standard_Integer i = 1; i <= nbTri; i++) {
//     Standard_Integer n1, n2, n3;
//     triangles(i).Get(n1, n2, n3);
//
//     // Degenerate check in 3D: triangle area ≈ 0
//     const gp_Pnt& P1 = theNodes(n1);
//     const gp_Pnt& P2 = theNodes(n2);
//     const gp_Pnt& P3 = theNodes(n3);
//
//     gp_Vec V1(P1, P2);
//     gp_Vec V2(P1, P3);
//     Standard_Real area = (V1 ^ V2).Magnitude() * 0.5;
//
//     if (area < Precision::SquareConfusion()) {
//       // Degenerate — mark for removal
//       degenerateIdx.Append(i);
//     }
//   }
//
//   // 2. Remove degenerate triangles by swapping with last valid triangle
//   //    (OCC preserves triangle count by compacting the array)
//   for (Standard_Integer i = 0; i < degenerateIdx.Size(); i++) {
//     Standard_Integer idxToRemove = degenerateIdx(i);
//     theMesh->RemoveTriangle(idxToRemove);
//   }
//
//   // 3. Cleanup orphaned nodes (not indexed by any triangle)
//   theMesh->RemoveOrphanNodes();
// }
```

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/mesh/mod.rs`

### Step 1: Add cull_degenerate_tris function

```rust
fn cull_degenerate_tris(indices: &mut Vec<i32>, vertices: &[Vec3]) -> usize {
    let mut out = Vec::with_capacity(indices.len());
    let mut removed = 0usize;
    for chunk in indices.chunks(4) {
        if chunk.len() < 4 || chunk[3] != -1 {
            out.extend_from_slice(chunk);
            continue;
        }
        let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            out.extend_from_slice(chunk);
            continue;
        }
        let area = (vertices[i0] - vertices[i1])
            .cross(vertices[i0] - vertices[i2])
            .length();
        if area > 1e-12 {
            out.extend_from_slice(chunk);
        } else {
            removed += 1;
        }
    }
    *indices = out;
    removed
}
```

### Step 2: Wire after Phase 2 and after Phase 3

After the face loop (before refine, ~line 275):

```rust
    let deg_after_fill = cull_degenerate_tris(&mut all_indices, &global_vertices);
    if deg_after_fill > 0 {
        log::debug!("[BRep mesh] removed {} degenerate tris after triangulation", deg_after_fill);
    }
```

After refine (before optimize, ~line 310):

```rust
    let deg_after_refine = cull_degenerate_tris(&mut all_indices, &global_vertices);
    if deg_after_refine > 0 {
        log::debug!("[BRep mesh] removed {} degenerate tris after refinement", deg_after_refine);
    }
```

### Step 3: Run tests

```bash
rtk cargo test -p rc3d-io mesh --lib
rtk cargo test -p rc3d-io --lib
```

### Step 4: Commit

```bash
git add crates/rc3d-io/src/step/brep/mesh/mod.rs
git commit -m "feat(rc3d-io): wire RemoveDegenerated after triangulation and after refine"
```

---

## Task 7: Fix config defaults

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/mesh/edge_disc.rs` (EdgeDiscConfig default)
- Modify: `crates/rc3d-io/src/step/brep/mesh/face_fill.rs` (FaceFillConfig default)
- Modify: `crates/rc3d-io/src/step/brep/mesh/refiner.rs` (RefineConfig default)
- Modify: `crates/rc3d-io/src/step/brep/mesh/mod.rs` (BRepMeshConfig default)
- Modify: `crates/rc3d-io/src/step/brep/heal/mod.rs` (HealConfig default)

### Step 1: Update each default impl

In `EdgeDiscConfig::default()`:
```rust
fn default() -> Self {
    Self { deflection: 0.01, angle_deflection: 0.1, min_points: 2, max_points: 256 }
}
```

In `FaceFillConfig::default()`:
```rust
fn default() -> Self {
    Self {
        enable_interior: true,
        deflection_interior: 0.01,
        min_size: 1e-3,
        min_size_relative: 0.01,
        max_adapt_iterations: 8,
    }
}
```

In `RefineConfig::default()`:
```rust
fn default() -> Self {
    Self {
        enable_post_refine: true,
        max_deflection: 0.01,
        max_iterations: 4,
        skip_refine_above: 512,
        max_tris: 8192,
        angular_deflection: 0.2,
    }
}
```

In `BRepMeshConfig::default()`:
```rust
fn default() -> Self {
    Self {
        edge: EdgeDiscConfig::default(),
        face: FaceFillConfig::default(),
        refine: RefineConfig::default(),
        optimize: OptimizeConfig::default(),
        relative_deflection: 0.005,
        same_parameter_tol: 1e-4,
    }
}
```

In `HealConfig::default()`:
```rust
fn default() -> Self {
    Self {
        gap_tolerance: 1e-4,
        fix_orientation: true,
        fix_reorder: true,
        fix_missing_seams: true,
        fix_vertex_tolerance: true,
        fix_small_area: true,
    }
}
```
Note: `run_check` field removed (Task 3).

### Step 2: Run tests

```bash
rtk cargo test -p rc3d-io --lib
```

### Step 3: Commit

```bash
git add crates/rc3d-io/src/step/brep/mesh/edge_disc.rs \
        crates/rc3d-io/src/step/brep/mesh/face_fill.rs \
        crates/rc3d-io/src/step/brep/mesh/refiner.rs \
        crates/rc3d-io/src/step/brep/mesh/mod.rs \
        crates/rc3d-io/src/step/brep/heal/mod.rs
git commit -m "fix(rc3d-io): correct config defaults — enable refine, relative deflection, fix checks"
```

---

## Task 8: Fix relative_deflection to use min()

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/mesh/report.rs`

### Step 1: Change `apply_relative_deflection`

Current (~line 101):
```rust
pub fn apply_relative_deflection(config: &mut BRepMeshConfig, shell_diag: f32) {
    if config.relative_deflection <= 0.0 || shell_diag <= 0.0 {
        return;
    }
    let rel = shell_diag * config.relative_deflection;
    config.edge.deflection = rel;
    config.face.deflection_interior = rel;
}
```

Replace with:
```rust
pub fn apply_relative_deflection(config: &mut BRepMeshConfig, shell_diag: f32) {
    if config.relative_deflection <= 0.0 || shell_diag <= 0.0 {
        return;
    }
    let rel = shell_diag * config.relative_deflection;
    // Use the tighter of explicit config vs relative — OCC Relative mode tightens
    // for small parts without loosening for large ones.
    config.edge.deflection = config.edge.deflection.min(rel);
    config.face.deflection_interior = config.face.deflection_interior.min(rel);
}
```

### Step 2: Run tests

```bash
rtk cargo test -p rc3d-io report --lib
```

### Step 3: Commit

```bash
git add crates/rc3d-io/src/step/brep/mesh/report.rs
git commit -m "fix(rc3d-io): relative_deflection uses min() to tighten for small parts"
```

**Phase A exit gate:** `rtk cargo test -p rc3d-io --lib` green. T2 tests pass with grid_fallback_rate < 10%.

---

# Phase B — Quality

## Task 9: Angular deflection in Steiner loop

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/mesh/refiner.rs` (add field)
- Modify: `crates/rc3d-io/src/step/brep/mesh/face_cdt.rs` (check angular in Steiner loop)
- Modify: `crates/rc3d-io/src/step/brep/mesh/face_fill.rs` (check angular in surface_fill_3d)

### Step 1: Add angular_deflection to RefineConfig

Already added in Task 7 defaults. The field `angular_deflection: f32` exists on `RefineConfig` (added in Task 7 step).

### Step 2: Add angular check in Steiner loop

In the Steiner insertion loop in `triangulate_uv_cdt_with_steiner` (Task 4), add angular deflection check alongside the linear check:

```rust
// Inside the deflection check for each edge (a, b):
// ... existing linear deflection check ...

// Angular deflection check
let na = face.surface.normal_native(uva.0, uva.1);
let nb = face.surface.normal_native(uvb.0, uvb.1);
let angle = (na.normalize().dot(nb.normalize())).acos().abs();
if angle > config.angular_deflection {
    splits.push((mid_uv.0 as f64, mid_uv.1 as f64, 0));
}
```

Same in `surface_fill_3d` Steiner loop.

### Step 3: Run tests

```bash
rtk cargo test -p rc3d-io face_cdt --lib
rtk cargo test -p rc3d-io face_fill --lib
rtk cargo test -p rc3d-io --lib
```

### Step 4: Commit

```bash
git add crates/rc3d-io/src/step/brep/mesh/refiner.rs \
        crates/rc3d-io/src/step/brep/mesh/face_cdt.rs \
        crates/rc3d-io/src/step/brep/mesh/face_fill.rs
git commit -m "feat(rc3d-io): angular deflection check in Steiner insertion loop"
```

---

## Task 10: fix_vertex_tolerance in heal_shell

**OCC reference:** `ShapeFix_Edge::FixVertexTolerance()`

```
// OCC ShapeFix_Edge::FixVertexTolerance algorithm:
//
// Standard_Boolean ShapeFix_Edge::FixVertexTolerance(
//   const TopoDS_Edge& edge,
//   const TopoDS_Face& face)
// {
//   TopoDS_Vertex Vf = TopExp::FirstVertex(edge, Standard_True);
//   TopoDS_Vertex Vl = TopExp::LastVertex(edge, Standard_True);
//
//   Standard_Real f, l;
//   Handle(Geom_Curve) C3D = BRep_Tool::Curve(edge, f, l);
//   if (C3D.IsNull()) return Standard_False;
//
//   // 1. Check gap at first vertex
//   gp_Pnt P_vertex_f = BRep_Tool::Pnt(Vf);
//   gp_Pnt P_curve_f  = C3D->Value(f);
//   Standard_Real gap_f = P_vertex_f.Distance(P_curve_f);
//
//   // 2. Check gap at last vertex
//   gp_Pnt P_vertex_l = BRep_Tool::Pnt(Vl);
//   gp_Pnt P_curve_l = C3D->Value(l);
//   Standard_Real gap_l = P_vertex_l.Distance(P_curve_l);
//
//   // 3. If gap exceeds current tolerance, extend tolerance
//   Standard_Real maxGap = Max(gap_f, gap_l);
//   Standard_Real currentTol = BRep_Tool::Tolerance(edge);
//
//   if (maxGap > currentTol) {
//     // Extend edge tolerance to cover the gap (+ 10% margin)
//     Standard_Real newTol = maxGap * 1.1;
//
//     // OCC also checks: is the gap within the face tolerance?
//     Standard_Real faceTol = BRep_Tool::Tolerance(face);
//     if (newTol > faceTol) {
//       // Gap exceeds face tolerance — this edge is problematic
//       // Still fix, but log warning
//     }
//
//     BRep_Builder B;
//     B.UpdateEdge(edge, newTol);
//     return Standard_True;
//   }
//   return Standard_False;
// }
```

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/heal/mod.rs`

### Step 1: Add fix_vertex_tolerance function

In `heal/mod.rs`, add:

```rust
fn fix_vertex_tolerance(reg: &mut BRepRegistry) -> usize {
    let mut fixed = 0usize;
    for (_, edge) in reg.edges.iter_mut() {
        let v_lo = reg.vertices.get(edge.v_low).map(|v| v.position);
        let v_hi = reg.vertices.get(edge.v_high).map(|v| v.position);
        if let (Some(p_lo), Some(p_hi)) = (v_lo, v_hi) {
            let curve_lo = edge.curve.d0(0.0);
            let curve_hi = edge.curve.d0(1.0);
            let gap_lo = (curve_lo - p_lo).length();
            let gap_hi = (curve_hi - p_hi).length();
            let max_gap = gap_lo.max(gap_hi);
            if max_gap > edge.tolerance {
                edge.tolerance = max_gap * 1.01; // 1% margin
                fixed += 1;
            }
        }
    }
    fixed
}
```

### Step 2: Call from heal_shell

In `heal_shell`, add before the orientation fix:

```rust
    if config.fix_vertex_tolerance {
        let fixed = fix_vertex_tolerance(reg);
        if fixed > 0 {
            log::debug!("[BRep heal] fixed vertex tolerance on {} edges", fixed);
        }
    }
```

### Step 3: Run tests

```bash
rtk cargo test -p rc3d-io heal --lib
```

### Step 4: Commit

```bash
git add crates/rc3d-io/src/step/brep/heal/mod.rs
git commit -m "feat(rc3d-io): fix_vertex_tolerance (ShapeFix_Edge equivalent)"
```

---

## Task 11: fix_small_area in heal_shell

**OCC reference:** `ShapeFix_Face::FixSmallArea()` / `BRepCheck_Face` area check

```
// OCC ShapeFix_Face area validation (simplified):
//
// Standard_Boolean ShapeFix_Face::FixSmallArea(const TopoDS_Face& face)
// {
//   // 1. Compute face bounding box via triangulation or surface bounds
//   Bnd_Box B;
//   BRepBndLib::Add(face, B);
//
//   // 2. Check if bounding box is void (no geometry)
//   if (B.IsVoid()) {
//     // Face has no geometric representation at all
//     return Standard_False;  // cannot fix → remove
//   }
//
//   // 3. Check if face is too small (bounding box diagonal < tolerance)
//   Standard_Real Xmin, Ymin, Zmin, Xmax, Ymax, Zmax;
//   B.Get(Xmin, Ymin, Zmin, Xmax, Ymax, Zmax);
//   Standard_Real diag = Sqrt(
//     (Xmax-Xmin)*(Xmax-Xmin) +
//     (Ymax-Ymin)*(Ymax-Ymin) +
//     (Zmax-Zmin)*(Zmax-Zmin)
//   );
//
//   Standard_Real faceTol = BRep_Tool::Tolerance(face);
//   // OCC uses 10× tolerance as threshold for "too small"
//   if (diag < 10.0 * faceTol) {
//     // Face is too small to mesh meaningfully
//     return Standard_False;  // skip → mark for removal
//   }
//
//   // 4. For our engine: also check wire edges + seam edges
//   TopoDS_Wire W = BRep_Tool::Wire(face);
//   if (W.IsNull() || BRep_Tool::IsEmpty(W)) {
//     // Empty wire AND no seam edges → zero-area face
//     return Standard_False;
//   }
//
//   return Standard_True;  // face is OK
// }
```

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/heal/mod.rs`

### Step 1: Add skip_face_keys to HealReport

```rust
pub struct HealReport {
    pub reordered_wires: usize,
    pub closed_gaps: usize,
    pub flipped_faces: usize,
    pub added_seams: usize,
    pub skip_face_keys: Vec<FaceKey>,  // NEW
}
```

### Step 2: Add fix_small_area function

```rust
fn fix_small_area(shell_key: ShellKey, reg: &BRepRegistry) -> Vec<FaceKey> {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return vec![],
    };
    let mut skip = Vec::new();
    for &(face_key, _) in &shell.faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        let wire = match reg.wires.get(face.outer_wire) {
            Some(w) => w,
            None => {
                skip.push(face_key);
                continue;
            }
        };
        // Zero area: empty wire with no seam edges
        if wire.edges.is_empty() && face.seam_edges.is_empty() {
            log::warn!("[BRep heal] face {:?} has zero area, marking for skip", face_key);
            skip.push(face_key);
        }
    }
    skip
}
```

### Step 3: Call from heal_shell

```rust
    if config.fix_small_area {
        report.skip_face_keys = fix_small_area(shell_key, reg);
    }
```

### Step 4: Wire skip_face_keys in mesh_brep_shell

In `mod.rs`, when building `face_infos`, skip faces in the skip list:

```rust
// After checking face exists (~line 146):
if skip_face_keys.contains(&face_key) {
    continue;
}
```

Add `skip_face_keys: &[FaceKey]` parameter to `mesh_brep_shell_with_report`. Update `step/mod.rs` caller to pass `&heal_report.skip_face_keys`.

### Step 5: Run tests

```bash
rtk cargo test -p rc3d-io heal --lib
rtk cargo test -p rc3d-io mesh --lib
rtk cargo test -p rc3d-io --lib
```

### Step 6: Commit

```bash
git add crates/rc3d-io/src/step/brep/heal/mod.rs \
        crates/rc3d-io/src/step/brep/mesh/mod.rs \
        crates/rc3d-io/src/step/mod.rs
git commit -m "feat(rc3d-io): fix_small_area detects zero-area faces; mesh skips them"
```

---

## Task 12: UvSource rename

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/mesh/face_uv.rs`
- Modify: `crates/rc3d-io/src/step/brep/mesh/mod.rs` (all references)
- Modify: `crates/rc3d-io/src/step/brep/mesh/report.rs` (all references)

### Step 1: Rename

In `face_uv.rs`:
```rust
pub enum UvSource {
    Pcurve,
    Synthetic,
    SurfaceFill,  // renamed from GridFallback
}
```

Grep for `GridFallback` across `mesh/` and replace all occurrences with `SurfaceFill`.

### Step 2: Run tests

```bash
rtk cargo test -p rc3d-io --lib
```

### Step 3: Commit

```bash
git add crates/rc3d-io/src/step/brep/mesh/
git commit -m "refactor(rc3d-io): rename UvSource::GridFallback → SurfaceFill"
```

---

# Phase C — Completeness

## Task 13: Assembly hierarchy

**Files:**
- Modify: `crates/rc3d-io/src/step/topology.rs` (parse NAUO)
- Modify: `crates/rc3d-io/src/step/assembly.rs` (build tree)
- Modify: `crates/rc3d-io/src/step/mod.rs` (emit tree)

### Step 1: Parse NAUO entities in topology.rs

Add to `collect_shells` or as a new function:

```rust
pub struct AssemblyNode {
    pub name: String,
    pub children: Vec<AssemblyNode>,
    pub shell_id: Option<u64>,
    pub transform: Option<Mat4>,
}

pub fn collect_assembly_tree(entities: &EntityIndex) -> Vec<AssemblyNode> {
    // Walk NAUO entities: NEXT_ASSEMBLY_USAGE_OCCURRENCE
    // Fields: (name, description, relating_product, related_product, reference_designator, transform)
    // Build parent→children map from relating→related references.
    // TODO during implementation: map exact STEP entity layout for NAUO params.
    todo!()
}
```

### Step 2: Build tree in assembly.rs

```rust
pub fn build_assembly_nodes(tree: &[AssemblyNode], shell_transforms: &HashMap<u64, Mat4>)
    -> Vec<AssemblyOutput> { ... }
```

### Step 3: Emit in step/mod.rs with Transform nodes

For each assembly node with a transform, emit `Separator → Transform → [Material → Shape]` instead of flat `Separator → Material → Shape`.

### Step 4: T3 integration test

```rust
#[test]
fn t3_assembly_has_transform_nodes() {
    let path = "test_data/AssemblyExample-Assembly.step";
    let graph = parse_step_file(path).unwrap();
    // Assert Transform nodes exist in the scene graph
}
```

### Step 5: Commit

```bash
git add crates/rc3d-io/src/step/topology.rs \
        crates/rc3d-io/src/step/assembly.rs \
        crates/rc3d-io/src/step/mod.rs
git commit -m "feat(rc3d-io): assembly hierarchy from NAUO entities with transforms"
```

---

## Task 14: Color/material transfer

**OCC reference:** `STEPCAFControl_Reader` XCAF color chain + `XCAFDoc_ColorTool`

```
// OCC STEPCAFControl_Reader color extraction (simplified):
//
// Standard_Boolean STEPCAFControl_Reader::TransferColors(
//   Handle(XCAFDoc_ColorTool) colorTool,
//   Handle(TDocStd_Document)   doc)
// {
//   // STEP entity chain for colors (AP214/AP242):
//   //
//   // STYLED_ITEM (#si)
//   //   → name: ''
//   //   → styles: (#presentation_style_assign, ...)
//   //   → item:   (#mapped_item | #shape_representation)
//   //
//   // PRESENTATION_STYLE_ASSIGNMENT (#psa)
//   //   → styles: (#surface_style_usage, ...)
//   //
//   // SURFACE_STYLE_USAGE (#ssu)
//   //   → side:   .POSITIVE. | .NEGATIVE. | .BOTH.
//   //   → style:  (#surface_side_style | #surface_style_rendering | ...)
//   //
//   // SURFACE_STYLE_FILL_AREA (#ssfa)
//   //   → fill_area: (#fill_area_style_colour)
//   //
//   // FILL_AREA_STYLE_COLOUR (#fasc)
//   //   → name: ''
//   //   → fill_colour: (#colour_rgb)
//   //
//   // COLOUR_RGB (#rgb)
//   //   → name: ''
//   //   → red:   0.0-1.0
//   //   → green: 0.0-1.0
//   //   → blue:  0.0-1.0
//   //
//   // SURFACE_STYLE_RENDERING (#ssr) — optional transparency
//   //   → rendering_properties: (#surface_rendering_properties)
//   //
//   // SURFACE_RENDERING_PROPERTIES (#srp) — optional
//   //   → transparency: 0.0-1.0  (0=opaque, 1=transparent)
//
//   // Mapping: face → color
//   // OCC walks STYLED_ITEM entities, matches the 'item' reference
//   // to the face's shape_representation, and extracts the color.
//   //
//   // For our engine:
//   //
//   // Pass 1: Build face_id → surface_id mapping (already done in topology.rs)
//   // Pass 2: Walk STYLED_ITEM entities:
//   //   for each STYLED_ITEM:
//   //     item_id = params[3]  // the geometric item
//   //     style_assign_ids = params[2]  // list of presentation style assignments
//   //     for each psa_id:
//   //       walk PRESENTATION_STYLE_ASSIGNMENT → SURFACE_STYLE_USAGE
//   //         → SURFACE_STYLE_FILL_AREA → FILL_AREA_STYLE_COLOUR → COLOUR_RGB
//   //       extract (R, G, B)
//   //       also check SURFACE_STYLE_RENDERING → SURFACE_RENDERING_PROPERTIES
//   //         → transparency
//   //     item_id → face_id → BRepFace.color = Some([R, G, B])
//   //
//   // The item_id in STYLED_ITEM can reference:
//   //   - A MAPPED_ITEM (which references a SHAPE_REPRESENTATION)
//   //   - A SHAPE_REPRESENTATION directly
//   //   - An ADVANCED_FACE directly
//   // We need to resolve the chain to find the face/surface the color applies to.
// }
```

**Files:**
- Modify: `crates/rc3d-io/src/step/topology.rs`
- Modify: `crates/rc3d-io/src/step/brep/topo.rs` (add color field to BRepFace)
- Modify: `crates/rc3d-io/src/step/brep/build.rs` (pass color through)
- Modify: `crates/rc3d-io/src/step/mod.rs` (use color in MaterialNode)

### Step 1: Add color field to BRepFace

In `topo.rs`:
```rust
pub struct BRepFace {
    // ... existing fields ...
    pub color: Option<[f32; 3]>,  // RGB from STYLED_ITEM
}
```

Update all `BRepFace { ... }` construction sites to include `color: None`.

### Step 2: Extract color from STYLED_ITEM chain

In `topology.rs`, add a function:
```rust
fn extract_face_colors(entities: &EntityIndex) -> HashMap<u64, [f32; 3]> {
    // Walk STYLED_ITEM → PRESENTATION_STYLE_ASSIGNMENT
    //   → SURFACE_STYLE_FILL_AREA → FILL_AREA_STYLE_COLOUR → COLOUR_RGB
    // Map face (surface) ID → RGB
    todo!()
}
```

### Step 3: Pass color through build_brep

In `build_brep`, after building each face, look up its color and set `face.color`.

### Step 4: Use color in step/mod.rs

```rust
let color = face.color.unwrap_or([0.9, 0.9, 0.9]);
let mat = MaterialNode {
    diffuse_color: Vec3::new(color[0], color[1], color[2]),
    base_color: Vec3::new(color[0], color[1], color[2]),
    roughness: 0.35, opacity: 1.0, ..Default::default()
};
```

### Step 5: Commit

```bash
git add crates/rc3d-io/src/step/topology.rs \
        crates/rc3d-io/src/step/brep/topo.rs \
        crates/rc3d-io/src/step/brep/build.rs \
        crates/rc3d-io/src/step/mod.rs
git commit -m "feat(rc3d-io): color/material transfer from STEP STYLED_ITEM entities"
```

---

## Task 15: T4 OCC reference gate

**Files:**
- Create: `crates/rc3d-io/src/step/brep/mesh/t4_quality.rs`
- Modify: `crates/rc3d-io/src/step/brep/mesh/mod.rs` (add `pub mod t4_quality;`)

### Step 1: Create t4_quality.rs

```rust
use crate::step::mesh_result::MeshResult;
use rc3d_core::math::Vec3;

/// Load an OCC-exported reference mesh in JSON format.
/// Format: {"vertices": [[x,y,z],...], "indices": [i0,i1,i2,-1,...]}
pub fn load_reference_mesh(json_path: &str) -> Option<MeshResult> {
    let text = std::fs::read_to_string(json_path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let vertices: Vec<Vec3> = v["vertices"].as_array()?.iter()
        .map(|arr| {
            let a = arr.as_array()?;
            Some(Vec3::new(
                a.get(0)?.as_f64()? as f32,
                a.get(1)?.as_f64()? as f32,
                a.get(2)?.as_f64()? as f32,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    let indices: Vec<i32> = v["indices"].as_array()?.iter()
        .filter_map(|v| v.as_i64().map(|n| n as i32))
        .collect();
    Some(MeshResult { vertices, indices, normals: vec![] })
}

/// Compute 95th-percentile Hausdorff distance from engine mesh to reference mesh.
/// Samples vertices from engine mesh, finds min distance to any reference triangle.
pub fn hausdorff_p95(engine: &MeshResult, reference: &MeshResult) -> f32 {
    let mut distances: Vec<f32> = engine.vertices.iter()
        .map(|v| point_to_mesh_distance(*v, reference))
        .collect();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (distances.len() as f32 * 0.95) as usize;
    *distances.get(idx).unwrap_or(&0.0)
}

fn point_to_mesh_distance(point: Vec3, mesh: &MeshResult) -> f32 {
    let mut min_dist = f32::MAX;
    for chunk in mesh.indices.chunks(4) {
        if chunk.len() < 4 { continue; }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
            continue;
        }
        let dist = point_to_triangle_distance(
            point, mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2],
        );
        min_dist = min_dist.min(dist);
    }
    min_dist
}

fn point_to_triangle_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    // Closest point on triangle to p, then distance.
    let ab = b - a; let ac = c - a; let ap = p - a;
    let d1 = ab.dot(ap); let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 { return (p - a).length(); }
    let bp = p - b;
    let d3 = ab.dot(bp); let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 { return (p - b).length(); }
    let cp = p - c;
    let d5 = ab.dot(cp); let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 { return (p - c).length(); }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (p - (a + ab * (d1 / (d1 - d3)))).length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (p - (a + ac * (d2 / (d2 - d6)))).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return (p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))))).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom; let w = vc * denom;
    (p - (a + ab * v + ac * w)).length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires OCC reference mesh in test_data/step/reference/"]
    fn t4_cube_hausdorff_within_deflection() {
        // Requires OCC-exported golden mesh for Cube.step
        let engine = mesh_step("test_data/step/analytic/Cube.step");
        let reference = load_reference_mesh("test_data/step/reference/Cube.json")
            .expect("reference mesh exists");
        let h = hausdorff_p95(&engine, &reference);
        assert!(h <= 0.10, "Hausdorff P95 {} exceeds deflection band", h);
    }
}
```

### Step 2: Add `pub mod t4_quality;` to mesh/mod.rs

### Step 3: Run

```bash
rtk cargo check -p rc3d-io
```

### Step 4: Commit

```bash
git add crates/rc3d-io/src/step/brep/mesh/t4_quality.rs \
        crates/rc3d-io/src/step/brep/mesh/mod.rs
git commit -m "test(rc3d-io): T4 OCC reference comparison harness (Hausdorff P95)"
```

---

## Task 16: AP242 entity name aliases

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/build.rs`

### Step 1: Add AP242 aliases

In `build_surface`, add alternative names:

```rust
"ELEMENTARY_SURFACE" | "SWEPT_SURFACE" => {
    // Supertype — should not appear directly. Log and skip.
    log::warn!("[BRep] encountered surface supertype '{}', skipping", record.name);
    None
}
```

In `build_curve`, add:
```rust
"BOUNDED_CURVE" => {
    let inner_id = geom::nth_ref(&record.params, 1)?;
    build_curve(inner_id, entities)
}
```

### Step 2: Test with any AP242 files in test_data

### Step 3: Commit

```bash
git add crates/rc3d-io/src/step/brep/build.rs
git commit -m "feat(rc3d-io): AP242 entity name aliases in surface/curve builders"
```

---

## Self-Review

**1. Spec coverage:** Every item in spec §6 (16 items) maps to a task. §4 config defaults → Task 7. §2.2 triangulation → Tasks 4-5. §2.1 heal pipeline → Tasks 1-3, 10-11. §2.3 assembly + materials → Tasks 13-14. §2.4 T4 → Task 15.

**2. Placeholder scan:** Two `todo!()` bodies remain — Task 13 Step 1 (assembly tree parsing) and Task 14 Step 2 (color chain extraction). These are P2 completeness items where the exact STEP entity layout needs validation against test files. Marked explicitly with "TODO during implementation: map exact STEP entity layout."

**3. Type consistency:** `AngularDeflection` field added to `RefineConfig` in Task 7. Referenced in Task 9 via `config.angular_deflection`. `skip_face_keys` added in Task 11, used in same task. `color: Option<[f32;3]>` added in Task 14, used in same task.

---

## Execution Handoff
