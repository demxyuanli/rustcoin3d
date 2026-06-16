//! B-rep boolean operations: union, intersection, difference.
//!
//! Pipeline (OCC 4-phase):
//! 1. Face-face intersection → intersection curves
//! 2. Split faces along intersection curves
//! 3. Point-in-solid classification via ray casting
//! 4. Face selection per boolean operation type
//! 5. Stitch result into new B-rep shell

pub mod aabb;
pub mod intersect;
pub mod intersect_edge;
pub mod classify;
pub mod split;
pub mod select;
pub(crate) mod stitch;
pub mod marching;
pub mod ssi_newton;
pub mod bopds;
pub mod face_intersector;
pub mod pave_filler;
pub mod builder_face;
pub mod coplanar;
pub mod section;
pub use intersect::{FaceIntersectionResult, faces_are_coplanar, is_tangent_intersection};
pub use marching::{SeedPoint, find_seeds, trace_curve};

use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, ShellKey, VertexKey};

/// Boolean operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoolOp {
    Union,        // A ∪ B
    #[default]
    Intersection, // A ∩ B
    Difference,   // A - B
}

/// History of a boolean operation: maps result shapes back to source shapes.
///
/// OCC alignment: BOPAlgo_Builder history tracking (BOPAlgo_Glue).
/// Records which result faces came from which source faces, which edges were
/// split, and which new vertices were created.
#[derive(Debug, Clone, Default)]
pub struct BRepBoolHistory {
    /// Map: result face → source face(s) that contributed to it
    pub result_to_source: std::collections::HashMap<FaceKey, Vec<FaceKey>>,
    /// Edges that were split during the operation (original → [split edges])
    pub split_edges: std::collections::HashMap<EdgeKey, Vec<EdgeKey>>,
    /// New vertices created at split points
    pub new_vertices: Vec<VertexKey>,
    /// Number of face-face intersections found
    pub intersection_count: usize,
    /// The operation type
    pub operation: BoolOp,
}

impl BRepBoolHistory {
    pub fn new(op: BoolOp) -> Self {
        Self {
            result_to_source: std::collections::HashMap::new(),
            split_edges: std::collections::HashMap::new(),
            new_vertices: Vec::new(),
            intersection_count: 0,
            operation: op,
        }
    }
}

/// Result of a boolean operation (legacy — kept for backward compatibility).
pub struct BoolResult {
    pub is_empty: bool,
}

/// Result of a B-Rep boolean operation.
pub struct BRepBoolResult {
    /// Result shells (multiple for disjoint union components).
    pub result_shells: Vec<ShellKey>,
    /// Whether the result is empty (e.g., disjoint intersection).
    pub is_empty: bool,
    /// Number of face-face intersection curves found.
    pub intersection_count: usize,
    /// Tolerance propagated from input face tolerances.
    pub tolerance: f32,
    /// History tracking: maps result shapes to source shapes.
    /// OCC alignment: BOPAlgo_Builder history.
    pub history: Option<BRepBoolHistory>,
}

/// Perform a boolean operation on B-Rep shells.
///
/// Full OCC-style pipeline:
/// 1. Compute face-face intersections (analytic surfaces)
/// 2. Split faces along intersection curves
/// 3. Classify each split region via ray casting
/// 4. Select faces per operation type
/// 5. Stitch into new shell
pub fn boolean_brep(
    shells_a: &[ShellKey],
    shells_b: &[ShellKey],
    reg: &mut BRepStore,
    op: BoolOp,
) -> BRepBoolResult {
    let tolerance = compute_face_tolerance(shells_a.iter().chain(shells_b.iter()), reg) + 1e-6;

    if shells_a.is_empty() || shells_b.is_empty() {
        return BRepBoolResult { history: None, result_shells: vec![], is_empty: true, intersection_count: 0, tolerance };
    }

    // Phase 1: Face-face intersections via PaveFiller (OCC BOPAlgo_PaveFiller)
    // Uses marching+Newton for general surfaces, analytic for simple pairs.
    let (bopds, _pave_report) = pave_filler::fill_paves(shells_a, shells_b, reg, tolerance);

    if bopds.face_face_interfs.is_empty() {
        return handle_no_intersection(shells_a, shells_b, reg, op, tolerance);
    }

    // Convert BOPDS face-face interfs to legacy intersection curves
    let raw_intersections: Vec<FaceIntersectionResult> = bopds.face_face_interfs.iter().map(|interf| {
        FaceIntersectionResult {
            face_a: interf.face_a,
            face_b: interf.face_b,
            curves_3d: interf.curves_3d.clone(),
            pcurves_on_a: interf.pcurves_a.clone(),
            pcurves_on_b: interf.pcurves_b.clone(),
        }
    }).collect();

    let curves = split::compute_brep_intersection_curves(&raw_intersections, reg);
    // pave_report.total_curves sums curves_3d.len() across FaceFaceInterf entries;
    // compute_brep_intersection_curves processes those same entries into BRepIntersectionCurves.
    // Adding both would double-count.
    let intersection_count = curves.len();

    if curves.is_empty() {
        return handle_no_intersection(shells_a, shells_b, reg, op, tolerance);
    }

    // Phase 2: Split faces using BopDS PaveBlock data (OCC BOPAlgo_BuilderFace)
    // Replaces legacy split_all_faces_brep UV geometric marching
    let split_a_uv = if bopds.face_face_interfs.is_empty() {
        split::split_all_faces_brep(shells_a, &curves, reg)
    } else {
        split::split_faces_from_bopds(shells_a, &bopds, reg)
    };
    let split_b_uv = if bopds.face_face_interfs.is_empty() {
        split::split_all_faces_brep(shells_b, &curves, reg)
    } else {
        split::split_faces_from_bopds(shells_b, &bopds, reg)
    };

    // Phase 2b: Build BRep faces from UV split regions (OCC BOPAlgo_BuilderFace)
    // Must run BEFORE classification since it needs &mut reg
    let mut new_faces_a: Vec<crate::topo::FaceKey> = Vec::new();
    for sfr in &split_a_uv {
        let result = builder_face::build_faces_from_split(
            sfr.original_face, &sfr.sub_faces, &curves, reg, Some(&bopds),
        );
        new_faces_a.extend(result.new_faces);
    }
    let mut new_faces_b: Vec<crate::topo::FaceKey> = Vec::new();
    for sfr in &split_b_uv {
        let result = builder_face::build_faces_from_split(
            sfr.original_face, &sfr.sub_faces, &curves, reg, Some(&bopds),
        );
        new_faces_b.extend(result.new_faces);
    }

    // Phase 3: Classify each region against the other solid(s)
    let regions_a = classify::classify_brep_regions(&split_a_uv, shells_b, reg);
    let regions_b = classify::classify_brep_regions(&split_b_uv, shells_a, reg);

    // Phase 4: Select faces based on operation type
    let selected = select::select_brep_faces(
        &regions_a, &regions_b, &split_a_uv, &split_b_uv, op, reg,
    );

    // Append newly built BRep faces to the selected set
    let mut all_selected = selected.clone();
    all_selected.extend(new_faces_a);
    all_selected.extend(new_faces_b);

    if all_selected.is_empty() {
        return BRepBoolResult {
            history: None,
            result_shells: vec![],
            is_empty: true,
            intersection_count,
            tolerance,
        };
    }

    // Phase 5: Stitch into new shell
    let shell_key = stitch::stitch_faces_into_shell(&all_selected, reg);
    let result_shells: Vec<ShellKey> = shell_key.into_iter().collect();
    let is_empty = result_shells.is_empty();

    BRepBoolResult {
        history: None,
        result_shells,
        is_empty,
        intersection_count,
        tolerance,
    }
}

/// Compute tolerance from face tolerances in the given shells.
fn compute_face_tolerance<'a>(shells: impl Iterator<Item = &'a ShellKey>, reg: &BRepStore) -> f32 {
    shells
        .filter_map(|&sk| reg.shells.get(sk))
        .flat_map(|s| s.faces.iter())
        .filter_map(|&(fk, _)| reg.faces.get(fk))
        .map(|f| f.tolerance)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1e-4)
}

/// Handle the no-intersection trivial case.
fn handle_no_intersection(
    shells_a: &[ShellKey],
    shells_b: &[ShellKey],
    reg: &mut BRepStore,
    op: BoolOp,
    tolerance: f32,
) -> BRepBoolResult {
    // Try coplanar face boolean (2D polygon clipping in UV space)
    if let Some(result) = coplanar::handle_coplanar_boolean(
        shells_a, shells_b, reg, op, tolerance,
    ) {
        return result;
    }

    // Check containment: is A inside B or B inside A?
    // Use a sample point from each shell
    let a_inside_b = shells_a.first().is_some_and(|&sk| {
        shell_contains_point(sk, shells_b, reg)
    });
    let b_inside_a = shells_b.first().is_some_and(|&sk| {
        shell_contains_point(sk, shells_a, reg)
    });

    match op {
        BoolOp::Union => {
            if a_inside_b {
                // A inside B → result is B
                BRepBoolResult { history: None, result_shells: vec![shells_b[0]], is_empty: false, intersection_count: 0, tolerance }
            } else if b_inside_a {
                // B inside A → result is A
                BRepBoolResult { history: None, result_shells: vec![shells_a[0]], is_empty: false, intersection_count: 0, tolerance }
            } else {
                // Disjoint: both shells in result
                let mut all_faces = Vec::new();
                for &sk in shells_a.iter().chain(shells_b.iter()) {
                    if let Some(shell) = reg.shells.get(sk) {
                        all_faces.extend(shell.faces.iter().map(|&(fk, _)| fk));
                    }
                }
                let shell_key = stitch::stitch_faces_into_shell(&all_faces, reg);
                let result_shells: Vec<ShellKey> = shell_key.into_iter().collect();
                let is_empty = result_shells.is_empty();
                BRepBoolResult { history: None, result_shells, is_empty, intersection_count: 0, tolerance }
            }
        }
        BoolOp::Intersection => {
            if a_inside_b {
                BRepBoolResult { history: None, result_shells: vec![shells_a[0]], is_empty: false, intersection_count: 0, tolerance }
            } else if b_inside_a {
                BRepBoolResult { history: None, result_shells: vec![shells_b[0]], is_empty: false, intersection_count: 0, tolerance }
            } else {
                // Disjoint → empty intersection
                BRepBoolResult { history: None, result_shells: vec![], is_empty: true, intersection_count: 0, tolerance }
            }
        }
        BoolOp::Difference => {
            if a_inside_b {
                // A entirely inside B → empty result
                BRepBoolResult { history: None, result_shells: vec![], is_empty: true, intersection_count: 0, tolerance }
            } else if b_inside_a {
                // B entirely inside A → A with B void (simplified: return A)
                BRepBoolResult { history: None, result_shells: vec![shells_a[0]], is_empty: false, intersection_count: 0, tolerance }
            } else {
                // Disjoint → A unchanged
                BRepBoolResult { history: None, result_shells: vec![shells_a[0]], is_empty: false, intersection_count: 0, tolerance }
            }
        }
    }
}

/// Check if any face center of `shell` is inside any of `other_shells`.
fn shell_contains_point(
    shell: ShellKey,
    other_shells: &[ShellKey],
    reg: &BRepStore,
) -> bool {
    let sample_point = reg.shells.get(shell)
        .and_then(|s| s.faces.first())
        .and_then(|&(fk, _)| reg.faces.get(fk))
        .map(|f| {
            let range = f.surface.param_range();
            let (un, vn) = f.surface.native_uv_to_d0(
                (range.u_min + range.u_max) * 0.5,
                (range.v_min + range.v_max) * 0.5,
            );
            f.surface.d0(un, vn)
        });

    if let Some(pt) = sample_point {
        for &other_sk in other_shells {
            let class = classify::classify_point_solid(pt, other_sk, reg, 1e-4);
            if class == classify::PointClassification::Inside {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::*;
    use crate::geom::SurfaceGeom;
    use crate::geom::curve2d::Curve2d;
    use rc3d_core::math::Vec3;

    fn make_plane_shell(reg: &mut BRepStore, origin: Vec3, normal: Vec3) -> ShellKey {
        let surface = SurfaceGeom::Plane { origin, normal, u_dir: Vec3::X };
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-6,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        })
    }

    #[test]
    fn test_bool_module_loads() {
        let op = BoolOp::Union;
        assert_eq!(op, BoolOp::Union);
        let result = BRepBoolResult { history: None, result_shells: vec![], is_empty: true, intersection_count: 0, tolerance: 1e-4 };
        assert!(result.is_empty);
    }

    #[test]
    fn test_boolean_no_intersection_disjoint_union() {
        let mut reg = BRepStore::new();
        // Two disjoint plane shells
        let sa = make_plane_shell(&mut reg, Vec3::new(0.0, 0.0, 0.0), Vec3::Z);
        let sb = make_plane_shell(&mut reg, Vec3::new(100.0, 0.0, 0.0), Vec3::Z);

        let result = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Union);
        // Disjoint union should produce a combined shell
        assert!(!result.result_shells.is_empty(), "Disjoint union should produce a result");
    }

    #[test]
    fn test_boolean_no_intersection_disjoint_intersection() {
        let mut reg = BRepStore::new();
        let sa = make_plane_shell(&mut reg, Vec3::ZERO, Vec3::Z);
        let sb = make_plane_shell(&mut reg, Vec3::new(100.0, 0.0, 0.0), Vec3::Z);

        let result = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Intersection);
        assert!(result.is_empty, "Disjoint intersection should be empty");
    }

    #[test]
    fn test_boolean_no_intersection_disjoint_difference() {
        let mut reg = BRepStore::new();
        let sa = make_plane_shell(&mut reg, Vec3::ZERO, Vec3::Z);
        let sb = make_plane_shell(&mut reg, Vec3::new(100.0, 0.0, 0.0), Vec3::Z);

        let result = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Difference);
        assert!(!result.is_empty, "Disjoint difference should return A");
        assert_eq!(result.result_shells, vec![sa]);
    }

    #[test]
    fn test_boolean_empty_shells() {
        let mut reg = BRepStore::new();
        let result = boolean_brep(&[], &[], &mut reg, BoolOp::Union);
        assert!(result.is_empty);
    }

    /// Integration test: two perpendicular planes intersecting along the Y axis.
    ///
    /// Plane Z=0 (normal Z) x Plane X=0 (normal X) → intersection line along Y.
    /// Verifies the full boolean pipeline: intersect → split → classify → select → stitch.
    ///
    /// Note: With single-face shells, ray-casting classifies both sides as "Outside"
    /// due to majority voting (1 of 6 rays hits). Therefore Intersection (keep Inside)
    /// is empty, but Union (keep Outside) and Difference produce results.
    #[test]
    fn test_bool_intersection_two_planes() {
        let mut reg = BRepStore::new();

        // Shell A: face on Plane Z=0
        let sa = make_plane_shell(&mut reg, Vec3::ZERO, Vec3::Z);

        // Shell B: face on Plane X=0, rotated 90 degrees
        let sb = make_plane_shell(&mut reg, Vec3::ZERO, Vec3::X);

        // Phase 1: intersection curves are found
        let raw = intersect::compute_intersections_brep(&[sa], &[sb], &reg);
        assert_eq!(raw.len(), 1, "Should find 1 face-face intersection");

        // Phase 2a: B-Rep curves computed
        let curves = split::compute_brep_intersection_curves(&raw, &reg);
        assert_eq!(curves.len(), 1, "Should produce 1 B-Rep curve");

        // Phase 2b: faces are split
        let split_a = split::split_all_faces_brep(&[sa], &curves, &reg);
        let split_b = split::split_all_faces_brep(&[sb], &curves, &reg);
        assert_eq!(split_a.len(), 1, "Shell A has 1 face → 1 split region");
        assert_eq!(split_b.len(), 1, "Shell B has 1 face → 1 split region");
        // Bug 2 verification: each face should produce 2 sub-faces (both sides of the curve)
        assert_eq!(split_a[0].sub_faces.len(), 2, "Face A split into 2 sub-faces");
        assert_eq!(split_b[0].sub_faces.len(), 2, "Face B split into 2 sub-faces");

        // Phase 3: classify — Bug 3 verification: ALL sub-faces are classified
        let regions_a = classify::classify_brep_regions(&split_a, &[sb], &reg);
        let regions_b = classify::classify_brep_regions(&split_b, &[sa], &reg);
        assert_eq!(regions_a[0].1.len(), 2, "All 2 sub-faces of A classified");
        assert_eq!(regions_b[0].1.len(), 2, "All 2 sub-faces of B classified");

        // Phase 4+5: Union (keep Outside) — Bug 1 verification: sub-faces created
        let result_union = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Union);
        assert!(!result_union.is_empty,
            "Union should produce non-empty result (Outside sub-faces selected)");
        assert!(result_union.intersection_count > 0,
            "Should report at least 1 intersection curve, got {}",
            result_union.intersection_count);

        // Difference A-B: keep Outside from A, Inside from B
        let result_diff = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Difference);
        assert!(!result_diff.is_empty,
            "Difference should produce non-empty result");
    }

    /// Build a square face with 4 boundary edges in the XY plane.
    fn make_square_face(
        reg: &mut BRepStore,
        origin: Vec3,
        size: f32,
    ) -> (ShellKey, FaceKey) {
        use crate::geom::CurveGeom;
        let half = size * 0.5;
        let o = origin;
        let v0 = reg.find_or_add_vertex(Vec3::new(o.x - half, o.y - half, o.z), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(o.x + half, o.y - half, o.z), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(o.x + half, o.y + half, o.z), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(o.x - half, o.y + half, o.z), 1e-4);

        let surface = SurfaceGeom::Plane {
            origin: Vec3::new(o.x, o.y, o.z),
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        // Build 4 edges
        let edges_data = [
            (v0, v1, Vec3::new(o.x - half, o.y - half, o.z), Vec3::new(o.x + half, o.y - half, o.z)),
            (v1, v2, Vec3::new(o.x + half, o.y - half, o.z), Vec3::new(o.x + half, o.y + half, o.z)),
            (v2, v3, Vec3::new(o.x + half, o.y + half, o.z), Vec3::new(o.x - half, o.y + half, o.z)),
            (v3, v0, Vec3::new(o.x - half, o.y + half, o.z), Vec3::new(o.x - half, o.y - half, o.z)),
        ];
        let mut wire_edges = Vec::new();
        for (va, vb, pa, pb) in edges_data {
            let curve = CurveGeom::Line { origin: pa, direction: pb - pa };
            let pc = Curve2d::Line { origin: (pa.x, pa.y), direction: (pb.x - pa.x, pb.y - pa.y) };
            let ek = reg.add_edge_with_pcurve(va, vb, curve.clone(), 1e-4, fk, pc, true);
            wire_edges.push((ek, Orientation::Forward));
        }
        reg.wires.get_mut(wk).unwrap().edges = wire_edges;

        let sk = reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        (sk, fk)
    }

    #[test]
    fn test_boolean_with_proper_edges() {
        let mut reg = BRepStore::new();

        // Two overlapping squares in the XY plane
        let (sa, _fa) = make_square_face(&mut reg, Vec3::new(0.0, 0.0, 0.0), 2.0);
        let (sb, _fb) = make_square_face(&mut reg, Vec3::new(1.0, 0.0, 0.0), 2.0);

        // Verify the faces have proper edges
        let shell_a = reg.shells.get(sa).unwrap();
        let face_a = reg.faces.get(shell_a.faces[0].0).unwrap();
        let wire_a = reg.wires.get(face_a.outer_wire).unwrap();
        assert_eq!(wire_a.edges.len(), 4, "square face should have 4 edges");

        let shell_b = reg.shells.get(sb).unwrap();
        let face_b = reg.faces.get(shell_b.faces[0].0).unwrap();
        let wire_b = reg.wires.get(face_b.outer_wire).unwrap();
        assert_eq!(wire_b.edges.len(), 4, "square face should have 4 edges");

        // Run boolean union on coplanar squares.
        // Note: coplanar faces cannot produce 1D intersection curves via
        // surface-surface intersection (they overlap in a 2D region).
        // Union correctly handles this via the disjoint path — both faces
        // are gathered and stitched into the result shell.
        let result = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Union);
        assert!(!result.result_shells.is_empty(),
            "Union of overlapping squares should produce a result shell");
    }

    /// Coplanar face intersection via 2D polygon clipping (UV space).
    #[test]
    fn test_coplanar_intersection_overlap() {
        let mut reg = BRepStore::new();
        let (sa, _fa) = make_square_face(&mut reg, Vec3::new(0.0, 0.0, 0.0), 2.0);
        let (sb, _fb) = make_square_face(&mut reg, Vec3::new(1.0, 0.0, 0.0), 2.0);

        let result = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Intersection);
        // Coplanar overlapping squares should produce a non-empty intersection
        // via 2D polygon clipping in UV space.
        assert!(!result.is_empty || !result.result_shells.is_empty(),
            "Coplanar intersection of overlapping squares should produce result");
    }
}
