//! B-rep boolean operations: union, intersection, difference.
//!
//! Pipeline (OCC 4-phase):
//! 1. Face-face intersection → intersection curves
//! 2. Split faces along intersection curves
//! 3. Point-in-solid classification via ray casting
//! 4. Face selection per boolean operation type
//! 5. Stitch result into new B-rep shell

pub mod intersect;
pub mod classify;
pub mod split;
pub mod select;
pub mod stitch;

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::ShellKey;

/// Boolean operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    Union,        // A ∪ B
    Intersection, // A ∩ B
    Difference,   // A - B
}

/// Result of a boolean operation (legacy — kept for backward compatibility).
pub struct BoolResult {
    pub is_empty: bool,
}

/// Result of a B-Rep boolean operation.
pub struct BRepBoolResult {
    /// The resulting shell, if any.
    pub result_shell: Option<ShellKey>,
    /// Whether the result is empty (e.g., disjoint intersection).
    pub is_empty: bool,
    /// Number of face-face intersection curves found.
    pub intersection_count: usize,
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
    reg: &mut BRepRegistry,
    op: BoolOp,
) -> BRepBoolResult {
    if shells_a.is_empty() || shells_b.is_empty() {
        return BRepBoolResult { result_shell: None, is_empty: true, intersection_count: 0 };
    }

    // Phase 1: Face-face intersections
    let raw_intersections = intersect::compute_intersections_brep(shells_a, shells_b, reg);

    if raw_intersections.is_empty() {
        // No intersection — handle trivial cases
        return handle_no_intersection(shells_a, shells_b, reg, op);
    }

    // Convert to B-Rep intersection curves with UV parameters
    let curves = split::compute_brep_intersection_curves(&raw_intersections, reg);
    let intersection_count = curves.len();

    if curves.is_empty() {
        return handle_no_intersection(shells_a, shells_b, reg, op);
    }

    // Phase 2: Split faces along intersection curves
    let split_a = split::split_all_faces_brep(shells_a, &curves, reg);
    let split_b = split::split_all_faces_brep(shells_b, &curves, reg);

    // Phase 3: Classify each region against the other solid
    let regions_a = classify::classify_brep_regions(&split_a, shells_b[0], reg);
    let regions_b = classify::classify_brep_regions(&split_b, shells_a[0], reg);

    // Phase 4: Select faces based on operation type
    let selected = select::select_brep_faces(
        &regions_a, &regions_b, &split_a, &split_b, op,
    );

    if selected.is_empty() {
        return BRepBoolResult {
            result_shell: None,
            is_empty: true,
            intersection_count,
        };
    }

    // Phase 5: Stitch into new shell
    let shell_key = stitch::stitch_faces_into_shell(&selected, reg);

    BRepBoolResult {
        result_shell: shell_key,
        is_empty: shell_key.is_none(),
        intersection_count,
    }
}

/// Handle the no-intersection trivial case.
fn handle_no_intersection(
    shells_a: &[ShellKey],
    shells_b: &[ShellKey],
    reg: &mut BRepRegistry,
    op: BoolOp,
) -> BRepBoolResult {
    // Check containment: is A inside B or B inside A?
    // Use a sample point from each shell
    let a_inside_b = shells_a.first().map_or(false, |&sk| {
        shell_contains_point(sk, shells_b, reg)
    });
    let b_inside_a = shells_b.first().map_or(false, |&sk| {
        shell_contains_point(sk, shells_a, reg)
    });

    match op {
        BoolOp::Union => {
            if a_inside_b {
                // A inside B → result is B
                BRepBoolResult { result_shell: Some(shells_b[0]), is_empty: false, intersection_count: 0 }
            } else if b_inside_a {
                // B inside A → result is A
                BRepBoolResult { result_shell: Some(shells_a[0]), is_empty: false, intersection_count: 0 }
            } else {
                // Disjoint: both shells in result
                let mut all_faces = Vec::new();
                for &sk in shells_a.iter().chain(shells_b.iter()) {
                    if let Some(shell) = reg.shells.get(sk) {
                        all_faces.extend(shell.faces.iter().map(|&(fk, _)| fk));
                    }
                }
                let shell_key = stitch::stitch_faces_into_shell(&all_faces, reg);
                BRepBoolResult { result_shell: shell_key, is_empty: shell_key.is_none(), intersection_count: 0 }
            }
        }
        BoolOp::Intersection => {
            if a_inside_b {
                BRepBoolResult { result_shell: Some(shells_a[0]), is_empty: false, intersection_count: 0 }
            } else if b_inside_a {
                BRepBoolResult { result_shell: Some(shells_b[0]), is_empty: false, intersection_count: 0 }
            } else {
                // Disjoint → empty intersection
                BRepBoolResult { result_shell: None, is_empty: true, intersection_count: 0 }
            }
        }
        BoolOp::Difference => {
            if a_inside_b {
                // A entirely inside B → empty result
                BRepBoolResult { result_shell: None, is_empty: true, intersection_count: 0 }
            } else if b_inside_a {
                // B entirely inside A → A with B void (simplified: return A)
                BRepBoolResult { result_shell: Some(shells_a[0]), is_empty: false, intersection_count: 0 }
            } else {
                // Disjoint → A unchanged
                BRepBoolResult { result_shell: Some(shells_a[0]), is_empty: false, intersection_count: 0 }
            }
        }
    }
}

/// Check if any face center of `shell` is inside any of `other_shells`.
fn shell_contains_point(
    shell: ShellKey,
    other_shells: &[ShellKey],
    reg: &BRepRegistry,
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
    use crate::step::brep::topo::*;
    use crate::step::brep::geom::SurfaceGeom;
    use rc3d_core::math::Vec3;

    fn make_plane_shell(reg: &mut BRepRegistry, origin: Vec3, normal: Vec3) -> ShellKey {
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
        let result = BRepBoolResult { result_shell: None, is_empty: true, intersection_count: 0 };
        assert!(result.is_empty);
    }

    #[test]
    fn test_boolean_no_intersection_disjoint_union() {
        let mut reg = BRepRegistry::new();
        // Two disjoint plane shells
        let sa = make_plane_shell(&mut reg, Vec3::new(0.0, 0.0, 0.0), Vec3::Z);
        let sb = make_plane_shell(&mut reg, Vec3::new(100.0, 0.0, 0.0), Vec3::Z);

        let result = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Union);
        // Disjoint union should produce a combined shell
        assert!(result.result_shell.is_some(), "Disjoint union should produce a result");
    }

    #[test]
    fn test_boolean_no_intersection_disjoint_intersection() {
        let mut reg = BRepRegistry::new();
        let sa = make_plane_shell(&mut reg, Vec3::ZERO, Vec3::Z);
        let sb = make_plane_shell(&mut reg, Vec3::new(100.0, 0.0, 0.0), Vec3::Z);

        let result = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Intersection);
        assert!(result.is_empty, "Disjoint intersection should be empty");
    }

    #[test]
    fn test_boolean_no_intersection_disjoint_difference() {
        let mut reg = BRepRegistry::new();
        let sa = make_plane_shell(&mut reg, Vec3::ZERO, Vec3::Z);
        let sb = make_plane_shell(&mut reg, Vec3::new(100.0, 0.0, 0.0), Vec3::Z);

        let result = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Difference);
        assert!(!result.is_empty, "Disjoint difference should return A");
        assert_eq!(result.result_shell, Some(sa));
    }

    #[test]
    fn test_boolean_empty_shells() {
        let mut reg = BRepRegistry::new();
        let result = boolean_brep(&[], &[], &mut reg, BoolOp::Union);
        assert!(result.is_empty);
    }
}
