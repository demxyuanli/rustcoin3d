//! BRepOffsetAPI_MakeThickSolid equivalent — shelling (hollowing) a solid.
//!
//! Phase 4: Framework implementation. Full shelling requires:
//! 1. Offset all faces inward by thickness
//! 2. Remove specified open faces
//! 3. Boolean difference: outer - inner
//! 4. Stitch connecting walls
//!
//! Current status: API skeleton with input validation.
//! Full implementation requires robust NURBS offset support.

use crate::step::brep::registry::BRepStore;
use crate::step::brep::topo::{FaceKey, ShellKey, SolidKey};
use crate::step::bool::{boolean_brep, BoolOp};
use crate::step::brep::geom::SurfaceGeom;

/// Result of a shelling operation.
#[derive(Debug)]
pub struct ShellResult {
    /// The hollow shell (outer - inner + connecting walls).
    pub result_shell: Option<ShellKey>,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Diagnostic message.
    pub message: String,
    /// Number of faces in the offset solid.
    pub offset_face_count: usize,
    /// Number of intersections during boolean.
    pub intersection_count: usize,
}

/// Create a hollow shell by offsetting inward and subtracting.
///
/// # Arguments
/// * `solid_key` - The solid to hollow
/// * `open_faces` - Faces to remove (creating openings)
/// * `thickness` - Wall thickness (positive = offset inward)
/// * `reg` - B-Rep registry (mutated)
pub fn make_thick_solid(
    solid_key: SolidKey,
    _open_faces: &[FaceKey],
    thickness: f32,
    reg: &mut BRepStore,
) -> ShellResult {
    // Validate inputs
    if thickness <= 0.0 {
        return ShellResult {
            result_shell: None,
            success: false,
            message: format!("Thickness must be positive, got {}", thickness),
            offset_face_count: 0,
            intersection_count: 0,
        };
    }

    let solid = match reg.solids.get(solid_key) {
        Some(s) => s,
        None => return ShellResult {
            result_shell: None,
            success: false,
            message: format!("Solid {:?} not found", solid_key),
            offset_face_count: 0,
            intersection_count: 0,
        },
    };

    let outer_shell = solid.outer_shell;

    // Step 1: Create inward offset solid
    let offset_result = offset_solid_faces(outer_shell, -thickness, reg);

    if offset_result.is_empty() {
        return ShellResult {
            result_shell: None,
            success: false,
            message: "Failed to create offset solid — no faces could be offset".to_string(),
            offset_face_count: 0,
            intersection_count: 0,
        };
    }

    let offset_face_count = offset_result.len();

    // Create inner shell from offset faces
    let inner_shell = reg.shells.insert(crate::step::brep::topo::BRepShell {
        faces: offset_result.iter().map(|&fk| (fk, crate::step::brep::topo::Orientation::Forward)).collect(),
        closed: false,
        step_id: None,
    });

    // Step 2: Boolean difference outer - inner
    let bool_result = boolean_brep(
        &[outer_shell],
        &[inner_shell],
        reg,
        BoolOp::Difference,
    );

    match bool_result.result_shells.first().copied() {
        Some(sk) => ShellResult {
            result_shell: Some(sk),
            success: true,
            message: format!(
                "Shelling complete: {} offset faces, {} intersections",
                offset_face_count, bool_result.intersection_count
            ),
            offset_face_count,
            intersection_count: bool_result.intersection_count,
        },
        None => ShellResult {
            result_shell: None,
            success: false,
            message: "Boolean difference produced empty result".to_string(),
            offset_face_count,
            intersection_count: bool_result.intersection_count,
        },
    }
}

/// Offset a shell's faces to create offset copies.
/// Returns FaceKeys of the new offset faces.
///
/// Currently supports:
/// - Planar faces: offset origin along normal
/// - Cylindrical faces: offset radius
/// - Spherical faces: offset radius
///
/// Unsupported surface types are skipped.
fn offset_solid_faces(
    shell_key: ShellKey,
    distance: f32,
    reg: &mut BRepStore,
) -> Vec<FaceKey> {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s.clone(),
        None => return vec![],
    };

    let mut new_faces = Vec::new();

    for &(face_key, _orient) in &shell.faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };

        // Create offset surface
        let offset_surface = match &face.surface {
            SurfaceGeom::Plane { origin, normal, u_dir } => {
                Some(SurfaceGeom::Plane {
                    origin: *origin + *normal * distance,
                    normal: *normal,
                    u_dir: *u_dir,
                })
            }
            SurfaceGeom::Cylinder { origin, axis, radius, .. } => {
                let new_r = *radius + distance;
                if new_r > 0.0 {
                    Some(SurfaceGeom::cylinder(*origin, *axis, new_r))
                } else {
                    None // Would invert — skip
                }
            }
            SurfaceGeom::Sphere { center, radius } => {
                let new_r = *radius + distance;
                if new_r > 0.0 {
                    Some(SurfaceGeom::Sphere {
                        center: *center,
                        radius: new_r,
                    })
                } else {
                    None
                }
            }
            SurfaceGeom::Offset { basis, distance: existing_dist } => {
                Some(SurfaceGeom::Offset {
                    basis: basis.clone(),
                    distance: existing_dist + distance,
                })
            }
            _ => {
                // Wrap in Offset for other surface types
                Some(SurfaceGeom::Offset {
                    basis: Box::new(face.surface.clone()),
                    distance,
                })
            }
        };

        if let Some(surface) = offset_surface {
            let wire = reg.wires.insert(crate::step::brep::topo::BRepWire { edges: vec![] });
            let new_fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
                surface,
                outer_wire: wire,
                inner_wires: vec![],
                same_sense: face.same_sense,
                tolerance: face.tolerance,
                seam_edges: vec![],
                color: face.color,
                degenerated_edges: vec![],
            });
            new_faces.push(new_fk);
        }
    }

    new_faces
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::topo::*;
    use rc3d_core::math::Vec3;

    fn make_test_solid(reg: &mut BRepStore) -> SolidKey {
        // Create a simple solid with one planar face
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
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
        let shell = reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: true,
            step_id: None,
        });
        reg.solids.insert(BRepSolid {
            outer_shell: shell,
            void_shells: vec![],
        })
    }

    #[test]
    fn test_make_thick_solid_api() {
        let mut reg = BRepStore::new();
        let result = make_thick_solid(
            SolidKey::from(slotmap::KeyData::from_ffi(0xDEAD)),
            &[],
            1.0,
            &mut reg,
        );
        assert!(!result.success, "Should fail for non-existent solid");
        assert!(result.message.contains("not found"));
    }

    #[test]
    fn test_make_thick_solid_negative_thickness() {
        let mut reg = BRepStore::new();
        let solid = make_test_solid(&mut reg);
        let result = make_thick_solid(solid, &[], -1.0, &mut reg);
        assert!(!result.success);
        assert!(result.message.contains("positive"));
    }

    #[test]
    fn test_offset_plane_face() {
        let mut reg = BRepStore::new();
        let solid = make_test_solid(&mut reg);
        let shell = reg.solids.get(solid).unwrap().outer_shell;
        let offset_faces = offset_solid_faces(shell, -1.0, &mut reg);
        assert_eq!(offset_faces.len(), 1, "Should offset 1 planar face");

        // Verify the offset face has shifted origin
        let face = reg.faces.get(offset_faces[0]).unwrap();
        if let SurfaceGeom::Plane { origin, .. } = &face.surface {
            assert!((origin.z - (-1.0)).abs() < 1e-6,
                "Plane should be offset by -1.0 in Z, got z={}", origin.z);
        } else {
            panic!("Expected Plane surface");
        }
    }

    #[test]
    fn test_offset_cylinder_face() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 5.0);
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
        let shell = reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: true,
            step_id: None,
        });

        // Offset inward by 1.0 → radius should become 4.0
        let offset_faces = offset_solid_faces(shell, -1.0, &mut reg);
        assert_eq!(offset_faces.len(), 1);
        let face = reg.faces.get(offset_faces[0]).unwrap();
        if let SurfaceGeom::Cylinder { radius, .. } = &face.surface {
            assert!((radius - 4.0).abs() < 1e-6, "Radius should be 4.0, got {}", radius);
        }
    }

    #[test]
    fn test_offset_cylinder_inversion_skipped() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 0.5);
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
        let shell = reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: true,
            step_id: None,
        });

        // Offset inward by 1.0 → would invert radius to -0.5, should be skipped
        let offset_faces = offset_solid_faces(shell, -1.0, &mut reg);
        assert_eq!(offset_faces.len(), 0, "Should skip inverted cylinder");
    }
}
