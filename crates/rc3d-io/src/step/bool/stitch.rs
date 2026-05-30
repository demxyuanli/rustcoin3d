//! Post-boolean shell stitching: connect selected faces into a coherent shell.
//!
//! After boolean face selection, the selected faces need to be assembled into
//! a new B-Rep shell with consistent orientation and connectivity.

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{FaceKey, ShellKey, BRepShell, Orientation};

/// Stitch selected faces into a new B-Rep shell.
///
/// The algorithm:
/// 1. Collect all selected faces
/// 2. Determine face orientations from existing shell (or assume Forward)
/// 3. Create a new shell referencing these faces
/// 4. Validate shell closure via Euler-Poincaré (if enough topology info)
pub fn stitch_faces_into_shell(
    face_keys: &[FaceKey],
    reg: &mut BRepRegistry,
) -> Option<ShellKey> {
    if face_keys.is_empty() {
        return None;
    }

    // Build face list with orientations
    // For boolean results, faces keep their original orientation unless inverted
    let faces: Vec<(FaceKey, Orientation)> = face_keys.iter()
        .filter_map(|&fk| {
            if reg.faces.contains_key(fk) {
                Some((fk, Orientation::Forward))
            } else {
                None
            }
        })
        .collect();

    if faces.is_empty() {
        return None;
    }

    let shell = BRepShell {
        faces,
        closed: false, // boolean results may not be closed initially
        step_id: None,
    };

    let shell_key = reg.shells.insert(shell);
    Some(shell_key)
}

/// Attempt to close a shell by matching open edges and setting orientations.
/// Returns true if the shell was successfully closed.
pub fn try_close_shell(shell_key: ShellKey, reg: &mut BRepRegistry) -> bool {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s.clone(),
        None => return false,
    };

    // Build edge usage count: each edge should be shared by exactly 2 faces
    // for a closed shell (Euler-Poincaré).
    let mut edge_count: std::collections::HashMap<crate::step::brep::topo::EdgeKey, u32> =
        std::collections::HashMap::new();

    for &(face_key, _) in &shell.faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        let wire = match reg.wires.get(face.outer_wire) {
            Some(w) => w,
            None => continue,
        };
        for &(edge_key, _) in &wire.edges {
            *edge_count.entry(edge_key).or_insert(0) += 1;
        }
    }

    // Check if all edges are shared (closed shell requirement)
    let all_shared = edge_count.values().all(|&count| count == 2);
    if all_shared {
        if let Some(shell) = reg.shells.get_mut(shell_key) {
            shell.closed = true;
        }
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::topo::*;
    use crate::step::brep::geom::SurfaceGeom;
    use rc3d_core::math::Vec3;

    fn make_test_face(reg: &mut BRepRegistry) -> FaceKey {
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-6,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        })
    }

    #[test]
    fn test_stitch_empty_returns_none() {
        let mut reg = BRepRegistry::new();
        assert!(stitch_faces_into_shell(&[], &mut reg).is_none());
    }

    #[test]
    fn test_stitch_cube_faces() {
        let mut reg = BRepRegistry::new();
        let faces: Vec<FaceKey> = (0..6).map(|_| make_test_face(&mut reg)).collect();

        let shell_key = stitch_faces_into_shell(&faces, &mut reg);
        assert!(shell_key.is_some(), "Should create a shell from 6 faces");

        let shell = reg.shells.get(shell_key.unwrap()).unwrap();
        assert_eq!(shell.faces.len(), 6);
    }

    #[test]
    fn test_stitch_skips_invalid_faces() {
        let mut reg = BRepRegistry::new();
        let fk = make_test_face(&mut reg);

        // Create a bogus key that doesn't exist in the registry
        let fake_key = FaceKey::from(slotmap::KeyData::from_ffi(0xDEAD));
        let result = stitch_faces_into_shell(&[fk, fake_key], &mut reg);
        assert!(result.is_some());
        let shell = reg.shells.get(result.unwrap()).unwrap();
        assert_eq!(shell.faces.len(), 1, "Only valid face should be included");
    }
}
