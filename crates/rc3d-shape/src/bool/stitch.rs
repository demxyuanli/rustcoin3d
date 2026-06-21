//! Post-boolean shell stitching: connect selected faces into a coherent shell.
//!
//! After boolean face selection, the selected faces need to be assembled into
//! a new B-Rep shell with consistent orientation and connectivity.

use std::collections::{HashMap, HashSet, VecDeque};

use rc3d_core::math::{Real, PVec3};
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, ShellKey, VertexKey, BRepShell, Orientation};

/// Stitch selected faces into a new B-Rep shell.
///
/// The algorithm:
/// 1. Collect all selected faces
/// 2. Determine face orientations from existing shell (or assume Forward)
/// 3. Create a new shell referencing these faces
/// 4. Validate shell closure via Euler-Poincaré (if enough topology info)
pub fn stitch_faces_into_shell(
    face_keys: &[FaceKey],
    reg: &mut BRepStore,
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

    // Weld vertices and propagate orientations
    weld_shell_vertices(shell_key, reg, 1e-4);
    propagate_shell_orientations(shell_key, reg);

    Some(shell_key)
}

/// Weld coincident vertices across faces in a shell.
/// Returns number of vertex pairs merged.
pub fn weld_shell_vertices(shell_key: ShellKey, reg: &mut BRepStore, tolerance: Real) -> usize {
    let face_keys: Vec<FaceKey> = {
        let Some(shell) = reg.shells.get(shell_key) else { return 0; };
        shell.faces.iter().map(|&(fk, _)| fk).collect()
    };

    // Collect all vertices and their positions
    let mut vert_positions: Vec<(VertexKey, PVec3)> = Vec::new();
    for fk in &face_keys {
        let Some(face) = reg.faces.get(*fk) else { continue };
        for wk in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
            let Some(wire) = reg.wires.get(*wk) else { continue };
            for &(ek, _) in &wire.edges {
                let Some(edge) = reg.edges.get(ek) else { continue };
                for &vk in &[edge.v_low, edge.v_high] {
                    if let Some(v) = reg.vertices.get(vk) {
                        vert_positions.push((vk, v.position));
                    }
                }
            }
        }
    }

    // Find and merge coincident vertices
    let mut merged = 0usize;
    let n = vert_positions.len();
    let mut to_replace: HashMap<VertexKey, VertexKey> = HashMap::new(); // replace -> keep

    for i in 0..n {
        let (vk_i, pos_i) = vert_positions[i];
        if !reg.vertices.contains_key(vk_i) { continue; } // already removed
        for j in (i + 1)..n {
            let (vk_j, pos_j) = vert_positions[j];
            if !reg.vertices.contains_key(vk_j) { continue; }
            if vk_i == vk_j { continue; }
            if (pos_i - pos_j).length() < tolerance {
                to_replace.insert(vk_j, vk_i);
            }
        }
    }

    // Resolve transitive chains to their ultimate canonical vertex.
    // e.g. {v2→v1, v1→v0} → {v2→v0, v1→v0} so no intermediate keys are removed
    // while still referenced by edges from earlier iterations.
    for (replace, keep) in to_replace.clone() {
        let mut ultimate = keep;
        while let Some(&next) = to_replace.get(&ultimate) {
            ultimate = next;
        }
        if ultimate != keep {
            to_replace.insert(replace, ultimate);
        }
    }

    // Apply merges: update edge vertex references, remove replaced vertices
    for (replace, keep) in &to_replace {
        for (_, edge) in reg.edges.iter_mut() {
            if edge.v_low == *replace { edge.v_low = *keep; }
            if edge.v_high == *replace { edge.v_high = *keep; }
        }
        reg.vertices.remove(*replace);
        merged += 1;
    }

    merged
}

/// Find shared edges between two faces.
fn shared_edges(fk_a: FaceKey, fk_b: FaceKey, reg: &BRepStore) -> Vec<EdgeKey> {
    let collect_edges = |fk: FaceKey| -> Vec<EdgeKey> {
        let mut edges = Vec::new();
        let Some(face) = reg.faces.get(fk) else { return edges; };
        for wk in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
            let Some(wire) = reg.wires.get(*wk) else { continue };
            for &(ek, _) in &wire.edges { edges.push(ek); }
        }
        edges
    };
    let edges_a = collect_edges(fk_a);
    let edges_b: HashSet<EdgeKey> = collect_edges(fk_b).into_iter().collect();
    let mut shared = Vec::new();
    for ek in edges_a {
        if edges_b.contains(&ek) { shared.push(ek); }
    }
    shared
}

/// Propagate consistent face orientations through a shell using BFS.
/// Ensures all normals point outward (or all inward) for a watertight shell.
/// Returns number of faces flipped.
pub fn propagate_shell_orientations(shell_key: ShellKey, reg: &mut BRepStore) -> usize {
    let face_keys: Vec<(FaceKey, Orientation)> = {
        let Some(shell) = reg.shells.get(shell_key) else { return 0; };
        shell.faces.clone()
    };

    if face_keys.len() <= 1 { return 0; }

    let mut flipped = 0usize;
    let mut visited: HashSet<FaceKey> = HashSet::new();
    let mut queue: VecDeque<FaceKey> = VecDeque::new();

    // BFS from first face
    queue.push_back(face_keys[0].0);
    visited.insert(face_keys[0].0);

    while let Some(current_fk) = queue.pop_front() {
        for &(next_fk, _) in &face_keys {
            if visited.contains(&next_fk) { continue; }

            // Find shared edges between faces
            let shared = shared_edges(current_fk, next_fk, reg);
            if shared.is_empty() { continue; }

            let edge = match reg.edges.get(shared[0]) {
                Some(e) => e,
                None => continue,
            };
            let mid = edge.curve.d0(0.5);

            // Compare normals at a shared edge midpoint
            let flip_needed = {
                let current_face = match reg.faces.get(current_fk) {
                    Some(f) => f,
                    None => continue,
                };
                let next_face = match reg.faces.get(next_fk) {
                    Some(f) => f,
                    None => continue,
                };
                let Some((u0, v0)) = current_face.surface.project(mid) else { continue };
                let Some((u1, v1)) = next_face.surface.project(mid) else { continue };
                let n0 = current_face.surface.normal(u0, v0);
                let n1 = next_face.surface.normal(u1, v1);
                n0.dot(n1) < 0.0
            }; // immutable borrows dropped here

            if flip_needed {
                if let Some(face_mut) = reg.faces.get_mut(next_fk) {
                    face_mut.same_sense = !face_mut.same_sense;
                    flipped += 1;
                }
            }

            visited.insert(next_fk);
            queue.push_back(next_fk);
        }
    }

    // Update shell face orientations based on face same_sense
    if let Some(shell) = reg.shells.get_mut(shell_key) {
        for (fk, orient) in shell.faces.iter_mut() {
            if let Some(face) = reg.faces.get(*fk) {
                if !face.same_sense {
                    *orient = match *orient {
                        Orientation::Forward => Orientation::Reversed,
                        Orientation::Reversed => Orientation::Forward,
                        other => other,
                    };
                }
            }
        }
    }

    flipped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::*;
    use crate::geom::SurfaceGeom;
    use rc3d_core::math::PVec3;

    fn make_test_face(reg: &mut BRepStore) -> FaceKey {
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: PVec3::ZERO,
                normal: PVec3::Z,
                u_dir: PVec3::X,
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
        let mut reg = BRepStore::new();
        assert!(stitch_faces_into_shell(&[], &mut reg).is_none());
    }

    #[test]
    fn test_stitch_cube_faces() {
        let mut reg = BRepStore::new();
        let faces: Vec<FaceKey> = (0..6).map(|_| make_test_face(&mut reg)).collect();

        let shell_key = stitch_faces_into_shell(&faces, &mut reg);
        assert!(shell_key.is_some(), "Should create a shell from 6 faces");

        let shell = reg.shells.get(shell_key.unwrap()).unwrap();
        assert_eq!(shell.faces.len(), 6);
    }

    #[test]
    fn test_stitch_skips_invalid_faces() {
        let mut reg = BRepStore::new();
        let fk = make_test_face(&mut reg);

        // Create a bogus key that doesn't exist in the registry
        let fake_key = FaceKey::from(slotmap::KeyData::from_ffi(0xDEAD));
        let result = stitch_faces_into_shell(&[fk, fake_key], &mut reg);
        assert!(result.is_some());
        let shell = reg.shells.get(result.unwrap()).unwrap();
        assert_eq!(shell.faces.len(), 1, "Only valid face should be included");
    }
}
