//! Face orientation fixing. T3.3

use std::collections::{HashSet, VecDeque};
use crate::topo::{ShellKey, FaceKey};
use crate::store::BRepRegistry;

/// Fix face orientations in a shell so all normals point consistently.
/// Returns number of faces flipped.
pub fn fix_shell_orientation(
    shell_key: ShellKey,
    reg: &mut BRepRegistry,
) -> usize {
    let face_keys: Vec<FaceKey> = {
        let shell = match reg.shells.get(shell_key) {
            Some(s) => s.faces.iter().map(|(fk, _)| *fk).collect(),
            None => return 0,
        };
        shell
    };

    if face_keys.len() <= 1 { return 0; }

    let mut flipped = 0;
    let mut visited: HashSet<FaceKey> = HashSet::new();
    let mut queue: VecDeque<FaceKey> = VecDeque::new();

    // Start BFS from first face
    queue.push_back(face_keys[0]);
    visited.insert(face_keys[0]);

    while let Some(current) = queue.pop_front() {
        for &next in &face_keys {
            if visited.contains(&next) { continue; }
            let shared = reg.find_shared_edges(current, next);
            if !shared.is_empty() {
                let current_face = match reg.faces.get(current) {
                    Some(f) => f, None => continue,
                };
                let next_face = match reg.faces.get(next) {
                    Some(f) => f, None => continue,
                };

                // Compare normals at a shared point
                let edge = reg.edges.get(shared[0]);
                if let (Some(e), Some(_pcurve)) = (edge, edge.and_then(|e| e.pcurves.get(&current))) {
                    let mid = e.curve.d0(0.5);
                    if let Some((u0, v0)) = current_face.surface.project(mid) {
                        if let Some((u1, v1)) = next_face.surface.project(mid) {
                            let n0 = current_face.surface.normal(u0, v0);
                            let n1 = next_face.surface.normal(u1, v1);
                            // If normals are opposite, flip next face
                            if n0.dot(n1) < 0.0 {
                                if let Some(face) = reg.faces.get_mut(next) {
                                    face.same_sense = !face.same_sense;
                                    flipped += 1;
                                }
                            }
                        }
                    }
                }
                visited.insert(next);
                queue.push_back(next);
            }
        }
    }

    flipped
}
