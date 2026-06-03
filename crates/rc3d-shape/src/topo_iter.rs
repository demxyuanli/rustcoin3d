//! Topology traversal utilities (OCC TopExp equivalent).
//!
//! Provides named functions for common topology iteration patterns.
//! Use these instead of inline `for &(ek, _) in &wire.edges` loops.

use crate::store::BRepStore;
use crate::topo::*;
use std::collections::HashSet;

/// Iterate all WireKeys of a face (outer_wire + inner_wires).
pub fn iter_wires_of_face(face: FaceKey, reg: &BRepStore) -> Vec<WireKey> {
    let Some(face) = reg.faces.get(face) else {
        return vec![];
    };
    let mut wires = Vec::with_capacity(1 + face.inner_wires.len());
    wires.push(face.outer_wire);
    wires.extend_from_slice(&face.inner_wires);
    wires
}

/// Iterate all (EdgeKey, Orientation) pairs in a face's wires.
pub fn iter_edge_orientations_of_face(
    face: FaceKey,
    reg: &BRepStore,
) -> Vec<(EdgeKey, Orientation)> {
    iter_wires_of_face(face, reg)
        .iter()
        .filter_map(|wk| reg.wires.get(*wk))
        .flat_map(|w| w.edges.iter().copied())
        .collect()
}

/// Iterate all EdgeKeys in a face's wires (deduplicated).
pub fn iter_edges_of_face(face: FaceKey, reg: &BRepStore) -> Vec<EdgeKey> {
    iter_edge_orientations_of_face(face, reg)
        .iter()
        .map(|(ek, _)| *ek)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

/// Iterate all unique EdgeKeys in a shell (deduplicated).
pub fn iter_edges_of_shell(shell: ShellKey, reg: &BRepStore) -> Vec<EdgeKey> {
    let Some(shell) = reg.shells.get(shell) else {
        return vec![];
    };
    let mut set = HashSet::new();
    for &(fk, _) in &shell.faces {
        for ek in iter_edges_of_face(fk, reg) {
            set.insert(ek);
        }
    }
    set.into_iter().collect()
}

/// Iterate all (FaceKey, Orientation) pairs in a shell.
pub fn iter_face_orientations_of_shell(
    shell: ShellKey,
    reg: &BRepStore,
) -> Vec<(FaceKey, Orientation)> {
    let Some(shell) = reg.shells.get(shell) else {
        return vec![];
    };
    shell.faces.clone()
}

/// Iterate all FaceKeys in a shell (without orientation).
pub fn iter_faces_of_shell(shell: ShellKey, reg: &BRepStore) -> Vec<FaceKey> {
    iter_face_orientations_of_shell(shell, reg)
        .iter()
        .map(|(fk, _)| *fk)
        .collect()
}

/// Get the two endpoint vertices of an edge.
pub fn iter_vertices_of_edge(edge: EdgeKey, reg: &BRepStore) -> Option<(VertexKey, VertexKey)> {
    reg.edges.get(edge).map(|e| (e.v_low, e.v_high))
}

/// Iterate all unique VertexKeys in a face (via its edges).
pub fn iter_vertices_of_face(face: FaceKey, reg: &BRepStore) -> Vec<VertexKey> {
    let mut set = HashSet::new();
    for ek in iter_edges_of_face(face, reg) {
        if let Some((vl, vh)) = iter_vertices_of_edge(ek, reg) {
            set.insert(vl);
            set.insert(vh);
        }
    }
    set.into_iter().collect()
}

/// Collect all unique EdgeKeys from a shell (convenience wrapper).
pub fn shell_edge_keys(shell: ShellKey, reg: &BRepStore) -> Vec<EdgeKey> {
    iter_edges_of_shell(shell, reg)
}

/// Collect all FaceKeys from a solid (outer + void shells).
pub fn iter_faces_of_solid(solid: SolidKey, reg: &BRepStore) -> Vec<FaceKey> {
    let Some(solid) = reg.solids.get(solid) else {
        return vec![];
    };
    let mut faces = iter_faces_of_shell(solid.outer_shell, reg);
    for &void_shell in &solid.void_shells {
        faces.extend(iter_faces_of_shell(void_shell, reg));
    }
    faces
}
