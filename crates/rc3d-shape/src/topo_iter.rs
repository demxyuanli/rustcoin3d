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

/// Iterate all (EdgeKey, Orientation) pairs in a face's outer wire only.
pub fn iter_edge_orientations_of_outer_wire(
    face: FaceKey,
    reg: &BRepStore,
) -> Vec<(EdgeKey, Orientation)> {
    let Some(face) = reg.faces.get(face) else {
        return vec![];
    };
    reg.wires
        .get(face.outer_wire)
        .map(|w| w.edges.iter().copied().collect())
        .unwrap_or_default()
}

/// Iterate all EdgeKeys in a face's outer wire only (deduplicated).
pub fn iter_edges_of_outer_wire(face: FaceKey, reg: &BRepStore) -> Vec<EdgeKey> {
    iter_edge_orientations_of_outer_wire(face, reg)
        .iter()
        .map(|(ek, _)| *ek)
        .collect::<HashSet<_>>()
        .into_iter()
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

// ═══════════════════════════════════════════════════════════════════════
//  Deep recursive traversal (OCC TopExp_Explorer equivalent)
// ═══════════════════════════════════════════════════════════════════════

/// Target shape type for deep exploration filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopExpTarget {
    Vertex,
    Edge,
    Wire,
    Face,
    Shell,
}

/// Recursively collect all edges of a shell.
///
/// OCC equivalent: `TopExp_Explorer(shell, TopAbs_EDGE)`
pub fn deep_edges_of_shell(shell_key: ShellKey, reg: &BRepStore) -> Vec<EdgeKey> {
    let mut edges = Vec::new();
    let mut seen = HashSet::new();
    let faces = iter_face_orientations_of_shell(shell_key, reg);
    for (fk, _) in faces {
        for wk in iter_wires_of_face(fk, reg) {
            let wire_edges = reg.wires.get(wk).map(|w| w.edges.clone()).unwrap_or_default();
            for (ek, _) in wire_edges {
                if seen.insert(ek) {
                    edges.push(ek);
                }
            }
        }
    }
    edges
}

/// Recursively collect all vertices of a face.
///
/// OCC equivalent: `TopExp_Explorer(face, TopAbs_VERTEX)`
pub fn deep_vertices_of_face(face_key: FaceKey, reg: &BRepStore) -> Vec<VertexKey> {
    let mut vertices = Vec::new();
    let mut seen = HashSet::new();
    for wk in iter_wires_of_face(face_key, reg) {
        let wire_edges = reg.wires.get(wk).map(|w| w.edges.clone()).unwrap_or_default();
        for (ek, _) in wire_edges {
            if let Some((v_lo, v_hi)) = iter_vertices_of_edge(ek, reg) {
                if seen.insert(v_lo) { vertices.push(v_lo); }
                if seen.insert(v_hi) { vertices.push(v_hi); }
            }
        }
    }
    vertices
}

/// Recursively collect all edges of a solid.
///
/// OCC equivalent: `TopExp_Explorer(solid, TopAbs_EDGE)`
pub fn deep_edges_of_solid(solid_key: SolidKey, reg: &BRepStore) -> Vec<EdgeKey> {
    let mut edges = Vec::new();
    let mut seen = HashSet::new();
    let solid = match reg.solids.get(solid_key) {
        Some(s) => s,
        None => return edges,
    };
    for shell_key in std::iter::once(&solid.outer_shell).chain(solid.void_shells.iter()) {
        for ek in deep_edges_of_shell(*shell_key, reg) {
            if seen.insert(ek) {
                edges.push(ek);
            }
        }
    }
    edges
}

/// Recursively collect all faces of a solid.
///
/// OCC equivalent: `TopExp_Explorer(solid, TopAbs_FACE)`
pub fn deep_faces_of_solid(solid_key: SolidKey, reg: &BRepStore) -> Vec<FaceKey> {
    let mut faces = Vec::new();
    let mut seen = HashSet::new();
    let solid = match reg.solids.get(solid_key) {
        Some(s) => s,
        None => return faces,
    };
    let shell_keys: Vec<ShellKey> = std::iter::once(solid.outer_shell)
        .chain(solid.void_shells.iter().copied())
        .collect();
    for sk in shell_keys {
        for fk in iter_faces_of_shell(sk, reg) {
            if seen.insert(fk) {
                faces.push(fk);
            }
        }
    }
    faces
}

/// Iterate all solids in a compound.
pub fn iter_solids_of_compound(compound: crate::topo::CompoundKey, reg: &BRepStore) -> Vec<crate::topo::SolidKey> {
    let Some(c) = reg.compounds.get(compound) else { return vec![]; };
    c.solids.clone()
}

/// Iterate all faces of a compsolid (outer + void shells).
pub fn iter_faces_of_compsolid(compsolid: crate::topo::SolidKey, reg: &BRepStore) -> Vec<crate::topo::FaceKey> {
    deep_faces_of_solid(compsolid, reg)
}

/// Recursively collect all vertices of a shell.
///
/// OCC equivalent: `TopExp_Explorer(shell, TopAbs_VERTEX)`
pub fn deep_vertices_of_shell(shell_key: ShellKey, reg: &BRepStore) -> Vec<VertexKey> {
    let mut vertices = Vec::new();
    let mut seen = HashSet::new();
    for fk in iter_faces_of_shell(shell_key, reg) {
        for vk in deep_vertices_of_face(fk, reg) {
            if seen.insert(vk) {
                vertices.push(vk);
            }
        }
    }
    vertices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use rc3d_core::math::Vec3;
    use std::collections::HashMap;

    /// Build a shell with two coplanar faces sharing one edge.
    ///
    /// Vertices: v0(0,0,0) v1(1,0,0) v2(1,1,0) v3(0,1,0)
    /// Edges: e01, e12, e23, e30
    /// Face A: full rectangle, normal=+Z, all 4 edges
    /// Face B: shares e01 only, normal=+Y, edges [e01]
    /// Shell: both faces
    fn setup_two_face_shell() -> (BRepStore, ShellKey, FaceKey, FaceKey) {
        let mut reg = BRepStore::new();

        // Vertices
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 1.0, 0.0), 1e-4);

        // Face A (rectangle, normal +Z)
        let face_a = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: WireKey::default(), // placeholder, set below
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        // Face B (shares e01, normal +Y)
        let face_b = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Y,
                u_dir: Vec3::X,
            },
            outer_wire: WireKey::default(), // placeholder, set below
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        // Edges (line curves)
        let line01 = CurveGeom::Line { origin: Vec3::new(0.0, 0.0, 0.0), direction: Vec3::X };
        let line12 = CurveGeom::Line { origin: Vec3::new(1.0, 0.0, 0.0), direction: Vec3::Y };
        let line23 = CurveGeom::Line { origin: Vec3::new(1.0, 1.0, 0.0), direction: Vec3::NEG_X };
        let line30 = CurveGeom::Line { origin: Vec3::new(0.0, 1.0, 0.0), direction: Vec3::NEG_Y };

        let e01 = reg.edges.insert(BRepEdge {
            curve: line01,
            tolerance: 1e-4,
            v_low: v0,
            v_high: v1,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: HashMap::from([(face_a, (Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) }, true)),
                                     (face_b, (Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) }, true))]),
        });
        let e12 = reg.edges.insert(BRepEdge {
            curve: line12,
            tolerance: 1e-4,
            v_low: v1,
            v_high: v2,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: HashMap::from([(face_a, (Curve2d::Line { origin: (0.0, 0.0), direction: (0.0, 1.0) }, true))]),
        });
        let e23 = reg.edges.insert(BRepEdge {
            curve: line23,
            tolerance: 1e-4,
            v_low: v2,
            v_high: v3,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: HashMap::from([(face_a, (Curve2d::Line { origin: (0.0, 0.0), direction: (-1.0, 0.0) }, true))]),
        });
        let e30 = reg.edges.insert(BRepEdge {
            curve: line30,
            tolerance: 1e-4,
            v_low: v3,
            v_high: v0,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: HashMap::from([(face_a, (Curve2d::Line { origin: (0.0, 0.0), direction: (0.0, -1.0) }, true))]),
        });

        // Wire A: 4 edges
        let wire_a = reg.wires.insert(BRepWire {
            edges: vec![
                (e01, Orientation::Forward),
                (e12, Orientation::Forward),
                (e23, Orientation::Forward),
                (e30, Orientation::Forward),
            ],
        });

        // Wire B: 1 shared edge
        let wire_b = reg.wires.insert(BRepWire {
            edges: vec![(e01, Orientation::Reversed)],
        });

        // Patch face outer wires
        reg.faces.get_mut(face_a).unwrap().outer_wire = wire_a;
        reg.faces.get_mut(face_b).unwrap().outer_wire = wire_b;

        // Shell with both faces
        let shell = reg.shells.insert(BRepShell {
            faces: vec![
                (face_a, Orientation::Forward),
                (face_b, Orientation::Forward),
            ],
            closed: false,
            step_id: None,
        });

        (reg, shell, face_a, face_b)
    }

    #[test]
    fn test_iter_wires_of_face() {
        let (reg, _shell, face_a, _face_b) = setup_two_face_shell();
        let wires = iter_wires_of_face(face_a, &reg);
        assert_eq!(wires.len(), 1, "face with 1 outer wire and 0 inner wires should return 1 wire");
    }

    #[test]
    fn test_iter_edges_of_face() {
        let (reg, _shell, face_a, _face_b) = setup_two_face_shell();
        let edges = iter_edges_of_face(face_a, &reg);
        assert_eq!(edges.len(), 4, "rectangular face should have 4 unique edges");
    }

    #[test]
    fn test_iter_edge_orientations_of_face() {
        let (reg, _shell, face_a, _face_b) = setup_two_face_shell();
        let oriented = iter_edge_orientations_of_face(face_a, &reg);
        assert_eq!(oriented.len(), 4, "rectangular face should have 4 oriented edges");
        assert!(oriented.iter().all(|(_, o)| *o == Orientation::Forward),
            "all orientations should be Forward in this test setup");
    }

    #[test]
    fn test_iter_edges_of_shell_dedup() {
        let (reg, shell, _face_a, _face_b) = setup_two_face_shell();
        let edges = iter_edges_of_shell(shell, &reg);
        assert_eq!(edges.len(), 4,
            "shell with 2 faces sharing 1 edge should have 4 unique edges (not 5)");
    }

    #[test]
    fn test_iter_faces_of_shell() {
        let (reg, shell, _face_a, _face_b) = setup_two_face_shell();
        let faces = iter_faces_of_shell(shell, &reg);
        assert_eq!(faces.len(), 2, "shell should contain 2 faces");
    }

    #[test]
    fn test_iter_vertices_of_edge() {
        let (reg, _shell, _face_a, _face_b) = setup_two_face_shell();
        // Use the first edge from face A's wire
        let face = reg.faces.get(_face_a).unwrap();
        let wire = reg.wires.get(face.outer_wire).unwrap();
        let (ek, _) = wire.edges[0];
        let (vl, vh) = iter_vertices_of_edge(ek, &reg).expect("edge should have vertices");
        assert!(vl < vh, "v_low should be less than v_high (SlotMap key ordering)");
    }

    #[test]
    fn test_iter_vertices_of_face() {
        let (reg, _shell, face_a, _face_b) = setup_two_face_shell();
        let verts = iter_vertices_of_face(face_a, &reg);
        assert_eq!(verts.len(), 4, "rectangular face should have 4 vertices");
    }

    #[test]
    fn test_missing_key_returns_empty() {
        let reg = BRepStore::new();
        let fake_face = FaceKey::default();
        let wires = iter_wires_of_face(fake_face, &reg);
        assert!(wires.is_empty(), "non-existent FaceKey should return empty vec");
        let edges = iter_edges_of_face(fake_face, &reg);
        assert!(edges.is_empty(), "non-existent FaceKey should return empty edges");
    }
}
