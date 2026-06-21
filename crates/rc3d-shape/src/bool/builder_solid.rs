//! BOPAlgo_BuilderSolid — reconstruct solid topology from split faces.
//!
//! OCC alignment: BOPAlgo_BuilderSolid is the phase after BOPAlgo_BuilderFace
//! that takes split faces from a boolean operation and reconstructs the solid
//! topology by building an edge-based adjacency graph, finding connected
//! components, classifying shells as outer/void, and assembling BRepSolid.
//!
//! Algorithm:
//! 1. Build adjacency graph: two faces are adjacent if they share at least one edge
//! 2. BFS to find connected face components (each = potential shell)
//! 3. For each component: build shell, check if closed, compute volume
//! 4. Largest positive volume → outer shell; other closed → void shells
//! 5. Assemble BRepSolid with outer_shell + void_shells

use std::collections::{HashMap, HashSet, VecDeque};

use rc3d_core::math::Real;

use crate::geom::properties::solid_volume;
use crate::store::BRepStore;
use crate::topo::{BRepShell, BRepSolid, FaceKey, Orientation, ShellKey, SolidKey, VertexKey};
use crate::topo_iter;

/// Reconstruct solids from a set of split faces.
///
/// Groups faces into connected components by edge sharing, builds shells,
/// classifies each shell as outer (largest positive volume) or void (negative
/// volume), and assembles `BRepSolid` entries. Returns the newly created
/// `SolidKey`s.
pub fn build_solids_from_faces(faces: &[FaceKey], reg: &mut BRepStore) -> Vec<SolidKey> {
    if faces.is_empty() {
        return vec![];
    }

    // 1. Build adjacency: face → adjacent faces (share at least one edge)
    let adjacency = build_face_adjacency(faces, reg);

    // 2. BFS: find connected face components
    let components = find_connected_components(faces, &adjacency);

    // 3. For each component: build shell, check closed, compute volume
    let mut candidates: Vec<CandidateShell> = Vec::new();
    for component in &components {
        if component.len() < 3 {
            continue;
        }
        let closed = is_shell_closed(component, reg);
        let Some(shell_key) = build_shell_from_faces(component, reg) else {
            continue;
        };
        // Compute volume via a temporary solid
        let temp_solid = reg.solids.insert(BRepSolid {
            outer_shell: shell_key,
            void_shells: vec![],
        });
        let vol = solid_volume(reg, temp_solid, 16);
        reg.solids.remove(temp_solid);
        candidates.push(CandidateShell {
            shell_key,
            closed,
            volume: vol,
        });
    }

    if candidates.is_empty() {
        return vec![];
    }

    // 4. Classify: positive volume → outer shell, negative volume → void shell.
    //    Each positive-volume closed shell becomes its own solid (disjoint
    //    components after boolean may produce multiple independent solids).
    //    Negative-volume closed shells are grouped as void shells under the
    //    largest outer shell (heuristic; full spatial containment test would
    //    be needed for proper void-to-outer assignment).
    let mut outer_candidates: Vec<ShellKey> = Vec::new();
    let mut void_candidates: Vec<ShellKey> = Vec::new();
    for c in &candidates {
        if !c.closed {
            continue;
        }
        if c.volume > 1e-12 {
            outer_candidates.push(c.shell_key);
        } else if c.volume < -1e-12 {
            void_candidates.push(c.shell_key);
        }
    }

    if outer_candidates.is_empty() {
        // No closed positive-volume shells: create a best-effort solid from
        // the largest component (even if open or zero volume).
        if let Some(c) = candidates.first() {
            let solid_key = reg.solids.insert(BRepSolid {
                outer_shell: c.shell_key,
                void_shells: vec![],
            });
            return vec![solid_key];
        }
        return vec![];
    }

    // 5. Assemble: each closed outer shell → one solid.
    //    Void shells are assigned to the first (largest) outer shell.
    let mut solids = Vec::new();
    for (i, &outer) in outer_candidates.iter().enumerate() {
        let voids_for_this = if i == 0 {
            void_candidates.clone()
        } else {
            vec![]
        };
        let solid_key = reg.solids.insert(BRepSolid {
            outer_shell: outer,
            void_shells: voids_for_this,
        });
        solids.push(solid_key);
    }
    solids
}

struct CandidateShell {
    shell_key: ShellKey,
    closed: bool,
    volume: Real,
}

/// Build face adjacency map: face → faces sharing at least one edge.
///
/// Uses unordered vertex-pair matching instead of EdgeKey identity because
/// `add_edge_with_pcurve` may assign distinct EdgeKeys to the same physical
/// edge when the two faces traverse it in opposite directions (normalized
/// curves don't match). Vertex-pair adjacency is deterministic regardless
/// of edge deduplication.
fn build_face_adjacency(faces: &[FaceKey], reg: &BRepStore) -> HashMap<FaceKey, Vec<FaceKey>> {
    // Map each face → set of unordered vertex pairs on its boundary
    let face_vpairs: HashMap<FaceKey, HashSet<(VertexKey, VertexKey)>> = faces
        .iter()
        .map(|&fk| {
            let vpairs = face_vertex_pairs(fk, reg);
            (fk, vpairs)
        })
        .collect();

    let mut adj: HashMap<FaceKey, Vec<FaceKey>> = HashMap::new();
    for &fk in faces {
        let mut neighbors: HashSet<FaceKey> = HashSet::new();
        let my_pairs = match face_vpairs.get(&fk) {
            Some(p) => p,
            None => continue,
        };
        for &other in faces {
            if other == fk {
                continue;
            }
            if let Some(other_pairs) = face_vpairs.get(&other) {
                if my_pairs.intersection(other_pairs).next().is_some() {
                    neighbors.insert(other);
                }
            }
        }
        adj.insert(fk, neighbors.into_iter().collect());
    }
    adj
}

/// Collect the unordered vertex pairs (min, max) for all edges of a face.
fn face_vertex_pairs(fk: FaceKey, reg: &BRepStore) -> HashSet<(VertexKey, VertexKey)> {
    let mut pairs = HashSet::new();
    for (ek, _) in topo_iter::iter_edge_orientations_of_face(fk, reg) {
        if let Some((vl, vh)) = topo_iter::iter_vertices_of_edge(ek, reg) {
            let ordered = if vl < vh { (vl, vh) } else { (vh, vl) };
            pairs.insert(ordered);
        }
    }
    pairs
}

/// BFS to find connected components in the face adjacency graph.
fn find_connected_components(
    faces: &[FaceKey],
    adj: &HashMap<FaceKey, Vec<FaceKey>>,
) -> Vec<Vec<FaceKey>> {
    let mut visited: HashSet<FaceKey> = HashSet::new();
    let mut components: Vec<Vec<FaceKey>> = Vec::new();

    for &fk in faces {
        if visited.contains(&fk) {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(fk);
        visited.insert(fk);
        while let Some(current) = queue.pop_front() {
            component.push(current);
            if let Some(neighbors) = adj.get(&current) {
                for &nb in neighbors {
                    if !visited.contains(&nb) {
                        visited.insert(nb);
                        queue.push_back(nb);
                    }
                }
            }
        }
        components.push(component);
    }
    components
}

/// Build a BRepShell from a set of faces.
fn build_shell_from_faces(faces: &[FaceKey], reg: &mut BRepStore) -> Option<ShellKey> {
    if faces.is_empty() {
        return None;
    }
    let face_entries: Vec<(FaceKey, Orientation)> = faces
        .iter()
        .filter(|&&fk| reg.faces.contains_key(fk))
        .map(|&fk| (fk, Orientation::Forward))
        .collect();
    if face_entries.is_empty() {
        return None;
    }
    let closed = is_shell_closed(faces, reg);
    let shell = BRepShell {
        faces: face_entries,
        closed,
        step_id: None,
    };
    Some(reg.shells.insert(shell))
}

/// Check if a set of faces forms a closed shell.
///
/// A shell is closed when every non-seam edge (vertex pair) in the face set
/// is shared by exactly two faces (manifold, watertight). Seam edges on
/// periodic surfaces (cylinder, sphere, torus) connect a vertex pair that
/// appears only once per face and may have count 1 in single-face shells.
fn is_shell_closed(faces: &[FaceKey], reg: &BRepStore) -> bool {
    if faces.len() < 3 {
        return false;
    }
    // Collect vertex pairs, tracking which are from seam edges
    let mut vp_count: HashMap<(VertexKey, VertexKey), usize> = HashMap::new();
    let mut seam_pairs: HashSet<(VertexKey, VertexKey)> = HashSet::new();
    for &fk in faces {
        let face = match reg.faces.get(fk) { Some(f) => f, None => continue };
        // Build set of vertex pairs from seam edges
        for &ek in &face.seam_edges {
            if let Some((vl, vh)) = topo_iter::iter_vertices_of_edge(ek, reg) {
                seam_pairs.insert(if vl < vh { (vl, vh) } else { (vh, vl) });
            }
        }
        for vp in face_vertex_pairs(fk, reg) {
            *vp_count.entry(vp).or_default() += 1;
        }
    }

    if vp_count.is_empty() {
        return false;
    }

    // Non-seam vertex pairs must appear exactly twice.
    // Seam vertex pairs may appear once (single-face periodic shell) or twice.
    vp_count.iter().all(|(vp, &c)| {
        c == 2 || (seam_pairs.contains(vp) && c == 1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Curve2d, CurveGeom, SurfaceGeom};
    use crate::topo::*;
    use rc3d_core::math::PVec3;

    /// Build axis-aligned cube faces (size×size×size) centered at `center`.
    /// Each face has 4 edges; each physical edge is shared by exactly 2 faces.
    fn make_box_faces(reg: &mut BRepStore, size: Real, center: PVec3) -> Vec<FaceKey> {
        let h = size * 0.5;
        let tol = 1e-4;
        let (cx, cy, cz) = (center.x, center.y, center.z);

        // 8 corner vertices
        let mut v = |x: Real, y: Real, z: Real| -> VertexKey {
            reg.find_or_add_vertex(PVec3::new(cx + x, cy + y, cz + z), tol)
        };
        let v000 = v(-h, -h, -h);
        let v100 = v(h, -h, -h);
        let v110 = v(h, h, -h);
        let v010 = v(-h, h, -h);
        let v001 = v(-h, -h, h);
        let v101 = v(h, -h, h);
        let v111 = v(h, h, h);
        let v011 = v(-h, h, h);

        // Face definitions: (surface, corner vertices in CCW order from outside)
        struct FaceSpec {
            surface: SurfaceGeom,
            corners: [VertexKey; 4],
        }

        let specs: [FaceSpec; 6] = [
            // -Z (bottom): normal = -Z
            FaceSpec {
                surface: SurfaceGeom::Plane { origin: center + PVec3::new(0.0, 0.0, -h), normal: -PVec3::Z, u_dir: PVec3::X },
                corners: [v000, v100, v110, v010],
            },
            // +Z (top): normal = +Z
            FaceSpec {
                surface: SurfaceGeom::Plane { origin: center + PVec3::new(0.0, 0.0, h), normal: PVec3::Z, u_dir: PVec3::X },
                corners: [v001, v011, v111, v101],
            },
            // -Y (front): normal = -Y
            FaceSpec {
                surface: SurfaceGeom::Plane { origin: center + PVec3::new(0.0, -h, 0.0), normal: -PVec3::Y, u_dir: PVec3::X },
                corners: [v000, v001, v101, v100],
            },
            // +Y (back): normal = +Y
            FaceSpec {
                surface: SurfaceGeom::Plane { origin: center + PVec3::new(0.0, h, 0.0), normal: PVec3::Y, u_dir: PVec3::X },
                corners: [v010, v110, v111, v011],
            },
            // -X (left): normal = -X
            FaceSpec {
                surface: SurfaceGeom::Plane { origin: center + PVec3::new(-h, 0.0, 0.0), normal: -PVec3::X, u_dir: PVec3::Y },
                corners: [v000, v010, v011, v001],
            },
            // +X (right): normal = +X
            FaceSpec {
                surface: SurfaceGeom::Plane { origin: center + PVec3::new(h, 0.0, 0.0), normal: PVec3::X, u_dir: PVec3::Y },
                corners: [v100, v101, v111, v110],
            },
        ];

        let mut faces = Vec::new();
        for spec in &specs {
            let wk = reg.wires.insert(BRepWire { edges: vec![] });
            let fk = reg.faces.insert(BRepFace {
                surface: spec.surface.clone(),
                outer_wire: wk,
                inner_wires: vec![],
                same_sense: true,
                tolerance: tol,
                seam_edges: vec![],
                color: None,
                degenerated_edges: vec![],
            });
            reg.trim_ranges.insert(fk, (0.0, size, 0.0, size));
            faces.push(fk);
        }

        // Populate wire edges — each edge is shared between exactly 2 faces
        for (i, spec) in specs.iter().enumerate() {
            let fk = faces[i];
            let mut wire_edges = Vec::new();
            let c = &spec.corners;
            let corner_pairs = [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])];

            for (va, vb) in corner_pairs {
                let pa = reg.vertices.get(va).unwrap().position;
                let pb = reg.vertices.get(vb).unwrap().position;
                let line = CurveGeom::Line { origin: pa, direction: pb - pa };
                let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
                let ek = reg.add_edge_with_pcurve(va, vb, line, tol, fk, pc, true);
                wire_edges.push((ek, Orientation::Forward));
            }
            reg.wires.get_mut(reg.faces.get(fk).unwrap().outer_wire)
                .unwrap()
                .edges = wire_edges;
        }

        faces
    }

    #[test]
    fn test_is_shell_closed_cube() {
        let mut reg = BRepStore::new();
        let faces = make_box_faces(&mut reg, 4.0, PVec3::ZERO);
        assert_eq!(faces.len(), 6);
        assert!(is_shell_closed(&faces, &reg), "cube of 6 faces should be closed");
    }

    #[test]
    fn test_is_shell_closed_open() {
        let mut reg = BRepStore::new();
        let faces = make_box_faces(&mut reg, 4.0, PVec3::ZERO);
        // Remove one face → open shell
        let open_faces: Vec<FaceKey> = faces[..5].to_vec();
        assert!(!is_shell_closed(&open_faces, &reg), "5-face shell should be open");
    }

    #[test]
    fn test_is_shell_closed_extra_face() {
        let mut reg = BRepStore::new();
        let faces = make_box_faces(&mut reg, 4.0, PVec3::ZERO);
        // Add a duplicate face → non-manifold (some edges will have 3 faces)
        let mut extra = faces.clone();
        extra.push(faces[0]);
        assert!(!is_shell_closed(&extra, &reg), "edge with 3 faces is non-manifold");
    }

    #[test]
    fn test_build_solids_from_faces_single_cube() {
        let mut reg = BRepStore::new();
        let faces = make_box_faces(&mut reg, 4.0, PVec3::ZERO);

        let solids = build_solids_from_faces(&faces, &mut reg);
        assert_eq!(solids.len(), 1, "should produce one solid");

        let solid = reg.solids.get(solids[0]).unwrap();
        let shell = reg.shells.get(solid.outer_shell).unwrap();
        assert_eq!(shell.faces.len(), 6);
        assert!(shell.closed);
        assert!(solid.void_shells.is_empty());

        // Volume should be ~64 (4×4×4)
        let vol = solid_volume(&reg, solids[0], 16);
        assert!((vol - 64.0).abs() < 10.0, "volume ~64, got {}", vol);
    }

    #[test]
    fn test_build_solids_from_faces_disjoint_components() {
        // Two separate cubes at different positions → two solids
        let mut reg = BRepStore::new();
        let faces1 = make_box_faces(&mut reg, 2.0, PVec3::new(-3.0, 0.0, 0.0));
        let faces2 = make_box_faces(&mut reg, 2.0, PVec3::new(3.0, 0.0, 0.0));

        let mut all_faces = faces1.clone();
        all_faces.extend(faces2);

        let solids = build_solids_from_faces(&all_faces, &mut reg);
        // Two disjoint cubes should produce 2 solids
        assert_eq!(solids.len(), 2, "two disjoint cubes → two solids, got {}", solids.len());
    }

    #[test]
    fn test_build_solids_from_faces_empty() {
        let mut reg = BRepStore::new();
        let solids = build_solids_from_faces(&[], &mut reg);
        assert!(solids.is_empty());
    }

    #[test]
    fn test_is_shell_closed_empty() {
        let reg = BRepStore::new();
        assert!(!is_shell_closed(&[], &reg));
        // Single face can't be closed
        let mut reg2 = BRepStore::new();
        let wk = reg2.wires.insert(BRepWire { edges: vec![] });
        let fk = reg2.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: PVec3::ZERO,
                normal: PVec3::Z,
                u_dir: PVec3::X,
            },
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        assert!(!is_shell_closed(&[fk], &reg2), "single face is not closed");
    }
}
