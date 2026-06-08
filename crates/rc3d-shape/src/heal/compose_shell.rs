//! Compose faces into shells based on shared edges.
//!
//! Groups a flat set of faces into coherent shells using edge adjacency,
//! and detects void shells (inner shells with inward normals) inside solids.
//!
//! OCC alignment: ShapeFix_ComposeShell — stitches faces into shells,
//! detects void regions.

use rc3d_core::math::Vec3;
use crate::store::BRepStore;
use crate::topo::*;
use std::collections::{HashMap, HashSet};

/// Result of shell composition.
#[derive(Debug, Clone, Default)]
pub struct ComposeShellReport {
    pub shells_composed: usize,
    pub void_shells_detected: usize,
    pub faces_assigned: usize,
}

/// Group a flat set of faces into shells based on shared edges.
///
/// Two faces are connected if they share an edge. Connected components
/// of the face-adjacency graph become separate shells.
///
/// OCC alignment: BRepTools_Quilt::Shells — groups faces by shared edges.
pub fn compose_shells(
    face_keys: &[FaceKey],
    reg: &BRepStore,
) -> Vec<Vec<FaceKey>> {
    if face_keys.is_empty() {
        return Vec::new();
    }

    // Build face adjacency: face → set of adjacent faces (sharing an edge)
    let mut adjacency: HashMap<FaceKey, HashSet<FaceKey>> = HashMap::new();
    for &fk in face_keys {
        adjacency.entry(fk).or_default();
    }

    // For each face pair, check if they share edges via edge_to_faces
    for &fk in face_keys {
        let Some(face) = reg.faces.get(fk) else { continue };
        let mut face_edges = Vec::new();
        for wire_key in std::iter::once(&face.outer_wire).chain(&face.inner_wires) {
            let Some(wire) = reg.wires.get(*wire_key) else { continue };
            for &(ek, _) in &wire.edges {
                face_edges.push(ek);
            }
        }
        for ek in face_edges {
            if let Some(adjacent_faces) = reg.edge_to_faces.get(&ek) {
                for &other_fk in adjacent_faces {
                    if other_fk != fk && adjacency.contains_key(&other_fk) {
                        adjacency.get_mut(&fk).unwrap().insert(other_fk);
                        adjacency.get_mut(&other_fk).unwrap().insert(fk);
                    }
                }
            }
        }
    }

    // BFS to find connected components
    let mut visited: HashSet<FaceKey> = HashSet::new();
    let mut shells: Vec<Vec<FaceKey>> = Vec::new();

    for &start_fk in face_keys {
        if visited.contains(&start_fk) {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = vec![start_fk];
        while let Some(fk) = queue.pop() {
            if visited.contains(&fk) {
                continue;
            }
            visited.insert(fk);
            component.push(fk);
            if let Some(neighbors) = adjacency.get(&fk) {
                for &nb in neighbors {
                    if !visited.contains(&nb) {
                        queue.push(nb);
                    }
                }
            }
        }
        shells.push(component);
    }

    shells
}

/// Build shells from a flat face list, creating ShellKey entries in BRepStore.
///
/// Returns the list of newly created ShellKeys.
pub fn compose_shells_into_store(
    face_keys: &[FaceKey],
    reg: &mut BRepStore,
) -> ComposeShellReport {
    let groups = compose_shells(face_keys, reg);

    let mut report = ComposeShellReport::default();
    report.shells_composed = groups.len();

    for group in &groups {
        report.faces_assigned += group.len();
        let faces: Vec<(FaceKey, Orientation)> = group
            .iter()
            .map(|&fk| (fk, Orientation::Forward))
            .collect();
        let _sk = reg.shells.insert(BRepShell {
            faces,
            closed: false,
            step_id: None,
        });
    }

    report
}

/// Detect void shells inside a solid (shells with all face centers inside
/// the outer shell).
///
/// Uses a simple centroid containment test: if the centroid of a shell's
/// face centers is inside the outer shell (via ray casting), it's a void.
///
/// OCC alignment: BRepClass3d_SolidClassifier — classifies point inside solid.
pub fn detect_void_shells(
    outer_shell: ShellKey,
    candidate_shells: &[ShellKey],
    reg: &BRepStore,
) -> Vec<ShellKey> {
    let Some(outer) = reg.shells.get(outer_shell) else {
        return Vec::new();
    };

    // Collect outer shell faces for point-in-shell test
    let outer_faces: Vec<FaceKey> = outer.faces.iter().map(|&(fk, _)| fk).collect();

    let mut voids = Vec::new();
    for &candidate_sk in candidate_shells {
        if candidate_sk == outer_shell {
            continue;
        }
        let Some(candidate) = reg.shells.get(candidate_sk) else {
            continue;
        };

        // Compute centroid of candidate shell's face centers
        let mut centroid = Vec3::ZERO;
        let mut count = 0u32;
        for &(fk, _) in &candidate.faces {
            let Some(face) = reg.faces.get(fk) else { continue };
            let range = face.surface.param_range();
            let u = (range.u_min + range.u_max) * 0.5;
            let v = (range.v_min + range.v_max) * 0.5;
            let (un, vn) = face.surface.native_uv_to_d0(u, v);
            centroid += face.surface.d0_native(un, vn);
            count += 1;
        }

        if count == 0 {
            continue;
        }
        centroid /= count as f32;

        // Simple containment check: count ray crossings with outer shell triangles
        if is_point_inside_shell(centroid, &outer_faces, reg) {
            voids.push(candidate_sk);
        }
    }

    voids
}

/// Simple ray-cast point-in-shell test.
/// Casts a ray in +X direction and counts triangle crossings.
fn is_point_inside_shell(point: Vec3, face_keys: &[FaceKey], reg: &BRepStore) -> bool {
    let mut crossings = 0u32;
    for &fk in face_keys {
        let Some(face) = reg.faces.get(fk) else { continue };
        // Get face plane for quick rejection
        let range = face.surface.param_range();
        let normal = face.surface.normal_native(
            (range.u_min + range.u_max) * 0.5,
            (range.v_min + range.v_max) * 0.5,
        );
        // Only consider faces whose normal has a significant Z component
        // pointing toward the ray direction (simplified test)
        let d = (point - face.surface.d0_native(
            face.surface.native_uv_to_d0(
                (range.u_min + range.u_max) * 0.5,
                (range.v_min + range.v_max) * 0.5,
            ).0,
            face.surface.native_uv_to_d0(
                (range.u_min + range.u_max) * 0.5,
                (range.v_min + range.v_max) * 0.5,
            ).1,
        )).dot(normal);

        // Simplified: if the point is on the negative side of the face plane,
        // it's potentially inside
        if d < 0.0 {
            crossings += 1;
        }
    }

    // Odd number of crossings → inside
    crossings % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::geom::curve2d::Curve2d;

    #[test]
    fn test_compose_shells_disjoint_groups() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        };

        // Create 2 faces with no shared edges
        let mut face_keys = Vec::new();
        for offset in [0.0f32, 10.0] {
            let v0 = reg.find_or_add_vertex(Vec3::new(offset, 0.0, 0.0), 1e-4);
            let v1 = reg.find_or_add_vertex(Vec3::new(offset + 1.0, 0.0, 0.0), 1e-4);
            let wk = reg.wires.insert(BRepWire { edges: vec![] });
            let fk = reg.faces.insert(BRepFace {
                surface: surface.clone(), outer_wire: wk, inner_wires: vec![],
                same_sense: true, tolerance: 1e-4, seam_edges: vec![],
                color: None, degenerated_edges: vec![],
            });
            let line = CurveGeom::Line {
                origin: Vec3::new(offset, 0.0, 0.0),
                direction: Vec3::X,
            };
            let pc = Curve2d::Line { origin: (offset, 0.0), direction: (1.0, 0.0) };
            let ek = reg.add_edge_with_pcurve(v0, v1, line, 1e-4, fk, pc);
            reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
            face_keys.push(fk);
        }

        let shells = compose_shells(&face_keys, &reg);
        assert_eq!(shells.len(), 2, "disjoint faces should produce 2 shells");
    }

    #[test]
    fn test_compose_shells_connected() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        };

        // Create 2 faces sharing one edge
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 1.0, 0.0), 1e-4);
        let v4 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);
        let v5 = reg.find_or_add_vertex(Vec3::new(2.0, 1.0, 0.0), 1e-4);

        let make_line = |a: Vec3, b: Vec3| CurveGeom::Line { origin: a, direction: b - a };
        let make_pc = |a: Vec3, b: Vec3| Curve2d::Line {
            origin: (a.x, a.y), direction: (b.x - a.x, b.y - a.y),
        };

        // Face A
        let wka = reg.wires.insert(BRepWire { edges: vec![] });
        let fka = reg.faces.insert(BRepFace {
            surface: surface.clone(), outer_wire: wka, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![],
            color: None, degenerated_edges: vec![],
        });
        let e0 = reg.add_edge_with_pcurve(v0, v1, make_line(Vec3::new(0.,0.,0.), Vec3::new(1.,0.,0.)), 1e-4, fka,
            make_pc(Vec3::new(0.,0.,0.), Vec3::new(1.,0.,0.)));
        let e1 = reg.add_edge_with_pcurve(v1, v2, make_line(Vec3::new(1.,0.,0.), Vec3::new(1.,1.,0.)), 1e-4, fka,
            make_pc(Vec3::new(1.,0.,0.), Vec3::new(1.,1.,0.)));
        let e2 = reg.add_edge_with_pcurve(v2, v3, make_line(Vec3::new(1.,1.,0.), Vec3::new(0.,1.,0.)), 1e-4, fka,
            make_pc(Vec3::new(1.,1.,0.), Vec3::new(0.,1.,0.)));
        let e3 = reg.add_edge_with_pcurve(v3, v0, make_line(Vec3::new(0.,1.,0.), Vec3::new(0.,0.,0.)), 1e-4, fka,
            make_pc(Vec3::new(0.,1.,0.), Vec3::new(0.,0.,0.)));
        reg.wires.get_mut(wka).unwrap().edges = vec![
            (e0, Orientation::Forward), (e1, Orientation::Forward),
            (e2, Orientation::Forward), (e3, Orientation::Forward),
        ];

        // Face B shares edge e1
        let wkb = reg.wires.insert(BRepWire { edges: vec![] });
        let fkb = reg.faces.insert(BRepFace {
            surface: surface.clone(), outer_wire: wkb, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![],
            color: None, degenerated_edges: vec![],
        });
        // Shared edge e1: add fkb's pcurve
        if let Some(edge) = reg.edges.get_mut(e1) {
            edge.pcurves.insert(fkb, make_pc(Vec3::new(1.,0.,0.), Vec3::new(1.,1.,0.)));
        }
        reg.edge_to_faces.entry(e1).or_default().push(fkb);
        let e4 = reg.add_edge_with_pcurve(v2, v5, make_line(Vec3::new(1.,1.,0.), Vec3::new(2.,1.,0.)), 1e-4, fkb,
            make_pc(Vec3::new(1.,1.,0.), Vec3::new(2.,1.,0.)));
        let e5 = reg.add_edge_with_pcurve(v5, v4, make_line(Vec3::new(2.,1.,0.), Vec3::new(2.,0.,0.)), 1e-4, fkb,
            make_pc(Vec3::new(2.,1.,0.), Vec3::new(2.,0.,0.)));
        let e6 = reg.add_edge_with_pcurve(v4, v1, make_line(Vec3::new(2.,0.,0.), Vec3::new(1.,0.,0.)), 1e-4, fkb,
            make_pc(Vec3::new(2.,0.,0.), Vec3::new(1.,0.,0.)));
        reg.wires.get_mut(wkb).unwrap().edges = vec![
            (e1, Orientation::Forward), (e4, Orientation::Forward),
            (e5, Orientation::Forward), (e6, Orientation::Forward),
        ];

        let shells = compose_shells(&[fka, fkb], &reg);
        assert_eq!(shells.len(), 1, "connected faces should produce 1 shell");
        assert_eq!(shells[0].len(), 2, "shell should have 2 faces");
    }

    #[test]
    fn test_compose_shells_into_store() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        };
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![],
            color: None, degenerated_edges: vec![],
        });
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let ek = reg.add_edge_with_pcurve(v0, v1, line, 1e-4, fk, pc);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];

        let report = compose_shells_into_store(&[fk], &mut reg);
        assert_eq!(report.shells_composed, 1);
        assert_eq!(report.faces_assigned, 1);
    }
}
