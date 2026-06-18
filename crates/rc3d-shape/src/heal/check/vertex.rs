//! BRepCheck_Vertex — vertex-on-curve and vertex-on-surface validation.

use crate::store::BRepStore;
use crate::topo::VertexKey;

use super::CheckStatus;

/// Check that each vertex lies on the 3D curves of all edges it belongs to.
///
/// For each edge referencing the vertex, the vertex position is compared
/// against the curve endpoint. If the distance exceeds the edge tolerance,
/// the vertex is flagged as `InvalidPointOnCurve`.
pub fn check_vertex_on_curves(vk: VertexKey, reg: &BRepStore) -> Vec<CheckStatus> {
    let mut statuses = Vec::new();
    let Some(vertex) = reg.vertices.get(vk) else {
        return statuses;
    };

    let edges = match reg.vertex_to_edges.get(&vk) {
        Some(edges) => edges.clone(),
        None => return statuses,
    };

    for &ek in &edges {
        let Some(edge) = reg.edges.get(ek) else {
            continue;
        };
        let t = if edge.v_low == vk { edge.t_min } else { edge.t_max };
        let curve_point = edge.curve.d0(t);
        let dist = (vertex.position - curve_point).length();
        let tol = edge.tolerance.max(1e-9);
        if dist > tol {
            statuses.push(CheckStatus::InvalidPointOnCurve);
        }
        // Only report once per vertex
        if !statuses.is_empty() {
            break;
        }
    }

    statuses
}

/// Check that each vertex lies on the surfaces of all faces it is incident to.
///
/// For each face reachable through the vertex's edges, the vertex is projected
/// onto the surface; if the projection distance exceeds the face tolerance,
/// the vertex is flagged as `InvalidPointOnSurface`.
pub fn check_vertex_on_surfaces(vk: VertexKey, reg: &BRepStore) -> Vec<CheckStatus> {
    let mut statuses = Vec::new();
    let Some(vertex) = reg.vertices.get(vk) else {
        return statuses;
    };

    let edges = match reg.vertex_to_edges.get(&vk) {
        Some(edges) => edges.clone(),
        None => return statuses,
    };

    let mut seen_faces = std::collections::HashSet::new();

    for &ek in &edges {
        let face_list = match reg.edge_to_faces.get(&ek) {
            Some(faces) => faces.clone(),
            None => continue,
        };
        for fk in face_list {
            if !seen_faces.insert(fk) {
                continue;
            }
            let Some(face) = reg.faces.get(fk) else {
                continue;
            };
            let Some((_u, _v)) = face.surface.project(vertex.position) else {
                // Projection failed — may be far from surface
                statuses.push(CheckStatus::InvalidPointOnSurface);
                continue;
            };
            let projected = face.surface.d0(_u, _v);
            let dist = (vertex.position - projected).length();
            let tol = face.tolerance.max(1e-9);
            if dist > tol {
                statuses.push(CheckStatus::InvalidPointOnSurface);
            }
        }
        if !statuses.is_empty() {
            break;
        }
    }

    statuses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepWire, Orientation};
    use rc3d_core::math::PVec3;

    fn build_test_reg() -> (BRepStore, VertexKey, VertexKey) {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let curve = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
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
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let ek = reg.add_edge_with_pcurve(v0, v1, curve, 1e-4, fk, pc, true);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        // Manually ensure inverted indices are populated
        reg.vertex_to_edges.entry(v0).or_default().push(ek);
        reg.vertex_to_edges.entry(v1).or_default().push(ek);
        reg.edge_to_faces.entry(ek).or_default().push(fk);
        (reg, v0, v1)
    }

    #[test]
    fn vertex_on_curve_valid() {
        let (reg, v0, _v1) = build_test_reg();
        let statuses = check_vertex_on_curves(v0, &reg);
        assert!(
            statuses.is_empty(),
            "vertex at curve endpoint should be valid, got {:?}",
            statuses
        );
    }

    #[test]
    fn vertex_on_curve_invalid_when_far() {
        let (mut reg, v0, _v1) = build_test_reg();
        // Move vertex far from where the edge expects
        if let Some(v) = reg.vertices.get_mut(v0) {
            v.position = PVec3::new(999.0, 0.0, 0.0);
        }
        let statuses = check_vertex_on_curves(v0, &reg);
        assert!(
            statuses.contains(&CheckStatus::InvalidPointOnCurve),
            "vertex far from curve should be flagged"
        );
    }

    #[test]
    fn vertex_on_surface_valid() {
        let (reg, v0, _v1) = build_test_reg();
        let statuses = check_vertex_on_surfaces(v0, &reg);
        assert!(
            statuses.is_empty(),
            "vertex on plane surface should be valid, got {:?}",
            statuses
        );
    }

    #[test]
    fn vertex_on_surface_invalid_when_off() {
        let (mut reg, v0, _v1) = build_test_reg();
        // Move vertex off the Z=0 plane
        if let Some(v) = reg.vertices.get_mut(v0) {
            v.position = PVec3::new(0.0, 0.0, 999.0);
        }
        let statuses = check_vertex_on_surfaces(v0, &reg);
        assert!(
            statuses.contains(&CheckStatus::InvalidPointOnSurface),
            "vertex off surface plane should be flagged"
        );
    }
}
