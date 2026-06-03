//! B-Rep fillet and chamfer operations (OCC BRepFilletAPI subset).
//!
//! Phase 3: Framework only — constant radius fillet on simple edges.
//! Full implementation deferred to Phase 4+ (requires robust edge chain
//! detection, variable radius support, and spring/back surface fitting).

use crate::step::brep::registry::BRepStore;
use crate::step::brep::topo::{EdgeKey, FaceKey};

/// Result of a fillet or chamfer operation.
#[derive(Debug)]
pub struct FilletResult {
    /// New cylindrical/toroidal faces created by the fillet.
    pub new_faces: Vec<FaceKey>,
    /// Adjacent faces that were trimmed to accommodate the fillet.
    pub modified_faces: Vec<FaceKey>,
    /// Original edge that was replaced.
    pub removed_edge: EdgeKey,
}

/// Fillet parameters for a single edge.
#[derive(Debug, Clone)]
pub struct FilletParams {
    /// Edge to fillet.
    pub edge: EdgeKey,
    /// Fillet radius (constant).
    pub radius: f32,
}

/// Chamfer parameters for a single edge.
#[derive(Debug, Clone)]
pub struct ChamferParams {
    /// Edge to chamfer.
    pub edge: EdgeKey,
    /// Chamfer distance from the edge on face A side.
    pub distance_a: f32,
    /// Chamfer distance from the edge on face B side.
    pub distance_b: f32,
}

/// Apply a constant-radius fillet to a single edge.
///
/// # Limitations (Phase 3)
/// - Only supports edges between two planar faces
/// - Only supports convex edges (dihedral angle < 180°)
/// - Does not handle edge chains (must fillet one edge at a time)
/// - No variable radius support
///
/// # Algorithm (when implemented)
/// 1. Find the two faces sharing this edge
/// 2. Verify both are planar (or one planar + one cylindrical)
/// 3. Compute the fillet cylinder/torus axis and center
/// 4. Create the fillet surface (cylindrical or toroidal)
/// 5. Trim adjacent faces to accommodate the fillet
/// 6. Add new edges and update wire connectivity
pub fn constant_radius_fillet(
    edge: EdgeKey,
    radius: f32,
    reg: &mut BRepStore,
) -> Result<FilletResult, String> {
    // Validate inputs
    if radius <= 0.0 {
        return Err(format!("Fillet radius must be positive, got {}", radius));
    }

    if !reg.edges.contains_key(edge) {
        return Err(format!("Edge {:?} not found in registry", edge));
    }

    // Find the two faces sharing this edge
    let face_keys: Vec<FaceKey> = reg.edge_to_faces
        .get(&edge)
        .cloned()
        .unwrap_or_default();

    if face_keys.len() != 2 {
        return Err(format!(
            "Edge {:?} must be shared by exactly 2 faces, found {}",
            edge, face_keys.len()
        ));
    }

    // TODO: Full fillet implementation
    // 1. Get face surfaces and verify planarity
    // 2. Compute dihedral angle
    // 3. Create fillet cylinder surface
    // 4. Trim adjacent faces
    // 5. Create new edges and vertices
    // 6. Update wire connectivity

    Err(format!(
        "constant_radius_fillet: not yet implemented (Phase 3 framework). \
         Edge {:?}, radius {}, faces {:?}",
        edge, radius, face_keys
    ))
}

/// Apply a chamfer to a single edge.
///
/// # Limitations (Phase 3)
/// - Only supports edges between two planar faces
/// - Symmetric and asymmetric chamfers supported via distance_a/distance_b
///
/// # Algorithm (when implemented)
/// 1. Find the two faces sharing this edge
/// 2. Compute chamfer plane from distance_a and distance_b
/// 3. Create chamfer face
/// 4. Trim adjacent faces
/// 5. Update topology
pub fn chamfer_edge(
    edge: EdgeKey,
    distance: f32,
    reg: &mut BRepStore,
) -> Result<FilletResult, String> {
    if distance <= 0.0 {
        return Err(format!("Chamfer distance must be positive, got {}", distance));
    }

    if !reg.edges.contains_key(edge) {
        return Err(format!("Edge {:?} not found in registry", edge));
    }

    Err(format!(
        "chamfer_edge: not yet implemented (Phase 3 framework). Edge {:?}",
        edge
    ))
}

/// Apply fillets to multiple edges with the same radius.
/// Returns results for each edge (success or error).
pub fn fillet_edges(
    params: &[FilletParams],
    reg: &mut BRepStore,
) -> Vec<Result<FilletResult, String>> {
    params.iter()
        .map(|p| constant_radius_fillet(p.edge, p.radius, reg))
        .collect()
}

/// Apply chamfers to multiple edges.
pub fn chamfer_edges(
    params: &[ChamferParams],
    reg: &mut BRepStore,
) -> Vec<Result<FilletResult, String>> {
    params.iter()
        .map(|p| chamfer_edge(p.edge, p.distance_a, reg))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::registry::BRepStore;

    #[test]
    fn test_fillet_api_exists() {
        let mut reg = BRepStore::new();
        // Create a dummy edge
        use crate::step::brep::topo::*;
        use crate::step::brep::geom::CurveGeom;
        use rc3d_core::math::Vec3;

        let v1 = reg.find_or_add_vertex(Vec3::ZERO, 1e-6);
        let v2 = reg.find_or_add_vertex(Vec3::X, 1e-6);
        let ek = reg.edges.insert(BRepEdge {
            curve: CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X },
            tolerance: 1e-6,
            v_low: v1,
            v_high: v2,
            pcurves: std::collections::HashMap::new(),
        });

        let result = constant_radius_fillet(ek, 1.0, &mut reg);
        assert!(result.is_err(), "Fillet should return error (not implemented)");
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("not yet implemented") || err_msg.contains("must be shared"),
            "Error should indicate limitation: {}", err_msg
        );
    }

    #[test]
    fn test_fillet_rejects_negative_radius() {
        let mut reg = BRepStore::new();
        use crate::step::brep::topo::*;
        let dummy_key = EdgeKey::from(slotmap::KeyData::from_ffi(0x1234));
        let result = constant_radius_fillet(dummy_key, -1.0, &mut reg);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("positive"));
    }

    #[test]
    fn test_chamfer_api_exists() {
        let mut reg = BRepStore::new();
        let dummy_key = EdgeKey::from(slotmap::KeyData::from_ffi(0x5678));
        let result = chamfer_edge(dummy_key, 1.0, &mut reg);
        assert!(result.is_err(), "Chamfer should return error (not implemented)");
    }

    #[test]
    fn test_fillet_edges_batch() {
        let mut reg = BRepStore::new();
        let params = vec![
            FilletParams { edge: EdgeKey::from(slotmap::KeyData::from_ffi(1)), radius: 0.5 },
            FilletParams { edge: EdgeKey::from(slotmap::KeyData::from_ffi(2)), radius: 0.5 },
        ];
        let results = fillet_edges(&params, &mut reg);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_err()), "All should fail (not implemented)");
    }
}
