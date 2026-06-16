//! B-Rep fillet and chamfer operations (OCC BRepFilletAPI subset).
//!
//! Phase 3: constant-radius fillet on planar edges.
//! Full implementation deferred to Phase 4+ (requires robust edge chain
//! detection, variable radius support, and spring/back surface fitting).

use rc3d_core::math::Real;
use std::collections::HashMap;

use rc3d_shape::BRepStore;
use rc3d_shape::topo::{
    BRepEdge, BRepFace, BRepWire, EdgeKey, FaceKey, Orientation, VertexKey,
};
use crate::step::brep::geom::{CurveGeom, SurfaceGeom};

/// Return (lo, hi) so that lo < hi.
fn ordered_pair(a: VertexKey, b: VertexKey) -> (VertexKey, VertexKey) {
    if a < b { (a, b) } else { (b, a) }
}

/// Result of a fillet or chamfer operation.
#[derive(Debug)]
pub struct FilletResult {
    /// New cylindrical/toroidal faces created by the fillet.
    pub new_faces: Vec<FaceKey>,
    /// Adjacent faces that the fillet replaces along.
    /// NOTE (Phase 3): These faces' wires are NOT yet modified — face trimming
    /// is deferred to Phase 4. The caller must perform wire splitting if needed.
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
    pub radius: Real,
    /// `true` for convex (external) edges, `false` for concave (re-entrant) edges.
    pub convex: bool,
}

/// Chamfer parameters for a single edge.
#[derive(Debug, Clone)]
pub struct ChamferParams {
    /// Edge to chamfer.
    pub edge: EdgeKey,
    /// Chamfer distance from the edge on face A side.
    pub distance_a: Real,
    /// Chamfer distance from the edge on face B side.
    pub distance_b: Real,
}

/// Apply a constant-radius fillet to a single edge.
///
/// # Limitations (Phase 3)
/// - Only supports edges between two planar faces
/// - Does not handle edge chains (must fillet one edge at a time)
/// - No variable radius support
///
/// # Algorithm
/// 1. Find the two faces sharing this edge
/// 2. Verify both are planar
/// 3. Compute the fillet cylinder axis and center
/// 4. Create the fillet surface (cylindrical)
/// 5. Create contact curve edges and fillet face
/// 6. Return the new and modified face keys
pub fn constant_radius_fillet(
    edge: EdgeKey,
    radius: Real,
    convex: bool,
    reg: &mut BRepStore,
) -> Result<FilletResult, String> {
    // Validate inputs
    if radius <= 0.0 {
        return Err(format!("Fillet radius must be positive, got {}", radius));
    }

    // Find the two faces sharing this edge
    let faces = reg
        .edge_to_faces
        .get(&edge)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    if faces.len() != 2 {
        return Err(format!(
            "Edge {:?} must be shared by exactly 2 faces, found {}",
            edge,
            faces.len()
        ));
    }

    let fk_a = faces[0];
    let fk_b = faces[1];
    let face_a = reg.faces.get(fk_a).ok_or("Face A not found")?;
    let face_b = reg.faces.get(fk_b).ok_or("Face B not found")?;

    // Step 1: Get edge geometry
    let brep_edge = reg.edges.get(edge)
        .ok_or_else(|| format!("Edge {:?} not found in registry", edge))?;
    let edge_mid = brep_edge.curve.d0(0.5);
    let raw_tangent = brep_edge.curve.d1(0.5);
    if raw_tangent.length() < 1e-10 {
        return Err("Degenerate edge (zero tangent)".to_string());
    }
    let edge_dir = raw_tangent.normalize();

    // Step 2: Get face normals — must be planar for Phase 3
    let (n1, n2) = match (&face_a.surface, &face_b.surface) {
        (
            SurfaceGeom::Plane { normal: n1, .. },
            SurfaceGeom::Plane { normal: n2, .. },
        ) => (*n1, *n2),
        _ => return Err("Fillet currently only supports planar faces".to_string()),
    };

    // Step 3: Dihedral angle
    let cos_theta = n1.dot(n2).max(-1.0).min(1.0);
    let theta = cos_theta.acos();
    let half = theta * 0.5;
    if half.sin() < 1e-6 {
        return Err("Faces are coplanar — cannot create fillet".to_string());
    }

    // Step 4: Bisector direction pointing into the solid
    let mut bisector = (n1 + n2).normalize();
    if bisector.length() < 1e-6 {
        // Opposite faces — pick arbitrary inside direction
        bisector = n1;
    }
    // For concave edges the fillet rolls inward — reverse the offset
    if !convex {
        bisector = -bisector;
    }

    // Step 5: Cylinder center line offset from edge midpoint
    let offset_dist = radius / half.sin();
    let center = edge_mid + bisector * offset_dist;

    // Create fillet cylinder surface
    let fillet_surface = SurfaceGeom::cylinder(center, edge_dir, radius);

    // Step 6: Compute contact line points
    // Derive extension from actual edge length so contact lines work at any model scale
    let edge_len = reg.vertices.get(brep_edge.v_low)
        .zip(reg.vertices.get(brep_edge.v_high))
        .map(|(a, b)| (b.position - a.position).length())
        .unwrap_or(1.0);
    let ext = edge_len * 2.0; // extend past both ends by 1× edge length
    let perp_a = edge_dir.cross(n1).normalize();
    let perp_b = edge_dir.cross(n2).normalize();
    let contact_dist = radius / half.tan();
    let edge_ext = edge_dir * ext;

    let base_a = edge_mid + perp_a * contact_dist;
    let base_b = edge_mid - perp_b * contact_dist;
    let cp_a1 = base_a - edge_ext;
    let cp_a2 = base_a + edge_ext;
    let cp_b1 = base_b - edge_ext;
    let cp_b2 = base_b + edge_ext;

    // Step 7: Create vertices
    let tolerance = face_a.tolerance.max(face_b.tolerance);
    let v_a1 = reg.find_or_add_vertex(cp_a1, tolerance);
    let v_a2 = reg.find_or_add_vertex(cp_a2, tolerance);
    let v_b1 = reg.find_or_add_vertex(cp_b1, tolerance);
    let v_b2 = reg.find_or_add_vertex(cp_b2, tolerance);

    // Create contact curve edges
    let line_dir = edge_ext * 2.0;
    let contact_curve_a = CurveGeom::Line {
        origin: cp_a1,
        direction: line_dir,
    };
    let contact_curve_b = CurveGeom::Line {
        origin: cp_b1,
        direction: line_dir,
    };

    let (v_lo_a, v_hi_a) = ordered_pair(v_a1, v_a2);
    let (v_lo_b, v_hi_b) = ordered_pair(v_b1, v_b2);

    let ek_a = reg.edges.insert(BRepEdge {
        curve: contact_curve_a,
        tolerance,
        v_low: v_lo_a,
        v_high: v_hi_a,
        t_min: 0.0,
        t_max: 1.0,
        pcurves: HashMap::new(),
    });
    let ek_b = reg.edges.insert(BRepEdge {
        curve: contact_curve_b,
        tolerance,
        v_low: v_lo_b,
        v_high: v_hi_b,
        t_min: 0.0,
        t_max: 1.0,
        pcurves: HashMap::new(),
    });

    // Create fillet face wire
    let fillet_wire = reg.wires.insert(BRepWire {
        edges: vec![(ek_a, Orientation::Forward), (ek_b, Orientation::Reversed)],
    });

    // Create fillet face
    let fillet_fk = reg.faces.insert(BRepFace {
        surface: fillet_surface,
        outer_wire: fillet_wire,
        inner_wires: vec![],
        same_sense: true,
        tolerance,
        seam_edges: vec![],
        color: None,
        degenerated_edges: vec![],
    });

    // Update edge_to_faces for new topology
    reg.edge_to_faces.insert(ek_a, vec![fillet_fk, fk_a]);
    reg.edge_to_faces.insert(ek_b, vec![fillet_fk, fk_b]);
    reg.edge_to_faces.remove(&edge);

    Ok(FilletResult {
        new_faces: vec![fillet_fk],
        modified_faces: vec![fk_a, fk_b],
        removed_edge: edge,
    })
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
    distance: Real,
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
    params
        .iter()
        .map(|p| constant_radius_fillet(p.edge, p.radius, p.convex, reg))
        .collect()
}

/// Apply chamfers to multiple edges.
pub fn chamfer_edges(
    params: &[ChamferParams],
    reg: &mut BRepStore,
) -> Vec<Result<FilletResult, String>> {
    params
        .iter()
        .map(|p| chamfer_edge(p.edge, p.distance_a, reg))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_shape::BRepStore;
    use rc3d_shape::topo::*;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use rc3d_core::math::PVec3;

    /// Helper: create two planar faces sharing an edge, with `edge_to_faces` populated.
    fn setup_two_planar_faces(
        reg: &mut BRepStore,
        n1: PVec3,
        n2: PVec3,
        edge_start: PVec3,
        edge_end: PVec3,
    ) -> (FaceKey, FaceKey, EdgeKey) {
        let v0 = reg.find_or_add_vertex(edge_start, 1e-6);
        let v1 = reg.find_or_add_vertex(edge_end, 1e-6);

        let curve = CurveGeom::Line {
            origin: edge_start,
            direction: edge_end - edge_start,
        };

        let face_keys: Vec<FaceKey> = [(n1, PVec3::Y), (n2, PVec3::X)]
            .iter()
            .map(|(normal, u_dir)| {
                reg.add_face(
                    SurfaceGeom::Plane {
                        origin: edge_start,
                        normal: *normal,
                        u_dir: *u_dir,
                    },
                    1e-6,
                )
            })
            .collect();

        let (v_lo, v_hi) = ordered_pair(v0, v1);
        let ek = reg.edges.insert(BRepEdge {
            curve,
            tolerance: 1e-6,
            v_low: v_lo,
            v_high: v_hi,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: HashMap::new(),
        });

        reg.edge_to_faces.insert(ek, vec![face_keys[0], face_keys[1]]);

        (face_keys[0], face_keys[1], ek)
    }

    #[test]
    fn test_fillet_api_exists() {
        let mut reg = BRepStore::new();

        // Two perpendicular planes sharing an edge along the Z axis:
        // Face A: normal = +X (plane x=0)
        // Face B: normal = +Y (plane y=0)
        // Edge: from (0,0,-10) to (0,0,10)
        let (fk_a, fk_b, ek) = setup_two_planar_faces(
            &mut reg,
            PVec3::X,
            PVec3::Y,
            PVec3::new(0.0, 0.0, -10.0),
            PVec3::new(0.0, 0.0, 10.0),
        );

        let result = constant_radius_fillet(ek, 1.0, true, &mut reg);
        assert!(result.is_ok(), "Fillet should succeed: {:?}", result.err());

        let fr = result.unwrap();
        assert_eq!(fr.new_faces.len(), 1, "One new fillet face should be created");
        assert!(fr.modified_faces.contains(&fk_a), "Face A should be in modified_faces");
        assert!(fr.modified_faces.contains(&fk_b), "Face B should be in modified_faces");
        assert_eq!(fr.removed_edge, ek, "Original edge should be recorded as removed");

        // Verify the fillet face has a cylinder surface with the correct radius
        let fillet_face = reg.faces.get(fr.new_faces[0]).unwrap();
        match &fillet_face.surface {
            SurfaceGeom::Cylinder { radius, .. } => {
                assert!((*radius - 1.0).abs() < 1e-5, "Radius should be 1.0, got {}", *radius);
            }
            other => panic!("Expected Cylinder surface, got {:?}", std::mem::discriminant(other)),
        }
    }

    #[test]
    fn test_fillet_rejects_negative_radius() {
        let mut reg = BRepStore::new();
        let dummy_key = EdgeKey::from(slotmap::KeyData::from_ffi(0x1234));
        let result = constant_radius_fillet(dummy_key, -1.0, true, &mut reg);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("positive"));
    }

    #[test]
    fn test_fillet_rejects_single_face_edge() {
        // Edge not shared by any faces should fail with "must be shared" error
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-6);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-6);
        let ek = reg.edges.insert(BRepEdge {
            curve: CurveGeom::Line {
                origin: PVec3::ZERO,
                direction: PVec3::X,
            },
            tolerance: 1e-6,
            v_low: v0,
            v_high: v1,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: std::collections::HashMap::new(),
        });
        // Don't add edge to edge_to_faces — it will be empty
        let result = constant_radius_fillet(ek, 1.0, true, &mut reg);
        assert!(result.is_err(), "Fillet should fail for edge with no faces");
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("must be shared"),
            "Error should indicate face count: {}",
            err_msg
        );
    }

    #[test]
    fn test_fillet_planar_90deg_geometry() {
        // Verify geometric correctness for a 90-degree planar fillet
        let mut reg = BRepStore::new();

        let (fk_a, fk_b, ek) = setup_two_planar_faces(
            &mut reg,
            PVec3::X,
            PVec3::Y,
            PVec3::new(0.0, 0.0, -10.0),
            PVec3::new(0.0, 0.0, 10.0),
        );

        let result = constant_radius_fillet(ek, 2.0, true, &mut reg);
        assert!(result.is_ok());

        let fr = result.unwrap();
        let fillet_face = reg.faces.get(fr.new_faces[0]).unwrap();

        // Verify cylinder surface
        match &fillet_face.surface {
            SurfaceGeom::Cylinder {
                origin,
                axis,
                radius,
                ..
            } => {
                // Radius should be 2.0
                assert!((*radius - 2.0).abs() < 1e-5, "Wrong radius: {}", *radius);

                // Axis should be parallel to Z (the edge direction)
                let axis_norm = axis.normalize();
                assert!(
                    (axis_norm.dot(PVec3::Z).abs() - 1.0).abs() < 1e-5,
                    "Cylinder axis should be parallel to Z, got {:?}",
                    axis_norm
                );

                // Center (origin) should be at (2.0, 2.0, 0.0) for the 90° case
                // bisector = (1,1,0)/sqrt(2), half=45°, offset = 2/sin(45) = 2.828
                // center = (0,0,0) + (0.707,0.707,0)*2.828 = (2.0, 2.0, 0.0)
                assert!((origin.x - 2.0).abs() < 1e-4, "Center x should be ~2.0, got {}", origin.x);
                assert!((origin.y - 2.0).abs() < 1e-4, "Center y should be ~2.0, got {}", origin.y);
                assert!(origin.z.abs() < 1e-4, "Center z should be ~0.0, got {}", origin.z);

                // Distance from cylinder axis to each face plane should equal radius
                let face_a = reg.faces.get(fk_a).unwrap();
                let face_b = reg.faces.get(fk_b).unwrap();
                if let (SurfaceGeom::Plane { origin: o1, normal: n1, .. },
                        SurfaceGeom::Plane { origin: o2, normal: n2, .. }) =
                    (&face_a.surface, &face_b.surface)
                {
                    let dist_a = (n1.dot(*origin - *o1)).abs();
                    let dist_b = (n2.dot(*origin - *o2)).abs();
                    assert!((dist_a - radius).abs() < 1e-4,
                        "Distance to face A should equal radius: {} vs {}", dist_a, radius);
                    assert!((dist_b - radius).abs() < 1e-4,
                        "Distance to face B should equal radius: {} vs {}", dist_b, radius);
                }
            }
            _ => panic!("Expected Cylinder surface"),
        }

        // Verify contact edges were created
        let fillet_wire = reg.wires.get(fillet_face.outer_wire).unwrap();
        assert_eq!(fillet_wire.edges.len(), 2, "Fillet wire should have 2 edges (contact curves)");
    }

    #[test]
    fn test_fillet_rejects_coplanar_faces() {
        // Two parallel/coplanar faces should be rejected
        let mut reg = BRepStore::new();

        let (_, _, ek) = setup_two_planar_faces(
            &mut reg,
            PVec3::X,
            PVec3::X, // Same normal — coplanar
            PVec3::new(0.0, 0.0, -10.0),
            PVec3::new(0.0, 0.0, 10.0),
        );

        let result = constant_radius_fillet(ek, 1.0, true, &mut reg);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("coplanar"));
    }

    #[test]
    fn test_fillet_rejects_non_planar_face() {
        // A cylinder face should be rejected (only planar supported)
        let mut reg = BRepStore::new();

        let (fk_a, _, ek) = setup_two_planar_faces(
            &mut reg,
            PVec3::X,
            PVec3::Y,
            PVec3::new(0.0, 0.0, -10.0),
            PVec3::new(0.0, 0.0, 10.0),
        );

        // Replace face A with a cylinder surface
        if let Some(face) = reg.faces.get_mut(fk_a) {
            face.surface = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 5.0);
        }

        let result = constant_radius_fillet(ek, 1.0, true, &mut reg);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("planar"));
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

        // Set up a valid edge + faces for the first param
        let (_, _, ek) = setup_two_planar_faces(
            &mut reg,
            PVec3::X,
            PVec3::Y,
            PVec3::new(0.0, 0.0, -10.0),
            PVec3::new(0.0, 0.0, 10.0),
        );

        let params = vec![
            FilletParams { edge: ek, radius: 0.5, convex: true },
            FilletParams {
                edge: EdgeKey::from(slotmap::KeyData::from_ffi(2)),
                radius: 0.5,
                convex: true,
            },
        ];
        let results = fillet_edges(&params, &mut reg);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok(), "First (valid edge) should succeed");
        assert!(results[1].is_err(), "Second (invalid edge) should fail");
    }
}
