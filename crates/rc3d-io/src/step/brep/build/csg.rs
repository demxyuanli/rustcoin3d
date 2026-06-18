//! CSG primitive → B-Rep conversion for STEP import.
//!
//! Converts CSG primitive entities (BLOCK, RIGHT_CIRCULAR_CYLINDER, etc.)
//! into B-Rep faces usable by the standard build pipeline.

use rc3d_core::math::{Real, PVec3};
use crate::step::parser::EntityIndex;
use crate::step::entity_geom;
use crate::step::topology::resolve_placement;
use rc3d_shape::BRepStore;
use rc3d_shape::topo::*;
use rc3d_shape::geom::{CurveGeom, SurfaceGeom, plane_tangent_basis, build_ortho_axes};
use rc3d_shape::geom::curve2d::Curve2d;

/// Build a B-Rep solid from a CSG primitive entity.
/// Returns the ShellKey if successful, None if the entity type is unknown or unsupported.
pub fn build_csg_primitive(
    entity_id: u64,
    entities: &EntityIndex,
    reg: &mut BRepStore,
) -> Option<ShellKey> {
    let record = entities.get(&entity_id)?;
    match record.name.as_str() {
        "BLOCK" => build_block(entity_id, entities, reg),
        "RIGHT_CIRCULAR_CYLINDER" => build_cylinder(entity_id, entities, reg),
        "RIGHT_CIRCULAR_CONE" => build_cone(entity_id, entities, reg),
        "SPHERE" => build_sphere(entity_id, entities, reg),
        "TORUS" => build_torus(entity_id, entities, reg),
        _ => {
            log::warn!("[CSG] Unknown CSG primitive type: {}", record.name);
            None
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Compute UV coordinates of a 3D point on a plane surface.
fn point_to_plane_uv(point: PVec3, origin: PVec3, u_dir: PVec3, v_dir: PVec3) -> (Real, Real) {
    let rel = point - origin;
    (rel.dot(u_dir), rel.dot(v_dir))
}

/// Create a rectangular planar face with 4 edges.
///
/// `corners` must be 4 points in CCW order when viewed from the outside
/// (normal pointing outward). `uv` must be the corresponding UV coordinates.
fn add_rectangular_face(
    reg: &mut BRepStore,
    face_origin: PVec3,
    face_normal: PVec3,
    u_dir: PVec3,
    corners: &[PVec3; 4],
    uv: &[(Real, Real); 4],
    tol: Real,
) -> FaceKey {
    let (u, _v) = plane_tangent_basis(face_normal, u_dir);
    let surface = SurfaceGeom::Plane { origin: face_origin, normal: face_normal, u_dir: u };

    let face_key = reg.add_face(surface, tol);

    let mut edge_keys: Vec<(EdgeKey, Orientation)> = Vec::with_capacity(4);
    for i in 0..4 {
        let j = (i + 1) % 4;
        let v0 = reg.find_or_add_vertex(corners[i], tol);
        let v1 = reg.find_or_add_vertex(corners[j], tol);

        let chord = corners[j] - corners[i];
        let chord_len = chord.length();
        let dir = if chord_len > 1e-12 { chord / chord_len } else { u_dir };

        let curve = CurveGeom::Line { origin: corners[i], direction: dir };

        let uv0 = uv[i];
        let uv1 = uv[j];
        let pc = Curve2d::Line {
            origin: uv0,
            direction: (uv1.0 - uv0.0, uv1.1 - uv0.1),
        };

        let ek = reg.add_edge_with_pcurve(v0, v1, curve, tol, face_key, pc, true);
        let (v_lo, _v_hi) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        let orient = if v0 == v_lo {
            Orientation::Forward
        } else {
            Orientation::Reversed
        };
        edge_keys.push((ek, orient));
    }

    // Replace the empty outer wire with the real one
    let wire_key = reg.wires.insert(BRepWire { edges: edge_keys });
    if let Some(face) = reg.faces.get_mut(face_key) {
        face.outer_wire = wire_key;
    }

    face_key
}

// ── BLOCK ────────────────────────────────────────────────────────

/// BLOCK: rectangular box.
///
/// STEP: `BLOCK('name', placement, x, y, z)`
///   - placement: AXIS2_PLACEMENT_3D
///   - x, y, z: dimensions along placement X, Y, Z axes
fn build_block(entity_id: u64, entities: &EntityIndex, reg: &mut BRepStore) -> Option<ShellKey> {
    let record = entities.get(&entity_id)?;
    let tol = reg.tolerance.model;

    // Extract parameters
    let placement_id = entity_geom::nth_ref(&record.params, 1)?;
    let dx = entity_geom::nth_real(&record.params, 2)?;
    let dy = entity_geom::nth_real(&record.params, 3)?;
    let dz = entity_geom::nth_real(&record.params, 4)?;

    let (origin, x_dir, z_dir) = resolve_placement(placement_id, entities)?;
    let y_dir = z_dir.cross(x_dir).normalize();

    // 8 vertices
    let v000 = origin;
    let v100 = origin + dx * x_dir;
    let v110 = origin + dx * x_dir + dy * y_dir;
    let v010 = origin + dy * y_dir;
    let v001 = origin + dz * z_dir;
    let v101 = origin + dx * x_dir + dz * z_dir;
    let v111 = origin + dx * x_dir + dy * y_dir + dz * z_dir;
    let v011 = origin + dy * y_dir + dz * z_dir;

    // Compute X-faces
    let face_xp_origin = origin + dx * x_dir;
    let face_xp_normal = x_dir;
    let face_xp_u_dir = y_dir;
    let face_xp_v_dir = z_dir;

    // Compute Y-faces
    let face_yp_origin = origin + dy * y_dir;
    let face_yp_normal = y_dir;
    let face_yp_u_dir = z_dir;

    // Compute Z-faces
    let face_zp_origin = origin + dz * z_dir;
    let face_zp_normal = z_dir;
    let face_zp_u_dir = x_dir;

    let mut face_keys = Vec::new();

    // +X face: normal = x_dir, vertices v100 → v110 → v111 → v101
    {
        let corners = [v100, v110, v111, v101];
        let uv = [
            point_to_plane_uv(v100, face_xp_origin, face_xp_u_dir, face_xp_v_dir),
            point_to_plane_uv(v110, face_xp_origin, face_xp_u_dir, face_xp_v_dir),
            point_to_plane_uv(v111, face_xp_origin, face_xp_u_dir, face_xp_v_dir),
            point_to_plane_uv(v101, face_xp_origin, face_xp_u_dir, face_xp_v_dir),
        ];
        let fk = add_rectangular_face(reg, face_xp_origin, face_xp_normal, face_xp_u_dir, &corners, &uv, tol);
        face_keys.push((fk, Orientation::Forward));
    }

    // -X face: normal = -x_dir, vertices v010 → v000 → v001 → v011
    {
        let corners = [v010, v000, v001, v011];
        let uv = [
            point_to_plane_uv(v010, origin, y_dir, z_dir),
            point_to_plane_uv(v000, origin, y_dir, z_dir),
            point_to_plane_uv(v001, origin, y_dir, z_dir),
            point_to_plane_uv(v011, origin, y_dir, z_dir),
        ];
        let fk = add_rectangular_face(reg, origin, -x_dir, y_dir, &corners, &uv, tol);
        face_keys.push((fk, Orientation::Forward));
    }

    // +Y face: normal = y_dir, vertices v110 → v010 → v011 → v111
    {
        let zp_v_dir = x_dir.normalize();
        let corners = [v110, v010, v011, v111];
        let uv = [
            point_to_plane_uv(v110, face_yp_origin, face_yp_u_dir, zp_v_dir),
            point_to_plane_uv(v010, face_yp_origin, face_yp_u_dir, zp_v_dir),
            point_to_plane_uv(v011, face_yp_origin, face_yp_u_dir, zp_v_dir),
            point_to_plane_uv(v111, face_yp_origin, face_yp_u_dir, zp_v_dir),
        ];
        let fk = add_rectangular_face(reg, face_yp_origin, face_yp_normal, face_yp_u_dir, &corners, &uv, tol);
        face_keys.push((fk, Orientation::Forward));
    }

    // -Y face: normal = -y_dir, vertices v000 → v100 → v101 → v001
    {
        let zp_v_dir = x_dir.normalize();
        let corners = [v000, v100, v101, v001];
        let uv = [
            point_to_plane_uv(v000, origin, z_dir, zp_v_dir),
            point_to_plane_uv(v100, origin, z_dir, zp_v_dir),
            point_to_plane_uv(v101, origin, z_dir, zp_v_dir),
            point_to_plane_uv(v001, origin, z_dir, zp_v_dir),
        ];
        let fk = add_rectangular_face(reg, origin, -y_dir, z_dir, &corners, &uv, tol);
        face_keys.push((fk, Orientation::Forward));
    }

    // +Z face: normal = z_dir, vertices v001 → v101 → v111 → v011
    {
        let corners = [v001, v101, v111, v011];
        let uv = [
            point_to_plane_uv(v001, face_zp_origin, face_zp_u_dir, y_dir),
            point_to_plane_uv(v101, face_zp_origin, face_zp_u_dir, y_dir),
            point_to_plane_uv(v111, face_zp_origin, face_zp_u_dir, y_dir),
            point_to_plane_uv(v011, face_zp_origin, face_zp_u_dir, y_dir),
        ];
        let fk = add_rectangular_face(reg, face_zp_origin, face_zp_normal, face_zp_u_dir, &corners, &uv, tol);
        face_keys.push((fk, Orientation::Forward));
    }

    // -Z face: normal = -z_dir, vertices v000 → v010 → v110 → v100
    {
        let corners = [v000, v010, v110, v100];
        let uv = [
            point_to_plane_uv(v000, origin, x_dir, y_dir),
            point_to_plane_uv(v010, origin, x_dir, y_dir),
            point_to_plane_uv(v110, origin, x_dir, y_dir),
            point_to_plane_uv(v100, origin, x_dir, y_dir),
        ];
        let fk = add_rectangular_face(reg, origin, -z_dir, x_dir, &corners, &uv, tol);
        face_keys.push((fk, Orientation::Forward));
    }

    let shell_key = reg.shells.insert(BRepShell {
        faces: face_keys,
        closed: true,
        step_id: Some(entity_id),
    });

    Some(shell_key)
}

// ── CYLINDER ─────────────────────────────────────────────────────

/// RIGHT_CIRCULAR_CYLINDER: right circular cylinder.
///
/// STEP: `RIGHT_CIRCULAR_CYLINDER('name', placement, height, radius)`
///   - placement: AXIS2_PLACEMENT_3D (Z axis = cylinder axis)
///   - height: along placement +Z
///   - radius: cylinder radius
fn build_cylinder(entity_id: u64, entities: &EntityIndex, reg: &mut BRepStore) -> Option<ShellKey> {
    let record = entities.get(&entity_id)?;
    let tol = reg.tolerance.model;

    let placement_id = entity_geom::nth_ref(&record.params, 1)?;
    let height = entity_geom::nth_real(&record.params, 2)?;
    let radius = entity_geom::nth_real(&record.params, 3)?;

    let (origin, _x_dir, axis) = resolve_placement(placement_id, entities)?;
    let (cx_dir, _cy_dir) = build_ortho_axes(axis);

    // Points at the seam (θ = 0, the x_dir direction)
    let bottom_center = origin;
    let top_center = origin + height * axis;
    let seam_bottom = bottom_center + radius * cx_dir;
    let seam_top = top_center + radius * cx_dir;

    let v_seam_bottom = reg.find_or_add_vertex(seam_bottom, tol);
    let v_seam_top = reg.find_or_add_vertex(seam_top, tol);

    let mut face_keys = Vec::new();

    // ── Cylindrical side face ──────────────────────────────────
    {
        let surface = SurfaceGeom::cylinder(bottom_center, axis, radius);
        let face_key = reg.add_face(surface, tol);

        // Bottom circular edge: u 0→2π at v=0
        let curve_bottom = CurveGeom::circle(bottom_center, axis, radius);
        let pc_bottom = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (std::f64::consts::TAU, 0.0),
        };
        let ek_bottom = reg.add_seam_edge(
            v_seam_bottom, v_seam_bottom,
            curve_bottom, tol, face_key, pc_bottom, true,
        );

        // Seam edge up: v 0→height at u=2π
        let curve_seam_up = CurveGeom::Line { origin: seam_bottom, direction: axis };
        let pc_seam_up = Curve2d::Line {
            origin: (std::f64::consts::TAU, 0.0),
            direction: (0.0, height),
        };
        let ek_seam = reg.add_edge_with_pcurve(
            v_seam_bottom, v_seam_top,
            curve_seam_up, tol, face_key, pc_seam_up, true,
        );

        // Top circular edge: u 2π→0 at v=height (reversed orientation)
        let curve_top = CurveGeom::circle(top_center, axis, radius);
        let pc_top = Curve2d::Line {
            origin: (0.0, height),
            direction: (std::f64::consts::TAU, 0.0),
        };
        let ek_top = reg.add_seam_edge(
            v_seam_top, v_seam_top,
            curve_top, tol, face_key, pc_top, true,
        );

        // Wire: bottom (Fwd), seam_up (Fwd), top (Rev), seam_down (Rev = reuse ek_seam reversed)
        let wire_edges = vec![
            (ek_bottom, Orientation::Forward),
            (ek_seam, Orientation::Forward),
            (ek_top, Orientation::Reversed),
            (ek_seam, Orientation::Reversed),
        ];
        let wire_key = reg.wires.insert(BRepWire { edges: wire_edges });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = wire_key;
            face.seam_edges.push(ek_seam);
        }
        face_keys.push((face_key, Orientation::Forward));
    }

    // ── Bottom cap face ────────────────────────────────────────
    {
        let (u_dir, _v_dir) = plane_tangent_basis(-axis, cx_dir);
        let surface = SurfaceGeom::Plane { origin: bottom_center, normal: -axis, u_dir };
        let face_key = reg.add_face(surface, tol);

        let curve_cap = CurveGeom::circle(bottom_center, -axis, radius);
        let pc_cap = Curve2d::Circle { center: (0.0, 0.0), radius };
        let ek_cap = reg.add_seam_edge(
            v_seam_bottom, v_seam_bottom,
            curve_cap, tol, face_key, pc_cap, true,
        );

        let wire_key = reg.wires.insert(BRepWire {
            edges: vec![(ek_cap, Orientation::Forward)],
        });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = wire_key;
        }
        face_keys.push((face_key, Orientation::Forward));
    }

    // ── Top cap face ───────────────────────────────────────────
    {
        let (u_dir, _v_dir) = plane_tangent_basis(axis, cx_dir);
        let surface = SurfaceGeom::Plane { origin: top_center, normal: axis, u_dir };
        let face_key = reg.add_face(surface, tol);

        let curve_cap = CurveGeom::circle(top_center, axis, radius);
        let pc_cap = Curve2d::Circle { center: (0.0, 0.0), radius };
        let ek_cap = reg.add_seam_edge(
            v_seam_top, v_seam_top,
            curve_cap, tol, face_key, pc_cap, true,
        );

        let wire_key = reg.wires.insert(BRepWire {
            edges: vec![(ek_cap, Orientation::Forward)],
        });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = wire_key;
        }
        face_keys.push((face_key, Orientation::Forward));
    }

    let shell_key = reg.shells.insert(BRepShell {
        faces: face_keys,
        closed: true,
        step_id: Some(entity_id),
    });

    Some(shell_key)
}

// ── CONE (stub) ──────────────────────────────────────────────────

/// RIGHT_CIRCULAR_CONE: stub — not yet implemented.
fn build_cone(_entity_id: u64, _entities: &EntityIndex, _reg: &mut BRepStore) -> Option<ShellKey> {
    log::info!("[CSG] RIGHT_CIRCULAR_CONE not yet implemented");
    None
}

// ── SPHERE (stub) ────────────────────────────────────────────────

/// SPHERE: stub — not yet implemented.
fn build_sphere(_entity_id: u64, _entities: &EntityIndex, _reg: &mut BRepStore) -> Option<ShellKey> {
    log::info!("[CSG] SPHERE not yet implemented");
    None
}

// ── TORUS (stub) ────────────────────────────────────────────────

/// TORUS: stub — not yet implemented.
fn build_torus(_entity_id: u64, _entities: &EntityIndex, _reg: &mut BRepStore) -> Option<ShellKey> {
    log::info!("[CSG] TORUS not yet implemented");
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_shape::BRepStore;

    fn make_store() -> BRepStore {
        BRepStore::with_tolerance(rc3d_shape::ToleranceContext::from_model(1e-6))
    }

    #[test]
    fn test_build_block_unit() {
        // Build a unit cube at origin with identity placement
        // We need an EntityIndex with a BLOCK entity and its dependencies.
        // For now, test that the function signature compiles and the helper works.
        let mut reg = make_store();
        let tol = reg.tolerance.model;

        // Create a rectangular face manually via the helper
        let _origin = PVec3::new(0.0, 0.0, 0.0);
        let corners = [
            PVec3::new(1.0, 0.0, 0.0),
            PVec3::new(1.0, 1.0, 0.0),
            PVec3::new(1.0, 1.0, 1.0),
            PVec3::new(1.0, 0.0, 1.0),
        ];
        let uv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let fk = add_rectangular_face(
            &mut reg,
            PVec3::new(1.0, 0.0, 0.0), // face origin
            PVec3::X,                    // normal
            PVec3::Y,                    // u_dir
            &corners,
            &uv,
            tol,
        );

        let face = reg.faces.get(fk).unwrap();
        assert!(matches!(face.surface, SurfaceGeom::Plane { .. }));
        let wire = reg.wires.get(face.outer_wire).unwrap();
        assert_eq!(wire.edges.len(), 4);

        // Each edge should have a PCurve for this face
        for (ek, _) in &wire.edges {
            let edge = reg.edges.get(*ek).unwrap();
            assert!(edge.pcurves.contains_key(&fk), "Edge missing PCurve");
        }
    }

    #[test]
    fn test_point_to_plane_uv() {
        let origin = PVec3::new(1.0, 0.0, 0.0);
        let u_dir = PVec3::Y;
        let v_dir = PVec3::Z;
        let p = PVec3::new(1.0, 2.0, 3.0);
        let (u, v) = point_to_plane_uv(p, origin, u_dir, v_dir);
        assert!((u - 2.0).abs() < 1e-10);
        assert!((v - 3.0).abs() < 1e-10);
    }
}
