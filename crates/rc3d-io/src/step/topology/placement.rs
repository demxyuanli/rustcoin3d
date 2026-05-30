//! Placement and direction resolution.
use super::super::parser::EntityIndex;
use super::super::value::StepValue;
use super::helpers::*;
use rc3d_core::math::Vec3;

pub fn resolve_point(point_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    let record = entities.get(&point_id)?;
    if record.name != "CARTESIAN_POINT" && record.name != "VERTEX_POINT" {
        return None;
    }
    if record.name == "VERTEX_POINT" {
        let inner = nth_ref(&record.params, 1)?;
        return resolve_point(inner, entities);
    }
    // CARTESIAN_POINT args: (name, (x, y, z))
    let coords = nth_list_params(&record.params, 1)?;
    if coords.len() < 3 { return None; }
    Some(Vec3::new(
        coords[0].as_real()? as f32,
        coords[1].as_real()? as f32,
        coords[2].as_real()? as f32,
    ))
}

/// AXIS1_PLACEMENT: rotation axis as (origin, unit direction).
pub fn resolve_axis1_placement(place_id: u64, entities: &EntityIndex) -> Option<(Vec3, Vec3)> {
    let record = entities.get(&place_id)?;
    if record.name != "AXIS1_PLACEMENT" {
        return None;
    }
    let origin_id = nth_ref(&record.params, 1)?;
    let axis_id = nth_ref(&record.params, 2)?;
    let origin = resolve_point(origin_id, entities)?;
    let axis = resolve_direction(axis_id, entities).unwrap_or(Vec3::Z);
    Some((origin, axis.normalize()))
}

/// Sweep/revolution axis from AXIS1_PLACEMENT or AXIS2_PLACEMENT_3D (Z axis).
pub fn resolve_sweep_axis(place_id: u64, entities: &EntityIndex) -> Option<(Vec3, Vec3)> {
    if let Some((origin, axis)) = resolve_axis1_placement(place_id, entities) {
        return Some((origin, axis));
    }
    resolve_placement(place_id, entities).map(|(origin, _, z)| (origin, z.normalize()))
}

pub fn resolve_placement(point_id: u64, entities: &EntityIndex) -> Option<(Vec3, Vec3, Vec3)> {
    let record = entities.get(&point_id)?;
    if record.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let origin_id = nth_ref(&record.params, 1)?;
    let axis_id = nth_ref(&record.params, 2)?;
    let refdir_id = nth_ref(&record.params, 3);

    let origin = resolve_point(origin_id, entities)?;
    let axis = resolve_direction(axis_id, entities).unwrap_or(Vec3::Z);
    let ref_dir = refdir_id.and_then(|id| resolve_direction(id, entities)).unwrap_or(Vec3::X);

    // Gram-Schmidt orthogonalization
    let z = axis.normalize();
    // Project ref_dir onto z's perpendicular plane
    let x_raw = ref_dir - z * ref_dir.dot(z);
    let x = if x_raw.length() > 1e-10 {
        x_raw.normalize()
    } else {
        // ref_dir is parallel to z, pick an arbitrary perpendicular X
        Vec3::Y.cross(z).normalize()
    };
    let _y = z.cross(x).normalize();

    Some((origin, x, z))
}

pub fn resolve_direction(dir_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    let record = entities.get(&dir_id)?;
    if record.name != "DIRECTION" { return None; }
    let coords = nth_list_params(&record.params, 1)?;
    if coords.len() < 3 { return None; }
    let v = Vec3::new(
        coords[0].as_real()? as f32,
        coords[1].as_real()? as f32,
        coords[2].as_real()? as f32,
    );
    let len = v.length();
    if len > 1e-10 { Some(v / len) } else { None }
}

/// Public direction resolution for use by geom and other modules.
pub fn resolve_direction_public(dir_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    resolve_direction(dir_id, entities)
}
