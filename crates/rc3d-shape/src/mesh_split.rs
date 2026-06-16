//! Extract sub-meshes by per-face triangle ranges (face-color draw batches).

use rc3d_core::math::Real;
use std::collections::{HashMap, HashSet};

use crate::emit_plan::FaceMaterialGroup;
use crate::mesh::refiner::extract_face_mesh_with_map;
use crate::mesh_result::MeshResult;
use crate::topo::FaceKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceTriRange {
    pub first_tri: usize,
    pub tri_count: usize,
}

#[derive(Debug, Clone)]
pub struct FaceDrawBatch {
    pub color: [Real; 3],
    pub opacity: Real,
    pub mesh: MeshResult,
}

/// Merge triangle ranges from the given faces into one sub-mesh.
pub fn extract_mesh_tri_ranges(mesh: &MeshResult, ranges: &[FaceTriRange]) -> MeshResult {
    let mut out = MeshResult::default();
    for range in ranges {
        if range.tri_count == 0 {
            continue;
        }
        let start = range.first_tri * 4;
        let end = start + range.tri_count * 4;
        let (part, _) = extract_face_mesh_with_map(
            &mesh.vertices,
            &mesh.normals,
            &mesh.indices,
            start,
            end,
        );
        out.append_from(&part);
    }
    out
}

fn ranges_for_faces(
    face_tri_ranges: &HashMap<FaceKey, FaceTriRange>,
    face_keys: &[FaceKey],
) -> Vec<FaceTriRange> {
    face_keys
        .iter()
        .filter_map(|fk| face_tri_ranges.get(fk).copied())
        .filter(|r| r.tri_count > 0)
        .collect()
}

/// Build draw batches for label default material + per-face color overrides.
pub fn face_material_draw_batches(
    mesh: &MeshResult,
    face_tri_ranges: &HashMap<FaceKey, FaceTriRange>,
    face_split_viable: bool,
    groups: &[FaceMaterialGroup],
    default_color: [Real; 3],
    default_opacity: Real,
) -> Vec<FaceDrawBatch> {
    if !face_split_viable || groups.is_empty() || face_tri_ranges.is_empty() {
        return vec![FaceDrawBatch {
            color: default_color,
            opacity: default_opacity,
            mesh: mesh.clone(),
        }];
    }

    let mut colored: HashSet<FaceKey> = HashSet::new();
    for group in groups {
        for &fk in &group.face_keys {
            colored.insert(fk);
        }
    }

    let mut batches = Vec::new();

    let default_faces: Vec<FaceKey> = face_tri_ranges
        .keys()
        .copied()
        .filter(|fk| !colored.contains(fk))
        .collect();
    let default_ranges = ranges_for_faces(face_tri_ranges, &default_faces);
    if !default_ranges.is_empty() {
        let sub = extract_mesh_tri_ranges(mesh, &default_ranges);
        if !sub.vertices.is_empty() && !sub.indices.is_empty() {
            batches.push(FaceDrawBatch {
                color: default_color,
                opacity: default_opacity,
                mesh: sub,
            });
        }
    }

    for group in groups {
        let ranges = ranges_for_faces(face_tri_ranges, &group.face_keys);
        if ranges.is_empty() {
            continue;
        }
        let sub = extract_mesh_tri_ranges(mesh, &ranges);
        if sub.vertices.is_empty() || sub.indices.is_empty() {
            continue;
        }
        batches.push(FaceDrawBatch {
            color: group.color,
            opacity: default_opacity,
            mesh: sub,
        });
    }

    if batches.is_empty() {
        batches.push(FaceDrawBatch {
            color: default_color,
            opacity: default_opacity,
            mesh: mesh.clone(),
        });
    }

    batches
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::PVec3;

    fn square_mesh() -> MeshResult {
        MeshResult {
            vertices: vec![
                PVec3::ZERO,
                PVec3::X,
                PVec3::Y,
                PVec3::new(1.0, 1.0, 0.0),
            ],
            normals: vec![PVec3::Z; 4],
            indices: vec![0, 1, 2, -1, 1, 3, 2, -1],
        }
    }

    #[test]
    fn extract_single_face_range() {
        let mesh = square_mesh();
        let sub = extract_mesh_tri_ranges(&mesh, &[FaceTriRange { first_tri: 1, tri_count: 1 }]);
        assert_eq!(sub.indices.len(), 4);
        assert_eq!(sub.vertices.len(), 3);
    }

    #[test]
    fn face_material_batches_fallback_without_groups() {
        let mesh = square_mesh();
        let batches = face_material_draw_batches(
            &mesh,
            &HashMap::new(),
            true,
            &[],
            [0.5, 0.5, 0.5],
            1.0,
        );
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].mesh.indices.len(), mesh.indices.len());
    }
}
