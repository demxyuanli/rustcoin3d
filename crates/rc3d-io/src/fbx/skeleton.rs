use std::collections::HashMap;

use rc3d_core::math::Mat4;
use rc3d_scene::animation::{Joint, Skeleton, VertexSkinData};

use super::types::*;

/// Resolve skeleton data for a specific geometry.
/// Returns the skeleton and per-vertex skin data if the mesh is skinned.
pub fn resolve_skeleton(
    geo_id: i64,
    objects: &HashMap<i64, FbxObject>,
    connections: &[FbxConnection],
) -> Option<ResolvedSkeleton> {
    // Find the Skin deformer connected to this geometry
    let skin_id = find_skin_for_geometry(geo_id, objects, connections)?;
    let skin_clusters = match objects.get(&skin_id)? {
        FbxObject::Deformer(FbxDeformer::Skin { clusters }) => clusters.clone(),
        _ => return None,
    };

    // Collect all bones (LimbNode models) referenced by clusters
    let mut bone_ids: Vec<i64> = Vec::new();
    let mut bone_id_to_index: HashMap<i64, usize> = HashMap::new();

    for &cluster_id in &skin_clusters {
        let bone_id = match objects.get(&cluster_id) {
            Some(FbxObject::Deformer(FbxDeformer::Cluster { bone_id, .. })) => *bone_id,
            _ => continue,
        };
        if !bone_id_to_index.contains_key(&bone_id) {
            bone_id_to_index.insert(bone_id, bone_ids.len());
            bone_ids.push(bone_id);
        }
    }

    if bone_ids.is_empty() {
        return None;
    }

    // Build parent relationships: parent must also be a skinned bone (cluster target)
    let mut parent_map: HashMap<i64, i64> = HashMap::new();
    for conn in connections {
        if bone_id_to_index.contains_key(&conn.child) && bone_id_to_index.contains_key(&conn.parent) {
            parent_map.insert(conn.child, conn.parent);
        }
    }

    // Build joints
    let joint_count = bone_ids.len();
    let mut joints: Vec<Joint> = Vec::with_capacity(joint_count);

    for (idx, &bone_id) in bone_ids.iter().enumerate() {
        let model = match objects.get(&bone_id) {
            Some(FbxObject::Model(m)) => m,
            _ => {
                joints.push(Joint {
                    name: format!("bone_{idx}"),
                    parent: joint_count,
                    bind_transform: Mat4::IDENTITY,
                    inverse_bind_matrix: Mat4::IDENTITY,
                });
                continue;
            }
        };

        let parent_idx = parent_map
            .get(&bone_id)
            .and_then(|pid| bone_id_to_index.get(pid))
            .copied()
            .unwrap_or(joint_count);

        // Get inverse bind matrix from cluster
        let inverse_bind = find_inverse_bind_for_bone(bone_id, objects, &skin_clusters);

        joints.push(Joint {
            name: model.name.clone(),
            parent: parent_idx,
            bind_transform: model.local_transform,
            inverse_bind_matrix: inverse_bind.unwrap_or(Mat4::IDENTITY),
        });
    }

    let skeleton = Skeleton::new(joints);

    // Determine vertex count from geometry
    let vertex_count = match objects.get(&geo_id) {
        Some(FbxObject::Geometry(geo)) => geo.positions.len(),
        _ => return None,
    };

    // Build per-vertex skin data
    let mut skin_data = vec![VertexSkinData::empty(); vertex_count];

    for &cluster_id in &skin_clusters {
        let (bone_id, indices, weights) = match objects.get(&cluster_id) {
            Some(FbxObject::Deformer(FbxDeformer::Cluster { bone_id, indices, weights, .. })) => {
                (*bone_id, indices.clone(), weights.clone())
            }
            _ => continue,
        };

        let bone_index = match bone_id_to_index.get(&bone_id) {
            Some(&i) => i as u32,
            None => continue,
        };

        for (&vi, &w) in indices.iter().zip(weights.iter()) {
            let vi = vi as usize;
            if vi >= vertex_count {
                continue;
            }
            let w = w as f32;
            if w < 1e-6 {
                continue;
            }

            // Find the slot with the smallest weight and replace it
            let sd = &mut skin_data[vi];
            let min_idx = (0..4)
                .min_by(|&a, &b| sd.bone_weights[a].partial_cmp(&sd.bone_weights[b]).unwrap())
                .unwrap();
            if w > sd.bone_weights[min_idx] {
                sd.bone_indices[min_idx] = bone_index;
                sd.bone_weights[min_idx] = w;
            }
        }
    }

    // Normalize weights
    for sd in &mut skin_data {
        let sum: f32 = sd.bone_weights.iter().sum();
        if sum > 1e-6 {
            for w in &mut sd.bone_weights {
                *w /= sum;
            }
        }
    }

    Some(ResolvedSkeleton {
        skeleton,
        skin_data,
    })
}

fn find_skin_for_geometry(
    geo_id: i64,
    objects: &HashMap<i64, FbxObject>,
    connections: &[FbxConnection],
) -> Option<i64> {
    // Geometry -> Skin: find Deformer(Skin) connected to geometry
    for conn in connections {
        if conn.child == geo_id {
            if matches!(objects.get(&conn.parent), Some(FbxObject::Deformer(FbxDeformer::Skin { .. }))) {
                return Some(conn.parent);
            }
        }
    }
    // Also check: Skin connected as child of Geometry
    for conn in connections {
        if conn.parent == geo_id {
            if matches!(objects.get(&conn.child), Some(FbxObject::Deformer(FbxDeformer::Skin { .. }))) {
                return Some(conn.child);
            }
        }
    }
    None
}

/// Maps FBX model (bone) IDs to joint indices, in consistent order for animation sampling.
pub fn bone_id_to_index_for_geometry(
    geo_id: i64,
    objects: &HashMap<i64, FbxObject>,
    connections: &[FbxConnection],
) -> HashMap<i64, usize> {
    let Some(skin_id) = find_skin_for_geometry(geo_id, objects, connections) else {
        return HashMap::new();
    };
    let skin_clusters = match objects.get(&skin_id) {
        Some(FbxObject::Deformer(FbxDeformer::Skin { clusters })) => clusters.clone(),
        _ => return HashMap::new(),
    };
    let mut map = HashMap::new();
    let mut next = 0usize;
    for &cluster_id in &skin_clusters {
        if let Some(FbxObject::Deformer(FbxDeformer::Cluster { bone_id, .. })) =
            objects.get(&cluster_id)
        {
            if !map.contains_key(bone_id) {
                map.insert(*bone_id, next);
                next += 1;
            }
        }
    }
    map
}

fn find_inverse_bind_for_bone(
    bone_id: i64,
    objects: &HashMap<i64, FbxObject>,
    skin_clusters: &[i64],
) -> Option<Mat4> {
    for &cluster_id in skin_clusters {
        if let Some(FbxObject::Deformer(FbxDeformer::Cluster { bone_id: bid, transform_link, .. })) =
            objects.get(&cluster_id)
        {
            if *bid == bone_id {
                return Some(*transform_link);
            }
        }
    }
    None
}
