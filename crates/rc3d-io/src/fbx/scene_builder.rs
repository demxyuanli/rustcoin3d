use std::collections::HashMap;

use rc3d_core::math::Vec3;
use rc3d_scene::node_data::{
    Coordinate3Node, IndexedFaceSetNode, MaterialNode, NormalNode, SeparatorNode, SkinnedMeshNode,
    TextureCoordinate2Node,
};
use rc3d_scene::{NodeData, SceneGraph};

use super::animation::resolve_animations;
use super::skeleton::{bone_id_to_index_for_geometry, resolve_skeleton};
use super::types::*;

pub fn build_scene(data: &FbxData) -> Result<SceneGraph, super::FbxError> {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(Default::default()));

    let materials: HashMap<i64, &MaterialNode> = data
        .objects
        .iter()
        .filter_map(|(&id, obj)| match obj {
            FbxObject::Material(mat) => Some((id, mat)),
            _ => None,
        })
        .collect();

    let mut geo_to_material: HashMap<i64, i64> = HashMap::new();
    for conn in &data.connections {
        if let Some(FbxObject::Geometry(_)) = data.objects.get(&conn.child) {
            for conn2 in &data.connections {
                if conn2.parent == conn.parent
                    && data
                        .objects
                        .get(&conn2.child)
                        .is_some_and(|o| matches!(o, FbxObject::Material(_)))
                {
                    geo_to_material.insert(conn.child, conn2.child);
                    break;
                }
            }
        }
    }

    for (&geo_id, obj) in &data.objects {
        if let FbxObject::Geometry(geo) = obj {
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));

            if let Some(&mat_id) = geo_to_material.get(&geo_id) {
                if let Some(mat) = materials.get(&mat_id) {
                    graph.add_child(sep, NodeData::Material((*mat).clone()));
                }
            } else {
                graph.add_child(
                    sep,
                    NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.9, 0.9, 0.9))),
                );
            }

            if let Some(resolved) = resolve_skeleton(geo_id, &data.objects, &data.connections) {
                if resolved.skeleton.joint_count() > 0 && !resolved.skin_data.is_empty() {
                    log::info!(
                        "FBX: geometry {} — {} joints, {} skinned verts",
                        geo_id,
                        resolved.skeleton.joint_count(),
                        resolved.skin_data.len(),
                    );
                    let bone_map =
                        bone_id_to_index_for_geometry(geo_id, &data.objects, &data.connections);
                    let clips = resolve_animations(&data.objects, &data.connections, &bone_map);
                    let clip = clips.into_iter().next();
                    if clip.is_some() {
                        log::info!("FBX: attaching animation clip to skinned geo {}", geo_id);
                    }
                    graph.add_child(
                        sep,
                        NodeData::SkinnedMesh(SkinnedMeshNode {
                            skeleton: resolved.skeleton,
                            skin_data: resolved.skin_data,
                            clip,
                        }),
                    );
                }
            }

            if !geo.positions.is_empty() {
                graph.add_child(
                    sep,
                    NodeData::Coordinate3(Coordinate3Node::from_points(geo.positions.clone())),
                );
            }
            if !geo.normals.is_empty() {
                graph.add_child(
                    sep,
                    NodeData::Normal(NormalNode::from_vectors(geo.normals.clone())),
                );
            }
            if !geo.uvs.is_empty() {
                graph.add_child(
                    sep,
                    NodeData::TextureCoordinate2(TextureCoordinate2Node::from_points(
                        geo.uvs.clone(),
                    )),
                );
            }
            if !geo.indices.is_empty() {
                graph.add_child(
                    sep,
                    NodeData::IndexedFaceSet(IndexedFaceSetNode {
                        coord_index: geo.indices.clone(),
                    }),
                );
            }
        }
    }

    if graph.children(root).map_or(true, |c| c.is_empty()) {
        log::warn!("FBX file contains no parseable geometry");
    }

    Ok(graph)
}
