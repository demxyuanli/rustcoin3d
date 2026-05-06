use std::collections::HashMap;
use std::path::Path;

use gltf::mesh::util::ReadIndices;
use rc3d_core::math::{Mat4, Quat, Vec3};
use rc3d_scene::node_data::{
    Coordinate3Node, IndexedFaceSetNode, MaterialNode, NodeData, NormalNode,
    SeparatorNode, TextureCoordinate2Node, TransformNode,
};
use rc3d_scene::SceneGraph;

#[derive(Debug, thiserror::Error)]
pub enum GltfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("glTF parse error: {0}")]
    Gltf(String),
    #[error("Base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),
}

impl From<gltf::Error> for GltfError {
    fn from(e: gltf::Error) -> Self {
        GltfError::Gltf(e.to_string())
    }
}

/// Parse a glTF 2.0 file (.gltf or .glb) into a SceneGraph.
pub fn parse_gltf_file(path: &Path) -> Result<SceneGraph, GltfError> {
    let (document, buffers, images) = gltf::import(path)?;
    build_scene(&document, &buffers, &images, path)
}

struct PrimitiveNodes {
    /// The separator wrapping the full primitive including coord/index/material.
    separator: rc3d_core::NodeId,
    /// The material node id within that separator.
    _material: rc3d_core::NodeId,
}

fn build_scene(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    source_path: &Path,
) -> Result<SceneGraph, GltfError> {
    let mut graph = SceneGraph::new();
    let base_dir = source_path.parent().unwrap_or(Path::new(""));

    let mut mesh_roots: HashMap<usize, Vec<PrimitiveNodes>> = HashMap::new();

    for mesh in document.meshes() {
        let mut primitive_nodes = Vec::new();
        for prim in mesh.primitives() {
            let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
            let positions: Vec<Vec3> = reader
                .read_positions()
                .map(|p| p.map(Vec3::from).collect())
                .unwrap_or_default();

            if positions.is_empty() {
                continue;
            }

            let separator_id = graph.add_root(NodeData::Separator(SeparatorNode));

            graph.add_child(
                separator_id,
                NodeData::Coordinate3(Coordinate3Node::from_points(positions)),
            );

            // Normals
            if let Some(normals_iter) = reader.read_normals() {
                let normals: Vec<Vec3> = normals_iter.map(Vec3::from).collect();
                graph.add_child(
                    separator_id,
                    NodeData::Normal(NormalNode::from_vectors(normals)),
                );
            }

            // Texcoords (first set only)
            if let Some(uv_iter) = reader.read_tex_coords(0) {
                let uvs: Vec<[f32; 2]> = uv_iter.into_f32().map(|uv| [uv[0], uv[1]]).collect();
                if !uvs.is_empty() {
                    graph.add_child(
                        separator_id,
                        NodeData::TextureCoordinate2(TextureCoordinate2Node::from_points(uvs)),
                    );
                }
            }

            // Indices
            let coord_index: Vec<i32> = build_coord_index(reader.read_indices(), &graph, separator_id);

            if !coord_index.is_empty() {
                graph.add_child(
                    separator_id,
                    NodeData::IndexedFaceSet(IndexedFaceSetNode { coord_index }),
                );
            }

            let material_node = build_material_node(&prim, images, base_dir);
            let material_id = graph.add_child(separator_id, material_node);

            // Morph targets (blend shapes)
            let morph_targets = build_morph_targets(&reader);
            let mesh_weights: Vec<f32> = mesh.weights()
                .map(|w| w.to_vec())
                .unwrap_or_default();
            if !morph_targets.is_empty() {
                let weights = if mesh_weights.is_empty() {
                    vec![0.0f32; morph_targets.len()]
                } else {
                    let mut w = mesh_weights;
                    w.resize(morph_targets.len(), 0.0);
                    w
                };
                graph.add_child(
                    separator_id,
                    NodeData::MorphTarget(rc3d_scene::MorphTargetNode {
                        targets: morph_targets,
                        weights,
                    }),
                );
            }

            primitive_nodes.push(PrimitiveNodes {
                separator: separator_id,
                _material: material_id,
            });
        }
        mesh_roots.insert(mesh.index(), primitive_nodes);
    }

    // Build node hierarchy from default scene
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or_else(|| GltfError::Gltf("no scene found".into()))?;

    for node in scene.nodes() {
        build_node(&node, &mut graph, None, &mesh_roots);
    }

    Ok(graph)
}

fn build_coord_index(
    indices: Option<ReadIndices<'_>>,
    graph: &SceneGraph,
    separator_id: rc3d_core::NodeId,
) -> Vec<i32> {
    match indices {
        Some(ReadIndices::U8(iter)) => {
            let mut out = Vec::new();
            for i in iter {
                out.push(i as i32);
                if out.len() % 3 == 0 {
                    out.push(-1);
                }
            }
            out
        }
        Some(ReadIndices::U16(iter)) => {
            let mut out = Vec::new();
            for i in iter {
                out.push(i as i32);
                if out.len() % 3 == 0 {
                    out.push(-1);
                }
            }
            out
        }
        Some(ReadIndices::U32(iter)) => {
            let mut out = Vec::new();
            for i in iter {
                out.push(i as i32);
                if out.len() % 3 == 0 {
                    out.push(-1);
                }
            }
            out
        }
        None => {
            // Non-indexed: build sequential indices
            let count = positions_len(graph, separator_id);
            let mut out = Vec::with_capacity(count + count / 3);
            for i in 0..count as u32 {
                out.push(i as i32);
                if (i + 1) % 3 == 0 {
                    out.push(-1);
                }
            }
            out
        }
    }
}

fn positions_len(graph: &SceneGraph, separator_id: rc3d_core::NodeId) -> usize {
    if let Some(entry) = graph.get(separator_id) {
        for &child in &entry.children {
            if let Some(child_entry) = graph.get(child) {
                if let NodeData::Coordinate3(c) = &child_entry.data {
                    return c.point.len();
                }
            }
        }
    }
    0
}

fn build_node(
    node: &gltf::Node,
    graph: &mut SceneGraph,
    parent_id: Option<rc3d_core::NodeId>,
    mesh_roots: &HashMap<usize, Vec<PrimitiveNodes>>,
) -> Option<rc3d_core::NodeId> {
    let separator_id = match parent_id {
        Some(pid) => graph.add_child(pid, NodeData::Separator(SeparatorNode)),
        None => graph.add_root(NodeData::Separator(SeparatorNode)),
    };

    // Apply node transform
    let (trans, rot, scale) = node.transform().decomposed();
    let has_transform = trans != [0.0, 0.0, 0.0]
        || rot != [0.0, 0.0, 0.0, 1.0]
        || scale != [1.0, 1.0, 1.0];

    if has_transform {
        let rotation = Mat4::from_quat(Quat::from_array([rot[0], rot[1], rot[2], rot[3]]));
        let transform = TransformNode {
            translation: Vec3::new(trans[0], trans[1], trans[2]),
            rotation,
            scale: Vec3::new(scale[0], scale[1], scale[2]),
            center: Vec3::ZERO,
        };
        graph.add_child(separator_id, NodeData::Transform(transform));
    }

    // Build light if present (requires KHR_lights_punctual feature)
    build_light_node(node, graph, separator_id);

    // Attach mesh if present
    if let Some(mesh) = node.mesh() {
        let mi = mesh.index();
        if let Some(primitives) = mesh_roots.get(&mi) {
            for prim in primitives {
                clone_subtree_contents(graph, prim.separator, separator_id);
            }
        }
    }

    // Children
    for child in node.children() {
        build_node(&child, graph, Some(separator_id), mesh_roots);
    }

    Some(separator_id)
}

fn build_light_node(
    node: &gltf::Node,
    graph: &mut SceneGraph,
    parent_id: rc3d_core::NodeId,
) {
    use gltf::khr_lights_punctual::Kind;

    let light_index = match node.light() {
        Some(l) => l,
        None => return,
    };

    let color = light_index.color();
    let intensity = light_index.intensity();
    let rgb = Vec3::new(color[0] * intensity, color[1] * intensity, color[2] * intensity);

    match light_index.kind() {
        Kind::Directional => {
            graph.add_child(
                parent_id,
                NodeData::DirectionalLight(rc3d_scene::node_data::DirectionalLightNode {
                    direction: Vec3::new(0.0, -1.0, 0.0),
                    color: rgb,
                    intensity: 1.0,
                    light_group: None,
                }),
            );
        }
        Kind::Point => {
            graph.add_child(
                parent_id,
                NodeData::PointLight(rc3d_scene::node_data::PointLightNode {
                    location: Vec3::ZERO,
                    color: rgb,
                    intensity: 1.0,
                    light_group: None,
                }),
            );
        }
        Kind::Spot {
            inner_cone_angle: _,
            outer_cone_angle,
        } => {
            graph.add_child(
                parent_id,
                NodeData::SpotLight(rc3d_scene::node_data::SpotLightNode {
                    location: Vec3::ZERO,
                    direction: Vec3::new(0.0, -1.0, 0.0),
                    color: rgb,
                    intensity: 1.0,
                    cut_off_angle: outer_cone_angle,
                    drop_off_rate: 4.0,
                    light_group: None,
                }),
            );
        }
    }
}

fn clone_subtree_contents(
    graph: &mut SceneGraph,
    src: rc3d_core::NodeId,
    dst: rc3d_core::NodeId,
) {
    let children: Vec<rc3d_core::NodeId> = graph
        .get(src)
        .map(|e| e.children.clone())
        .unwrap_or_default();

    for child_id in children {
        clone_node_recursive(graph, child_id, dst);
    }
}

fn clone_node_recursive(
    graph: &mut SceneGraph,
    src: rc3d_core::NodeId,
    parent: rc3d_core::NodeId,
) {
    // Collect data first under immutable borrow, then mutate.
    let data = graph.get(src).map(|e| e.data.clone());
    let children: Vec<rc3d_core::NodeId> = graph
        .get(src)
        .map(|e| e.children.clone())
        .unwrap_or_default();

    let Some(data) = data else { return };

    let new_id = match &data {
        NodeData::Separator(_) => graph.add_child(parent, NodeData::Separator(SeparatorNode)),
        NodeData::Coordinate3(c) => graph.add_child(parent, NodeData::Coordinate3(c.clone())),
        NodeData::Normal(n) => graph.add_child(parent, NodeData::Normal(n.clone())),
        NodeData::TextureCoordinate2(t) => {
            graph.add_child(parent, NodeData::TextureCoordinate2(t.clone()))
        }
        NodeData::Material(m) => graph.add_child(parent, NodeData::Material(m.clone())),
        NodeData::IndexedFaceSet(i) => {
            graph.add_child(parent, NodeData::IndexedFaceSet(i.clone()))
        }
        other => graph.add_child(parent, other.clone()),
    };

    for child_id in children {
        clone_node_recursive(graph, child_id, new_id);
    }
}

fn resolve_texture_path(
    texture: &gltf::texture::Texture,
    images: &[gltf::image::Data],
    base_dir: &Path,
) -> Option<String> {
    let source = texture.source();

    // 1. Try source name as a file path hint
    if let Some(name) = source.name().filter(|n| !n.is_empty()) {
        let named_path = base_dir.join(name);
        if named_path.is_file() {
            return Some(named_path.to_string_lossy().to_string());
        }
        if Path::new(name).is_file() {
            return Some(name.to_string());
        }
        return Some(name.to_string());
    }

    // 2. For embedded textures, extract to temp file
    let source_index = source.index();
    if source_index < images.len() {
        let img_data = &images[source_index];

        if let Ok(temp_dir) = std::env::temp_dir().canonicalize() {
            let rc3d_temp = temp_dir.join("rc3d_gltf_textures");
            let _ = std::fs::create_dir_all(&rc3d_temp);
            let filename = format!("embedded_{}_{}.png", source_index, texture.index());
            let temp_path = rc3d_temp.join(&filename);

            if !temp_path.is_file() {
                let rgba_data = match img_data.format {
                    gltf::image::Format::R8G8B8 => {
                        img_data.pixels.chunks_exact(3)
                            .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255u8])
                            .collect::<Vec<u8>>()
                    }
                    gltf::image::Format::R8G8B8A8 => {
                        img_data.pixels.clone()
                    }
                    _ => {
                        img_data.pixels.clone()
                    }
                };

                if let Some(img) = image::RgbaImage::from_raw(
                    img_data.width,
                    img_data.height,
                    rgba_data,
                ) {
                    let _ = img.save(&temp_path);
                }
            }

            if temp_path.is_file() {
                return Some(temp_path.to_string_lossy().to_string());
            }
        }
    }

    None
}

fn build_material_node(
    primitive: &gltf::Primitive,
    images: &[gltf::image::Data],
    base_dir: &Path,
) -> NodeData {
    let mat = primitive.material();
    let pbr = mat.pbr_metallic_roughness();
    let base_color = pbr.base_color_factor();
    let metallic = pbr.metallic_factor();
    let roughness = pbr.roughness_factor();
    let base = Vec3::new(base_color[0], base_color[1], base_color[2]);
    let opacity = base_color[3];

    let albedo_texture = pbr
        .base_color_texture()
        .and_then(|tex| resolve_texture_path(&tex.texture(), images, base_dir));

    let normal_texture = mat
        .normal_texture()
        .and_then(|tex| resolve_texture_path(&tex.texture(), images, base_dir));

    let emissive_texture = mat
        .emissive_texture()
        .and_then(|tex| resolve_texture_path(&tex.texture(), images, base_dir));

    let metallic_roughness_texture = pbr
        .metallic_roughness_texture()
        .and_then(|tex| resolve_texture_path(&tex.texture(), images, base_dir));

    let occlusion_texture = mat
        .occlusion_texture()
        .and_then(|tex| resolve_texture_path(&tex.texture(), images, base_dir));

    let emissive = mat.emissive_factor();
    let emissive_color = Vec3::new(emissive[0], emissive[1], emissive[2]);
    let ambient = Vec3::new(
        emissive[0].max(base.x * 0.1),
        emissive[1].max(base.y * 0.1),
        emissive[2].max(base.z * 0.1),
    );

    let alpha_mode = match mat.alpha_mode() {
        gltf::material::AlphaMode::Opaque => rc3d_scene::AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => rc3d_scene::AlphaMode::Mask,
        gltf::material::AlphaMode::Blend => rc3d_scene::AlphaMode::Blend,
    };

    NodeData::Material(MaterialNode {
        diffuse_color: base,
        ambient_color: ambient,
        specular_color: Vec3::new(0.04, 0.04, 0.04),
        shininess: (1.0 - roughness).max(0.01) * 128.0,
        base_color: base,
        metallic,
        roughness,
        albedo_texture,
        normal_texture,
        opacity,
        emissive_color,
        emissive_texture,
        metallic_roughness_texture,
        occlusion_texture,
        alpha_mode,
        alpha_cutoff: mat.alpha_cutoff().unwrap_or(0.5),
        double_sided: mat.double_sided(),
        anisotropic: 0.0,
        light_group: None,
    })
}

fn build_morph_targets<'a, 's, F>(
    reader: &gltf::mesh::Reader<'a, 's, F>,
) -> Vec<rc3d_scene::MorphTarget>
where
    F: Clone + Fn(gltf::buffer::Buffer<'a>) -> Option<&'s [u8]>,
{
    let mut targets = Vec::new();
    for (i, (positions, normals, _tangents)) in reader.read_morph_targets().enumerate() {
        let position_deltas: Vec<Vec3> = positions
            .map(|p| p.map(|d| Vec3::new(d[0], d[1], d[2])).collect())
            .unwrap_or_default();

        if position_deltas.is_empty() {
            continue;
        }

        let normal_deltas: Option<Vec<Vec3>> = normals
            .map(|n| n.map(|d| Vec3::new(d[0], d[1], d[2])).collect());

        let name = format!("morph_{i}");

        targets.push(rc3d_scene::MorphTarget {
            name,
            position_deltas,
            normal_deltas,
            tangent_deltas: None,
        });
    }
    targets
}
