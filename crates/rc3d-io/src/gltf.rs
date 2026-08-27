use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use base64::Engine;

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
    let (document, buffers, images) = import_gltf(path)?;
    build_scene(&document, &buffers, &images, path)
}

fn uses_handled_unvalidated_extension(gltf: &gltf::Gltf) -> bool {
    gltf.extensions_required()
        .chain(gltf.extensions_used())
        .any(|e| e == "KHR_draco_mesh_compression" || e == "KHR_texture_basisu")
}

fn import_gltf(
    path: &Path,
) -> Result<(gltf::Document, Vec<gltf::buffer::Data>, Vec<gltf::image::Data>), GltfError> {
    let file = File::open(path)?;
    let gltf = gltf::Gltf::from_reader_without_validation(BufReader::new(file))?;
    if !uses_handled_unvalidated_extension(&gltf) {
        return gltf::import(path).map_err(Into::into);
    }
    let buffers = gltf::import_buffers(&gltf.document, path.parent(), gltf.blob)?;
    let images = import_images_with_ktx2(&gltf.document, path.parent(), &buffers)?;
    Ok((gltf.document, buffers, images))
}

fn import_images_with_ktx2(
    document: &gltf::Document,
    base: Option<&Path>,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<gltf::image::Data>, GltfError> {
    let mut images = Vec::new();
    for image in document.images() {
        match gltf::image::Data::from_source(image.source(), base, buffers) {
            Ok(data) => images.push(data),
            Err(png_err) => {
                let bytes = read_image_bytes(&image, base, buffers).map_err(GltfError::Gltf)?;
                if crate::ktx2::looks_like_ktx2(&bytes) || mime_is_ktx2(&image) {
                    let decoded = crate::ktx2::decode_to_rgba(&bytes).map_err(GltfError::Gltf)?;
                    images.push(gltf::image::Data {
                        pixels: decoded.rgba,
                        format: gltf::image::Format::R8G8B8A8,
                        width: decoded.width,
                        height: decoded.height,
                    });
                } else {
                    return Err(png_err.into());
                }
            }
        }
    }
    Ok(images)
}

fn mime_is_ktx2(image: &gltf::Image<'_>) -> bool {
    match image.source() {
        gltf::image::Source::Uri { mime_type, uri } => {
            mime_type == Some("image/ktx2")
                || uri.rsplit('.').next().is_some_and(|ext| ext.eq_ignore_ascii_case("ktx2"))
        }
        gltf::image::Source::View { mime_type, .. } => mime_type == "image/ktx2",
    }
}

fn read_image_bytes(
    image: &gltf::Image<'_>,
    base: Option<&Path>,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<u8>, String> {
    match image.source() {
        gltf::image::Source::Uri { uri, .. } => read_uri_bytes(base, uri),
        gltf::image::Source::View { view, .. } => {
            let buf = buffers
                .get(view.buffer().index())
                .ok_or_else(|| "KTX2 buffer index out of range".to_string())?;
            let start = view.offset();
            let end = start
                .checked_add(view.length())
                .ok_or_else(|| "KTX2 bufferView overflow".to_string())?;
            if end > buf.len() {
                return Err("KTX2 bufferView exceeds buffer".into());
            }
            Ok(buf[start..end].to_vec())
        }
    }
}

fn read_uri_bytes(base: Option<&Path>, uri: &str) -> Result<Vec<u8>, String> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let b64 = rest
            .split(";base64,")
            .nth(1)
            .ok_or_else(|| "invalid data URI".to_string())?;
        return base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| e.to_string());
    }
    let path = if let Some(file) = uri.strip_prefix("file://") {
        PathBuf::from(file)
    } else if let Some(base) = base {
        base.join(uri)
    } else {
        PathBuf::from(uri)
    };
    std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))
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
            let mut positions: Vec<Vec3> = reader
                .read_positions()
                .map(|p| p.map(Vec3::from).collect())
                .unwrap_or_default();
            let mut draco_normals: Option<Vec<Vec3>> = None;
            let mut draco_uvs: Option<Vec<[f32; 2]>> = None;
            let mut draco_indices: Option<Vec<i32>> = None;

            if positions.is_empty() {
                match crate::draco::decode_primitive(document, &prim, buffers) {
                    Ok(Some(decoded)) => {
                        positions = decoded.positions;
                        draco_normals = decoded.normals;
                        draco_uvs = decoded.texcoords;
                        draco_indices = Some(decoded.indices);
                    }
                    Ok(None) => continue,
                    Err(e) => return Err(GltfError::Gltf(e)),
                }
            }

            if positions.is_empty() {
                continue;
            }

            let separator_id = graph.add_root(NodeData::Separator(SeparatorNode));

            graph.add_child(
                separator_id,
                NodeData::Coordinate3(Coordinate3Node::from_points(positions)),
            );

            if let Some(normals) = draco_normals {
                graph.add_child(
                    separator_id,
                    NodeData::Normal(NormalNode::from_vectors(normals)),
                );
            } else if let Some(normals_iter) = reader.read_normals() {
                let normals: Vec<Vec3> = normals_iter.map(Vec3::from).collect();
                graph.add_child(
                    separator_id,
                    NodeData::Normal(NormalNode::from_vectors(normals)),
                );
            }

            if let Some(uvs) = draco_uvs {
                if !uvs.is_empty() {
                    graph.add_child(
                        separator_id,
                        NodeData::TextureCoordinate2(TextureCoordinate2Node::from_points(uvs)),
                    );
                }
            } else if let Some(uv_iter) = reader.read_tex_coords(0) {
                let uvs: Vec<[f32; 2]> = uv_iter.into_f32().map(|uv| [uv[0], uv[1]]).collect();
                if !uvs.is_empty() {
                    graph.add_child(
                        separator_id,
                        NodeData::TextureCoordinate2(TextureCoordinate2Node::from_points(uvs)),
                    );
                }
            }

            let coord_index: Vec<i32> = if let Some(indices) = draco_indices {
                indices
            } else {
                build_coord_index(reader.read_indices(), &graph, separator_id)
            };

            if !coord_index.is_empty() {
                graph.add_child(
                    separator_id,
                    NodeData::IndexedFaceSet(IndexedFaceSetNode::from_coord_index(coord_index)),
                );
            }

            let material_node = build_material_node(document, &prim, images, base_dir);
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

fn is_raster_image_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "bmp" | "tif" | "tiff" | "webp")
    )
}

fn texture_image_index(
    document: &gltf::Document,
    texture: &gltf::texture::Texture,
) -> Option<usize> {
    if let Some(idx) = texture
        .extension_value("KHR_texture_basisu")
        .and_then(|v| v.get("source"))
        .and_then(|v| v.as_u64())
    {
        return Some(idx as usize);
    }
    let src = document
        .as_json()
        .textures
        .get(texture.index())?
        .source
        .value();
    if src == u32::MAX as usize {
        None
    } else {
        Some(src)
    }
}

fn extract_embedded_png(
    img_data: &gltf::image::Data,
    source_index: usize,
    texture_index: usize,
) -> Option<String> {
    let temp_dir = std::env::temp_dir().canonicalize().ok()?;
    let rc3d_temp = temp_dir.join("rc3d_gltf_textures");
    let _ = std::fs::create_dir_all(&rc3d_temp);
    let filename = format!("embedded_{source_index}_{texture_index}.png");
    let temp_path = rc3d_temp.join(&filename);

    if !temp_path.is_file() {
        let rgba_data = match img_data.format {
            gltf::image::Format::R8G8B8 => img_data
                .pixels
                .chunks_exact(3)
                .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255u8])
                .collect::<Vec<u8>>(),
            gltf::image::Format::R8G8B8A8 => img_data.pixels.clone(),
            _ => img_data.pixels.clone(),
        };

        if let Some(img) = image::RgbaImage::from_raw(img_data.width, img_data.height, rgba_data)
        {
            let _ = img.save(&temp_path);
        }
    }

    temp_path.is_file().then(|| temp_path.to_string_lossy().to_string())
}

fn resolve_texture_path(
    document: &gltf::Document,
    texture: &gltf::texture::Texture,
    images: &[gltf::image::Data],
    base_dir: &Path,
) -> Option<String> {
    let source_index = texture_image_index(document, texture)?;

    if let Some(image) = document.images().nth(source_index) {
        if let gltf::image::Source::Uri { uri, .. } = image.source() {
            let named_path = base_dir.join(uri);
            if named_path.is_file() && is_raster_image_path(&named_path) {
                return Some(named_path.to_string_lossy().to_string());
            }
            let direct = Path::new(uri);
            if direct.is_file() && is_raster_image_path(direct) {
                return Some(uri.to_string());
            }
        }
        if let Some(name) = image.name().filter(|n| !n.is_empty()) {
            let named_path = base_dir.join(name);
            if named_path.is_file() && is_raster_image_path(&named_path) {
                return Some(named_path.to_string_lossy().to_string());
            }
        }
    }

    images
        .get(source_index)
        .and_then(|img| extract_embedded_png(img, source_index, texture.index()))
}

fn json_vec3(v: Option<&serde_json::Value>) -> Option<Vec3> {
    let a = v?.as_array()?;
    if a.len() < 3 {
        return None;
    }
    Some(Vec3::new(
        a[0].as_f64()? as f32,
        a[1].as_f64()? as f32,
        a[2].as_f64()? as f32,
    ))
}

fn material_extension<'a>(
    document: &'a gltf::Document,
    mat: &gltf::Material,
    name: &str,
) -> Option<&'a serde_json::Value> {
    let idx = mat.index()?;
    document
        .as_json()
        .materials
        .get(idx)?
        .extensions
        .as_ref()?
        .others
        .get(name)
}

fn build_material_node(
    document: &gltf::Document,
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
        .and_then(|tex| resolve_texture_path(document, &tex.texture(), images, base_dir));

    let normal_texture = mat
        .normal_texture()
        .and_then(|tex| resolve_texture_path(document, &tex.texture(), images, base_dir));

    let emissive_texture = mat
        .emissive_texture()
        .and_then(|tex| resolve_texture_path(document, &tex.texture(), images, base_dir));

    let metallic_roughness_texture = pbr
        .metallic_roughness_texture()
        .and_then(|tex| resolve_texture_path(document, &tex.texture(), images, base_dir));

    let occlusion_texture = mat
        .occlusion_texture()
        .and_then(|tex| resolve_texture_path(document, &tex.texture(), images, base_dir));

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

    let alpha_cutoff = mat.alpha_cutoff().unwrap_or(0.5);
    let sheen_ext = material_extension(document, &mat, "KHR_materials_sheen");
    let sheen_color = json_vec3(sheen_ext.and_then(|v| v.get("sheenColorFactor")))
        .unwrap_or(Vec3::ZERO);
    let sheen_roughness = sheen_ext
        .and_then(|v| v.get("sheenRoughnessFactor"))
        .and_then(|v| v.as_f64())
        .map(|x| x as f32)
        .unwrap_or(0.0);
    let clearcoat_ext = material_extension(document, &mat, "KHR_materials_clearcoat");
    let clearcoat_factor = clearcoat_ext
        .and_then(|v| v.get("clearcoatFactor"))
        .and_then(|v| v.as_f64())
        .map(|x| x as f32)
        .unwrap_or(0.0);
    let clearcoat_roughness = clearcoat_ext
        .and_then(|v| v.get("clearcoatRoughnessFactor"))
        .and_then(|v| v.as_f64())
        .map(|x| x as f32)
        .unwrap_or(0.0);
    let aniso_ext = material_extension(document, &mat, "KHR_materials_anisotropy");
    let anisotropic = aniso_ext
        .and_then(|v| v.get("anisotropyStrength"))
        .and_then(|v| v.as_f64())
        .map(|x| x as f32)
        .unwrap_or(0.0);
    let iri_ext = material_extension(document, &mat, "KHR_materials_iridescence");
    let iridescence_factor = iri_ext
        .and_then(|v| v.get("iridescenceFactor"))
        .and_then(|v| v.as_f64())
        .map(|x| x as f32)
        .unwrap_or(0.0);
    let iridescence_ior = iri_ext
        .and_then(|v| v.get("iridescenceIor"))
        .and_then(|v| v.as_f64())
        .map(|x| x as f32)
        .unwrap_or(1.3);
    let iridescence_thickness_min = iri_ext
        .and_then(|v| v.get("iridescenceThicknessMinimum"))
        .and_then(|v| v.as_f64())
        .map(|x| x as f32)
        .unwrap_or(100.0);
    let iridescence_thickness_max = iri_ext
        .and_then(|v| v.get("iridescenceThicknessMaximum"))
        .and_then(|v| v.as_f64())
        .map(|x| x as f32)
        .unwrap_or(400.0);

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
        alpha_cutoff,
        double_sided: mat.double_sided(),
        anisotropic,
        clearcoat_factor,
        clearcoat_roughness,
        specular_factor: 1.0,
        specular_color_factor: Vec3::ONE,
        transmission_factor: mat.transmission().map(|t| t.transmission_factor()).unwrap_or(0.0),
        ior: mat.ior().unwrap_or(1.5),
        sheen_color,
        sheen_roughness,
        iridescence_factor,
        iridescence_ior,
        iridescence_thickness_min,
        iridescence_thickness_max,
        toon_steps: 0.0,
        visualize_normals: false,
        visualize_depth: false,
        light_group: None,
        custom_wgsl: None,
        custom_uniforms: [0.0; 4],
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
