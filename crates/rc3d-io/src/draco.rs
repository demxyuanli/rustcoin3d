//! Decode `KHR_draco_mesh_compression` primitive payloads into CPU arrays.

use draco_core::{
    DecoderBuffer, FaceIndex, Mesh, MeshDecoder, PointAttribute, PointIndex,
};
use rc3d_core::math::Vec3;

pub struct DecodedDracoMesh {
    pub positions: Vec<Vec3>,
    pub normals: Option<Vec<Vec3>>,
    pub texcoords: Option<Vec<[f32; 2]>>,
    pub indices: Vec<i32>,
}

pub fn decode_primitive(
    document: &gltf::Document,
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
) -> Result<Option<DecodedDracoMesh>, String> {
    let Some(ext) = primitive.extension_value("KHR_draco_mesh_compression") else {
        return Ok(None);
    };
    let view_idx = ext
        .get("bufferView")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "KHR_draco_mesh_compression missing bufferView".to_string())?
        as usize;
    let view = document
        .views()
        .nth(view_idx)
        .ok_or_else(|| format!("Draco bufferView {view_idx} out of range"))?;
    let buf = buffers
        .get(view.buffer().index())
        .ok_or_else(|| "Draco buffer index out of range".to_string())?;
    let start = view.offset();
    let end = start
        .checked_add(view.length())
        .ok_or_else(|| "Draco bufferView overflow".to_string())?;
    if end > buf.len() {
        return Err("Draco bufferView exceeds buffer".into());
    }
    let bytes = &buf[start..end];

    let mut mesh = Mesh::new();
    MeshDecoder::new()
        .decode(&mut DecoderBuffer::new(bytes), &mut mesh)
        .map_err(|e| format!("Draco decode failed: {e:?}"))?;

    let attr_map = ext.get("attributes").and_then(|v| v.as_object());
    let unique_id = |semantic: &str| -> Option<u32> {
        attr_map
            .and_then(|m| m.get(semantic))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
    };

    let npoints = mesh.num_points();
    let pos_id = unique_id("POSITION").ok_or_else(|| {
        "KHR_draco_mesh_compression missing POSITION attribute id".to_string()
    })?;
    let pos_attr = mesh
        .attribute_by_unique_id(pos_id)
        .ok_or_else(|| format!("Draco POSITION unique id {pos_id} not in mesh"))?;
    let positions = read_vec3_attr(pos_attr, npoints)
        .ok_or_else(|| "failed to read Draco POSITION".to_string())?;
    if positions.is_empty() {
        return Err("Draco mesh has no positions".into());
    }

    let normals = unique_id("NORMAL")
        .and_then(|id| mesh.attribute_by_unique_id(id))
        .and_then(|attr| read_vec3_attr(attr, npoints));
    let texcoords = unique_id("TEXCOORD_0")
        .and_then(|id| mesh.attribute_by_unique_id(id))
        .and_then(|attr| read_vec2_attr(attr, npoints));

    let mut indices = Vec::new();
    let nfaces = mesh.num_faces();
    if nfaces > 0 {
        indices.reserve(nfaces * 4);
        for i in 0..nfaces {
            let face = mesh.face(FaceIndex(i as u32));
            indices.push(face[0].0 as i32);
            indices.push(face[1].0 as i32);
            indices.push(face[2].0 as i32);
            indices.push(-1);
        }
    } else {
        indices.reserve(npoints + npoints / 3);
        for i in 0..npoints as u32 {
            indices.push(i as i32);
            if (i + 1) % 3 == 0 {
                indices.push(-1);
            }
        }
    }

    Ok(Some(DecodedDracoMesh {
        positions,
        normals,
        texcoords,
        indices,
    }))
}

fn attr_value_bytes(attr: &PointAttribute, point_i: u32) -> Option<&[u8]> {
    let avi = attr.mapped_index(PointIndex(point_i));
    let stride = attr.byte_stride().max(0) as usize;
    if stride == 0 {
        return None;
    }
    let data = attr.buffer().data();
    let off = avi.0 as usize * stride;
    data.get(off..off + stride)
}

fn read_f32_components(attr: &PointAttribute, point_i: u32, want: usize) -> Option<Vec<f32>> {
    use draco_core::DataType;
    let bytes = attr_value_bytes(attr, point_i)?;
    let comps = (attr.num_components() as usize).min(want);
    let mut out = vec![0.0f32; want];
    match attr.data_type() {
        DataType::Float32 => {
            for i in 0..comps {
                let s = i * 4;
                out[i] = f32::from_le_bytes(bytes.get(s..s + 4)?.try_into().ok()?);
            }
        }
        DataType::Float64 => {
            for i in 0..comps {
                let s = i * 8;
                out[i] = f64::from_le_bytes(bytes.get(s..s + 8)?.try_into().ok()?) as f32;
            }
        }
        _ => return None,
    }
    Some(out)
}

fn read_vec3_attr(attr: &PointAttribute, npoints: usize) -> Option<Vec<Vec3>> {
    let mut out = Vec::with_capacity(npoints);
    for i in 0..npoints {
        let v = read_f32_components(attr, i as u32, 3)?;
        out.push(Vec3::new(v[0], v[1], v[2]));
    }
    Some(out)
}

fn read_vec2_attr(attr: &PointAttribute, npoints: usize) -> Option<Vec<[f32; 2]>> {
    let mut out = Vec::with_capacity(npoints);
    for i in 0..npoints {
        let v = read_f32_components(attr, i as u32, 2)?;
        out.push([v[0], v[1]]);
    }
    Some(out)
}
