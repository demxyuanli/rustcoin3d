use std::collections::HashMap;
use std::io::{Read, Seek};

use fbxcel::low::v7400::AttributeValue;
use fbxcel::pull_parser::v7400::attribute::loaders::DirectLoader;
use rc3d_core::math::{Mat4, Vec3, Vec4};
use rc3d_scene::node_data::MaterialNode;

use super::types::*;
use super::FbxError;

pub struct FbxParser<R> {
    parser: fbxcel::pull_parser::v7400::Parser<R>,
}

impl<R: Seek + Read> FbxParser<R> {
    pub fn new(parser: fbxcel::pull_parser::v7400::Parser<R>) -> Self {
        Self { parser }
    }

    pub fn parse(&mut self) -> Result<FbxData, FbxError> {
        let mut data = FbxData::default();

        loop {
            let event = self.parser.next_event().map_err(perr)?;
            match event {
                fbxcel::pull_parser::v7400::Event::StartNode(node) => {
                    let name = node.name().to_owned();
                    match name.as_str() {
                        "Objects" => self.collect_objects(&mut data)?,
                        "Connections" => self.collect_connections(&mut data)?,
                        _ => self.parser.skip_current_node().map_err(perr)?,
                    }
                }
                fbxcel::pull_parser::v7400::Event::EndFbx(_) => break,
                _ => {}
            }
        }

        // Resolve references
        self.resolve_curve_nodes(&mut data);
        self.resolve_skin_clusters(&mut data);

        Ok(data)
    }

    fn collect_objects(&mut self, data: &mut FbxData) -> Result<(), FbxError> {
        loop {
            let event = self.parser.next_event().map_err(perr)?;
            match event {
                fbxcel::pull_parser::v7400::Event::StartNode(node) => {
                    let name = node.name().to_owned();
                    match name.as_str() {
                        "Geometry" => {
                            let (id, _, _) = read_model_header(node)?;
                            let geo = self.read_geometry()?;
                            if !geo.positions.is_empty() {
                                data.objects.insert(id, FbxObject::Geometry(geo));
                            }
                        }
                        "Material" => {
                            let (id, _, _) = read_model_header(node)?;
                            let mat = self.read_material()?;
                            data.objects.insert(id, FbxObject::Material(mat));
                        }
                        "Model" => {
                            let (id, name, model_type) = read_model_header(node)?;
                            let model = self.read_model_body(name, model_type)?;
                            data.objects.insert(id, FbxObject::Model(model));
                        }
                        "Deformer" => {
                            let (id, _name, deformer_type) = read_model_header(node)?;
                            let deformer = self.read_deformer_body(deformer_type)?;
                            data.objects.insert(id, FbxObject::Deformer(deformer));
                        }
                        "AnimationStack" => {
                            let (id, name, _) = read_model_header(node)?;
                            self.parser.skip_current_node().map_err(perr)?;
                            data.objects.insert(id, FbxObject::AnimationStack(FbxAnimationStack { name }));
                        }
                        "AnimationLayer" => {
                            let (id, name, _) = read_model_header(node)?;
                            self.parser.skip_current_node().map_err(perr)?;
                            data.objects.insert(id, FbxObject::AnimationLayer(FbxAnimationLayer { name }));
                        }
                        "AnimationCurveNode" => {
                            let (id, name, _) = read_model_header(node)?;
                            self.parser.skip_current_node().map_err(perr)?;
                            data.objects.insert(
                                id,
                                FbxObject::AnimationCurveNode(FbxAnimationCurveNode {
                                    name,
                                    target_property: String::new(),
                                    target_model_id: None,
                                }),
                            );
                        }
                        "AnimationCurve" => {
                            let (id, _, _) = read_model_header(node)?;
                            let curve = self.read_animation_curve_body()?;
                            data.objects.insert(id, FbxObject::AnimationCurve(curve));
                        }
                        _ => self.parser.skip_current_node().map_err(perr)?,
                    }
                }
                fbxcel::pull_parser::v7400::Event::EndNode => return Ok(()),
                fbxcel::pull_parser::v7400::Event::EndFbx(_) => {
                    return Err(FbxError::Parse("Unexpected end in Objects".into()));
                }
            }
        }
    }

    fn collect_connections(&mut self, data: &mut FbxData) -> Result<(), FbxError> {
        loop {
            let event = self.parser.next_event().map_err(perr)?;
            match event {
                fbxcel::pull_parser::v7400::Event::StartNode(node) => {
                    let name = node.name().to_owned();
                    if name == "C" || name == "Connect" {
                        let vals = read_all_attrs(node)?;
                        let get_i64 = |v: &AttributeValue| -> Option<i64> {
                            match v {
                                AttributeValue::I64(i) => Some(*i),
                                AttributeValue::I32(i) => Some(*i as i64),
                                _ => None,
                            }
                        };
                        if vals.len() >= 3 {
                            if let (Some(c), Some(p)) = (get_i64(&vals[1]), get_i64(&vals[2])) {
                                data.connections.push(FbxConnection { child: c, parent: p });
                            }
                        }
                    } else {
                        self.parser.skip_current_node().map_err(perr)?;
                    }
                }
                fbxcel::pull_parser::v7400::Event::EndNode => return Ok(()),
                fbxcel::pull_parser::v7400::Event::EndFbx(_) => {
                    return Err(FbxError::Parse("Unexpected end in Connections".into()));
                }
            }
        }
    }

    fn read_geometry(&mut self) -> Result<FbxGeometry, FbxError> {
        let mut geo = FbxGeometry::default();

        loop {
            let event = self.parser.next_event().map_err(perr)?;
            match event {
                fbxcel::pull_parser::v7400::Event::StartNode(node) => {
                    let name = node.name().to_owned();
                    match name.as_str() {
                        "Vertices" => {
                            if let Some(arr) = read_attr_f32_pair(node)? {
                                geo.positions = arr
                                    .chunks_exact(3)
                                    .map(|c| Vec3::new(c[0], c[1], c[2]))
                                    .collect();
                            }
                        }
                        "PolygonVertexIndex" => {
                            if let Some(arr) = read_attr_i32_pair(node)? {
                                geo.indices = convert_polygon_indices(&arr);
                            }
                        }
                        "Normals" => {
                            if let Some(arr) = read_attr_f32_pair(node)? {
                                geo.normals = arr
                                    .chunks_exact(3)
                                    .map(|c| Vec3::new(c[0], c[1], c[2]))
                                    .collect();
                            }
                        }
                        "UV" => {
                            if let Some(arr) = read_attr_f32_pair(node)? {
                                geo.uvs = arr.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
                            }
                        }
                        _ => self.parser.skip_current_node().map_err(perr)?,
                    }
                }
                fbxcel::pull_parser::v7400::Event::EndNode => return Ok(geo),
                fbxcel::pull_parser::v7400::Event::EndFbx(_) => {
                    return Err(FbxError::Parse("Unexpected end in Geometry".into()));
                }
            }
        }
    }

    fn read_material(&mut self) -> Result<MaterialNode, FbxError> {
        let mut mat = MaterialNode::from_diffuse(Vec3::new(0.8, 0.8, 0.8));

        loop {
            let event = self.parser.next_event().map_err(perr)?;
            match event {
                fbxcel::pull_parser::v7400::Event::StartNode(node) => {
                    let name = node.name().to_owned();
                    match name.as_str() {
                        "DiffuseColor" | "Diffuse" => {
                            if let Some(c) = read_color_attr(node)? {
                                mat.diffuse_color = c;
                                mat.base_color = c;
                            }
                        }
                        "SpecularColor" | "Specular" => {
                            if let Some(c) = read_color_attr(node)? {
                                mat.specular_color = c;
                            }
                        }
                        "AmbientColor" | "Ambient" => {
                            if let Some(c) = read_color_attr(node)? {
                                mat.ambient_color = c;
                            }
                        }
                        "Shininess" | "ShininessExponent" => {
                            if let Some(v) = read_first_f32_attr(node)? {
                                mat.shininess = v;
                            }
                        }
                        "Opacity" | "TransparencyFactor" => {
                            if let Some(v) = read_first_f32_attr(node)? {
                                mat.opacity = 1.0 - v;
                            }
                        }
                        "EmissiveColor" => {
                            if let Some(c) = read_color_attr(node)? {
                                mat.emissive_color = c;
                            }
                        }
                        _ => self.parser.skip_current_node().map_err(perr)?,
                    }
                }
                fbxcel::pull_parser::v7400::Event::EndNode => return Ok(mat),
                fbxcel::pull_parser::v7400::Event::EndFbx(_) => {
                    return Err(FbxError::Parse("Unexpected end in Material".into()));
                }
            }
        }
    }

    fn read_model_body(
        &mut self,
        name: String,
        model_type: String,
    ) -> Result<FbxModel, FbxError> {
        let mut model = FbxModel {
            name,
            model_type,
            local_transform: Mat4::IDENTITY,
            translation: None,
            rotation: None,
            scaling: None,
            pre_rotation: None,
            post_rotation: None,
        };

        loop {
            let event = self.parser.next_event().map_err(perr)?;
            match event {
                fbxcel::pull_parser::v7400::Event::StartNode(node) => {
                    let name = node.name().to_owned();
                    match name.as_str() {
                        "Properties70" | "Properties" => {
                            self.read_model_properties(&mut model)?;
                        }
                        _ => self.parser.skip_current_node().map_err(perr)?,
                    }
                }
                fbxcel::pull_parser::v7400::Event::EndNode => {
                    let t = model.translation.unwrap_or(Vec3::ZERO);
                    let r = model.rotation.unwrap_or(Vec3::ZERO);
                    let s = model.scaling.unwrap_or(Vec3::ONE);

                    let rx = Mat4::from_rotation_x(r.x.to_radians());
                    let ry = Mat4::from_rotation_y(r.y.to_radians());
                    let rz = Mat4::from_rotation_z(r.z.to_radians());

                    let mut transform =
                        Mat4::from_translation(t) * rz * ry * rx * Mat4::from_scale(s);

                    if let Some(pre) = model.pre_rotation {
                        let pre_mat = Mat4::from_rotation_z(pre.z.to_radians())
                            * Mat4::from_rotation_y(pre.y.to_radians())
                            * Mat4::from_rotation_x(pre.x.to_radians());
                        transform = transform * pre_mat;
                    }

                    model.local_transform = transform;
                    return Ok(model);
                }
                fbxcel::pull_parser::v7400::Event::EndFbx(_) => {
                    return Err(FbxError::Parse("Unexpected end in Model".into()));
                }
            }
        }
    }

    fn read_model_properties(&mut self, model: &mut FbxModel) -> Result<(), FbxError> {
        loop {
            let event = self.parser.next_event().map_err(perr)?;
            match event {
                fbxcel::pull_parser::v7400::Event::StartNode(node) => {
                    let name = node.name().to_owned();
                    if name == "P" || name == "Property" {
                        let vals = read_all_attrs(node)?;
                        let get_str = |i: usize, vals: &[AttributeValue]| -> Option<String> {
                            vals.get(i).and_then(|v| {
                                if let AttributeValue::String(s) = v { Some(s.clone()) } else { None }
                            })
                        };
                        let get_f64 = |i: usize, vals: &[AttributeValue]| -> Option<f64> {
                            vals.get(i).and_then(|v| {
                                v.get_f64()
                                    .or_else(|| v.get_f32().map(|f| f as f64))
                            })
                        };

                        if let Some(prop_name) = get_str(0, &vals) {
                            match prop_name.as_str() {
                                "Lcl Translation" => {
                                    model.translation = Some(Vec3::new(
                                        get_f64(4, &vals).unwrap_or(0.0) as f32,
                                        get_f64(5, &vals).unwrap_or(0.0) as f32,
                                        get_f64(6, &vals).unwrap_or(0.0) as f32,
                                    ));
                                }
                                "Lcl Rotation" => {
                                    model.rotation = Some(Vec3::new(
                                        get_f64(4, &vals).unwrap_or(0.0) as f32,
                                        get_f64(5, &vals).unwrap_or(0.0) as f32,
                                        get_f64(6, &vals).unwrap_or(0.0) as f32,
                                    ));
                                }
                                "Lcl Scaling" => {
                                    model.scaling = Some(Vec3::new(
                                        get_f64(4, &vals).unwrap_or(1.0) as f32,
                                        get_f64(5, &vals).unwrap_or(1.0) as f32,
                                        get_f64(6, &vals).unwrap_or(1.0) as f32,
                                    ));
                                }
                                "PreRotation" => {
                                    model.pre_rotation = Some(Vec3::new(
                                        get_f64(4, &vals).unwrap_or(0.0) as f32,
                                        get_f64(5, &vals).unwrap_or(0.0) as f32,
                                        get_f64(6, &vals).unwrap_or(0.0) as f32,
                                    ));
                                }
                                "PostRotation" => {
                                    model.post_rotation = Some(Vec3::new(
                                        get_f64(4, &vals).unwrap_or(0.0) as f32,
                                        get_f64(5, &vals).unwrap_or(0.0) as f32,
                                        get_f64(6, &vals).unwrap_or(0.0) as f32,
                                    ));
                                }
                                _ => {}
                            }
                        }
                    } else {
                        self.parser.skip_current_node().map_err(perr)?;
                    }
                }
                fbxcel::pull_parser::v7400::Event::EndNode => return Ok(()),
                fbxcel::pull_parser::v7400::Event::EndFbx(_) => {
                    return Err(FbxError::Parse("Unexpected end in Properties".into()));
                }
            }
        }
    }

    fn read_deformer_body(&mut self, deformer_type: String) -> Result<FbxDeformer, FbxError> {
        let mut indices: Vec<i32> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut transform: Option<Mat4> = None;
        let mut transform_link: Option<Mat4> = None;

        loop {
            let event = self.parser.next_event().map_err(perr)?;
            match event {
                fbxcel::pull_parser::v7400::Event::StartNode(node) => {
                    let name = node.name().to_owned();
                    match name.as_str() {
                        "Indexes" => {
                            indices = read_attr_i32_pair(node)?.unwrap_or_default();
                        }
                        "Weights" => {
                            weights = read_attr_f64_pair(node)?.unwrap_or_default();
                        }
                        "Transform" => {
                            transform = read_matrix(node)?;
                        }
                        "TransformLink" => {
                            transform_link = read_matrix(node)?;
                        }
                        _ => self.parser.skip_current_node().map_err(perr)?,
                    }
                }
                fbxcel::pull_parser::v7400::Event::EndNode => {
                    let deformer = match deformer_type.as_str() {
                        "Skin" => FbxDeformer::Skin { clusters: Vec::new() },
                        "Cluster" => FbxDeformer::Cluster {
                            bone_id: 0,
                            indices,
                            weights,
                            transform: transform.unwrap_or(Mat4::IDENTITY),
                            transform_link: transform_link.unwrap_or(Mat4::IDENTITY),
                        },
                        "BlendShape" => FbxDeformer::BlendShape { channels: Vec::new() },
                        _ => FbxDeformer::Skin { clusters: Vec::new() },
                    };
                    return Ok(deformer);
                }
                fbxcel::pull_parser::v7400::Event::EndFbx(_) => {
                    return Err(FbxError::Parse("Unexpected end in Deformer".into()));
                }
            }
        }
    }

    fn read_animation_curve_body(&mut self) -> Result<FbxAnimationCurve, FbxError> {
        let mut curve = FbxAnimationCurve::default();

        loop {
            let event = self.parser.next_event().map_err(perr)?;
            match event {
                fbxcel::pull_parser::v7400::Event::StartNode(node) => {
                    let name = node.name().to_owned();
                    match name.as_str() {
                        "KeyTime" => {
                            curve.times = read_attr_f64_pair(node)?.unwrap_or_default();
                        }
                        "KeyValueFloat" | "KeyValueDouble" => {
                            curve.values = read_attr_f64_pair(node)?.unwrap_or_default();
                        }
                        _ => self.parser.skip_current_node().map_err(perr)?,
                    }
                }
                fbxcel::pull_parser::v7400::Event::EndNode => return Ok(curve),
                fbxcel::pull_parser::v7400::Event::EndFbx(_) => {
                    return Err(FbxError::Parse("Unexpected end in AnimationCurve".into()));
                }
            }
        }
    }

    /// After parsing, resolve animation curve node targets (model / bone IDs).
    fn resolve_curve_nodes(&self, data: &mut FbxData) {
        let mut curve_node_to_model: HashMap<i64, i64> = HashMap::new();
        for conn in &data.connections {
            if matches!(
                data.objects.get(&conn.child),
                Some(FbxObject::AnimationCurveNode(_))
            ) && matches!(data.objects.get(&conn.parent), Some(FbxObject::Model(_)))
            {
                curve_node_to_model.insert(conn.child, conn.parent);
            }
        }
        for (cn_id, model_id) in curve_node_to_model {
            if let Some(FbxObject::AnimationCurveNode(cn)) = data.objects.get_mut(&cn_id) {
                cn.target_model_id = Some(model_id);
            }
        }
    }

    fn resolve_skin_clusters(&self, data: &mut FbxData) {
        let mut skin_to_clusters: HashMap<i64, Vec<i64>> = HashMap::new();
        for conn in &data.connections {
            if matches!(data.objects.get(&conn.child), Some(FbxObject::Deformer(FbxDeformer::Cluster { .. }))) {
                skin_to_clusters
                    .entry(conn.parent)
                    .or_default()
                    .push(conn.child);
            }
        }

        // Update Skin deformers with their cluster IDs
        for (&skin_id, clusters) in &skin_to_clusters {
            if let Some(FbxObject::Deformer(FbxDeformer::Skin { clusters: ref mut c })) =
                data.objects.get_mut(&skin_id)
            {
                *c = clusters.clone();
            }
        }

        // Cluster bone_id: Model(LimbNode) <-> Cluster (either orientation)
        for conn in &data.connections {
            if let Some(FbxObject::Deformer(FbxDeformer::Cluster { .. })) =
                data.objects.get(&conn.child)
            {
                if let Some(FbxObject::Model(m)) = data.objects.get(&conn.parent) {
                    if m.is_limb_node() {
                        if let Some(FbxObject::Deformer(FbxDeformer::Cluster { ref mut bone_id, .. })) =
                            data.objects.get_mut(&conn.child)
                        {
                            *bone_id = conn.parent;
                        }
                    }
                }
            }
            if let Some(FbxObject::Deformer(FbxDeformer::Cluster { .. })) =
                data.objects.get(&conn.parent)
            {
                if let Some(FbxObject::Model(m)) = data.objects.get(&conn.child) {
                    if m.is_limb_node() {
                        if let Some(FbxObject::Deformer(FbxDeformer::Cluster { ref mut bone_id, .. })) =
                            data.objects.get_mut(&conn.parent)
                        {
                            *bone_id = conn.child;
                        }
                    }
                }
            }
        }
    }
}

// --- Attribute helpers ---

fn read_model_header<R: Seek + Read>(
    start_node: fbxcel::pull_parser::v7400::StartNode<'_, R>,
) -> Result<(i64, String, String), FbxError> {
    let mut attrs = start_node.attributes();
    let id = match attrs.load_next(DirectLoader).map_err(perr)? {
        Some(AttributeValue::I64(i)) => i,
        Some(AttributeValue::I32(i)) => i as i64,
        _ => 0,
    };
    let name = match attrs.load_next(DirectLoader).map_err(perr)? {
        Some(AttributeValue::String(s)) => s,
        _ => String::new(),
    };
    let type_ = match attrs.load_next(DirectLoader).map_err(perr)? {
        Some(AttributeValue::String(s)) => s,
        _ => String::new(),
    };
    Ok((id, name, type_))
}

fn read_all_attrs<R: Seek + Read>(
    start_node: fbxcel::pull_parser::v7400::StartNode<'_, R>,
) -> Result<Vec<AttributeValue>, FbxError> {
    let mut attrs = start_node.attributes();
    let mut result = Vec::new();
    loop {
        let v = attrs.load_next(DirectLoader).map_err(perr)?;
        match v {
            Some(val) => result.push(val),
            None => break,
        }
    }
    Ok(result)
}

fn read_attr_f32_pair<R: Seek + Read>(
    start_node: fbxcel::pull_parser::v7400::StartNode<'_, R>,
) -> Result<Option<Vec<f32>>, FbxError> {
    let mut attrs = start_node.attributes();
    let _len = attrs.load_next(DirectLoader).map_err(perr)?;
    let data = attrs.load_next(DirectLoader).map_err(perr)?;
    match data {
        Some(AttributeValue::ArrF32(v)) => Ok(Some(v)),
        Some(AttributeValue::ArrF64(v)) => Ok(Some(v.into_iter().map(|v| v as f32).collect())),
        _ => Ok(None),
    }
}

fn read_attr_f64_pair<R: Seek + Read>(
    start_node: fbxcel::pull_parser::v7400::StartNode<'_, R>,
) -> Result<Option<Vec<f64>>, FbxError> {
    let mut attrs = start_node.attributes();
    let _len = attrs.load_next(DirectLoader).map_err(perr)?;
    let data = attrs.load_next(DirectLoader).map_err(perr)?;
    match data {
        Some(AttributeValue::ArrF64(v)) => Ok(Some(v)),
        Some(AttributeValue::ArrF32(v)) => Ok(Some(v.into_iter().map(|v| v as f64).collect())),
        _ => Ok(None),
    }
}

fn read_attr_i32_pair<R: Seek + Read>(
    start_node: fbxcel::pull_parser::v7400::StartNode<'_, R>,
) -> Result<Option<Vec<i32>>, FbxError> {
    let mut attrs = start_node.attributes();
    let _len = attrs.load_next(DirectLoader).map_err(perr)?;
    let data = attrs.load_next(DirectLoader).map_err(perr)?;
    match data {
        Some(AttributeValue::ArrI32(v)) => Ok(Some(v)),
        Some(AttributeValue::ArrI64(v)) => Ok(Some(v.into_iter().map(|v| v as i32).collect())),
        _ => Ok(None),
    }
}

fn read_color_attr<R: Seek + Read>(
    start_node: fbxcel::pull_parser::v7400::StartNode<'_, R>,
) -> Result<Option<Vec3>, FbxError> {
    let vals = read_attr_f32_pair(start_node)?;
    match vals {
        Some(v) if v.len() >= 3 => Ok(Some(Vec3::new(v[0], v[1], v[2]))),
        _ => Ok(None),
    }
}

fn read_first_f32_attr<R: Seek + Read>(
    start_node: fbxcel::pull_parser::v7400::StartNode<'_, R>,
) -> Result<Option<f32>, FbxError> {
    let vals = read_attr_f32_pair(start_node)?;
    Ok(vals.and_then(|v| v.into_iter().next()))
}

/// Read a 4x4 matrix from 16 f64 values.
fn read_matrix<R: Seek + Read>(
    start_node: fbxcel::pull_parser::v7400::StartNode<'_, R>,
) -> Result<Option<Mat4>, FbxError> {
    let vals = read_attr_f64_pair(start_node)?;
    match vals {
        Some(v) if v.len() >= 16 => {
            // FBX stores matrices in row-major order
            let cols = [
                Vec4::new(v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32),
                Vec4::new(v[4] as f32, v[5] as f32, v[6] as f32, v[7] as f32),
                Vec4::new(v[8] as f32, v[9] as f32, v[10] as f32, v[11] as f32),
                Vec4::new(v[12] as f32, v[13] as f32, v[14] as f32, v[15] as f32),
            ];
            let m = Mat4::from_cols(cols[0], cols[1], cols[2], cols[3]);
            Ok(Some(m.transpose())) // Convert row-major to column-major
        }
        _ => Ok(None),
    }
}

fn convert_polygon_indices(raw: &[i32]) -> Vec<i32> {
    let mut faces: Vec<Vec<i32>> = Vec::new();
    let mut current: Vec<i32> = Vec::new();
    for &idx in raw {
        if idx < 0 {
            current.push(!idx);
            if !current.is_empty() {
                faces.push(std::mem::take(&mut current));
            }
        } else {
            current.push(idx);
        }
    }
    if !current.is_empty() {
        faces.push(current);
    }

    let mut result = Vec::new();
    for face in &faces {
        if face.len() < 3 {
            continue;
        }
        for i in 1..face.len() - 1 {
            result.push(face[0]);
            result.push(face[i]);
            result.push(face[i + 1]);
            result.push(-1);
        }
    }
    result
}

fn perr(e: fbxcel::pull_parser::Error) -> FbxError {
    FbxError::Parse(e.to_string())
}
