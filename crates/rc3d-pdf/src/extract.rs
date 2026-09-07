//! Scene-graph geometry extraction for 3D PDF export.
//!
//! Drives the shared [`rc3d_scene::traversal::scene_traverse`] visitor so the
//! accumulated model matrices match what the renderer uses (Separator /
//! Transform / Rotation / MultipleCopy / Switch / … semantics come from the
//! structural kernel). Every supported shape becomes its own world-space
//! mesh (so each ends up as an individually pickable object in the PDF),
//! tagged with the diffuse colour and albedo texture of the material state
//! active at that point of the traversal.

use std::collections::HashMap;

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::node_data::{IndexedFaceSetNode, MaterialNode};
use rc3d_scene::traversal::{ChildPolicy, SceneVisitor, SeparatorPolicy};
use rc3d_scene::{NodeData, NodeEntry, SceneGraph};

use crate::u3d::{U3dMesh, U3dTexture};

/// One collected world-space mesh with its shading colour and texture.
#[derive(Clone, Debug, Default)]
pub struct ExtractedMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    triangles: Vec<[u32; 3]>,
    texcoords: Vec<[f32; 2]>,
    /// Diffuse RGBA from the traversal material state; `None` = no colour.
    pub diffuse: Option<[f32; 4]>,
    /// Albedo texture resolved from the material state; `None` = untextured.
    pub texture: Option<U3dTexture>,
    /// Named Separator ancestry (outermost first) used to build the U3D
    /// GroupNode assembly tree; empty = part belongs to the world root.
    pub assembly: Vec<String>,
}

impl ExtractedMesh {
    /// Wrap into the U3D encoder payload.
    pub fn into_u3d_mesh(self, name: &str) -> U3dMesh {
        U3dMesh {
            name: name.to_string(),
            positions: self.positions,
            normals: self.normals,
            triangles: self.triangles,
            texcoords: self.texcoords,
            diffuse: self.diffuse,
            texture: self.texture,
            assembly: self.assembly,
        }
    }

    /// Bounding sphere over the collected positions.
    pub fn bounds(&self) -> ([f32; 3], f32) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for p in &self.positions {
            for i in 0..3 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
        if self.positions.is_empty() {
            return ([0.0; 3], 1.0);
        }
        let center = [
            0.5 * (min[0] + max[0]),
            0.5 * (min[1] + max[1]),
            0.5 * (min[2] + max[2]),
        ];
        let mut r2 = 0.0_f32;
        for p in &self.positions {
            let dx = p[0] - center[0];
            let dy = p[1] - center[1];
            let dz = p[2] - center[2];
            r2 = r2.max(dx * dx + dy * dy + dz * dz);
        }
        (center, r2.sqrt().max(1e-4))
    }
}

/// Extract all supported shape geometry from `graph` in world space, one
/// `ExtractedMesh` per shape (multi-material index face sets are split per
/// material group).
pub fn extract_scene(graph: &SceneGraph) -> Vec<ExtractedMesh> {
    let mut collector = MeshCollector::new();
    for &root in graph.roots() {
        rc3d_scene::traversal::scene_traverse(&mut collector, graph, root);
    }
    collector.meshes
}

/// Per-separator pending geometry state (parallel to the render collector):
/// `Coordinate3`/`TextureCoordinate2` arrays are consumed by the next
/// `IndexedFaceSet` in the same separator scope, and a `Material` colours /
/// textures the shapes that follow it until the scope is left.
struct MeshCollector {
    meshes: Vec<ExtractedMesh>,
    matrix: Mat4,
    points_stack: Vec<Option<Vec<Vec3>>>,
    pending_points: Option<Vec<Vec3>>,
    uv_stack: Vec<Option<Vec<[f32; 2]>>>,
    pending_uv: Option<Vec<[f32; 2]>>,
    color: Option<[f32; 4]>,
    color_stack: Vec<Option<[f32; 4]>>,
    /// Active material albedo texture path (read from the file system once).
    texture_path: Option<String>,
    texture_stack: Vec<Option<String>>,
    tex_cache: HashMap<String, Option<U3dTexture>>,
}

impl MeshCollector {
    fn new() -> Self {
        Self {
            meshes: Vec::new(),
            matrix: Mat4::IDENTITY,
            points_stack: Vec::new(),
            pending_points: None,
            uv_stack: Vec::new(),
            pending_uv: None,
            color: None,
            color_stack: Vec::new(),
            texture_path: None,
            texture_stack: Vec::new(),
            tex_cache: HashMap::new(),
        }
    }

    /// Diffuse RGBA of a scene material.
    fn color_of(mat: &MaterialNode) -> Option<[f32; 4]> {
        let d = mat.diffuse_color.to_array();
        Some([
            d[0].clamp(0.0, 1.0),
            d[1].clamp(0.0, 1.0),
            d[2].clamp(0.0, 1.0),
            mat.opacity.clamp(0.0, 1.0),
        ])
    }

    /// Decode an image file into a PNG texture resource. Failures yield
    /// `None`; the mesh then simply keeps its flat colour.
    fn load_texture(path: &str) -> Option<U3dTexture> {
        let raw = std::fs::read(path).ok()?;
        let img = image::load_from_memory(&raw).ok()?;
        let (width, height) = (img.width(), img.height());
        let has_alpha = img.color().has_alpha();
        let mut png = Vec::new();
        let encoded = {
            use image::codecs::png::PngEncoder;
            use image::{ExtendedColorType, ImageEncoder as _};
            let enc = PngEncoder::new(&mut png);
            if has_alpha {
                enc.write_image(
                    img.to_rgba8().as_raw(),
                    width,
                    height,
                    ExtendedColorType::Rgba8,
                )
            } else {
                enc.write_image(
                    img.to_rgb8().as_raw(),
                    width,
                    height,
                    ExtendedColorType::Rgb8,
                )
            }
            .is_ok()
        };
        if !encoded || png.is_empty() {
            return None;
        }
        // Resource label derived from the file stem (must stay ASCII-safe).
        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| {
                let cleaned: String = s
                    .chars()
                    .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
                    .collect();
                if cleaned.is_empty() {
                    "Texture".to_string()
                } else {
                    cleaned
                }
            })
            .unwrap_or_else(|| "Texture".to_string());
        Some(U3dTexture {
            name,
            width,
            height,
            has_alpha,
            png,
        })
    }

    /// Transform `local` into world space and store it as a new mesh.
    fn emit(
        &mut self,
        local: &rc3d_mesh::TriangleMesh,
        diffuse: Option<[f32; 4]>,
        assembly: &[String],
    ) {
        self.emit_range(local, 0, local.tri_indices.len() / 3, diffuse, None, assembly);
    }

    /// Like [`Self::emit`] but only for triangles `[tri_begin, tri_end)`,
    /// optionally binding the current material texture. `TriangleMesh`
    /// carries adjacency caches that a material-group slice must not share.
    fn emit_range(
        &mut self,
        local: &rc3d_mesh::TriangleMesh,
        tri_begin: usize,
        tri_end: usize,
        diffuse: Option<[f32; 4]>,
        texture: Option<String>,
        assembly: &[String],
    ) {
        if local.positions.is_empty() || tri_begin >= tri_end {
            return;
        }
        let tri_end = tri_end.min(local.tri_indices.len() / 3);
        let m = self.matrix;
        let uv_ok = local.texcoords.len() == local.positions.len() && !local.texcoords.is_empty();
        let mut out = ExtractedMesh {
            diffuse,
            ..Default::default()
        };
        out.assembly = assembly.to_vec();
        for p in &local.positions {
            let w = m.transform_point3(*p);
            out.positions.push(w.to_array());
        }
        // Inverse-transpose of the 3x3 for correct normal orientation.
        let normal_m = m.inverse().transpose();
        for n in &local.normals {
            let wn = normal_m.transform_vector3(*n).normalize_or_zero();
            let n3 = if wn.length_squared() < 0.5 { Vec3::Y } else { wn };
            out.normals.push(n3.to_array());
        }
        if uv_ok && texture.is_some() {
            out.texcoords.extend_from_slice(&local.texcoords);
        }
        for tri in local.tri_indices[tri_begin * 3..tri_end * 3].chunks(3) {
            out.triangles.push([tri[0], tri[1], tri[2]]);
        }
        // Resolve the texture only when UVs travelled with the mesh.
        if let Some(path) = texture {
            if !out.texcoords.is_empty() {
                out.texture = self.resolve_texture(&path);
            }
        }
        self.meshes.push(out);
    }

    /// Cache-load the image behind a material texture path.
    fn resolve_texture(&mut self, path: &str) -> Option<U3dTexture> {
        if let Some(cached) = self.tex_cache.get(path) {
            return cached.clone();
        }
        let loaded = Self::load_texture(path);
        self.tex_cache.insert(path.to_string(), loaded.clone());
        loaded
    }

    /// Outer-to-inner names of the named Separators that contain `node`
    /// (walking the graph's single-parent links). Unnamed separators stay
    /// transparent; structural scopes without names yield no group, keeping
    /// flat exports byte-identical to the pre-hierarchy format.
    fn group_path_of(graph: &SceneGraph, node: NodeId) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut cur = graph.get(node).and_then(|e| e.parent);
        while let Some(id) = cur {
            let Some(entry) = graph.get(id) else { break };
            if let NodeData::Separator(_) = &entry.data {
                if let Some(name) = entry.name.as_deref() {
                    if !name.is_empty() {
                        out.push(sanitize_ident(name));
                    }
                }
            }
            cur = entry.parent;
        }
        out.reverse();
        out
    }

    fn push_shape(&mut self, local: &rc3d_mesh::TriangleMesh, assembly: &[String]) {
        let tex = self.texture_path.clone();
        self.emit_range(local, 0, local.tri_indices.len() / 3, self.color, tex, assembly);
    }

    fn push_indexed_face_set(
        &mut self,
        points: Vec<Vec3>,
        uv: Option<Vec<[f32; 2]>>,
        ifs: &IndexedFaceSetNode,
        assembly: &[String],
    ) {
        if points.is_empty() || ifs.coord_index.is_empty() {
            return;
        }
        let mesh = match uv {
            Some(uv) if uv.len() == points.len() => {
                rc3d_mesh::TriangleMesh::from_indexed_face_set_tex(&points, &uv, &ifs.coord_index)
            }
            _ => rc3d_mesh::TriangleMesh::from_indexed_face_set(&points, &ifs.coord_index),
        };
        if !ifs.material_groups.is_empty() && !ifs.materials.is_empty() {
            // Per-face materials: split the index buffer into one mesh per
            // material group, matching three.js BufferGeometry.group ranges
            // (group.start / count are expressed in tessellated indices).
            for group in &ifs.material_groups {
                let tri_begin = (group.start / 3) as usize;
                let tri_end = ((group.start + group.count) / 3) as usize;
                if tri_begin >= tri_end || tri_end > mesh.tri_indices.len() / 3 {
                    continue;
                }
                let Some(mat) = ifs.materials.get(group.material_index as usize) else {
                    continue;
                };
                self.emit_range(&mesh, tri_begin, tri_end, Self::color_of(mat), None, assembly);
            }
            return;
        }
        self.emit(&mesh, self.color, assembly);
    }
}

/// Draw-identifier-safe label: keep ASCII alphanumerics and `_`/`-`, and
/// fold everything else (spaces, CJK, punctuation) to `_` so the name stays
/// valid inside a U3D node palette and the Acrobat model tree.
fn sanitize_ident(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    if cleaned.is_empty() {
        "_".to_string()
    } else {
        cleaned
    }
}

impl rc3d_scene::traversal::TraversalMatrices for MeshCollector {
    fn model_matrix(&self) -> Mat4 {
        self.matrix
    }
    fn set_model_matrix(&mut self, matrix: Mat4) {
        self.matrix = matrix;
    }
    fn view_matrix(&self) -> Mat4 {
        Mat4::IDENTITY
    }
}

impl SceneVisitor for MeshCollector {
    fn enter_separator(&mut self) {
        self.points_stack.push(self.pending_points.take());
        self.uv_stack.push(self.pending_uv.take());
        self.color_stack.push(self.color);
        self.texture_stack.push(self.texture_path.take());
    }

    fn leave_separator(&mut self) {
        self.pending_points = self.points_stack.pop().flatten();
        self.pending_uv = self.uv_stack.pop().flatten();
        self.color = self.color_stack.pop().flatten();
        self.texture_path = self.texture_stack.pop().flatten();
    }

    fn visit_node(
        &mut self,
        graph: &SceneGraph,
        node: NodeId,
        entry: &NodeEntry,
    ) -> ChildPolicy {
        // Assembly ancestry (named Separators, outermost first) is resolved
        // from the graph's parent links at the shape node itself, so sibling
        // Coordinate3 / IndexedFaceSet property nodes share the same group.
        let assembly = Self::group_path_of(graph, node);
        match &entry.data {
            NodeData::Material(mat) => {
                self.color = Self::color_of(mat);
                self.texture_path = mat.albedo_texture.clone();
            }
            NodeData::Coordinate3(c3) => {
                self.pending_points = Some(c3.point.clone());
            }
            NodeData::TextureCoordinate2(tc2) => {
                self.pending_uv = Some(tc2.point.clone());
            }
            NodeData::IndexedFaceSet(ifs) => {
                if let Some(points) = self.pending_points.take() {
                    let uv = self.pending_uv.take();
                    self.push_indexed_face_set(points, uv, ifs, &assembly);
                }
            }
            NodeData::Cube(c) => {
                let m = rc3d_mesh::tessellate_cube(c.width, c.height, c.depth);
                self.push_shape(&m, &assembly);
            }
            NodeData::Sphere(s) => {
                let m = rc3d_mesh::tessellate_sphere(s.radius, 24, 16);
                self.push_shape(&m, &assembly);
            }
            NodeData::Cylinder(cy) => {
                let m = rc3d_mesh::tessellate_cylinder(cy.radius, cy.height, 24);
                self.push_shape(&m, &assembly);
            }
            NodeData::Cone(co) => {
                let m = rc3d_mesh::tessellate_cone(co.bottom_radius, co.height, 24);
                self.push_shape(&m, &assembly);
            }
            NodeData::Torus(t) => {
                let m = rc3d_mesh::tessellate_torus(t.major_radius, t.minor_radius, 32, 20);
                self.push_shape(&m, &assembly);
            }
            _ => {}
        }
        ChildPolicy::Recurse
    }

    /// RenderCollector-style: flatten direct Transform children so sibling
    /// Coordinate3/IndexedFaceSet pairs under the root Separator share the
    /// same accumulated transform.
    fn separator_policy(&self) -> SeparatorPolicy {
        SeparatorPolicy::ScopedPush
    }
}
