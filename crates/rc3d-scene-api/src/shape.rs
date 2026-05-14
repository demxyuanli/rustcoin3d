//! Shape trait and built-in geometry types.

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::Aabb;
use rc3d_mesh::topology::TriangleMesh;
use rc3d_scene::node_data::{
    Coordinate3Node, ConeNode, CubeNode, CylinderNode, IndexedFaceSetNode, IndexedLineSetNode,
    NodeData, NormalNode, SphereNode,
};
use rc3d_scene::SceneGraph;

use crate::material::Material;

/// A 3D shape that can be compiled into internal geometry.
pub trait Shape {
    fn compile(&self) -> TriangleMesh;
    fn aabb(&self) -> Aabb;
    fn shape_material(&self) -> Option<&Material>;
    fn shape_translation(&self) -> Option<&Vec3>;
    fn shape_rotation(&self) -> Option<&Mat4>;
    fn shape_scale(&self) -> Option<&Vec3>;
    fn has_transform(&self) -> bool;
    fn node_data(&self) -> NodeData;

    /// Add geometry nodes to the scene graph under `parent`.
    ///
    /// Default: adds a single `node_data()` child. Shapes that require multiple
    /// sibling nodes (e.g., Coordinate3 + IndexedFaceSet) override this.
    fn add_geometry_nodes(&self, graph: &mut SceneGraph, parent: rc3d_core::NodeId) {
        graph.add_child(parent, self.node_data());
    }
}

// ─── Builder helpers shared by all shapes ───

/// Internal builder state for transform + material.
#[derive(Clone, Debug, Default)]
pub(crate) struct ShapeProps {
    pub translation: Option<Vec3>,
    pub rotation: Option<Mat4>,
    pub scale: Option<Vec3>,
    pub material: Option<Material>,
}

impl ShapeProps {
    pub(crate) fn has_transform(&self) -> bool {
        self.translation.is_some() || self.rotation.is_some() || self.scale.is_some()
    }
}

macro_rules! shape_props {
    ($self:ident) => {
        fn shape_material(&$self) -> Option<&Material> { $self.props.material.as_ref() }
        fn shape_translation(&$self) -> Option<&Vec3> { $self.props.translation.as_ref() }
        fn shape_rotation(&$self) -> Option<&Mat4> { $self.props.rotation.as_ref() }
        fn shape_scale(&$self) -> Option<&Vec3> { $self.props.scale.as_ref() }
        fn has_transform(&$self) -> bool { $self.props.has_transform() }
    };
}

macro_rules! builder_methods {
    () => {
        pub fn at(mut self, x: f32, y: f32, z: f32) -> Self {
            self.props.translation = Some(Vec3::new(x, y, z));
            self
        }

        pub fn scale(mut self, sx: f32, sy: f32, sz: f32) -> Self {
            self.props.scale = Some(Vec3::new(sx, sy, sz));
            self
        }

        pub fn rotate_x(mut self, angle_rad: f32) -> Self {
            let r = self.props.rotation.get_or_insert(Mat4::IDENTITY);
            *r = Mat4::from_rotation_x(angle_rad) * *r;
            self
        }

        pub fn rotate_y(mut self, angle_rad: f32) -> Self {
            let r = self.props.rotation.get_or_insert(Mat4::IDENTITY);
            *r = Mat4::from_rotation_y(angle_rad) * *r;
            self
        }

        pub fn rotate_z(mut self, angle_rad: f32) -> Self {
            let r = self.props.rotation.get_or_insert(Mat4::IDENTITY);
            *r = Mat4::from_rotation_z(angle_rad) * *r;
            self
        }

        pub fn material(mut self, mat: Material) -> Self {
            self.props.material = Some(mat);
            self
        }
    };
}

// ─── Cube ───

#[derive(Clone, Debug)]
pub struct Cube {
    pub width: f32,
    pub height: f32,
    pub depth: f32,
    pub(crate) props: ShapeProps,
}

impl Default for Cube {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 1.0,
            depth: 1.0,
            props: ShapeProps::default(),
        }
    }
}

impl Cube {
    pub fn width(mut self, v: f32) -> Self { self.width = v; self }
    pub fn height(mut self, v: f32) -> Self { self.height = v; self }
    pub fn depth(mut self, v: f32) -> Self { self.depth = v; self }
    builder_methods!();
}

impl Shape for Cube {
    fn compile(&self) -> TriangleMesh {
        rc3d_mesh::tessellate_cube(self.width, self.height, self.depth)
    }
    fn aabb(&self) -> Aabb {
        let hw = self.width / 2.0;
        let hh = self.height / 2.0;
        let hd = self.depth / 2.0;
        Aabb { min: Vec3::new(-hw, -hh, -hd), max: Vec3::new(hw, hh, hd) }
    }
    shape_props!(self);
    fn node_data(&self) -> NodeData {
        NodeData::Cube(CubeNode { width: self.width, height: self.height, depth: self.depth })
    }
}

// ─── Sphere ───

#[derive(Clone, Debug)]
pub struct Sphere {
    pub radius: f32,
    pub(crate) slices: u32,
    pub(crate) stacks: u32,
    pub(crate) props: ShapeProps,
}

impl Default for Sphere {
    fn default() -> Self {
        Self { radius: 1.0, slices: 24, stacks: 16, props: ShapeProps::default() }
    }
}

impl Sphere {
    pub fn radius(mut self, v: f32) -> Self { self.radius = v; self }
    pub fn slices(mut self, v: u32) -> Self { self.slices = v; self }
    pub fn stacks(mut self, v: u32) -> Self { self.stacks = v; self }
    builder_methods!();
}

impl Shape for Sphere {
    fn compile(&self) -> TriangleMesh {
        rc3d_mesh::tessellate_sphere(self.radius, self.slices, self.stacks)
    }
    fn aabb(&self) -> Aabb {
        let r = self.radius;
        Aabb { min: Vec3::new(-r, -r, -r), max: Vec3::new(r, r, r) }
    }
    shape_props!(self);
    fn node_data(&self) -> NodeData {
        NodeData::Sphere(SphereNode { radius: self.radius })
    }
}

// ─── Cone ───

#[derive(Clone, Debug)]
pub struct Cone {
    pub bottom_radius: f32,
    pub height: f32,
    pub(crate) segments: u32,
    pub(crate) props: ShapeProps,
}

impl Default for Cone {
    fn default() -> Self {
        Self { bottom_radius: 1.0, height: 2.0, segments: 24, props: ShapeProps::default() }
    }
}

impl Cone {
    pub fn bottom_radius(mut self, v: f32) -> Self { self.bottom_radius = v; self }
    pub fn height(mut self, v: f32) -> Self { self.height = v; self }
    pub fn segments(mut self, v: u32) -> Self { self.segments = v; self }
    builder_methods!();
}

impl Shape for Cone {
    fn compile(&self) -> TriangleMesh {
        rc3d_mesh::tessellate_cone(self.bottom_radius, self.height, self.segments)
    }
    fn aabb(&self) -> Aabb {
        let r = self.bottom_radius;
        let hh = self.height / 2.0;
        Aabb { min: Vec3::new(-r, -hh, -r), max: Vec3::new(r, hh, r) }
    }
    shape_props!(self);
    fn node_data(&self) -> NodeData {
        NodeData::Cone(ConeNode { bottom_radius: self.bottom_radius, height: self.height })
    }
}

// ─── Cylinder ───

#[derive(Clone, Debug)]
pub struct Cylinder {
    pub radius: f32,
    pub height: f32,
    pub(crate) segments: u32,
    pub(crate) props: ShapeProps,
}

impl Default for Cylinder {
    fn default() -> Self {
        Self { radius: 1.0, height: 2.0, segments: 24, props: ShapeProps::default() }
    }
}

impl Cylinder {
    pub fn radius(mut self, v: f32) -> Self { self.radius = v; self }
    pub fn height(mut self, v: f32) -> Self { self.height = v; self }
    pub fn segments(mut self, v: u32) -> Self { self.segments = v; self }
    builder_methods!();
}

impl Shape for Cylinder {
    fn compile(&self) -> TriangleMesh {
        rc3d_mesh::tessellate_cylinder(self.radius, self.height, self.segments)
    }
    fn aabb(&self) -> Aabb {
        let r = self.radius;
        let hh = self.height / 2.0;
        Aabb { min: Vec3::new(-r, -hh, -r), max: Vec3::new(r, hh, r) }
    }
    shape_props!(self);
    fn node_data(&self) -> NodeData {
        NodeData::Cylinder(CylinderNode { radius: self.radius, height: self.height })
    }
}

// ─── Mesh (IndexedFaceSet wrapper) ───

#[derive(Clone, Debug)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub normals: Option<Vec<Vec3>>,
    pub texcoords: Option<Vec<[f32; 2]>>,
    pub(crate) props: ShapeProps,
    /// Pre-compiled TriangleMesh from from_raw / from_indexed.
    pub(crate) cached_topology: Option<TriangleMesh>,
}

impl Mesh {
    /// Construct from raw non-indexed triangle positions (AoS, every 3 = one triangle).
    pub fn from_raw(positions: &[Vec3], indices: &[u32]) -> Self {
        let topology = TriangleMesh::from_indexed(positions, indices);
        Self {
            positions: positions.to_vec(),
            indices: indices.to_vec(),
            normals: None,
            texcoords: None,
            props: ShapeProps::default(),
            cached_topology: Some(topology),
        }
    }

    /// Attach per-vertex normals.
    pub fn with_normals(mut self, normals: Vec<Vec3>) -> Self {
        self.normals = Some(normals);
        self
    }

    /// Attach per-vertex texture coordinates.
    pub fn with_texcoords(mut self, uvs: Vec<[f32; 2]>) -> Self {
        self.texcoords = Some(uvs);
        self
    }

    /// Construct from raw positions and coord_index (Inventor/VRML face set format).
    ///
    /// Polygons are fan-triangulated internally; `node_data()` will produce
    /// correct OIV coord_index with `-1` terminators for each triangle.
    pub fn from_face_set(positions: &[Vec3], coord_index: &[i32]) -> Self {
        let topology = TriangleMesh::from_indexed_face_set(positions, coord_index);
        let indices = topology.tri_indices.clone();
        Self {
            positions: positions.to_vec(),
            indices,
            normals: None,
            texcoords: None,
            props: ShapeProps::default(),
            cached_topology: Some(topology),
        }
    }

    builder_methods!();
}

impl Shape for Mesh {
    fn compile(&self) -> TriangleMesh {
        if let Some(ref t) = self.cached_topology {
            return t.clone();
        }
        TriangleMesh::from_indexed(&self.positions, &self.indices)
    }

    fn aabb(&self) -> Aabb {
        if let Some(ref t) = self.cached_topology {
            return t.bounding_box();
        }
        if self.positions.is_empty() {
            return Aabb::empty();
        }
        let mut aabb = Aabb::from_point(self.positions[0]);
        for p in &self.positions[1..] {
            aabb = aabb.union(&Aabb::from_point(*p));
        }
        aabb
    }

    shape_props!(self);

    fn node_data(&self) -> NodeData {
        // Coord_index in OIV format: each triangle terminated by -1
        let tri_count = self.indices.len() / 3;
        let mut coord_index = Vec::with_capacity(tri_count * 4);
        for tri in self.indices.chunks_exact(3) {
            coord_index.push(tri[0] as i32);
            coord_index.push(tri[1] as i32);
            coord_index.push(tri[2] as i32);
            coord_index.push(-1);
        }
        NodeData::IndexedFaceSet(IndexedFaceSetNode { coord_index })
    }

    fn add_geometry_nodes(&self, graph: &mut SceneGraph, parent: rc3d_core::NodeId) {
        graph.add_child(
            parent,
            NodeData::Coordinate3(Coordinate3Node::from_points(self.positions.clone())),
        );
        if let Some(ref normals) = self.normals {
            graph.add_child(
                parent,
                NodeData::Normal(NormalNode::from_vectors(normals.clone())),
            );
        }
        graph.add_child(parent, self.node_data());
    }
}

// ─── LineSet (IndexedLineSet wrapper) ───

#[derive(Clone, Debug)]
pub struct LineSet {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub line_width: f32,
    pub(crate) props: ShapeProps,
}

impl Default for LineSet {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            indices: Vec::new(),
            line_width: 1.0,
            props: ShapeProps::default(),
        }
    }
}

impl LineSet {
    pub fn from_lines(positions: &[Vec3], indices: &[u32]) -> Self {
        Self {
            positions: positions.to_vec(),
            indices: indices.to_vec(),
            line_width: 1.0,
            props: ShapeProps::default(),
        }
    }

    pub fn from_line_strip(positions: &[Vec3]) -> Self {
        let mut indices = Vec::with_capacity((positions.len() - 1) * 2);
        for i in 0..positions.len().saturating_sub(1) {
            indices.push(i as u32);
            indices.push((i + 1) as u32);
        }
        Self {
            positions: positions.to_vec(),
            indices,
            line_width: 1.0,
            props: ShapeProps::default(),
        }
    }

    pub fn line_width(mut self, v: f32) -> Self { self.line_width = v; self }
    builder_methods!();
}

impl Shape for LineSet {
    fn compile(&self) -> TriangleMesh {
        TriangleMesh::empty()
    }
    fn aabb(&self) -> Aabb {
        if self.positions.is_empty() {
            return Aabb::empty();
        }
        let mut aabb = Aabb::from_point(self.positions[0]);
        for p in &self.positions[1..] {
            aabb = aabb.union(&Aabb::from_point(*p));
        }
        aabb
    }
    shape_props!(self);
    fn node_data(&self) -> NodeData {
        let coord_index: Vec<i32> = self.indices.iter().map(|&i| i as i32).collect();
        NodeData::IndexedLineSet(IndexedLineSetNode {
            coord_index,
            line_width: self.line_width,
        })
    }
    fn add_geometry_nodes(&self, graph: &mut SceneGraph, parent: rc3d_core::NodeId) {
        graph.add_child(
            parent,
            NodeData::Coordinate3(Coordinate3Node::from_points(self.positions.clone())),
        );
        graph.add_child(parent, self.node_data());
    }
}

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube_default_aabb() {
        let cube = Cube::default();
        let aabb = cube.aabb();
        assert_eq!(aabb.min, Vec3::new(-0.5, -0.5, -0.5));
        assert_eq!(aabb.max, Vec3::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn test_cube_compile_vertices() {
        let mesh = Cube::default().compile();
        // 6 faces × 6 vertices per face (2 triangles, flat normals) = 36
        assert_eq!(mesh.positions.len(), 36);
        assert_eq!(mesh.faces.len(), 12); // 6 faces × 2 triangles
    }

    #[test]
    fn test_cube_scaled_aabb() {
        let cube = Cube::default().width(2.0).height(1.0).depth(3.0);
        let aabb = cube.aabb();
        assert_eq!(aabb.min, Vec3::new(-1.0, -0.5, -1.5));
        assert_eq!(aabb.max, Vec3::new(1.0, 0.5, 1.5));
    }

    #[test]
    fn test_sphere_compile() {
        let mesh = Sphere::default().compile();
        assert!(!mesh.positions.is_empty());
        assert!(!mesh.faces.is_empty());
    }

    #[test]
    fn test_cone_compile() {
        let mesh = Cone::default().compile();
        assert!(!mesh.positions.is_empty());
    }

    #[test]
    fn test_cylinder_compile() {
        let mesh = Cylinder::default().compile();
        assert!(!mesh.positions.is_empty());
    }

    #[test]
    fn test_mesh_from_raw() {
        let pos = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let idx = vec![0u32, 1, 2];
        let mesh = Mesh::from_raw(&pos, &idx);
        let compiled = mesh.compile();
        assert_eq!(compiled.faces.len(), 1);
    }

    #[test]
    fn test_lineset_from_strip() {
        let pts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let ls = LineSet::from_line_strip(&pts);
        assert_eq!(ls.indices.len(), 4); // 2 segments × 2 indices
        assert_eq!(ls.indices, vec![0, 1, 1, 2]);
    }

    #[test]
    fn test_shape_builder_transform() {
        let cube = Cube::default().at(1.0, 2.0, 3.0).scale(2.0, 2.0, 2.0);
        assert!(cube.has_transform());
    }

    #[test]
    fn test_shape_builder_material() {
        let mat = Material::pbr().base_color(1.0, 0.0, 0.0);
        let cube = Cube::default().material(mat.clone());
        assert!(cube.shape_material().is_some());
        assert_eq!(cube.shape_material().unwrap().base_color, Vec3::new(1.0, 0.0, 0.0));
    }
}
