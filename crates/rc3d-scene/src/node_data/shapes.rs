//! Shape and mesh nodes.
use rc3d_core::math::Vec3;
use serde::{Deserialize, Serialize};
use crate::animation::{AnimationClip, Skeleton, VertexSkinData};
use super::properties::MaterialNode;

/// Shape: renders the first 3 coordinates as a triangle.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TriangleNode;

/// Shape: axis-aligned box.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CubeNode {
    pub width: f32,
    pub height: f32,
    pub depth: f32,
}

impl Default for CubeNode {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 1.0,
            depth: 1.0,
        }
    }
}

/// Shape: UV sphere.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SphereNode {
    pub radius: f32,
}

impl Default for SphereNode {
    fn default() -> Self {
        Self { radius: 1.0 }
    }
}

/// Shape: cone.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConeNode {
    pub bottom_radius: f32,
    pub height: f32,
}

impl Default for ConeNode {
    fn default() -> Self {
        Self {
            bottom_radius: 1.0,
            height: 2.0,
        }
    }
}

/// Shape: cylinder.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CylinderNode {
    pub radius: f32,
    pub height: f32,
}

impl Default for CylinderNode {
    fn default() -> Self {
        Self {
            radius: 1.0,
            height: 2.0,
        }
    }
}

/// Shape: torus (donut).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TorusNode {
    /// Distance from the center of the torus to the center of the tube.
    pub major_radius: f32,
    /// Radius of the tube.
    pub minor_radius: f32,
}

impl Default for TorusNode {
    fn default() -> Self {
        Self {
            major_radius: 1.0,
            minor_radius: 0.25,
        }
    }
}

/// Shape: arbitrary triangle mesh from vertex/index arrays.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct IndexedFaceSetNode {
    pub coord_index: Vec<i32>,
    /// three.js BufferGeometry.groups: ranges of tessellated indices with a material slot.
    /// Empty = single draw using the traversal material.
    #[serde(default)]
    pub material_groups: Vec<FaceMaterialGroup>,
    /// Palette indexed by [`FaceMaterialGroup::material_index`]. Empty = traversal material.
    #[serde(default)]
    pub materials: Vec<MaterialNode>,
    /// Per-tessellated-triangle CAD face id. Empty = triangle index is the face id.
    #[serde(default)]
    pub face_ids: Vec<u32>,
}

impl IndexedFaceSetNode {
    pub fn from_coord_index(coord_index: Vec<i32>) -> Self {
        Self {
            coord_index,
            ..Default::default()
        }
    }

    /// CAD face id for tessellated triangle `tri`.
    pub fn face_id(&self, tri: u32) -> u32 {
        self.face_ids
            .get(tri as usize)
            .copied()
            .unwrap_or(tri)
    }
}

/// One material range on an [`IndexedFaceSetNode`] (index-buffer `start`/`count`, like three.js).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaceMaterialGroup {
    /// First element in the tessellated triangle index buffer.
    pub start: u32,
    /// Number of indices in this group (multiple of 3).
    pub count: u32,
    /// Index into [`IndexedFaceSetNode::materials`].
    pub material_index: u32,
}

impl FaceMaterialGroup {
    /// Compact consecutive per-triangle material slots into index-buffer groups.
    pub fn compact_from_triangle_slots(slots: &[u32]) -> Vec<Self> {
        if slots.is_empty() {
            return Vec::new();
        }
        let mut groups = Vec::new();
        let mut start_tri = 0u32;
        let mut current = slots[0];
        for (i, &slot) in slots.iter().enumerate().skip(1) {
            if slot != current {
                groups.push(Self {
                    start: start_tri * 3,
                    count: (i as u32 - start_tri) * 3,
                    material_index: current,
                });
                start_tri = i as u32;
                current = slot;
            }
        }
        groups.push(Self {
            start: start_tri * 3,
            count: (slots.len() as u32 - start_tri) * 3,
            material_index: current,
        });
        groups
    }
}

/// Shape: line segments from vertex/index arrays (Coin3D SoIndexedLineSet).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IndexedLineSetNode {
    pub coord_index: Vec<i32>,
    pub line_width: f32,
    /// RGBA color for the edge overlay pass.
    #[serde(default = "default_indexed_line_color")]
    pub color: [f32; 4],
}

fn default_indexed_line_color() -> [f32; 4] {
    [0.5, 0.5, 0.5, 1.0]
}

impl Default for IndexedLineSetNode {
    fn default() -> Self {
        Self {
            coord_index: Vec::new(),
            line_width: 1.0,
            color: default_indexed_line_color(),
        }
    }
}

/// Skeletal skinning data for mesh geometry under the same separator (Coin-style sidecar).
/// Place before [`Coordinate3`](Coordinate3Node) / [`IndexedFaceSet`](IndexedFaceSetNode) so the
/// render collector can attach weights to the generated draw call.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SkinnedMeshNode {
    pub skeleton: Skeleton,
    pub skin_data: Vec<VertexSkinData>,
    pub clip: Option<AnimationClip>,
}

/// A single morph target (blend shape) storing per-vertex deltas.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MorphTarget {
    pub name: String,
    /// Per-vertex position deltas (same length as Coordinate3Node::point).
    pub position_deltas: Vec<Vec3>,
    /// Per-vertex normal deltas (same length as NormalNode::vector, if present).
    pub normal_deltas: Option<Vec<Vec3>>,
    /// Per-vertex tangent deltas (same length as the tangent array, if present).
    pub tangent_deltas: Option<Vec<[f32; 4]>>,
}

/// Stores morph target (blend shape) data and per-instance weights.
/// Parent this node alongside the geometry it affects within a Separator.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct MorphTargetNode {
    /// The morph targets (blend shapes) for this mesh.
    pub targets: Vec<MorphTarget>,
    /// Current weight for each target (same length as targets).
    pub weights: Vec<f32>,
}

/// GPU-instanced mesh: renders one reference geometry N times with per-instance transforms.
/// Each transform is a 4×4 model matrix in column-major order.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct InstancedMeshNode {
    /// Instance model matrices (column-major 4×4).
    pub transforms: Vec<[[f32; 4]; 4]>,
}

/// One packed geometry range inside a [`BatchedMeshNode`] (three.js `BatchedMesh` geometry id).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct BatchedGeometry {
    /// First index in [`BatchedMeshNode::indices`].
    pub index_first: u32,
    /// Index count (multiple of 3).
    pub index_count: u32,
}

/// One drawable instance of a packed geometry.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BatchedInstance {
    pub geometry: u32,
    /// Local model matrix, column-major 4×4.
    pub transform: [[f32; 4]; 4],
    pub visible: bool,
    /// Multiplies the traversal material (`[1,1,1,1]` = unchanged).
    pub color: [f32; 4],
}

impl Default for BatchedInstance {
    fn default() -> Self {
        Self {
            geometry: 0,
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            visible: true,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Multi-geometry GPU batch (three.js `BatchedMesh`).
///
/// Distinct meshes share one vertex/index buffer; each instance draws a geometry
/// range with its own transform. Same-range instances batch through the existing
/// opaque instancing / multi-draw path.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct BatchedMeshNode {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub texcoords: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub geometries: Vec<BatchedGeometry>,
    pub instances: Vec<BatchedInstance>,
}

impl BatchedMeshNode {
    /// Append a geometry and return its id.
    pub fn add_geometry(
        &mut self,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        texcoords: &[[f32; 2]],
        tangents: &[[f32; 4]],
        indices: &[u32],
    ) -> u32 {
        let v_off = self.positions.len() as u32;
        self.positions.extend_from_slice(positions);
        if normals.len() == positions.len() {
            self.normals.extend_from_slice(normals);
        } else {
            self.normals
                .extend(std::iter::repeat([0.0, 1.0, 0.0]).take(positions.len()));
        }
        if texcoords.len() == positions.len() {
            self.texcoords.extend_from_slice(texcoords);
        } else {
            self.texcoords
                .extend(std::iter::repeat([0.0, 0.0]).take(positions.len()));
        }
        if tangents.len() == positions.len() {
            self.tangents.extend_from_slice(tangents);
        } else {
            self.tangents
                .extend(std::iter::repeat([1.0, 0.0, 0.0, 1.0]).take(positions.len()));
        }
        let i_off = self.indices.len() as u32;
        self.indices.extend(indices.iter().map(|i| i + v_off));
        let id = self.geometries.len() as u32;
        self.geometries.push(BatchedGeometry {
            index_first: i_off,
            index_count: indices.len() as u32,
        });
        id
    }

    /// Append an instance of `geometry` and return its id.
    pub fn add_instance(&mut self, geometry: u32, transform: [[f32; 4]; 4]) -> u32 {
        let id = self.instances.len() as u32;
        self.instances.push(BatchedInstance {
            geometry,
            transform,
            visible: true,
            color: [1.0, 1.0, 1.0, 1.0],
        });
        id
    }
}

