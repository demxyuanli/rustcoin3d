//! Scene graph node payloads (cameras, geometry, lights).
//!
//! ## Depth convention (`PerspectiveCameraNode` / `OrthographicCameraNode`)
//!
//! Set [`PerspectiveCameraNode::reverse_depth`] or [`OrthographicCameraNode::reverse_depth`] to build
//! a projection whose clip-space Z ordering matches **reverse-Z** rendering (WebGPU:
//! `CompareFunction::Greater`, depth clear `0.0`, HZB max pyramid). Leave `false` for **forward-Z**
//! (`Less`, clear `1.0`, HZB min pyramid).
//!
//! `DrawCall::depth_reversed_z` (see `rc3d-render`) is derived from the active projection matrix via
//! `rc3d_core::depth_reversed_z_from_projection`; keep camera projection and GPU depth state aligned.
//! If one frame mixes draw calls built from incompatible projections, `render_draw_calls` logs a
//! warning and uses the first visible draw call for pipeline depth mode.

use std::sync::Arc;

use rc3d_core::math::{Mat4, Vec3, Vec4};
use rc3d_core::NodeId;

use crate::animation::{AnimationClip, Skeleton, VertexSkinData};
use crate::node_handler::NodeHandler;

/// Behavioral marker: saves/restores all state elements during traversal.
#[derive(Clone, Debug, Default)]
pub struct SeparatorNode;

/// Ordered container of children (no state save/restore).
#[derive(Clone, Debug, Default)]
pub struct GroupNode;

/// Stores vertex positions.
#[derive(Clone, Debug)]
pub struct Coordinate3Node {
    pub point: Vec<Vec3>,
}

impl Coordinate3Node {
    pub fn from_points(points: Vec<Vec3>) -> Self {
        Self { point: points }
    }
}

/// Per-vertex 2D texture coordinates (parallel to [`Coordinate3Node::point`] when used with IFS).
#[derive(Clone, Debug)]
pub struct TextureCoordinate2Node {
    pub point: Vec<[f32; 2]>,
}

impl TextureCoordinate2Node {
    pub fn from_points(points: Vec<[f32; 2]>) -> Self {
        Self { point: points }
    }
}

/// Stores per-vertex normals.
#[derive(Clone, Debug)]
pub struct NormalNode {
    pub vector: Vec<Vec3>,
}

impl NormalNode {
    pub fn from_vectors(vectors: Vec<Vec3>) -> Self {
        Self { vector: vectors }
    }
}

/// Stores material properties with full PBR support.
#[derive(Clone, Debug)]
pub struct MaterialNode {
    pub diffuse_color: Vec3,
    pub ambient_color: Vec3,
    pub specular_color: Vec3,
    pub shininess: f32,
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub albedo_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub opacity: f32,
    pub emissive_color: Vec3,
    pub emissive_texture: Option<String>,
    /// Packed Occlusion/Roughness/Metallic texture (glTF ORM convention).
    pub metallic_roughness_texture: Option<String>,
    pub occlusion_texture: Option<String>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

/// Alpha rendering mode following glTF conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

impl Default for AlphaMode {
    fn default() -> Self {
        Self::Opaque
    }
}

impl MaterialNode {
    pub fn from_diffuse(diffuse: Vec3) -> Self {
        Self {
            diffuse_color: diffuse,
            ambient_color: diffuse * 0.2,
            specular_color: Vec3::new(0.5, 0.5, 0.5),
            shininess: 32.0,
            base_color: diffuse,
            metallic: 0.0,
            roughness: 0.5,
            albedo_texture: None,
            normal_texture: None,
            opacity: 1.0,
            emissive_color: Vec3::ZERO,
            emissive_texture: None,
            metallic_roughness_texture: None,
            occlusion_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

impl Default for MaterialNode {
    fn default() -> Self {
        Self {
            diffuse_color: Vec3::new(0.8, 0.8, 0.8),
            ambient_color: Vec3::new(0.2, 0.2, 0.2),
            specular_color: Vec3::new(0.0, 0.0, 0.0),
            shininess: 0.0,
            base_color: Vec3::new(0.8, 0.8, 0.8),
            metallic: 0.0,
            roughness: 0.5,
            albedo_texture: None,
            normal_texture: None,
            opacity: 1.0,
            emissive_color: Vec3::ZERO,
            emissive_texture: None,
            metallic_roughness_texture: None,
            occlusion_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

/// 3D transformation: translation, rotation, scale.
#[derive(Clone, Debug)]
pub struct TransformNode {
    pub translation: Vec3,
    pub rotation: Mat4,
    pub scale: Vec3,
    pub center: Vec3,
}

impl Default for TransformNode {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Mat4::IDENTITY,
            scale: Vec3::ONE,
            center: Vec3::ZERO,
        }
    }
}

impl TransformNode {
    pub fn from_translation(t: Vec3) -> Self {
        Self {
            translation: t,
            ..Default::default()
        }
    }

    pub fn to_matrix(&self) -> Mat4 {
        let c = Mat4::from_translation(self.center);
        let ci = Mat4::from_translation(-self.center);
        let t = Mat4::from_translation(self.translation);
        let s = Mat4::from_scale(self.scale);
        t * c * self.rotation * s * ci
    }
}

/// Shape: renders the first 3 coordinates as a triangle.
#[derive(Clone, Debug, Default)]
pub struct TriangleNode;

/// Shape: axis-aligned box.
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
pub struct SphereNode {
    pub radius: f32,
}

impl Default for SphereNode {
    fn default() -> Self {
        Self { radius: 1.0 }
    }
}

/// Shape: cone.
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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

/// Shape: arbitrary triangle mesh from vertex/index arrays.
#[derive(Clone, Debug, Default)]
pub struct IndexedFaceSetNode {
    pub coord_index: Vec<i32>,
}

/// Skeletal skinning data for mesh geometry under the same separator (Coin-style sidecar).
/// Place before [`Coordinate3`](Coordinate3Node) / [`IndexedFaceSet`](IndexedFaceSetNode) so the
/// render collector can attach weights to the generated draw call.
#[derive(Clone, Debug)]
pub struct SkinnedMeshNode {
    pub skeleton: Skeleton,
    pub skin_data: Vec<VertexSkinData>,
    pub clip: Option<AnimationClip>,
}

/// A single morph target (blend shape) storing per-vertex deltas.
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug, Default)]
pub struct MorphTargetNode {
    /// The morph targets (blend shapes) for this mesh.
    pub targets: Vec<MorphTarget>,
    /// Current weight for each target (same length as targets).
    pub weights: Vec<f32>,
}

/// Camera with perspective projection.
#[derive(Clone, Debug)]
pub struct PerspectiveCameraNode {
    pub position: Vec3,
    pub orientation: Mat4,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
    /// When `true`, builds reverse-Z clip mapping (near plane -> ndc_z ~ 1, far -> ~0). Match WebGPU
    /// depth test `Greater` and clear `0.0`.
    pub reverse_depth: bool,
}

impl Default for PerspectiveCameraNode {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            orientation: Mat4::IDENTITY,
            fov: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 100.0,
            aspect: 1.0,
            reverse_depth: false,
        }
    }
}

impl PerspectiveCameraNode {
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3, fov: f32, aspect: f32) -> Self {
        Self {
            position: eye,
            orientation: Mat4::look_at_rh(eye, target, up),
            fov,
            near: 0.1,
            far: 100.0,
            aspect,
            reverse_depth: false,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        self.orientation
    }

    pub fn projection_matrix(&self) -> Mat4 {
        // WebGPU clip Z in [0, w]; xy unchanged vs OpenGL NDC.
        let fov_clamped = self.fov.clamp(f32::EPSILON, std::f32::consts::PI - f32::EPSILON);
        let f = 1.0 / (fov_clamped * 0.5).tan();
        let d = (self.far - self.near).max(f32::EPSILON);
        let aspect = self.aspect.max(f32::EPSILON);
        if self.reverse_depth {
            let a = self.near / d;
            let b = self.near * self.far / d;
            Mat4::from_cols(
                Vec4::new(f / aspect, 0.0, 0.0, 0.0),
                Vec4::new(0.0, f, 0.0, 0.0),
                Vec4::new(0.0, 0.0, a, -1.0),
                Vec4::new(0.0, 0.0, b, 0.0),
            )
        } else {
            let nf = 1.0 / (self.near - self.far);
            Mat4::from_cols(
                Vec4::new(f / aspect, 0.0, 0.0, 0.0),
                Vec4::new(0.0, f, 0.0, 0.0),
                Vec4::new(0.0, 0.0, self.far * nf, -1.0),
                Vec4::new(0.0, 0.0, self.near * self.far * nf, 0.0),
            )
        }
    }
}

/// Camera with orthographic projection.
#[derive(Clone, Debug)]
pub struct OrthographicCameraNode {
    pub position: Vec3,
    pub orientation: Mat4,
    pub height: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
    /// When `true`, ndc_z increases toward the near plane (reverse-Z); pair with `Greater` + clear `0`.
    pub reverse_depth: bool,
}

impl Default for OrthographicCameraNode {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            orientation: Mat4::IDENTITY,
            height: 2.0,
            near: 0.1,
            far: 100.0,
            aspect: 1.0,
            reverse_depth: false,
        }
    }
}

impl OrthographicCameraNode {
    pub fn view_matrix(&self) -> Mat4 {
        self.orientation
    }

    pub fn projection_matrix(&self) -> Mat4 {
        let half_h = self.height.max(f32::EPSILON) / 2.0;
        let half_w = half_h * self.aspect;
        let rml = half_w * 2.0;
        let tmb = half_h * 2.0;
        let fmn = self.far - self.near;
        if self.reverse_depth {
            Mat4::from_cols(
                Vec4::new(2.0 / rml, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 2.0 / tmb, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0 / fmn, 0.0),
                Vec4::new(0.0, 0.0, self.far / fmn, 1.0),
            )
        } else {
            Mat4::from_cols(
                Vec4::new(2.0 / rml, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 2.0 / tmb, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0 / (self.near - self.far), 0.0),
                Vec4::new(0.0, 0.0, -self.near / fmn, 1.0),
            )
        }
    }
}

/// Directional (infinite) light.
#[derive(Clone, Debug)]
pub struct DirectionalLightNode {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl Default for DirectionalLightNode {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::ONE,
            intensity: 1.0,
        }
    }
}

/// Point light.
#[derive(Clone, Debug)]
pub struct PointLightNode {
    pub location: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl Default for PointLightNode {
    fn default() -> Self {
        Self {
            location: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 1.0,
        }
    }
}

/// Spot light.
#[derive(Clone, Debug)]
pub struct SpotLightNode {
    pub location: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub cut_off_angle: f32,
    pub drop_off_rate: f32,
}

impl Default for SpotLightNode {
    fn default() -> Self {
        Self {
            location: Vec3::ZERO,
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::ONE,
            intensity: 1.0,
            cut_off_angle: 0.785,
            drop_off_rate: 0.0,
        }
    }
}

/// Event callback node: marker for scene-graph event routing.
///
/// When HandleEventAction encounters this node during traversal,
/// the application-level handler decides whether to consume the event.
/// The `enabled` flag controls whether the node participates in routing.
#[derive(Clone, Debug)]
pub struct EventCallbackNode {
    pub enabled: bool,
}

impl Default for EventCallbackNode {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Pick style: controls whether this node (and its children) can be picked.
#[derive(Clone, Debug)]
pub struct PickStyleNode {
    pub pickable: bool,
}

impl Default for PickStyleNode {
    fn default() -> Self {
        Self { pickable: true }
    }
}

/// One LOD level: a group of children rendered at this detail level.
#[derive(Clone, Debug)]
pub struct LodLevel {
    pub children: Vec<NodeId>,
    pub max_distance: f32,
}

/// LOD switch node (Coin3D SoLOD / SoLevelOfDetail pattern).
/// Selects one child group based on camera distance.
#[derive(Clone, Debug)]
pub struct LodNode {
    pub levels: Vec<LodLevel>,
    pub current_level: usize,
}

impl Default for LodNode {
    fn default() -> Self {
        Self {
            levels: Vec::new(),
            current_level: 0,
        }
    }
}

/// Section/cutting plane node (Coin3D SoClipPlane pattern).
#[derive(Clone, Debug)]
pub struct SectionPlaneNode {
    pub plane: [f32; 4],
    pub enabled: bool,
}

impl Default for SectionPlaneNode {
    fn default() -> Self {
        Self { plane: [0.0, 1.0, 0.0, 0.0], enabled: true }
    }
}

/// Switch node: traverses one child based on index (Coin3D SoSwitch pattern).
/// which_child: -1 = all, -2 = none, 0..N = specific child.
#[derive(Clone, Debug)]
pub struct SwitchNode {
    pub which_child: i32,
    pub children: Vec<rc3d_core::NodeId>,
}

impl Default for SwitchNode {
    fn default() -> Self {
        Self { which_child: -1, children: Vec::new() }
    }
}

/// MultipleCopy node: repeats child traversal with offset transforms
/// (Coin3D SoMultipleCopy pattern).
#[derive(Clone, Debug)]
pub struct MultipleCopyNode {
    pub copies: Vec<rc3d_core::math::Mat4>,
    pub children: Vec<rc3d_core::NodeId>,
}

impl Default for MultipleCopyNode {
    fn default() -> Self {
        Self { copies: Vec::new(), children: Vec::new() }
    }
}

/// Screen-space 2D text label (Coin3D SoText2 pattern).
#[derive(Clone, Debug)]
pub struct Text2Node {
    pub string: String,
    pub position: [f32; 2],
    pub size: f32,
    pub color: [f32; 4],
}

impl Default for Text2Node {
    fn default() -> Self {
        Self { string: String::new(), position: [0.0, 0.0], size: 16.0, color: [1.0, 1.0, 1.0, 1.0] }
    }
}

/// World-space 3D text label (Coin3D SoText3 pattern).
#[derive(Clone, Debug)]
pub struct Text3Node {
    pub string: String,
    pub position: Vec3,
    pub size: f32,
    pub color: [f32; 4],
}

impl Default for Text3Node {
    fn default() -> Self {
        Self { string: String::new(), position: Vec3::ZERO, size: 16.0, color: [1.0, 1.0, 1.0, 1.0] }
    }
}

/// Central node type enum.
#[derive(Clone, Debug)]
pub enum MeasurementType {
    Distance,
    Angle,
    Radius,
    Diameter,
}

#[derive(Clone, Debug)]
pub struct MeasurementNode {
    pub points: Vec<rc3d_core::math::Vec3>,
    pub measurement_type: MeasurementType,
    pub label: String,
    pub color: [f32; 4],
    pub value: f32,
}

impl Default for MeasurementNode {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            measurement_type: MeasurementType::Distance,
            label: String::new(),
            color: [1.0, 1.0, 0.0, 1.0],
            value: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub enum MarkupElement {
    Line {
        start: [f32; 2],
        end: [f32; 2],
        color: [f32; 4],
        width: f32,
    },
    Rect {
        origin: [f32; 2],
        size: [f32; 2],
        color: [f32; 4],
        filled: bool,
    },
    Circle {
        center: [f32; 2],
        radius: f32,
        color: [f32; 4],
    },
    Freehand {
        points: Vec<[f32; 2]>,
        color: [f32; 4],
        width: f32,
    },
    Text {
        position: [f32; 2],
        string: String,
        size: f32,
        color: [f32; 4],
    },
    /// Dimension line with extension lines and arrowheads.
    Dimension {
        start: [f32; 2],
        end: [f32; 2],
        /// Offset direction (normalized) for extension lines.
        offset_dir: [f32; 2],
        /// Extension line length.
        extension_len: f32,
        /// Arrowhead size.
        arrow_size: f32,
        /// Measurement label (e.g., "12.34 m").
        label: String,
        color: [f32; 4],
    },
}

#[derive(Clone, Debug)]
pub struct MarkupNode {
    pub elements: Vec<MarkupElement>,
    pub layer_name: String,
    pub visible: bool,
}

impl Default for MarkupNode {
    fn default() -> Self {
        Self {
            elements: Vec::new(),
            layer_name: String::new(),
            visible: true,
        }
    }
}

/// Central node type enum.
#[derive(Clone, Debug)]
pub enum NodeData {
    // Grouping
    Separator(SeparatorNode),
    Group(GroupNode),
    // Properties
    Transform(TransformNode),
    Coordinate3(Coordinate3Node),
    TextureCoordinate2(TextureCoordinate2Node),
    Normal(NormalNode),
    Material(MaterialNode),
    // Shapes
    Triangle(TriangleNode),
    Cube(CubeNode),
    Sphere(SphereNode),
    Cone(ConeNode),
    Cylinder(CylinderNode),
    IndexedFaceSet(IndexedFaceSetNode),
    SkinnedMesh(SkinnedMeshNode),
    MorphTarget(MorphTargetNode),
    // Cameras
    PerspectiveCamera(PerspectiveCameraNode),
    OrthographicCamera(OrthographicCameraNode),
    // Lights
    DirectionalLight(DirectionalLightNode),
    PointLight(PointLightNode),
    SpotLight(SpotLightNode),
    /// User-defined behavior via [`NodeHandler`] (traversal / future collect hooks).
    HandlerNode(Arc<dyn NodeHandler>),
    /// Event routing callback (Coin3D SoEventCallback pattern).
    EventCallback(EventCallbackNode),
    /// Pick style: controls node pickability (Coin3D SoPickStyle pattern).
    PickStyle(PickStyleNode),
    /// Level-of-detail switch (Coin3D SoLOD pattern).
    Lod(LodNode),
    Switch(SwitchNode),
    MultipleCopy(MultipleCopyNode),
    /// Section/cutting plane (Coin3D SoClipPlane pattern).
    SectionPlane(SectionPlaneNode),
    /// Screen-space 2D text label (Coin3D SoText2 pattern).
    Text2(Text2Node),
    /// World-space 3D text label (Coin3D SoText3 pattern).
    Text3(Text3Node),
    Measurement(MeasurementNode),
    Markup(MarkupNode),
}

/// Describes a named field on a node type.
#[derive(Clone, Debug)]
pub struct FieldDescriptor {
    pub name: &'static str,
    pub field_index: u16,
}

impl NodeData {
    /// Returns the fields exposed by this node type.
    pub fn field_descriptors(&self) -> Vec<FieldDescriptor> {
        match self {
            NodeData::Transform(_) => vec![
                FieldDescriptor { name: "translation", field_index: 0 },
                FieldDescriptor { name: "rotation", field_index: 1 },
                FieldDescriptor { name: "scale", field_index: 2 },
                FieldDescriptor { name: "center", field_index: 3 },
            ],
            NodeData::Material(_) => vec![
                FieldDescriptor { name: "diffuseColor", field_index: 0 },
                FieldDescriptor { name: "specularColor", field_index: 1 },
                FieldDescriptor { name: "shininess", field_index: 2 },
                FieldDescriptor { name: "opacity", field_index: 3 },
            ],
            NodeData::DirectionalLight(_) => vec![
                FieldDescriptor { name: "direction", field_index: 0 },
                FieldDescriptor { name: "color", field_index: 1 },
                FieldDescriptor { name: "intensity", field_index: 2 },
            ],
            NodeData::PointLight(_) => vec![
                FieldDescriptor { name: "location", field_index: 0 },
                FieldDescriptor { name: "color", field_index: 1 },
                FieldDescriptor { name: "intensity", field_index: 2 },
                FieldDescriptor { name: "cutoff_distance", field_index: 3 },
            ],
            NodeData::SpotLight(_) => vec![
                FieldDescriptor { name: "location", field_index: 0 },
                FieldDescriptor { name: "direction", field_index: 1 },
                FieldDescriptor { name: "color", field_index: 2 },
                FieldDescriptor { name: "intensity", field_index: 3 },
                FieldDescriptor { name: "cut_off_angle", field_index: 4 },
                FieldDescriptor { name: "drop_off_rate", field_index: 5 },
            ],
            NodeData::PerspectiveCamera(_) => vec![
                FieldDescriptor { name: "fov", field_index: 0 },
                FieldDescriptor { name: "near", field_index: 1 },
                FieldDescriptor { name: "far", field_index: 2 },
                FieldDescriptor { name: "reverse_depth", field_index: 3 },
            ],
            NodeData::OrthographicCamera(_) => vec![
                FieldDescriptor { name: "height", field_index: 0 },
                FieldDescriptor { name: "near", field_index: 1 },
                FieldDescriptor { name: "far", field_index: 2 },
                FieldDescriptor { name: "reverse_depth", field_index: 3 },
            ],
            NodeData::SectionPlane(_) => vec![
                FieldDescriptor { name: "plane", field_index: 0 },
                FieldDescriptor { name: "enabled", field_index: 1 },
            ],
            NodeData::Lod(_) => vec![
                FieldDescriptor { name: "current_level", field_index: 0 },
            ],
            NodeData::Switch(_) => vec![
                FieldDescriptor { name: "which_child", field_index: 0 },
            ],
            NodeData::Text2(_) => vec![
                FieldDescriptor { name: "string", field_index: 0 },
                FieldDescriptor { name: "position", field_index: 1 },
                FieldDescriptor { name: "size", field_index: 2 },
                FieldDescriptor { name: "color", field_index: 3 },
            ],
            NodeData::Text3(_) => vec![
                FieldDescriptor { name: "string", field_index: 0 },
                FieldDescriptor { name: "position", field_index: 1 },
                FieldDescriptor { name: "size", field_index: 2 },
                FieldDescriptor { name: "color", field_index: 3 },
            ],
            NodeData::EventCallback(_) => vec![
                FieldDescriptor { name: "enabled", field_index: 0 },
            ],
            NodeData::PickStyle(_) => vec![
                FieldDescriptor { name: "pickable", field_index: 0 },
            ],
            NodeData::Markup(_) => vec![
                FieldDescriptor { name: "visible", field_index: 0 },
                FieldDescriptor { name: "layer_name", field_index: 1 },
            ],
            NodeData::Measurement(_) => vec![
                FieldDescriptor { name: "value", field_index: 0 },
                FieldDescriptor { name: "label", field_index: 1 },
                FieldDescriptor { name: "color", field_index: 2 },
            ],
            // Nodes with no runtime fields
            NodeData::Separator(_)
            | NodeData::Group(_)
            | NodeData::Coordinate3(_)
            | NodeData::TextureCoordinate2(_)
            | NodeData::Normal(_)
            | NodeData::Triangle(_)
            | NodeData::Cube(_)
            | NodeData::Sphere(_)
            | NodeData::Cone(_)
            | NodeData::Cylinder(_)
            | NodeData::IndexedFaceSet(_)
            | NodeData::SkinnedMesh(_)
            | NodeData::MorphTarget(_)
            | NodeData::HandlerNode(_)
            | NodeData::MultipleCopy(_) => vec![],
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            NodeData::Separator(_) => "Separator",
            NodeData::Group(_) => "Group",
            NodeData::Transform(_) => "Transform",
            NodeData::Coordinate3(_) => "Coordinate3",
            NodeData::TextureCoordinate2(_) => "TextureCoordinate2",
            NodeData::Normal(_) => "Normal",
            NodeData::Material(_) => "Material",
            NodeData::Triangle(_) => "Triangle",
            NodeData::Cube(_) => "Cube",
            NodeData::Sphere(_) => "Sphere",
            NodeData::Cone(_) => "Cone",
            NodeData::Cylinder(_) => "Cylinder",
            NodeData::IndexedFaceSet(_) => "IndexedFaceSet",
            NodeData::SkinnedMesh(_) => "SkinnedMesh",
            NodeData::MorphTarget(_) => "MorphTarget",
            NodeData::PerspectiveCamera(_) => "PerspectiveCamera",
            NodeData::OrthographicCamera(_) => "OrthographicCamera",
            NodeData::DirectionalLight(_) => "DirectionalLight",
            NodeData::PointLight(_) => "PointLight",
            NodeData::SpotLight(_) => "SpotLight",
            NodeData::HandlerNode(h) => h.handler_name(),
            NodeData::EventCallback(_) => "EventCallback",
            NodeData::PickStyle(_) => "PickStyle",
            NodeData::Lod(_) => "Lod",
            NodeData::Switch(_) => "Switch",
            NodeData::MultipleCopy(_) => "MultipleCopy",
            NodeData::SectionPlane(_) => "SectionPlane",
            NodeData::Text2(_) => "Text2",
            NodeData::Text3(_) => "Text3",
            NodeData::Measurement(_) => "Measurement",
            NodeData::Markup(_) => "Markup",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── field_descriptors completeness ──

    #[test]
    fn test_material_descriptors() {
        let d = NodeData::Material(MaterialNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"diffuseColor"));
        assert!(names.contains(&"opacity"));
        assert!(!d.is_empty());
    }

    #[test]
    fn test_directional_light_descriptors() {
        let d = NodeData::DirectionalLight(DirectionalLightNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"direction"));
        assert!(names.contains(&"color"));
        assert!(names.contains(&"intensity"));
    }

    #[test]
    fn test_spot_light_descriptors() {
        let d = NodeData::SpotLight(SpotLightNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"cut_off_angle"));
        assert!(names.contains(&"drop_off_rate"));
    }

    #[test]
    fn test_perspective_camera_descriptors() {
        let d = NodeData::PerspectiveCamera(PerspectiveCameraNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"fov"));
        assert!(names.contains(&"near"));
        assert!(names.contains(&"far"));
        assert!(names.contains(&"reverse_depth"));
    }

    #[test]
    fn test_section_plane_descriptors() {
        let d = NodeData::SectionPlane(SectionPlaneNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"plane"));
        assert!(names.contains(&"enabled"));
    }

    #[test]
    fn test_markup_descriptors() {
        let d = NodeData::Markup(MarkupNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"visible"));
        assert!(names.contains(&"layer_name"));
    }

    #[test]
    fn test_measurement_descriptors() {
        let d = NodeData::Measurement(MeasurementNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"value"));
        assert!(names.contains(&"label"));
        assert!(names.contains(&"color"));
    }

    #[test]
    fn test_shape_nodes_have_no_descriptors() {
        // Structural/geometry nodes should return empty field lists
        assert!(NodeData::Separator(SeparatorNode).field_descriptors().is_empty());
        assert!(NodeData::Group(GroupNode).field_descriptors().is_empty());
        assert!(NodeData::Cube(CubeNode::default()).field_descriptors().is_empty());
        assert!(NodeData::Cylinder(CylinderNode::default()).field_descriptors().is_empty());
        assert!(NodeData::IndexedFaceSet(IndexedFaceSetNode::default())
            .field_descriptors()
            .is_empty());
    }

    #[test]
    fn test_transform_descriptors() {
        let d = NodeData::Transform(TransformNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"translation"));
        assert!(names.contains(&"rotation"));
        assert!(names.contains(&"scale"));
    }

    // ── type_name completeness ──

    #[test]
    fn test_type_name_is_consistent() {
        // Each variant's type_name should match the variant name
        assert_eq!(NodeData::Separator(SeparatorNode).type_name(), "Separator");
        assert_eq!(NodeData::Material(MaterialNode::default()).type_name(), "Material");
        assert_eq!(
            NodeData::PerspectiveCamera(PerspectiveCameraNode::default()).type_name(),
            "PerspectiveCamera"
        );
        assert_eq!(
            NodeData::DirectionalLight(DirectionalLightNode::default()).type_name(),
            "DirectionalLight"
        );
        assert_eq!(NodeData::SectionPlane(SectionPlaneNode::default()).type_name(), "SectionPlane");
        assert_eq!(NodeData::Markup(MarkupNode::default()).type_name(), "Markup");
        assert_eq!(
            NodeData::Measurement(MeasurementNode::default()).type_name(),
            "Measurement"
        );
    }

    #[test]
    fn test_field_descriptor_indices_are_sequential() {
        let d = NodeData::SpotLight(SpotLightNode::default()).field_descriptors();
        for (i, fd) in d.iter().enumerate() {
            assert_eq!(fd.field_index, i as u16, "field {} should have index {}", fd.name, i);
        }
    }
}
