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

/// Stores material properties.
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
    pub opacity: f32,
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
            opacity: 1.0,
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
            opacity: 1.0,
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
        ci * s * self.rotation * c * t
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
        let f = 1.0 / (self.fov * 0.5).tan();
        if self.reverse_depth {
            let d = self.far - self.near;
            let a = self.near / d;
            let b = self.near * self.far / d;
            Mat4::from_cols(
                Vec4::new(f / self.aspect, 0.0, 0.0, 0.0),
                Vec4::new(0.0, f, 0.0, 0.0),
                Vec4::new(0.0, 0.0, a, -1.0),
                Vec4::new(0.0, 0.0, b, 0.0),
            )
        } else {
            let nf = 1.0 / (self.near - self.far);
            Mat4::from_cols(
                Vec4::new(f / self.aspect, 0.0, 0.0, 0.0),
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
        let half_h = self.height / 2.0;
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
}

impl NodeData {
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
            NodeData::PerspectiveCamera(_) => "PerspectiveCamera",
            NodeData::OrthographicCamera(_) => "OrthographicCamera",
            NodeData::DirectionalLight(_) => "DirectionalLight",
            NodeData::PointLight(_) => "PointLight",
            NodeData::SpotLight(_) => "SpotLight",
            NodeData::HandlerNode(h) => h.handler_name(),
            NodeData::EventCallback(_) => "EventCallback",
            NodeData::Lod(_) => "Lod",
            NodeData::Switch(_) => "Switch",
            NodeData::MultipleCopy(_) => "MultipleCopy",
            NodeData::SectionPlane(_) => "SectionPlane",
            NodeData::Text2(_) => "Text2",
            NodeData::Text3(_) => "Text3",
        }
    }
}
