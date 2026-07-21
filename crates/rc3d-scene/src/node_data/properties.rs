//! Property nodes: coordinates, material, transform, hints, bindings.
use rc3d_core::math::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

/// Stores vertex positions.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Coordinate3Node {
    pub point: Vec<Vec3>,
}

impl Coordinate3Node {
    pub fn from_points(points: Vec<Vec3>) -> Self {
        Self { point: points }
    }
}

/// Per-vertex 2D texture coordinates (parallel to [`Coordinate3Node::point`] when used with IFS).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextureCoordinate2Node {
    pub point: Vec<[f32; 2]>,
}

impl TextureCoordinate2Node {
    pub fn from_points(points: Vec<[f32; 2]>) -> Self {
        Self { point: points }
    }
}

/// Stores per-vertex normals.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NormalNode {
    pub vector: Vec<Vec3>,
}

impl NormalNode {
    pub fn from_vectors(vectors: Vec<Vec3>) -> Self {
        Self { vector: vectors }
    }
}

/// Stores material properties with full PBR support.
#[derive(Serialize, Deserialize, Clone, Debug)]
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
    /// Anisotropic roughness (0.0 = isotropic GGX, 1.0 = fully anisotropic).
    pub anisotropic: f32,
    pub light_group: Option<String>,
}

/// Alpha rendering mode following glTF conventions.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[derive(Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
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
            anisotropic: 0.0,
            light_group: None,
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
            anisotropic: 0.0,
            light_group: None,
        }
    }
}

/// 3D transformation: translation, rotation, scale.
#[derive(Serialize, Deserialize, Clone, Debug)]
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
        Self { translation: t, ..Default::default() }
    }
    pub fn from_rotation(r: Mat4) -> Self {
        Self { rotation: r, ..Default::default() }
    }
    pub fn from_scale(s: Vec3) -> Self {
        Self { scale: s, ..Default::default() }
    }
    pub fn from_trs(t: Vec3, r: Mat4, s: Vec3) -> Self {
        Self { translation: t, rotation: r, scale: s, ..Default::default() }
    }

    pub fn to_matrix(&self) -> Mat4 {
        let c = Mat4::from_translation(self.center);
        let ci = Mat4::from_translation(-self.center);
        let t = Mat4::from_translation(self.translation);
        let s = Mat4::from_scale(self.scale);
        t * c * self.rotation * s * ci
    }
}

/// Global environment settings (Coin3D SoEnvironment).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EnvironmentNode {
    pub ambient_intensity: f32,
    pub ambient_color: Vec3,
    pub attenuation: Vec3,
    pub fog_color: Vec3,
    pub fog_visibility: f32,
}
impl Default for EnvironmentNode {
    fn default() -> Self {
        Self { ambient_intensity: 0.2, ambient_color: Vec3::ONE, attenuation: Vec3::new(0.0, 0.0, 1.0), fog_color: Vec3::ONE, fog_visibility: 0.0 }
    }
}

/// Shape rendering hints (Coin3D SoShapeHints).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexOrdering { Unknown, Clockwise, CounterClockwise }
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeType { Unknown, Solid, FaceSet, LineSet, PointSet }
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceType { Unknown, Convex }
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ShapeHintsNode {
    pub vertex_ordering: VertexOrdering,
    pub shape_type: ShapeType,
    pub face_type: FaceType,
    pub crease_angle: f32,
}
impl Default for ShapeHintsNode {
    fn default() -> Self {
        Self { vertex_ordering: VertexOrdering::Unknown, shape_type: ShapeType::Unknown, face_type: FaceType::Convex, crease_angle: 0.5 }
    }
}

/// Annotation node: renders children as overlay without depth test (Coin3D SoAnnotation).
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct AnnotationNode;

/// Resets the current model matrix to identity (Coin3D SoResetTransform).
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ResetTransformNode;

/// 2D texture coordinate transform (Coin3D SoTexture2Transform).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Texture2TransformNode {
    pub translation: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
    pub center: [f32; 2],
}
impl Default for Texture2TransformNode {
    fn default() -> Self { Self { translation: [0.0; 2], rotation: 0.0, scale: [1.0; 2], center: [0.0; 2] } }
}

/// Per-vertex material binding (Coin3D SoMaterialBinding).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialBinding { Default, Overall, PerPart, PerPartIndexed, PerFace, PerFaceIndexed, PerVertex, PerVertexIndexed }
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MaterialBindingNode { pub value: MaterialBinding }
impl Default for MaterialBindingNode { fn default() -> Self { Self { value: MaterialBinding::Default } } }
