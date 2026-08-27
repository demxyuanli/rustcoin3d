//! Property nodes: coordinates, material, transform, hints, bindings.
use rc3d_core::math::{Mat4, Quat, Vec3};
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

fn default_iridescence_ior() -> f32 {
    1.3
}
fn default_iridescence_thickness_min() -> f32 {
    100.0
}
fn default_iridescence_thickness_max() -> f32 {
    400.0
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
    /// Clearcoat factor (0.0–1.0) for KHR_materials_clearcoat GLTF extension.
    pub clearcoat_factor: f32,
    /// Clearcoat roughness (0.0–1.0).
    pub clearcoat_roughness: f32,
    /// PBR specular factor (KHR_materials_specular) — multiplies dielectric F0. Default 1.0.
    pub specular_factor: f32,
    /// PBR specular color (KHR_materials_specular) — tints F0. Default [1,1,1].
    pub specular_color_factor: Vec3,
    /// Transmission factor (KHR_materials_transmission) — 0.0 = opaque, 1.0 = full transmission.
    pub transmission_factor: f32,
    /// Index of refraction for transmission (default 1.5 for glass).
    pub ior: f32,
    /// Sheen color (KHR_materials_sheen). Zero disables the layer.
    #[serde(default)]
    pub sheen_color: Vec3,
    /// Sheen roughness (KHR_materials_sheen).
    #[serde(default)]
    pub sheen_roughness: f32,
    /// Thin-film iridescence factor (KHR_materials_iridescence). Zero disables.
    #[serde(default)]
    pub iridescence_factor: f32,
    /// Thin-film IOR (KHR_materials_iridescence). Default 1.3.
    #[serde(default = "default_iridescence_ior")]
    pub iridescence_ior: f32,
    /// Thin-film thickness minimum in nm.
    #[serde(default = "default_iridescence_thickness_min")]
    pub iridescence_thickness_min: f32,
    /// Thin-film thickness maximum in nm.
    #[serde(default = "default_iridescence_thickness_max")]
    pub iridescence_thickness_max: f32,
    /// Cel-shading bands (three.js MeshToonMaterial). 0 = off.
    #[serde(default)]
    pub toon_steps: f32,
    /// Output world normals as color (three.js MeshNormalMaterial).
    #[serde(default)]
    pub visualize_normals: bool,
    /// Output camera-distance grayscale (three.js MeshDepthMaterial).
    #[serde(default)]
    pub visualize_depth: bool,
    pub light_group: Option<String>,
    /// Optional custom WGSL (Three.js ShaderMaterial). Snippet or full shader.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_wgsl: Option<String>,
    /// User vec4 forwarded to custom shaders as `u.custom`.
    #[serde(default)]
    pub custom_uniforms: [f32; 4],
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
            clearcoat_factor: 0.0,
            clearcoat_roughness: 0.0,
            specular_factor: 1.0,
            specular_color_factor: Vec3::ONE,
            transmission_factor: 0.0,
            ior: 1.5,
            sheen_color: Vec3::ZERO,
            sheen_roughness: 0.0,
            iridescence_factor: 0.0,
            iridescence_ior: 1.3,
            iridescence_thickness_min: 100.0,
            iridescence_thickness_max: 400.0,
            toon_steps: 0.0,
            visualize_normals: false,
            visualize_depth: false,
            light_group: None,
            custom_wgsl: None,
            custom_uniforms: [0.0; 4],
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
            clearcoat_factor: 0.0,
            clearcoat_roughness: 0.0,
            specular_factor: 1.0,
            specular_color_factor: Vec3::ONE,
            transmission_factor: 0.0,
            ior: 1.5,
            sheen_color: Vec3::ZERO,
            sheen_roughness: 0.0,
            iridescence_factor: 0.0,
            iridescence_ior: 1.3,
            iridescence_thickness_min: 100.0,
            iridescence_thickness_max: 400.0,
            toon_steps: 0.0,
            visualize_normals: false,
            visualize_depth: false,
            light_group: None,
            custom_wgsl: None,
            custom_uniforms: [0.0; 4],
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

/// Axis-angle rotation (Coin3D `SoRotation`). Multiplies the current model matrix.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RotationNode {
    pub axis: Vec3,
    pub angle: f32,
}

impl Default for RotationNode {
    fn default() -> Self {
        Self {
            axis: Vec3::Z,
            angle: 0.0,
        }
    }
}

impl RotationNode {
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let len = axis.length();
        if len < 1e-8 {
            Self {
                axis: Vec3::Z,
                angle: 0.0,
            }
        } else {
            Self {
                axis: axis / len,
                angle,
            }
        }
    }

    pub fn from_quat(q: Quat) -> Self {
        let (axis, angle) = q.to_axis_angle();
        Self { axis, angle }
    }

    pub fn to_matrix(&self) -> Mat4 {
        if self.axis.length_squared() < 1e-12 {
            Mat4::IDENTITY
        } else {
            Mat4::from_axis_angle(self.axis.normalize(), self.angle)
        }
    }
}

/// Cardinal-axis rotation (Coin3D `SoRotationXYZ`).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RotationAxis {
    #[default]
    X,
    Y,
    Z,
}

/// Rotation about X, Y, or Z (Coin3D `SoRotationXYZ`).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RotationXYZNode {
    pub axis: RotationAxis,
    pub angle: f32,
}

impl Default for RotationXYZNode {
    fn default() -> Self {
        Self {
            axis: RotationAxis::X,
            angle: 0.0,
        }
    }
}

impl RotationXYZNode {
    pub fn to_matrix(&self) -> Mat4 {
        match self.axis {
            RotationAxis::X => Mat4::from_rotation_x(self.angle),
            RotationAxis::Y => Mat4::from_rotation_y(self.angle),
            RotationAxis::Z => Mat4::from_rotation_z(self.angle),
        }
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
