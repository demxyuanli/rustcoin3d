use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::{Appearance, DisplayMode, EdgeStyle, FillStyle};
use std::any::Any;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementId(pub u16);

pub trait Element: Any + std::fmt::Debug + Send + Sync {
    fn element_id(&self) -> ElementId;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn clone_box(&self) -> Box<dyn Element>;
}

impl Clone for Box<dyn Element> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// --- Concrete Elements ---

#[derive(Clone, Debug)]
pub struct ModelMatrixElement {
    pub matrix: Mat4,
}

impl Default for ModelMatrixElement {
    fn default() -> Self {
        Self {
            matrix: Mat4::IDENTITY,
        }
    }
}

impl Element for ModelMatrixElement {
    fn element_id(&self) -> ElementId {
        ElementId(0)
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug)]
pub struct ViewMatrixElement {
    pub matrix: Mat4,
}

impl Default for ViewMatrixElement {
    fn default() -> Self {
        Self { matrix: Mat4::IDENTITY }
    }
}

impl Element for ViewMatrixElement {
    fn element_id(&self) -> ElementId { ElementId(1) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug)]
pub struct ProjectionMatrixElement {
    pub matrix: Mat4,
}

impl Default for ProjectionMatrixElement {
    fn default() -> Self {
        Self { matrix: Mat4::IDENTITY }
    }
}

impl Element for ProjectionMatrixElement {
    fn element_id(&self) -> ElementId { ElementId(2) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug, Default)]
pub struct CoordinateElement {
    pub points: Vec<Vec3>,
}

impl Element for CoordinateElement {
    fn element_id(&self) -> ElementId { ElementId(3) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug, Default)]
pub struct TextureCoordinate2Element {
    pub coords: Vec<[f32; 2]>,
}

impl Element for TextureCoordinate2Element {
    fn element_id(&self) -> ElementId {
        ElementId(7)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Element> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Debug, Default)]
pub struct NormalElement {
    pub vectors: Vec<Vec3>,
}

impl Element for NormalElement {
    fn element_id(&self) -> ElementId { ElementId(4) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug)]
pub struct MaterialElement {
    pub diffuse: Vec3,
    pub ambient: Vec3,
    pub specular: Vec3,
    pub shininess: f32,
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub albedo_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub opacity: f32,
    pub emissive_color: Vec3,
    pub emissive_texture: Option<String>,
    pub metallic_roughness_texture: Option<String>,
    pub occlusion_texture: Option<String>,
    pub alpha_mode: crate::AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    /// Anisotropic roughness (0.0 = isotropic).
    pub anisotropic: f32,
    /// Clearcoat factor (0.0–1.0) for KHR_materials_clearcoat GLTF extension.
    pub clearcoat_factor: f32,
    /// Clearcoat roughness (0.0–1.0).
    pub clearcoat_roughness: f32,
    pub specular_factor: f32,
    pub specular_color_factor: Vec3,
    pub transmission_factor: f32,
    pub ior: f32,
    pub sheen_color: Vec3,
    pub sheen_roughness: f32,
    pub iridescence_factor: f32,
    pub iridescence_ior: f32,
    pub iridescence_thickness_min: f32,
    pub iridescence_thickness_max: f32,
    pub toon_steps: f32,
    pub visualize_normals: bool,
    pub visualize_depth: bool,
    pub custom_wgsl: Option<String>,
    pub custom_uniforms: [f32; 4],
}

impl Default for MaterialElement {
    fn default() -> Self {
        Self {
            diffuse: Vec3::new(0.9, 0.9, 0.9),
            ambient: Vec3::new(0.25, 0.25, 0.25),
            specular: Vec3::new(0.0, 0.0, 0.0),
            shininess: 0.0,
            base_color: Vec3::new(0.94, 0.94, 0.94),
            metallic: 0.0,
            roughness: 0.35,
            albedo_texture: None,
            normal_texture: None,
            opacity: 1.0,
            emissive_color: Vec3::ZERO,
            emissive_texture: None,
            metallic_roughness_texture: None,
            occlusion_texture: None,
            alpha_mode: crate::AlphaMode::Opaque,
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
            custom_wgsl: None,
            custom_uniforms: [0.0; 4],
        }
    }
}

impl Element for MaterialElement {
    fn element_id(&self) -> ElementId { ElementId(5) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug, Default)]
pub struct LightElement {
    pub lights: Vec<LightData>,
}

#[derive(Clone, Debug)]
pub struct LightData {
    pub light_type: LightType,
    pub direction: Vec3,
    pub location: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub cut_off_angle: f32,
    pub drop_off_rate: f32,
    /// Packed as `light_positions.xyz` for [`LightType::Hemisphere`] (ground irradiance).
    pub ground_color: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightType {
    Directional,
    Point,
    Spot,
    Hemisphere,
}

impl Element for LightElement {
    fn element_id(&self) -> ElementId { ElementId(6) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

/// Coin3D `SoDrawStyle` analogue: inherited until a Separator pops the stack.
/// `mode` stays the last explicit preset; `fill` / `edges` are the resolved axes.
#[derive(Clone, Copy, Debug)]
pub struct DisplayModeElement {
    pub mode: DisplayMode,
    pub fill: FillStyle,
    pub edges: EdgeStyle,
}

impl Default for DisplayModeElement {
    fn default() -> Self {
        let mode = DisplayMode::default();
        Self {
            mode,
            fill: mode.fill(),
            edges: mode.edges(),
        }
    }
}

impl DisplayModeElement {
    pub fn appearance(self) -> Appearance {
        Appearance {
            fill: self.fill,
            edges: self.edges,
        }
    }

    pub fn set_appearance(&mut self, app: Appearance) {
        self.fill = app.fill;
        self.edges = app.edges;
        self.mode = app.to_display_mode();
    }

    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        self.mode = mode;
        self.fill = mode.fill();
        self.edges = mode.edges();
    }
}

impl Element for DisplayModeElement {
    fn element_id(&self) -> ElementId {
        ElementId(8)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Element> {
        Box::new(*self)
    }
}

pub const NUM_ELEMENT_TYPES: usize = 9;
