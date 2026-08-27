//! PBR material builder.

use rc3d_core::math::Vec3;
use rc3d_scene::node_data::{AlphaMode, MaterialNode};

/// Material descriptor, mapping directly to `MaterialNode`.
#[derive(Clone, Debug)]
pub struct Material {
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
    pub metallic_roughness_texture: Option<String>,
    pub occlusion_texture: Option<String>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub anisotropic: f32,
    pub clearcoat_factor: f32,
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
    pub light_group: Option<String>,
    pub custom_wgsl: Option<String>,
    pub custom_uniforms: [f32; 4],
}

impl Default for Material {
    fn default() -> Self {
        Self {
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            ambient_color: Vec3::new(0.1, 0.1, 0.1),
            specular_color: Vec3::new(0.04, 0.04, 0.04),
            shininess: 32.0,
            base_color: Vec3::new(0.7, 0.7, 0.7),
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

impl Material {
    pub fn pbr() -> Self {
        Self::default()
    }

    pub fn diffuse(color: Vec3) -> Self {
        Self {
            diffuse_color: color,
            ambient_color: color * 0.15,
            base_color: color,
            ..Default::default()
        }
    }

    pub fn base_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.diffuse_color = Vec3::new(r, g, b);
        self.ambient_color = Vec3::new(r * 0.15, g * 0.15, b * 0.15);
        self.base_color = Vec3::new(r, g, b);
        self
    }

    pub fn metallic(mut self, v: f32) -> Self {
        self.metallic = v;
        self
    }

    pub fn roughness(mut self, v: f32) -> Self {
        self.roughness = v;
        self
    }

    pub fn opacity(mut self, v: f32) -> Self {
        self.opacity = v;
        self
    }

    pub fn shininess(mut self, v: f32) -> Self {
        self.shininess = v;
        self
    }

    pub fn emissive(mut self, r: f32, g: f32, b: f32) -> Self {
        self.emissive_color = Vec3::new(r, g, b);
        self
    }

    pub fn textures(
        mut self,
        albedo: Option<&str>,
        normal: Option<&str>,
        orm: Option<&str>,
    ) -> Self {
        self.albedo_texture = albedo.map(String::from);
        self.normal_texture = normal.map(String::from);
        self.metallic_roughness_texture = orm.map(String::from);
        self
    }

    pub(crate) fn to_node(&self) -> MaterialNode {
        MaterialNode {
            diffuse_color: self.diffuse_color,
            ambient_color: self.ambient_color,
            specular_color: self.specular_color,
            shininess: self.shininess,
            base_color: self.base_color,
            metallic: self.metallic,
            roughness: self.roughness,
            albedo_texture: self.albedo_texture.clone(),
            normal_texture: self.normal_texture.clone(),
            opacity: self.opacity,
            emissive_color: self.emissive_color,
            emissive_texture: self.emissive_texture.clone(),
            metallic_roughness_texture: self.metallic_roughness_texture.clone(),
            occlusion_texture: self.occlusion_texture.clone(),
            alpha_mode: self.alpha_mode,
            alpha_cutoff: self.alpha_cutoff,
            double_sided: self.double_sided,
            anisotropic: self.anisotropic,
            clearcoat_factor: self.clearcoat_factor,
            clearcoat_roughness: self.clearcoat_roughness,
            specular_factor: self.specular_factor,
            specular_color_factor: self.specular_color_factor,
            transmission_factor: self.transmission_factor,
            ior: self.ior,
            sheen_color: self.sheen_color,
            sheen_roughness: self.sheen_roughness,
            iridescence_factor: self.iridescence_factor,
            iridescence_ior: self.iridescence_ior,
            iridescence_thickness_min: self.iridescence_thickness_min,
            iridescence_thickness_max: self.iridescence_thickness_max,
            toon_steps: self.toon_steps,
            visualize_normals: self.visualize_normals,
            visualize_depth: self.visualize_depth,
            light_group: self.light_group.clone(),
            custom_wgsl: self.custom_wgsl.clone(),
            custom_uniforms: self.custom_uniforms,
        }
    }

    pub fn custom_shader(mut self, wgsl: impl Into<String>) -> Self {
        self.custom_wgsl = Some(wgsl.into());
        self
    }

    pub fn custom_uniforms(mut self, v: [f32; 4]) -> Self {
        self.custom_uniforms = v;
        self
    }

    pub fn transmission(mut self, factor: f32, ior: f32) -> Self {
        self.transmission_factor = factor;
        self.ior = ior;
        self
    }

    pub fn sheen(mut self, color: Vec3, roughness: f32) -> Self {
        self.sheen_color = color;
        self.sheen_roughness = roughness;
        self
    }

    pub fn anisotropic(mut self, strength: f32) -> Self {
        self.anisotropic = strength;
        self
    }

    /// Thin-film iridescence (KHR_materials_iridescence). `thickness_max_nm` is the upper film thickness.
    pub fn iridescence(mut self, factor: f32, ior: f32, thickness_max_nm: f32) -> Self {
        self.iridescence_factor = factor;
        self.iridescence_ior = ior;
        self.iridescence_thickness_max = thickness_max_nm;
        self
    }

    pub fn toon(mut self, steps: f32) -> Self {
        self.toon_steps = steps;
        self
    }

    pub fn visualize_normals(mut self, on: bool) -> Self {
        self.visualize_normals = on;
        self
    }

    /// Camera-distance grayscale (three.js MeshDepthMaterial analog).
    pub fn visualize_depth(mut self, on: bool) -> Self {
        self.visualize_depth = on;
        self
    }
}
