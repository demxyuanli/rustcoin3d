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
    pub light_group: Option<String>,
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
            light_group: None,
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
            light_group: self.light_group.clone(),
        }
    }
}
