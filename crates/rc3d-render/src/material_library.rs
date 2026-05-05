use std::collections::HashMap;

use rc3d_scene::MaterialNode;

/// Handle to a material stored in the MaterialLibrary.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct MaterialId(u32);

/// Entry in the material library with cached GPU bind group.
#[derive(Clone)]
pub struct MaterialEntry {
    pub material: MaterialNode,
    /// Cached PBR material bind group (albedo texture + normal map + sampler).
    /// Rebuilt when textures are updated.
    pub bind_group: Option<wgpu::BindGroup>,
}

/// Shared material definitions with GPU bind group caching.
///
/// Materials are referenced by name (from import) or by MaterialId.
/// Bind groups are pre-built and cached, avoiding per-frame creation
/// for materials that reuse the same textures.
#[derive(Clone, Default)]
pub struct MaterialLibrary {
    entries: HashMap<String, MaterialEntry>,
    id_to_name: HashMap<MaterialId, String>,
    name_to_id: HashMap<String, MaterialId>,
    next_id: u32,
    /// Bind group layout reference (set during initialization).
    material_bgl: Option<wgpu::BindGroupLayout>,
}

impl MaterialLibrary {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            id_to_name: HashMap::new(),
            name_to_id: HashMap::new(),
            next_id: 1,
            material_bgl: None,
        }
    }

    /// Set the bind group layout used for material textures.
    pub fn set_bind_group_layout(&mut self, bgl: &wgpu::BindGroupLayout) {
        self.material_bgl = Some(bgl.clone());
    }

    /// Insert or update a material by name. Returns the MaterialId.
    /// Invalidates any cached bind group for this material.
    pub fn insert(&mut self, name: impl Into<String>, material: MaterialNode) -> MaterialId {
        let name = name.into();
        if let Some(&id) = self.name_to_id.get(&name) {
            let entry = self.entries.get_mut(&name).unwrap();
            entry.material = material;
            entry.bind_group = None; // Invalidate cache
            id
        } else {
            let id = MaterialId(self.next_id);
            self.next_id += 1;
            self.name_to_id.insert(name.clone(), id);
            self.id_to_name.insert(id, name.clone());
            self.entries.insert(name, MaterialEntry {
                material,
                bind_group: None,
            });
            id
        }
    }

    /// Look up a material by name.
    pub fn get(&self, name: &str) -> Option<&MaterialNode> {
        self.entries.get(name).map(|e| &e.material)
    }

    /// Look up a material by ID.
    pub fn get_by_id(&self, id: MaterialId) -> Option<&MaterialNode> {
        self.id_to_name
            .get(&id)
            .and_then(|name| self.entries.get(name))
            .map(|e| &e.material)
    }

    /// Look up material ID by name.
    pub fn id(&self, name: &str) -> Option<MaterialId> {
        self.name_to_id.get(name).copied()
    }

    /// Look up the cached bind group for a material. Returns None if not yet built.
    pub fn bind_group(&self, id: MaterialId) -> Option<&wgpu::BindGroup> {
        self.id_to_name
            .get(&id)
            .and_then(|name| self.entries.get(name))
            .and_then(|e| e.bind_group.as_ref())
    }

    /// Build or rebuild the bind group for a material.
    /// Requires `device`, `texture_cache`, and `queue` to load textures.
    /// Returns true if the bind group was created or updated.
    pub fn build_bind_group(
        &mut self,
        id: MaterialId,
        device: &wgpu::Device,
        texture_cache: &mut crate::texture_cache::TextureCache,
        queue: &wgpu::Queue,
    ) -> bool {
        let bgl = match &self.material_bgl {
            Some(bgl) => bgl,
            None => return false,
        };

        let Some(name) = self.id_to_name.get(&id).cloned() else {
            return false;
        };

        let entry = self.entries.get(&name);
        let Some(entry) = entry else { return false };

        let albedo_path = entry.material.albedo_texture.clone();
        let albedo_handle = match &albedo_path {
            Some(p) => texture_cache.load_path(device, queue, p.as_str()),
            None => texture_cache.white_handle(),
        };

        let normal_path = entry.material.normal_texture.clone();
        let normal_handle = match &normal_path {
            Some(p) => texture_cache.load_path(device, queue, p.as_str()),
            None => texture_cache.default_normal_handle(),
        };

        let mr_handle = match &entry.material.metallic_roughness_texture {
            Some(p) => texture_cache.load_path(device, queue, p.as_str()),
            None => texture_cache.white_handle(),
        };
        let emissive_handle = match &entry.material.emissive_texture {
            Some(p) => texture_cache.load_path(device, queue, p.as_str()),
            None => texture_cache.white_handle(),
        };
        let occlusion_handle = match &entry.material.occlusion_texture {
            Some(p) => texture_cache.load_path(device, queue, p.as_str()),
            None => texture_cache.white_handle(),
        };

        let bg = texture_cache.pbr_material_bind_group(
            device, bgl, albedo_handle, normal_handle, mr_handle, emissive_handle, occlusion_handle,
        );

        if let Some(entry) = self.entries.get_mut(&name) {
            entry.bind_group = Some(bg.clone());
            true
        } else {
            false
        }
    }

    /// Build bind groups for all materials that don't have one yet.
    pub fn build_all_bind_groups(
        &mut self,
        device: &wgpu::Device,
        texture_cache: &mut crate::texture_cache::TextureCache,
        queue: &wgpu::Queue,
    ) {
        let ids: Vec<MaterialId> = self.id_to_name.keys().copied().collect();
        for id in ids {
            self.build_bind_group(id, device, texture_cache, queue);
        }
    }

    /// Invalidate all cached bind groups (e.g. on device reset).
    pub fn invalidate_bind_groups(&mut self) {
        for entry in self.entries.values_mut() {
            entry.bind_group = None;
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
