use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::{Appearance, DisplayMode, EdgeStyle, FillStyle};
use std::sync::Arc;

use crate::vertex::Vertex;

/// GPU skinning inputs carried on a draw call (`SkinnedMeshNode` in the scene graph).
#[derive(Clone, Debug)]
pub struct SkinnedMeshDrawPayload {
    pub skeleton: rc3d_scene::animation::Skeleton,
    pub skin_data: Arc<Vec<rc3d_scene::animation::VertexSkinData>>,
    pub clip: Option<rc3d_scene::animation::AnimationClip>,
}

/// Collected draw data from scene graph traversal.
#[derive(Clone, Debug)]
pub struct DrawCall {
    pub vertices: Arc<Vec<Vertex>>,
    pub indices: Option<Arc<Vec<u32>>>,
    /// Crease + boundary feature edges (shaded overlay, selection edges).
    pub edge_positions: Arc<Vec<[f32; 3]>>,
    /// Full topological edge line list (wireframe mode).
    pub wireframe_edge_positions: Arc<Vec<[f32; 3]>>,
    pub mvp: Mat4,
    pub model_matrix: Mat4,
    pub camera_pos: Vec3,
    /// Index into `LightSetTable` for shared light parameters.
    pub light_set_id: u32,
    /// Pre-computed hash of light params for fast draw-call grouping.
    pub light_key: u64,
    pub diffuse_color: Vec3,
    pub ambient_color: Vec3,
    pub specular_color: Vec3,
    pub shininess: f32,
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub anisotropic: f32,
    pub opacity: f32,
    pub albedo_path: Option<Arc<str>>,
    pub normal_path: Option<Arc<str>>,
    pub emissive_color: Vec3,
    pub emissive_path: Option<Arc<str>>,
    pub metallic_roughness_path: Option<Arc<str>>,
    pub occlusion_path: Option<Arc<str>>,
    pub alpha_mode: rc3d_scene::AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    /// Clearcoat factor [0,1] for KHR_materials_clearcoat GLTF extension.
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
    /// Custom WGSL (ShaderMaterial). Empty = PBR path.
    pub custom_wgsl: Option<std::sync::Arc<str>>,
    pub custom_uniforms: [f32; 4],
    pub aabb: Option<rc3d_core::Aabb>,
    pub display_mode: DisplayMode,
    pub fill_style: FillStyle,
    pub edge_style: EdgeStyle,
    pub selected: bool,
    pub overlay_color: Option<[f32; 4]>,
    pub mesh_hash: Option<u64>,
    pub meshlet_data: Option<Arc<rc3d_mesh::MeshletData>>,
    pub projection_orthographic: bool,
    /// Align with camera projection (e.g. `PerspectiveCameraNode::reverse_depth` in `rc3d-scene`) and
    /// renderer depth ops; inferred via `rc3d_core::depth_reversed_z_from_projection`.
    pub depth_reversed_z: bool,
    /// Overlay draw call: rendered without depth test (Annotation children).
    pub is_overlay: bool,
    pub node_type_label: Arc<str>,
    /// Instance transforms for GPU instancing (e.g., MultipleCopy).
    /// If set, the draw call is instanced with these model matrices.
    pub instance_transforms: Option<Arc<Vec<Mat4>>>,
    /// Morph target (blend shape) weights. Empty = no morph targets.
    /// When non-empty, the renderer allocates a storage buffer with per-target
    /// position deltas (and optional normal deltas) for the vertex shader.
    pub morph_weights: Vec<f32>,
    /// Packed morph target deltas: each target has [position_deltas, normal_deltas_opt].
    /// Length = targets.len(); each position_deltas.len() = vertex_count.
    pub morph_target_deltas: Option<Arc<rc3d_scene::MorphTargetNode>>,
    /// Skeletal skinning: compute pass writes animated vertices into the mesh vertex buffer.
    pub skinning: Option<Arc<SkinnedMeshDrawPayload>>,
    /// First index into [`Self::indices`] (three.js BufferGeometry.groups `start`).
    /// `index_draw_count == 0` means draw the whole index buffer.
    pub index_first: u32,
    /// Index count for this draw. Zero = use the full uploaded index buffer.
    pub index_draw_count: u32,
}

impl DrawCall {
    /// `(first_index, index_count)` for this draw, or `None` to use the full GPU mesh.
    pub fn index_draw_range(&self) -> Option<(u32, u32)> {
        if self.index_draw_count == 0 {
            None
        } else {
            Some((self.index_first, self.index_draw_count))
        }
    }

    /// Clamp this draw's index range against an uploaded mesh `index_count`.
    pub fn resolved_index_range(&self, mesh_index_count: u32) -> (u32, u32) {
        if self.index_draw_count == 0 {
            (0, mesh_index_count)
        } else {
            let first = self.index_first.min(mesh_index_count);
            let end = first.saturating_add(self.index_draw_count).min(mesh_index_count);
            (first, end.saturating_sub(first))
        }
    }

    pub fn appearance(&self) -> Appearance {
        Appearance {
            fill: self.fill_style,
            edges: self.edge_style,
        }
    }

    /// Packed into `pbr_alpha_flags.w`: toon bands, 100+ normals, 200+ depth.
    pub fn shade_mode_w(&self) -> f32 {
        if self.visualize_depth {
            200.0
        } else if self.visualize_normals {
            100.0
        } else {
            self.toon_steps
        }
    }
}

impl Default for DrawCall {
    fn default() -> Self {
        Self {
            vertices: Arc::new(Vec::new()),
            indices: None,
            edge_positions: Arc::new(Vec::new()),
            wireframe_edge_positions: Arc::new(Vec::new()),
            mvp: Mat4::IDENTITY,
            model_matrix: Mat4::IDENTITY,
            camera_pos: Vec3::ZERO,
            light_set_id: 0,
            light_key: 0,
            diffuse_color: Vec3::ZERO,
            ambient_color: Vec3::ZERO,
            specular_color: Vec3::ZERO,
            shininess: 1.0,
            base_color: Vec3::ZERO,
            metallic: 0.0,
            roughness: 0.5,
            anisotropic: 0.0,
            opacity: 1.0,
            albedo_path: None,
            normal_path: None,
            emissive_color: Vec3::ZERO,
            emissive_path: None,
            metallic_roughness_path: None,
            occlusion_path: None,
            alpha_mode: rc3d_scene::AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
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
            aabb: None,
            display_mode: DisplayMode::ShadedWithEdges,
            fill_style: FillStyle::Shaded,
            edge_style: EdgeStyle::Crease,
            selected: false,
            overlay_color: None,
            mesh_hash: None,
            meshlet_data: None,
            projection_orthographic: false,
            depth_reversed_z: false,
            is_overlay: false,
            node_type_label: Arc::from("Unknown"),
            instance_transforms: None,
            morph_weights: Vec::new(),
            morph_target_deltas: None,
            skinning: None,
            index_first: 0,
            index_draw_count: 0,
        }
    }
}

/// Default HOOPS-style ghost opacity for unselected filled geometry.
pub const GHOST_UNSELECTED_OPACITY: f32 = 0.25;

/// HOOPS Isolate/Ghost: when enabled and anything is selected, unselected
/// filled draws become translucent. Selected geometry keeps its material.
/// No-op when the selection is empty (the rest of the scene stays shaded).
pub fn apply_ghost_unselected(draw_calls: &mut [DrawCall], enabled: bool, opacity: f32) {
    if !enabled {
        return;
    }
    if !draw_calls.iter().any(|dc| dc.selected) {
        return;
    }
    let ghost_a = opacity.clamp(0.02, 0.95);
    for dc in draw_calls.iter_mut() {
        if dc.selected || dc.is_overlay || dc.custom_wgsl.is_some() {
            continue;
        }
        if dc.vertices.is_empty() && dc.meshlet_data.is_none() {
            continue;
        }
        if !dc.appearance().wants_filled() {
            continue;
        }
        dc.opacity = dc.opacity.min(ghost_a);
        dc.alpha_mode = rc3d_scene::AlphaMode::Blend;
        dc.double_sided = true;
        dc.edge_style = EdgeStyle::None;
    }
}

/// Default HOOPS X-ray fill opacity (all filled geometry, edges kept).
pub const XRAY_FILL_OPACITY: f32 = 0.28;

/// HOOPS X-ray visual style: every filled draw becomes translucent and keeps
/// (or gains) crease edges. Unlike Isolate/Ghost this is not gated on selection.
pub fn apply_xray(draw_calls: &mut [DrawCall], enabled: bool, opacity: f32) {
    if !enabled {
        return;
    }
    let xray_a = opacity.clamp(0.02, 0.95);
    for dc in draw_calls.iter_mut() {
        if dc.is_overlay || dc.custom_wgsl.is_some() {
            continue;
        }
        if dc.vertices.is_empty() && dc.meshlet_data.is_none() {
            continue;
        }
        if !dc.appearance().wants_filled() {
            continue;
        }
        dc.opacity = dc.opacity.min(xray_a);
        dc.alpha_mode = rc3d_scene::AlphaMode::Blend;
        dc.double_sided = true;
        if dc.edge_style == EdgeStyle::None {
            dc.edge_style = EdgeStyle::Crease;
        }
    }
}

/// Recomputes MVP and projection-related flags using the given view/projection
/// without re-traversing the scene graph (e.g. orbit camera overlay on collected geometry).
pub fn apply_world_camera(
    draw_calls: &mut [DrawCall],
    view_matrix: Mat4,
    projection: Mat4,
    camera_pos: Vec3,
) {
    apply_world_camera_ex(draw_calls, view_matrix, projection, camera_pos, false);
}

pub fn apply_world_camera_ex(
    draw_calls: &mut [DrawCall],
    view_matrix: Mat4,
    projection: Mat4,
    camera_pos: Vec3,
    orthographic: bool,
) {
    let depth_reversed_z = rc3d_core::depth_reversed_z_from_projection(projection);
    for dc in draw_calls.iter_mut() {
        dc.mvp = projection * view_matrix * dc.model_matrix;
        dc.camera_pos = camera_pos;
        dc.depth_reversed_z = depth_reversed_z;
        dc.projection_orthographic = orthographic;
    }
}

/// Extract `projection * view` from a draw call after [`apply_world_camera`].
#[inline]
pub fn view_projection_from_draw_call(dc: &DrawCall) -> Mat4 {
    dc.mvp * dc.model_matrix.inverse()
}
