use rc3d_actions::{LightData, LightType, State};
use rc3d_core::math::{Mat4, Vec3, Vec4};
use rc3d_core::{DisplayMode, NodeId};
use rc3d_scene::{NodeData, SceneGraph};
use slotmap::Key;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::material_library::MaterialLibrary;
use crate::vertex::{Vertex, MAX_LIGHTS};

// Re-export shape_cache items for backwards compatibility via crate::render_action::
pub use crate::shape_cache::{
    feature_crease_angle, set_feature_crease_angle, clamp_edge_positions,
    ShapeKey, CachedShapeData, MAX_EDGE_POSITIONS, MESHLET_TRIANGLE_THRESHOLD,
};

// Re-export light_packing items
pub use crate::light_packing::{hash_light_params, PackedLights};

// Re-export traversal items
pub use crate::traversal::{
    convert_draw_calls_to_cache_textures,
    populate_cache_from_draw_calls,
    merge_chunk_into_cache, invalidate_cache_for_subtree,
    TraversalChunk,
};

fn material_element_for_node(
    mat: &rc3d_scene::MaterialNode,
    entry_name: Option<&str>,
    library: Option<&MaterialLibrary>,
) -> rc3d_actions::MaterialElement {
    let src = if let (Some(name), Some(lib)) = (entry_name, library) {
        lib.get(name).unwrap_or(mat)
    } else {
        mat
    };
    rc3d_actions::MaterialElement {
        diffuse: src.diffuse_color,
        ambient: src.ambient_color,
        specular: src.specular_color,
        shininess: src.shininess,
        base_color: src.base_color,
        metallic: src.metallic,
        roughness: src.roughness,
        albedo_texture: src.albedo_texture.clone(),
        normal_texture: src.normal_texture.clone(),
        opacity: src.opacity,
        emissive_color: src.emissive_color,
        emissive_texture: src.emissive_texture.clone(),
        metallic_roughness_texture: src.metallic_roughness_texture.clone(),
        occlusion_texture: src.occlusion_texture.clone(),
        alpha_mode: src.alpha_mode,
        alpha_cutoff: src.alpha_cutoff,
        double_sided: src.double_sided,
        anisotropic: src.anisotropic,
    }
}

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
    pub aabb: Option<rc3d_core::Aabb>,
    pub display_mode: DisplayMode,
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
            aabb: None,
            display_mode: DisplayMode::ShadedWithEdges,
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
    let depth_reversed_z = rc3d_core::depth_reversed_z_from_projection(projection);
    for dc in draw_calls.iter_mut() {
        dc.mvp = projection * view_matrix * dc.model_matrix;
        dc.camera_pos = camera_pos;
        dc.depth_reversed_z = depth_reversed_z;
        dc.projection_orthographic = false;
    }
}

/// Extract `projection * view` from a draw call after [`apply_world_camera`].
#[inline]
pub fn view_projection_from_draw_call(dc: &DrawCall) -> Mat4 {
    dc.mvp * dc.model_matrix.inverse()
}

/// Opaque cache-target pointer wrapper.
/// SAFETY INVARIANT: This type is Send because `RenderCollector` is only ever
/// accessed from a single thread during traversal. The `cache_ptr` is set by the
/// same thread that created the `FlatDrawCache`, and `get_mut()` is only called
/// during that thread's traversal. This invariant MUST be maintained - if
/// `RenderCollector` is ever accessed from multiple threads, this would cause
/// data races and undefined behavior.
struct CacheTarget {
    ptr: *mut crate::flat_draw_cache::FlatDrawCache,
    _invariant: std::marker::PhantomData<&'static mut ()>,
}

unsafe impl Send for CacheTarget {}

impl CacheTarget {
    fn null() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            _invariant: std::marker::PhantomData,
        }
    }
    fn is_null(&self) -> bool {
        self.ptr.is_null()
    }
    fn set(&mut self, cache: &mut crate::flat_draw_cache::FlatDrawCache) {
        self.ptr = cache as *mut _;
    }
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut crate::flat_draw_cache::FlatDrawCache {
        &mut *self.ptr
    }
}

/// Opaque texture-table pointer wrapper (same safety invariant as `CacheTarget`).
struct TextureTableTarget {
    ptr: *mut crate::global_tables::TexturePathTable,
    _invariant: std::marker::PhantomData<&'static mut ()>,
}

unsafe impl Send for TextureTableTarget {}

impl TextureTableTarget {
    fn null() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            _invariant: std::marker::PhantomData,
        }
    }
    fn is_null(&self) -> bool {
        self.ptr.is_null()
    }
    fn set(&mut self, table: &mut crate::global_tables::TexturePathTable) {
        self.ptr = table as *mut _;
    }
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut crate::global_tables::TexturePathTable {
        &mut *self.ptr
    }
}

/// Traverses the scene graph, accumulates state, and collects draw calls.
pub struct RenderCollector {
    pub state: State,
    pub draw_calls: Vec<DrawCall>,
    pub camera_pos: Vec3,
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
    pub projection_orthographic: bool,
    pub global_display_mode: DisplayMode,
    pub material_library: Option<MaterialLibrary>,
    mesh_cache: HashMap<ShapeKey, CachedShapeData>,
    pub light_sets: crate::light_set::LightSetTable,
    /// Optional flat-cache to populate during traversal.
    /// Wrapped for Send-safety; only valid during single-threaded traversal.
    cache_ptr: CacheTarget,
    /// Optional texture table for interning texture paths during direct-emit.
    texture_table_ptr: TextureTableTarget,
    hidden_nodes: HashSet<NodeId>,
    /// Environment state (accumulated from EnvironmentNode)
    pub ambient_intensity: f32,
    pub ambient_color: rc3d_core::math::Vec3,
    pub fog_color: rc3d_core::math::Vec3,
    pub fog_visibility: f32,
    /// Shape hints (vertex ordering for face culling)
    pub vertex_ordering: i32,
    /// Material binding mode (0=default, 1=overall, 2=per_part, 3=per_face, 4=per_vertex)
    pub material_binding: i32,
    /// Texture coordinate transform (2D scale+rotation+translation)
    pub tex2_transform: Option<(rc3d_core::math::Vec2, f32, rc3d_core::math::Vec2)>,
    /// True while traversing inside an Annotation node.
    pub inside_annotation: bool,
    /// Stereo rendering mode (set by StereoCameraNode).
    pub stereo_mode: Option<rc3d_scene::node_data::StereoMode>,
    /// Effect draw commands collected during traversal (Decal, Volume, PointCloud).
    pub effect_commands: crate::render_passes::pass_effects::EffectCommands,
}

impl RenderCollector {
    pub fn new() -> Self {
        Self {
            state: State::new(),
            draw_calls: Vec::new(),
            camera_pos: Vec3::new(0.0, 0.0, 5.0),
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            projection_orthographic: false,
            global_display_mode: DisplayMode::ShadedWithEdges,
            material_library: None,
            mesh_cache: HashMap::new(),
            light_sets: crate::light_set::LightSetTable::new(),
            cache_ptr: CacheTarget::null(),
            texture_table_ptr: TextureTableTarget::null(),
            hidden_nodes: HashSet::new(),
            ambient_intensity: 0.2,
            ambient_color: rc3d_core::math::Vec3::ONE,
            fog_color: rc3d_core::math::Vec3::ONE,
            fog_visibility: 0.0,
            vertex_ordering: 0,
            material_binding: 0,
            tex2_transform: None,
            inside_annotation: false,
            stereo_mode: None,
            effect_commands: crate::render_passes::pass_effects::EffectCommands::default(),
        }
    }

    /// Set the flat-cache target for direct emit during traversal.
    /// When set, each draw call is pushed to the cache immediately (no post-conversion needed).
    /// The `texture_table` is used to intern texture paths and fill in texture IDs
    /// in the cache entries during emit.
    pub fn set_cache_target(
        &mut self,
        cache: &mut crate::flat_draw_cache::FlatDrawCache,
        texture_table: &mut crate::global_tables::TexturePathTable,
    ) {
        self.cache_ptr.set(cache);
        self.texture_table_ptr.set(texture_table);
    }

    /// Pre-allocate draw_calls capacity (use previous frame's count as hint).
    pub fn reserve_draw_calls(&mut self, capacity: usize) {
        self.draw_calls.reserve(capacity);
    }

    pub fn traverse(&mut self, graph: &SceneGraph, root: NodeId) {
        self.traverse_node(graph, root);
    }

    /// Push a GpuDrawData entry to the cache from a freshly-built DrawCall.
    /// Called inline during emit to skip the post-traversal conversion step.
    fn emit_to_cache(&self, dc: &DrawCall) {
        let cache = unsafe { self.cache_ptr.get_mut() };

        // Compute texture IDs via the texture table (if available)
        let tex_ids = if !self.texture_table_ptr.is_null() {
            let tt = unsafe { self.texture_table_ptr.get_mut() };
            crate::traversal::intern_draw_call_tex_ids(dc, tt)
        } else {
            [u16::MAX; 5]
        };

        let (gpu, meta) = crate::traversal::draw_call_to_cache_entries(dc, tex_ids);

        cache.gpu_data.push(gpu);
        cache.metadata.push(meta);

        // Track node-to-draw mapping (caller updates after traversal)
        cache.total_triangles += dc.indices.as_ref().map_or(
            (dc.vertices.len() / 3) as u64,
            |idx| (idx.len() / 3) as u64,
        );
        cache.groups_dirty = true;
    }

    pub fn invalidate_mesh_cache(&mut self) {
        self.mesh_cache.clear();
    }

    pub fn set_hidden_nodes(&mut self, hidden_nodes: &HashSet<NodeId>) {
        self.hidden_nodes = hidden_nodes.clone();
    }

    fn traverse_node(&mut self, graph: &SceneGraph, node: NodeId) {
        if self.hidden_nodes.contains(&node) {
            return;
        }
        let Some(entry) = graph.get(node) else {
            return;
        };
        let is_selected = graph.is_selected(node);
        let node_display_mode = entry.display_mode;
        let node_type_label = entry.data.type_name();

        match &entry.data {
            NodeData::Separator(_) => {
                self.state.push_all();
                let base = self.state.model_matrix();
                let mut accum = base;
                for &child in &entry.children {
                    let Some(ce) = graph.get(child) else { continue };
                    if let NodeData::Transform(t) = &ce.data {
                        accum *= t.to_matrix();
                        self.state.set_model_matrix(accum);
                        for &gc in &ce.children {
                            self.traverse_node(graph, gc);
                        }
                    } else {
                        self.state.set_model_matrix(accum);
                        self.traverse_node(graph, child);
                    }
                }
                self.state.pop_all();
            }
            NodeData::Group(_) | NodeData::File(_) => {
                for &child in &entry.children { self.traverse_node(graph, child); }
            }
            NodeData::Decal(decal) => {
                self.effect_commands.decals.push(crate::render_passes::pass_effects::DecalDrawCommand {
                    model_matrix: self.state.model_matrix(),
                    position: decal.position,
                    direction: decal.direction,
                    size: decal.size,
                    texture_path: decal.texture_path.clone(),
                    color: decal.color,
                    opacity: decal.opacity,
                    is_overlay: self.inside_annotation,
                });
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::ReflectionPlane(rp) => {
                if !rp.enabled { for &child in &entry.children { self.traverse_node(graph, child); } return; }
                // Mirror view matrix across the reflection plane
                let view = self.state.view_matrix();
                let n = rp.normal.normalize();
                let o = rp.origin;
                let eye = view.inverse().w_axis.truncate();
                let d = -(eye - o).dot(n);
                let refl = Mat4::from_cols(
                    Vec4::new(1.0-2.0*n.x*n.x, -2.0*n.y*n.x, -2.0*n.z*n.x, 0.0),
                    Vec4::new(-2.0*n.x*n.y, 1.0-2.0*n.y*n.y, -2.0*n.z*n.y, 0.0),
                    Vec4::new(-2.0*n.x*n.z, -2.0*n.y*n.z, 1.0-2.0*n.z*n.z, 0.0),
                    Vec4::new(2.0*d*n.x, 2.0*d*n.y, 2.0*d*n.z, 1.0),
                );
                let mirrored = view * refl;
                let saved = self.state.view_matrix();
                self.state.set_view_matrix(mirrored);
                for &child in &entry.children { self.traverse_node(graph, child); }
                self.state.set_view_matrix(saved);
            }
            // TODO: Stereo dual-viewport — needs second render pass with offset eye matrices
            NodeData::StereoCamera(_) => { for &child in &entry.children { self.traverse_node(graph, child); } }
            // TODO: wgpu lacks native DXR/VKRT — deferred until wgpu adds ray tracing support
            NodeData::RayTracing(_) => { for &child in &entry.children { self.traverse_node(graph, child); } }
            NodeData::Volume(volume) => {
                self.effect_commands.volumes.push(crate::render_passes::pass_effects::VolumeDrawCommand {
                    model_matrix: self.state.model_matrix(),
                    dimensions: volume.dimensions,
                    texture_path: volume.texture_path.clone(),
                    density_scale: volume.density_scale,
                    color_map: volume.color_map,
                    is_overlay: self.inside_annotation,
                });
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::PointCloud(point_cloud) => {
                self.effect_commands.point_clouds.push(crate::render_passes::pass_effects::PointCloudDrawCommand {
                    model_matrix: self.state.model_matrix(),
                    file_path: point_cloud.file_path.clone(),
                    max_visible_points: point_cloud.max_visible_points,
                    point_size: point_cloud.point_size,
                    color: point_cloud.color,
                    is_overlay: self.inside_annotation,
                });
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::Billboard(b) => {
                let current = self.state.model_matrix();
                let inv = self.state.view_matrix().inverse();
                let facing = if b.axis_aligned {
                    let fwd = Vec3::new(inv.w_axis.x, 0.0, inv.w_axis.z).normalize();
                    Mat4::look_at_rh(Vec3::ZERO, fwd, Vec3::Y)
                } else {
                    Mat4::from_cols(inv.x_axis, inv.y_axis, inv.z_axis, Mat4::IDENTITY.w_axis)
                };
                self.state.set_model_matrix(current * facing);
                for &child in &entry.children { self.traverse_node(graph, child); }
                self.state.set_model_matrix(current);
            }
            NodeData::ResetTransform(_) => {
                let saved = self.state.model_matrix();
                self.state.set_model_matrix(Mat4::IDENTITY);
                for &child in &entry.children { self.traverse_node(graph, child); }
                self.state.set_model_matrix(saved);
            }
            NodeData::ExplodedView(ev) => {
                let base = self.state.model_matrix();
                for &child in &entry.children {
                    let offset = ev.direction * ev.factor;
                    self.state.set_model_matrix(base * Mat4::from_translation(offset));
                    self.traverse_node(graph, child);
                }
                self.state.set_model_matrix(base);
            }
            NodeData::ShapeHints(sh) => {
                self.vertex_ordering = match sh.vertex_ordering {
                    rc3d_scene::node_data::VertexOrdering::Clockwise => 1,
                    rc3d_scene::node_data::VertexOrdering::CounterClockwise => 2,
                    _ => 0,
                };
                for &child in &entry.children { self.traverse_node(graph, child); }
            }
            NodeData::MaterialBinding(mb) => {
                self.material_binding = match mb.value {
                    rc3d_scene::node_data::MaterialBinding::Overall => 1,
                    rc3d_scene::node_data::MaterialBinding::PerPart => 2,
                    rc3d_scene::node_data::MaterialBinding::PerFace => 3,
                    rc3d_scene::node_data::MaterialBinding::PerVertex => 4,
                    _ => 0,
                };
                for &child in &entry.children { self.traverse_node(graph, child); }
            }
            NodeData::Texture2Transform(t2t) => {
                self.tex2_transform = Some((
                    rc3d_core::math::Vec2::new(t2t.translation[0], t2t.translation[1]),
                    t2t.rotation,
                    rc3d_core::math::Vec2::new(t2t.scale[0], t2t.scale[1]),
                ));
                for &child in &entry.children { self.traverse_node(graph, child); }
            }
            NodeData::Annotation(_) => {
                let was_inside_annotation = self.inside_annotation;
                self.inside_annotation = true;
                for &child in &entry.children { self.traverse_node(graph, child); }
                self.inside_annotation = was_inside_annotation;
            }
            NodeData::AnnotationSet(ann) => {
                if ann.visible {
                    let set_matrix = self.state.model_matrix();
                    for el in &ann.elements {
                        let (element, el_model) =
                            rc3d_scene::annotation::prepare_annotation_for_render(
                                graph, set_matrix, el,
                            );
                        self.effect_commands.annotation_elements.push(
                            crate::render_passes::pass_effects::ProjectedAnnotation {
                                element,
                                model_matrix: el_model,
                                style: ann.style.clone(),
                            },
                        );
                    }
                }
            }
            NodeData::Environment(env) => {
                self.ambient_intensity = env.ambient_intensity;
                self.ambient_color = env.ambient_color;
                if env.fog_visibility > 0.0 {
                    self.fog_color = env.fog_color;
                    self.fog_visibility = env.fog_visibility;
                }
                for &child in &entry.children { self.traverse_node(graph, child); }
            }
            NodeData::IndexedLineSet(ils) => {
                let coord = self.state.coordinate();
                let mvp = self.state.projection_matrix() * self.state.view_matrix() * self.state.model_matrix();
                let mut edge_positions: Vec<[f32; 3]> = Vec::new();
                let mut aabb = rc3d_core::Aabb::empty();
                for i in (0..ils.coord_index.len()).step_by(2) {
                    if i + 1 < ils.coord_index.len() {
                        let a = ils.coord_index[i].max(0) as usize;
                        let b = ils.coord_index[i + 1].max(0) as usize;
                        if a < coord.points.len() && b < coord.points.len() {
                            let pa = coord.points[a]; let pb = coord.points[b];
                            edge_positions.push([pa.x, pa.y, pa.z]);
                            edge_positions.push([pb.x, pb.y, pb.z]);
                            aabb = aabb.union(&rc3d_core::Aabb::from_point(pa));
                            aabb = aabb.union(&rc3d_core::Aabb::from_point(pb));
                        }
                    }
                }
                if !edge_positions.is_empty() {
                    self.draw_calls.push(DrawCall {
                        vertices: Arc::new(Vec::new()),
                        edge_positions: Arc::new(edge_positions),
                        mvp,
                        model_matrix: self.state.model_matrix(),
                        camera_pos: self.camera_pos,
                        aabb: Some(aabb),
                        overlay_color: Some([0.5, 0.5, 0.5, 1.0]),
                        node_type_label: Arc::from("IndexedLineSet"),
                        is_overlay: self.inside_annotation,
                        ..Default::default()
                    });
                }
                for &child in &entry.children { self.traverse_node(graph, child); }
            }
            NodeData::Switch(sw) => {
                match sw.which_child {
                    -2 => {} // none
                    -1 => {
                        for &child in &sw.children {
                            self.traverse_node(graph, child);
                        }
                    }
                    idx if idx >= 0 => {
                        let i = idx as usize;
                        if i < sw.children.len() {
                            self.traverse_node(graph, sw.children[i]);
                        }
                    }
                    _ => {}
                }
            }
            NodeData::MultipleCopy(mc) => {
                let base = self.state.model_matrix();
                for &copy_mat in &mc.copies {
                    self.state.set_model_matrix(base * copy_mat);
                    for &child in &mc.children {
                        self.traverse_node(graph, child);
                    }
                }
                self.state.set_model_matrix(base);
            }
            NodeData::Lod(lod) => {
                let level = lod.current_level.min(lod.levels.len().saturating_sub(1));
                if let Some(level_data) = lod.levels.get(level) {
                    for &child in &level_data.children {
                        self.traverse_node(graph, child);
                    }
                }
            }
            NodeData::HandlerNode(h) => {
                h.traverse(graph, node, &entry.children, &mut |id| self.traverse_node(graph, id));
            }
            NodeData::EventCallback(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::PickStyle(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::SectionPlane(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::Text2(_) | NodeData::Text3(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::Measurement(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::Markup(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::MorphTarget(mt) => {
                self.state.set_morph_targets(mt.clone());
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::SkinnedMesh(sm) => {
                self.state.set_skinned_mesh(sm.clone());
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
                self.state.clear_skinned_mesh();
            }
            NodeData::Transform(t) => {
                let current = self.state.model_matrix();
                self.state.set_model_matrix(current * t.to_matrix());
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::Coordinate3(coord) => {
                self.state.set_coordinate(coord.point.clone());
            }
            NodeData::TextureCoordinate2(tex) => {
                self.state.set_texture_coordinate2(tex.point.clone());
            }
            NodeData::Normal(norm) => {
                self.state.set_normal(norm.vector.clone());
            }
            NodeData::Material(mat) => {
                let el = material_element_for_node(mat, entry.name.as_deref(), self.material_library.as_ref());
                self.state.set_material(el);
            }
            NodeData::PerspectiveCamera(cam) => {
                self.state.set_view_matrix(cam.view_matrix());
                self.state.set_projection_matrix(cam.projection_matrix());
                self.view_matrix = cam.view_matrix();
                self.projection_matrix = cam.projection_matrix();
                self.camera_pos = cam.position;
                self.projection_orthographic = false;
            }
            NodeData::OrthographicCamera(cam) => {
                self.state.set_view_matrix(cam.view_matrix());
                self.state.set_projection_matrix(cam.projection_matrix());
                self.view_matrix = cam.view_matrix();
                self.projection_matrix = cam.projection_matrix();
                self.camera_pos = cam.position;
                self.projection_orthographic = true;
            }
            NodeData::DirectionalLight(light) => {
                self.state.add_light(LightData {
                    light_type: LightType::Directional,
                    direction: light.direction,
                    location: Vec3::ZERO,
                    color: light.color,
                    intensity: light.intensity,
                    cut_off_angle: 0.0,
                    drop_off_rate: 0.0,
                });
            }
            NodeData::PointLight(light) => {
                self.state.add_light(LightData {
                    light_type: LightType::Point,
                    direction: Vec3::ZERO,
                    location: light.location,
                    color: light.color,
                    intensity: light.intensity,
                    cut_off_angle: 0.0,
                    drop_off_rate: 0.0,
                });
            }
            NodeData::SpotLight(light) => {
                self.state.add_light(LightData {
                    light_type: LightType::Spot,
                    direction: light.direction,
                    location: light.location,
                    color: light.color,
                    intensity: light.intensity,
                    cut_off_angle: light.cut_off_angle,
                    drop_off_rate: light.drop_off_rate,
                });
            }
            NodeData::AreaLight(light) => {
                self.state.add_light(LightData {
                    light_type: LightType::Point,
                    direction: light.direction, location: light.position,
                    color: light.color, intensity: light.intensity,
                    cut_off_angle: 0.0, drop_off_rate: 0.0,
                });
                for &child in &entry.children { self.traverse_node(graph, child); }
            }
            NodeData::Triangle(_) => {
                let coord = self.state.coordinate();
                if coord.points.len() < 3 {
                    return;
                }
                let normals = self.state.normal();
                let face_n = if normals.vectors.len() >= 3 {
                    normals.vectors[..3].to_vec()
                } else {
                    let c = (coord.points[1] - coord.points[0])
                        .cross(coord.points[2] - coord.points[0]);
                    let n = rc3d_core::utils::math::safe_normalize(c, Vec3::Y);
                    vec![n, n, n]
                };
                let positions = vec![coord.points[0], coord.points[1], coord.points[2]];
                let mut mesh = rc3d_mesh::TriangleMesh::from_tris(&positions);
                mesh.compute_tangents();
                let edge_feature = mesh.edge_line_positions_feature(
                    feature_crease_angle(),
                );
                let edge_full = mesh.edge_line_positions();
                let mut vertices = Vec::with_capacity(3);
                for (i, v) in mesh.phong_buffers().0.iter().enumerate() {
                    let n = if i < face_n.len() { face_n[i].to_array() } else { [v[3], v[4], v[5]] };
                    vertices.push(Vertex {
                        position: [v[0], v[1], v[2]],
                        normal: n,
                        texcoord: [v[6], v[7]],
                        tangent: [v[8], v[9], v[10], v[11]],
                    });
                }
                self.emit_draw_call_with_edges(
                    vertices,
                    Some((0..3u32).collect()),
                    edge_feature,
                    edge_full,
                    is_selected,
                    node_display_mode,
                    node_type_label,
                );
            }
            NodeData::Cube(cube) => {
                self.emit_cached_shape(
                    ShapeKey::Cube {
                        w: cube.width.to_bits(),
                        h: cube.height.to_bits(),
                        d: cube.depth.to_bits(),
                    },
                    || rc3d_mesh::tessellate_cube(cube.width, cube.height, cube.depth),
                    is_selected,
                    node_display_mode,
                    node_type_label,
                );
            }
            NodeData::Sphere(sphere) => {
                const SLICES: u32 = 24;
                const STACKS: u32 = 16;
                self.emit_cached_shape(
                    ShapeKey::Sphere {
                        r: sphere.radius.to_bits(),
                        slices: SLICES,
                        stacks: STACKS,
                    },
                    || rc3d_mesh::tessellate_sphere(sphere.radius, SLICES, STACKS),
                    is_selected,
                    node_display_mode,
                    node_type_label,
                );
            }
            NodeData::Cone(cone) => {
                const SEGMENTS: u32 = 24;
                self.emit_cached_shape(
                    ShapeKey::Cone {
                        r: cone.bottom_radius.to_bits(),
                        h: cone.height.to_bits(),
                        segments: SEGMENTS,
                    },
                    || rc3d_mesh::tessellate_cone(cone.bottom_radius, cone.height, SEGMENTS),
                    is_selected,
                    node_display_mode,
                    node_type_label,
                );
            }
            NodeData::Cylinder(cyl) => {
                const SEGMENTS: u32 = 24;
                self.emit_cached_shape(
                    ShapeKey::Cylinder {
                        r: cyl.radius.to_bits(),
                        h: cyl.height.to_bits(),
                        segments: SEGMENTS,
                    },
                    || rc3d_mesh::tessellate_cylinder(cyl.radius, cyl.height, SEGMENTS),
                    is_selected,
                    node_display_mode,
                    node_type_label,
                );
            }
            NodeData::Torus(torus) => {
                const MAJOR_SEGMENTS: u32 = 32;
                const MINOR_SEGMENTS: u32 = 16;
                self.emit_cached_shape(
                    ShapeKey::Torus {
                        major_r: torus.major_radius.to_bits(),
                        minor_r: torus.minor_radius.to_bits(),
                        major_segments: MAJOR_SEGMENTS,
                        minor_segments: MINOR_SEGMENTS,
                    },
                    || rc3d_mesh::tessellate_torus(torus.major_radius, torus.minor_radius, MAJOR_SEGMENTS, MINOR_SEGMENTS),
                    is_selected,
                    node_display_mode,
                    node_type_label,
                );
            }
            NodeData::IndexedFaceSet(ifs) => {
                let coord = self.state.coordinate();
                if coord.points.is_empty() {
                    return;
                }
                let points_sig = if coord.points.len() >= 2 {
                    let first = coord.points[0].to_array();
                    let last = coord.points[coord.points.len() - 1].to_array();
                    [
                        first[0].to_bits(),
                        first[1].to_bits(),
                        first[2].to_bits(),
                        last[0].to_bits(),
                        last[1].to_bits(),
                        last[2].to_bits(),
                    ]
                } else {
                    let p = coord.points[0].to_array();
                    [
                        p[0].to_bits(),
                        p[1].to_bits(),
                        p[2].to_bits(),
                        p[0].to_bits(),
                        p[1].to_bits(),
                        p[2].to_bits(),
                    ]
                };
                let index_sig = if ifs.coord_index.len() >= 2 {
                    [ifs.coord_index[0], ifs.coord_index[ifs.coord_index.len() - 1]]
                } else if ifs.coord_index.len() == 1 {
                    [ifs.coord_index[0], ifs.coord_index[0]]
                } else {
                    [0, 0]
                };
                let tex_el = self.state.texture_coordinate2();
                let (tex_len, tex_sig) = if tex_el.coords.is_empty() {
                    (0u32, [0u32; 4])
                } else {
                    let first = tex_el.coords[0];
                    let last = tex_el.coords[tex_el.coords.len() - 1];
                    (
                        tex_el.coords.len() as u32,
                        [
                            first[0].to_bits(),
                            first[1].to_bits(),
                            last[0].to_bits(),
                            last[1].to_bits(),
                        ],
                    )
                };
                let norm_el = self.state.normal();
                let (normal_len, normal_sig) =
                    if norm_el.vectors.len() == coord.points.len() && !norm_el.vectors.is_empty()
                    {
                        let first = norm_el.vectors[0].to_array();
                        let last = norm_el.vectors[norm_el.vectors.len() - 1].to_array();
                        (
                            norm_el.vectors.len() as u32,
                            [
                                first[0].to_bits(),
                                first[1].to_bits(),
                                first[2].to_bits(),
                                last[0].to_bits(),
                                last[1].to_bits(),
                                last[2].to_bits(),
                            ],
                        )
                    } else {
                        (0u32, [0u32; 6])
                    };
                let key = ShapeKey::IndexedFaceSet {
                    node: node.data().as_ffi(),
                    coord_len: coord.points.len() as u32,
                    coord_index_len: ifs.coord_index.len() as u32,
                    points_sig,
                    index_sig,
                    tex_len,
                    tex_sig,
                    normal_len,
                    normal_sig,
                };
                let cached: Option<CachedShapeData> = match self.mesh_cache.entry(key) {
                    Entry::Occupied(occupied) => Some(occupied.get().clone()),
                    Entry::Vacant(vacant) => {
                        let mut mesh = {
                            let use_tex =
                                !tex_el.coords.is_empty() && tex_el.coords.len() == coord.points.len();
                            if use_tex {
                                rc3d_mesh::TriangleMesh::from_indexed_face_set_tex(
                                    &coord.points,
                                    &tex_el.coords,
                                    &ifs.coord_index,
                                )
                            } else {
                                rc3d_mesh::TriangleMesh::from_indexed_face_set(
                                    &coord.points,
                                    &ifs.coord_index,
                                )
                            }
                        };
                        if norm_el.vectors.len() == mesh.positions.len()
                            && !norm_el.vectors.is_empty()
                        {
                            let computed_backup = mesh.normals.clone();
                            mesh.normals.clone_from(&norm_el.vectors);
                            for (i, n) in mesh.normals.iter_mut().enumerate() {
                                let len = n.length();
                                if len > 1e-20 {
                                    *n /= len;
                                } else {
                                    let fb =
                                        computed_backup.get(i).copied().unwrap_or(Vec3::Y);
                                    let l = fb.length();
                                    *n = if l > 1e-20 { fb / l } else { Vec3::Y };
                                }
                            }
                        }
                        mesh.compute_tangents();
                        if mesh.positions.is_empty() {
                            None
                        } else {
                            let (phong_verts, indices) = mesh.phong_buffers();
                            let edge_feature = mesh.edge_line_positions_feature(
                                feature_crease_angle(),
                            );
                            let edge_full = mesh.edge_line_positions();
                            let local_aabb = mesh.bounding_box();
                            let vertices: Vec<Vertex> = phong_verts
                                .iter()
                                .map(|v| Vertex {
                                    position: [v[0], v[1], v[2]],
                                    normal: [v[3], v[4], v[5]],
                                    texcoord: [v[6], v[7]],
                                    tangent: [v[8], v[9], v[10], v[11]],
                                })
                                .collect();
                            let tri_count = indices.len() / 3;
                            let meshlet_data = if tri_count > MESHLET_TRIANGLE_THRESHOLD
                                && self.state.skinned_mesh().is_none()
                            {
                                let md = rc3d_mesh::build_meshlets_from_mesh(
                                    &mesh.positions,
                                    &mesh.normals,
                                    &mesh.texcoords,
                                    &mesh.tangents,
                                    &mesh.tri_indices,
                                );
                                log::info!(
                                    "Meshlet: {} tris -> {} meshlets ({} verts)",
                                    tri_count,
                                    md.total_meshlets,
                                    md.vertices.len(),
                                );
                                Some(Arc::new(md))
                            } else {
                                None
                            };
                            Some(vacant.insert((
                                Arc::new(vertices),
                                Arc::new(indices),
                                Arc::new(edge_feature),
                                Arc::new(edge_full),
                                local_aabb,
                                meshlet_data,
                            )).clone())
                        }
                    }
                };
                if let Some((vertices, indices, edge_feature, edge_full, local_aabb, meshlet_data)) =
                    cached
                {
                    let edge_feature = clamp_edge_positions(edge_feature.clone());
                    let edge_full = clamp_edge_positions(edge_full.clone());
                    self.emit_draw_call_with_cached_aabb(
                        vertices.clone(),
                        Some(indices.clone()),
                        edge_feature,
                        edge_full,
                        local_aabb.clone(),
                        meshlet_data.clone(),
                        is_selected,
                        node_display_mode,
                        node_type_label,
                    );
                }
            }
            NodeData::Custom(_, _) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
        }
    }

    fn emit_cached_shape<F>(
        &mut self,
        key: ShapeKey,
        build_mesh: F,
        selected: bool,
        node_display_mode: Option<DisplayMode>,
        node_type_label: &str,
    ) where
        F: FnOnce() -> rc3d_mesh::TriangleMesh,
    {
        match self.mesh_cache.entry(key) {
            Entry::Occupied(_) => {}
            Entry::Vacant(vacant) => {
                let mut mesh = build_mesh();
                if mesh.positions.is_empty() {
                    vacant.insert((
                        Arc::new(Vec::new()),
                        Arc::new(Vec::new()),
                        Arc::new(Vec::new()),
                        Arc::new(Vec::new()),
                        rc3d_core::Aabb::empty(),
                        None,
                    ));
                    return;
                }
                mesh.compute_tangents();
                let (phong_verts, indices) = mesh.phong_buffers();
                let edge_feature = mesh.edge_line_positions_feature(
                    feature_crease_angle(),
                );
                let edge_full = mesh.edge_line_positions();
                let local_aabb = mesh.bounding_box();
                let vertices: Vec<Vertex> = phong_verts
                    .iter()
                    .map(|v| Vertex {
                        position: [v[0], v[1], v[2]],
                        normal: [v[3], v[4], v[5]],
                        texcoord: [v[6], v[7]],
                        tangent: [v[8], v[9], v[10], v[11]],
                    })
                    .collect();
                vacant.insert((
                    Arc::new(vertices),
                    Arc::new(indices),
                    Arc::new(edge_feature),
                    Arc::new(edge_full),
                    local_aabb,
                    None,
                ));
            }
        }

        if let Some((vertices, indices, edge_feature, edge_full, local_aabb, meshlet_data)) = self.mesh_cache.get(&key) {
            let edge_feature = clamp_edge_positions(Arc::clone(edge_feature));
            let edge_full = clamp_edge_positions(Arc::clone(edge_full));
            self.emit_draw_call_with_cached_aabb(
                Arc::clone(vertices),
                Some(Arc::clone(indices)),
                edge_feature,
                edge_full,
                local_aabb.clone(),
                meshlet_data.clone(),
                selected,
                node_display_mode,
                node_type_label,
            );
        }
    }

    fn emit_draw_call_with_cached_aabb(
        &mut self,
        vertices: Arc<Vec<Vertex>>,
        indices: Option<Arc<Vec<u32>>>,
        edge_positions: Arc<Vec<[f32; 3]>>,
        wireframe_edge_positions: Arc<Vec<[f32; 3]>>,
        local_aabb: rc3d_core::Aabb,
        meshlet_data: Option<Arc<rc3d_mesh::MeshletData>>,
        selected: bool,
        node_display_mode: Option<DisplayMode>,
        node_type_label: &str,
    ) {
        let model = self.state.model_matrix();
        let mvp = self.state.projection_matrix() * self.state.view_matrix() * model;
        let mat = self.state.material();
        let aabb = Some(local_aabb.transform(model));
        let packed = self.collect_lights();
        let light_key = {
            let (ref light_dirs, ref light_colors, ref light_types, ref light_positions, ref spot_params, light_count) = packed;
            hash_light_params(light_dirs, light_colors, light_types, light_positions, spot_params, light_count)
        };

        self.draw_calls.push(DrawCall {
            vertices,
            is_overlay: self.inside_annotation,
            indices,
            edge_positions,
            wireframe_edge_positions,
            mvp,
            model_matrix: model,
            camera_pos: self.camera_pos,
            light_set_id: self.light_sets.intern(light_key, packed),
            light_key,
            diffuse_color: mat.diffuse,
            ambient_color: mat.ambient,
            specular_color: mat.specular,
            shininess: mat.shininess,
            base_color: mat.base_color,
            metallic: mat.metallic,
            roughness: mat.roughness,
            anisotropic: mat.anisotropic,
            opacity: mat.opacity,
            albedo_path: mat
                .albedo_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            normal_path: mat
                .normal_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            emissive_color: mat.emissive_color,
            emissive_path: mat
                .emissive_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            metallic_roughness_path: mat
                .metallic_roughness_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            occlusion_path: mat
                .occlusion_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            alpha_mode: mat.alpha_mode,
            alpha_cutoff: mat.alpha_cutoff,
            double_sided: mat.double_sided,
            aabb,
            display_mode: node_display_mode.unwrap_or(DisplayMode::ShadedWithEdges),
            selected,
            overlay_color: None,
            mesh_hash: None,
            meshlet_data,
            projection_orthographic: self.projection_orthographic,
            depth_reversed_z: rc3d_core::depth_reversed_z_from_projection(
                self.state.projection_matrix(),
            ),
            node_type_label: Arc::from(node_type_label),
            instance_transforms: None,
            morph_weights: self.state.morph_targets().map(|mt| mt.weights.clone()).unwrap_or_default(),
            morph_target_deltas: self.state.morph_targets().map(|mt| Arc::new(mt.clone())),
            skinning: self.skinning_payload_for_draw(),
        });
        if !self.cache_ptr.is_null() {
            self.emit_to_cache(self.draw_calls.last().unwrap());
        }
    }

    fn emit_draw_call_with_edges(
        &mut self,
        vertices: Vec<Vertex>,
        indices: Option<Vec<u32>>,
        edge_feature: Vec<[f32; 3]>,
        edge_wireframe: Vec<[f32; 3]>,
        selected: bool,
        node_display_mode: Option<DisplayMode>,
        node_type_label: &str,
    ) {
        let model = self.state.model_matrix();
        let mvp = self.state.projection_matrix() * self.state.view_matrix() * model;
        let mat = self.state.material();
        let packed = self.collect_lights();
        let light_key = {
            let (ref light_dirs, ref light_colors, ref light_types, ref light_positions, ref spot_params, light_count) = packed;
            hash_light_params(light_dirs, light_colors, light_types, light_positions, spot_params, light_count)
        };

        let aabb = if vertices.is_empty() {
            None
        } else {
            let first = model.transform_point3(Vec3::from_array(vertices[0].position));
            let mut aabb = rc3d_core::Aabb::from_point(first);
            for v in &vertices[1..] {
                let p = model.transform_point3(Vec3::from_array(v.position));
                aabb = aabb.union(&rc3d_core::Aabb::from_point(p));
            }
            Some(aabb)
        };

        self.draw_calls.push(DrawCall {
            vertices: Arc::new(vertices),
            is_overlay: self.inside_annotation,
            indices: indices.map(Arc::new),
            edge_positions: Arc::new(edge_feature),
            wireframe_edge_positions: Arc::new(edge_wireframe),
            mvp,
            model_matrix: model,
            camera_pos: self.camera_pos,
            light_set_id: self.light_sets.intern(light_key, packed),
            light_key,
            diffuse_color: mat.diffuse,
            ambient_color: mat.ambient,
            specular_color: mat.specular,
            shininess: mat.shininess,
            base_color: mat.base_color,
            metallic: mat.metallic,
            roughness: mat.roughness,
            anisotropic: mat.anisotropic,
            opacity: mat.opacity,
            albedo_path: mat
                .albedo_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            normal_path: mat
                .normal_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            emissive_color: mat.emissive_color,
            emissive_path: mat
                .emissive_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            metallic_roughness_path: mat
                .metallic_roughness_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            occlusion_path: mat
                .occlusion_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            alpha_mode: mat.alpha_mode,
            alpha_cutoff: mat.alpha_cutoff,
            double_sided: mat.double_sided,
            aabb,
            display_mode: node_display_mode.unwrap_or(DisplayMode::ShadedWithEdges),
            selected,
            overlay_color: None,
            mesh_hash: None,
            meshlet_data: None,
            projection_orthographic: self.projection_orthographic,
            depth_reversed_z: rc3d_core::depth_reversed_z_from_projection(
                self.state.projection_matrix(),
            ),
            node_type_label: Arc::from(node_type_label),
            instance_transforms: None,
            morph_weights: self.state.morph_targets().map(|mt| mt.weights.clone()).unwrap_or_default(),
            morph_target_deltas: self.state.morph_targets().map(|mt| Arc::new(mt.clone())),
            skinning: self.skinning_payload_for_draw(),
        });
        if !self.cache_ptr.is_null() {
            self.emit_to_cache(self.draw_calls.last().unwrap());
        }
    }

    fn skinning_payload_for_draw(&self) -> Option<Arc<SkinnedMeshDrawPayload>> {
        self.state.skinned_mesh().map(|sm| {
            Arc::new(SkinnedMeshDrawPayload {
                skeleton: sm.skeleton.clone(),
                skin_data: Arc::new(sm.skin_data.clone()),
                clip: sm.clip.clone(),
            })
        })
    }

    fn collect_lights(&self) -> PackedLights {
        let mut dirs = [[0.0f32; 4]; MAX_LIGHTS];
        let mut colors = [[0.0f32; 4]; MAX_LIGHTS];
        let mut types = [[0.0f32; 4]; MAX_LIGHTS];
        let mut positions = [[0.0f32; 4]; MAX_LIGHTS];
        let mut spot_params = [[0.0f32; 4]; MAX_LIGHTS];
        let mut count = 0u32;
        let mut warned = false;
        let total_lights = self.state.lights().len();
        for light in self.state.lights() {
            if (count as usize) < MAX_LIGHTS {
                let idx = count as usize;
                dirs[idx] = [light.direction.x, light.direction.y, light.direction.z, 0.0];
                let c = light.color * light.intensity;
                colors[idx] = [c.x, c.y, c.z, 1.0];
                positions[idx] = [light.location.x, light.location.y, light.location.z, 1.0];
                types[idx][0] = match light.light_type {
                    LightType::Directional => 0.0,
                    LightType::Point => 1.0,
                    LightType::Spot => 2.0,
                };
                spot_params[idx] = [light.cut_off_angle.cos(), light.drop_off_rate, 0.0, 0.0];
                count += 1;
            } else {
                warned = true;
                break;
            }
        }
        if warned {
            log::warn!("MAX_LIGHTS ({}) exceeded; {} lights truncated", MAX_LIGHTS, total_lights.saturating_sub(MAX_LIGHTS));
        }
        if count == 0 {
            dirs[0] = [0.0, 0.0, -1.0, 0.0];
            colors[0] = [1.0, 1.0, 1.0, 1.0];
            types[0][0] = 0.0;
            count = 1;
        }
        (dirs, colors, types, positions, spot_params, count)
    }
}

impl Default for RenderCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl rc3d_actions::Action for RenderCollector {
    fn kind(&self) -> rc3d_actions::ActionKind {
        rc3d_actions::ActionKind::GLRender
    }

    fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        self.traverse(graph, root);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;
    use rc3d_core::math::Vec3;
    use rc3d_scene::{
        AnnotationNode, Coordinate3Node, GroupNode, IndexedLineSetNode, NodeData, SceneGraph,
    };

    #[test]
    fn view_projection_from_draw_call_strips_model() {
        let model = Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 0.0));
        let view = Mat4::look_at_rh(glam::Vec3::new(0.0, 2.0, 8.0), glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let expected_vp = proj * view;
        let mut dc = DrawCall::default();
        dc.model_matrix = model;
        apply_world_camera(
            std::slice::from_mut(&mut dc),
            view,
            proj,
            Vec3::new(0.0, 2.0, 8.0),
        );
        let extracted = view_projection_from_draw_call(&dc);
        assert!(
            extracted.abs_diff_eq(expected_vp, 1e-4),
            "VP should match camera after stripping model"
        );
    }

    #[test]
    fn draw_call_vp_must_win_over_stale_cached_scene_vp() {
        let view_a = Mat4::look_at_rh(glam::Vec3::new(0.0, 2.0, 8.0), glam::Vec3::ZERO, glam::Vec3::Y);
        let view_b = Mat4::look_at_rh(glam::Vec3::new(5.0, 2.0, 8.0), glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let stale_vp = proj * view_a;
        let mut dc = DrawCall::default();
        apply_world_camera(std::slice::from_mut(&mut dc), view_b, proj, Vec3::new(5.0, 2.0, 8.0));
        let fresh_vp = view_projection_from_draw_call(&dc);
        let chosen = if fresh_vp != Mat4::IDENTITY {
            fresh_vp
        } else {
            stale_vp
        };
        assert!(chosen.abs_diff_eq(proj * view_b, 1e-4));
        assert!(!chosen.abs_diff_eq(stale_vp, 1e-4));
    }

    #[test]
    fn annotation_children_emit_overlay_draw_calls() {
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Group(GroupNode));
        let annotation = graph.add_child(root, NodeData::Annotation(AnnotationNode));
        graph.add_child(
            annotation,
            NodeData::Coordinate3(Coordinate3Node::from_points(vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
            ])),
        );
        graph.add_child(
            annotation,
            NodeData::IndexedLineSet(IndexedLineSetNode {
                coord_index: vec![0, 1],
                line_width: 1.0,
            }),
        );

        let mut collector = RenderCollector::new();
        collector.traverse(&graph, root);

        assert_eq!(collector.draw_calls.len(), 1);
        assert!(collector.draw_calls[0].is_overlay);
        assert!(!collector.inside_annotation);
    }

    #[test]
    fn collector_collects_effect_commands_from_decal_volume_pointcloud() {
        use rc3d_scene::node_data::{
            DecalNode, PointCloudNode, SeparatorNode, VolumeNode,
        };

        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        graph.add_child(root, NodeData::Decal(DecalNode {
            position: Vec3::new(1.0, 0.0, 0.0),
            direction: Vec3::NEG_Y,
            size: [2.0, 2.0],
            texture_path: "d.png".to_string(),
            color: [1.0, 0.0, 0.0, 0.5],
            opacity: 0.8,
        }));
        graph.add_child(root, NodeData::Volume(VolumeNode {
            dimensions: [32, 32, 32],
            texture_path: "v.raw".to_string(),
            density_scale: 1.0,
            color_map: [[0.0; 4]; 4],
        }));
        graph.add_child(root, NodeData::PointCloud(PointCloudNode {
            file_path: "p.bin".to_string(),
            max_visible_points: 1000,
            point_size: 2.0,
            color: [0.0, 1.0, 0.0, 1.0],
        }));

        let mut collector = RenderCollector::new();
        collector.traverse(&graph, root);

        assert_eq!(collector.effect_commands.decals.len(), 1);
        assert_eq!(collector.effect_commands.decals[0].texture_path, "d.png");
        assert_eq!(collector.effect_commands.volumes.len(), 1);
        assert_eq!(collector.effect_commands.volumes[0].texture_path, "v.raw");
        assert_eq!(collector.effect_commands.point_clouds.len(), 1);
        assert_eq!(collector.effect_commands.point_clouds[0].file_path, "p.bin");
        assert_eq!(collector.effect_commands.point_clouds[0].max_visible_points, 1000);
    }

    #[test]
    fn inside_annotation_sets_is_overlay_on_effect_commands() {
        use rc3d_scene::node_data::{AnnotationNode, DecalNode, SeparatorNode};

        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        let annotation = graph.add_child(root, NodeData::Annotation(AnnotationNode::default()));
        graph.add_child(annotation, NodeData::Decal(DecalNode {
            texture_path: "overlay.png".to_string(),
            ..Default::default()
        }));

        let mut collector = RenderCollector::new();
        collector.traverse(&graph, root);

        assert_eq!(collector.effect_commands.decals.len(), 1);
        assert!(collector.effect_commands.decals[0].is_overlay);
    }

    #[test]
    fn collector_produces_draw_calls_for_all_geometry_types() {
        use rc3d_scene::node_data::{
            ConeNode, CubeNode, CylinderNode, MaterialNode, SeparatorNode, SphereNode,
            TorusNode, TransformNode, TriangleNode,
        };
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        graph.add_child(root, NodeData::Material(MaterialNode::default()));
        let geom = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(1.0, 0.0, 0.0))),
        );
        graph.add_child(geom, NodeData::Cube(CubeNode::default()));
        graph.add_child(geom, NodeData::Sphere(SphereNode::default()));
        graph.add_child(geom, NodeData::Cone(ConeNode::default()));
        graph.add_child(geom, NodeData::Cylinder(CylinderNode::default()));
        graph.add_child(geom, NodeData::Torus(TorusNode::default()));
        // Triangle and other unit shapes produce valid draw calls
        graph.add_child(root, NodeData::Triangle(TriangleNode));

        let mut collector = RenderCollector::new();
        collector.traverse(&graph, root);

        assert!(
            collector.draw_calls.len() >= 5,
            "expected at least 4 draw calls (Cube/Sphere/Cone/Cylinder), got {}",
            collector.draw_calls.len()
        );
    }

    #[test]
    fn nested_transform_chains_to_children() {
        use rc3d_scene::node_data::{CubeNode, SeparatorNode, TransformNode};
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        let parent = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(1.0, 2.0, 3.0))),
        );
        let child = graph.add_child(
            parent,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(4.0, 5.0, 6.0))),
        );
        graph.add_child(child, NodeData::Cube(CubeNode::default()));

        let mut collector = RenderCollector::new();
        collector.traverse(&graph, root);

        assert!(!collector.draw_calls.is_empty());
        let dc = &collector.draw_calls[0];
        let expected = Vec3::new(5.0, 7.0, 9.0);
        let actual = dc.model_matrix.w_axis.truncate();
        assert!(
            (actual - expected).length() < 0.01,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn empty_scene_produces_no_draw_calls() {
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Separator(
            rc3d_scene::node_data::SeparatorNode,
        ));
        let mut collector = RenderCollector::new();
        collector.traverse(&graph, root);
        assert!(collector.draw_calls.is_empty());
        assert!(collector.effect_commands.is_empty());
    }

    #[test]
    fn annotation_marks_draw_calls_and_effects_as_overlay() {
        use rc3d_scene::node_data::{
            AnnotationNode, CubeNode, DecalNode, MaterialNode, SeparatorNode,
        };
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        let ann = graph.add_child(root, NodeData::Annotation(AnnotationNode::default()));
        graph.add_child(ann, NodeData::Material(MaterialNode::default()));
        graph.add_child(ann, NodeData::Cube(CubeNode::default()));
        graph.add_child(ann, NodeData::Decal(DecalNode {
            texture_path: "ann_decal.png".to_string(),
            ..Default::default()
        }));

        let mut collector = RenderCollector::new();
        collector.traverse(&graph, root);

        // All draw calls under annotation should have is_overlay
        for dc in &collector.draw_calls {
            assert!(dc.is_overlay, "draw call should be overlay inside annotation");
        }
        for dc in &collector.effect_commands.decals {
            assert!(dc.is_overlay, "decal should be overlay inside annotation");
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Incremental traversal adapter: dirty flag propagation + FlatDrawCache fill
// ──────────────────────────────────────────────────────────────────────────

use crate::flat_draw_cache::FlatDrawCache;
use crate::global_tables::TexturePathTable;

/// Incremental traversal: collect draw data from dirty subtrees into FlatDrawCache.
/// Static subtrees are reused from previous frames' cache.
pub fn traverse_into_cache(
    graph: &SceneGraph,
    cache: &mut FlatDrawCache,
    texture_table: &mut TexturePathTable,
    hidden_nodes: &HashSet<rc3d_core::NodeId>,
) {
    let dirty_roots = crate::dirty_flags::collect_dirty_roots(graph);

    if dirty_roots.is_empty() && !cache.gpu_data.is_empty() {
        // Nothing changed AND cache is populated — reuse as-is
        cache.ensure_groups_sorted();
        return;
    }

    // Count total nodes to decide full vs incremental rebuild
    let total_nodes = crate::traversal::count_all_nodes(graph);
    if dirty_roots.len() as f64 > total_nodes as f64 * 0.5 {
        // Full rebuild — emit directly to cache during traversal
        cache.clear();
        for &root in graph.roots() {
            let mut collector = RenderCollector::new();
            collector.set_cache_target(cache, texture_table);
            collector.set_hidden_nodes(hidden_nodes);
            collector.traverse(graph, root);
        }
    } else {
        // Incremental: remove dirty entries, re-traverse dirty subtrees
        for &dirty_root in &dirty_roots {
            crate::traversal::invalidate_cache_for_subtree(cache, dirty_root);
            let mut collector = RenderCollector::new();
            collector.set_cache_target(cache, texture_table);
            collector.set_hidden_nodes(hidden_nodes);
            collector.traverse(graph, dirty_root);
        }
    }

    cache.groups_dirty = true;
    cache.ensure_groups_sorted();
    // Note: caller is responsible for clear_all_dirty_flags
}
