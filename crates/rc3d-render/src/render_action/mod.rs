use rc3d_core::math::{Mat4, Vec3, Vec4};
use rc3d_core::{Appearance, DisplayMode, EdgeStyle, NodeId};
use rc3d_scene::{
    billboard_facing, scene_traverse, ChildPolicy, LightData, LightType, MaterialElement, NodeData, NodeEntry,
    SceneGraph, SceneVisitor, SeparatorPolicy, State, TraversalMatrices,
};
use slotmap::Key;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::material_library::MaterialLibrary;
use crate::vertex::Vertex;

// Re-export shape_cache items for backwards compatibility via crate::render_action::
pub use crate::shape_cache::{
    feature_crease_angle, set_feature_crease_angle, clamp_edge_positions,
    ShapeKey, CachedShapeData, MAX_EDGE_POSITIONS, MESHLET_TRIANGLE_THRESHOLD,
};

// Re-export light_packing items
pub use crate::light_packing::{hash_light_params, collect_lights, PackedLights};

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
) -> MaterialElement {
    let src = if let (Some(name), Some(lib)) = (entry_name, library) {
        lib.get(name).unwrap_or(mat)
    } else {
        mat
    };
    MaterialElement {
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
        clearcoat_factor: src.clearcoat_factor,
        clearcoat_roughness: src.clearcoat_roughness,
        specular_factor: src.specular_factor,
        specular_color_factor: src.specular_color_factor,
        transmission_factor: src.transmission_factor,
        ior: src.ior,
        sheen_color: src.sheen_color,
        sheen_roughness: src.sheen_roughness,
        iridescence_factor: src.iridescence_factor,
        iridescence_ior: src.iridescence_ior,
        iridescence_thickness_min: src.iridescence_thickness_min,
        iridescence_thickness_max: src.iridescence_thickness_max,
        toon_steps: src.toon_steps,
        visualize_normals: src.visualize_normals,
        visualize_depth: src.visualize_depth,
        custom_wgsl: src.custom_wgsl.clone(),
        custom_uniforms: src.custom_uniforms,
    }
}

mod draw_call;
pub use draw_call::{
    apply_ghost_unselected, apply_world_camera, apply_world_camera_ex,
    view_projection_from_draw_call, DrawCall, GHOST_UNSELECTED_OPACITY, SkinnedMeshDrawPayload,
};


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
    /// True while traversing inside an Annotation node.
    pub inside_annotation: bool,
    /// Pending instance transforms (set by InstancedMeshNode traversal).
    /// Applied to the next emitted draw call, then cleared.
    pub pending_instance_transforms: Option<Arc<Vec<Mat4>>>,
    /// Effect draw commands collected during traversal (Decal, Volume, PointCloud).
    pub effect_commands: crate::render_passes::pass_effects::EffectCommands,
    /// Last LightProbe wins (v1: infinite / no falloff). Zero intensity disables SH IBL.
    pub light_probe_sh: [[f32; 4]; 9],
    pub light_probe_intensity: f32,
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
            inside_annotation: false,
            pending_instance_transforms: None,
            effect_commands: crate::render_passes::pass_effects::EffectCommands::default(),
            light_probe_sh: [[0.0; 4]; 9],
            light_probe_intensity: 0.0,
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
        self.state.set_display_mode(self.global_display_mode);
        scene_traverse(self, graph, root);
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
        cache.total_triangles += if dc.index_draw_count > 0 {
            (dc.index_draw_count / 3) as u64
        } else {
            dc.indices.as_ref().map_or(
                (dc.vertices.len() / 3) as u64,
                |idx| (idx.len() / 3) as u64,
            )
        };
        cache.groups_dirty = true;
    }

    pub fn invalidate_mesh_cache(&mut self) {
        self.mesh_cache.clear();
    }

    pub fn set_hidden_nodes(&mut self, hidden_nodes: &HashSet<NodeId>) {
        self.hidden_nodes = hidden_nodes.clone();
    }

    fn traverse_entry_children(&mut self, graph: &SceneGraph, entry: &NodeEntry) -> ChildPolicy {
        for &child in &entry.children {
            scene_traverse(self, graph, child);
        }
        ChildPolicy::Skip
    }
}

impl TraversalMatrices for RenderCollector {
    fn model_matrix(&self) -> Mat4 {
        self.state.model_matrix()
    }

    fn set_model_matrix(&mut self, matrix: Mat4) {
        self.state.set_model_matrix(matrix);
    }

    fn view_matrix(&self) -> Mat4 {
        self.state.view_matrix()
    }
}


include!("visit.rs");

include!("emit.rs");

impl Default for RenderCollector {
    fn default() -> Self {
        Self::new()
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
                ..Default::default()
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
            ..Default::default()
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
