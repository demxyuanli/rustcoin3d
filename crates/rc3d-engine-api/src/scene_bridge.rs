//! Bridge from `rc3d-scene-api` and `rc3d-effects` to the low-level engine.
//!
//! Converts `Scene` → `SceneGraph`, stores `RenderConfig` / `EffectGraph`,
//! and provides file import conveniences.
//!
//! Note: `impl App` extension methods (`from_scene`, `with_effects_config`, etc.)
//! remain in `rc3d-app/src/scene_bridge.rs` because they access private `App` fields.

use rc3d_core::math::{Mat4, Vec3};
use rc3d_scene::SceneGraph;

// ══════════════════════════════════════════════════════════════════════════════
// Dynamic NURBS surface with view-dependent re-tessellation
// ══════════════════════════════════════════════════════════════════════════════

/// A NURBS surface that re-tessellates when the camera settles.
///
/// During camera interaction (orbit/pan/zoom), tessellation is frozen to maintain FPS.
/// Once the camera has been still for `settle_frames`, the surface re-tessellates
/// with screen-space criteria. This avoids the per-frame CPU + GPU churn of dynamic
/// subdivision during continuous camera movement.
pub struct DynamicSurface {
    surface: rc3d_nurbs::NurbsRenderSurface,
    coord_node: rc3d_core::NodeId,
    normal_node: rc3d_core::NodeId,
    ifs_node: rc3d_core::NodeId,
    last_tess_eye: Vec3,
    light_dir: Vec3,
    still_frames: u32,
    settle_frames: u32,
    move_threshold: f32,
    max_px: f32,
    silhouette_px: f32,
    terminator_px: f32,
    angle_tol: f32,
    max_depth: usize,
}

impl DynamicSurface {
    /// Create a new dynamic surface and add its scene-graph nodes under `parent`.
    ///
    /// The nodes (Separator + Coordinate3 + Normal + IndexedFaceSet) are created
    /// with an initial tessellation and will be updated each frame as the camera moves.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        graph: &mut SceneGraph,
        parent: rc3d_core::NodeId,
        surface: rc3d_nurbs::NurbsRenderSurface,
        camera_pos: Vec3,
        light_dir: Vec3,
        mvp: &Mat4,
        viewport: (f32, f32),
        settle_frames: u32,
        move_threshold: f32,
        max_px: f32,
        silhouette_px: f32,
        terminator_px: f32,
        angle_tol: f32,
        max_depth: usize,
    ) -> Self {
        use rc3d_scene::node_data::*;

        let ts = surface.tessellate_screen_space(
            mvp, viewport, camera_pos, light_dir, max_px, silhouette_px, terminator_px, angle_tol, max_depth,
        );

        let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));
        let coord = graph.add_child(
            sep,
            NodeData::Coordinate3(Coordinate3Node::from_points(ts.positions)),
        );
        let normal = graph.add_child(
            sep,
            NodeData::Normal(NormalNode::from_vectors(ts.normals)),
        );
        let coord_index = triangle_indices_to_coord_index(&ts.indices);
        let ifs = graph.add_child(
            sep,
            NodeData::IndexedFaceSet(IndexedFaceSetNode {
                coord_index,
            }),
        );

        Self {
            surface,
            coord_node: coord,
            normal_node: normal,
            ifs_node: ifs,
            last_tess_eye: camera_pos,
            light_dir,
            still_frames: 0,
            settle_frames,
            move_threshold,
            max_px,
            silhouette_px,
            terminator_px,
            angle_tol,
            max_depth,
        }
    }

    /// Re-tessellate only when the camera has settled.
    ///
    /// `camera_moved` should be true during active interaction (orbit/pan/zoom).
    /// When false for `settle_frames` consecutive frames AND the camera has moved
    /// more than `move_threshold` since the last tessellation, the surface is
    /// re-tessellated.
    pub fn update(
        &mut self,
        graph: &mut SceneGraph,
        camera_pos: Vec3,
        mvp: &Mat4,
        viewport: (f32, f32),
        camera_moved: bool,
    ) {
        if camera_moved {
            self.still_frames = 0;
            return;
        }
        self.still_frames += 1;
        if self.still_frames < self.settle_frames {
            return;
        }
        // Reset to avoid re-tessellating every frame after settling
        self.still_frames = 0;

        let dist = (camera_pos - self.last_tess_eye).length();
        if dist < self.move_threshold {
            return;
        }
        self.last_tess_eye = camera_pos;

        let ts = self.surface.tessellate_screen_space(
            mvp, viewport, camera_pos, self.light_dir,
            self.max_px, self.silhouette_px, self.terminator_px,
            self.angle_tol, self.max_depth,
        );

        // Update Coordinate3 in-place
        if let Some(entry) = graph.get_mut(self.coord_node) {
            if let rc3d_scene::node_data::NodeData::Coordinate3(coord) = &mut entry.data {
                coord.point = ts.positions;
                entry.dirty_flags |= rc3d_scene::node_entry::dirty_flags::GEOMETRY;
            }
        }

        // Update Normal in-place
        if let Some(entry) = graph.get_mut(self.normal_node) {
            if let rc3d_scene::node_data::NodeData::Normal(normal) = &mut entry.data {
                normal.vector = ts.normals;
                entry.dirty_flags |= rc3d_scene::node_entry::dirty_flags::GEOMETRY;
            }
        }

        // Update IndexedFaceSet index array
        if let Some(entry) = graph.get_mut(self.ifs_node) {
            if let rc3d_scene::node_data::NodeData::IndexedFaceSet(ifs) = &mut entry.data {
                ifs.coord_index = triangle_indices_to_coord_index(&ts.indices);
                entry.dirty_flags |= rc3d_scene::node_entry::dirty_flags::GEOMETRY;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// File import conveniences
// ══════════════════════════════════════════════════════════════════════════════

/// Import an STL file into a `SceneGraph` (thin wrapper over `rc3d_io::stl`).
///
/// Usage:
/// ```ignore
/// let graph = load_stl("model.stl")?;
/// let mut scene = Scene::new();
/// scene.merge(graph);
/// ```
pub fn load_stl_file(path: &std::path::Path) -> Result<SceneGraph, String> {
    rc3d_io::stl::parse_stl_file(path).map_err(|e| format!("STL parse error: {e}"))
}

/// Import STL from bytes.
pub fn load_stl_bytes(data: &[u8]) -> Result<SceneGraph, String> {
    rc3d_io::stl::parse_stl(data).map_err(|e| format!("STL parse error: {e}"))
}

/// Import an OBJ file into a `SceneGraph`.
pub fn load_obj_file(path: &std::path::Path) -> Result<SceneGraph, String> {
    rc3d_io::obj::parse_obj_file(path).map_err(|e| format!("OBJ parse error: {e}"))
}

/// Import OBJ from text.
pub fn load_obj_str(text: &str) -> Result<SceneGraph, String> {
    rc3d_io::obj::parse_obj(text).map_err(|e| format!("OBJ parse error: {e}"))
}

/// Import a glTF/GLB file into a `SceneGraph`.
pub fn load_gltf_file(path: &std::path::Path) -> Result<SceneGraph, String> {
    rc3d_io::gltf::parse_gltf_file(path).map_err(|e| format!("glTF parse error: {e}"))
}

/// Import an FBX file into a `SceneGraph`.
pub fn load_fbx_file(path: &std::path::Path) -> Result<SceneGraph, String> {
    rc3d_io::fbx::parse_fbx_file(path).map_err(|e| format!("FBX parse error: {e}"))
}

/// Ensure the scene graph has at least one camera and one directional light.
///
/// Prevents render-blocking failures when using `App::from_scene` with an incomplete scene:
/// - No camera → identity projection → all geometry outside NDC → invisible
/// - No light → zero illumination → black screen
pub fn ensure_camera_and_light(graph: &mut SceneGraph) {
    use rc3d_scene::node_data::*;

    let mut has_camera = false;
    let mut has_light = false;

    for &root in graph.roots() {
        check_camera_light_recursive(graph, root, &mut has_camera, &mut has_light);
        if has_camera && has_light {
            break;
        }
    }

    if !has_camera || !has_light {
        let root = graph.roots().first().copied().unwrap_or_else(|| {
            graph.add_root(NodeData::Separator(SeparatorNode))
        });

        if !has_camera {
            graph.add_child(
                root,
                NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                    Vec3::new(0.0, 2.0, 8.0),
                    Vec3::ZERO,
                    Vec3::Y,
                    std::f32::consts::FRAC_PI_4,
                    800.0 / 600.0,
                )),
            );
        }
        if !has_light {
            graph.add_child(
                root,
                NodeData::DirectionalLight(DirectionalLightNode {
                    direction: Vec3::new(-0.5, -1.0, -0.3).normalize(),
                    color: Vec3::ONE,
                    intensity: 1.0,
                    light_group: None,
                }),
            );
        }
    }
}

fn check_camera_light_recursive(
    graph: &SceneGraph,
    node: rc3d_core::NodeId,
    has_camera: &mut bool,
    has_light: &mut bool,
) {
    let Some(entry) = graph.get(node) else { return };
    match &entry.data {
        rc3d_scene::node_data::NodeData::PerspectiveCamera(_)
        | rc3d_scene::node_data::NodeData::OrthographicCamera(_) => *has_camera = true,
        rc3d_scene::node_data::NodeData::DirectionalLight(_)
        | rc3d_scene::node_data::NodeData::PointLight(_) => *has_light = true,
        _ => {}
    }
    if *has_camera && *has_light {
        return;
    }
    for &child in &entry.children {
        check_camera_light_recursive(graph, child, has_camera, has_light);
        if *has_camera && *has_light {
            return;
        }
    }
}

/// Convert flat triangle indices `[i0, i1, i2, i3, i4, i5, ...]` into
/// Open Inventor coord_index format `[i0, i1, i2, -1, i3, i4, i5, -1, ...]`.
pub fn triangle_indices_to_coord_index(tri_indices: &[u32]) -> Vec<i32> {
    let tri_count = tri_indices.len() / 3;
    let mut coord_index = Vec::with_capacity(tri_count * 4);
    for tri in tri_indices.chunks_exact(3) {
        coord_index.push(tri[0] as i32);
        coord_index.push(tri[1] as i32);
        coord_index.push(tri[2] as i32);
        coord_index.push(-1);
    }
    coord_index
}
