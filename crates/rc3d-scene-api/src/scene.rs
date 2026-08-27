//! Scene container and NodeHandle.

use rc3d_core::{DisplayMode, EdgeStyle, FillStyle, NodeId, VisualStyle};
use rc3d_scene::node_data::{
    FontNode, FontStyle, NodeData, RotationAxis, RotationNode, RotationXYZNode, SeparatorNode,
    SpriteNode, TransformNode,
};
use rc3d_scene::SceneGraph;

use crate::camera::{CubeCamera, OrthographicCamera, PerspectiveCamera};
use crate::light::{DirectionalLight, HemisphereLight, LightProbe, PointLight};
use crate::shape::Shape;

/// Opaque handle to a node in the scene graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeHandle(pub(crate) NodeId);

impl NodeHandle {
    pub fn id(&self) -> NodeId {
        self.0
    }

    pub(crate) fn from_id(id: NodeId) -> Self {
        Self(id)
    }
}

/// High-level scene container.
///
/// Hides the internal `SceneGraph` and `NodeData` types. Users describe
/// *what* they want (shapes, lights, cameras) without touching the low-level API.
pub struct Scene {
    graph: SceneGraph,
    root: NodeId,
    camera_handle: Option<NodeHandle>,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    pub fn new() -> Self {
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        Self { graph, root, camera_handle: None }
    }

    /// Add a shape to the scene. Each shape is automatically wrapped in a
    /// `Separator` for state isolation, so transforms and materials don't leak.
    pub fn add(&mut self, shape: impl Shape + 'static) -> NodeHandle {
        let sep = self.graph.add_child(self.root, NodeData::Separator(SeparatorNode));

        // Optional transform
        if shape.has_transform() {
            self.graph.add_child(
                sep,
                NodeData::Transform(TransformNode {
                    translation: shape.shape_translation().copied().unwrap_or(rc3d_core::math::Vec3::ZERO),
                    rotation: shape.shape_rotation().copied().unwrap_or(rc3d_core::math::Mat4::IDENTITY),
                    scale: shape.shape_scale().copied().unwrap_or(rc3d_core::math::Vec3::ONE),
                    center: rc3d_core::math::Vec3::ZERO,
                }),
            );
        }

        // Optional material
        if let Some(mat) = shape.shape_material() {
            self.graph.add_child(sep, NodeData::Material(mat.to_node()));
        }

        // Geometry nodes (may add Coordinate3/Normal siblings before the shape node)
        shape.add_geometry_nodes(&mut self.graph, sep);

        NodeHandle(sep)
    }

    /// Set the active perspective camera.
    pub fn set_camera(&mut self, camera: PerspectiveCamera) -> NodeHandle {
        let node = self.graph.add_child(self.root, camera.to_node());
        self.camera_handle = Some(NodeHandle(node));
        NodeHandle(node)
    }

    /// Set the active orthographic camera.
    pub fn set_ortho_camera(&mut self, camera: OrthographicCamera) -> NodeHandle {
        let node = self.graph.add_child(self.root, camera.to_node());
        self.camera_handle = Some(NodeHandle(node));
        NodeHandle(node)
    }

    /// Add a local cubemap reflection probe (three.js CubeCamera analog).
    pub fn add_cube_camera(&mut self, camera: CubeCamera) -> NodeHandle {
        NodeHandle(self.graph.add_child(self.root, camera.to_node()))
    }

    /// Enable stereo presentation (three.js StereoEffect analog). Uses `camera()` as the base eye.
    pub fn add_stereo_camera(&mut self, stereo: crate::StereoCamera) -> NodeHandle {
        let base = self.camera_handle.map(|h| h.0).unwrap_or_default();
        NodeHandle(self.graph.add_child(self.root, stereo.to_node(base)))
    }

    /// Add a camera-facing textured quad (three.js Sprite analog).
    pub fn add_sprite(&mut self, sprite: SpriteNode) -> NodeHandle {
        NodeHandle(self.graph.add_child(self.root, NodeData::Sprite(sprite)))
    }

    /// Add a directional light to the scene.
    pub fn add_light(&mut self, light: DirectionalLight) -> NodeHandle {
        NodeHandle(self.graph.add_child(self.root, light.to_node()))
    }

    /// Add a point light to the scene.
    pub fn add_point_light(&mut self, light: PointLight) -> NodeHandle {
        NodeHandle(self.graph.add_child(self.root, light.to_node()))
    }

    /// Add a hemisphere (sky/ground) ambient light.
    pub fn add_hemisphere_light(&mut self, light: HemisphereLight) -> NodeHandle {
        NodeHandle(self.graph.add_child(self.root, light.to_node()))
    }

    /// Add an L2 spherical-harmonic irradiance probe (three.js LightProbe analog).
    pub fn add_light_probe(&mut self, probe: LightProbe) -> NodeHandle {
        NodeHandle(self.graph.add_child(self.root, probe.to_node()))
    }

    /// Add an axis-angle rotation property node (Coin3D SoRotation analog).
    pub fn add_rotation(&mut self, axis: rc3d_core::math::Vec3, angle: f32) -> NodeHandle {
        NodeHandle(
            self.graph
                .add_child(self.root, NodeData::Rotation(RotationNode::from_axis_angle(axis, angle))),
        )
    }

    /// Add a cardinal-axis rotation property node (Coin3D SoRotationXYZ analog).
    pub fn add_rotation_xyz(&mut self, axis: RotationAxis, angle: f32) -> NodeHandle {
        NodeHandle(
            self.graph
                .add_child(self.root, NodeData::RotationXYZ(RotationXYZNode { axis, angle })),
        )
    }

    /// Add a font property node (Coin3D SoFont analog) for subsequent Text2 / Text3.
    pub fn add_font(&mut self, name: impl Into<String>, size: f32, style: FontStyle) -> NodeHandle {
        NodeHandle(self.graph.add_child(
            self.root,
            NodeData::Font(FontNode {
                name: name.into(),
                size,
                style,
            }),
        ))
    }

    /// Add a group of shapes (explicit Separator boundary).
    pub fn add_group(&mut self, group: &crate::group::Group) -> NodeHandle {
        group.add_to_graph(&mut self.graph, self.root)
    }

    /// Merge an external `SceneGraph` (e.g. from `rc3d-io` STL parser) into this scene.
    ///
    /// All root nodes and their subtrees are copied under this scene's root.
    pub fn merge(&mut self, other: rc3d_scene::SceneGraph) -> NodeHandle {
        let group = self.graph.add_child(self.root,
            rc3d_scene::node_data::NodeData::Separator(rc3d_scene::node_data::SeparatorNode));
        for &root in other.roots() {
            copy_subtree(&other, &mut self.graph, root, group);
        }
        NodeHandle(group)
    }

    /// Get the camera handle if one was set.
    pub fn camera(&self) -> Option<NodeHandle> {
        self.camera_handle
    }

    /// Create a type-safe query over the scene graph.
    pub fn query(&self) -> crate::query::Query<'_> {
        crate::query::Query::new(&self.graph)
    }

    /// Finalize and return the internal `SceneGraph` for rendering.
    pub fn build(self) -> SceneGraph {
        self.graph
    }

    /// Coin3D subtree `SoDrawStyle`: set `mode` on this handle (typically a Separator).
    pub fn set_subtree_display_mode(&mut self, handle: NodeHandle, mode: DisplayMode) {
        self.graph.set_display_mode(handle.0, mode);
    }

    pub fn set_subtree_fill_style(&mut self, handle: NodeHandle, fill: FillStyle) {
        self.graph.set_fill_style(handle.0, fill);
    }

    pub fn set_subtree_edge_style(&mut self, handle: NodeHandle, edges: EdgeStyle) {
        self.graph.set_edge_style(handle.0, edges);
    }

    pub fn set_subtree_visual_style(&mut self, handle: NodeHandle, style: &VisualStyle) {
        self.graph.apply_visual_style(handle.0, style);
    }

    /// Resolve PMI names on every `AnnotationSet` and stamp unbound points.
    pub fn bind_pmi(&mut self) -> usize {
        self.graph.bind_pmi()
    }

    pub fn set_face_tint(&mut self, handle: NodeHandle, face: u32, color: [f32; 4]) {
        self.graph.set_face_tint(handle.0, face, color);
    }

    pub fn set_edge_tint(&mut self, handle: NodeHandle, triangle: u32, edge: u8, color: [f32; 4]) {
        self.graph.set_edge_tint(handle.0, triangle, edge, color);
    }

    /// Borrow the internal graph (for read-only queries).
    pub fn graph(&self) -> &SceneGraph {
        &self.graph
    }

    /// Mutably borrow the internal graph (for engine integration).
    pub fn graph_mut(&mut self) -> &mut SceneGraph {
        &mut self.graph
    }
}

/// Deep-copy a subtree from src to dst.
fn copy_subtree(
    src: &SceneGraph,
    dst: &mut SceneGraph,
    src_id: NodeId,
    dst_parent: NodeId,
) -> NodeId {
    let entry = src.get(src_id).expect("node exists");
    let new_id = dst.add_child(dst_parent, entry.data.clone());
    if let Some(dst_entry) = dst.get_mut(new_id) {
        dst_entry.display_mode = entry.display_mode;
        dst_entry.fill_style = entry.fill_style;
        dst_entry.edge_style = entry.edge_style;
        dst_entry.name = entry.name.clone();
    }
    if let Some(children) = src.children(src_id) {
        for &child in children {
            copy_subtree(src, dst, child, new_id);
        }
    }
    new_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Cube;
    use crate::material::Material;

    #[test]
    fn test_empty_scene() {
        let scene = Scene::new();
        let graph = scene.build();
        assert_eq!(graph.roots().len(), 1);
    }

    #[test]
    fn test_add_cube_creates_subtree() {
        let mut scene = Scene::new();
        let handle = scene.add(Cube::default());
        let graph = scene.build();

        // The cube's Separator should have 2 children: cube node (+ possibly
        // transform/material if defaults trigger)
        assert!(graph.get(handle.id()).is_some());
        assert!(!graph.children(handle.id()).unwrap().is_empty());
    }

    #[test]
    fn test_add_cube_with_material() {
        let mut scene = Scene::new();
        let mat = Material::pbr().base_color(1.0, 0.0, 0.0);
        scene.add(Cube::default().material(mat));
        let graph = scene.build();
        // SceneGraph created successfully with material
        assert!(graph.node_count() > 0);
    }

    #[test]
    fn test_set_camera() {
        let mut scene = Scene::new();
        let cam = PerspectiveCamera::look_at(
            rc3d_core::math::Vec3::new(2.0, 2.0, 4.0),
            rc3d_core::math::Vec3::ZERO,
            rc3d_core::math::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        );
        let h = scene.set_camera(cam);
        assert!(scene.camera().is_some());
        assert_eq!(scene.camera().unwrap(), h);
    }

    #[test]
    fn test_add_light() {
        let mut scene = Scene::new();
        let light = DirectionalLight::sun(
            rc3d_core::math::Vec3::new(-1.0, -1.0, -1.0),
            1.0,
        );
        scene.add_light(light);
        let graph = scene.build();
        assert!(graph.node_count() > 1);
    }
}
