//! Central `NodeData` enum and serialization.
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::custom_node::CustomNodeData;
use crate::node_handler::NodeHandler;

use super::advanced::*;
use super::cameras::*;
use super::control::*;
use super::effects::*;
use super::grouping::*;
use super::lights::*;
use super::properties::*;
use super::shapes::*;

/// Central node type enum.
#[derive(Debug)]
pub enum NodeData {
    // Grouping
    Separator(SeparatorNode),
    Group(GroupNode),
            Billboard(BillboardNode),
    // Properties
    Transform(TransformNode),
    Coordinate3(Coordinate3Node),
    TextureCoordinate2(TextureCoordinate2Node),
    Normal(NormalNode),
    Material(MaterialNode),
    // Shapes
    Triangle(TriangleNode),
    Cube(CubeNode),
    Sphere(SphereNode),
    Cone(ConeNode),
    Cylinder(CylinderNode),
    /// Shape: torus (donut).
    Torus(TorusNode),
    IndexedFaceSet(IndexedFaceSetNode),
    IndexedLineSet(IndexedLineSetNode),
    SkinnedMesh(SkinnedMeshNode),
    MorphTarget(MorphTargetNode),
    // Cameras
    StereoCamera(StereoCameraNode),
    PerspectiveCamera(PerspectiveCameraNode),
    OrthographicCamera(OrthographicCameraNode),
    // Lights
    DirectionalLight(DirectionalLightNode),
    PointLight(PointLightNode),
    SpotLight(SpotLightNode),
    /// Area light: rectangle or disc emitter with realistic soft shadows.
    AreaLight(AreaLightNode),
    /// User-defined behavior via [`NodeHandler`] (traversal / future collect hooks).
    HandlerNode(Arc<dyn NodeHandler>),
    /// Event routing callback (Coin3D SoEventCallback pattern).
    EventCallback(EventCallbackNode),
    /// Pick style: controls node pickability (Coin3D SoPickStyle pattern).
    PickStyle(PickStyleNode),
    /// Level-of-detail switch (Coin3D SoLOD pattern).
    Lod(LodNode),
    Switch(SwitchNode),
    MultipleCopy(MultipleCopyNode),
    /// Section/cutting plane (Coin3D SoClipPlane pattern).
    SectionPlane(SectionPlaneNode),
    /// Screen-space 2D text label (Coin3D SoText2 pattern).
    Text2(Text2Node),
    /// World-space 3D text label (Coin3D SoText3 pattern).
    Text3(Text3Node),
    Measurement(MeasurementNode),
    Markup(MarkupNode),
    AnnotationSet(AnnotationSetNode),
    /// External file reference (SoFile/SoWWWInline equivalent).
    File(FileNode),
    Environment(EnvironmentNode),
    ShapeHints(ShapeHintsNode),
    Annotation(AnnotationNode),
    ResetTransform(ResetTransformNode),
    Texture2Transform(Texture2TransformNode),
    MaterialBinding(MaterialBindingNode),
    /// Screen-space projected texture overlay.
    ExplodedView(ExplodedViewNode),
    RayTracing(RayTracingNode),
    Volume(VolumeNode),
    PointCloud(PointCloudNode),
    ReflectionPlane(ReflectionPlaneNode),
    Decal(DecalNode),
    /// User-defined node type registered via `NodeTypeRegistry`.
    Custom(u16, Box<dyn CustomNodeData>),
}

impl Clone for NodeData {
    fn clone(&self) -> Self {
        match self {
            NodeData::Separator(v) => NodeData::Separator(v.clone()),
            NodeData::Group(v) => NodeData::Group(v.clone()),
            NodeData::Billboard(v) => NodeData::Billboard(v.clone()),
            NodeData::Transform(v) => NodeData::Transform(v.clone()),
            NodeData::Coordinate3(v) => NodeData::Coordinate3(v.clone()),
            NodeData::TextureCoordinate2(v) => NodeData::TextureCoordinate2(v.clone()),
            NodeData::Normal(v) => NodeData::Normal(v.clone()),
            NodeData::Material(v) => NodeData::Material(v.clone()),
            NodeData::Triangle(v) => NodeData::Triangle(v.clone()),
            NodeData::Cube(v) => NodeData::Cube(v.clone()),
            NodeData::Sphere(v) => NodeData::Sphere(v.clone()),
            NodeData::Cone(v) => NodeData::Cone(v.clone()),
            NodeData::Cylinder(v) => NodeData::Cylinder(v.clone()),
            NodeData::Torus(v) => NodeData::Torus(v.clone()),
            NodeData::IndexedFaceSet(v) => NodeData::IndexedFaceSet(v.clone()),
            NodeData::IndexedLineSet(v) => NodeData::IndexedLineSet(v.clone()),
            NodeData::SkinnedMesh(v) => NodeData::SkinnedMesh(v.clone()),
            NodeData::MorphTarget(v) => NodeData::MorphTarget(v.clone()),
            NodeData::StereoCamera(v) => NodeData::StereoCamera(v.clone()),
            NodeData::PerspectiveCamera(v) => NodeData::PerspectiveCamera(v.clone()),
            NodeData::OrthographicCamera(v) => NodeData::OrthographicCamera(v.clone()),
            NodeData::DirectionalLight(v) => NodeData::DirectionalLight(v.clone()),
            NodeData::PointLight(v) => NodeData::PointLight(v.clone()),
            NodeData::SpotLight(v) => NodeData::SpotLight(v.clone()),
            NodeData::AreaLight(v) => NodeData::AreaLight(v.clone()),
            NodeData::HandlerNode(h) => NodeData::HandlerNode(Arc::clone(h)),
            NodeData::EventCallback(v) => NodeData::EventCallback(v.clone()),
            NodeData::PickStyle(v) => NodeData::PickStyle(v.clone()),
            NodeData::Lod(v) => NodeData::Lod(v.clone()),
            NodeData::Switch(v) => NodeData::Switch(v.clone()),
            NodeData::MultipleCopy(v) => NodeData::MultipleCopy(v.clone()),
            NodeData::SectionPlane(v) => NodeData::SectionPlane(v.clone()),
            NodeData::Text2(v) => NodeData::Text2(v.clone()),
            NodeData::Text3(v) => NodeData::Text3(v.clone()),
            NodeData::Measurement(v) => NodeData::Measurement(v.clone()),
            NodeData::Markup(v) => NodeData::Markup(v.clone()),
            NodeData::AnnotationSet(v) => NodeData::AnnotationSet(v.clone()),
            NodeData::Custom(id, d) => NodeData::Custom(*id, d.clone_box()),
            NodeData::Environment(v) => NodeData::Environment(v.clone()),
            NodeData::ShapeHints(v) => NodeData::ShapeHints(v.clone()),
            NodeData::Annotation(v) => NodeData::Annotation(v.clone()),
            NodeData::ResetTransform(v) => NodeData::ResetTransform(v.clone()),
            NodeData::Texture2Transform(v) => NodeData::Texture2Transform(v.clone()),
            NodeData::MaterialBinding(v) => NodeData::MaterialBinding(v.clone()),
            NodeData::ExplodedView(v) => NodeData::ExplodedView(v.clone()),
            NodeData::RayTracing(v) => NodeData::RayTracing(v.clone()),
            NodeData::Volume(v) => NodeData::Volume(v.clone()),
            NodeData::PointCloud(v) => NodeData::PointCloud(v.clone()),
            NodeData::ReflectionPlane(v) => NodeData::ReflectionPlane(v.clone()),
            NodeData::Decal(v) => NodeData::Decal(v.clone()),
            NodeData::File(v) => NodeData::File(v.clone()),
        }
    }
}

/// Describes a named field on a node type.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FieldDescriptor {
    pub name: &'static str,
    pub field_index: u16,
}

impl NodeData {
    /// Returns the fields exposed by this node type.
    pub fn field_descriptors(&self) -> Vec<FieldDescriptor> {
        match self {
            NodeData::Transform(_) => vec![
                FieldDescriptor { name: "translation", field_index: 0 },
                FieldDescriptor { name: "rotation", field_index: 1 },
                FieldDescriptor { name: "scale", field_index: 2 },
                FieldDescriptor { name: "center", field_index: 3 },
            ],
            NodeData::Material(_) => vec![
                FieldDescriptor { name: "diffuseColor", field_index: 0 },
                FieldDescriptor { name: "specularColor", field_index: 1 },
                FieldDescriptor { name: "shininess", field_index: 2 },
                FieldDescriptor { name: "opacity", field_index: 3 },
            ],
            NodeData::DirectionalLight(_) => vec![
                FieldDescriptor { name: "direction", field_index: 0 },
                FieldDescriptor { name: "color", field_index: 1 },
                FieldDescriptor { name: "intensity", field_index: 2 },
            ],
            NodeData::PointLight(_) => vec![
                FieldDescriptor { name: "location", field_index: 0 },
                FieldDescriptor { name: "color", field_index: 1 },
                FieldDescriptor { name: "intensity", field_index: 2 },
                FieldDescriptor { name: "cutoff_distance", field_index: 3 },
            ],
            NodeData::SpotLight(_) => vec![
                FieldDescriptor { name: "location", field_index: 0 },
                FieldDescriptor { name: "direction", field_index: 1 },
                FieldDescriptor { name: "color", field_index: 2 },
                FieldDescriptor { name: "intensity", field_index: 3 },
                FieldDescriptor { name: "cut_off_angle", field_index: 4 },
                FieldDescriptor { name: "drop_off_rate", field_index: 5 },
            ],
            NodeData::AreaLight(_) => vec![
                FieldDescriptor { name: "color", field_index: 0 },
                FieldDescriptor { name: "intensity", field_index: 1 },
                FieldDescriptor { name: "width", field_index: 2 },
                FieldDescriptor { name: "height", field_index: 3 },
            ],
            NodeData::PerspectiveCamera(_) => vec![
                FieldDescriptor { name: "fov", field_index: 0 },
                FieldDescriptor { name: "near", field_index: 1 },
                FieldDescriptor { name: "far", field_index: 2 },
                FieldDescriptor { name: "reverse_depth", field_index: 3 },
            ],
            NodeData::StereoCamera(_) => vec![
                FieldDescriptor { name: "interocular", field_index: 0 },
                FieldDescriptor { name: "convergence", field_index: 1 },
            ],
            NodeData::OrthographicCamera(_) => vec![
                FieldDescriptor { name: "height", field_index: 0 },
                FieldDescriptor { name: "near", field_index: 1 },
                FieldDescriptor { name: "far", field_index: 2 },
                FieldDescriptor { name: "reverse_depth", field_index: 3 },
            ],
            NodeData::SectionPlane(_) => vec![
                FieldDescriptor { name: "plane", field_index: 0 },
                FieldDescriptor { name: "enabled", field_index: 1 },
                FieldDescriptor { name: "cap_color", field_index: 2 },
                FieldDescriptor { name: "cap_enabled", field_index: 3 },
            ],
            NodeData::Lod(_) => vec![
                FieldDescriptor { name: "current_level", field_index: 0 },
            ],
            NodeData::Switch(_) => vec![
                FieldDescriptor { name: "which_child", field_index: 0 },
            ],
            NodeData::Text2(_) => vec![
                FieldDescriptor { name: "string", field_index: 0 },
                FieldDescriptor { name: "position", field_index: 1 },
                FieldDescriptor { name: "size", field_index: 2 },
                FieldDescriptor { name: "color", field_index: 3 },
            ],
            NodeData::Text3(_) => vec![
                FieldDescriptor { name: "string", field_index: 0 },
                FieldDescriptor { name: "position", field_index: 1 },
                FieldDescriptor { name: "size", field_index: 2 },
                FieldDescriptor { name: "color", field_index: 3 },
            ],
            NodeData::EventCallback(_) => vec![
                FieldDescriptor { name: "enabled", field_index: 0 },
            ],
            NodeData::PickStyle(_) => vec![
                FieldDescriptor { name: "pickable", field_index: 0 },
            ],
            NodeData::Markup(_) => vec![
                FieldDescriptor { name: "visible", field_index: 0 },
                FieldDescriptor { name: "layer_name", field_index: 1 },
            ],
            NodeData::Measurement(_) => vec![
                FieldDescriptor { name: "value", field_index: 0 },
                FieldDescriptor { name: "label", field_index: 1 },
                FieldDescriptor { name: "color", field_index: 2 },
            ],
            // Nodes with no runtime fields
            NodeData::Separator(_)
            | NodeData::Group(_)
            | NodeData::Billboard(_)
            | NodeData::Coordinate3(_)
            | NodeData::TextureCoordinate2(_)
            | NodeData::Normal(_)
            | NodeData::Triangle(_)
            | NodeData::IndexedLineSet(_)
            | NodeData::File(_)
            | NodeData::Environment(_)
            | NodeData::ShapeHints(_)
            | NodeData::Annotation(_)
            | NodeData::AnnotationSet(_)
            | NodeData::ResetTransform(_)
            | NodeData::Texture2Transform(_)
            | NodeData::MaterialBinding(_)
            | NodeData::RayTracing(_)
            | NodeData::Volume(_)
            | NodeData::PointCloud(_)
            | NodeData::ExplodedView(_)
            | NodeData::ReflectionPlane(_)
            | NodeData::Cube(_)
            | NodeData::Sphere(_)
            | NodeData::Cone(_)
            | NodeData::Cylinder(_)
            | NodeData::Torus(_)
            | NodeData::IndexedFaceSet(_)
            | NodeData::SkinnedMesh(_)
            | NodeData::MorphTarget(_)
            | NodeData::HandlerNode(_)
            | NodeData::MultipleCopy(_) => vec![],
            NodeData::Custom(_, d) => d.field_descriptors(),
            NodeData::Decal(_) => vec![FieldDescriptor { name: "opacity", field_index: 0 }],
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            NodeData::Separator(_) => "Separator",
            NodeData::Group(_) => "Group",
            NodeData::Billboard(_) => "Billboard",
            NodeData::Transform(_) => "Transform",
            NodeData::Coordinate3(_) => "Coordinate3",
            NodeData::TextureCoordinate2(_) => "TextureCoordinate2",
            NodeData::Normal(_) => "Normal",
            NodeData::Material(_) => "Material",
            NodeData::Triangle(_) => "Triangle",
            NodeData::Cube(_) => "Cube",
            NodeData::Sphere(_) => "Sphere",
            NodeData::Cone(_) => "Cone",
            NodeData::Cylinder(_) => "Cylinder",
            NodeData::Torus(_) => "Torus",
            NodeData::IndexedFaceSet(_) => "IndexedFaceSet",
            NodeData::IndexedLineSet(_) => "IndexedLineSet",
            NodeData::SkinnedMesh(_) => "SkinnedMesh",
            NodeData::MorphTarget(_) => "MorphTarget",
            NodeData::StereoCamera(_) => "StereoCamera",
            NodeData::PerspectiveCamera(_) => "PerspectiveCamera",
            NodeData::OrthographicCamera(_) => "OrthographicCamera",
            NodeData::DirectionalLight(_) => "DirectionalLight",
            NodeData::PointLight(_) => "PointLight",
            NodeData::SpotLight(_) => "SpotLight",
            NodeData::AreaLight(_) => "AreaLight",
            NodeData::HandlerNode(h) => h.handler_name(),
            NodeData::EventCallback(_) => "EventCallback",
            NodeData::PickStyle(_) => "PickStyle",
            NodeData::Lod(_) => "Lod",
            NodeData::Switch(_) => "Switch",
            NodeData::MultipleCopy(_) => "MultipleCopy",
            NodeData::SectionPlane(_) => "SectionPlane",
            NodeData::Text2(_) => "Text2",
            NodeData::Text3(_) => "Text3",
            NodeData::Measurement(_) => "Measurement",
            NodeData::Markup(_) => "Markup",
            NodeData::Custom(_, d) => d.type_name(),
            NodeData::Environment(_) => "Environment",
            NodeData::ShapeHints(_) => "ShapeHints",
            NodeData::Annotation(_) => "Annotation",
            NodeData::AnnotationSet(_) => "AnnotationSet",
            NodeData::ResetTransform(_) => "ResetTransform",
            NodeData::Texture2Transform(_) => "Texture2Transform",
            NodeData::MaterialBinding(_) => "MaterialBinding",
            NodeData::ExplodedView(_) => "ExplodedView",
            NodeData::RayTracing(_) => "RayTracing",
            NodeData::Volume(_) => "Volume",
            NodeData::PointCloud(_) => "PointCloud",
            NodeData::ReflectionPlane(_) => "ReflectionPlane",
            NodeData::Decal(_) => "Decal",
            NodeData::File(_) => "File",
        }
    }
}

// Manual Serialize/Deserialize to handle HandlerNode (Arc<dyn NodeHandler>).
impl Serialize for NodeData {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            NodeData::Separator(v) => s.serialize_newtype_variant("NodeData", 0, "Separator", v),
            NodeData::Group(v) => s.serialize_newtype_variant("NodeData", 1, "Group", v),
            NodeData::Billboard(v) => s.serialize_newtype_variant("NodeData", 48, "Billboard", v),
            NodeData::Transform(v) => s.serialize_newtype_variant("NodeData", 2, "Transform", v),
            NodeData::Coordinate3(v) => s.serialize_newtype_variant("NodeData", 3, "Coordinate3", v),
            NodeData::TextureCoordinate2(v) => s.serialize_newtype_variant("NodeData", 4, "TextureCoordinate2", v),
            NodeData::Normal(v) => s.serialize_newtype_variant("NodeData", 5, "Normal", v),
            NodeData::Material(v) => s.serialize_newtype_variant("NodeData", 6, "Material", v),
            NodeData::Triangle(v) => s.serialize_newtype_variant("NodeData", 7, "Triangle", v),
            NodeData::Cube(v) => s.serialize_newtype_variant("NodeData", 8, "Cube", v),
            NodeData::Sphere(v) => s.serialize_newtype_variant("NodeData", 9, "Sphere", v),
            NodeData::Cone(v) => s.serialize_newtype_variant("NodeData", 10, "Cone", v),
            NodeData::Cylinder(v) => s.serialize_newtype_variant("NodeData", 11, "Cylinder", v),
            NodeData::Torus(v) => s.serialize_newtype_variant("NodeData", 50, "Torus", v),
            NodeData::IndexedFaceSet(v) => s.serialize_newtype_variant("NodeData", 12, "IndexedFaceSet", v),
            NodeData::IndexedLineSet(v) => s.serialize_newtype_variant("NodeData", 34, "IndexedLineSet", v),
            NodeData::SkinnedMesh(v) => s.serialize_newtype_variant("NodeData", 13, "SkinnedMesh", v),
            NodeData::MorphTarget(v) => s.serialize_newtype_variant("NodeData", 14, "MorphTarget", v),
            NodeData::StereoCamera(v) => s.serialize_newtype_variant("NodeData", 44, "StereoCamera", v),
            NodeData::RayTracing(v) => s.serialize_newtype_variant("NodeData", 45, "RayTracing", v),
            NodeData::Volume(v) => s.serialize_newtype_variant("NodeData", 46, "Volume", v),
            NodeData::PointCloud(v) => s.serialize_newtype_variant("NodeData", 47, "PointCloud", v),
            NodeData::PerspectiveCamera(v) => s.serialize_newtype_variant("NodeData", 15, "PerspectiveCamera", v),
            NodeData::OrthographicCamera(v) => s.serialize_newtype_variant("NodeData", 16, "OrthographicCamera", v),
            NodeData::DirectionalLight(v) => s.serialize_newtype_variant("NodeData", 17, "DirectionalLight", v),
            NodeData::PointLight(v) => s.serialize_newtype_variant("NodeData", 18, "PointLight", v),
            NodeData::SpotLight(v) => s.serialize_newtype_variant("NodeData", 19, "SpotLight", v),
            NodeData::AreaLight(v) => s.serialize_newtype_variant("NodeData", 32, "AreaLight", v),
            NodeData::EventCallback(v) => s.serialize_newtype_variant("NodeData", 20, "EventCallback", v),
            NodeData::PickStyle(v) => s.serialize_newtype_variant("NodeData", 21, "PickStyle", v),
            NodeData::Lod(v) => s.serialize_newtype_variant("NodeData", 22, "Lod", v),
            NodeData::Switch(v) => s.serialize_newtype_variant("NodeData", 23, "Switch", v),
            NodeData::MultipleCopy(v) => s.serialize_newtype_variant("NodeData", 24, "MultipleCopy", v),
            NodeData::SectionPlane(v) => s.serialize_newtype_variant("NodeData", 25, "SectionPlane", v),
            NodeData::Text2(v) => s.serialize_newtype_variant("NodeData", 26, "Text2", v),
            NodeData::Text3(v) => s.serialize_newtype_variant("NodeData", 27, "Text3", v),
            NodeData::Measurement(v) => s.serialize_newtype_variant("NodeData", 28, "Measurement", v),
            NodeData::Markup(v) => s.serialize_newtype_variant("NodeData", 29, "Markup", v),
            NodeData::AnnotationSet(v) => s.serialize_newtype_variant("NodeData", 49, "AnnotationSet", v),
            NodeData::HandlerNode(h) => s.serialize_newtype_variant("NodeData", 30, "HandlerNode", &h.handler_name()),
            NodeData::Environment(v) => s.serialize_newtype_variant("NodeData", 36, "Environment", v),
            NodeData::ShapeHints(v) => s.serialize_newtype_variant("NodeData", 37, "ShapeHints", v),
            NodeData::Annotation(v) => s.serialize_newtype_variant("NodeData", 38, "Annotation", v),
            NodeData::ResetTransform(v) => s.serialize_newtype_variant("NodeData", 39, "ResetTransform", v),
            NodeData::Texture2Transform(v) => s.serialize_newtype_variant("NodeData", 40, "Texture2Transform", v),
            NodeData::MaterialBinding(v) => s.serialize_newtype_variant("NodeData", 41, "MaterialBinding", v),
            NodeData::ExplodedView(v) => s.serialize_newtype_variant("NodeData", 42, "ExplodedView", v),
            NodeData::ReflectionPlane(v) => s.serialize_newtype_variant("NodeData", 43, "ReflectionPlane", v),
            NodeData::Decal(v) => s.serialize_newtype_variant("NodeData", 33, "Decal", v),
            NodeData::File(v) => s.serialize_newtype_variant("NodeData", 35, "File", v),
            NodeData::Custom(type_id, d) => {
                let payload = (type_id, d.serialize_custom());
                s.serialize_newtype_variant("NodeData", 31, "Custom", &payload)
            }
        }
    }
}

impl<'de> Deserialize<'de> for NodeData {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // Use a helper enum for deserialization, then map HandlerNode -> default
        #[derive(Deserialize)]
        enum NodeDataHelper {
            Separator(SeparatorNode),
            Group(GroupNode),
            Billboard(BillboardNode),
            Transform(TransformNode),
            Coordinate3(Coordinate3Node),
            TextureCoordinate2(TextureCoordinate2Node),
            Normal(NormalNode),
            Material(MaterialNode),
            Triangle(TriangleNode),
            Cube(CubeNode),
            Sphere(SphereNode),
            Cone(ConeNode),
            Cylinder(CylinderNode),
            Torus(TorusNode),
            IndexedFaceSet(IndexedFaceSetNode),
            IndexedLineSet(IndexedLineSetNode),
            SkinnedMesh(SkinnedMeshNode),
            MorphTarget(MorphTargetNode),
            StereoCamera(StereoCameraNode),
            PerspectiveCamera(PerspectiveCameraNode),
            OrthographicCamera(OrthographicCameraNode),
            DirectionalLight(DirectionalLightNode),
            PointLight(PointLightNode),
            SpotLight(SpotLightNode),
            AreaLight(AreaLightNode),
            #[serde(rename = "HandlerNode")]
            Handler(String),
            Custom((u16, String)),
            Environment(EnvironmentNode),
            ShapeHints(ShapeHintsNode),
            Annotation(AnnotationNode),
            ResetTransform(ResetTransformNode),
            Texture2Transform(Texture2TransformNode),
            MaterialBinding(MaterialBindingNode),
            ExplodedView(ExplodedViewNode),
            RayTracing(RayTracingNode),
            Volume(VolumeNode),
            PointCloud(PointCloudNode),
            ReflectionPlane(ReflectionPlaneNode),
            Decal(DecalNode),
            File(FileNode),
            EventCallback(EventCallbackNode),
            PickStyle(PickStyleNode),
            Lod(LodNode),
            Switch(SwitchNode),
            MultipleCopy(MultipleCopyNode),
            SectionPlane(SectionPlaneNode),
            Text2(Text2Node),
            Text3(Text3Node),
            Measurement(MeasurementNode),
            Markup(MarkupNode),
            AnnotationSet(AnnotationSetNode),
        }
        match NodeDataHelper::deserialize(d)? {
            NodeDataHelper::Separator(v) => Ok(NodeData::Separator(v)),
            NodeDataHelper::Group(v) => Ok(NodeData::Group(v)),
            NodeDataHelper::Billboard(v) => Ok(NodeData::Billboard(v)),
            NodeDataHelper::Transform(v) => Ok(NodeData::Transform(v)),
            NodeDataHelper::Coordinate3(v) => Ok(NodeData::Coordinate3(v)),
            NodeDataHelper::TextureCoordinate2(v) => Ok(NodeData::TextureCoordinate2(v)),
            NodeDataHelper::Normal(v) => Ok(NodeData::Normal(v)),
            NodeDataHelper::Material(v) => Ok(NodeData::Material(v)),
            NodeDataHelper::Triangle(v) => Ok(NodeData::Triangle(v)),
            NodeDataHelper::Cube(v) => Ok(NodeData::Cube(v)),
            NodeDataHelper::Sphere(v) => Ok(NodeData::Sphere(v)),
            NodeDataHelper::Cone(v) => Ok(NodeData::Cone(v)),
            NodeDataHelper::Cylinder(v) => Ok(NodeData::Cylinder(v)),
            NodeDataHelper::Torus(v) => Ok(NodeData::Torus(v)),
            NodeDataHelper::IndexedFaceSet(v) => Ok(NodeData::IndexedFaceSet(v)),
            NodeDataHelper::IndexedLineSet(v) => Ok(NodeData::IndexedLineSet(v)),
            NodeDataHelper::SkinnedMesh(v) => Ok(NodeData::SkinnedMesh(v)),
            NodeDataHelper::MorphTarget(v) => Ok(NodeData::MorphTarget(v)),
            NodeDataHelper::StereoCamera(v) => Ok(NodeData::StereoCamera(v)),
            NodeDataHelper::PerspectiveCamera(v) => Ok(NodeData::PerspectiveCamera(v)),
            NodeDataHelper::OrthographicCamera(v) => Ok(NodeData::OrthographicCamera(v)),
            NodeDataHelper::DirectionalLight(v) => Ok(NodeData::DirectionalLight(v)),
            NodeDataHelper::PointLight(v) => Ok(NodeData::PointLight(v)),
            NodeDataHelper::SpotLight(v) => Ok(NodeData::SpotLight(v)),
            NodeDataHelper::AreaLight(v) => Ok(NodeData::AreaLight(v)),
            NodeDataHelper::Handler(_name) => Ok(NodeData::HandlerNode(Arc::new(
                crate::node_handler::DummyHandler,
            ))),
            NodeDataHelper::Environment(v) => Ok(NodeData::Environment(v)),
            NodeDataHelper::ShapeHints(v) => Ok(NodeData::ShapeHints(v)),
            NodeDataHelper::Annotation(v) => Ok(NodeData::Annotation(v)),
            NodeDataHelper::ResetTransform(v) => Ok(NodeData::ResetTransform(v)),
            NodeDataHelper::Texture2Transform(v) => Ok(NodeData::Texture2Transform(v)),
            NodeDataHelper::MaterialBinding(v) => Ok(NodeData::MaterialBinding(v)),
            NodeDataHelper::ExplodedView(v) => Ok(NodeData::ExplodedView(v)),
            NodeDataHelper::RayTracing(v) => Ok(NodeData::RayTracing(v)),
            NodeDataHelper::Volume(v) => Ok(NodeData::Volume(v)),
            NodeDataHelper::PointCloud(v) => Ok(NodeData::PointCloud(v)),
            NodeDataHelper::ReflectionPlane(v) => Ok(NodeData::ReflectionPlane(v)),
            NodeDataHelper::Decal(v) => Ok(NodeData::Decal(v)),
            NodeDataHelper::File(v) => Ok(NodeData::File(v)),
            NodeDataHelper::Custom((_type_id, ref _data)) => {
                // Defer to registry for deserialization; fallback to DummyHandler
                Ok(NodeData::HandlerNode(Arc::new(
                    crate::node_handler::DummyHandler,
                )))
            }
            NodeDataHelper::EventCallback(v) => Ok(NodeData::EventCallback(v)),
            NodeDataHelper::PickStyle(v) => Ok(NodeData::PickStyle(v)),
            NodeDataHelper::Lod(v) => Ok(NodeData::Lod(v)),
            NodeDataHelper::Switch(v) => Ok(NodeData::Switch(v)),
            NodeDataHelper::MultipleCopy(v) => Ok(NodeData::MultipleCopy(v)),
            NodeDataHelper::SectionPlane(v) => Ok(NodeData::SectionPlane(v)),
            NodeDataHelper::Text2(v) => Ok(NodeData::Text2(v)),
            NodeDataHelper::Text3(v) => Ok(NodeData::Text3(v)),
            NodeDataHelper::Measurement(v) => Ok(NodeData::Measurement(v)),
            NodeDataHelper::Markup(v) => Ok(NodeData::Markup(v)),
            NodeDataHelper::AnnotationSet(v) => Ok(NodeData::AnnotationSet(v)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── field_descriptors completeness ──

    #[test]
    fn test_material_descriptors() {
        let d = NodeData::Material(MaterialNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"diffuseColor"));
        assert!(names.contains(&"opacity"));
        assert!(!d.is_empty());
    }

    #[test]
    fn test_directional_light_descriptors() {
        let d = NodeData::DirectionalLight(DirectionalLightNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"direction"));
        assert!(names.contains(&"color"));
        assert!(names.contains(&"intensity"));
    }

    #[test]
    fn test_spot_light_descriptors() {
        let d = NodeData::SpotLight(SpotLightNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"cut_off_angle"));
        assert!(names.contains(&"drop_off_rate"));
    }

    #[test]
    fn test_perspective_camera_descriptors() {
        let d = NodeData::PerspectiveCamera(PerspectiveCameraNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"fov"));
        assert!(names.contains(&"near"));
        assert!(names.contains(&"far"));
        assert!(names.contains(&"reverse_depth"));
    }

    #[test]
    fn test_section_plane_descriptors() {
        let d = NodeData::SectionPlane(SectionPlaneNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"plane"));
        assert!(names.contains(&"enabled"));
        assert!(names.contains(&"cap_color"));
        assert!(names.contains(&"cap_enabled"));
    }

    #[test]
    fn test_markup_descriptors() {
        let d = NodeData::Markup(MarkupNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"visible"));
        assert!(names.contains(&"layer_name"));
    }

    #[test]
    fn test_measurement_descriptors() {
        let d = NodeData::Measurement(MeasurementNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"value"));
        assert!(names.contains(&"label"));
        assert!(names.contains(&"color"));
    }

    #[test]
    fn test_shape_nodes_have_no_descriptors() {
        // Structural/geometry nodes should return empty field lists
        assert!(NodeData::Separator(SeparatorNode).field_descriptors().is_empty());
        assert!(NodeData::Group(GroupNode).field_descriptors().is_empty());
        assert!(NodeData::Cube(CubeNode::default()).field_descriptors().is_empty());
        assert!(NodeData::Cylinder(CylinderNode::default()).field_descriptors().is_empty());
        assert!(NodeData::Torus(TorusNode::default()).field_descriptors().is_empty());
        assert!(NodeData::IndexedFaceSet(IndexedFaceSetNode::default())
            .field_descriptors()
            .is_empty());
    }

    #[test]
    fn test_transform_descriptors() {
        let d = NodeData::Transform(TransformNode::default()).field_descriptors();
        let names: Vec<&str> = d.iter().map(|fd| fd.name).collect();
        assert!(names.contains(&"translation"));
        assert!(names.contains(&"rotation"));
        assert!(names.contains(&"scale"));
    }

    // ── type_name completeness ──

    #[test]
    fn test_type_name_is_consistent() {
        // Each variant's type_name should match the variant name
        assert_eq!(NodeData::Separator(SeparatorNode).type_name(), "Separator");
        assert_eq!(NodeData::Material(MaterialNode::default()).type_name(), "Material");
        assert_eq!(
            NodeData::PerspectiveCamera(PerspectiveCameraNode::default()).type_name(),
            "PerspectiveCamera"
        );
        assert_eq!(
            NodeData::DirectionalLight(DirectionalLightNode::default()).type_name(),
            "DirectionalLight"
        );
        assert_eq!(NodeData::SectionPlane(SectionPlaneNode::default()).type_name(), "SectionPlane");
        assert_eq!(NodeData::Torus(TorusNode::default()).type_name(), "Torus");
        assert_eq!(NodeData::Markup(MarkupNode::default()).type_name(), "Markup");
        assert_eq!(
            NodeData::Measurement(MeasurementNode::default()).type_name(),
            "Measurement"
        );
    }

    #[test]
    fn test_field_descriptor_indices_are_sequential() {
        let d = NodeData::SpotLight(SpotLightNode::default()).field_descriptors();
        for (i, fd) in d.iter().enumerate() {
            assert_eq!(fd.field_index, i as u16, "field {} should have index {}", fd.name, i);
        }
    }
}
