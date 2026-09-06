use rc3d_scene::node_data::{
    AnnotationSetNode, AreaLightNode, BatchedMeshNode, BillboardNode, ConeNode, CubeNode,
    CylinderNode, DirectionalLightNode, EnvironmentNode, FontNode, HemisphereLightNode,
    InstancedMeshNode, LightProbeNode, LodNode, MarkupNode, MaterialNode, MeasurementNode,
    NodeData, OrthographicCameraNode, PerspectiveCameraNode, PointCloudNode, PointLightNode,
    SectionPlaneNode, SeparatorNode, SphereNode, SpotLightNode, SpriteNode, StereoCameraNode,
    SwitchNode, Text2Node, Text3Node, TransformNode,
};
use rc3d_scene::ParticleEmitter;

use crate::ui::types::NodeDataType;

pub(crate) fn node_data_for_type(t: NodeDataType) -> NodeData {
    match t {
        NodeDataType::Cube => NodeData::Cube(CubeNode::default()),
        NodeDataType::Sphere => NodeData::Sphere(SphereNode::default()),
        NodeDataType::Cylinder => NodeData::Cylinder(CylinderNode::default()),
        NodeDataType::Cone => NodeData::Cone(ConeNode::default()),
        NodeDataType::Separator => NodeData::Separator(SeparatorNode),
        NodeDataType::DirectionalLight => {
            NodeData::DirectionalLight(DirectionalLightNode::default())
        }
        NodeDataType::PointLight => NodeData::PointLight(PointLightNode::default()),
        NodeDataType::SpotLight => NodeData::SpotLight(SpotLightNode::default()),
        NodeDataType::PerspectiveCamera => {
            NodeData::PerspectiveCamera(PerspectiveCameraNode::default())
        }
        NodeDataType::OrthographicCamera => {
            NodeData::OrthographicCamera(OrthographicCameraNode::default())
        }
        NodeDataType::Text2 => NodeData::Text2(Text2Node::default()),
        NodeDataType::Text3 => NodeData::Text3(Text3Node::default()),
        NodeDataType::Font => NodeData::Font(FontNode::default()),
        NodeDataType::Transform => NodeData::Transform(TransformNode::default()),
        NodeDataType::Material => NodeData::Material(MaterialNode::default()),
        NodeDataType::HemisphereLight => NodeData::HemisphereLight(HemisphereLightNode::default()),
        NodeDataType::AreaLight => NodeData::AreaLight(AreaLightNode::default()),
        NodeDataType::LightProbe => NodeData::LightProbe(LightProbeNode::default()),
        NodeDataType::StereoCamera => NodeData::StereoCamera(StereoCameraNode::default()),
        NodeDataType::Billboard => NodeData::Billboard(BillboardNode::default()),
        NodeDataType::Sprite => NodeData::Sprite(SpriteNode::default()),
        NodeDataType::Lod => NodeData::Lod(LodNode::default()),
        NodeDataType::Switch => NodeData::Switch(SwitchNode::default()),
        NodeDataType::Environment => NodeData::Environment(EnvironmentNode::default()),
        NodeDataType::SectionPlane => NodeData::SectionPlane(SectionPlaneNode::default()),
        NodeDataType::AnnotationSet => NodeData::AnnotationSet(AnnotationSetNode::default()),
        NodeDataType::Measurement => NodeData::Measurement(MeasurementNode::default()),
        NodeDataType::Markup => NodeData::Markup(MarkupNode::default()),
        NodeDataType::InstancedMesh => NodeData::InstancedMesh(default_instanced_mesh()),
        NodeDataType::BatchedMesh => NodeData::BatchedMesh(default_batched_mesh()),
        NodeDataType::Particles => {
            NodeData::PointCloud(PointCloudNode::with_emitter(ParticleEmitter::fountain()))
        }
    }
}

fn identity_m() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn translated_m(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ]
}

fn default_instanced_mesh() -> InstancedMeshNode {
    InstancedMeshNode {
        transforms: vec![
            identity_m(),
            translated_m(1.5, 0.0, 0.0),
            translated_m(-1.5, 0.0, 0.0),
        ],
    }
}

fn default_batched_mesh() -> BatchedMeshNode {
    let mut node = BatchedMeshNode::default();
    let positions = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let normals = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
    let texcoords = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
    let tangents = [
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ];
    let geom = node.add_geometry(&positions, &normals, &texcoords, &tangents, &[0, 1, 2]);
    node.add_instance(geom, identity_m());
    node.add_instance(geom, translated_m(1.25, 0.0, 0.0));
    node
}
