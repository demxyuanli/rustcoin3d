//! Advanced rendering effects: Decal, Volume, PointCloud.
//! Each effect is lazy-initialized when first needed.

mod collect;
mod render;

pub use collect::{DecalDrawCommand, EffectCommands, PointCloudDrawCommand, ProjectedAnnotation, VolumeDrawCommand, collect_effect_nodes};
pub use render::{DecalPass, PointCloudPass, VolumePass};

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::Vec3;
    use rc3d_scene::{
        DecalNode, GroupNode, NodeData, PointCloudNode, SceneGraph, TransformNode, VolumeNode,
    };

    #[test]
    fn collect_effect_nodes_preserves_decal_volume_and_point_cloud_fields() {
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Group(GroupNode));
        graph.add_child(
            root,
            NodeData::Decal(DecalNode {
                position: Vec3::new(1.0, 2.0, 3.0),
                direction: Vec3::NEG_Y,
                size: [4.0, 5.0],
                texture_path: "decal.png".to_string(),
                color: [1.0, 0.5, 0.25, 0.75],
                opacity: 0.8,
            }),
        );
        graph.add_child(
            root,
            NodeData::Volume(VolumeNode {
                dimensions: [16, 32, 64],
                texture_path: "volume.raw".to_string(),
                density_scale: 2.5,
                color_map: [[0.1, 0.2, 0.3, 0.4]; 4],
            }),
        );
        graph.add_child(
            root,
            NodeData::PointCloud(PointCloudNode {
                file_path: "points.bin".to_string(),
                max_visible_points: 1234,
                point_size: 3.0,
                color: [0.2, 0.4, 0.6, 1.0],
            }),
        );

        let commands = collect_effect_nodes(&graph);

        assert_eq!(commands.decals.len(), 1);
        assert_eq!(commands.decals[0].texture_path, "decal.png");
        assert_eq!(commands.decals[0].position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(commands.decals[0].size, [4.0, 5.0]);
        assert_eq!(commands.decals[0].opacity, 0.8);
        assert_eq!(commands.volumes.len(), 1);
        assert_eq!(commands.volumes[0].texture_path, "volume.raw");
        assert_eq!(commands.volumes[0].dimensions, [16, 32, 64]);
        assert_eq!(commands.volumes[0].density_scale, 2.5);
        assert_eq!(commands.point_clouds.len(), 1);
        assert_eq!(commands.point_clouds[0].file_path, "points.bin");
        assert_eq!(commands.point_clouds[0].max_visible_points, 1234);
        assert_eq!(commands.point_clouds[0].point_size, 3.0);
    }

    #[test]
    fn collect_effect_nodes_recurses_through_effect_children_and_transforms() {
        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Group(GroupNode));
        let transform = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(10.0, 0.0, 0.0))),
        );
        let decal = graph.add_child(
            transform,
            NodeData::Decal(DecalNode {
                texture_path: "outer.png".to_string(),
                ..Default::default()
            }),
        );
        graph.add_child(
            decal,
            NodeData::PointCloud(PointCloudNode {
                file_path: "nested.bin".to_string(),
                ..Default::default()
            }),
        );

        let commands = collect_effect_nodes(&graph);

        assert_eq!(commands.decals.len(), 1);
        assert_eq!(commands.point_clouds.len(), 1);
        assert_eq!(commands.decals[0].model_matrix.w_axis.x, 10.0);
        assert_eq!(commands.point_clouds[0].model_matrix.w_axis.x, 10.0);
        assert_eq!(commands.point_clouds[0].file_path, "nested.bin");
    }
}
