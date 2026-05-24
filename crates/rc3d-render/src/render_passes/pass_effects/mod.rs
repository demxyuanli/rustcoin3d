//! Advanced rendering effects: Decal, Volume, PointCloud.
//! Each effect is lazy-initialized when first needed.

mod collect;
mod render;

pub use collect::{AnnotationVisibility, DecalDrawCommand, EffectCommands, PointCloudDrawCommand, ProjectedAnnotation, VolumeDrawCommand, collect_effect_nodes};
pub use render::{DecalPass, PointCloudPass, VolumePass};

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::{Mat4, Vec3};
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

    #[test]
    fn annotation_on_node_gets_transform_model_matrix() {
        use rc3d_scene::annotation::{AnnotationLabelMode, AnnotationPoint, AnnotationStyle};
        use rc3d_scene::node_data::{
            AnnotationElement, AnnotationNode, AnnotationSetNode, CubeNode, MaterialNode,
            SeparatorNode, TransformNode,
        };

        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        let cube_x = -3.5f32;
        let cube_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        let cube_tf = graph.add_child(
            cube_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(cube_x, 0.2, 0.0))),
        );
        graph.add_child(
            cube_tf,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(1.0, 0.6, 0.2))),
        );
        graph.add_child(cube_tf, NodeData::Cube(CubeNode { width: 1.5, height: 1.5, depth: 1.5 }));

        let ann = graph.add_child(root, NodeData::Annotation(AnnotationNode));
        let hw = 0.75f32;
        graph.add_child(
            ann,
            NodeData::AnnotationSet(AnnotationSetNode {
                style: AnnotationStyle::default(),
                elements: vec![AnnotationElement::Dimension {
                    start: AnnotationPoint::on_node(cube_tf, [-hw, -hw, 0.0]),
                    end: AnnotationPoint::on_node(cube_tf, [hw, -hw, 0.0]),
                    offset_dir: [0.0, -1.0, 0.0],
                    extension_len: 0.3,
                    arrow_size: 0.15,
                    label: String::new(),
                    label_mode: AnnotationLabelMode::Auto,
                    color: [1.0, 0.4, 0.0, 1.0],
                }],
                visible: true,
            }),
        );

        let cmds = collect_effect_nodes(&graph);
        assert_eq!(cmds.annotation_elements.len(), 1);
        let pa = &cmds.annotation_elements[0];
        let world_origin = pa.model_matrix.transform_point3(Vec3::ZERO);
        assert!(
            (world_origin.x - cube_x).abs() < 1e-3,
            "world x={} expected {}",
            world_origin.x,
            cube_x
        );
        assert!((world_origin.y - 0.2).abs() < 1e-3);

        match &cmds.annotation_elements[0].element {
            AnnotationElement::Dimension { start, .. } => {
                assert!(start.node.is_none());
                assert_eq!(start.local, [-hw, -hw, 0.0]);
            }
            _ => panic!("dimension expected"),
        }
    }

    /// Full-scene projection: cube left, cylinder right on screen (same camera as markup_dimensions).
    #[test]
    fn markup_dimensions_layout_projects_to_correct_screen_halves() {
        use crate::render_action::RenderCollector;
        use crate::render_passes::pass_markup::projection::project_point_vp;
        use rc3d_scene::node_data::{
            AnnotationElement, AnnotationSetNode, CubeNode, CylinderNode, PerspectiveCameraNode,
            SeparatorNode, TransformNode,
        };

        let mut graph = SceneGraph::new();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(4.0, 3.0, 12.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        let cube_x = -3.5f32;
        let cube_tf = {
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            let tf = graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(cube_x, 0.0, 0.0))),
            );
            graph.add_child(tf, NodeData::Cube(CubeNode { width: 1.5, height: 1.5, depth: 1.5 }));
            let hw = 0.75f32;
            graph.add_child(
                tf,
                NodeData::AnnotationSet(AnnotationSetNode {
                    style: rc3d_scene::annotation::AnnotationStyle::default(),
                    elements: vec![AnnotationElement::Dimension {
                        start: [-hw, 0.0, 0.0].into(),
                        end: [hw, 0.0, 0.0].into(),
                        offset_dir: [0.0, -1.0, 0.0],
                        extension_len: 0.3,
                        arrow_size: 0.15,
                        label: String::new(),
                        label_mode: rc3d_scene::annotation::AnnotationLabelMode::Auto,
                        color: [1.0, 0.4, 0.0, 1.0],
                    }],
                    visible: true,
                }),
            );
            tf
        };

        let cyl_x = 3.5f32;
        let cyl_tf = {
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            let tf = graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(cyl_x, -0.3, 0.0))),
            );
            graph.add_child(tf, NodeData::Cylinder(CylinderNode { radius: 0.8, height: 2.0 }));
            graph.add_child(
                tf,
                NodeData::AnnotationSet(AnnotationSetNode {
                    style: rc3d_scene::annotation::AnnotationStyle::default(),
                    elements: vec![AnnotationElement::Dimension {
                        start: [0.8, 0.0, 0.0].into(),
                        end: [-0.8, 0.0, 0.0].into(),
                        offset_dir: [0.0, 1.0, 0.0],
                        extension_len: 0.3,
                        arrow_size: 0.15,
                        label: String::new(),
                        label_mode: rc3d_scene::annotation::AnnotationLabelMode::Auto,
                        color: [0.1, 0.5, 1.0, 1.0],
                    }],
                    visible: true,
                }),
            );
            tf
        };

        let _ = (cube_tf, cyl_tf);

        let mut collector = RenderCollector::new();
        collector.traverse(&graph, root);

        let vp = collector.projection_matrix * collector.view_matrix;
        assert_ne!(vp, Mat4::IDENTITY);

        let screen_w = 800.0f32;
        let screen_h = 600.0f32;
        let depth_rev = false;

        let cube_pa = collector
            .effect_commands
            .annotation_elements
            .iter()
            .find(|pa| pa.model_matrix.w_axis.x < -1.0)
            .expect("cube annotation");
        let cyl_pa = collector
            .effect_commands
            .annotation_elements
            .iter()
            .find(|pa| pa.model_matrix.w_axis.x > 1.0)
            .expect("cylinder annotation");

        let cube_scr = project_point_vp(
            Vec3::ZERO,
            cube_pa.model_matrix,
            vp,
            screen_w,
            screen_h,
            depth_rev,
        )
        .expect("cube projects");
        let cyl_scr = project_point_vp(
            Vec3::ZERO,
            cyl_pa.model_matrix,
            vp,
            screen_w,
            screen_h,
            depth_rev,
        )
        .expect("cylinder projects");

        assert!(
            cube_scr[0] < screen_w * 0.45,
            "cube annotation should be on left, got x={}",
            cube_scr[0]
        );
        assert!(
            cyl_scr[0] > screen_w * 0.55,
            "cylinder annotation should be on right, got x={}",
            cyl_scr[0]
        );
        assert!(
            (cube_scr[0] - cyl_scr[0]).abs() > 200.0,
            "cube and cylinder screen x should be far apart"
        );
    }
}
