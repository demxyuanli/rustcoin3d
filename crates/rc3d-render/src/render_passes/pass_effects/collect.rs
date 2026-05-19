use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::node_data::AnnotationElement;
use rc3d_scene::{NodeData, SceneGraph};

#[derive(Clone, Debug, Default)]
pub struct EffectCommands {
    pub decals: Vec<DecalDrawCommand>,
    pub volumes: Vec<VolumeDrawCommand>,
    pub point_clouds: Vec<PointCloudDrawCommand>,
    pub annotation_elements: Vec<ProjectedAnnotation>,
}

impl EffectCommands {
    pub fn is_empty(&self) -> bool {
        self.decals.is_empty() && self.volumes.is_empty() && self.point_clouds.is_empty() && self.annotation_elements.is_empty()
    }
}

/// A 3D annotation element collected during scene traversal, with its world-space transform.
/// Projection to screen coordinates happens in the render pass.
#[derive(Clone, Debug)]
pub struct ProjectedAnnotation {
    pub element: AnnotationElement,
    pub model_matrix: Mat4,
}

#[derive(Clone, Debug)]
pub struct DecalDrawCommand {
    pub model_matrix: Mat4,
    pub position: Vec3,
    pub direction: Vec3,
    pub size: [f32; 2],
    pub texture_path: String,
    pub color: [f32; 4],
    pub opacity: f32,
    pub is_overlay: bool,
}

#[derive(Clone, Debug)]
pub struct VolumeDrawCommand {
    pub model_matrix: Mat4,
    pub dimensions: [u32; 3],
    pub texture_path: String,
    pub density_scale: f32,
    pub color_map: [[f32; 4]; 4],
    pub is_overlay: bool,
}

#[derive(Clone, Debug)]
pub struct PointCloudDrawCommand {
    pub model_matrix: Mat4,
    pub file_path: String,
    pub max_visible_points: u32,
    pub point_size: f32,
    pub color: [f32; 4],
    pub is_overlay: bool,
}

pub fn collect_effect_nodes(graph: &SceneGraph) -> EffectCommands {
    let mut commands = EffectCommands::default();
    for &root in graph.roots() {
        collect_effect_recursive(graph, root, Mat4::IDENTITY, false, &mut commands);
    }
    commands
}

fn collect_effect_recursive(
    graph: &SceneGraph,
    node: NodeId,
    model_matrix: Mat4,
    inside_annotation: bool,
    commands: &mut EffectCommands,
) {
    let Some(entry) = graph.get(node) else { return };
    match &entry.data {
        NodeData::Decal(decal) => {
            commands.decals.push(DecalDrawCommand {
                model_matrix,
                position: decal.position,
                direction: decal.direction,
                size: decal.size,
                texture_path: decal.texture_path.clone(),
                color: decal.color,
                opacity: decal.opacity,
                is_overlay: inside_annotation,
            });
            for &child in &entry.children {
                collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
            }
        }
        NodeData::Volume(volume) => {
            commands.volumes.push(VolumeDrawCommand {
                model_matrix,
                dimensions: volume.dimensions,
                texture_path: volume.texture_path.clone(),
                density_scale: volume.density_scale,
                color_map: volume.color_map,
                is_overlay: inside_annotation,
            });
            for &child in &entry.children {
                collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
            }
        }
        NodeData::PointCloud(point_cloud) => {
            commands.point_clouds.push(PointCloudDrawCommand {
                model_matrix,
                file_path: point_cloud.file_path.clone(),
                max_visible_points: point_cloud.max_visible_points,
                point_size: point_cloud.point_size,
                color: point_cloud.color,
                is_overlay: inside_annotation,
            });
            for &child in &entry.children {
                collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
            }
        }
        NodeData::Transform(transform) => {
            let next_model = model_matrix * transform.to_matrix();
            for &child in &entry.children {
                collect_effect_recursive(graph, child, next_model, inside_annotation, commands);
            }
        }
        NodeData::ResetTransform(_) => {
            for &child in &entry.children {
                collect_effect_recursive(graph, child, Mat4::IDENTITY, inside_annotation, commands);
            }
        }
        NodeData::Separator(_) => {
            let mut accum = model_matrix;
            for &child in &entry.children {
                let Some(ce) = graph.get(child) else { continue };
                if let NodeData::Transform(t) = &ce.data {
                    accum = accum * t.to_matrix();
                    for &gc in &ce.children {
                        collect_effect_recursive(graph, gc, accum, inside_annotation, commands);
                    }
                } else {
                    collect_effect_recursive(graph, child, accum, inside_annotation, commands);
                }
            }
        }
        NodeData::Annotation(_) => {
            for &child in &entry.children {
                collect_effect_recursive(graph, child, model_matrix, true, commands);
            }
        }
        NodeData::AnnotationSet(ann) => {
            if ann.visible {
                for el in &ann.elements {
                    commands.annotation_elements.push(ProjectedAnnotation {
                        element: el.clone(),
                        model_matrix,
                    });
                }
            }
        }
        NodeData::Switch(sw) => match sw.which_child {
            -2 => {}
            -1 => {
                for &child in &sw.children {
                    collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
                }
            }
            idx if idx >= 0 => {
                if let Some(&child) = sw.children.get(idx as usize) {
                    collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
                }
            }
            _ => {}
        },
        NodeData::MultipleCopy(mc) => {
            for &copy_matrix in &mc.copies {
                let next_model = model_matrix * copy_matrix;
                for &child in &mc.children {
                    collect_effect_recursive(graph, child, next_model, inside_annotation, commands);
                }
            }
        }
        NodeData::Lod(lod) => {
            let level = lod.current_level.min(lod.levels.len().saturating_sub(1));
            if let Some(level_data) = lod.levels.get(level) {
                for &child in &level_data.children {
                    collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
                }
            }
        }
        _ => {
            for &child in &entry.children {
                collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
            }
        }
    }
}
