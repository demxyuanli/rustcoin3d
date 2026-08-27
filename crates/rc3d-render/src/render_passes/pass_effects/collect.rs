use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::annotation::prepare_annotation_for_render;
use rc3d_scene::node_data::{AnnotationElement, AnnotationStyle};
use rc3d_scene::{NodeData, PointCloudNode, SceneGraph};
use slotmap::Key;

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

/// Visibility flags computed per-annotation during projection.
#[derive(Clone, Debug, Default)]
pub struct AnnotationVisibility {
    pub back_facing: bool,
    pub outside_ndc: bool,
    /// Deferred: occlusion check (Phase 1.5)
    pub occluded: bool,
}

/// A 3D annotation element collected during scene traversal, with its world-space transform.
/// Projection to screen coordinates happens in the render pass.
#[derive(Clone, Debug)]
pub struct ProjectedAnnotation {
    pub element: AnnotationElement,
    pub model_matrix: Mat4,
    pub style: AnnotationStyle,
    pub visibility: AnnotationVisibility,
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

/// GPU point sprite (64-byte aligned: xyz size + rgba + velocity/age + extra).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPoint {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub velocity: [f32; 4],
    pub extra: [f32; 4],
}

/// CPU snapshot of a particle emitter for the GPU compute path.
#[derive(Clone, Copy, Debug)]
pub struct GpuEmitterParams {
    pub origin: [f32; 3],
    pub origin_jitter: [f32; 3],
    pub velocity: [f32; 3],
    pub velocity_jitter: [f32; 3],
    pub acceleration: [f32; 3],
    pub spawn_rate: f32,
    pub max_particles: u32,
    pub lifetime: f32,
    pub lifetime_jitter: f32,
    pub start_color: [f32; 4],
    pub end_color: [f32; 4],
    pub start_size: f32,
    pub end_size: f32,
}

impl GpuEmitterParams {
    pub fn from_emitter(e: &rc3d_scene::ParticleEmitter) -> Self {
        Self {
            origin: [e.origin.x, e.origin.y, e.origin.z],
            origin_jitter: [e.origin_jitter.x, e.origin_jitter.y, e.origin_jitter.z],
            velocity: [e.velocity.x, e.velocity.y, e.velocity.z],
            velocity_jitter: [e.velocity_jitter.x, e.velocity_jitter.y, e.velocity_jitter.z],
            acceleration: [e.acceleration.x, e.acceleration.y, e.acceleration.z],
            spawn_rate: e.spawn_rate,
            max_particles: e.max_particles,
            lifetime: e.lifetime,
            lifetime_jitter: e.lifetime_jitter,
            start_color: e.start_color,
            end_color: e.end_color,
            start_size: e.start_size,
            end_size: e.end_size,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PointCloudDrawCommand {
    pub model_matrix: Mat4,
    pub file_path: String,
    pub max_visible_points: u32,
    pub point_size: f32,
    pub color: [f32; 4],
    pub is_overlay: bool,
    /// In-memory particles / uploaded points. Empty = file-backed or GPU-sim.
    pub points: std::sync::Arc<Vec<GpuPoint>>,
    pub sim_key: u64,
    pub emitter: Option<GpuEmitterParams>,
}

pub fn pack_point_cloud_gpu(pc: &PointCloudNode) -> std::sync::Arc<Vec<GpuPoint>> {
    if pc.particles.is_empty() {
        return std::sync::Arc::new(Vec::new());
    }
    let n = pc.particles.len().min(pc.max_visible_points as usize);
    let mut out = Vec::with_capacity(n);
    for p in pc.particles.iter().take(n) {
        out.push(GpuPoint {
            position: [p.position.x, p.position.y, p.position.z, p.size.max(0.1)],
            color: p.color,
            velocity: [p.velocity.x, p.velocity.y, p.velocity.z, p.age],
            extra: [p.lifetime, 0.0, 0.0, 0.0],
        });
    }
    std::sync::Arc::new(out)
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
            let gpu_emitter = point_cloud
                .emitter
                .as_ref()
                .filter(|e| !e.simulate_on_cpu)
                .map(GpuEmitterParams::from_emitter);
            commands.point_clouds.push(PointCloudDrawCommand {
                model_matrix,
                file_path: point_cloud.file_path.clone(),
                max_visible_points: point_cloud.max_visible_points,
                point_size: point_cloud.point_size,
                color: point_cloud.color,
                is_overlay: inside_annotation,
                points: if gpu_emitter.is_some() {
                    std::sync::Arc::new(Vec::new())
                } else {
                    pack_point_cloud_gpu(point_cloud)
                },
                sim_key: node.data().as_ffi(),
                emitter: gpu_emitter,
            });
            for &child in &entry.children {
                collect_effect_recursive(graph, child, model_matrix, inside_annotation, commands);
            }
        }
        NodeData::Transform(_) | NodeData::Rotation(_) | NodeData::RotationXYZ(_) => {
            let next_model = match entry.data.local_matrix() {
                Some(lm) => model_matrix * lm,
                None => model_matrix,
            };
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
                if let Some(lm) = ce.data.local_matrix() {
                    accum *= lm;
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
                    let (element, el_model) =
                        prepare_annotation_for_render(graph, model_matrix, el);
                    commands.annotation_elements.push(ProjectedAnnotation {
                        element,
                        model_matrix: el_model,
                        style: ann.style.clone(),
                        visibility: AnnotationVisibility::default(),
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
