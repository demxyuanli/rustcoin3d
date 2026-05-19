//! Screen-space glyphon labels for 3D annotation elements.

use glam::{Mat4, Vec3};
use rc3d_scene::node_data::AnnotationElement;

use super::projection::project_point_vp;
use crate::render_passes::pass_effects::EffectCommands;
use crate::render_passes::pass_text::TextDrawCommand;

fn dimension_dim_line_screen_mid(
    start: [f32; 3],
    end: [f32; 3],
    offset_dir: [f32; 3],
    extension_len: f32,
    model: Mat4,
    scene_vp: Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
) -> Option<[f32; 2]> {
    let dim_dir = Vec3::new(end[0] - start[0], end[1] - start[1], end[2] - start[2]);
    let dim_len = dim_dir.length().max(0.001);
    let ndim = dim_dir / dim_len;
    let ext = extension_len;
    let dl_start = Vec3::new(
        start[0] + offset_dir[0] - ndim.x * ext,
        start[1] + offset_dir[1] - ndim.y * ext,
        start[2] + offset_dir[2] - ndim.z * ext,
    );
    let dl_end = Vec3::new(
        end[0] + offset_dir[0] + ndim.x * ext,
        end[1] + offset_dir[1] + ndim.y * ext,
        end[2] + offset_dir[2] + ndim.z * ext,
    );
    let s0 = project_point_vp(dl_start, model, scene_vp, screen_w, screen_h, depth_reversed_z)?;
    let s1 = project_point_vp(dl_end, model, scene_vp, screen_w, screen_h, depth_reversed_z)?;
    Some([(s0[0] + s1[0]) * 0.5, (s0[1] + s1[1]) * 0.5])
}

/// Build HUD text draw commands for Dimension and Leader annotation labels.
pub fn build_annotation_label_commands(
    effect_commands: &EffectCommands,
    scene_vp: Mat4,
    screen_w: u32,
    screen_h: u32,
    depth_reversed_z: bool,
) -> Vec<TextDrawCommand> {
    let sw = screen_w.max(1) as f32;
    let sh = screen_h.max(1) as f32;
    let mut out = Vec::new();

    for pa in &effect_commands.annotation_elements {
        match &pa.element {
            AnnotationElement::Dimension {
                start,
                end,
                offset_dir,
                extension_len,
                label,
                color,
                ..
            } => {
                if label.is_empty() {
                    continue;
                }
                let Some(screen) = dimension_dim_line_screen_mid(
                    *start,
                    *end,
                    *offset_dir,
                    *extension_len,
                    pa.model_matrix,
                    scene_vp,
                    sw,
                    sh,
                    depth_reversed_z,
                ) else {
                    continue;
                };
                out.push(TextDrawCommand {
                    string: label.clone(),
                    screen_pos: screen,
                    size: 14.0,
                    color: *color,
                    is_3d: false,
                });
            }
            AnnotationElement::Leader {
                anchor,
                label_offset,
                text,
                color,
                ..
            } => {
                if text.is_empty() {
                    continue;
                }
                let anchor_local = Vec3::from(*anchor);
                let Some(ap) = project_point_vp(
                    anchor_local,
                    pa.model_matrix,
                    scene_vp,
                    sw,
                    sh,
                    depth_reversed_z,
                ) else {
                    continue;
                };
                out.push(TextDrawCommand {
                    string: text.clone(),
                    screen_pos: [ap[0] + label_offset[0], ap[1] + label_offset[1]],
                    size: 14.0,
                    color: *color,
                    is_3d: false,
                });
            }
            _ => {}
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_passes::pass_effects::ProjectedAnnotation;
    use rc3d_scene::node_data::AnnotationElement;

    #[test]
    fn dimension_label_projects_to_screen() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let scene_vp = proj * view;
        let mut cmds = EffectCommands::default();
        cmds.annotation_elements.push(ProjectedAnnotation {
            element: AnnotationElement::Dimension {
                start: [-1.0, 0.0, 0.0],
                end: [1.0, 0.0, 0.0],
                offset_dir: [0.0, -1.0, 0.0],
                extension_len: 0.2,
                arrow_size: 0.1,
                label: "W=2".into(),
                color: [1.0, 1.0, 1.0, 1.0],
            },
            model_matrix: Mat4::IDENTITY,
        });
        let labels = build_annotation_label_commands(&cmds, scene_vp, 800, 600, false);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].string, "W=2");
        assert!(labels[0].screen_pos[0] > 0.0 && labels[0].screen_pos[0] < 800.0);
    }
}
