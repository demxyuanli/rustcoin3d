use crate::render_passes::pass_effects::ProjectedAnnotation;
use crate::vertex::MarkupVertex;
use glam::Vec3;

use super::projection::{pixels_per_world_unit_at, project_point_vp};

/// Project 3D annotation elements to 2D MarkupVertex.
pub(super) fn project_annotation_elements(
    elements: &[ProjectedAnnotation],
    scene_vp: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
) -> Vec<MarkupVertex> {
    use rc3d_scene::node_data::AnnotationElement;
    let mut out = Vec::new();

    for pa in elements {
        let model = pa.model_matrix;
        let proj_pt = |p: &[f32; 3]| {
            project_point_vp(
                Vec3::new(p[0], p[1], p[2]),
                model,
                scene_vp,
                screen_w,
                screen_h,
                depth_reversed_z,
            )
        };

        match &pa.element {
            AnnotationElement::Dimension { start, end, offset_dir, extension_len, arrow_size, color, .. } => {
                let color = *color;

                let dim_dir_3d = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
                let dim_len_3d = (dim_dir_3d[0].powi(2) + dim_dir_3d[1].powi(2) + dim_dir_3d[2].powi(2))
                    .sqrt()
                    .max(0.001);
                let ndim_3d = [
                    dim_dir_3d[0] / dim_len_3d,
                    dim_dir_3d[1] / dim_len_3d,
                    dim_dir_3d[2] / dim_len_3d,
                ];

                let off_len = (offset_dir[0].powi(2) + offset_dir[1].powi(2) + offset_dir[2].powi(2))
                    .sqrt()
                    .max(0.001);
                let off_n = [
                    offset_dir[0] / off_len,
                    offset_dir[1] / off_len,
                    offset_dir[2] / off_len,
                ];

                let ext = *extension_len;
                let asz = *arrow_size;
                let ahw = asz * 0.4;

                let points_3d: [[f32; 3]; 12] = [
                    [start[0], start[1], start[2]],
                    [end[0], end[1], end[2]],
                    [
                        start[0] + offset_dir[0],
                        start[1] + offset_dir[1],
                        start[2] + offset_dir[2],
                    ],
                    [end[0] + offset_dir[0], end[1] + offset_dir[1], end[2] + offset_dir[2]],
                    [
                        start[0] + offset_dir[0] - ndim_3d[0] * ext,
                        start[1] + offset_dir[1] - ndim_3d[1] * ext,
                        start[2] + offset_dir[2] - ndim_3d[2] * ext,
                    ],
                    [
                        end[0] + offset_dir[0] + ndim_3d[0] * ext,
                        end[1] + offset_dir[1] + ndim_3d[1] * ext,
                        end[2] + offset_dir[2] + ndim_3d[2] * ext,
                    ],
                    [
                        start[0] + offset_dir[0] - ndim_3d[0] * ext + ndim_3d[0] * asz + off_n[0] * ahw,
                        start[1] + offset_dir[1] - ndim_3d[1] * ext + ndim_3d[1] * asz + off_n[1] * ahw,
                        start[2] + offset_dir[2] - ndim_3d[2] * ext + ndim_3d[2] * asz + off_n[2] * ahw,
                    ],
                    [
                        start[0] + offset_dir[0] - ndim_3d[0] * ext + ndim_3d[0] * asz - off_n[0] * ahw,
                        start[1] + offset_dir[1] - ndim_3d[1] * ext + ndim_3d[1] * asz - off_n[1] * ahw,
                        start[2] + offset_dir[2] - ndim_3d[2] * ext + ndim_3d[2] * asz - off_n[2] * ahw,
                    ],
                    [
                        end[0] + offset_dir[0] + ndim_3d[0] * ext - ndim_3d[0] * asz + off_n[0] * ahw,
                        end[1] + offset_dir[1] + ndim_3d[1] * ext - ndim_3d[1] * asz + off_n[1] * ahw,
                        end[2] + offset_dir[2] + ndim_3d[2] * ext - ndim_3d[2] * asz + off_n[2] * ahw,
                    ],
                    [
                        end[0] + offset_dir[0] + ndim_3d[0] * ext - ndim_3d[0] * asz - off_n[0] * ahw,
                        end[1] + offset_dir[1] + ndim_3d[1] * ext - ndim_3d[1] * asz - off_n[1] * ahw,
                        end[2] + offset_dir[2] + ndim_3d[2] * ext - ndim_3d[2] * asz - off_n[2] * ahw,
                    ],
                    [
                        start[0] + offset_dir[0] - ndim_3d[0] * ext + ndim_3d[0] * asz,
                        start[1] + offset_dir[1] - ndim_3d[1] * ext + ndim_3d[1] * asz,
                        start[2] + offset_dir[2] - ndim_3d[2] * ext + ndim_3d[2] * asz,
                    ],
                    [
                        end[0] + offset_dir[0] + ndim_3d[0] * ext - ndim_3d[0] * asz,
                        end[1] + offset_dir[1] + ndim_3d[1] * ext - ndim_3d[1] * asz,
                        end[2] + offset_dir[2] + ndim_3d[2] * ext - ndim_3d[2] * asz,
                    ],
                ];

                let mut scr = [[0.0f32; 2]; 12];
                let mut ok = true;
                for i in 0..12 {
                    if let Some(p) = proj_pt(&points_3d[i]) {
                        scr[i] = p;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }

                out.push(MarkupVertex { position: [scr[4][0], scr[4][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[5][0], scr[5][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[0][0], scr[0][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[2][0], scr[2][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[1][0], scr[1][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[3][0], scr[3][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[4][0], scr[4][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[6][0], scr[6][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[4][0], scr[4][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[7][0], scr[7][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[6][0], scr[6][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[10][0], scr[10][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[7][0], scr[7][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[10][0], scr[10][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[5][0], scr[5][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[8][0], scr[8][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[5][0], scr[5][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[9][0], scr[9][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[8][0], scr[8][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[11][0], scr[11][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[9][0], scr[9][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[11][0], scr[11][1], 0.0], color });
            }
            AnnotationElement::Leader { anchor, label_offset, color, .. } => {
                let Some(ap) = proj_pt(anchor) else { continue };
                let color = *color;
                // Leader line and label anchor: screen-space offset (pixels), fixed on screen.
                let lp = [ap[0] + label_offset[0], ap[1] + label_offset[1]];
                out.push(MarkupVertex { position: [ap[0], ap[1], 0.0], color });
                out.push(MarkupVertex { position: [lp[0], lp[1], 0.0], color });
                let r = 2.0;
                let n = 8;
                let mut prev = [ap[0] + r, ap[1]];
                for i in 1..=n {
                    let a = (i as f32 / n as f32) * std::f32::consts::TAU;
                    let curr = [ap[0] + r * a.cos(), ap[1] + r * a.sin()];
                    out.push(MarkupVertex { position: [prev[0], prev[1], 0.0], color });
                    out.push(MarkupVertex { position: [curr[0], curr[1], 0.0], color });
                    prev = curr;
                }
            }
            AnnotationElement::Datum { position, size, color } => {
                let color = *color;
                let anchor = Vec3::new(position[0], position[1], position[2]);
                let Some(c) = proj_pt(position) else { continue };
                let px_per_unit = pixels_per_world_unit_at(
                    anchor, model, scene_vp, screen_w, screen_h, depth_reversed_z,
                );
                let r = (*size * px_per_unit).max(4.0);
                out.push(MarkupVertex { position: [c[0] - r, c[1], 0.0], color });
                out.push(MarkupVertex { position: [c[0] + r, c[1], 0.0], color });
                out.push(MarkupVertex { position: [c[0], c[1] - r, 0.0], color });
                out.push(MarkupVertex { position: [c[0], c[1] + r, 0.0], color });
            }
        }
    }

    out
}
