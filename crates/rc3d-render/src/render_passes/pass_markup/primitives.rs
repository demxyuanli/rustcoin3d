use crate::render_passes::pass_effects::ProjectedAnnotation;
use crate::vertex::MarkupVertex;

use super::projection::project_point;

/// Project 3D annotation elements to 2D MarkupVertex.
pub(super) fn project_annotation_elements(
    elements: &[ProjectedAnnotation],
    view: glam::Mat4,
    proj: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
) -> Vec<MarkupVertex> {
    use rc3d_scene::node_data::AnnotationElement;
    let mut out = Vec::new();

    for pa in elements {
        let proj_pt = |p: &[f32; 3]| {
            let pos = glam::Vec3::new(p[0], p[1], p[2]);
            project_point(pos, pa.model_matrix, view, proj, screen_w, screen_h, depth_reversed_z)
        };

        match &pa.element {
            AnnotationElement::Dimension { start, end, offset_dir, extension_len, arrow_size, color, .. } => {
                let color = *color;

                // ── Compute ALL geometry in 3D on a single annotation plane ──
                let dim_dir_3d = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
                let dim_len_3d = (dim_dir_3d[0].powi(2) + dim_dir_3d[1].powi(2) + dim_dir_3d[2].powi(2)).sqrt().max(0.001);
                let ndim_3d = [dim_dir_3d[0] / dim_len_3d, dim_dir_3d[1] / dim_len_3d, dim_dir_3d[2] / dim_len_3d];

                // Normalized offset direction (perpendicular to measurement line, for arrow opening)
                let off_len = (offset_dir[0].powi(2) + offset_dir[1].powi(2) + offset_dir[2].powi(2)).sqrt().max(0.001);
                let off_n = [offset_dir[0] / off_len, offset_dir[1] / off_len, offset_dir[2] / off_len];

                let ext = *extension_len;
                let asz = *arrow_size;
                let ahw = asz * 0.4; // arrow half-width

                // Measurement points on object
                let points_3d: [[f32; 3]; 12] = [
                    // 0: s0 — measurement start
                    [start[0], start[1], start[2]],
                    // 1: e0 — measurement end
                    [end[0], end[1], end[2]],
                    // 2: ext_s — extension anchor start (start + offset)
                    [start[0] + offset_dir[0], start[1] + offset_dir[1], start[2] + offset_dir[2]],
                    // 3: ext_e — extension anchor end (end + offset)
                    [end[0] + offset_dir[0], end[1] + offset_dir[1], end[2] + offset_dir[2]],
                    // 4: dl_start — dim-line start (offset + extended left)
                    [start[0] + offset_dir[0] - ndim_3d[0] * ext, start[1] + offset_dir[1] - ndim_3d[1] * ext, start[2] + offset_dir[2] - ndim_3d[2] * ext],
                    // 5: dl_end — dim-line end (offset + extended right)
                    [end[0] + offset_dir[0] + ndim_3d[0] * ext, end[1] + offset_dir[1] + ndim_3d[1] * ext, end[2] + offset_dir[2] + ndim_3d[2] * ext],
                    // 6: aL1 — left arrow wing 1 (dl_start → right, opening perpendicular)
                    [start[0] + offset_dir[0] - ndim_3d[0] * ext + ndim_3d[0] * asz + off_n[0] * ahw,
                     start[1] + offset_dir[1] - ndim_3d[1] * ext + ndim_3d[1] * asz + off_n[1] * ahw,
                     start[2] + offset_dir[2] - ndim_3d[2] * ext + ndim_3d[2] * asz + off_n[2] * ahw],
                    // 7: aL2 — left arrow wing 2
                    [start[0] + offset_dir[0] - ndim_3d[0] * ext + ndim_3d[0] * asz - off_n[0] * ahw,
                     start[1] + offset_dir[1] - ndim_3d[1] * ext + ndim_3d[1] * asz - off_n[1] * ahw,
                     start[2] + offset_dir[2] - ndim_3d[2] * ext + ndim_3d[2] * asz - off_n[2] * ahw],
                    // 8: aR1 — right arrow wing 1 (dl_end → left, opening perpendicular)
                    [end[0] + offset_dir[0] + ndim_3d[0] * ext - ndim_3d[0] * asz + off_n[0] * ahw,
                     end[1] + offset_dir[1] + ndim_3d[1] * ext - ndim_3d[1] * asz + off_n[1] * ahw,
                     end[2] + offset_dir[2] + ndim_3d[2] * ext - ndim_3d[2] * asz + off_n[2] * ahw],
                    // 9: aR2 — right arrow wing 2
                    [end[0] + offset_dir[0] + ndim_3d[0] * ext - ndim_3d[0] * asz - off_n[0] * ahw,
                     end[1] + offset_dir[1] + ndim_3d[1] * ext - ndim_3d[1] * asz - off_n[1] * ahw,
                     end[2] + offset_dir[2] + ndim_3d[2] * ext - ndim_3d[2] * asz - off_n[2] * ahw],
                    // 10: aL_tip — left arrow tip (for the back edge)
                    [start[0] + offset_dir[0] - ndim_3d[0] * ext + ndim_3d[0] * asz,
                     start[1] + offset_dir[1] - ndim_3d[1] * ext + ndim_3d[1] * asz,
                     start[2] + offset_dir[2] - ndim_3d[2] * ext + ndim_3d[2] * asz],
                    // 11: aR_tip — right arrow tip
                    [end[0] + offset_dir[0] + ndim_3d[0] * ext - ndim_3d[0] * asz,
                     end[1] + offset_dir[1] + ndim_3d[1] * ext - ndim_3d[1] * asz,
                     end[2] + offset_dir[2] + ndim_3d[2] * ext - ndim_3d[2] * asz],
                ];

                // ── Project all 12 points to screen ──
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
                if !ok { continue; }

                // Dimension line
                out.push(MarkupVertex { position: [scr[4][0], scr[4][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[5][0], scr[5][1], 0.0], color });
                // Extension lines (measurement → offset anchor)
                out.push(MarkupVertex { position: [scr[0][0], scr[0][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[2][0], scr[2][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[1][0], scr[1][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[3][0], scr[3][1], 0.0], color });
                // Left arrow (dl_start → wings + back edge)
                out.push(MarkupVertex { position: [scr[4][0], scr[4][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[6][0], scr[6][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[4][0], scr[4][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[7][0], scr[7][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[6][0], scr[6][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[10][0], scr[10][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[7][0], scr[7][1], 0.0], color });
                out.push(MarkupVertex { position: [scr[10][0], scr[10][1], 0.0], color });
                // Right arrow (dl_end → wings + back edge)
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

                // ── Fixed world axes (labeled plane coords) ──
                let world_x = glam::Vec3::X;
                let world_y = glam::Vec3::Y;

                // ── Pixels-per-world-unit at anchor depth (via world X) ──
                let test_3d = [anchor[0] + 0.1, anchor[1], anchor[2]];
                let ts = proj_pt(&test_3d).unwrap_or(ap);
                let px_per_unit = ((ts[0] - ap[0]).powi(2) + (ts[1] - ap[1]).powi(2)).sqrt().max(1.0) / 0.1;

                // ── Label position in 3D (fixed world axes, not camera-relative) ──
                let world_dx = label_offset[0] / px_per_unit;
                let world_dy = label_offset[1] / px_per_unit;
                let label_3d = [
                    anchor[0] + world_x.x * world_dx + world_y.x * world_dy,
                    anchor[1] + world_x.y * world_dx + world_y.y * world_dy,
                    anchor[2] + world_x.z * world_dx + world_y.z * world_dy,
                ];
                let Some(lp) = proj_pt(&label_3d) else { continue };

                // Leader line (both endpoints in 3D, fixed in world)
                out.push(MarkupVertex { position: [ap[0], ap[1], 0.0], color });
                out.push(MarkupVertex { position: [lp[0], lp[1], 0.0], color });
                // Dot at anchor (screen-space circle, cosmetic)
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
                let sz = *size;

                // ── Fixed world diagonals (X+Z and X-Z on world XZ plane, damped by Y) ──
                let d1 = glam::Vec3::new(sz * 0.707, sz * 0.1, sz * 0.707);
                let d2 = glam::Vec3::new(sz * 0.707, sz * 0.1, -sz * 0.707);

                let se_3d = [position[0] + d1.x, position[1] + d1.y, position[2] + d1.z];
                let nw_3d = [position[0] - d1.x, position[1] - d1.y, position[2] - d1.z];
                let ne_3d = [position[0] + d2.x, position[1] + d2.y, position[2] + d2.z];
                let sw_3d = [position[0] - d2.x, position[1] - d2.y, position[2] - d2.z];

                // ── Project all 5 points ──
                let Some(c) = proj_pt(position) else { continue };
                let nw = proj_pt(&nw_3d).unwrap_or(c);
                let se = proj_pt(&se_3d).unwrap_or(c);
                let ne = proj_pt(&ne_3d).unwrap_or(c);
                let sw = proj_pt(&sw_3d).unwrap_or(c);

                // Cross (two diagonal lines, fixed in world)
                out.push(MarkupVertex { position: [nw[0], nw[1], 0.0], color });
                out.push(MarkupVertex { position: [se[0], se[1], 0.0], color });
                out.push(MarkupVertex { position: [ne[0], ne[1], 0.0], color });
                out.push(MarkupVertex { position: [sw[0], sw[1], 0.0], color });
            }
        }
    }

    out
}
