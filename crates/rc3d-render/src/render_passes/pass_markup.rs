//! Screen-space markup overlay rendering.
//!
//! Collects all MarkupNodes from the scene graph and renders their elements
//! as screen-space line geometry with depth_compare: Always (overlay).

use wgpu::util::DeviceExt;

use crate::vertex::MarkupVertex;
use rc3d_core::NodeId;
use rc3d_scene::node_data::{MarkupElement, NodeData};
use rc3d_scene::SceneGraph;

/// Flatten all visible markup elements from the scene graph into colored line vertices.
pub fn collect_markup_lines(
    graph: &SceneGraph,
    root: NodeId,
    _surface_w: u32,
    _surface_h: u32,
) -> Vec<MarkupVertex> {
    let mut vertices = Vec::new();
    collect_recursive(graph, root, &mut vertices);
    vertices
}

fn collect_recursive(graph: &SceneGraph, node: NodeId, out: &mut Vec<MarkupVertex>) {
    let Some(entry) = graph.get(node) else { return };

    if let NodeData::Markup(m) = &entry.data {
        if m.visible {
            for el in &m.elements {
                push_element_vertices(el, out);
            }
        }
    }

    for &child in &entry.children {
        collect_recursive(graph, child, out);
    }
}

fn push_element_vertices(el: &MarkupElement, out: &mut Vec<MarkupVertex>) {
    match *el {
        MarkupElement::Line { start, end, color, .. } => {
            out.push(MarkupVertex { position: [start[0], start[1], 0.0], color });
            out.push(MarkupVertex { position: [end[0], end[1], 0.0], color });
        }
        MarkupElement::Rect {
            origin, size, color, ..
        } => {
            let x0 = origin[0];
            let y0 = origin[1];
            let x1 = x0 + size[0];
            let y1 = y0 + size[1];
            let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
            for i in 0..4 {
                out.push(MarkupVertex { position: [corners[i][0], corners[i][1], 0.0], color });
                out.push(MarkupVertex { position: [corners[(i + 1) % 4][0], corners[(i + 1) % 4][1], 0.0], color });
            }
        }
        MarkupElement::Circle { center, radius, color, .. } => {
            let n_segments = 64usize;
            let mut prev = [center[0] + radius, center[1]];
            for i in 1..=n_segments {
                let angle = (i as f32 / n_segments as f32) * std::f32::consts::TAU;
                let curr = [center[0] + radius * angle.cos(), center[1] + radius * angle.sin()];
                out.push(MarkupVertex { position: [prev[0], prev[1], 0.0], color });
                out.push(MarkupVertex { position: [curr[0], curr[1], 0.0], color });
                prev = curr;
            }
        }
        MarkupElement::Freehand { ref points, color, .. } => {
            for w in points.windows(2) {
                out.push(MarkupVertex { position: [w[0][0], w[0][1], 0.0], color });
                out.push(MarkupVertex { position: [w[1][0], w[1][1], 0.0], color });
            }
        }
        MarkupElement::Dimension {
            start, end, offset_dir, extension_len, arrow_size, color, ..
        } => {
            let dl_start = [start[0] + offset_dir[0] * extension_len, start[1] + offset_dir[1] * extension_len];
            let dl_end = [end[0] + offset_dir[0] * extension_len, end[1] + offset_dir[1] * extension_len];
            // Dimension line
            out.push(MarkupVertex { position: [dl_start[0], dl_start[1], 0.0], color });
            out.push(MarkupVertex { position: [dl_end[0], dl_end[1], 0.0], color });
            // Extension lines
            out.push(MarkupVertex { position: [start[0], start[1], 0.0], color });
            out.push(MarkupVertex { position: [dl_start[0], dl_start[1], 0.0], color });
            out.push(MarkupVertex { position: [end[0], end[1], 0.0], color });
            out.push(MarkupVertex { position: [dl_end[0], dl_end[1], 0.0], color });
            // Arrowheads
            let dir = [(end[0] - start[0]), (end[1] - start[1])];
            let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(0.001);
            let ndir = [dir[0] / len, dir[1] / len];
            let perp = [-ndir[1], ndir[0]];
            let asz = arrow_size;
            // Start arrowhead
            let p1 = [dl_start[0] + perp[0] * asz * 0.5, dl_start[1] + perp[1] * asz * 0.5];
            let p2 = [dl_start[0] - perp[0] * asz * 0.5, dl_start[1] - perp[1] * asz * 0.5];
            out.push(MarkupVertex { position: [dl_start[0], dl_start[1], 0.0], color });
            out.push(MarkupVertex { position: [p1[0], p1[1], 0.0], color });
            out.push(MarkupVertex { position: [dl_start[0], dl_start[1], 0.0], color });
            out.push(MarkupVertex { position: [p2[0], p2[1], 0.0], color });
            // End arrowhead
            let e1 = [dl_end[0] + perp[0] * asz * 0.5, dl_end[1] + perp[1] * asz * 0.5];
            let e2 = [dl_end[0] - perp[0] * asz * 0.5, dl_end[1] - perp[1] * asz * 0.5];
            out.push(MarkupVertex { position: [dl_end[0], dl_end[1], 0.0], color });
            out.push(MarkupVertex { position: [e1[0], e1[1], 0.0], color });
            out.push(MarkupVertex { position: [dl_end[0], dl_end[1], 0.0], color });
            out.push(MarkupVertex { position: [e2[0], e2[1], 0.0], color });
        }
        MarkupElement::AngleDimension { center, arm1, arm2, radius, color, .. } => {
            let a1 = (arm1[1] - center[1]).atan2(arm1[0] - center[0]);
            let a2 = (arm2[1] - center[1]).atan2(arm2[0] - center[0]);
            let n = 32usize;
            let span = a2 - a1;
            let mut prev = [center[0] + radius * a1.cos(), center[1] + radius * a1.sin()];
            for i in 1..=n {
                let t = i as f32 / n as f32;
                let angle = a1 + span * t;
                let curr = [center[0] + radius * angle.cos(), center[1] + radius * angle.sin()];
                out.push(MarkupVertex { position: [prev[0], prev[1], 0.0], color });
                out.push(MarkupVertex { position: [curr[0], curr[1], 0.0], color });
                prev = curr;
            }
            // Arms to center
            out.push(MarkupVertex { position: [center[0], center[1], 0.0], color });
            out.push(MarkupVertex { position: [arm1[0], arm1[1], 0.0], color });
            out.push(MarkupVertex { position: [center[0], center[1], 0.0], color });
            out.push(MarkupVertex { position: [arm2[0], arm2[1], 0.0], color });
        }
        MarkupElement::RadialDimension { center, perimeter, color, .. } => {
            out.push(MarkupVertex { position: [center[0], center[1], 0.0], color });
            out.push(MarkupVertex { position: [perimeter[0], perimeter[1], 0.0], color });
        }
        MarkupElement::DiameterDimension { p1, p2, center, color, .. } => {
            out.push(MarkupVertex { position: [p1[0], p1[1], 0.0], color });
            out.push(MarkupVertex { position: [p2[0], p2[1], 0.0], color });
            let s = 3.0;
            out.push(MarkupVertex { position: [center[0] - s, center[1] - s, 0.0], color });
            out.push(MarkupVertex { position: [center[0] + s, center[1] + s, 0.0], color });
            out.push(MarkupVertex { position: [center[0] + s, center[1] - s, 0.0], color });
            out.push(MarkupVertex { position: [center[0] - s, center[1] + s, 0.0], color });
        }
        MarkupElement::Leader { anchor, label_pos, color, .. } => {
            out.push(MarkupVertex { position: [anchor[0], anchor[1], 0.0], color });
            out.push(MarkupVertex { position: [label_pos[0], label_pos[1], 0.0], color });
            let r = 2.0;
            let n = 8;
            let mut prev = [anchor[0] + r, anchor[1]];
            for i in 1..=n {
                let a = (i as f32 / n as f32) * std::f32::consts::TAU;
                let curr = [anchor[0] + r * a.cos(), anchor[1] + r * a.sin()];
                out.push(MarkupVertex { position: [prev[0], prev[1], 0.0], color });
                out.push(MarkupVertex { position: [curr[0], curr[1], 0.0], color });
                prev = curr;
            }
        }
        MarkupElement::Callout { anchor, label_pos, radius, color, .. } => {
            out.push(MarkupVertex { position: [anchor[0], anchor[1], 0.0], color });
            out.push(MarkupVertex { position: [label_pos[0], label_pos[1], 0.0], color });
            let n = 8;
            let sr = 2.0;
            let mut prev = [anchor[0] + sr, anchor[1]];
            for i in 1..=n {
                let a = (i as f32 / n as f32) * std::f32::consts::TAU;
                let curr = [anchor[0] + sr * a.cos(), anchor[1] + sr * a.sin()];
                out.push(MarkupVertex { position: [prev[0], prev[1], 0.0], color });
                out.push(MarkupVertex { position: [curr[0], curr[1], 0.0], color });
                prev = curr;
            }
            let n_seg = 32usize;
            let mut prev_b = [label_pos[0] + radius, label_pos[1]];
            for i in 1..=n_seg {
                let a = (i as f32 / n_seg as f32) * std::f32::consts::TAU;
                let curr = [label_pos[0] + radius * a.cos(), label_pos[1] + radius * a.sin()];
                out.push(MarkupVertex { position: [prev_b[0], prev_b[1], 0.0], color });
                out.push(MarkupVertex { position: [curr[0], curr[1], 0.0], color });
                prev_b = curr;
            }
        }
        MarkupElement::Text { .. } => {
            // Text elements are deferred to HUD/glyphon layer.
        }
    }
}

/// Build an orthographic projection that maps pixel coordinates [0..w, 0..h]
/// to NDC [-1..1] with Y flipped (screen-space: origin top-left, Y down).
fn screen_space_ortho(w: f32, h: f32) -> glam::Mat4 {
    glam::Mat4::from_cols(
        glam::Vec4::new(2.0 / w, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, -2.0 / h, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
        glam::Vec4::new(-1.0, 1.0, 0.0, 1.0),
    )
}

/// Project a 3D world-space point to 2D screen coordinates.
fn project_point(
    pos: glam::Vec3,
    model: glam::Mat4,
    view: glam::Mat4,
    proj: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
) -> Option<[f32; 2]> {
    let clip = proj * view * model * pos.extend(1.0);
    if clip.w.abs() < 0.0001 {
        return None;
    }
    let ndc = glam::vec3(clip.x, clip.y, clip.z) / clip.w;
    // Clip check: skip points behind camera or beyond far plane
    if depth_reversed_z {
        if ndc.z > 1.0 || ndc.z < 0.0 { return None; }
    } else {
        if ndc.z < 0.0 || ndc.z > 1.0 { return None; }
    }
    Some([
        (ndc.x * 0.5 + 0.5) * screen_w,
        (0.5 - ndc.y * 0.5) * screen_h,
    ])
}

/// Project 3D annotation elements to 2D MarkupVertex.
fn project_annotation_elements(
    elements: &[super::pass_effects::ProjectedAnnotation],
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
                // Use XZ-plane diagonals with a small Y component for visibility from all angles
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

/// Render markup lines as an overlay.
pub fn pass_markup(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    surface_w: u32,
    surface_h: u32,
    view_matrix: glam::Mat4,
    projection_matrix: glam::Mat4,
    depth_reversed_z: bool,
    effect_commands: &super::pass_effects::EffectCommands,
) {
    // ── Combine 2D screen-space markup + projected 3D annotations ──
    let projected = project_annotation_elements(
        &effect_commands.annotation_elements,
        view_matrix,
        projection_matrix,
        surface_w as f32,
        surface_h as f32,
        depth_reversed_z,
    );
    let combined = if projected.is_empty() {
        std::borrow::Cow::Borrowed(&renderer.frame.markup_vertices)
    } else {
        let mut v = renderer.frame.markup_vertices.clone();
        v.extend_from_slice(&projected);
        std::borrow::Cow::Owned(v)
    };

    if combined.is_empty() {
        return;
    }

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Markup Overlay"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_pipeline(&renderer.gpu.pipelines.markup_lines);

    let mvp = screen_space_ortho(surface_w as f32, surface_h as f32).to_cols_array_2d();
    let uniforms = crate::vertex::FlatUniforms {
        mvp,
        color: [0.0; 4], // unused — fs_markup uses vertex colors
    };

    if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
        let vb = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("markup vb"),
            contents: bytemuck::cast_slice(&combined),
            usage: wgpu::BufferUsages::VERTEX,
        });
        pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.draw(0..combined.len() as u32, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_scene::node_data::{MarkupElement, MarkupNode, NodeData};
    use rc3d_scene::SceneGraph;

    fn make_markup_graph(elements: Vec<MarkupElement>, visible: bool) -> (SceneGraph, NodeId) {
        let mut g = SceneGraph::new();
        let markup = MarkupNode {
            elements,
            visible,
            ..Default::default()
        };
        let id = g.add_root(NodeData::Markup(markup));
        (g, id)
    }

    // ── Empty / hidden ──

    #[test]
    fn test_empty_scene_yields_no_vertices() {
        let g = SceneGraph::new();
        let v = collect_markup_lines(&g, NodeId::default(), 800, 600);
        assert!(v.is_empty());
    }

    #[test]
    fn test_hidden_markup_yields_no_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Line {
                start: [0.0, 0.0],
                end: [10.0, 10.0],
                color: [1.0, 0.0, 0.0, 1.0],
                width: 1.0,
            }],
            false,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert!(v.is_empty());
    }

    // ── Line ──

    #[test]
    fn test_line_element_produces_two_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Line {
                start: [10.0, 20.0],
                end: [30.0, 40.0],
                color: [1.0, 0.0, 0.0, 0.8],
                width: 2.0,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].position, [10.0, 20.0, 0.0]);
        assert_eq!(v[1].position, [30.0, 40.0, 0.0]);
    }

    // ── Rect ──

    #[test]
    fn test_rect_element_produces_eight_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Rect {
                origin: [0.0, 0.0],
                size: [100.0, 50.0],
                color: [1.0, 0.0, 0.0, 0.6],
                filled: false,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // 4 edges × 2 vertices = 8
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_rect_covers_correct_corners() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Rect {
                origin: [50.0, 50.0],
                size: [100.0, 100.0],
                color: [1.0, 0.0, 0.0, 0.6],
                filled: false,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // Should have corners at (50,50), (150,50), (150,150), (50,150)
        let positions: Vec<[f32; 2]> = v.iter().map(|lv| [lv.position[0], lv.position[1]]).collect();
        assert!(positions.contains(&[50.0, 50.0]));
        assert!(positions.contains(&[150.0, 50.0]));
        assert!(positions.contains(&[150.0, 150.0]));
        assert!(positions.contains(&[50.0, 150.0]));
    }

    // ── Circle ──

    #[test]
    fn test_circle_produces_64_segment_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Circle {
                center: [400.0, 300.0],
                radius: 50.0,
                color: [1.0, 0.0, 0.0, 0.6],
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // 64 segments × 2 = 128 vertices
        assert_eq!(v.len(), 128);
    }

    #[test]
    fn test_circle_vertices_near_center_radius() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Circle {
                center: [0.0, 0.0],
                radius: 10.0,
                color: [1.0, 0.0, 0.0, 0.6],
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        for lv in &v {
            let dist = (lv.position[0].powi(2) + lv.position[1].powi(2)).sqrt();
            assert!((dist - 10.0).abs() < 0.2, "distance {} not ≈ 10", dist);
        }
    }

    // ── Freehand ──

    #[test]
    fn test_freehand_two_points_produces_two_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Freehand {
                points: vec![[0.0, 0.0], [10.0, 10.0]],
                color: [0.0, 1.0, 0.0, 0.8],
                width: 2.0,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_freehand_polyline_connects_points() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Freehand {
                points: vec![[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]],
                color: [0.0, 1.0, 0.0, 0.8],
                width: 2.0,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // 2 segments * 2 = 4 vertices
        assert_eq!(v.len(), 4);
        // First segment
        assert_eq!(v[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(v[1].position, [5.0, 5.0, 0.0]);
        // Second segment
        assert_eq!(v[2].position, [5.0, 5.0, 0.0]);
        assert_eq!(v[3].position, [10.0, 0.0, 0.0]);
    }

    #[test]
    fn test_freehand_single_point_no_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Freehand {
                points: vec![[0.0, 0.0]],
                color: [0.0, 1.0, 0.0, 0.8],
                width: 2.0,
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // windows(2) on a single element gives empty iterator
        assert!(v.is_empty());
    }

    // ── Dimension ──

    #[test]
    fn test_dimension_produces_six_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Dimension {
                start: [10.0, 10.0],
                end: [110.0, 10.0],
                offset_dir: [0.0, -1.0],
                extension_len: 20.0,
                arrow_size: 5.0,
                label: String::new(),
                color: [1.0, 1.0, 0.0, 1.0],
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        // 3 extension/dim lines + 4 arrow lines = 14 vertices
        assert!(v.len() >= 14, "expected >=14 vertices, got {}", v.len());
    }

    // ── Text (deferred) ──

    #[test]
    fn test_text_element_no_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Text {
                position: [0.0, 0.0],
                string: "hello".into(),
                size: 14.0,
                color: [1.0, 1.0, 1.0, 1.0],
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert!(v.is_empty());
    }

    // ── Multiple elements ──

    // ── Angle Dimension ──

    #[test]
    fn test_angle_dimension_produces_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::AngleDimension {
                center: [100.0, 100.0],
                arm1: [150.0, 100.0],
                arm2: [100.0, 50.0],
                radius: 40.0,
                color: [1.0, 1.0, 0.0, 1.0],
                label: "45°".into(),
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert!(v.len() > 0, "angle dimension should produce vertices");
    }

    #[test]
    fn test_radial_dimension_produces_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::RadialDimension {
                center: [100.0, 100.0],
                perimeter: [150.0, 100.0],
                color: [1.0, 1.0, 0.0, 1.0],
                label: "R=50".into(),
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_diameter_dimension_produces_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::DiameterDimension {
                p1: [50.0, 100.0],
                p2: [150.0, 100.0],
                center: [100.0, 100.0],
                color: [1.0, 1.0, 0.0, 1.0],
                label: "D=100".into(),
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert!(v.len() >= 6, "diameter dimension should have main line + cross mark");
    }

    #[test]
    fn test_leader_produces_vertices() {
        let (g, root) = make_markup_graph(
            vec![MarkupElement::Leader {
                anchor: [50.0, 50.0],
                label_pos: [150.0, 100.0],
                text: "note".into(),
                color: [1.0, 1.0, 0.0, 1.0],
            }],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert!(v.len() >= 2, "leader line + dot should produce vertices");
    }

    #[test]
    fn test_multiple_elements_combined() {
        let (g, root) = make_markup_graph(
            vec![
                MarkupElement::Line {
                    start: [0.0, 0.0],
                    end: [10.0, 10.0],
                    color: [1.0, 0.0, 0.0, 0.8],
                    width: 1.0,
                },
                MarkupElement::Line {
                    start: [10.0, 10.0],
                    end: [20.0, 20.0],
                    color: [1.0, 0.0, 0.0, 0.8],
                    width: 1.0,
                },
            ],
            true,
        );
        let v = collect_markup_lines(&g, root, 800, 600);
        assert_eq!(v.len(), 4);
    }
}
