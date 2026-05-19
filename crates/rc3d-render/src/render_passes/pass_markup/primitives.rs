use crate::render_passes::pass_effects::ProjectedAnnotation;
use crate::render_passes::pass_text::TextDrawCommand;
use crate::vertex::MarkupVertex;
use glam::Vec3;
use rc3d_scene::annotation::{
    angle_dimension_label_point, angle_dimension_lines, angle_degrees, datum_cross_points,
    diameter_dimension_points, distance_3d, linear_dimension_points, radial_dimension_points,
    resolve_angle_label, resolve_diameter_label, resolve_length_label, resolve_radius_label,
    AnnotationLabelMode, AnnotationStyle,
};
use rc3d_scene::node_data::AnnotationElement;

use super::projection::{
    leader_label_local_3d, ndc_to_screen, project_point_ndc, screen_baseline_from_model_tangent,
};

/// Per-frame label projection (screen position + plane tangent baseline).
struct LabelProj<'a> {
    model: glam::Mat4,
    scene_vp: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl LabelProj<'_> {
    /// Same `scene_vp * model` path as annotation lines (NDC, then to pixels).
    fn screen_pos(&self, p: &[f32; 3]) -> Option<[f32; 2]> {
        let ndc = project_point_ndc(
            glam::Vec3::from(*p),
            self.model,
            self.scene_vp,
            self.depth_reversed_z,
        )?;
        Some(ndc_to_screen(ndc, self.screen_w, self.screen_h))
    }

    fn baseline_from_tangent(&self, at: [f32; 3], tangent: [f32; 3]) -> f32 {
        screen_baseline_from_model_tangent(
            glam::Vec3::from(at),
            glam::Vec3::from(tangent),
            self.model,
            self.scene_vp,
            self.screen_w,
            self.screen_h,
            self.depth_reversed_z,
        )
    }
}

fn offset_point_along(dir: [f32; 3], from: [f32; 3], distance: f32) -> [f32; 3] {
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt().max(1e-6);
    let s = distance / len;
    [
        from[0] + dir[0] * s,
        from[1] + dir[1] * s,
        from[2] + dir[2] * s,
    ]
}

fn tangent_xyz(from: [f32; 3], to: [f32; 3]) -> [f32; 3] {
    [to[0] - from[0], to[1] - from[1], to[2] - from[2]]
}

fn push_line_ndc(out: &mut Vec<MarkupVertex>, a: [f32; 3], b: [f32; 3], color: [f32; 4]) {
    out.push(MarkupVertex { position: a, color });
    out.push(MarkupVertex { position: b, color });
}

fn push_segment_3d(
    out: &mut Vec<MarkupVertex>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    a: [f32; 3],
    b: [f32; 3],
    color: [f32; 4],
) -> bool {
    let (Some(sa), Some(sb)) = (proj(&a), proj(&b)) else {
        return false;
    };
    push_line_ndc(out, sa, sb, color);
    true
}

fn project_points_3d(
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    points: &[[f32; 3]],
) -> Option<Vec<[f32; 3]>> {
    let mut ndc = Vec::with_capacity(points.len());
    for p in points {
        ndc.push(proj(p)?);
    }
    Some(ndc)
}

fn push_circle_3d(
    out: &mut Vec<MarkupVertex>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    center: [f32; 3],
    radius: f32,
    color: [f32; 4],
    segments: u32,
) {
    let n = segments.max(8);
    let mut prev = proj(&[center[0] + radius, center[1], center[2]]);
    for i in 1..=n {
        let a = (i as f32 / n as f32) * std::f32::consts::TAU;
        let p = [
            center[0] + radius * a.cos(),
            center[1] + radius * a.sin(),
            center[2],
        ];
        let curr = proj(&p);
        if let (Some(pa), Some(pb)) = (prev, curr) {
            push_line_ndc(out, pa, pb, color);
        }
        prev = curr;
    }
}

fn push_plane_label(
    labels: &mut Vec<TextDrawCommand>,
    text: String,
    screen_pos: [f32; 2],
    baseline_angle_rad: f32,
    color: [f32; 4],
    font_size: f32,
) {
    if text.is_empty() {
        return;
    }
    labels.push(TextDrawCommand {
        string: text,
        screen_pos,
        size: font_size,
        color,
        is_3d: false,
        plane_aligned: true,
        baseline_angle_rad,
    });
}

/// Label at a model-local point; baseline follows `tangent` on the annotation plane.
fn push_model_plane_label(
    labels: &mut Vec<TextDrawCommand>,
    lp: &LabelProj<'_>,
    text: String,
    at: [f32; 3],
    tangent: [f32; 3],
    color: [f32; 4],
    font_size: f32,
) {
    let Some(screen_pos) = lp.screen_pos(&at) else {
        return;
    };
    let baseline = lp.baseline_from_tangent(at, tangent);
    push_plane_label(labels, text, screen_pos, baseline, color, font_size);
}

fn draw_linear_dimension(
    out: &mut Vec<MarkupVertex>,
    labels: &mut Vec<TextDrawCommand>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    lp: &LabelProj<'_>,
    start: [f32; 3],
    end: [f32; 3],
    offset_dir: [f32; 3],
    extension_len: f32,
    arrow_size: f32,
    label: &str,
    label_mode: &AnnotationLabelMode,
    color: [f32; 4],
    style: &AnnotationStyle,
) {
    let points = linear_dimension_points(start, end, offset_dir, extension_len, arrow_size);
    let Some(v) = project_points_3d(proj, &points) else {
        return;
    };
    if v.len() < 12 {
        return;
    }
    let ndc: [[f32; 3]; 12] = std::array::from_fn(|i| v[i]);

    push_line_ndc(out, ndc[4], ndc[5], color);
    push_line_ndc(out, ndc[0], ndc[2], color);
    push_line_ndc(out, ndc[1], ndc[3], color);
    push_line_ndc(out, ndc[4], ndc[6], color);
    push_line_ndc(out, ndc[4], ndc[7], color);
    push_line_ndc(out, ndc[6], ndc[10], color);
    push_line_ndc(out, ndc[7], ndc[10], color);
    push_line_ndc(out, ndc[5], ndc[8], color);
    push_line_ndc(out, ndc[5], ndc[9], color);
    push_line_ndc(out, ndc[8], ndc[11], color);
    push_line_ndc(out, ndc[9], ndc[11], color);

    let dist = distance_3d(start, end);
    let text = resolve_length_label(label, label_mode, dist, style);
    let mid_3d = [
        (points[4][0] + points[5][0]) * 0.5,
        (points[4][1] + points[5][1]) * 0.5,
        (points[4][2] + points[5][2]) * 0.5,
    ];
    let label_at = offset_point_along(offset_dir, mid_3d, style.font_size * 0.012);
    let tangent = tangent_xyz(points[4], points[5]);
    push_model_plane_label(labels, lp, text, label_at, tangent, color, style.font_size);
}

fn draw_angle_dimension(
    out: &mut Vec<MarkupVertex>,
    labels: &mut Vec<TextDrawCommand>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    lp: &LabelProj<'_>,
    center: [f32; 3],
    arm1: [f32; 3],
    arm2: [f32; 3],
    radius: f32,
    label: &str,
    label_mode: &AnnotationLabelMode,
    color: [f32; 4],
    style: &AnnotationStyle,
) {
    for (a, b) in angle_dimension_lines(center, arm1, arm2, radius, style.arc_segments) {
        push_segment_3d(out, proj, a, b, color);
    }
    let deg = angle_degrees(center, arm1, arm2);
    let text = resolve_angle_label(label, label_mode, deg, style);
    let label_pt = angle_dimension_label_point(center, arm1, arm2, radius);
    let tangent = tangent_xyz(center, label_pt);
    push_model_plane_label(labels, lp, text, label_pt, tangent, color, style.font_size);
}

fn draw_radial_dimension(
    out: &mut Vec<MarkupVertex>,
    labels: &mut Vec<TextDrawCommand>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    lp: &LabelProj<'_>,
    center: [f32; 3],
    perimeter: [f32; 3],
    arrow_size: f32,
    label: &str,
    label_mode: &AnnotationLabelMode,
    color: [f32; 4],
    style: &AnnotationStyle,
) {
    let pts = radial_dimension_points(center, perimeter, arrow_size);
    let Some(scr) = project_points_3d(proj, &pts) else {
        return;
    };
    if scr.len() < 6 {
        return;
    }
    push_line_ndc(out, scr[0], scr[1], color);
    push_line_ndc(out, scr[2], scr[3], color);
    push_line_ndc(out, scr[4], scr[5], color);

    let r = distance_3d(center, perimeter);
    let text = resolve_radius_label(label, label_mode, r, style);
    let mid = [
        (center[0] + perimeter[0]) * 0.5,
        (center[1] + perimeter[1]) * 0.5,
        (center[2] + perimeter[2]) * 0.5,
    ];
    let tangent = tangent_xyz(center, perimeter);
    push_model_plane_label(labels, lp, text, mid, tangent, color, style.font_size);
}

fn draw_diameter_dimension(
    out: &mut Vec<MarkupVertex>,
    labels: &mut Vec<TextDrawCommand>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    lp: &LabelProj<'_>,
    center: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    arrow_size: f32,
    label: &str,
    label_mode: &AnnotationLabelMode,
    color: [f32; 4],
    style: &AnnotationStyle,
) {
    let pts = diameter_dimension_points(center, p1, p2, arrow_size);
    let Some(scr) = project_points_3d(proj, &pts) else {
        return;
    };
    if scr.len() < 10 {
        return;
    }
    push_line_ndc(out, scr[0], scr[1], color);
    push_line_ndc(out, scr[2], scr[3], color);
    push_line_ndc(out, scr[4], scr[5], color);
    push_line_ndc(out, scr[6], scr[7], color);
    push_line_ndc(out, scr[8], scr[9], color);

    let d = distance_3d(p1, p2);
    let text = resolve_diameter_label(label, label_mode, d, style);
    let mid = [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5, (p1[2] + p2[2]) * 0.5];
    let tangent = tangent_xyz(p1, p2);
    push_model_plane_label(labels, lp, text, mid, tangent, color, style.font_size);
}

fn draw_leader(
    out: &mut Vec<MarkupVertex>,
    labels: &mut Vec<TextDrawCommand>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    lp: &LabelProj<'_>,
    anchor: [f32; 3],
    label_offset: [f32; 2],
    leader_offset_scale: f32,
    text: &str,
    color: [f32; 4],
    font_size: f32,
    callout_radius: Option<f32>,
    arc_segments: u32,
) {
    let label_3d = leader_label_local_3d(anchor, label_offset, leader_offset_scale);
    let Some(ap) = proj(&anchor) else {
        return;
    };
    let Some(label_ndc) = proj(&label_3d) else {
        return;
    };
    push_line_ndc(out, ap, label_ndc, color);
    let dot_r = 0.03;
    push_circle_3d(out, proj, anchor, dot_r, color, 8);
    if let Some(r) = callout_radius {
        let world_r = r * leader_offset_scale;
        push_circle_3d(out, proj, label_3d, world_r, color, arc_segments);
    }
    if !text.is_empty() {
        let tangent = tangent_xyz(anchor, label_3d);
        push_model_plane_label(labels, lp, text.to_string(), label_3d, tangent, color, font_size);
    }
}

/// Project 3D annotation elements to overlay lines and plane-aligned labels.
pub(super) fn project_annotation_elements(
    elements: &[ProjectedAnnotation],
    scene_vp: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
    labels: &mut Vec<TextDrawCommand>,
) -> Vec<MarkupVertex> {
    let mut out = Vec::new();

    for pa in elements {
        let model = pa.model_matrix;
        let style = &pa.style;
        let proj_ndc = |p: &[f32; 3]| {
            project_point_ndc(
                Vec3::new(p[0], p[1], p[2]),
                model,
                scene_vp,
                depth_reversed_z,
            )
        };
        let lp = LabelProj {
            model,
            scene_vp,
            screen_w,
            screen_h,
            depth_reversed_z,
            _marker: std::marker::PhantomData,
        };

        match &pa.element {
            AnnotationElement::Dimension {
                start,
                end,
                offset_dir,
                extension_len,
                arrow_size,
                label,
                label_mode,
                color,
            } => {
                let eff = style.merge_with_element(Some(*extension_len), Some(*arrow_size));
                draw_linear_dimension(
                    &mut out,
                    labels,
                    &proj_ndc,
                    &lp,
                    start.coords(),
                    end.coords(),
                    *offset_dir,
                    eff.extension_len,
                    eff.arrow_size,
                    label,
                    label_mode,
                    *color,
                    &eff,
                );
            }
            AnnotationElement::AngleDimension {
                center,
                arm1,
                arm2,
                radius,
                label,
                label_mode,
                color,
            } => draw_angle_dimension(
                &mut out,
                labels,
                &proj_ndc,
                &lp,
                center.coords(),
                arm1.coords(),
                arm2.coords(),
                *radius,
                label,
                label_mode,
                *color,
                style,
            ),
            AnnotationElement::RadialDimension {
                center,
                perimeter,
                label,
                label_mode,
                arrow_size,
                color,
            } => {
                let eff = style.merge_with_element(None, Some(*arrow_size));
                draw_radial_dimension(
                    &mut out,
                    labels,
                    &proj_ndc,
                    &lp,
                    center.coords(),
                    perimeter.coords(),
                    eff.arrow_size,
                    label,
                    label_mode,
                    *color,
                    &eff,
                );
            }
            AnnotationElement::DiameterDimension {
                center,
                p1,
                p2,
                label,
                label_mode,
                arrow_size,
                color,
            } => {
                let eff = style.merge_with_element(None, Some(*arrow_size));
                draw_diameter_dimension(
                    &mut out,
                    labels,
                    &proj_ndc,
                    &lp,
                    center.coords(),
                    p1.coords(),
                    p2.coords(),
                    eff.arrow_size,
                    label,
                    label_mode,
                    *color,
                    &eff,
                );
            }
            AnnotationElement::Leader {
                anchor,
                label_offset,
                text,
                color,
            } => draw_leader(
                &mut out,
                labels,
                &proj_ndc,
                &lp,
                anchor.coords(),
                *label_offset,
                style.leader_offset_scale,
                text,
                *color,
                style.font_size,
                None,
                style.arc_segments,
            ),
            AnnotationElement::Callout {
                anchor,
                label_offset,
                text,
                radius,
                color,
            } => draw_leader(
                &mut out,
                labels,
                &proj_ndc,
                &lp,
                anchor.coords(),
                *label_offset,
                style.leader_offset_scale,
                text,
                *color,
                style.font_size,
                Some(*radius),
                style.arc_segments,
            ),
            AnnotationElement::Datum { position, size, color } => {
                let color = *color;
                for (a, b) in datum_cross_points(position.coords(), *size) {
                    push_segment_3d(&mut out, &proj_ndc, a, b, color);
                }
            }
        }
    }

    out
}
