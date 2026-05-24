use crate::render_passes::pass_effects::{AnnotationVisibility, ProjectedAnnotation};
use crate::vertex::MarkupVertex;
use crate::world_label::{
    angle_dimension_label_basis, extent_label_basis, leader_label_basis,
    linear_dimension_label_basis, normalize3, resolve_label_height_world, WorldLabelCommand,
};
use glam::Vec3;
use glyphon::{Attrs, Buffer, FontSystem, Metrics, Shaping};
use rc3d_scene::annotation::{
    angle_dimension_label_point, angle_dimension_lines, angle_degrees, datum_cross_points,
    diameter_dimension_points, distance_3d, linear_dimension_points, radial_dimension_points,
    resolve_angle_label, resolve_diameter_label, resolve_length_label, resolve_radius_label,
    AnnotationLabelMode, AnnotationStyle,
};
use rc3d_scene::node_data::{AnnotationElement, DatumTargetType, GdtMaterialCondition, GdtSymbol};

use super::projection::{leader_label_local_3d, project_point_ndc};

/// Per-frame context for world label height (screen-pixel floor).
struct LabelCtx<'a> {
    model: glam::Mat4,
    scene_vp: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
    _marker: std::marker::PhantomData<&'a ()>,
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

fn push_world_label(
    labels: &mut Vec<WorldLabelCommand>,
    ctx: &LabelCtx<'_>,
    style: &AnnotationStyle,
    extension_len: f32,
    text: String,
    at: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
    color: [f32; 4],
) {
    if text.is_empty() {
        return;
    }
    let tangent = normalize3(tangent, [1.0, 0.0, 0.0]);
    let bitangent = normalize3(bitangent, [0.0, 1.0, 0.0]);
    let height_world = resolve_label_height_world(
        extension_len,
        style.label_height_factor,
        style.font_size,
        at,
        bitangent,
        ctx.model,
        ctx.scene_vp,
        ctx.screen_w,
        ctx.screen_h,
        ctx.depth_reversed_z,
    );
    labels.push(WorldLabelCommand {
        string: text,
        model_matrix: ctx.model,
        at,
        tangent,
        bitangent,
        height_world,
        screen_height_px: style.font_size,
        color,
    });
}

/// Measure text width in world-space units given font size in screen pixels.
/// Returns (world_width, world_height) or None if measurement fails.
fn measure_text_world(
    text: &str,
    font_size: f32,
    font_system: &mut FontSystem,
    label_attrs: Attrs<'_>,
    at: [f32; 3],
    up_axis: [f32; 3],
    model: glam::Mat4,
    scene_vp: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
    extension_len: f32,
    label_height_factor: f32,
) -> Option<(f32, f32)> {
    let mut buffer = Buffer::new(font_system, Metrics::new(font_size, 1.0));
    buffer.set_text(font_system, text, label_attrs, Shaping::Advanced);
    buffer.shape_until_scroll(font_system, false);

    let max_line_w = buffer
        .layout_runs()
        .map(|run| run.line_w)
        .fold(0.0f32, |a, b| if a > b { a } else { b });

    if max_line_w <= 0.0 {
        return None;
    }

    let world_h = resolve_label_height_world(
        extension_len,
        label_height_factor,
        font_size,
        at,
        up_axis,
        model,
        scene_vp,
        screen_w,
        screen_h,
        depth_reversed_z,
    );
    let world_w = world_h * (max_line_w / font_size);

    Some((world_w, world_h))
}

fn draw_linear_dimension(
    out: &mut Vec<MarkupVertex>,
    labels: &mut Vec<WorldLabelCommand>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    ctx: &LabelCtx<'_>,
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
    let label_at = offset_point_along(offset_dir, mid_3d, extension_len * 0.04);
    let tangent = tangent_xyz(points[4], points[5]);
    let (_, bitangent) = linear_dimension_label_basis(start, end, offset_dir);
    push_world_label(
        labels,
        ctx,
        style,
        extension_len,
        text,
        label_at,
        tangent,
        bitangent,
        color,
    );
}

fn draw_angle_dimension(
    out: &mut Vec<MarkupVertex>,
    labels: &mut Vec<WorldLabelCommand>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    ctx: &LabelCtx<'_>,
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
    let (tangent, bitangent) = angle_dimension_label_basis(center, arm1, arm2);
    push_world_label(
        labels,
        ctx,
        style,
        style.extension_len,
        text,
        label_pt,
        tangent,
        bitangent,
        color,
    );
}

fn draw_radial_dimension(
    out: &mut Vec<MarkupVertex>,
    labels: &mut Vec<WorldLabelCommand>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    ctx: &LabelCtx<'_>,
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
    let (tangent, bitangent) = extent_label_basis(center, perimeter);
    push_world_label(
        labels,
        ctx,
        style,
        style.extension_len,
        text,
        mid,
        tangent,
        bitangent,
        color,
    );
}

fn draw_diameter_dimension(
    out: &mut Vec<MarkupVertex>,
    labels: &mut Vec<WorldLabelCommand>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    ctx: &LabelCtx<'_>,
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
    let (tangent, bitangent) = extent_label_basis(p1, p2);
    push_world_label(
        labels,
        ctx,
        style,
        style.extension_len,
        text,
        mid,
        tangent,
        bitangent,
        color,
    );
}

fn draw_leader(
    out: &mut Vec<MarkupVertex>,
    labels: &mut Vec<WorldLabelCommand>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    ctx: &LabelCtx<'_>,
    anchor: [f32; 3],
    label_offset: [f32; 2],
    leader_offset_scale: f32,
    text: &str,
    color: [f32; 4],
    style: &AnnotationStyle,
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
        let leader_tangent = tangent_xyz(anchor, label_3d);
        let (tangent, bitangent) = leader_label_basis(leader_tangent);
        push_world_label(
            labels,
            ctx,
            style,
            style.extension_len,
            text.to_string(),
            label_3d,
            tangent,
            bitangent,
            color,
        );
    }
}

fn gdt_symbol_text(sym: GdtSymbol) -> &'static str {
    match sym {
        GdtSymbol::Straightness => "\u{2014}",
        GdtSymbol::Flatness => "\u{230F}",
        GdtSymbol::Circularity => "\u{25CB}",
        GdtSymbol::Cylindricity => "\u{232D}",
        GdtSymbol::ProfileOfLine => "\u{2312}",
        GdtSymbol::ProfileOfSurface => "\u{2313}",
        GdtSymbol::Angularity => "\u{2220}",
        GdtSymbol::Perpendicularity => "\u{22A5}",
        GdtSymbol::Parallelism => "\u{2225}",
        GdtSymbol::Position => "\u{2316}",
        GdtSymbol::Concentricity => "\u{25CE}",
        GdtSymbol::Symmetry => "\u{232F}",
        GdtSymbol::CircularRunout => "R",
        GdtSymbol::TotalRunout => "TR",
    }
}

fn gdt_mc_text(mc: GdtMaterialCondition) -> &'static str {
    match mc {
        GdtMaterialCondition::MaximumMaterial => "\u{24C2}",
        GdtMaterialCondition::LeastMaterial => "\u{24C1}",
        GdtMaterialCondition::RegardlessOfFeature => "\u{24C8}",
    }
}

/// Draw a wireframe rectangle with a vertical compartment divider for FCF / datum area.
fn draw_fcf_frame(
    out: &mut Vec<MarkupVertex>,
    proj: &impl Fn(&[f32; 3]) -> Option<[f32; 3]>,
    center: [f32; 3],
    half_w: f32,
    half_h: f32,
    color: [f32; 4],
) {
    let corners = [
        [center[0] - half_w, center[1] - half_h, center[2]], // BL
        [center[0] + half_w, center[1] - half_h, center[2]], // BR
        [center[0] + half_w, center[1] + half_h, center[2]], // TR
        [center[0] - half_w, center[1] + half_h, center[2]], // TL
    ];
    for i in 0..4 {
        let j = (i + 1) % 4;
        push_segment_3d(out, proj, corners[i], corners[j], color);
    }
    // Vertical divider between symbol and tolerance compartments
    let div_start = [
        center[0] - half_w * 0.3,
        center[1] - half_h,
        center[2],
    ];
    let div_end = [
        center[0] - half_w * 0.3,
        center[1] + half_h,
        center[2],
    ];
    push_segment_3d(out, proj, div_start, div_end, color);
}

/// Collect all key point coordinates for an annotation element.
fn element_key_points(element: &AnnotationElement) -> Vec<[f32; 3]> {
    match element {
        AnnotationElement::Dimension { start, end, offset_dir, .. } => {
            let s = glam::Vec3::from(start.coords());
            let e = glam::Vec3::from(end.coords());
            vec![start.coords(), end.coords(), ((s + e) * 0.5 + glam::Vec3::from(*offset_dir)).into()]
        }
        AnnotationElement::AngleDimension { center, arm1, arm2, .. } => {
            vec![center.coords(), arm1.coords(), arm2.coords()]
        }
        AnnotationElement::RadialDimension { center, perimeter, .. } => {
            vec![center.coords(), perimeter.coords()]
        }
        AnnotationElement::DiameterDimension { center, p1, p2, .. } => {
            vec![center.coords(), p1.coords(), p2.coords()]
        }
        AnnotationElement::Leader { anchor, .. } => vec![anchor.coords()],
        AnnotationElement::Callout { anchor, .. } => vec![anchor.coords()],
        AnnotationElement::Datum { position, .. } => vec![position.coords()],
        AnnotationElement::GdtFeatureControlFrame { position, leader_target, .. } => {
            let mut pts = vec![position.coords()];
            if let Some(lt) = leader_target {
                pts.push(lt.coords());
            }
            pts
        }
        AnnotationElement::GdtDatumTarget { position, .. } => vec![position.coords()],
        AnnotationElement::ChamferDimension { start, end, offset_dir, .. } => {
            let s = glam::Vec3::from(start.coords());
            let e = glam::Vec3::from(end.coords());
            vec![start.coords(), end.coords(), ((s + e) * 0.5 + glam::Vec3::from(*offset_dir)).into()]
        }
        AnnotationElement::OrdinateDimension { feature, datum, .. } => {
            vec![feature.coords(), datum.coords()]
        }
        AnnotationElement::SurfaceFinish { position, .. } => vec![position.coords()],
        AnnotationElement::WeldSymbol { position, .. } => vec![position.coords()],
        AnnotationElement::DatumIdentifier { position, .. } => vec![position.coords()],
    }
}

/// Get the first key point's NDC (for depth sampling).
fn element_any_ndc_point(
    element: &AnnotationElement,
    model: glam::Mat4,
    scene_vp: glam::Mat4,
    depth_reversed_z: bool,
) -> Option<[f32; 3]> {
    for p in element_key_points(element) {
        if let Some(ndc) = project_point_ndc(glam::Vec3::from(p), model, scene_vp, depth_reversed_z) {
            return Some(ndc);
        }
    }
    None
}

/// Check if any key point of the element is within NDC bounds.
fn element_any_point_visible(
    element: &AnnotationElement,
    model: glam::Mat4,
    scene_vp: glam::Mat4,
    depth_reversed_z: bool,
) -> bool {
    element_key_points(element).iter().any(|p| {
        project_point_ndc(glam::Vec3::from(*p), model, scene_vp, depth_reversed_z).is_some()
    })
}

/// Check if a point at NDC `p` is occluded by sampling the depth buffer.
/// Returns true if the depth buffer has a closer value than `p.z`.
fn is_ndc_occluded(
    p: [f32; 3],
    depth_buf: &[f32],
    buf_w: u32,
    buf_h: u32,
    depth_reversed_z: bool,
) -> bool {
    let px = ((p[0] * 0.5 + 0.5) * buf_w as f32) as i32;
    let py = ((0.5 - p[1] * 0.5) * buf_h as f32) as i32;
    if px < 0 || py < 0 { return false; }
    let (px, py) = (px as u32, py as u32);
    if px >= buf_w || py >= buf_h { return false; }
    let sampled = depth_buf[(py * buf_w + px) as usize];
    if depth_reversed_z {
        sampled > p[2] // reversed: closer = larger z
    } else {
        sampled < p[2] // normal: closer = smaller z
    }
}

/// Project 3D annotation elements to overlay lines and world-space labels.
pub(super) fn project_annotation_elements(
    elements: &[ProjectedAnnotation],
    scene_vp: glam::Mat4,
    screen_w: f32,
    screen_h: f32,
    depth_reversed_z: bool,
    occlusion: Option<(&[f32], u32, u32)>, // (depth_buf, buf_w, buf_h)
    mut font_system: Option<(&mut FontSystem, Attrs<'_>)>,
    labels: &mut Vec<WorldLabelCommand>,
) -> Vec<MarkupVertex> {
    let mut out = Vec::new();

    for pa in elements {
        let model = pa.model_matrix;
        let style = &pa.style;

        // --- visibility culling ---
        // Back-face culling skipped: annotation plane normal depends on arbitrary
        // point ordering (start/end, offset sign). NDC culling alone is sufficient.
        // --- visibility culling ---
        let mut visibility = AnnotationVisibility::default();
        visibility.outside_ndc =
            !element_any_point_visible(&pa.element, model, scene_vp, depth_reversed_z);
        // Occlusion: sample depth buffer at anchor NDC
        if let Some((depth_buf, buf_w, buf_h)) = occlusion {
            if let Some(anchor_ndc) = element_any_ndc_point(&pa.element, model, scene_vp, depth_reversed_z) {
                visibility.occluded = is_ndc_occluded(anchor_ndc, depth_buf, buf_w, buf_h, depth_reversed_z);
            }
        }
        if visibility.outside_ndc || visibility.occluded {
            continue;
        }
        // --- end culling ---

        let proj_ndc = |p: &[f32; 3]| {
            project_point_ndc(
                Vec3::new(p[0], p[1], p[2]),
                model,
                scene_vp,
                depth_reversed_z,
            )
        };
        let ctx = LabelCtx {
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
                    &ctx,
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
                &ctx,
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
                    &ctx,
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
                    &ctx,
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
                &ctx,
                anchor.coords(),
                *label_offset,
                style.leader_offset_scale,
                text,
                *color,
                style,
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
                &ctx,
                anchor.coords(),
                *label_offset,
                style.leader_offset_scale,
                text,
                *color,
                style,
                Some(*radius),
                style.arc_segments,
            ),
            AnnotationElement::Datum { position, size, color } => {
                let color = *color;
                for (a, b) in datum_cross_points(position.coords(), *size) {
                    push_segment_3d(&mut out, &proj_ndc, a, b, color);
                }
            }
            AnnotationElement::GdtFeatureControlFrame {
                symbol,
                tolerance,
                diameter,
                datum_primary,
                datum_secondary,
                material_condition,
                position,
                leader_target,
                color,
            } => {
                let frame_pos = position.coords();

                // Build text: e.g. "⌓ Ø0.05 Ⓜ A B"
                let sym_str = gdt_symbol_text(*symbol);
                let mut fcf_text = if *diameter {
                    format!("{} Ø{:.2}", sym_str, tolerance)
                } else {
                    format!("{} {:.2}", sym_str, tolerance)
                };
                if let Some(ref mc) = material_condition {
                    fcf_text.push_str(&format!(" {}", gdt_mc_text(*mc)));
                }
                if let Some(ref d1) = datum_primary {
                    fcf_text.push_str(&format!(" {}", d1));
                }
                if let Some(ref d2) = datum_secondary {
                    fcf_text.push_str(&format!(" {}", d2));
                }

                // Leader line from frame to feature
                if let Some(ref target) = leader_target {
                    push_segment_3d(&mut out, &proj_ndc, frame_pos, target.coords(), *color);
                }

                // Frame sizing: try precise text measurement first, fallback to heuristic.
                let mut sized = false;
                if let Some((fs, attrs)) = font_system.as_mut() {
                    if let Some((text_w, text_h)) = measure_text_world(
                        &fcf_text,
                        style.font_size,
                        fs,
                        *attrs,
                        frame_pos,
                        [0.0, 1.0, 0.0],
                        ctx.model,
                        ctx.scene_vp,
                        ctx.screen_w,
                        ctx.screen_h,
                        ctx.depth_reversed_z,
                        style.extension_len,
                        style.label_height_factor,
                    ) {
                        let frame_hh = text_h * 0.55;
                        let frame_hw = text_w * 0.5 + text_h * 0.25;
                        draw_fcf_frame(&mut out, &proj_ndc, frame_pos, frame_hw, frame_hh, *color);
                        sized = true;
                    }
                }
                if !sized {
                    // Fallback: heuristic sizing from character count
                    let font_h_world = resolve_label_height_world(
                        style.extension_len,
                        style.label_height_factor,
                        style.font_size,
                        frame_pos,
                        [0.0, 1.0, 0.0], // vertical axis
                        ctx.model,
                        ctx.scene_vp,
                        ctx.screen_w,
                        ctx.screen_h,
                        ctx.depth_reversed_z,
                    );
                    let char_count = fcf_text.chars().count().max(1) as f32;
                    let frame_hh = font_h_world * 0.6;
                    let frame_hw = font_h_world * char_count * 0.3 + font_h_world * 0.3;
                    draw_fcf_frame(&mut out, &proj_ndc, frame_pos, frame_hw, frame_hh, *color);
                }

                push_world_label(
                    labels,
                    &ctx,
                    style,
                    style.extension_len,
                    fcf_text,
                    frame_pos,
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    *color,
                );
            }
            AnnotationElement::GdtDatumTarget {
                position,
                label,
                target_type,
                size,
                color,
            } => {
                let pos = position.coords();
                match target_type {
                    DatumTargetType::Point => {
                        let r = *size * 0.5;
                        push_circle_3d(&mut out, &proj_ndc, pos, r, *color, 16);
                        // Cross within circle
                        push_segment_3d(
                            &mut out,
                            &proj_ndc,
                            [pos[0] - r, pos[1], pos[2]],
                            [pos[0] + r, pos[1], pos[2]],
                            *color,
                        );
                        push_segment_3d(
                            &mut out,
                            &proj_ndc,
                            [pos[0], pos[1] - r, pos[2]],
                            [pos[0], pos[1] + r, pos[2]],
                            *color,
                        );
                    }
                    DatumTargetType::Line => {
                        push_segment_3d(
                            &mut out,
                            &proj_ndc,
                            [pos[0] - *size, pos[1], pos[2]],
                            [pos[0] + *size, pos[1], pos[2]],
                            *color,
                        );
                        push_circle_3d(
                            &mut out,
                            &proj_ndc,
                            [pos[0] - *size, pos[1], pos[2]],
                            *size * 0.1,
                            *color,
                            8,
                        );
                        push_circle_3d(
                            &mut out,
                            &proj_ndc,
                            [pos[0] + *size, pos[1], pos[2]],
                            *size * 0.1,
                            *color,
                            8,
                        );
                    }
                    DatumTargetType::Area => {
                        let r = *size * 0.5;
                        // Hatched rectangle
                        draw_fcf_frame(&mut out, &proj_ndc, pos, r, r, *color);
                        // Diagonal hatch lines
                        for i in 0..4 {
                            let off = (i as f32 - 1.5) * r * 0.5;
                            push_segment_3d(
                                &mut out,
                                &proj_ndc,
                                [pos[0] - r + off, pos[1] - r, pos[2]],
                                [pos[0] + r + off, pos[1] + r, pos[2]],
                                *color,
                            );
                        }
                    }
                }
                push_world_label(
                    labels,
                    &ctx,
                    style,
                    style.extension_len,
                    label.clone(),
                    pos,
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    *color,
                );
            }
            AnnotationElement::ChamferDimension {
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
                let formatted_label = if label.is_empty() {
                    "C".to_string()
                } else {
                    format!("C{}", label)
                };
                draw_linear_dimension(
                    &mut out,
                    labels,
                    &proj_ndc,
                    &ctx,
                    start.coords(),
                    end.coords(),
                    *offset_dir,
                    eff.extension_len,
                    eff.arrow_size,
                    &formatted_label,
                    label_mode,
                    *color,
                    &eff,
                );
            }
            AnnotationElement::OrdinateDimension {
                feature,
                datum,
                axis_dir,
                jog_length,
                offset,
                label,
                label_mode,
                color,
            } => {
                let f = feature.coords();
                let d = datum.coords();
                let axis = Vec3::from(*axis_dir);
                let perp = if axis.x.abs() > 0.5 {
                    Vec3::new(0.0, offset.signum(), 0.0)
                } else {
                    Vec3::new(offset.signum(), 0.0, 0.0)
                };
                let jog_start = [
                    d[0] + axis.x * *jog_length,
                    d[1] + axis.y * *jog_length,
                    d[2] + axis.z * *jog_length,
                ];
                let jog_mid = [
                    jog_start[0] + perp.x,
                    jog_start[1] + perp.y,
                    jog_start[2] + perp.z,
                ];
                let label_pt = [
                    jog_mid[0] + axis.x * 0.1,
                    jog_mid[1] + axis.y * 0.1,
                    jog_mid[2] + axis.z * 0.1,
                ];

                // Jogged leader
                push_segment_3d(&mut out, &proj_ndc, f, jog_start, *color);
                push_segment_3d(&mut out, &proj_ndc, jog_start, jog_mid, *color);
                // Short horizontal extension at label
                push_segment_3d(
                    &mut out,
                    &proj_ndc,
                    jog_mid,
                    [
                        jog_mid[0] + axis.x * 0.3,
                        jog_mid[1] + axis.y * 0.3,
                        jog_mid[2] + axis.z * 0.3,
                    ],
                    *color,
                );

                let dist = distance_3d(f, d);
                let text = resolve_length_label(label, label_mode, dist, style);
                push_world_label(
                    labels,
                    &ctx,
                    style,
                    style.extension_len,
                    text,
                    label_pt,
                    axis.into(),
                    [perp.y, -perp.x, 0.0],
                    *color,
                );
            }
            // New annotation types -- rendering to be added later
            AnnotationElement::SurfaceFinish { .. } => {}
            AnnotationElement::WeldSymbol { .. } => {}
            AnnotationElement::DatumIdentifier { .. } => {}
        }
    }

    out
}
