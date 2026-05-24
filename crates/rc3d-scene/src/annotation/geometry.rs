//! 3D annotation geometry (single-plane, world units).

use rc3d_core::math::Vec3;

fn v3(p: [f32; 3]) -> Vec3 {
    Vec3::from(p)
}

fn to_arr(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.length_squared() > 1e-12 {
        v.normalize()
    } else {
        fallback
    }
}

/// Linear dimension: 12 key points (same layout as legacy `Dimension` renderer).
pub fn linear_dimension_points(
    start: [f32; 3],
    end: [f32; 3],
    offset_dir: [f32; 3],
    extension_len: f32,
    arrow_size: f32,
) -> [[f32; 3]; 12] {
    let dim_dir_3d = v3(end) - v3(start);
    let dim_len_3d = dim_dir_3d.length().max(0.001);
    let ndim_3d = dim_dir_3d / dim_len_3d;

    let off_len = v3(offset_dir).length().max(0.001);
    let off_n = v3(offset_dir) / off_len;

    let ext = extension_len;
    let asz = arrow_size;
    let ahw = asz * 0.4;

    let s = v3(start);
    let e = v3(end);
    let o = v3(offset_dir);

    [
        to_arr(s),
        to_arr(e),
        to_arr(s + o),
        to_arr(e + o),
        to_arr(s + o - ndim_3d * ext),
        to_arr(e + o + ndim_3d * ext),
        to_arr(s + o - ndim_3d * ext + ndim_3d * asz + off_n * ahw),
        to_arr(s + o - ndim_3d * ext + ndim_3d * asz - off_n * ahw),
        to_arr(e + o + ndim_3d * ext - ndim_3d * asz + off_n * ahw),
        to_arr(e + o + ndim_3d * ext - ndim_3d * asz - off_n * ahw),
        to_arr(s + o - ndim_3d * ext + ndim_3d * asz),
        to_arr(e + o + ndim_3d * ext - ndim_3d * asz),
    ]
}

/// Distance between two 3D points.
pub fn distance_3d(a: [f32; 3], b: [f32; 3]) -> f32 {
    v3(a).distance(v3(b))
}

/// Angle in degrees between rays `center->arm1` and `center->arm2`.
pub fn angle_degrees(center: [f32; 3], arm1: [f32; 3], arm2: [f32; 3]) -> f32 {
    let da = v3(arm1) - v3(center);
    let db = v3(arm2) - v3(center);
    if da.length_squared() < 1e-12 || db.length_squared() < 1e-12 {
        return 0.0;
    }
    let cos_a = da.normalize().dot(db.normalize()).clamp(-1.0, 1.0);
    cos_a.acos().to_degrees()
}

/// Line segments for an angular dimension (extension rays + arc).
pub fn angle_dimension_lines(
    center: [f32; 3],
    arm1: [f32; 3],
    arm2: [f32; 3],
    radius: f32,
    arc_segments: u32,
) -> Vec<([f32; 3], [f32; 3])> {
    let c = v3(center);
    let mut d1 = v3(arm1) - c;
    let mut d2 = v3(arm2) - c;
    if d1.length_squared() < 1e-12 || d2.length_squared() < 1e-12 {
        return Vec::new();
    }
    d1 = d1.normalize();
    d2 = d2.normalize();
    let r = radius.max(0.001);
    let n = d1.cross(d2);
    let plane_n = if n.length_squared() > 1e-12 {
        n.normalize()
    } else {
        Vec3::Y
    };

    let p1 = c + d1 * r;
    let p2 = c + d2 * r;
    let mut lines = vec![(to_arr(c), to_arr(p1)), (to_arr(c), to_arr(p2))];

    let mut prev = p1;
    let segs = arc_segments.max(4);
    for i in 1..=segs {
        let t = i as f32 / segs as f32;
        let dir = slerp_direction(d1, d2, t, plane_n);
        let curr = c + dir * r;
        lines.push((to_arr(prev), to_arr(curr)));
        prev = curr;
    }

    lines
}

/// Mid-arc point for label placement.
pub fn angle_dimension_label_point(
    center: [f32; 3],
    arm1: [f32; 3],
    arm2: [f32; 3],
    radius: f32,
) -> [f32; 3] {
    let c = v3(center);
    let d1 = normalize_or(v3(arm1) - c, Vec3::X);
    let d2 = normalize_or(v3(arm2) - c, Vec3::Y);
    let n = d1.cross(d2);
    let plane_n = if n.length_squared() > 1e-12 {
        n.normalize()
    } else {
        Vec3::Y
    };
    let mid = slerp_direction(d1, d2, 0.5, plane_n);
    to_arr(c + mid * radius.max(0.001))
}

fn slerp_direction(a: Vec3, b: Vec3, t: f32, axis: Vec3) -> Vec3 {
    let mut b = b;
    // Orient b to the same hemisphere as a around axis.
    if a.cross(b).dot(axis) < 0.0 {
        b = -b;
    }
    let omega = a.dot(b).clamp(-1.0, 1.0).acos();
    if omega < 1e-6 {
        return a;
    }
    let sin_o = omega.sin();
    let w0 = ((1.0 - t) * omega).sin() / sin_o;
    let w1 = (t * omega).sin() / sin_o;
    (a * w0 + b * w1).normalize()
}

/// Radial dimension: center to perimeter with arrowhead at perimeter.
pub fn radial_dimension_points(
    center: [f32; 3],
    perimeter: [f32; 3],
    arrow_size: f32,
) -> [[f32; 3]; 6] {
    let c = v3(center);
    let p = v3(perimeter);
    let dir = normalize_or(p - c, Vec3::X);
    let perp = normalize_or(dir.cross(Vec3::Y), Vec3::Z);
    let ahw = arrow_size * 0.4;
    let tip = p;
    let base = p - dir * arrow_size;
    [
        to_arr(c),
        to_arr(p),
        to_arr(base + perp * ahw),
        to_arr(tip),
        to_arr(base - perp * ahw),
        to_arr(tip),
    ]
}

/// Diameter dimension: line through `p1` and `p2` with arrows at both ends.
pub fn diameter_dimension_points(
    center: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    arrow_size: f32,
) -> [[f32; 3]; 10] {
    let _ = center;
    let a = v3(p1);
    let b = v3(p2);
    let dir = normalize_or(b - a, Vec3::X);
    let perp = normalize_or(dir.cross(Vec3::Y), Vec3::Z);
    let ahw = arrow_size * 0.4;

    let a_base = a + dir * arrow_size;
    let b_base = b - dir * arrow_size;

    [
        to_arr(a),
        to_arr(b),
        to_arr(a_base + perp * ahw),
        to_arr(a),
        to_arr(a_base - perp * ahw),
        to_arr(a),
        to_arr(b_base + perp * ahw),
        to_arr(b),
        to_arr(b_base - perp * ahw),
        to_arr(b),
    ]
}

/// Datum cross: two diagonals in a plane (uses fixed slight Y tilt for visibility).
pub fn datum_cross_points(position: [f32; 3], size: f32) -> [([f32; 3], [f32; 3]); 2] {
    let sz = size;
    let d1 = Vec3::new(sz * 0.707, sz * 0.1, sz * 0.707);
    let d2 = Vec3::new(sz * 0.707, sz * 0.1, -sz * 0.707);
    let pos = v3(position);
    let se = pos + d1;
    let nw = pos - d1;
    let ne = pos + d2;
    let sw = pos - d2;
    [(to_arr(nw), to_arr(se)), (to_arr(ne), to_arr(sw))]
}

/// Key points for a GD&T feature control frame (position, four corners for box outline).
pub fn gdt_fcf_key_points(position: [f32; 3], leader_target: Option<[f32; 3]>) -> Vec<[f32; 3]> {
    let mut pts = vec![position];
    if let Some(lt) = leader_target {
        pts.push(lt);
    }
    pts
}

/// Key points for a GD&T datum target (position center point).
pub fn gdt_datum_target_key_points(position: [f32; 3]) -> Vec<[f32; 3]> {
    vec![position]
}

/// Key points for a chamfer dimension (start, end, offset midpoint).
pub fn chamfer_dimension_key_points(
    start: [f32; 3],
    end: [f32; 3],
    offset_dir: [f32; 3],
) -> Vec<[f32; 3]> {
    let mid = to_arr(v3(start) + (v3(end) - v3(start)) * 0.5 + v3(offset_dir));
    vec![start, end, mid]
}

/// Key points for an ordinate dimension (feature, datum, jog corner).
pub fn ordinate_dimension_key_points(
    feature: [f32; 3],
    datum: [f32; 3],
    axis_dir: [f32; 3],
    jog_length: f32,
    offset: f32,
) -> Vec<[f32; 3]> {
    let axis = normalize_or(v3(axis_dir), Vec3::X);
    let perp = normalize_or(axis.cross(Vec3::Y), Vec3::Z);
    let jog_corner = v3(feature) + axis * jog_length + perp * offset;
    vec![feature, datum, to_arr(jog_corner)]
}

/// Key points for a surface finish annotation (position only).
pub fn surface_finish_key_points(
    position: [f32; 3],
    _direction: [f32; 3],
) -> Vec<[f32; 3]> {
    vec![position]
}

/// Key points for a weld symbol annotation (position only).
pub fn weld_symbol_key_points(
    position: [f32; 3],
    _arrow_dir: [f32; 3],
) -> Vec<[f32; 3]> {
    vec![position]
}

/// Key points for a datum identifier triangle (position + apex).
pub fn datum_identifier_key_points(
    position: [f32; 3],
    size: f32,
) -> Vec<[f32; 3]> {
    let p = v3(position);
    let apex = p + Vec3::new(0.0, size, 0.0);
    vec![position, to_arr(apex)]
}
