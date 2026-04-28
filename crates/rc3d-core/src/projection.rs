//! Helpers for interpreting projection matrices (WebGPU clip / NDC conventions).

use glam::{Mat4, Vec4};

fn ndc_z_clip(projection: Mat4, z_eye: f32) -> Option<f32> {
    let c = projection * Vec4::new(0.0, 0.0, z_eye, 1.0);
    if c.w.abs() < 1e-20 {
        return None;
    }
    let z = c.z / c.w;
    z.is_finite().then_some(z)
}

/// Returns true when NDC (or post-projective) depth increases toward the camera along the
/// forward view ray (0,0,-zn) to (0,0,-zf), i.e. nearer surfaces map to **larger** depth
/// values than farther ones — the layout expected by a max-reduced HZB built from the same
/// depth buffer (reverse-Z style).
///
/// For uninitialized or non-standard matrices this returns `false` (safe default: skip
/// reverse-Z-only GPU paths such as max-pyramid occlusion until a matching min pyramid exists).
pub fn depth_reversed_z_from_projection(projection: Mat4) -> bool {
    if let Some((zn, zf)) = try_extract_perspective_clip_distances(projection) {
        if let Some(out) = probe_depth_order(projection, -zn * 1.0001, -zf * 0.9999) {
            return out;
        }
    }
    if let Some((zn, zf)) = try_extract_orthographic_clip_distances(projection) {
        if let Some(out) = probe_depth_order(projection, -zn * 1.0001, -zf * 0.9999) {
            return out;
        }
    }
    false
}

fn probe_depth_order(projection: Mat4, z_closer: f32, z_farther: f32) -> Option<bool> {
    let n_closer = ndc_z_clip(projection, z_closer)?;
    let n_farther = ndc_z_clip(projection, z_farther)?;
    if !(-0.5..=1.5).contains(&n_closer) || !(-0.5..=1.5).contains(&n_farther) {
        return None;
    }
    Some(n_closer > n_farther + 1e-5)
}

/// WebGPU-style perspective with `clip.w = -z_eye` uses `z_axis.w ≈ -1`.
/// Forward-Z finite perspective has `z_axis.z < 0`; reverse-Z finite uses `z_axis.z > 0`
/// (see `rc3d-scene` optional reverse-depth projection matrices).
fn try_extract_perspective_clip_distances(projection: Mat4) -> Option<(f32, f32)> {
    if projection.z_axis.w > -0.5 {
        return None;
    }
    let a = projection.z_axis.z;
    let b = projection.w_axis.z;
    if !a.is_finite() || !b.is_finite() || a.abs() < 1e-12 {
        return None;
    }

    if a > 0.0 {
        let zn = b / (1.0 + a);
        let zf = b / a;
        if !zn.is_finite() || !zf.is_finite() || zn <= 0.0 || zf <= zn {
            return None;
        }
        return Some((zn, zf));
    }

    let zn = b / a;
    if !zn.is_finite() || zn <= 0.0 {
        return None;
    }
    let denom = 1.0 + a;
    if denom.abs() < 1e-5 {
        return None;
    }
    let zf = a * zn / denom;
    if !zf.is_finite() || zf <= zn {
        return None;
    }
    Some((zn, zf))
}

/// Orthographic `clip.w == 1` uses `z_axis.w == 0` and linear `clip.z = cz * z + cw`.
/// Reverse-Z ortho uses `z_axis.z > 0` (ndc near plane > far plane).
fn try_extract_orthographic_clip_distances(projection: Mat4) -> Option<(f32, f32)> {
    if projection.z_axis.w < -0.5 {
        return None;
    }
    let cz = projection.z_axis.z;
    let cw = projection.w_axis.z;
    if cz.abs() < 1e-20 {
        return None;
    }

    if cz > 0.0 {
        let zn = (cw - 1.0) / cz;
        let zf = cw / cz;
        if !zn.is_finite() || !zf.is_finite() || zn <= 0.0 || zf <= zn {
            return None;
        }
        return Some((zn, zf));
    }

    let zn = cw / cz;
    if !zn.is_finite() || zn <= 0.0 {
        return None;
    }
    let zf = zn - 1.0 / cz;
    if !zf.is_finite() || zf <= zn {
        return None;
    }
    Some((zn, zf))
}
