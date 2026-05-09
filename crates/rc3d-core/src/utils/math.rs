//! Math utilities: remap, safe normalize, interpolation.
//!
//! These are generic math helpers used across the rendering, IO, and application crates.

use glam::Vec3;

/// Remap a value from `[in_min, in_max]` to `[0.0, 1.0]`, clamped.
///
/// Used for normalizing slider positions, progress bars, LOD bias, etc.
///
/// # Examples
///
/// ```
/// use rc3d_core::utils::math::remap;
/// assert!((remap(50.0, 0.0, 100.0) - 0.5).abs() < 1e-6);
/// assert_eq!(remap(200.0, 0.0, 100.0), 1.0); // clamped
/// assert_eq!(remap(-10.0, 0.0, 100.0), 0.0); // clamped
/// ```
#[inline]
pub fn remap(value: f32, in_min: f32, in_max: f32) -> f32 {
    ((value - in_min) / (in_max - in_min)).clamp(0.0, 1.0)
}

/// Linear interpolation: `a + (b - a) * t`.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Normalize a `Vec3`, returning `fallback` if the vector is too small.
///
/// Uses `length_squared() > 1e-20` as the validity threshold, consistent
/// with the rest of the codebase.
#[inline]
pub fn safe_normalize(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.length_squared() > 1e-20 {
        v.normalize()
    } else {
        fallback
    }
}

/// Convert an index buffer to a triangle count.
///
/// Every 3 indices form one triangle; used for draw-call statistics,
/// meshlet threshold decisions, and BVH construction.
#[inline]
pub fn triangle_count(indices: &[u32]) -> usize {
    indices.len() / 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_midpoint() {
        assert!((remap(5.0, 0.0, 10.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn remap_clamped() {
        assert_eq!(remap(20.0, 0.0, 10.0), 1.0);
        assert_eq!(remap(-5.0, 0.0, 10.0), 0.0);
    }

    #[test]
    fn safe_normalize_valid() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = safe_normalize(v, Vec3::Y);
        assert!((n.length() - 1.0).abs() < 1e-6);
        assert!((n.x - 0.6).abs() < 1e-6);
    }

    #[test]
    fn safe_normalize_zero() {
        let n = safe_normalize(Vec3::ZERO, Vec3::Y);
        assert_eq!(n, Vec3::Y);
    }

    #[test]
    fn triangle_count_standard() {
        assert_eq!(triangle_count(&[0, 1, 2, 2, 1, 3]), 2);
        assert_eq!(triangle_count(&[0, 1, 2]), 1);
        assert_eq!(triangle_count(&[0, 1]), 0);
    }
}
