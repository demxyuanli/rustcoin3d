//! Robust 2D geometric predicates for Delaunay triangulation.

use rc3d_core::math::Real;
/// A lightweight 2D point using f64 for robust arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point2d {
    pub x: f64,
    pub y: f64,
}

impl Point2d {
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn from_f32(x: Real, y: Real) -> Self {
        Self {
            x: x as f64,
            y: y as f64,
        }
    }

    #[inline]
    pub fn to_f32(self) -> (Real, Real) {
        (self.x as Real, self.y as Real)
    }

    #[inline]
    pub fn dist_sq(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    #[inline]
    pub fn dist(self, other: Self) -> f64 {
        self.dist_sq(other).sqrt()
    }
}

impl std::ops::Sub for Point2d {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Add for Point2d {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Mul<f64> for Point2d {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }
}

/// Robust 2D orientation test using Shewchuk adaptive-exact arithmetic.
/// Returns >0 if a->b->c is CCW, <0 if CW, 0 if collinear.
#[inline]
pub fn robust_orient2d(a: Point2d, b: Point2d, c: Point2d) -> f64 {
    super::delabella::adaptive_orient2d(a.x, a.y, b.x, b.y, c.x, c.y)
}

/// Robust in-circle test using Shewchuk adaptive-exact arithmetic.
/// Returns >0 if d is inside the circumcircle of (a, b, c), <0 if outside, 0 on.
/// The triangle (a, b, c) must be oriented CCW for correct sign.
#[inline]
pub fn robust_in_circle(a: Point2d, b: Point2d, c: Point2d, d: Point2d) -> f64 {
    super::delabella::adaptive_incircle(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y)
}

/// UV bounding span of four points (max axis extent).
#[inline]
fn uv_span4(a: Point2d, b: Point2d, c: Point2d, d: Point2d) -> f64 {
    let min_x = a.x.min(b.x).min(c.x).min(d.x);
    let max_x = a.x.max(b.x).max(c.x).max(d.x);
    let min_y = a.y.min(b.y).min(c.y).min(d.y);
    let max_y = a.y.max(b.y).max(c.y).max(d.y);
    (max_x - min_x).max(max_y - min_y)
}

/// Fast in-circle test using f64 determinant (sufficient for incremental CDT).
#[inline]
pub fn fast_in_circle(a: Point2d, b: Point2d, c: Point2d, d: Point2d) -> f64 {
    let ax = a.x - d.x;
    let ay = a.y - d.y;
    let bx = b.x - d.x;
    let by = b.y - d.y;
    let cx = c.x - d.x;
    let cy = c.y - d.y;
    (ax * ax + ay * ay) * (bx * cy - cx * by)
        - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay)
}

/// Adaptive in-circle: fast f64 when safe, robust exact when UV span is large or ambiguous.
#[inline]
pub fn adaptive_in_circle(a: Point2d, b: Point2d, c: Point2d, d: Point2d) -> f64 {
    let span = uv_span4(a, b, c, d);
    if span > 1e4 {
        return robust_in_circle(a, b, c, d);
    }
    let fast = fast_in_circle(a, b, c, d);
    let safety = (span * span * 1e-15).max(1e-12);
    if fast.abs() > safety {
        fast
    } else {
        robust_in_circle(a, b, c, d)
    }
}

/// Compute circumcenter and squared radius of triangle (a, b, c).
/// Returns None if the three points are collinear.
pub fn circumcircle(a: Point2d, b: Point2d, c: Point2d) -> Option<(Point2d, f64)> {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if d.abs() < 1e-30 {
        return None;
    }
    let a_sq = a.x * a.x + a.y * a.y;
    let b_sq = b.x * b.x + b.y * b.y;
    let c_sq = c.x * c.x + c.y * c.y;

    let cx = (a_sq * (b.y - c.y) + b_sq * (c.y - a.y) + c_sq * (a.y - b.y)) / d;
    let cy = (a_sq * (c.x - b.x) + b_sq * (a.x - c.x) + c_sq * (b.x - a.x)) / d;

    let center = Point2d::new(cx, cy);
    let r_sq = center.dist_sq(a);
    Some((center, r_sq))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orient2d_ccw() {
        let a = Point2d::new(0.0, 0.0);
        let b = Point2d::new(1.0, 0.0);
        let c = Point2d::new(0.0, 1.0);
        assert!(robust_orient2d(a, b, c) > 0.0);
    }

    #[test]
    fn test_orient2d_cw() {
        let a = Point2d::new(0.0, 0.0);
        let b = Point2d::new(0.0, 1.0);
        let c = Point2d::new(1.0, 0.0);
        assert!(robust_orient2d(a, b, c) < 0.0);
    }

    #[test]
    fn test_in_circle() {
        let a = Point2d::new(0.0, 0.0);
        let b = Point2d::new(1.0, 0.0);
        let c = Point2d::new(0.0, 1.0);
        let inside = Point2d::new(0.25, 0.25);
        let outside = Point2d::new(2.0, 2.0);
        assert!(robust_in_circle(a, b, c, inside) > 0.0);
        assert!(robust_in_circle(a, b, c, outside) < 0.0);
    }

    #[test]
    fn test_circumcircle() {
        let a = Point2d::new(0.0, 0.0);
        let b = Point2d::new(1.0, 0.0);
        let c = Point2d::new(0.0, 1.0);
        let (center, r_sq) = circumcircle(a, b, c).unwrap();
        let expected_r_sq = 0.5;
        assert!((center.x - 0.5).abs() < 1e-10);
        assert!((center.y - 0.5).abs() < 1e-10);
        assert!((r_sq - expected_r_sq).abs() < 1e-10);
    }

    #[test]
    fn test_orient2d_near_collinear_adaptive() {
        // Nearly collinear points where naive f64 returns ~0.
        // Adaptive-exact should correctly detect CCW.
        let a = Point2d::new(0.0, 0.0);
        let b = Point2d::new(1.0, 0.0);
        let c = Point2d::new(0.5, 1e-15);
        let result = robust_orient2d(a, b, c);
        assert!(result > 0.0, "adaptive orient2d should detect CCW, got {}", result);
    }
}
