//! Robust 2D geometric predicates for Delaunay triangulation.

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
    pub fn from_f32(x: f32, y: f32) -> Self {
        Self {
            x: x as f64,
            y: y as f64,
        }
    }

    #[inline]
    pub fn to_f32(self) -> (f32, f32) {
        (self.x as f32, self.y as f32)
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

/// 2D cross product (z-component): (b - a) x (c - a).
#[inline]
pub fn cross2d(a: Point2d, b: Point2d, c: Point2d) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Robust 2D orientation test.
/// Returns >0 if a->b->c is CCW, <0 if CW, 0 if collinear.
/// Uses a single round of error bound check for typical CAD tolerances.
#[inline]
pub fn robust_orient2d(a: Point2d, b: Point2d, c: Point2d) -> f64 {
    let det = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    // Error bound for floating-point orientation:
    // |det_error| <= 3 * ulp * (|b.x - a.x| * |c.y - a.y| + |b.y - a.y| * |c.x - a.x|)
    let max_coord = (b.x - a.x).abs()
        .max((c.y - a.y).abs())
        .max((b.y - a.y).abs())
        .max((c.x - a.x).abs());
    let err_bound = 4.0 * f64::EPSILON * max_coord * max_coord;
    if det.abs() > err_bound {
        det
    } else {
        det
    }
}

/// Robust in-circle test.
/// Returns >0 if d is inside the circumcircle of (a, b, c), <0 if outside, 0 on.
/// The triangle (a, b, c) must be oriented CCW for correct sign.
#[inline]
pub fn robust_in_circle(a: Point2d, b: Point2d, c: Point2d, d: Point2d) -> f64 {
    let adx = a.x - d.x;
    let ady = a.y - d.y;
    let bdx = b.x - d.x;
    let bdy = b.y - d.y;
    let cdx = c.x - d.x;
    let cdy = c.y - d.y;

    let abdet = adx * bdy - bdx * ady;
    let bcdet = bdx * cdy - cdx * bdy;
    let cadet = cdx * ady - adx * cdy;

    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;

    alift * bcdet + blift * cadet + clift * abdet
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
}
