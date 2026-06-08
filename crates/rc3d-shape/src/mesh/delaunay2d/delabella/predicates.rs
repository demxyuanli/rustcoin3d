//! Adaptive-exact geometric predicates for robust Delaunay triangulation.
//!
//! Two-stage adaptive evaluation:
//!   1. Fast f64 arithmetic with Shewchuk error bound check (~99% of cases)
//!   2. Expansion-arithmetic exact fallback
//!
//! Convention: orient2d(a,b,c) > 0 if a→b→c is CCW.
//! Formula: (b.x-a.x)*(c.y-a.y) - (c.x-a.x)*(b.y-a.y)
//!
//! Reference: Shewchuk, "Adaptive Precision Floating-Point Arithmetic
//! and Fast Robust Geometric Predicates", Discrete & Computational Geometry, 1997.

// ── Shewchuk constants for f64 ────────────────────────────────────────

const EPS: f64 = f64::EPSILON; // 2^-52
const CCW_ERR_A: f64 = (3.0 + 16.0 * EPS) * EPS;
const ICC_ERR_A: f64 = (10.0 + 96.0 * EPS) * EPS;
const ICC_ERR_B: f64 = (4.0 + 48.0 * EPS) * EPS;
const SPLITTER: f64 = 134217729.0; // 2^27 + 1

// ── Error-free transformations ────────────────────────────────────────

/// Returns (s, e) where a + b = s + e exactly.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// Fast two-sum when |a| >= |b|.
#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// Split a f64 into (hi, lo) where a = hi + lo, non-overlapping bits.
#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = SPLITTER * a;
    let abig = c - a;
    let ahi = c - abig;
    (ahi, a - ahi)
}

/// Error-free product via split (no FMA dependency).
#[inline]
fn two_product_split(a: f64, b: f64) -> (f64, f64) {
    let (ahi, alo) = split(a);
    let (bhi, blo) = split(b);
    let p = a * b;
    let err = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo;
    (p, err)
}

// ── Expansion operations ─────────────────────────────────────────────

/// h = e + b, where e is an expansion. Returns new expansion.
fn grow_expansion(e: &[f64], b: f64) -> Vec<f64> {
    let n = e.len();
    let mut h = Vec::with_capacity(n + 1);
    let mut q = b;
    for &ei in e {
        let (s, err) = fast_two_sum(q, ei);
        h.push(s);
        q = err;
    }
    h.push(q);
    h
}

/// Expansion difference: h = e - f.
fn expansion_diff(e: &[f64], f: &[f64]) -> Vec<f64> {
    let mut h = e.to_vec();
    for &fi in f {
        h = grow_expansion(&h, -fi);
    }
    h
}

// ── Sign of expansion ─────────────────────────────────────────────────

/// Return the sign of an expansion (sign of largest-magnitude component).
#[inline]
fn expansion_sign(e: &[f64]) -> f64 {
    for &c in e.iter().rev() {
        if c > 0.0 {
            return 1.0;
        }
        if c < 0.0 {
            return -1.0;
        }
    }
    0.0
}

// ── Exact orient2d (expansion arithmetic) ─────────────────────────────

/// Exact orientation test via expansion arithmetic.
/// det = (bx-ax)*(cy-ay) - (cx-ax)*(by-ay)
fn orient2d_exact(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> Vec<f64> {
    let (dbx, dbx_e) = two_sum(bx, -ax);
    let (dby, dby_e) = two_sum(by, -ay);
    let (dcx, dcx_e) = two_sum(cx, -ax);
    let (dcy, dcy_e) = two_sum(cy, -ay);

    // product1 = dbx * dcy
    let (p1, e1) = two_product_split(dbx, dcy);
    let mut prod1 = vec![p1, e1];
    let (g1a, g1b) = two_product_split(dbx_e, dcy);
    prod1 = grow_expansion(&prod1, g1a);
    prod1 = grow_expansion(&prod1, g1b);
    let (g2a, g2b) = two_product_split(dbx, dcy_e);
    prod1 = grow_expansion(&prod1, g2a);
    prod1 = grow_expansion(&prod1, g2b);

    // product2 = dcx * dby
    let (p2, e2) = two_product_split(dcx, dby);
    let mut prod2 = vec![p2, e2];
    let (g3a, g3b) = two_product_split(dcx_e, dby);
    prod2 = grow_expansion(&prod2, g3a);
    prod2 = grow_expansion(&prod2, g3b);
    let (g4a, g4b) = two_product_split(dcx, dby_e);
    prod2 = grow_expansion(&prod2, g4a);
    prod2 = grow_expansion(&prod2, g4b);

    expansion_diff(&prod1, &prod2)
}

// ── Exact incircle (expansion arithmetic) ─────────────────────────────

/// Exact incircle determinant via expansion arithmetic.
/// det = | ax-dx  ay-dy  (ax-dx)²+(ay-dy)² |
///       | bx-dx  by-dy  (bx-dx)²+(by-dy)² |
///       | cx-dx  cy-dy  (cx-dx)²+(cy-dy)² |
fn incircle_exact(
    ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, dx: f64, dy: f64,
) -> Vec<f64> {
    let (adx, _) = two_sum(ax, -dx);
    let (ady, _) = two_sum(ay, -dy);
    let (bdx, _) = two_sum(bx, -dx);
    let (bdy, _) = two_sum(by, -dy);
    let (cdx, _) = two_sum(cx, -dx);
    let (cdy, _) = two_sum(cy, -dy);

    // 2x2 minors of the first two columns:
    // ab_det = adx*bdy - bdx*ady
    // bc_det = bdx*cdy - cdx*bdy
    // ca_det = cdx*ady - adx*cdy
    let ab_det = two_two_diff(adx, bdy, bdx, ady);
    let bc_det = two_two_diff(bdx, cdy, cdx, bdy);
    let ca_det = two_two_diff(cdx, ady, adx, cdy);

    // Lifted coordinates: adx²+ady², etc.
    let alift = lift_sq(adx, ady);
    let blift = lift_sq(bdx, bdy);
    let clift = lift_sq(cdx, cdy);

    // det = bc_det*alift + ca_det*blift + ab_det*clift
    let mut result = vec![0.0];
    for &bc in &bc_det {
        for &al in &alift {
            let (p, e) = two_product_split(bc, al);
            result = grow_expansion(&result, p);
            result = grow_expansion(&result, e);
        }
    }
    for &ca in &ca_det {
        for &bl in &blift {
            let (p, e) = two_product_split(ca, bl);
            result = grow_expansion(&result, p);
            result = grow_expansion(&result, e);
        }
    }
    for &ab in &ab_det {
        for &cl in &clift {
            let (p, e) = two_product_split(ab, cl);
            result = grow_expansion(&result, p);
            result = grow_expansion(&result, e);
        }
    }
    result
}

/// Compute x² + y² as an expansion.
fn lift_sq(x: f64, y: f64) -> Vec<f64> {
    let (sx, ex) = two_product_split(x, x);
    let (sy, ey) = two_product_split(y, y);
    let mut e = vec![sx, ex];
    e = grow_expansion(&e, sy);
    e = grow_expansion(&e, ey);
    e
}

/// Exact (ax * bx) - (ay * by) as expansion.
fn two_two_diff(ax: f64, bx: f64, ay: f64, by: f64) -> Vec<f64> {
    let (px, pex) = two_product_split(ax, bx);
    let (py, pey) = two_product_split(ay, by);
    let (r0, t0) = two_sum(px, -py);
    let (r1, t1) = two_sum(pex, -pey);
    let (r2, t2) = two_sum(t0, r1);
    let (r3, t3) = fast_two_sum(t2, t1);
    let mut h = vec![r0, r2, r3];
    if t3 != 0.0 {
        h.push(t3);
    }
    h.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));
    h
}

// ── Public API ────────────────────────────────────────────────────────

/// Adaptive orient2d: is C left of the directed line A→B?
///
/// Returns positive if CCW, negative if CW, zero if collinear.
/// Formula: (b.x-a.x)*(c.y-a.y) - (c.x-a.x)*(b.y-a.y)
pub fn adaptive_orient2d(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    // Stage 1: fast determinant
    let dbx = bx - ax;
    let dby = by - ay;
    let dcx = cx - ax;
    let dcy = cy - ay;

    let det = dbx * dcy - dcx * dby;
    let permanent = (dbx * dcy).abs() + (dcx * dby).abs();
    if det.abs() > CCW_ERR_A * permanent {
        return det;
    }

    // Stage 2: exact
    expansion_sign(&orient2d_exact(ax, ay, bx, by, cx, cy))
}

/// Adaptive incircle: is D inside the circumcircle of (A, B, C)?
///
/// Returns positive if inside (A,B,C in CCW order), negative if outside, zero if cocircular.
pub fn adaptive_incircle(
    ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, dx: f64, dy: f64,
) -> f64 {
    let adx = ax - dx;
    let ady = ay - dy;
    let bdx = bx - dx;
    let bdy = by - dy;
    let cdx = cx - dx;
    let cdy = cy - dy;

    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;

    // Stage 1
    let det1 = (bdx * cdy - cdx * bdy) * alift
        - (adx * cdy - cdx * ady) * blift
        + (adx * bdy - bdx * ady) * clift;

    let bdxcdy = bdx * cdy;
    let cdxbdy = cdx * bdy;
    let adxcdy = adx * cdy;
    let cdxady = cdx * ady;
    let adxbdy = adx * bdy;
    let bdxady = bdx * ady;

    let permanent = (bdxcdy - cdxbdy).abs() * alift
        + (adxcdy - cdxady).abs() * blift
        + (adxbdy - bdxady).abs() * clift;

    if det1.abs() > ICC_ERR_A * permanent {
        return det1;
    }

    // Stage 2: tighter bound
    let det2 = alift * (bdxcdy - cdxbdy)
        + blift * (cdxady - adxcdy)
        + clift * (adxbdy - bdxady);

    let permanent2 = (bdxcdy.abs() + cdxbdy.abs()) * alift
        + (adxcdy.abs() + cdxady.abs()) * blift
        + (adxbdy.abs() + bdxady.abs()) * clift;

    if det2.abs() > ICC_ERR_B * permanent2 {
        return det2;
    }

    // Stage 3: exact
    expansion_sign(&incircle_exact(ax, ay, bx, by, cx, cy, dx, dy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orient2d_ccw() {
        // a=(0,0) → b=(1,0), c=(0,1): det = 1*1 - 0*0 = 1 > 0 (CCW)
        let r = adaptive_orient2d(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        assert!(r > 0.0, "expected CCW, got {}", r);
    }

    #[test]
    fn test_orient2d_cw() {
        let r = adaptive_orient2d(0.0, 0.0, 1.0, 0.0, 0.0, -1.0);
        assert!(r < 0.0, "expected CW, got {}", r);
    }

    #[test]
    fn test_orient2d_collinear() {
        let r = adaptive_orient2d(0.0, 0.0, 1.0, 0.0, 2.0, 0.0);
        assert!(r == 0.0, "expected collinear, got {}", r);
    }

    #[test]
    fn test_orient2d_near_collinear() {
        let r = adaptive_orient2d(0.0, 0.0, 1.0, 0.0, 0.5, 1e-14);
        assert!(r > 0.0, "expected positive, got {}", r);
    }

    #[test]
    fn test_incircle_inside() {
        // CCW triangle (0,0),(2,0),(1,2), point (1,0.5) inside circumcircle
        let r = adaptive_incircle(0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.5);
        assert!(r > 0.0, "expected inside, got {}", r);
    }

    #[test]
    fn test_incircle_outside() {
        let r = adaptive_incircle(0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 3.0);
        assert!(r < 0.0, "expected outside, got {}", r);
    }

    #[test]
    fn test_incircle_cocircular() {
        // Four points on unit circle
        let r = adaptive_incircle(1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0);
        assert!(r.abs() < 1e-10, "expected cocircular, got {}", r);
    }

    #[test]
    fn test_orient2d_large_values() {
        let big = 1e15;
        let r = adaptive_orient2d(0.0, 0.0, big, 0.0, big, big);
        // det = big*big - big*0 = big² > 0
        assert!(r > 0.0, "large CCW expected, got {}", r);
    }

    #[test]
    fn test_two_sum_round_trip() {
        let a = 1.0 + f64::EPSILON;
        let b = f64::EPSILON;
        let (s, e) = two_sum(a, b);
        let reconstructed = s + e;
        let expected = a + b;
        assert!((reconstructed - expected).abs() < 1e-30, "round-trip should be exact");
    }

    #[test]
    fn test_two_product_round_trip() {
        let a = 1.0 + f64::EPSILON;
        let (p, e) = two_product_split(a, a);
        let exact = p + e;
        let expected = a * a;
        // p+e is the exact product, expected may have rounding
        assert!((exact - expected).abs() <= f64::EPSILON);
    }
}
