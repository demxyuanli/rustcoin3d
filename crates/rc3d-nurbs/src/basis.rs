/// Evaluate the i-th B-spline basis function of degree p at parameter t.
/// Uses the Cox-de Boor recurrence.
pub fn bspline_basis(i: usize, p: usize, t: f32, knots: &[f32]) -> f32 {
    if p == 0 {
        if t >= knots[i] && t < knots[i + 1] {
            return 1.0;
        }
        // Include the rightmost boundary for clamped knot vectors
        let n = knots.len();
        if i + 1 < n && (t - knots[i + 1]).abs() < f32::EPSILON && i == n - 2 {
            return 1.0;
        }
        return 0.0;
    }

    let denom1 = knots[i + p] - knots[i];
    let left = if denom1 > 0.0 {
        ((t - knots[i]) / denom1) * bspline_basis(i, p - 1, t, knots)
    } else {
        0.0
    };

    let denom2 = knots[i + p + 1] - knots[i + 1];
    let right = if denom2 > 0.0 {
        ((knots[i + p + 1] - t) / denom2) * bspline_basis(i + 1, p - 1, t, knots)
    } else {
        0.0
    };

    left + right
}

pub use rc3d_core::utils::bspline::bspline_bases;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knot::{find_span, open_uniform_knots};

    #[test]
    fn test_partition_of_unity() {
        let knots = open_uniform_knots(3, 7);
        let span = find_span(3, &knots, 0.5);
        let bases = bspline_bases(span, 3, 0.5, &knots);
        let sum: f32 = bases.iter().map(|(_, v)| v).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {}", sum);
    }

    #[test]
    fn test_basis_nonnegative() {
        let knots = open_uniform_knots(3, 7);
        for t_i in 0..=20 {
            let t = t_i as f32 / 20.0;
            let span = find_span(3, &knots, t);
            let bases = bspline_bases(span, 3, t, &knots);
            for (_, v) in bases {
                assert!(v >= -1e-6, "negative basis value {} at t={}", v, t);
            }
        }
    }
}
