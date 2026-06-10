//! B-spline basis evaluation utilities (knot span, basis functions).

/// Knot-span index for parameter `t`.
/// Returns the index `i` such that knots[i] <= t < knots[i+1].
pub fn find_span(degree: usize, knots: &[f32], t: f32) -> usize {
    let n = knots.len() - degree - 1;
    if t >= knots[n] {
        return n - 1;
    }
    for i in (degree..n).rev() {
        if t >= knots[i] {
            return i;
        }
    }
    degree
}

/// Non-zero basis functions at `t` for the active span.
/// Returns Vec of (basis_index, value) pairs.
pub fn bspline_bases(span: usize, degree: usize, t: f32, knots: &[f32]) -> Vec<(usize, f32)> {
    if degree == 0 {
        return vec![(span, 1.0)];
    }
    let stride = degree + 1;
    let mut n = vec![0.0f32; stride * stride];
    n[0] = 1.0;
    for j in 1..=degree {
        let row_j = j * stride;
        let row_prev = (j - 1) * stride;
        for i in 0..=j {
            let left = if i >= 1 && span + i >= j {
                let idx_lo = span + i - j;
                if idx_lo + j < knots.len() {
                    let denom = knots[idx_lo + j] - knots[idx_lo];
                    if denom > 1e-10 {
                        (t - knots[idx_lo]) / denom * n[row_prev + i - 1]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let right = if i < j && span + i + 1 >= j && span + i + 1 < knots.len() {
                let idx_lo = span + i + 1 - j;
                let denom = knots[idx_lo + j] - knots[idx_lo];
                if denom > 1e-10 {
                    (knots[idx_lo + j] - t) / denom * n[row_prev + i]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            n[row_j + i] = left + right;
        }
    }
    let row_deg = degree * stride;
    let start = span.saturating_sub(degree);
    (0..=degree)
        .map(|i| (start + i, n[row_deg + i]))
        .collect()
}
