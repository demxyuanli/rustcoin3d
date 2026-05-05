/// Build a uniform knot vector for a curve of given degree and control point count.
/// Example: degree=3, n_cp=7 → [0,0,0,0, 1,2,3, 4,4,4,4]
pub fn uniform_knots(degree: usize, n_control_points: usize) -> Vec<f32> {
    let n_knots = n_control_points + degree + 1;
    let mut knots = Vec::with_capacity(n_knots);
    for i in 0..n_knots {
        knots.push(i as f32);
    }
    knots
}

/// Build an open-uniform (clamped) knot vector.
/// First and last `degree+1` knots are repeated, interior knots are evenly spaced.
pub fn open_uniform_knots(degree: usize, n_control_points: usize) -> Vec<f32> {
    let n_knots = n_control_points + degree + 1;
    let mut knots = Vec::with_capacity(n_knots);
    let n_interior = n_knots.saturating_sub(2 * (degree + 1));
    for _ in 0..=degree {
        knots.push(0.0);
    }
    for i in 1..=n_interior {
        knots.push(i as f32 / (n_interior + 1) as f32);
    }
    for _ in 0..=degree {
        knots.push(1.0);
    }
    knots
}

/// Find the knot span index: the index `i` such that knots[i] <= t < knots[i+1].
/// Returns `degree` if t < knots[degree], or `n-1-degree-1` if t >= knots[n-degree-1].
pub fn find_span(degree: usize, knots: &[f32], t: f32) -> usize {
    let n = knots.len();
    let last = n - degree - 1;
    if t >= knots[last] {
        return last.saturating_sub(1);
    }
    if t <= knots[degree] {
        return degree;
    }
    for i in degree..last {
        if t >= knots[i] && t < knots[i + 1] {
            return i;
        }
    }
    degree
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_uniform_cubic() {
        let knots = open_uniform_knots(3, 7);
        assert_eq!(&knots[0..4], &[0.0; 4]);
        assert_eq!(&knots[7..11], &[1.0; 4]);
    }

    #[test]
    fn test_find_span_mid() {
        let knots = open_uniform_knots(3, 7);
        let span = find_span(3, &knots, 0.5);
        assert!(span >= 3 && span <= 6);
    }
}
