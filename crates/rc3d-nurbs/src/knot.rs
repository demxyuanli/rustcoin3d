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
    knots.resize(knots.len() + degree + 1, 0.0);
    for i in 1..=n_interior {
        knots.push(i as f32 / (n_interior + 1) as f32);
    }
    knots.resize(knots.len() + degree + 1, 1.0);
    knots
}

pub use rc3d_core::utils::bspline::find_span;

const KNOT_EPS: f32 = 1e-10;

/// Count how many times value `t` appears in the knot vector (epsilon comparison).
pub fn knot_multiplicity(knots: &[f32], t: f32) -> usize {
    knots.iter().filter(|&&k| (k - t).abs() < KNOT_EPS).count()
}

/// Extract unique knot values with their multiplicities, preserving order.
pub fn unique_knots(knots: &[f32]) -> Vec<(f32, usize)> {
    let mut result: Vec<(f32, usize)> = Vec::new();
    for &k in knots {
        if let Some(last) = result.last_mut() {
            if (last.0 - k).abs() < KNOT_EPS {
                last.1 += 1;
                continue;
            }
        }
        result.push((k, 1));
    }
    result
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

    #[test]
    fn test_unique_knots_cubic() {
        let knots = open_uniform_knots(3, 7);
        // Produces [0,0,0,0, 0.25, 0.5, 0.75, 1,1,1,1]
        let uniq = unique_knots(&knots);
        assert_eq!(uniq.len(), 5);
        assert_eq!(uniq[0], (0.0, 4));
        assert_eq!(uniq[3], (0.75, 1));
        assert_eq!(uniq[4], (1.0, 4));
    }

    #[test]
    fn test_knot_multiplicity_basic() {
        let knots = vec![0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0];
        assert_eq!(knot_multiplicity(&knots, 0.0), 2);
        assert_eq!(knot_multiplicity(&knots, 0.5), 3);
        assert_eq!(knot_multiplicity(&knots, 1.0), 2);
        assert_eq!(knot_multiplicity(&knots, 0.7), 0);
    }
}
