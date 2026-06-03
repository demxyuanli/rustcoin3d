//! Arc length estimation for STEP curve entities.
//!
//! Estimates curve arc length by adaptive sampling and chord length summation.

use rc3d_core::math::Vec3;
use super::super::parser::EntityIndex;
use super::super::entity_geom as geom;

/// Estimate the arc length of a curve entity from `t0` to `t1` using
/// adaptive chordal sampling with the given tolerance.
///
/// The curve is sampled at intervals determined by the tolerance, and
/// the sum of chord lengths between consecutive sample points is returned.
pub fn curve_arc_length(
    curve_id: u64,
    entities: &EntityIndex,
    t0: f32,
    t1: f32,
    tolerance: f32,
) -> f32 {
    let (low, high) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };

    // Determine sample count: start with a reasonable base and increase
    // based on tolerance (tighter tolerance → more samples).
    let base_samples = 16;
    let extra = ((1.0 / tolerance.max(1e-6)).sqrt() * 4.0) as usize;
    let n = (base_samples + extra).min(512).max(4);

    // Sample the curve across the full [0, 1] range
    // We need start/end points that roughly bound the curve.
    // Since sample_curve needs actual 3D start/end for LINE curves,
    // we evaluate at t0 and t1 first.
    let pts = sample_curve_range(curve_id, entities, low, high, n);

    if pts.len() < 2 {
        return 0.0;
    }

    let mut arc = 0.0f32;
    for w in pts.windows(2) {
        arc += (w[1] - w[0]).length();
    }
    arc
}

/// Sample a curve at `n` evenly-spaced parameter values in [t0, t1].
///
/// For LINE curves, uses the actual start/end points from the full curve sample.
/// For other curves, evaluates at evenly-spaced t values via `sample_curve`.
fn sample_curve_range(
    curve_id: u64,
    entities: &EntityIndex,
    t0: f32,
    t1: f32,
    n: usize,
) -> Vec<Vec3> {
    let record = match entities.get(&curve_id) {
        Some(r) => r,
        None => return vec![],
    };

    match record.name.as_str() {
        "LINE" => {
            // For lines, sample_curve returns [start, end].
            // Interpolate between t0 and t1.
            let full = geom::sample_curve(
                curve_id, entities, Vec3::ZERO, Vec3::ZERO, 0.01,
            );
            if full.len() < 2 {
                return vec![];
            }
            let p0 = full[0];
            let p1 = full[1];
            let mut result = Vec::with_capacity(n + 1);
            for i in 0..=n {
                let t = t0 + (t1 - t0) * (i as f32) / (n as f32);
                result.push(p0 + (p1 - p0) * t);
            }
            result
        }
        "CIRCLE" | "ELLIPSE" => {
            // sample_curve returns n points around the full circle/ellipse.
            // We need to restrict to [t0, t1].
            let full = geom::sample_curve(
                curve_id, entities, Vec3::ZERO, Vec3::ZERO, 0.001,
            );
            if full.is_empty() {
                return vec![];
            }
            let total = full.len().saturating_sub(1); // full curve has n+1 points (closed)
            let i0 = (t0 * total as f32) as usize;
            let i1 = ((t1 * total as f32) as usize + 1).min(full.len());
            if i0 < i1 && i1 <= full.len() {
                full[i0..i1].to_vec()
            } else {
                vec![]
            }
        }
        _ => {
            // Generic: sample the full curve and map t to point indices
            let full = geom::sample_curve(
                curve_id, entities, Vec3::ZERO, Vec3::ZERO, 0.01,
            );
            if full.len() < 2 {
                return vec![];
            }
            let total = full.len().saturating_sub(1);
            let i0 = (t0 * total as f32) as usize;
            let i1 = ((t1 * total as f32) as usize + 1).min(full.len());
            if i0 < i1 && i1 <= full.len() {
                full[i0..i1].to_vec()
            } else {
                vec![]
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::parser;

    fn make_entities(data_section: &str) -> EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data_section
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_line_arc_length_positive() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (1.0, 0.0, 0.0));
#10 = LINE('', #1, #2);\
",
        );
        // A line from (0,0,0) in direction (1,0,0) has unit length per parameter unit.
        // Sample from t=0 to t=1 should give ~1.0
        let len = curve_arc_length(10, &entities, 0.0, 1.0, 0.01);
        assert!(len > 0.0, "line arc length should be positive");
        assert!((len - 1.0).abs() < 0.1, "line arc length from 0 to 1 should be ~1.0, got {}", len);
    }

    #[test]
    fn test_circle_arc_length_approximate() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = DIRECTION('', (1.0, 0.0, 0.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#10 = CIRCLE('', #4, 1.0);\
",
        );
        // Full unit circle arc length ≈ 2π
        let len = curve_arc_length(10, &entities, 0.0, 1.0, 0.001);
        assert!(len > 0.0, "circle arc length should be positive");
        assert!((len - 2.0 * std::f32::consts::PI).abs() < 0.2,
            "circle arc length should be ≈2π, got {}", len);
    }

    #[test]
    fn test_arc_length_nonexistent_curve() {
        let entities = make_entities("");
        assert_eq!(curve_arc_length(999, &entities, 0.0, 1.0, 0.01), 0.0);
    }
}
