//! BSpline curve evaluation benchmarks (d0, d0+d1+d2) at varying degrees.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rc3d_core::math::{Real, PVec3};
use rc3d_shape::geom::CurveGeom;

/// Build a clamped (non-rational) BSpline of given degree with `num_ctrl` control points.
/// Control points trace a sine wave in the XZ plane for geometric variation.
fn make_bspline(degree: usize, num_ctrl: usize) -> CurveGeom {
    let control_points: Vec<PVec3> = (0..num_ctrl)
        .map(|i| {
            let t = i as Real / (num_ctrl - 1) as Real;
            PVec3::new(t, (t * std::f64::consts::TAU).sin(), 0.0)
        })
        .collect();
    let n = control_points.len();
    let n_inner = n.saturating_sub(degree); // number of inner knot intervals
    let total_knots = n + degree + 1;
    let mut knots = Vec::with_capacity(total_knots);
    // Leading multiplicity = degree + 1
    for _ in 0..=degree {
        knots.push(0.0);
    }
    // Inner knots: uniform spacing
    for i in 1..n_inner {
        knots.push(i as Real / n_inner as Real);
    }
    // Trailing multiplicity = degree + 1
    for _ in 0..=degree {
        knots.push(1.0);
    }
    CurveGeom::BSpline {
        degree,
        control_points,
        knots,
        weights: None,
    }
}

pub fn bench_bspline_d0_degree3(c: &mut Criterion) {
    let spline = make_bspline(3, 7);
    let ts: Vec<Real> = (0..1000).map(|i| i as Real / 999.0).collect();
    c.bench_function("bspline_d0_degree3", |b| {
        b.iter(|| {
            for &t in &ts {
                black_box(spline.d0(black_box(t)));
            }
        })
    });
}

pub fn bench_bspline_d012_degree3(c: &mut Criterion) {
    let spline = make_bspline(3, 7);
    let ts: Vec<Real> = (0..1000).map(|i| i as Real / 999.0).collect();
    c.bench_function("bspline_d012_degree3", |b| {
        b.iter(|| {
            for &t in &ts {
                black_box(spline.d012(black_box(t)));
            }
        })
    });
}

pub fn bench_bspline_d0_degree8(c: &mut Criterion) {
    let spline = make_bspline(8, 16);
    let ts: Vec<Real> = (0..1000).map(|i| i as Real / 999.0).collect();
    c.bench_function("bspline_d0_degree8", |b| {
        b.iter(|| {
            for &t in &ts {
                black_box(spline.d0(black_box(t)));
            }
        })
    });
}

criterion_group!(benches, bench_bspline_d0_degree3, bench_bspline_d012_degree3, bench_bspline_d0_degree8);
criterion_main!(benches);
