//! Spatial index benchmarks: insert and nearest-neighbor lookup.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use rc3d_core::math::Real;
use rc3d_core::utils::spatial::SpatialIndexF64;

/// Generate `n` random 3D points in [0, 10)^3.
fn random_points(n: usize) -> Vec<[Real; 3]> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            [
                rng.gen::<Real>() * 10.0,
                rng.gen::<Real>() * 10.0,
                rng.gen::<Real>() * 10.0,
            ]
        })
        .collect()
}

pub fn bench_spatial_insert_1k(c: &mut Criterion) {
    let points = random_points(1000);
    c.bench_function("spatial_insert_1k", |b| {
        b.iter(|| {
            let mut idx = SpatialIndexF64::<usize>::with_cell_size(1e-4);
            for (i, &p) in points.iter().enumerate() {
                idx.insert(i, p);
            }
            black_box(idx)
        })
    });
}

pub fn bench_spatial_find_near_1k(c: &mut Criterion) {
    let points = random_points(1000);
    let mut idx = SpatialIndexF64::<usize>::with_cell_size(1e-4);
    for (i, &p) in points.iter().enumerate() {
        idx.insert(i, p);
    }
    let queries: Vec<[Real; 3]> = random_points(200);
    let positions: Vec<[Real; 3]> = points.clone();

    c.bench_function("spatial_find_near_1k", |b| {
        b.iter(|| {
            for &q in &queries {
                black_box(idx.find_near(q, black_box(0.01), |i: usize| positions[i]));
            }
        })
    });
}

criterion_group!(benches, bench_spatial_insert_1k, bench_spatial_find_near_1k);
criterion_main!(benches);
