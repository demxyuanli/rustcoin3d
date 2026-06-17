//! Geometric property benchmarks: face area and solid volume.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rc3d_core::math::PVec3;
use rc3d_shape::geom::{SurfaceGeom, face_area, solid_volume};
use rc3d_shape::store::BRepStore;
use rc3d_shape::topo::*;

/// Create a 10x10 plane face with the given origin/normal/u_dir.
fn add_plane_face(reg: &mut BRepStore, origin: PVec3, normal: PVec3, u_dir: PVec3) -> FaceKey {
    let wk = reg.wires.insert(BRepWire { edges: vec![] });
    let fk = reg.faces.insert(BRepFace {
        surface: SurfaceGeom::Plane { origin, normal, u_dir },
        outer_wire: wk,
        inner_wires: vec![],
        same_sense: true,
        tolerance: 1e-4,
        seam_edges: vec![],
        color: None,
        degenerated_edges: vec![],
    });
    // 10x10 domain
    reg.trim_ranges.insert(fk, (0.0, 10.0, 0.0, 10.0));
    fk
}

pub fn bench_face_area_plane(c: &mut Criterion) {
    let mut reg = BRepStore::new();
    let fk = add_plane_face(
        &mut reg,
        PVec3::ZERO,
        PVec3::new(0.0, 0.0, 1.0),
        PVec3::new(1.0, 0.0, 0.0),
    );
    c.bench_function("face_area_plane", |b| {
        b.iter(|| face_area(black_box(&reg), black_box(fk), black_box(32)))
    });
}

pub fn bench_face_area_sphere(c: &mut Criterion) {
    let mut reg = BRepStore::new();
    let wk = reg.wires.insert(BRepWire { edges: vec![] });
    let fk = reg.faces.insert(BRepFace {
        surface: SurfaceGeom::Sphere {
            center: PVec3::ZERO,
            radius: 5.0,
        },
        outer_wire: wk,
        inner_wires: vec![],
        same_sense: true,
        tolerance: 1e-4,
        seam_edges: vec![],
        color: None,
        degenerated_edges: vec![],
    });
    c.bench_function("face_area_sphere", |b| {
        b.iter(|| face_area(black_box(&reg), black_box(fk), black_box(32)))
    });
}

pub fn bench_solid_volume_cube(c: &mut Criterion) {
    let mut reg = BRepStore::new();
    let specs = [
        (PVec3::ZERO, PVec3::new(1.0, 0.0, 0.0), PVec3::new(0.0, 0.0, 1.0), PVec3::new(0.0, 1.0, 0.0)),
        (PVec3::new(1.0, 0.0, 0.0), PVec3::new(1.0, 0.0, 0.0), PVec3::new(0.0, 0.0, 1.0), PVec3::new(0.0, 1.0, 0.0)),
        (PVec3::ZERO, PVec3::new(0.0, 1.0, 0.0), PVec3::new(1.0, 0.0, 0.0), PVec3::new(0.0, 0.0, 1.0)),
        (PVec3::new(0.0, 1.0, 0.0), PVec3::new(0.0, 1.0, 0.0), PVec3::new(1.0, 0.0, 0.0), PVec3::new(0.0, 0.0, 1.0)),
        (PVec3::ZERO, PVec3::new(0.0, 0.0, 1.0), PVec3::new(1.0, 0.0, 0.0), PVec3::new(0.0, 1.0, 0.0)),
        (PVec3::new(0.0, 0.0, 1.0), PVec3::new(0.0, 0.0, 1.0), PVec3::new(1.0, 0.0, 0.0), PVec3::new(0.0, 1.0, 0.0)),
    ];
    let mut faces = Vec::new();
    for (origin, normal, u_dir, _v_dir) in specs {
        let fk = add_plane_face(&mut reg, origin, normal, u_dir);
        faces.push((fk, Orientation::Forward));
    }
    let sk = reg.shells.insert(BRepShell {
        faces,
        closed: true,
        step_id: None,
    });
    let solid_key = reg.solids.insert(BRepSolid {
        outer_shell: sk,
        void_shells: vec![],
    });

    c.bench_function("solid_volume_cube", |b| {
        b.iter(|| solid_volume(black_box(&reg), black_box(solid_key), black_box(16)))
    });
}

criterion_group!(benches, bench_face_area_plane, bench_face_area_sphere, bench_solid_volume_cube);
criterion_main!(benches);
