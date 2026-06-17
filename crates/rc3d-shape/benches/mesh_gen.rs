//! Mesh generation benchmarks: plane face and cylindrical face tessellation.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rc3d_core::math::PVec3;
use rc3d_shape::geom::{CurveGeom, SurfaceGeom};
use rc3d_shape::geom::curve2d::Curve2d;
use rc3d_shape::mesh::{mesh_brep_shell, BRepMeshConfig};
use rc3d_shape::store::BRepStore;
use rc3d_shape::topo::*;

/// Build a 10x10 plane face with a 4-edge wire boundary.
fn build_plane_shell(reg: &mut BRepStore) -> ShellKey {
    let face_key = reg.add_face(
        SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        },
        1e-4,
    );
    let edges_data = [
        (PVec3::ZERO, PVec3::new(10.0, 0.0, 0.0), (0.0, 0.0), (10.0, 0.0)),
        (PVec3::new(10.0, 0.0, 0.0), PVec3::new(10.0, 10.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
        (PVec3::new(10.0, 10.0, 0.0), PVec3::new(0.0, 10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
        (PVec3::new(0.0, 10.0, 0.0), PVec3::ZERO, (0.0, 10.0), (0.0, 0.0)),
    ];
    let mut wire_edges = Vec::new();
    for (a, b, uv_a, uv_b) in edges_data {
        let v0 = reg.find_or_add_vertex(a, 1e-4);
        let v1 = reg.find_or_add_vertex(b, 1e-4);
        let curve_3d = CurveGeom::Line {
            origin: a,
            direction: b - a,
        };
        let pcurve = Curve2d::Line {
            origin: uv_a,
            direction: (uv_b.0 - uv_a.0, uv_b.1 - uv_a.1),
        };
        let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve, true);
        wire_edges.push((ek, Orientation::Forward));
    }
    let outer_wire = reg.wires.insert(BRepWire { edges: wire_edges });
    if let Some(face) = reg.faces.get_mut(face_key) {
        face.outer_wire = outer_wire;
    }
    reg.shells.insert(BRepShell {
        faces: vec![(face_key, Orientation::Forward)],
        closed: false,
        step_id: None,
    })
}

/// Build a cylindrical face (radius=1, height=2, full 360 degrees).
fn build_cylinder_shell(reg: &mut BRepStore) -> ShellKey {
    let face_key = reg.add_face(
        SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 1.0),
        1e-4,
    );
    // Four edge wire: two arcs at z=0 and z=2, two seam lines
    let v0 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-4);
    let v1 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 2.0), 1e-4);
    let v2 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-4); // seam reuse
    let v3 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 2.0), 1e-4); // seam reuse

    let circle_z0 = CurveGeom::circle(PVec3::ZERO, PVec3::Z, 1.0);
    let circle_z2 = CurveGeom::circle(PVec3::new(0.0, 0.0, 2.0), PVec3::Z, 1.0);
    let line_seam = CurveGeom::Line {
        origin: PVec3::new(1.0, 0.0, 0.0),
        direction: PVec3::new(0.0, 0.0, 2.0),
    };

    let pc_arc_z0 = Curve2d::Line {
        origin: (0.0, 0.0),
        direction: (std::f64::consts::TAU, 0.0),
    };
    let pc_arc_z2 = Curve2d::Line {
        origin: (0.0, 2.0),
        direction: (std::f64::consts::TAU, 0.0),
    };
    let pc_seam_lo = Curve2d::Line {
        origin: (0.0, 0.0),
        direction: (0.0, 2.0),
    };
    let pc_seam_hi = Curve2d::Line {
        origin: (std::f64::consts::TAU, 0.0),
        direction: (0.0, 2.0),
    };

    let e_arc_z0 = reg.add_edge_with_pcurve(v0, v2, circle_z0, 1e-4, face_key, pc_arc_z0, true);
    let e_line_lo = reg.add_edge_with_pcurve(v0, v1, line_seam.clone(), 1e-4, face_key, pc_seam_lo, true);
    let e_arc_z2 = reg.add_edge_with_pcurve(v3, v1, circle_z2, 1e-4, face_key, pc_arc_z2, true);
    let e_line_hi = reg.add_edge_with_pcurve(v2, v3, line_seam, 1e-4, face_key, pc_seam_hi, true);

    let outer_wire = reg.wires.insert(BRepWire {
        edges: vec![
            (e_arc_z0, Orientation::Forward),
            (e_line_lo, Orientation::Forward),
            (e_arc_z2, Orientation::Reversed),
            (e_line_hi, Orientation::Forward),
        ],
    });
    if let Some(face) = reg.faces.get_mut(face_key) {
        face.outer_wire = outer_wire;
    }
    reg.shells.insert(BRepShell {
        faces: vec![(face_key, Orientation::Forward)],
        closed: false,
        step_id: None,
    })
}

pub fn bench_mesh_plane_face(c: &mut Criterion) {
    let mut reg = BRepStore::new();
    let shell_key = build_plane_shell(&mut reg);
    let config = BRepMeshConfig::default();
    c.bench_function("mesh_plane_face", |b| {
        b.iter(|| mesh_brep_shell(black_box(shell_key), black_box(&reg), black_box(&config), black_box(&[])))
    });
}

pub fn bench_mesh_cylinder_face(c: &mut Criterion) {
    let mut reg = BRepStore::new();
    let shell_key = build_cylinder_shell(&mut reg);
    let config = BRepMeshConfig::default();
    c.bench_function("mesh_cylinder_face", |b| {
        b.iter(|| mesh_brep_shell(black_box(shell_key), black_box(&reg), black_box(&config), black_box(&[])))
    });
}

criterion_group!(benches, bench_mesh_plane_face, bench_mesh_cylinder_face);
criterion_main!(benches);
