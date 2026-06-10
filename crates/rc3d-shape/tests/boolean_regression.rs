//! Boolean operation regression tests — Phase E.
//!
//! Run: cargo test -p rc3d-shape --test boolean_regression -- --nocapture

use rc3d_core::math::Vec3;
use rc3d_shape::bool::{boolean_brep, BoolOp};
use rc3d_shape::geom::{Curve2d, CurveGeom, SurfaceGeom};
use rc3d_shape::store::BRepStore;
use rc3d_shape::topo::{BRepFace, BRepShell, BRepWire, Orientation, ShellKey, FaceKey};

/// Build single square face shell — matches pattern from bool::tests::make_square_face.
fn make_square_shell(reg: &mut BRepStore, origin: Vec3, size: f32, normal: Vec3, u_dir: Vec3) -> (ShellKey, FaceKey) {
    let half = size * 0.5;
    let o = origin;
    let v_dir = normal.cross(u_dir).normalize();

    let corners = [
        o - u_dir * half - v_dir * half,
        o + u_dir * half - v_dir * half,
        o + u_dir * half + v_dir * half,
        o - u_dir * half + v_dir * half,
    ];
    let v0 = reg.find_or_add_vertex(corners[0], 1e-4);
    let v1 = reg.find_or_add_vertex(corners[1], 1e-4);
    let v2 = reg.find_or_add_vertex(corners[2], 1e-4);
    let v3 = reg.find_or_add_vertex(corners[3], 1e-4);

    let surface = SurfaceGeom::Plane { origin: o, normal, u_dir };
    let wk = reg.wires.insert(BRepWire { edges: vec![] });
    let fk = reg.faces.insert(BRepFace {
        surface,
        outer_wire: wk,
        inner_wires: vec![], same_sense: true, tolerance: 1e-4,
        seam_edges: vec![], color: None, degenerated_edges: vec![],
    });

    let edges_data = [(v0,v1,corners[0],corners[1]), (v1,v2,corners[1],corners[2]),
                      (v2,v3,corners[2],corners[3]), (v3,v0,corners[3],corners[0])];
    let mut wire_edges = Vec::new();
    for (va, vb, pa, pb) in edges_data {
        let curve = CurveGeom::Line { origin: pa, direction: pb - pa };
        let pc = Curve2d::Line { origin: (pa.x, pa.y), direction: (pb.x - pa.x, pb.y - pa.y) };
        let ek = reg.add_edge_with_pcurve(va, vb, curve, 1e-4, fk, (pc, true));
        wire_edges.push((ek, Orientation::Forward));
    }
    reg.wires.get_mut(wk).unwrap().edges = wire_edges;

    let sk = reg.shells.insert(BRepShell { faces: vec![(fk, Orientation::Forward)], closed: false, step_id: None });
    (sk, fk)
}

// ── E1: Two overlapping squares (coplanar, XY plane) ────────────────────

#[test]
fn e1_coplanar_union() {
    let mut reg = BRepStore::new();
    let (sa, _) = make_square_shell(&mut reg, Vec3::new(0.0, 0.0, 0.0), 2.0, Vec3::Z, Vec3::X);
    let (sb, _) = make_square_shell(&mut reg, Vec3::new(1.0, 0.0, 0.0), 2.0, Vec3::Z, Vec3::X);
    let r = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Union);
    eprintln!("E1 coplanar-union: shells={} empty={} ints={}", r.result_shells.len(), r.is_empty, r.intersection_count);
    assert!(!r.result_shells.is_empty(), "union should produce result shell");
}

#[test]
fn e2_coplanar_intersection() {
    let mut reg = BRepStore::new();
    let (sa, _) = make_square_shell(&mut reg, Vec3::new(0.0, 0.0, 0.0), 2.0, Vec3::Z, Vec3::X);
    let (sb, _) = make_square_shell(&mut reg, Vec3::new(1.0, 0.0, 0.0), 2.0, Vec3::Z, Vec3::X);
    let r = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Intersection);
    eprintln!("E2 coplanar-intersection: shells={} empty={} ints={}", r.result_shells.len(), r.is_empty, r.intersection_count);
}

#[test]
fn e3_coplanar_difference() {
    let mut reg = BRepStore::new();
    let (sa, _) = make_square_shell(&mut reg, Vec3::new(0.0, 0.0, 0.0), 2.0, Vec3::Z, Vec3::X);
    let (sb, _) = make_square_shell(&mut reg, Vec3::new(1.0, 0.0, 0.0), 2.0, Vec3::Z, Vec3::X);
    let r = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Difference);
    eprintln!("E3 coplanar-diff: shells={} empty={} ints={}", r.result_shells.len(), r.is_empty, r.intersection_count);
}

// ── E4: Disjoint intersection ───────────────────────────────────────────

#[test]
fn e4_perpendicular_faces_int() {
    let mut reg = BRepStore::new();
    // Face in XY plane
    let (sa, _) = make_square_shell(&mut reg, Vec3::new(0.0, 0.0, 0.0), 2.0, Vec3::Z, Vec3::X);
    // Face in XZ plane (perpendicular)
    let (sb, _) = make_square_shell(&mut reg, Vec3::new(0.0, 0.0, 0.0), 2.0, Vec3::Y, Vec3::X);
    let r = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Intersection);
    eprintln!("E4 perpendicular-int: shells={} empty={} ints={}", r.result_shells.len(), r.is_empty, r.intersection_count);
    // Perpendicular planes intersect along a line — should find intersection curves
}
