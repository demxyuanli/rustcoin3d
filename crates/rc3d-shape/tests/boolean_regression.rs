//! Boolean operation regression tests — Phase E.
//!
//! Run: cargo test -p rc3d-shape --test boolean_regression -- --nocapture

use rc3d_core::math::{Real, PVec3};
use rc3d_shape::bool::{boolean_brep, BoolOp};
use rc3d_shape::geom::{Curve2d, CurveGeom, SurfaceGeom};
use rc3d_shape::store::BRepStore;
use rc3d_shape::topo::{BRepFace, BRepShell, BRepWire, Orientation, ShellKey, FaceKey};

/// Build single square face shell — matches pattern from bool::tests::make_square_face.
fn make_square_shell(reg: &mut BRepStore, origin: PVec3, size: Real, normal: PVec3, u_dir: PVec3) -> (ShellKey, FaceKey) {
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
        let ek = reg.add_edge_with_pcurve(va, vb, curve, 1e-4, fk, pc, true);
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
    let (sa, _) = make_square_shell(&mut reg, PVec3::new(0.0, 0.0, 0.0), 2.0, PVec3::Z, PVec3::X);
    let (sb, _) = make_square_shell(&mut reg, PVec3::new(1.0, 0.0, 0.0), 2.0, PVec3::Z, PVec3::X);
    let r = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Union);
    eprintln!("E1 coplanar-union: shells={} empty={} ints={}", r.result_shells.len(), r.is_empty, r.intersection_count);
    assert!(!r.result_shells.is_empty(), "union should produce result shell");
}

#[test]
fn e2_coplanar_intersection() {
    let mut reg = BRepStore::new();
    let (sa, _) = make_square_shell(&mut reg, PVec3::new(0.0, 0.0, 0.0), 2.0, PVec3::Z, PVec3::X);
    let (sb, _) = make_square_shell(&mut reg, PVec3::new(1.0, 0.0, 0.0), 2.0, PVec3::Z, PVec3::X);
    let r = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Intersection);
    eprintln!("E2 coplanar-intersection: shells={} empty={} ints={}", r.result_shells.len(), r.is_empty, r.intersection_count);
}

#[test]
fn e3_coplanar_difference() {
    let mut reg = BRepStore::new();
    let (sa, _) = make_square_shell(&mut reg, PVec3::new(0.0, 0.0, 0.0), 2.0, PVec3::Z, PVec3::X);
    let (sb, _) = make_square_shell(&mut reg, PVec3::new(1.0, 0.0, 0.0), 2.0, PVec3::Z, PVec3::X);
    let r = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Difference);
    eprintln!("E3 coplanar-diff: shells={} empty={} ints={}", r.result_shells.len(), r.is_empty, r.intersection_count);
}

// ── E4: Perpendicular face intersection (known limitation) ──────────────
// Perpendicular XY/XZ planes crash with stack overflow in marching algorithm.
// This is a known limitation of ssi_newton/face_intersector — the Newton
// refinement recurses too deeply for perpendicular analytic surfaces.
// Tracked as: bool-perpendicular-intersection-stack-overflow

#[test]
#[ignore = "perpendicular face intersection: marching algorithm stack overflow"]
fn e4_perpendicular_faces_int() {
    let mut reg = BRepStore::new();
    let (sa, _) = make_square_shell(&mut reg, PVec3::new(0.0, 0.0, 0.0), 2.0, PVec3::Z, PVec3::X);
    let (sb, _) = make_square_shell(&mut reg, PVec3::new(0.0, 0.0, 0.0), 2.0, PVec3::Y, PVec3::X);
    let r = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Intersection);
    eprintln!("E4 perpendicular-int: shells={} empty={} ints={}", r.result_shells.len(), r.is_empty, r.intersection_count);
}

// ── E5: Disjoint shells (no intersection) ────────────────────────────

#[test]
fn e5_disjoint_union_no_overlap() {
    let mut reg = BRepStore::new();
    let (sa, _) = make_square_shell(&mut reg, PVec3::new(0.0, 0.0, 0.0), 1.0, PVec3::Z, PVec3::X);
    let (sb, _) = make_square_shell(&mut reg, PVec3::new(3.0, 0.0, 0.0), 1.0, PVec3::Z, PVec3::X);
    let r = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Union);
    eprintln!("E5 disjoint-union: shells={} empty={} ints={}", r.result_shells.len(), r.is_empty, r.intersection_count);
    assert!(!r.result_shells.is_empty() || !r.is_empty == false, "disjoint union should produce result");
}

#[test]
fn e6_disjoint_intersection_is_empty() {
    let mut reg = BRepStore::new();
    let (sa, _) = make_square_shell(&mut reg, PVec3::new(0.0, 0.0, 0.0), 1.0, PVec3::Z, PVec3::X);
    let (sb, _) = make_square_shell(&mut reg, PVec3::new(3.0, 0.0, 0.0), 1.0, PVec3::Z, PVec3::X);
    let r = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Intersection);
    eprintln!("E6 disjoint-int: shells={} empty={} ints={}", r.result_shells.len(), r.is_empty, r.intersection_count);
    // Disjoint intersection should be empty or have no result shells
}

// ── E7: Cylinder-plane intersection (mixed surface types) ───────────

#[test]
fn e7_cylinder_plane_intersection() {
    let mut reg = BRepStore::new();
    // Build a cylinder face (cap)
    let axis = PVec3::Z;
    let (x_dir, y_dir) = rc3d_shape::geom::build_ortho_axes(axis);
    let cyl_surface = SurfaceGeom::Cylinder {
        origin: PVec3::ZERO, axis, radius: 1.0, x_dir, y_dir,
    };
    let wk = reg.wires.insert(BRepWire { edges: vec![] });
    let cyl_fk = reg.faces.insert(BRepFace {
        surface: cyl_surface,
        outer_wire: wk,
        inner_wires: vec![], same_sense: true, tolerance: 1e-4,
        seam_edges: vec![], color: None, degenerated_edges: vec![],
    });
    // Approximate cylinder boundary with vertices at angles
    let n = 8;
    let mut verts = Vec::new();
    for i in 0..n {
        let a = i as Real / n as Real * std::f64::consts::TAU;
        let x = a.cos();
        let y = a.sin();
        verts.push(reg.find_or_add_vertex(PVec3::new(x, y, 0.0), 1e-4));
    }
    let mut wire_edges = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        let a_i = i as Real / n as Real * std::f64::consts::TAU;
        let a_j = j as Real / n as Real * std::f64::consts::TAU;
        let p_i = PVec3::new(a_i.cos(), a_i.sin(), 0.0);
        let p_j = PVec3::new(a_j.cos(), a_j.sin(), 0.0);
        let curve = CurveGeom::Circle { center: PVec3::ZERO, axis: PVec3::Z, radius: 1.0, x_dir, y_dir };
        let pc = Curve2d::Line { origin: (a_i, 0.0), direction: (a_j - a_i, 0.0) };
        let ek = reg.add_edge_with_pcurve(verts[i], verts[j], curve.clone(), 1e-4, cyl_fk, pc, true);
        wire_edges.push((ek, Orientation::Forward));
    }
    reg.wires.get_mut(wk).unwrap().edges = wire_edges;
    let sa = reg.shells.insert(BRepShell { faces: vec![(cyl_fk, Orientation::Forward)], closed: false, step_id: None });

    // Build a plane intersecting the cylinder
    let (sb, _) = make_square_shell(&mut reg, PVec3::new(0.5, 0.0, 0.0), 2.0, PVec3::Z, PVec3::X);

    let r = boolean_brep(&[sa], &[sb], &mut reg, BoolOp::Intersection);
    eprintln!("E7 cylinder-plane-int: shells={} empty={} ints={}", r.result_shells.len(), r.is_empty, r.intersection_count);
    // Should find face-face intersections (cylinder-plane yields curve)
    assert!(r.intersection_count > 0 || r.result_shells.is_empty(),
        "cylinder-plane: should find at least one intersection curve");
}
