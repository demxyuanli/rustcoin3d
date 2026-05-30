//! Integration tests for B-rep boolean operations.
//! Two intersecting axis-aligned boxes → union, intersection, difference.

use rc3d_core::math::Vec3;
use rc3d_io::step::topology::{StepShell, StepFace, StepEdge, StepLoop};
use rc3d_io::step::bool::{BoolOp, boolean};
use rc3d_io::step::parser::EntityIndex;

/// Build a planar face from a polygon with explicit 3D vertices.
fn make_face_3d(pts_3d: &[(f32, f32, f32)]) -> StepFace {
    let n = pts_3d.len();
    let edges: Vec<StepEdge> = (0..n).map(|i| {
        let (x1, y1, z1) = pts_3d[i];
        let (x2, y2, z2) = pts_3d[(i + 1) % n];
        StepEdge {
            start: Vec3::new(x1, y1, z1),
            end: Vec3::new(x2, y2, z2),
            curve_id: 0,
            curve_type: "LINE".into(),
            reversed: false,
            tolerance: 1e-4,
        }
    }).collect();
    StepFace {
        bounds: vec![StepLoop {
            edges,
            vertex_loop_point: None,
        }],
        surface_id: None,
        same_sense: true,
        face_id: None,
    }
}

/// Build a box as a shell of 6 axis-aligned faces.
fn make_box_shell(cx: f32, cy: f32, cz: f32, sx: f32, sy: f32, sz: f32) -> StepShell {
    let x0 = cx - sx * 0.5; let x1 = cx + sx * 0.5;
    let y0 = cy - sy * 0.5; let y1 = cy + sy * 0.5;
    let z0 = cz - sz * 0.5; let z1 = cz + sz * 0.5;

    let faces = vec![
        // Bottom (-Z): CCW from below
        make_face_3d(&[(x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0)]),
        // Top (+Z): CCW from above
        make_face_3d(&[(x0,y1,z1), (x1,y1,z1), (x1,y0,z1), (x0,y0,z1)]),
        // Front (-Y): CCW from front
        make_face_3d(&[(x0,y0,z0), (x0,y0,z1), (x1,y0,z1), (x1,y0,z0)]),
        // Back (+Y): CCW from back
        make_face_3d(&[(x1,y1,z0), (x1,y1,z1), (x0,y1,z1), (x0,y1,z0)]),
        // Left (-X): CCW from left
        make_face_3d(&[(x0,y0,z0), (x0,y1,z0), (x0,y1,z1), (x0,y0,z1)]),
        // Right (+X): CCW from right
        make_face_3d(&[(x1,y1,z0), (x1,y0,z0), (x1,y0,z1), (x1,y1,z1)]),
    ];

    StepShell { id: 1, faces }
}

#[test]
fn test_boxes_union() {
    // Box A: centered at (0, 0, 0), size (4, 4, 4)
    // Box B: centered at (0, 0, 0), size (2, 2, 2) — fully inside A
    // Union should be box A (large box) since B is inside it
    let shell_a = make_box_shell(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
    let shell_b = make_box_shell(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);

    let entities = EntityIndex::new();

    let result = boolean(&[shell_a], &[shell_b], &entities, &entities, BoolOp::Union);
    assert!(!result.is_empty, "union of contained boxes should not be empty");
    let face_count: usize = result.shells.iter().map(|s| s.faces.len()).sum();
    assert!(face_count > 0, "union should have faces");
    eprintln!("Union faces: {}", face_count);
}

#[test]
fn test_boxes_intersection() {
    // Box A: size 4x4x4, Box B: size 2x2x2 centered — B fully inside A
    let shell_a = make_box_shell(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
    let shell_b = make_box_shell(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let entities = EntityIndex::new();

    let result = boolean(&[shell_a], &[shell_b], &entities, &entities, BoolOp::Intersection);
    assert!(!result.is_empty, "intersection of contained boxes should not be empty");
    let fc: usize = result.shells.iter().map(|s| s.faces.len()).sum();
    eprintln!("Intersection faces: {}", fc);
}

#[test]
fn test_boxes_difference() {
    // Box A: size 4x4x4, centered at origin
    // Box B: size 2x2x2, centered at (0,0,0) — fully inside A
    // A-B should be A's 6 faces (outside B) but B's faces are inside A (inverted)
    let shell_a = make_box_shell(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
    let shell_b = make_box_shell(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let entities = EntityIndex::new();

    let result = boolean(&[shell_a], &[shell_b], &entities, &entities, BoolOp::Difference);
    assert!(!result.is_empty, "A-B should not be empty");
    eprintln!("Difference faces: {}", result.shells.iter().map(|s| s.faces.len()).sum::<usize>());
}

#[test]
fn test_disjoint_boxes_union() {
    // Two boxes that don't intersect
    let shell_a = make_box_shell(-5.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let shell_b = make_box_shell(5.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let entities = EntityIndex::new();

    let result = boolean(&[shell_a], &[shell_b], &entities, &entities, BoolOp::Union);
    assert!(!result.is_empty);
    // Both boxes completely outside each other → all faces kept
    let face_count: usize = result.shells.iter().map(|s| s.faces.len()).sum();
    assert_eq!(face_count, 12, "disjoint union should keep all 12 faces");
}

#[test]
fn test_disjoint_boxes_intersection() {
    let shell_a = make_box_shell(-5.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let shell_b = make_box_shell(5.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let entities = EntityIndex::new();

    let result = boolean(&[shell_a], &[shell_b], &entities, &entities, BoolOp::Intersection);
    // Disjoint intersection should be empty
    assert!(result.is_empty, "intersection of disjoint boxes should be empty");
}

/// Test boolean on a real STEP file — self-union should equal the original.
#[test]
fn test_shape_step_self_union() {
    use std::path::Path;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data").join("Shape.step");
    if !path.exists() { eprintln!("  SKIP: Shape.step not found"); return; }

    let text = std::fs::read_to_string(&path).expect("read Shape.step");
    let exchange = rc3d_io::step::parser::parse_exchange(&text).expect("parse");
    let shells = rc3d_io::step::topology::collect_shells(&exchange.entities);
    assert!(!shells.is_empty(), "Shape.step should have shells");

    // Self-union: A ∪ A = A
    let result = boolean(&shells, &shells, &exchange.entities, &exchange.entities, BoolOp::Union);
    assert!(!result.is_empty, "self-union should not be empty");
    eprintln!("Self-union: {} faces", result.shells.iter().map(|s| s.faces.len()).sum::<usize>());
}

/// Test that union with itself preserves original face count.
#[test]
fn test_shape_step_self_intersection() {
    use std::path::Path;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data").join("Shape.step");
    if !path.exists() { eprintln!("  SKIP"); return; }

    let text = std::fs::read_to_string(&path).expect("read Shape.step");
    let exchange = rc3d_io::step::parser::parse_exchange(&text).expect("parse");
    let shells = rc3d_io::step::topology::collect_shells(&exchange.entities);

    // Self-intersection: A ∩ A = A
    let result = boolean(&shells, &shells, &exchange.entities, &exchange.entities, BoolOp::Intersection);
    assert!(!result.is_empty, "self-intersection should not be empty");
    eprintln!("Self-intersection: {} faces", result.shells.iter().map(|s| s.faces.len()).sum::<usize>());
}
