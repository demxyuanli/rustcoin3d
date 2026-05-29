//! Assembly tree and hierarchical SceneGraph import (cs.step: Cube + Sphere).
//! Run: cargo test -p rc3d-io --test step_assembly_tree --release -- --nocapture

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_io::step::assembly;
use rc3d_io::step::brep::build_brep;
use rc3d_io::step::import_options::StepImportOptions;
use rc3d_io::step::parser;
use rc3d_io::{import_step_with_options, parse_step_with_options};
use rc3d_scene::node_data::{Coordinate3Node, NodeData, TransformNode};
use rc3d_scene::SceneGraph;
use std::path::Path;

fn test_data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

fn subtree_has_indexed_face_set(graph: &SceneGraph, id: NodeId) -> bool {
    let Some(entry) = graph.get(id) else {
        return false;
    };
    if matches!(entry.data, NodeData::IndexedFaceSet(_)) {
        return true;
    }
    entry
        .children
        .iter()
        .any(|&c| subtree_has_indexed_face_set(graph, c))
}

fn count_mesh_part_children(graph: &SceneGraph, parent: NodeId) -> usize {
    graph
        .children(parent)
        .map(|kids| {
            kids.iter()
                .filter(|&&cid| subtree_has_indexed_face_set(graph, cid))
                .count()
        })
        .unwrap_or(0)
}

fn bbox_from_subtree(graph: &SceneGraph, id: NodeId) -> Option<(Vec3, Vec3)> {
    let mut mn = Vec3::splat(f32::MAX);
    let mut mx = Vec3::splat(f32::MIN);
    let mut any = false;
    fn walk(graph: &SceneGraph, id: NodeId, mn: &mut Vec3, mx: &mut Vec3, any: &mut bool) {
        let Some(entry) = graph.get(id) else {
            return;
        };
        if let NodeData::Coordinate3(Coordinate3Node { point }) = &entry.data {
            for &p in point {
                mn.x = mn.x.min(p.x);
                mn.y = mn.y.min(p.y);
                mn.z = mn.z.min(p.z);
                mx.x = mx.x.max(p.x);
                mx.y = mx.y.max(p.y);
                mx.z = mx.z.max(p.z);
                *any = true;
            }
        }
        for &c in &entry.children {
            walk(graph, c, mn, mx, any);
        }
    }
    walk(graph, id, &mut mn, &mut mx, &mut any);
    if any {
        Some((mn, mx))
    } else {
        None
    }
}

#[test]
fn assembly_tree_cs_step_products_and_shells() {
    let path = test_data("cs.step");
    if !path.exists() {
        println!("SKIP: cs.step not found");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("read cs.step");
    let exchange = parser::parse_exchange_with_options(&text, &StepImportOptions::default())
        .expect("parse");
    let tree = assembly::build_assembly_tree(&exchange.entities);
    assert!(
        tree.nodes.len() >= 3,
        "expected root + Cube + Sphere products, got {}",
        tree.nodes.len()
    );
    let with_shells: Vec<_> = tree
        .nodes
        .iter()
        .filter(|n| !n.shells.is_empty())
        .collect();
    assert!(
        with_shells.len() >= 2,
        "expected >=2 assembly nodes with shells, got {}",
        with_shells.len()
    );
    let instances = assembly::extract_shell_instances(&exchange.entities);
    assert!(
        instances.len() >= 2,
        "expected >=2 shell instances (Cube+Sphere), got {}",
        instances.len()
    );
    let brep = build_brep(&exchange.entities).expect("brep");
    assert!(brep.root_solids.len() >= 2);
    let product_names: Vec<_> = with_shells.iter().map(|n| n.name.as_str()).collect();
    assert!(
        product_names.iter().any(|n| n.contains("Cube")),
        "missing Cube product in tree"
    );
    assert!(
        product_names.iter().any(|n| n.contains("Sphere")),
        "missing Sphere product in tree"
    );
}

#[test]
fn hierarchical_scene_graph_cs_step() {
    let path = test_data("cs.step");
    if !path.exists() {
        println!("SKIP: cs.step not found");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("read cs.step");
    let graph = parse_step_with_options(&text, &StepImportOptions::default()).expect("import");

    let root = graph.roots()[0];
    let mesh_parent = graph
        .children(root)
        .and_then(|kids| {
            kids.iter().find_map(|&cid| {
                if count_mesh_part_children(&graph, cid) >= 2 {
                    Some(cid)
                } else {
                    None
                }
            })
        })
        .unwrap_or(root);

    let part_count = count_mesh_part_children(&graph, mesh_parent);
    assert!(
        part_count >= 2,
        "expected >=2 mesh parts under assembly container, got {part_count}"
    );

    let mut part_bboxes = Vec::new();
    if let Some(kids) = graph.children(mesh_parent) {
        for &cid in kids {
            if subtree_has_indexed_face_set(&graph, cid) {
                if let Some(bb) = bbox_from_subtree(&graph, cid) {
                    part_bboxes.push(bb);
                }
            }
        }
    }
    assert!(
        part_bboxes.len() >= 2,
        "expected per-part bboxes, got {}",
        part_bboxes.len()
    );

    let (a_min, a_max) = part_bboxes[0];
    let (b_min, b_max) = part_bboxes[1];
    let a_center = (a_min + a_max) * 0.5;
    let b_center = (b_min + b_max) * 0.5;
    let a_diag = (a_max - a_min).length();
    let b_diag = (b_max - b_min).length();
    assert!(a_diag > 1.0 && b_diag > 1.0, "parts should have non-trivial extent");

    // Cube vs sphere: distinct extent or center (both may share origin placement).
    let separated = (a_center - b_center).length() > 0.5
        || (a_diag - b_diag).abs() > 2.0;
    assert!(
        separated,
        "parts should differ in size or placement: centers {:?} {:?} diags {:.2} {:.2}",
        a_center,
        b_center,
        a_diag,
        b_diag
    );
}

#[test]
fn import_result_exposes_assembly_tree_with_product_ids() {
    let path = test_data("cs.step");
    if !path.exists() {
        println!("SKIP: cs.step not found");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("read cs.step");
    let result =
        import_step_with_options(&text, &StepImportOptions::default()).expect("import");
    assert!(result.report.assembly_node_count >= 2);
    let with_shells: Vec<_> = result
        .assembly_tree
        .nodes
        .iter()
        .filter(|n| !n.shells.is_empty())
        .collect();
    assert!(with_shells.len() >= 2);
    assert!(
        with_shells.iter().any(|n| n.product_id > 0),
        "assembly nodes should carry STEP product_id"
    );
    assert!(!result.entities.is_empty());
}

fn subtree_transforms(graph: &SceneGraph, id: NodeId) -> Vec<Mat4> {
    let mut out = Vec::new();
    fn walk(graph: &SceneGraph, id: NodeId, parent: Mat4, out: &mut Vec<Mat4>) {
        let Some(entry) = graph.get(id) else {
            return;
        };
        let local = match &entry.data {
            NodeData::Transform(TransformNode { rotation, .. }) => *rotation,
            _ => Mat4::IDENTITY,
        };
        let world = parent * local;
        if matches!(entry.data, NodeData::Transform(_)) {
            out.push(world);
        }
        for &c in &entry.children {
            walk(graph, c, world, out);
        }
    }
    walk(graph, id, Mat4::IDENTITY, &mut out);
    out
}

#[test]
fn hierarchical_scene_graph_carries_part_transforms() {
    let path = test_data("cs.step");
    if !path.exists() {
        println!("SKIP: cs.step not found");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("read");
    let graph = parse_step_with_options(&text, &StepImportOptions::default()).expect("import");
    let root = graph.roots()[0];
    let transforms = subtree_transforms(&graph, root);
    // cs.step may use identity placement; hierarchy must still attach Transform nodes per part.
    let part_count = graph
        .children(root)
        .map(|kids| {
            kids.iter()
                .filter(|&&cid| {
                    graph
                        .children(cid)
                        .map(|c| {
                            c.iter().any(|&gc| {
                                matches!(graph.get(gc).map(|e| &e.data), Some(NodeData::Transform(_)))
                            })
                        })
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0);
    assert!(part_count >= 2, "expected hierarchical part separators");
}
