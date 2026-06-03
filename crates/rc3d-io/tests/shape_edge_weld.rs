//! Regression: duplicate EdgeKeys (same vertex pair) must weld boundary samples.
//! Run: cargo test -p rc3d-io --test shape_edge_weld --release -- --nocapture
//!
//! Uses B-Rep parse + heal only (no scene emit / shell mesh) so small STEP files finish in seconds.

use rc3d_io::step::brep::build_brep;
use rc3d_io::step::brep::heal::{auto_heal_shell, HealLevel};
use rc3d_io::step::parser;
use rc3d_io::step::StepImportOptions;
use rc3d_shape::mesh::edge_disc::EdgeDiscConfig;
use rc3d_shape::store::BRepStore;
use rc3d_shape::{
    check_shell_topo_diag, measure_equivalent_edge_weld_gap, measure_face_boundary_surface_gap,
};
use std::path::Path;

fn test_data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

/// Parse STEP → BRep → heal. Does not build scene graph or tessellate faces.
fn load_healed_store(step_file: &str) -> BRepStore {
    let path = test_data(step_file);
    let text = std::fs::read_to_string(&path).expect("read step");
    let options = StepImportOptions::default();
    let exchange = parser::parse_exchange_with_options(&text, &options).expect("parse");
    let brep = build_brep(&exchange.entities).expect("brep");
    let mut reg = brep.registry;
    for &sk in &brep.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let _ = auto_heal_shell(solid.outer_shell, &mut reg, HealLevel::Standard, 5);
        }
    }
    reg
}

fn assert_shape_weld_gap(step_file: &str, max_gap: f32) {
    let path = test_data(step_file);
    if !path.exists() {
        println!("SKIP {step_file}");
        return;
    }
    let reg = load_healed_store(step_file);
    let cfg = EdgeDiscConfig::default();
    let gap = measure_equivalent_edge_weld_gap(&reg, &cfg);
    println!("{step_file}: equiv_edge_weld_gap={gap:.6}");
    assert!(
        gap <= max_gap,
        "{step_file} duplicate-edge weld gap {gap} > {max_gap}"
    );
}

fn assert_shape_boundary_on_surface(step_file: &str, max_gap: f32) {
    let path = test_data(step_file);
    if !path.exists() {
        println!("SKIP {step_file}");
        return;
    }
    let reg = load_healed_store(step_file);
    let cfg = EdgeDiscConfig::default();
    let mut max_all = 0.0f32;
    for (_sk, solid) in reg.solids.iter() {
        let gap = measure_face_boundary_surface_gap(&reg, solid.outer_shell, &cfg);
        println!(
            "{step_file} shell {:?} boundary_surface_gap={gap:.6}",
            solid.outer_shell
        );
        max_all = max_all.max(gap);
    }
    assert!(
        max_all <= max_gap,
        "{step_file} face boundary off-surface gap {max_all} > {max_gap}"
    );
}

#[test]
fn shape_boundary_on_surface() {
    assert_shape_boundary_on_surface("Shape.step", 1e-3);
}

fn assert_shape_topo_sewing(step_file: &str, max_wire_gap: f32, max_pcurve_drift: f32) {
    let path = test_data(step_file);
    if !path.exists() {
        println!("SKIP {step_file}");
        return;
    }
    let reg = load_healed_store(step_file);
    let mut max_w = 0.0f32;
    let mut max_d = 0.0f32;
    for (_sk, solid) in reg.solids.iter() {
        let diag = check_shell_topo_diag(solid.outer_shell, &reg);
        println!(
            "{step_file} shell {:?} topo max_wire_gap={:.6} max_pcurve_drift={:.6}",
            solid.outer_shell, diag.max_wire_junction_gap, diag.max_pcurve_drift
        );
        max_w = max_w.max(diag.max_wire_junction_gap);
        max_d = max_d.max(diag.max_pcurve_drift);
    }
    assert!(
        max_w <= max_wire_gap,
        "{step_file} wire junction gap {max_w} > {max_wire_gap}"
    );
    assert!(
        max_d <= max_pcurve_drift,
        "{step_file} pcurve drift {max_d} > {max_pcurve_drift}"
    );
}

#[test]
fn shape_topo_sewing_after_import() {
    assert_shape_topo_sewing("Shape.step", 0.05, 0.05);
}

#[test]
fn shape_equiv_edge_weld() {
    assert_shape_weld_gap("Shape.step", 1e-2);
}

#[test]
fn shape1_equiv_edge_weld() {
    assert_shape_weld_gap("Shape-1.step", 1e-2);
}

#[test]
fn shape2_equiv_edge_weld() {
    assert_shape_weld_gap("Shape-2.step", 1e-2);
}

#[test]
fn shape_corpus_equiv_edge_weld() {
    assert_shape_weld_gap("Shape.step", 1e-2);
    assert_shape_weld_gap("Shape-1.step", 1e-2);
    assert_shape_weld_gap("Shape-2.step", 1e-2);
}
