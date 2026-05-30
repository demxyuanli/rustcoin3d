//! Manual STEP debug probes (ignored by default).
//! Run: cargo test -p rc3d-io --test step_debug --release -- --ignored --nocapture

use std::path::Path;
use rc3d_io::step::brep::build::{build_brep, build_curve};
use rc3d_io::step::brep::geom::{CurveGeom, SurfaceGeom};
use rc3d_io::step::parser;

fn test_data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

#[test]
#[ignore]
fn debug_bspline_entities() {
    let path = test_data("Shape.step");
    let text = std::fs::read_to_string(&path).expect("read");
    let exchange = parser::parse_exchange(&text).expect("parse");
    for &eid in &[27, 26] {
        if let Some(rec) = exchange.entities.get(&eid) {
            println!("#{}: name={}", eid, rec.name);
        }
    }
    let brep = build_brep(&exchange.entities).expect("brep");
    for (ek, edge) in brep.registry.edges.iter() {
        if matches!(edge.curve_3d, CurveGeom::BSpline(_)) {
            println!("edge {:?} bspline", ek);
        }
    }
}

#[test]
#[ignore]
fn debug_inner_wall_revolution() {
    let path = test_data("Shape.step");
    let text = std::fs::read_to_string(&path).unwrap();
    let exchange = parser::parse_exchange(&text).unwrap();
    for id in [72u64, 34] {
        if let Some(c) = build_curve(id, &exchange.entities) {
            println!("curve #{id}: {:?}", std::mem::discriminant(&c));
            for t in [0.0, 0.5, 1.0] {
                let p = c.d0(t);
                println!("  d0({t:.2}) = ({:.2},{:.2},{:.2})", p.x, p.y, p.z);
            }
        }
    }
    let brep = build_brep(&exchange.entities).unwrap();
    for (fk, face) in brep.registry.faces.iter() {
        if let SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } = &face.surface {
            println!(
                "Face {:?}: Rev org=({:.1},{:.1},{:.1})",
                fk, axis_origin.x, axis_origin.y, axis_origin.z
            );
            let _ = (generatrix, axis_dir);
        }
    }
}
