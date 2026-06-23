//! Mesh quality stats for Shape corpus.
//! Run: cargo test -p rc3d-io --test shape_mesh_diag --release -- --nocapture

use rc3d_core::math::Real;
use rc3d_core::math::PVec3;
use rc3d_io::step::brep::build_brep;
use rc3d_io::step::brep::geom::SurfaceGeom;
use rc3d_io::step::brep::heal::{auto_heal_shell, HealLevel};
use rc3d_io::step::brep::mesh::{mesh_brep_shell_with_report, BRepMeshConfig};
use rc3d_io::step::parser;
use std::path::Path;

fn test_data(name: &str) -> std::path::PathBuf {
    let test_data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data");
    let candidate = test_data.join(name);
    if candidate.exists() {
        return candidate;
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../steps")
        .join(name)
}

fn surface_kind(surface: &SurfaceGeom) -> &'static str {
    match surface {
        SurfaceGeom::Plane { .. } => "Plane",
        SurfaceGeom::Revolution { .. } => "Revolution",
        SurfaceGeom::BSpline(_) => "BSpline",
        SurfaceGeom::Offset { .. } => "Offset",
        SurfaceGeom::Cylinder { .. } => "Cylinder",
        SurfaceGeom::Cone { .. } => "Cone",
        SurfaceGeom::Sphere { .. } => "Sphere",
        SurfaceGeom::Torus { .. } => "Torus",
        SurfaceGeom::Extrusion { .. } => "Extrusion",
    }
}

fn mesh_quality(verts: &[PVec3], indices: &[i32], shell_diag: Real) -> (usize, Real, Real, usize) {
    let mut max_edge = 0.0_f64;
    let mut max_aspect = 0.0_f64;
    let mut sliver_count = 0usize;
    let sliver_thresh = shell_diag * 0.05;

    for chunk in indices.chunks(4) {
        if chunk.len() < 4 || chunk[3] != -1 {
            continue;
        }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() {
            continue;
        }
        let p0 = verts[i0];
        let p1 = verts[i1];
        let p2 = verts[i2];
        let e0 = (p1 - p0).length();
        let e1 = (p2 - p1).length();
        let e2 = (p0 - p2).length();
        max_edge = max_edge.max(e0).max(e1).max(e2);
        let emin = e0.min(e1).min(e2).max(1e-12);
        let emax = e0.max(e1).max(e2);
        let aspect = emax / emin;
        max_aspect = max_aspect.max(aspect);
        if emax > sliver_thresh && aspect > 50.0 {
            sliver_count += 1;
        }
    }
    (sliver_count, max_edge, max_aspect, indices.len() / 4)
}

fn run_file(step_file: &str) {
    let path = test_data(step_file);
    if !path.exists() {
        println!("SKIP {step_file}");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("read");
    let exchange = parser::parse_exchange(&text).expect("parse");
    let brep = build_brep(&exchange.entities).expect("brep");
    let mut reg = brep.registry;
    let mut skip = Vec::new();
    for &sk in &brep.root_solids {
        if let Some(s) = reg.solids.get(sk) {
            let h = auto_heal_shell(s.outer_shell, &mut reg, HealLevel::Standard, 5);
            skip.extend(h.skip_face_keys);
        }
    }
    let cfg = BRepMeshConfig::preview();
    for &sk in &brep.root_solids {
        let outer_shell = reg.solids.get(sk).expect("solid").outer_shell;
        let out = mesh_brep_shell_with_report(outer_shell, &mut reg, &cfg, &skip);
        let diag = out.report.shell_diag.max(1.0);
        let (slivers, max_e, max_a, tris) =
            mesh_quality(&out.mesh.vertices, &out.mesh.indices, diag);
        println!(
            "\n=== {step_file} tris={} verts={} slivers={} max_edge={:.2} max_aspect={:.0} grid_fb={} ===",
            tris,
            out.mesh.vertices.len(),
            slivers,
            max_e,
            max_a,
            out.report.grid_fallback_count,
        );
        let mut top: Vec<_> = out
            .report
            .faces
            .iter()
            .map(|fs| {
                let kind = reg
                    .faces
                    .get(fs.face_key)
                    .map(|f| surface_kind(&f.surface))
                    .unwrap_or("?");
                (fs.tri_count, fs.grid_fallback, fs.max_chord_error, kind, fs.face_key)
            })
            .collect();
        top.sort_by(|a, b| b.0.cmp(&a.0));
        for (tris, fb, chord, kind, fk) in top.iter().take(8) {
            println!(
                "  {:?} {} tris={} grid_fb={} chord={:.4}",
                fk, kind, tris, fb, chord
            );
        }
    }
}

#[test]
#[ignore = "diagnostic tool: no assertions, manual inspection only"]
fn shape_mesh_quality_report() {
    for f in [
        "Shape.step",
        "Shape-2.step",
        "Shape-1.step",
        "Cube.step",
        "cs.step",
        "OffsetPlaneHoleEdge.step",
    ] {
        run_file(f);
    }
}
