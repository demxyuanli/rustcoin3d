//! Tier-2 freeform acceptance: Shape / Shape-1 / Shape-2 (T2 progress gate).
//! Run: cargo test -p rc3d-io --test shape_corpus --release -- --test-threads=1

use rc3d_core::math::Vec3;
use rc3d_io::step::brep::build_brep;
use rc3d_io::step::brep::heal::{auto_heal_shell, HealLevel};
use rc3d_io::step::brep::mesh::{mesh_brep_shell_with_report, BRepMeshConfig};
use rc3d_io::step::brep::mesh::report::deflection_from_report;
use rc3d_io::step::brep::{deflection_within_band, hausdorff_meshes};
use rc3d_io::step::brep::mesh::t4_quality::DeflectionMetrics;
use rc3d_io::step::adapter::AdapterMode;
use rc3d_io::step::import_options::StepImportOptions;
use rc3d_io::step::mesh_result::MeshResult;
use rc3d_io::step::parser;
use rc3d_io::{parse_stl_triangles};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct ShapeExpect {
    file: &'static str,
    min_tris: usize,
    min_verts: usize,
    min_face_ratio: usize,
}

struct CorpusRun {
    tris: usize,
    verts: usize,
    meshed: usize,
    total_faces: usize,
    grid_fallback_rate: f32,
    deflection: DeflectionMetrics,
    secs: f32,
}

fn test_data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

/// Single B-Rep parse → heal → mesh path (no scene graph / edge overlay).
fn run_corpus_brep(file: &str) -> CorpusRun {
    run_corpus_brep_adapter(file, AdapterMode::CompatMerge)
}

fn run_corpus_brep_adapter(file: &str, adapter_mode: AdapterMode) -> CorpusRun {
    let path = test_data(file);
    assert!(path.exists(), "missing test data: {file}");
    let start = Instant::now();
    let text = std::fs::read_to_string(&path).expect("read step");
    let options = StepImportOptions {
        adapter_mode,
        ..StepImportOptions::default()
    };
    let exchange = parser::parse_exchange_with_options(&text, &options).expect("parse");
    let brep = build_brep(&exchange.entities).expect("brep");
    let mut reg = brep.registry;
    let mut skip_face_keys = Vec::new();
    for &sk in &brep.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let heal = auto_heal_shell(solid.outer_shell, &mut reg, HealLevel::Standard, 5);
            skip_face_keys.extend(heal.skip_face_keys);
        }
    }
    let mesh_config = BRepMeshConfig::default();
    let mut tris = 0usize;
    let mut verts = 0usize;
    let mut meshed = 0usize;
    let mut total_faces = 0usize;
    let mut grid_fallback = 0usize;
    let mut deflection = DeflectionMetrics::default();
    for &sk in &brep.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let out = mesh_brep_shell_with_report(solid.outer_shell, &reg, &mesh_config, &skip_face_keys);
            total_faces += out.report.face_count;
            grid_fallback += out.report.grid_fallback_count;
            meshed += out.report.meshed_faces;
            tris += out.mesh.indices.len() / 4;
            verts += out.mesh.vertices.len();
            let mesh = out.mesh;
            deflection = deflection_from_report(&out.report);
            try_hausdorff_vs_reference(file, &mesh);
            if std::env::var("SHAPE_FACE_DIAG").is_ok() && file == "Shape.step" {
                use rc3d_io::step::brep::geom::SurfaceGeom;
                eprintln!("--- Shape.step per-face mesh ---");
                for fs in &out.report.faces {
                    let face = reg.faces.get(fs.face_key).expect("face");
                    let wire_n = reg
                        .wires
                        .get(face.outer_wire)
                        .map(|w| w.edges.len())
                        .unwrap_or(0);
                    let kind = match &face.surface {
                        SurfaceGeom::Plane { .. } => "Plane",
                        SurfaceGeom::Revolution { .. } => "Revolution",
                        SurfaceGeom::BSpline(_) => "BSpline",
                        SurfaceGeom::Offset { .. } => "Offset",
                        SurfaceGeom::Cylinder { .. } => "Cylinder",
                        SurfaceGeom::Cone { .. } => "Cone",
                        SurfaceGeom::Sphere { .. } => "Sphere",
                        SurfaceGeom::Torus { .. } => "Torus",
                        SurfaceGeom::Extrusion { .. } => "Extrusion",
                    };
                    eprintln!(
                        "  {:?} {} wire_edges={} tris={} uv={:?} grid_fb={} chord={:.4}",
                        fs.face_key,
                        kind,
                        wire_n,
                        fs.tri_count,
                        fs.uv_source,
                        fs.grid_fallback,
                        fs.max_chord_error,
                    );
                }
                eprintln!("--- mesh bbox ---");
                let mut mn = Vec3::splat(f32::MAX);
                let mut mx = Vec3::splat(f32::MIN);
                for v in &mesh.vertices {
                    mn = mn.min(*v);
                    mx = mx.max(*v);
                }
                eprintln!("  min={:?} max={:?} diag={:.3}", mn, mx, (mx - mn).length());
            }
            for fs in &out.report.faces {
                if !fs.grid_fallback {
                    assert!(
                        fs.tri_count > 0,
                        "{}: face {:?} uv {:?}",
                        file,
                        fs.face_key,
                        fs.uv_source,
                    );
                }
            }
        }
    }
    let rate = if total_faces > 0 {
        grid_fallback as f32 / total_faces as f32
    } else {
        0.0
    };
    CorpusRun {
        tris,
        verts,
        meshed,
        total_faces,
        grid_fallback_rate: rate,
        deflection,
        secs: start.elapsed().as_secs_f32(),
    }
}

fn ref_stl_path(step_file: &str) -> PathBuf {
    test_data("ref").join(format!("{step_file}.stl"))
}

fn try_hausdorff_vs_reference(step_file: &str, engine: &MeshResult) {
    let ref_path = ref_stl_path(step_file);
    if !ref_path.exists() {
        return;
    }
    let data = std::fs::read(&ref_path).expect("read reference stl");
    let tris = parse_stl_triangles(&data).expect("parse reference stl");
    let mut ref_mesh = MeshResult::default();
    for tri in tris {
        let base = ref_mesh.vertices.len() as i32;
        ref_mesh.vertices.push(Vec3::from(tri.vertices[0]));
        ref_mesh.vertices.push(Vec3::from(tri.vertices[1]));
        ref_mesh.vertices.push(Vec3::from(tri.vertices[2]));
        ref_mesh.indices.extend_from_slice(&[base, base + 1, base + 2, -1]);
    }
    let h = hausdorff_meshes(engine, &ref_mesh, 512);
    println!(
        "  {step_file} T4 Hausdorff vs ref: sym_p95={:.4} sym_max={:.4} ({} samples)",
        h.symmetric_p95, h.symmetric_max, h.sample_count
    );
}

fn assert_shape(expect: &ShapeExpect, run: &CorpusRun, mesh_config: &BRepMeshConfig) {
    println!(
        "  {}: {} tris, {} verts, {}/{} faces meshed, grid_fallback {:.1}%, T4 p95={:.4} max={:.4}, {:.2}s",
        expect.file,
        run.tris,
        run.verts,
        run.meshed,
        run.total_faces,
        run.grid_fallback_rate * 100.0,
        run.deflection.p95,
        run.deflection.max,
        run.secs
    );
    assert!(
        run.tris >= expect.min_tris,
        "{}: expected >= {} tris, got {}",
        expect.file,
        expect.min_tris,
        run.tris
    );
    assert!(
        run.verts >= expect.min_verts,
        "{}: expected >= {} verts, got {}",
        expect.file,
        expect.min_verts,
        run.verts
    );
    assert!(
        run.tris >= expect.min_face_ratio,
        "{}: boundary-only? {} tris (need >> face count)",
        expect.file,
        run.tris
    );
    assert!(
        run.grid_fallback_rate < 0.95,
        "{}: grid_fallback_rate {:.1}% exceeds 95% target",
        expect.file,
        run.grid_fallback_rate * 100.0
    );
    let band = mesh_config.face.deflection_interior * 2.0;
    if !deflection_within_band(&run.deflection, mesh_config) {
        eprintln!(
            "[T4 warn] {} deflection p95 {:.4} max {:.4} exceeds band {:.4}",
            expect.file, run.deflection.p95, run.deflection.max, band
        );
    }
}

/// CompatMerge vs StrictFidelity: entity count and keyword parity on T2 corpus (B-lax params).
#[test]
fn strict_fidelity_index_parity() {
    const FILES: [&str; 3] = ["Shape.step", "Shape-1.step", "Shape-2.step"];
    for file in FILES {
        let path = test_data(file);
        if !path.exists() {
            eprintln!("SKIP: {file} not found");
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read step");

        let compat_opts = StepImportOptions {
            adapter_mode: AdapterMode::CompatMerge,
            ..StepImportOptions::default()
        };
        let strict_opts = StepImportOptions {
            adapter_mode: AdapterMode::StrictFidelity,
            ..StepImportOptions::default()
        };
        let compat_ex =
            parser::parse_exchange_with_options(&text, &compat_opts).expect("compat parse");
        let strict_ex =
            parser::parse_exchange_with_options(&text, &strict_opts).expect("strict parse");

        assert_eq!(
            compat_ex.entities.len(),
            strict_ex.entities.len(),
            "{file}: entity count"
        );
        let mut param_diffs = 0usize;
        for (id, rec) in &compat_ex.entities {
            let other = strict_ex
                .entities
                .get(id)
                .unwrap_or_else(|| panic!("{file}: strict missing #{id}"));
            assert_eq!(rec.name, other.name, "{file}: #{id} keyword");
            if rec.params != other.params {
                param_diffs += 1;
            }
        }
        println!(
            "  {file}: index parity OK ({} entities, {} param diffs logged)",
            compat_ex.entities.len(),
            param_diffs
        );
    }
}

/// Optional StrictFidelity mesh ratio gate (not required for release).
#[test]
#[ignore = "Strict mesh not a release gate; see step-part21-gaps spec"]
fn strict_fidelity_mesh_optional() {
    const FILES: [&str; 3] = ["Shape.step", "Shape-1.step", "Shape-2.step"];
    const TRI_RATIO_MIN: f64 = 0.85;
    const TRI_RATIO_MAX: f64 = 1.15;
    const MIN_STRICT_MESH_FRAC: f64 = 0.50;

    let mut mesh_gated = 0usize;
    for file in FILES {
        let path = test_data(file);
        if !path.exists() {
            continue;
        }
        let compat_run = run_corpus_brep_adapter(file, AdapterMode::CompatMerge);
        assert!(compat_run.tris > 0, "{file}: CompatMerge must produce triangles");

        let strict_run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_corpus_brep_adapter(file, AdapterMode::StrictFidelity)
        }));
        let strict_run = match strict_run {
            Ok(run) => run,
            Err(_) => {
                eprintln!("  {file}: StrictFidelity mesh panicked");
                continue;
            }
        };
        if strict_run.tris == 0 {
            continue;
        }
        let frac = strict_run.tris as f64 / compat_run.tris as f64;
        if frac < MIN_STRICT_MESH_FRAC {
            continue;
        }
        let ratio = frac;
        if ratio >= TRI_RATIO_MIN && ratio <= TRI_RATIO_MAX {
            mesh_gated += 1;
        }
    }
    assert!(mesh_gated >= 1, "at least one file in optional mesh band");
}

#[test]
fn t2_shape_bottle() {
    println!("\n=== T2 Shape.step ===");
    let run = run_corpus_brep("Shape.step");
    let mesh_config = BRepMeshConfig::default();
    assert_shape(
        &ShapeExpect {
            file: "Shape.step",
            min_tris: 100,
            min_verts: 500,
            min_face_ratio: 50,
        },
        &run,
        &mesh_config,
    );
}

#[test]
fn t2_shape1() {
    println!("\n=== T2 Shape-1.step ===");
    let run = run_corpus_brep("Shape-1.step");
    assert_shape(
        &ShapeExpect {
            file: "Shape-1.step",
            min_tris: 2000,
            min_verts: 3000,
            min_face_ratio: 600,
        },
        &run,
        &BRepMeshConfig::default(),
    );
}

#[test]
fn t2_shape2() {
    println!("\n=== T2 Shape-2.step ===");
    let run = run_corpus_brep("Shape-2.step");
    assert_shape(
        &ShapeExpect {
            file: "Shape-2.step",
            min_tris: 3800,
            min_verts: 2000,  // Quantized dedup produces ~2423 verts; threshold with margin
            min_face_ratio: 2800,
        },
        &run,
        &BRepMeshConfig::default(),
    );
}
