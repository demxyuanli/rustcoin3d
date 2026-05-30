//! Export STEP or STL inputs to ASCII STL.
//!
//! Usage:
//!   cargo run -p rc3d-examples --example export_mesh -- <input.step|input.stl> [output.stl]
//!   cargo run -p rc3d-examples --example export_mesh -- Shape.step out/Shape.stl --per-face=out/faces
//!   cargo run -p rc3d-examples --example export_mesh -- Shape.step --no-heal
//!
//! Options:
//!   --per-face=DIR   Also write one ASCII STL per meshed B-Rep face
//!   --no-heal        Skip auto_heal_shell before meshing (STEP only)
//!   --adapter=MODE   compat (default) or strict

use std::env;
use std::path::{Path, PathBuf};

use rc3d_io::step::adapter::AdapterMode;
use rc3d_io::step::import_options::StepImportOptions;
use rc3d_io::{default_stl_output, export_file_to_ascii_stl, MeshExportOptions};

fn main() {
    let args: Vec<String> = env::args().collect();
    let positional: Vec<&str> = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .collect();

    let Some(input_arg) = positional.first() else {
        print_usage();
        return;
    };

    let input = Path::new(input_arg);
    if !input.exists() {
        eprintln!("Input not found: {}", input.display());
        std::process::exit(1);
    }

    let output = positional
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_stl_output(input));

    let per_face_dir = args
        .iter()
        .find_map(|a| a.strip_prefix("--per-face="))
        .map(PathBuf::from);

    let no_heal = args.iter().any(|a| a == "--no-heal");
    let adapter_mode = args
        .iter()
        .find_map(|a| a.strip_prefix("--adapter="))
        .map(|v| match v.to_lowercase().as_str() {
            "strict" => AdapterMode::StrictFidelity,
            "compat" | "compatmerge" => AdapterMode::CompatMerge,
            other => {
                eprintln!("Unknown adapter mode '{other}', using compat");
                AdapterMode::CompatMerge
            }
        })
        .unwrap_or(AdapterMode::CompatMerge);

    let options = MeshExportOptions {
        import_options: StepImportOptions {
            adapter_mode,
            ..StepImportOptions::default()
        },
        heal: !no_heal,
        ..MeshExportOptions::default()
    };

    match export_file_to_ascii_stl(input, &output, &options, per_face_dir.as_deref()) {
        Ok(summary) => {
            println!("Input:  {}", summary.input.display());
            println!("Output: {}", summary.output.display());
            println!(
                "Mesh: {} tris, {} verts, {}/{} faces meshed, grid_fallback {:.1}%",
                summary.tri_count,
                summary.vert_count,
                summary.meshed_faces,
                summary.total_faces,
                summary.grid_fallback_rate * 100.0,
            );
            if summary.per_face_files > 0 {
                println!("Per-face files: {}", summary.per_face_files);
            }
            if let Some(dir) = per_face_dir {
                println!("Per-face dir: {}", dir.display());
            }
        }
        Err(e) => {
            eprintln!("Export failed: {e}");
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage: export_mesh <input.step|input.stl> [output.stl] [--per-face=DIR] [--no-heal] [--adapter=compat|strict]"
    );
    eprintln!("Example:");
    eprintln!("  cargo run -p rc3d-examples --example export_mesh -- test_data/Shape.step test_output/Shape.stl");
}
