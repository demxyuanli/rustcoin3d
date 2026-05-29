//! Export each face of Shape.step as a separate STL file
//! Run: cargo test -p rc3d-io --test export_faces --release -- --nocapture

use rc3d_core::math::Vec3;
use rc3d_io::step::brep::build_brep;
use rc3d_io::step::brep::heal::{auto_heal_shell, HealLevel};
use rc3d_io::step::brep::mesh::{mesh_brep_shell_with_report, BRepMeshConfig};
use rc3d_io::step::parser;
use rc3d_io::step::mesh_result::MeshResult;
use rc3d_io::step::brep::geom::SurfaceGeom;
use std::path::Path;
use std::io::Write;

fn test_data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data").join(name)
}

fn write_binary_stl(path: &Path, vertices: &[Vec3], indices: &[i32]) {
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&[0u8; 80]).unwrap();
    let tri_count = indices.len() / 4;
    f.write_all(&(tri_count as u32).to_le_bytes()).unwrap();
    for chunk in indices.chunks(4) {
        if chunk.len() < 3 { continue; }
        let v0 = vertices[chunk[0] as usize];
        let v1 = vertices[chunk[1] as usize];
        let v2 = vertices[chunk[2] as usize];
        let n = (v1 - v0).cross(v2 - v0);
        let n = if n.length_squared() > 1e-12 { n.normalize() } else { Vec3::Z };
        f.write_all(&n.x.to_le_bytes()).unwrap();
        f.write_all(&n.y.to_le_bytes()).unwrap();
        f.write_all(&n.z.to_le_bytes()).unwrap();
        for v in [v0, v1, v2] {
            f.write_all(&v.x.to_le_bytes()).unwrap();
            f.write_all(&v.y.to_le_bytes()).unwrap();
            f.write_all(&v.z.to_le_bytes()).unwrap();
        }
        f.write_all(&[0u8; 2]).unwrap();
    }
}

#[test]
fn export_per_face_stl() {
    let path = test_data("Shape.step");
    let text = std::fs::read_to_string(&path).unwrap();
    let exchange = parser::parse_exchange(&text).unwrap();
    let brep = build_brep(&exchange.entities).unwrap();
    let mut reg = brep.registry;
    let mut skip = Vec::new();
    for &sk in &brep.root_solids {
        if let Some(s) = reg.solids.get(sk) {
            let h = auto_heal_shell(s.outer_shell, &mut reg, HealLevel::Standard, 5);
            skip.extend(h.skip_face_keys);
        }
    }
    let cfg = BRepMeshConfig::default();
    let mut mesh = MeshResult::default();
    let mut report = None;
    for &sk in &brep.root_solids {
        if let Some(s) = reg.solids.get(sk) {
            let out = mesh_brep_shell_with_report(s.outer_shell, &reg, &cfg, &skip);
            report = Some(out.report);
            mesh = out.mesh;
        }
    }

    let out_dir = test_data("faces");
    std::fs::create_dir_all(&out_dir).unwrap();

    println!("\n=== Per-face STL export ===");
    for fs in &report.unwrap().faces {
        if fs.tri_count == 0 { continue; }
        let face = reg.faces.get(fs.face_key).unwrap();
        let kind = match &face.surface {
            SurfaceGeom::Revolution { .. } => "Rev",
            SurfaceGeom::BSpline(_) => "BSpl",
            SurfaceGeom::Offset { .. } => "Offs",
            SurfaceGeom::Plane { .. } => "Plane",
            _ => "Other",
        };
        let uv = match fs.uv_source {
            rc3d_io::step::brep::mesh::face_uv::UvSource::Pcurve => "pc",
            _ => "syn",
        };

        let start = fs.first_tri * 4;
        let end = (start + fs.tri_count * 4).min(mesh.indices.len());
        let face_idx: Vec<i32> = if start < mesh.indices.len() {
            mesh.indices[start..end].to_vec()
        } else {
            Vec::new()
        };

        let fk_str = format!("{:?}", fs.face_key);
        let fk_id = fk_str.trim_start_matches("FaceKey(").trim_end_matches(')');
        let name = format!("face_{}_{}{}_{}tris.stl", fk_id, kind, uv, fs.tri_count);
        write_binary_stl(&out_dir.join(&name), &mesh.vertices, &face_idx);
        let sz = std::fs::metadata(&out_dir.join(&name)).map(|m| m.len()).unwrap_or(0);
        println!("  {:?} {} {}tris first_tri={} -> {} ({}B)",
            fs.face_key, kind, fs.tri_count, fs.first_tri, name, sz);
    }
    println!("\nDone: {:?}", out_dir);
}
