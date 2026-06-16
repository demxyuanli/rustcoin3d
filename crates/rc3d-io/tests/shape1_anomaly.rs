//! Per-face anomaly scan for Shape-1.step — locates faces emitting out-of-bbox vertices.
//! Run: cargo test -p rc3d-io --test shape1_anomaly --release -- --nocapture

use rc3d_core::math::PVec3;
use rc3d_io::step::brep::build_brep;
use rc3d_io::step::brep::geom::SurfaceGeom;
use rc3d_io::step::brep::heal::{auto_heal_shell, HealLevel};
use rc3d_io::step::brep::mesh::{mesh_brep_shell_with_report, BRepMeshConfig};
use rc3d_io::step::parser;
use std::collections::HashSet;
use std::path::Path;

fn test_data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
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

fn mesh_bbox(verts: &[PVec3]) -> (PVec3, PVec3) {
    let mut mn = PVec3::splat(f64::MAX);
    let mut mx = PVec3::splat(f64::MIN);
    for v in verts {
        mn = mn.min(*v);
        mx = mx.max(*v);
    }
    (mn, mx)
}

fn face_vertex_indices(indices: &[i32], first_tri: usize, tri_count: usize) -> HashSet<usize> {
    let mut out = HashSet::new();
    for ti in first_tri..first_tri.saturating_add(tri_count) {
        let base = ti * 4;
        if base + 3 >= indices.len() {
            break;
        }
        for k in 0..3 {
            let gi = indices[base + k];
            if gi >= 0 {
                out.insert(gi as usize);
            }
        }
    }
    out
}

#[test]
fn shape1_per_face_anomaly_scan() {
    let path = test_data("Shape-1.step");
    if !path.exists() {
        println!("SKIP: Shape-1.step not found");
        return;
    }

    let text = std::fs::read_to_string(&path).expect("read step");
    let exchange = parser::parse_exchange(&text).expect("parse");
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
    let mut combined_verts = Vec::new();
    let mut combined_indices = Vec::new();

    for &sk in &brep.root_solids {
        let solid = reg.solids.get(sk).expect("solid");
        let out = mesh_brep_shell_with_report(
            solid.outer_shell,
            &reg,
            &mesh_config,
            &skip_face_keys,
        );

        let offset = combined_verts.len();
        combined_verts.extend_from_slice(&out.mesh.vertices);
        for &idx in &out.mesh.indices {
            if idx == -1 {
                combined_indices.push(-1);
            } else {
                combined_indices.push(idx + offset as i32);
            }
        }

        let mut brep_mn = PVec3::splat(f64::MAX);
        let mut brep_mx = PVec3::splat(f64::MIN);
        let mut bad_brep_verts = 0usize;
        for (vk, v) in reg.vertices.iter() {
            brep_mn = brep_mn.min(v.position);
            brep_mx = brep_mx.max(v.position);
            if v.position.x.abs() > 500.0
                || v.position.y.abs() > 500.0
                || v.position.z.abs() > 500.0
            {
                bad_brep_verts += 1;
                if bad_brep_verts <= 20 {
                    println!("  bad BRep vertex {:?} pos={:?}", vk, v.position);
                }
            }
        }
        let brep_diag = (brep_mx - brep_mn).length().max(1.0);
        let limit = 500.0_f64;

        let (mn, mx) = mesh_bbox(&combined_verts);
        let mesh_diag = (mx - mn).length();

        println!(
            "\n=== Shape-1 shell {:?} brep_bbox min={:?} max={:?} diag={:.2} ===",
            sk, brep_mn, brep_mx, brep_diag
        );
        println!(
            "  mesh_bbox min={:?} max={:?} diag={:.2} limit={:.2}",
            mn, mx, mesh_diag, limit
        );
        println!(
            "  {} faces, {} tris, grid_fallback={}, bad_brep_verts={}",
            out.report.face_count,
            out.report.total_tris,
            out.report.grid_fallback_count,
            bad_brep_verts
        );

        let mut anomaly_faces = 0usize;
        for fs in &out.report.faces {
            if skip_face_keys.contains(&fs.face_key) {
                continue;
            }
            let face = reg.faces.get(fs.face_key).expect("face");
            let wire_n = reg
                .wires
                .get(face.outer_wire)
                .map(|w| w.edges.len())
                .unwrap_or(0);
            let gis = face_vertex_indices(&combined_indices, fs.first_tri, fs.tri_count);
            let mut bad = 0usize;
            let mut worst = PVec3::ZERO;
            let mut worst_abs = 0.0_f64;
            for gi in &gis {
                if *gi >= combined_verts.len() {
                    bad += 1;
                    continue;
                }
                let p = combined_verts[*gi];
                let a = p.x.abs().max(p.y.abs()).max(p.z.abs());
                if a > worst_abs {
                    worst_abs = a;
                    worst = p;
                }
                if p.x.abs() > limit || p.y.abs() > limit || p.z.abs() > limit {
                    bad += 1;
                }
            }
            if bad > 0 {
                anomaly_faces += 1;
                println!(
                    "  ANOMALY {:?} {} wire={} tris={} uv={:?} grid_fb={} bad_verts={}/{} worst={:?}",
                    fs.face_key,
                    surface_kind(&face.surface),
                    wire_n,
                    fs.tri_count,
                    fs.uv_source,
                    fs.grid_fallback,
                    bad,
                    gis.len(),
                    worst,
                );
            }
        }
        println!("  anomaly_faces={}/{}", anomaly_faces, out.report.meshed_faces);

        let total_bad = combined_verts
            .iter()
            .filter(|p| p.x.abs() > limit || p.y.abs() > limit || p.z.abs() > limit)
            .count();
        println!("  total_bad_vertices={}/{}", total_bad, combined_verts.len());
        assert_eq!(
            bad_brep_verts, 0,
            "BRep vertices must stay within +/-500 after heal (was seam v-range bug)"
        );
        assert_eq!(
            total_bad, 0,
            "mesh vertices must stay within trim bbox limit"
        );
    }
}
