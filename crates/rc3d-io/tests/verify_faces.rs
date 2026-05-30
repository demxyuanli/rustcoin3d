//! Verify per-face STL exports: bbox, degeneracy, vertex sanity
//! Run: cargo test -p rc3d-io --test verify_faces --release -- --nocapture

use rc3d_core::math::Vec3;
use rc3d_io::parse_stl_triangles;
use std::path::Path;
use std::fs;
use std::collections::HashSet;

fn test_data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data").join(name)
}

// Simple hash for vertex dedup
fn vhash(v: &[f32; 3]) -> u64 {
    (v[0].to_bits() as u64).wrapping_mul(31)
        .wrapping_add(v[1].to_bits() as u64).wrapping_mul(31)
        .wrapping_add(v[2].to_bits() as u64)
}

#[test]
fn verify_face_stls() {
    let faces_dir = test_data("faces");
    let mut entries: Vec<_> = fs::read_dir(&faces_dir).unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "stl").unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    println!("\n=== Per-face STL verification ===\n");
    if entries.is_empty() {
        println!("  No STL files found in {:?} — run export_mesh example with --per-face first", faces_dir);
        return;
    }
    println!("  {:45} {:>6} {:>6} {:>6} {:>6} {:>6}","file","tris","verts","diag","degen","max_edge");

    for entry in &entries {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy();
        let data = fs::read(&path).unwrap();
        let tris = match parse_stl_triangles(&data) {
            Ok(t) => t,
            Err(e) => { println!("  {} ERROR: {}", name, e); continue; }
        };

        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        let mut degens = 0usize;
        let mut vert_keys = HashSet::new();
        let mut max_edge = 0.0f32;

        for tri in &tris {
            let a = Vec3::new(tri.vertices[0][0], tri.vertices[0][1], tri.vertices[0][2]);
            let b = Vec3::new(tri.vertices[1][0], tri.vertices[1][1], tri.vertices[1][2]);
            let c = Vec3::new(tri.vertices[2][0], tri.vertices[2][1], tri.vertices[2][2]);
            mn = mn.min(a).min(b).min(c);
            mx = mx.max(a).max(b).max(c);
            vert_keys.insert(vhash(&tri.vertices[0]));
            vert_keys.insert(vhash(&tri.vertices[1]));
            vert_keys.insert(vhash(&tri.vertices[2]));
            let area = (b - a).cross(c - a).length() * 0.5;
            if area < 1e-12 { degens += 1; }
            max_edge = max_edge.max((a-b).length()).max((b-c).length()).max((c-a).length());
        }

        let diag = (mx - mn).length();
        println!("  {:45} {:>6} {:>6} {:>6.1} {:>6} {:>6.1}",
            name, tris.len(), vert_keys.len(), diag, degens, max_edge);
    }

    // Also verify the full mesh for comparison
    println!("\n=== Comparison: our full mesh vs OCCT reference ===");
    for (label, file) in [("our", "shape-our.stl"), ("occt", "shape-tri.stl")] {
        let path = test_data(file);
        let data = match fs::read(&path) {
            Ok(d) => d,
            Err(e) => { println!("  {}: {} (skip)", label, e); continue; }
        };
        let tris = match parse_stl_triangles(&data) {
            Ok(t) => t,
            Err(e) => { println!("  {}: parse error {} (skip)", label, e); continue; }
        };
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for tri in &tris {
            for v in &tri.vertices {
                let p = Vec3::new(v[0], v[1], v[2]);
                mn = mn.min(p); mx = mx.max(p);
            }
        }
        println!("  {}: {} tris, diag={:.1}, mn=({:.1},{:.1},{:.1}), mx=({:.1},{:.1},{:.1})",
            label, tris.len(), (mx-mn).length(), mn.x, mn.y, mn.z, mx.x, mx.y, mx.z);
    }
}
