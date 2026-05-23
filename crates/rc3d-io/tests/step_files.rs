//! One-by-one STEP file tests to isolate crashes.
//! Run with: cargo test -p rc3d-io --test step_files -- --nocapture

use rc3d_io::parse_step_file;
use std::path::Path;

fn load(name: &str, size_mb: f64) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data").join(name);
    println!("\n=== {} ({:.1} MB) ===", name, size_mb);
    if !path.exists() { println!("  SKIP"); return; }
    let start = std::time::Instant::now();
    match parse_step_file(&path) {
        Ok(graph) => {
            let mut meshes = 0u32;
            let mut verts = 0usize;
            let mut idxs = 0usize;
            let mut stack: Vec<_> = graph.roots().to_vec();
            while let Some(id) = stack.pop() {
                if let Some(e) = graph.get(id) {
                    use rc3d_scene::NodeData;
                    match &e.data {
                        NodeData::IndexedFaceSet(ifs) => { meshes += 1; idxs += ifs.coord_index.len(); }
                        NodeData::Coordinate3(c) => { verts += c.point.len(); }
                        _ => {}
                    }
                    stack.extend(e.children.iter().copied());
                }
            }
            println!("  OK: {} roots, {} meshes, {} verts, {} idxs ({:.2}s)",
                graph.roots().len(), meshes, verts, idxs, start.elapsed().as_secs_f32());
        }
        Err(e) => println!("  FAIL ({:.2}s): {}", start.elapsed().as_secs_f32(), e),
    }
}

#[test] fn t_assembly() { load("AssemblyExample-Assembly.step", 0.5); }
#[test] fn t_shape() { load("Shape.step", 0.1); }
#[test] fn t_shape1() { load("Shape-1.step", 0.3); }
#[test] fn t_shape2() { load("Shape-2.step", 0.8); }
#[test] fn t_bender() { load("bender assembly v54.step", 32.0); }
#[test] fn t_end4() { load("end4.stp", 86.0); }
