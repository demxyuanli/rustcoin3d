//! One-by-one STEP file tests to isolate crashes.
//! Run with: cargo test -p rc3d-io --test step_files -- --nocapture

use rc3d_io::{parse_step_file, write_step_from_entities};
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

/// Round-trip test: parse → write (Path B) → re-parse → compare mesh stats.
#[test]
fn t_roundtrip_write_entities() {
    use rc3d_io::parse_step_file;
    use std::fs;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data").join("Shape.step");
    if !path.exists() { println!("  SKIP"); return; }

    // Phase 1: Parse original
    let text = fs::read_to_string(&path).expect("read Shape.step");
    let exchange = rc3d_io::step::parser::parse_exchange(&text).expect("parse");
    let orig_mesh = {
        let shells = rc3d_io::step::topology::collect_shells(&exchange.entities);
        rc3d_io::step::tessellate::tessellate_faces(
            &shells.iter().flat_map(|s| s.faces.iter()).cloned().collect::<Vec<_>>(),
            &exchange.entities,
        )
    };

    // Phase 2: Write via Path B
    let written = write_step_from_entities(&exchange.entities);
    assert!(written.len() > 1000, "written output too short: {} bytes", written.len());
    // Debug: save output
    let _ = std::fs::write(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/_roundtrip_debug.step"),
        &written,
    );
    // Print first line and entity count
    let entity_count = written.matches('=').count();
    println!("  Written {} bytes, ~{} entities", written.len(), entity_count);

    // Phase 3: Re-parse written output
    let exchange2 = rc3d_io::step::parser::parse_exchange(&written).expect("re-parse");
    let shells2 = rc3d_io::step::topology::collect_shells(&exchange2.entities);

    // Phase 4: Compare mesh stats
    let mesh2 = rc3d_io::step::tessellate::tessellate_faces(
        &shells2.iter().flat_map(|s| s.faces.iter()).cloned().collect::<Vec<_>>(),
        &exchange2.entities,
    );

    println!("  Original: {} vertices, {} indices", orig_mesh.vertices.len(), orig_mesh.indices.len());
    println!("  Written:  {} vertices, {} indices", mesh2.vertices.len(), mesh2.indices.len());

    // Vertex count should be within 5% (minor differences from ID renumbering)
    let v_ratio = mesh2.vertices.len() as f64 / orig_mesh.vertices.len().max(1) as f64;
    assert!(v_ratio > 0.95 && v_ratio < 1.05,
        "vertex count mismatch: orig={} written={} ratio={:.3}",
        orig_mesh.vertices.len(), mesh2.vertices.len(), v_ratio);

    println!("  ROUNDTRIP OK");
}

#[test]
fn test_shared_topology_import() {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"), "/../../test_data/AssemblyExample-Assembly.step"
    ));
    if !path.exists() {
        println!("SKIP: test data not found");
        return;
    }

    let bytes = std::fs::read(path).expect("read file");
    let text = String::from_utf8_lossy(&bytes);
    let exchange = rc3d_io::step::parser::parse_exchange(&text).expect("parse");

    // Verify HEADER parsing
    if let Some(ref header) = exchange.header {
        println!("  FILE_SCHEMA: {:?}", header.file_schema);
        println!("  FILE_NAME: {}", header.file_name.name);
    }

    // Verify validation
    let report = rc3d_io::step::validate::validate(&exchange.entities);
    println!("  Validation: {} errors, {} warnings",
        report.errors.len(), report.warnings.len());
    assert!(report.topology_info.shells > 0, "should find shells");
    assert!(report.topology_info.faces > 0, "should find faces");

    // Verify B-Rep pipeline
    let brep_result = rc3d_io::step::brep::build_brep(&exchange.entities)
        .expect("B-Rep build");
    let reg = &brep_result.registry;
    println!("  B-Rep: {} vertices, {} edges, {} faces, {} shells",
        reg.vertices.len(), reg.edges.len(), reg.faces.len(), reg.shells.len());
    assert!(reg.vertices.len() > 0, "should have vertices");
    assert!(reg.edges.len() > 0, "should have edges");
    assert!(!brep_result.root_solids.is_empty(), "should have solids");

    // Verify mesh output
    let mesh_config = rc3d_io::step::brep::mesh::BRepMeshConfig::default();
    for &sk in &brep_result.root_solids {
        let solid = reg.solids.get(sk).unwrap();
        let mesh = rc3d_io::step::brep::mesh::mesh_brep_shell(solid.outer_shell, reg, &mesh_config);
        println!("  Mesh: {} vertices, {} indices", mesh.vertices.len(), mesh.indices.len());
        assert!(!mesh.vertices.is_empty(), "should produce mesh vertices");
    }

    println!("  B-Rep PIPELINE OK");
}
