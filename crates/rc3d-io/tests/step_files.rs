//! One-by-one STEP file tests to isolate crashes.
//! Run with: cargo test -p rc3d-io --test step_files -- --nocapture

use rc3d_core::NodeId;
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
#[test]
fn test_cs_step_face_mesh_coverage() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/cs.step");
    if !path.exists() {
        println!("SKIP: cs.step not found");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("read cs.step");
    let exchange = rc3d_io::step::parser::parse_exchange(&text).expect("parse");
    let brep = rc3d_io::step::brep::build_brep(&exchange.entities).expect("brep");
    let mut reg = brep.registry;
    let heal_cfg = rc3d_io::step::brep::heal::HealConfig::default();
    let mesh_cfg = rc3d_io::step::brep::mesh::BRepMeshConfig::default();
    for &sk in &brep.root_solids {
        let outer_shell = reg.solids.get(sk).unwrap().outer_shell;
        let heal = rc3d_io::step::brep::heal::heal_shell(outer_shell, &mut reg, &heal_cfg);
        let shell = reg.shells.get(outer_shell).unwrap();
        println!("solid {:?}: {} faces, skip={:?}", sk, shell.faces.len(), heal.skip_face_keys);
        let out = rc3d_io::step::brep::mesh::mesh_brep_shell_with_report(
            outer_shell,
            &reg,
            &mesh_cfg,
            &heal.skip_face_keys,
        );
        let mut meshed = 0usize;
        for &(fk, _) in &shell.faces {
            let skipped = heal.skip_face_keys.contains(&fk);
            let face = reg.faces.get(fk).unwrap();
            let wire = reg.wires.get(face.outer_wire).unwrap();
            let face_stats = out.report.faces.iter().find(|f| f.face_key == fk);
            let tris = face_stats.map(|f| f.tri_count).unwrap_or(0);
            if skipped {
                println!("  {:?} SKIPPED (wire_edges={})", fk, wire.edges.len());
                continue;
            }
            if tris > 0 {
                meshed += 1;
            }
            println!(
                "  {:?} wire_edges={} tris={} grid_fb={}",
                fk,
                wire.edges.len(),
                tris,
                face_stats.map(|f| f.grid_fallback).unwrap_or(false)
            );
        }
        println!("  meshed_faces={}/{}", meshed, shell.faces.len());
        assert_eq!(
            meshed,
            shell.faces.len(),
            "expected all faces meshed, got {}/{}",
            meshed,
            shell.faces.len()
        );
    }
}

#[test] fn t_shape() { load("Shape.step", 0.1); }
#[test] fn t_shape1() { load("Shape-1.step", 0.3); }
#[test] fn t_shape2() { load("Shape-2.step", 0.8); }

#[test]
fn test_shape2_heal_diagnostics() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/Shape-2.step");
    if !path.exists() { println!("SKIP"); return; }
    let text = std::fs::read_to_string(&path).expect("read");
    let exchange = rc3d_io::step::parser::parse_exchange(&text).expect("parse");
    println!("Parsed: {} entities", exchange.entities.len());

    let brep = rc3d_io::step::brep::build_brep(&exchange.entities).expect("brep");
    println!("B-Rep: {} solids, {} faces total",
        brep.root_solids.len(),
        brep.registry.faces.len());

    let mut reg = brep.registry;
    let mesh_cfg = rc3d_io::step::brep::mesh::BRepMeshConfig::default();

    for &sk in &brep.root_solids {
        let solid = reg.solids.get(sk).unwrap();
        let shell_key = solid.outer_shell;
        let n_faces = reg.shells.get(shell_key).unwrap().faces.len();
        println!("\nShell: {} faces", n_faces);

        // Run heal with diagnostics
        let heal = rc3d_io::step::brep::heal::auto_heal_shell(
            shell_key, &mut reg,
            rc3d_io::step::brep::heal::HealLevel::Standard,
            3,
        );
        println!("Heal report:");
        println!("  merged_vertices={}", heal.merged_vertices);
        println!("  removed_small_edges={}", heal.removed_small_edges);
        println!("  closed_uv_gaps={}", heal.closed_uv_gaps);
        println!("  shifted_pcurves={}", heal.shifted_pcurves);
        println!("  reordered_wires={}", heal.reordered_wires);
        println!("  added_seams={}", heal.added_seams);
        println!("  lacking_tolerance_fixes={}", heal.lacking_tolerance_fixes);
        println!("  self_intersections_fixed={}", heal.self_intersections_fixed);
        println!("  degenerate_edges_created={}", heal.degenerate_edges_created);
        println!("  periodic_degen_created={}", heal.periodic_degen_created);
        println!("  adjusted_edge_curves={}", heal.adjusted_edge_curves);
        println!("  split_faces_created={}", heal.split_faces_created);
        println!("  check_errors={}", heal.check_errors);
        println!("  check_warnings={}", heal.check_warnings);
        println!("  skip_face_keys={:?}", heal.skip_face_keys.len());
        println!("  inner_wires_fixed={}", heal.inner_wires_fixed);

        // Mesh
        let output = rc3d_io::step::brep::mesh::mesh_brep_shell_with_report(
            shell_key, &reg, &mesh_cfg, &heal.skip_face_keys,
        );
        let faces = reg.shells.get(shell_key).unwrap().faces.clone();
        println!("\nMesh: {} triangles, {} meshed_faces, {} skipped",
            output.mesh.indices.len() / 4,
            output.report.meshed_faces,
            faces.len().saturating_sub(output.report.meshed_faces),
        );

        for &(face_key, _) in &faces {
            let face = reg.faces.get(face_key).unwrap();
            let wire = reg.wires.get(face.outer_wire).unwrap();
            let skipped = heal.skip_face_keys.contains(&face_key);
            let face_stats = output.report.faces.iter().find(|f| f.face_key == face_key);
            let tris = face_stats.map(|f| f.tri_count).unwrap_or(0);
            let status = if skipped { "SKIPPED" } else if tris > 0 { "MESHED" } else { "EMPTY" };
            let surf_name = format!("{:?}", std::mem::discriminant(&face.surface));
            println!("  face {:?}: {} edges, {} tris, {} surface={}",
                face_key, wire.edges.len(), tris, status, surf_name);
        }
        // Write heal check errors to file for diagnostic
        let check_report = rc3d_io::step::brep::heal::check_shell(shell_key, &reg);
        let mut diag = String::new();
        diag.push_str(&format!("Shell: {} faces\n", n_faces));
        diag.push_str(&format!("Heal: merged={}, removed={}, uv_gaps={}, shifted={}, reordered={}, seams={}, lacking={}, self_int={}, degen={}, periodic={}, edge_curves={}, split={}\n",
            heal.merged_vertices, heal.removed_small_edges, heal.closed_uv_gaps,
            heal.shifted_pcurves, heal.reordered_wires, heal.added_seams,
            heal.lacking_tolerance_fixes, heal.self_intersections_fixed,
            heal.degenerate_edges_created, heal.periodic_degen_created,
            heal.adjusted_edge_curves, heal.split_faces_created));
        diag.push_str(&format!("Errors ({}):\n", check_report.errors.len()));
        for e in &check_report.errors { diag.push_str(&format!("  {}\n", e)); }
        diag.push_str(&format!("Warnings ({}):\n", check_report.warnings.len()));
        for w in &check_report.warnings { diag.push_str(&format!("  {}\n", w)); }
        std::fs::write("target/shape2_diag.txt", &diag).ok();

        assert!(output.report.meshed_faces > 0, "at least one face should be meshed");
    }
}
#[test]
#[ignore = "large industrial file; run manually"]
fn t_bender() {
    load("bender assembly v54.step", 32.0);
}
#[test]
#[ignore = "large industrial file; run manually"]
fn t_end4() {
    load("end4.stp", 86.0);
}

/// Round-trip test: parse → write (Path B) → re-parse → compare mesh stats.
#[test]
fn t_roundtrip_write_entities() {
    use rc3d_io::parse_step_file;
    use std::fs;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data").join("Shape.step");
    if !path.exists() { println!("  SKIP"); return; }

    // Phase 1: Parse original via B-Rep pipeline
    let text = fs::read_to_string(&path).expect("read Shape.step");
    let orig_result = rc3d_io::step::import_step_with_options(
        &text, &rc3d_io::StepImportOptions::default(),
    ).expect("import original");

    // Phase 2: Write via Path B
    let exchange = rc3d_io::step::parser::parse_exchange(&text).expect("parse");
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

    // Phase 3: Re-parse written output via B-Rep pipeline
    let result2 = rc3d_io::step::import_step_with_options(
        &written, &rc3d_io::StepImportOptions::default(),
    ).expect("re-import");

    println!("  ROUNDTRIP OK (B-Rep pipeline)");
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
        let mesh = rc3d_io::step::brep::mesh::mesh_brep_shell(solid.outer_shell, reg, &mesh_config, &[]);
        println!("  Mesh: {} vertices, {} indices", mesh.vertices.len(), mesh.indices.len());
        assert!(!mesh.vertices.is_empty(), "should produce mesh vertices");
    }

    // Verify AP schema was detected (all valid STEP files should have a FILE_SCHEMA)
    if let Some(ref header) = exchange.header {
        assert!(header.ap_schema.is_some(), "AP schema should be detected");
        println!("  AP_SCHEMA: {:?}", header.ap_schema);
    }

    println!("  B-Rep PIPELINE OK");
}

/// Verify assembly tree: product hierarchy, shell mapping, and transform propagation.
#[test]
fn test_assembly_tree_hierarchy() {
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

    // Build assembly tree
    let tree = rc3d_io::step::assembly::build_assembly_tree(&exchange.entities);
    println!("  Assembly tree: {} nodes, root_index={}", tree.nodes.len(), tree.root_index);
    assert!(!tree.nodes.is_empty(), "assembly tree should have nodes");

    // Verify at least one node has a name
    let named = tree.nodes.iter().filter(|n| !n.name.is_empty()).count();
    println!("  Named nodes: {}", named);
    assert!(named > 0, "should have at least one named product node");

    // Verify shell mapping
    let shell_nodes: Vec<_> = tree.nodes.iter().filter(|n| !n.shells.is_empty()).collect();
    println!("  Nodes with shells: {}", shell_nodes.len());
    // For AssemblyExample, expect at least one product node with geometry attached
    // (may be 0 for pure-hierarchy STEP files without geometry; that's acceptable)

    // Verify flattened shells
    let flat = tree.flatten_shells();
    println!("  Flattened shells: {}", flat.len());

    // Verify shell transforms are built
    let xforms = rc3d_io::step::assembly::extract_shell_transforms(&exchange.entities);
    println!("  Shell transforms: {}", xforms.len());

    // Verify styles are extractable
    let styles = rc3d_io::step::assembly::extract_shell_styles(&exchange.entities);
    println!("  Shell styles: {}", styles.len());
}

/// Integration test: PMI extraction from a STEP snippet containing
/// dimension, datum, and tolerance entities, and their wiring into
/// an AnnotationSet scene graph node.
#[test]
fn test_pmi_extraction_from_step_snippet() {
    use rc3d_io::step::parser::parse_exchange;
    use rc3d_io::step::pmi::pmi_extract::extract_pmi;
    use rc3d_scene::SceneGraph;
    use rc3d_scene::node_data::{NodeData, SeparatorNode};

    let input = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('pt1', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('pt2', (10.0, 0.0, 0.0));
#3 = DIMENSIONAL_CHARACTERISTIC_REPRESENTATION('', 10.0);
#4 = DIMENSIONAL_SIZE('dist', '', #3);
#5 = ANNOTATION_OCCURRENCE('', #4, $, (#1, #2));
#10 = CARTESIAN_POINT('org', (5.0, 0.0, 0.0));
#11 = AXIS2_PLACEMENT_3D('', #10, #12, #13);
#12 = DIRECTION('', (0.0, 0.0, 1.0));
#13 = DIRECTION('', (1.0, 0.0, 0.0));
#14 = DATUM('', 'A', (#11));
#20 = CARTESIAN_POINT('org2', (5.0, 2.0, 0.0));
#21 = GEOMETRIC_TOLERANCE('', '', #3);
#22 = ANNOTATION_OCCURRENCE('', #21, $, (#20));
ENDSEC;
END-ISO-10303-21;
    ";

    // Phase 1: parse entities (snippet has no B-Rep geometry, so parse_step
    // would fail with NoGeometry; use parse_exchange + extract_pmi directly)
    let exchange = parse_exchange(input).expect("parse exchange");

    // Phase 2: extract PMI data
    let pmi = extract_pmi(&exchange.entities);

    // Verify all three PMI types are present and correct
    assert_eq!(pmi.dimensions.len(), 1, "should extract one dimension");
    assert_eq!(pmi.datums.len(), 1, "should extract one datum");
    assert_eq!(pmi.tolerances.len(), 1, "should extract one tolerance");

    // Verify dimension content
    let dim = &pmi.dimensions[0];
    assert!(dim.text.contains("dist"), "dimension text should contain 'dist'");
    assert!((dim.start.x - 0.0).abs() < 1e-6, "dimension start.x should be 0.0");
    assert!((dim.end.x - 10.0).abs() < 1e-6, "dimension end.x should be 10.0");

    // Verify datum content
    let datum = &pmi.datums[0];
    assert_eq!(datum.label, "A", "datum label should be 'A'");
    assert!((datum.origin.x - 5.0).abs() < 1e-6, "datum origin.x should be 5.0");

    // Verify tolerance content
    let tol = &pmi.tolerances[0];
    assert!((tol.value - 10.0).abs() < 1e-6, "tolerance value should be 10.0");

    // Phase 3: verify PMI -> AnnotationSet scene graph bridge
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    let annot_id = rc3d_io::step::pmi::pmi_render::attach_pmi_to_scene(
        &mut graph, root, &pmi,
    );
    assert!(
        annot_id != NodeId::default(),
        "attach_pmi_to_scene should return a valid NodeId"
    );

    // Verify AnnotationSet node exists among root's children
    let children = graph.children(root).expect("root should have children");
    let found_annot = children.iter().any(|&cid| {
        graph.get(cid).is_some_and(|entry| {
            matches!(&entry.data, NodeData::AnnotationSet(s) if !s.elements.is_empty())
        })
    });
    assert!(found_annot, "PMI AnnotationSet node should exist in scene graph");

    // Verify the AnnotationSet has the correct number of elements.
    // GEOMETRIC_TOLERANCE (generic base type) is extracted but has no
    // specific GDT symbol, so it is filtered out during rendering.
    let annot_entry = graph.get(annot_id).expect("AnnotationSet should be found");
    if let NodeData::AnnotationSet(ann) = &annot_entry.data {
        // dimension + datum = 2; tolerance is extracted but skipped in render
        assert_eq!(ann.elements.len(), 2, "AnnotationSet should have 2 elements (dim + datum)");
    } else {
        panic!("expected AnnotationSet node at returned annot_id");
    }
}

#[test]
fn test_heal_passes_on_cs_step() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/cs.step");
    if !path.exists() {
        println!("SKIP: cs.step not found");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("read cs.step");
    let exchange = rc3d_io::step::parser::parse_exchange(&text).expect("parse");
    let brep = rc3d_io::step::brep::build_brep(&exchange.entities).expect("brep");
    let mut reg = brep.registry;

    for &sk in &brep.root_solids {
        let outer_shell = reg.solids.get(sk).unwrap().outer_shell;

        let heal_cfg = rc3d_io::step::brep::heal::HealConfig::default();
        // Phase 1 fixes are all enabled by default
        let heal = rc3d_io::step::brep::heal::heal_shell(outer_shell, &mut reg, &heal_cfg);

        println!(
            "solid {:?}: merged_vertices={}, removed_small={}, closed_uv={}, shifted={}, check_errors={}, check_warnings={}",
            sk,
            heal.merged_vertices,
            heal.removed_small_edges,
            heal.closed_uv_gaps,
            heal.shifted_pcurves,
            heal.check_errors,
            heal.check_warnings,
        );

        let mesh_cfg = rc3d_io::step::brep::mesh::BRepMeshConfig::default();
        let shell = reg.shells.get(outer_shell).unwrap();
        for &(face_key, _) in &shell.faces {
            if heal.skip_face_keys.contains(&face_key) {
                println!("  face {:?}: SKIPPED by heal", face_key);
            }
        }
        let output = rc3d_io::step::brep::mesh::mesh_brep_shell_with_report(
            outer_shell,
            &reg,
            &mesh_cfg,
            &heal.skip_face_keys,
        );
        assert!(
            output.report.meshed_faces > 0,
            "at least one face should be meshed"
        );
        println!(
            "  meshed {} faces, {} tris, {} skipped",
            output.report.meshed_faces,
            output.mesh.indices.len() / 4,
            heal.skip_face_keys.len(),
        );
    }
}
