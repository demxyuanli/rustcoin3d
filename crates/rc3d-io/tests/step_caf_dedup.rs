//! CAF transfer dedup integration (CDSR / MAPPED_ITEM -> shared mesh slots).
//! Run: rtk cargo test -p rc3d-io --test step_caf_dedup -- --nocapture

use rc3d_core::math::PVec3;
use rc3d_core::math::PMat4;
use rc3d_io::step::assembly::AssemblyContext;
use rc3d_io::step::brep::BRepBuildOptions;
use rc3d_io::step::parser::parse_exchange;
use rc3d_io::step::{emit_plan_options_from_step, import_step_with_options, StepCafTransfer, StepImportOptions};
use rc3d_shape::{AttributeBag, SceneEmitPlan, ShapeKind, SolidKey, XdeLabel, PmiEntry};
use std::collections::{HashMap, HashSet};
use std::path::Path;

fn test_data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

fn count_cdsr_entities(text: &str) -> usize {
    parse_exchange(text)
        .map(|ex| {
            ex.entities
                .values()
                .filter(|r| r.name == "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION")
                .count()
        })
        .unwrap_or(0)
}

/// Shell step ids referenced by the assembly tree / shell instance list.
fn assembly_shell_step_ids(text: &str) -> Vec<u64> {
    let exchange = parse_exchange(text).expect("parse");
    let ctx = AssemblyContext::build(&exchange.entities);
    let mut ids: HashSet<u64> = HashSet::new();
    for (shell_id, _) in ctx.shell_instances(&exchange.entities) {
        ids.insert(shell_id);
    }
    let tree = ctx.assembly_tree(&exchange.entities);
    for node in &tree.nodes {
        for &shell_id in &node.shells {
            ids.insert(shell_id);
        }
    }
    ids.into_iter().collect()
}

#[test]
fn cdsr_dedup_index_assembly_example_compresses_shell_keys() {
    let path = test_data("AssemblyExample-Assembly.step");
    if !path.exists() {
        println!("SKIP: AssemblyExample-Assembly.step not found");
        return;
    }

    let text = std::fs::read_to_string(&path).expect("read");
    let cdsr_count = count_cdsr_entities(&text);
    assert!(cdsr_count >= 1, "fixture should contain CDSR entities");

    let exchange = parse_exchange(&text).expect("parse");
    let shell_ids = assembly_shell_step_ids(&text);
    assert!(
        shell_ids.len() >= 2,
        "assembly fixture should reference multiple shell step ids, got {}",
        shell_ids.len()
    );

    let brep = rc3d_io::step::brep::build_brep(&exchange.entities).expect("brep");
    let mut shell_to_solid = HashMap::new();
    for &sk in &brep.root_solids {
        let solid = brep.registry.solids.get(sk).unwrap();
        let shell = brep.registry.shells.get(solid.outer_shell).unwrap();
        if let Some(step_id) = shell.step_id {
            shell_to_solid.insert(step_id, sk);
        }
    }

    let unique_solids = StepCafTransfer::count_deduped_solids(
        &exchange.entities,
        &shell_to_solid,
        &shell_ids,
    )
    .expect("dedup resolve");

    println!(
        "AssemblyExample: cdsr={} assembly_shells={} mapped_shells={} unique_solids={}",
        cdsr_count,
        shell_ids.len(),
        shell_to_solid.len(),
        unique_solids,
    );

    assert!(
        shell_to_solid.len() >= 2,
        "B-Rep should expose multiple shell step ids"
    );
    assert!(
        unique_solids <= shell_to_solid.len(),
        "dedup resolve must not increase solid count"
    );
    assert!(
        cdsr_count >= 1,
        "fixture should exercise CDSR entity parsing"
    );
}

#[test]
fn cdsr_dedup_cs_step_distinct_solids() {
    let path = test_data("cs.step");
    if !path.exists() {
        println!("SKIP: cs.step not found");
        return;
    }

    let text = std::fs::read_to_string(&path).expect("read");
    let exchange = parse_exchange(&text).expect("parse");
    let transfer = StepCafTransfer::transfer(
        &exchange.entities,
        &BRepBuildOptions::from_import(&StepImportOptions::default()),
    )
    .expect("transfer");

    let mut solid_keys = HashSet::new();
    for (_shape_id, node) in transfer.document.shapes.iter() {
        if let ShapeKind::Solid(sk) = node.kind {
            solid_keys.insert(sk);
        }
    }

    println!(
        "cs.step: shapes={} unique_solids={} root_solids={}",
        transfer.document.shapes.len(),
        solid_keys.len(),
        transfer.root_solids.len(),
    );

    assert!(
        transfer.root_solids.len() >= 2,
        "Cube + Sphere should yield >=2 root solids"
    );
    assert!(
        solid_keys.len() >= 2,
        "transfer should retain >=2 unique SolidKeys"
    );
    assert_eq!(
        transfer.document.shapes.len(),
        solid_keys.len(),
        "shape node count should match unique solids"
    );

    let mut document = transfer.document;
    let plan = document
        .build_emit_plan(&emit_plan_options_from_step(&StepImportOptions::default()))
        .expect("emit plan");
    assert!(
        !plan.mesh_table.is_empty(),
        "emit plan should tessellate at least one mesh slot"
    );
    assert!(
        plan.instances.len() >= 2,
        "Cube + Sphere should emit two instances"
    );
    assert_eq!(
        plan.instances.len(),
        plan.mesh_table.len(),
        "distinct parts should use distinct mesh slots"
    );
}

#[test]
fn caf_transfer_end_to_end_document_labels_and_plan() {
    let path = test_data("AssemblyExample-Assembly.step");
    if !path.exists() {
        println!("SKIP");
        return;
    }

    let text = std::fs::read_to_string(&path).expect("read");
    let result = import_step_with_options(&text, &StepImportOptions::default()).expect("import");
    let doc = &result.document;

    assert!(
        !doc.labels.root_labels.is_empty() || !doc.labels.labels.is_empty(),
        "XDE label forest should be populated for assembly"
    );
    assert!(!doc.roots.is_empty() || !doc.shapes.is_empty());

    let mut doc_mut = result.document;
    let plan = doc_mut
        .build_emit_plan(&emit_plan_options_from_step(&StepImportOptions::default()))
        .expect("plan");

    assert!(!plan.instances.is_empty());
    assert!(!plan.mesh_table.is_empty());
    if !plan.hierarchy.is_empty() {
        let hierarchy_instances: usize = count_hierarchy_instances(&plan.hierarchy);
        assert_eq!(
            hierarchy_instances,
            plan.instances.len(),
            "hierarchy instance count should match flat instances"
        );
    }
}

fn count_hierarchy_instances(nodes: &[rc3d_shape::EmitNode]) -> usize {
    nodes
        .iter()
        .map(|n| n.instances.len() + count_hierarchy_instances(&n.children))
        .sum()
}

fn mesh_slot_use(plan: &SceneEmitPlan) -> HashMap<u32, usize> {
    let mut slot_use: HashMap<u32, usize> = HashMap::new();
    for inst in &plan.instances {
        *slot_use.entry(inst.mesh_slot).or_default() += 1;
    }
    slot_use
}

fn assert_shared_mesh_slots(plan: &SceneEmitPlan) {
    assert!(
        plan.instances.len() > plan.mesh_table.len(),
        "P0-3: need more instances than mesh slots, got instances={} mesh_slots={}",
        plan.instances.len(),
        plan.mesh_table.len()
    );
    let slot_use = mesh_slot_use(plan);
    assert!(
        slot_use.values().any(|&c| c > 1),
        "P0-3: at least one mesh slot should be referenced by multiple instances, use={slot_use:?}"
    );
}

fn unique_solid_keys(document: &rc3d_shape::ShapeDocument) -> HashSet<SolidKey> {
    document
        .shapes
        .iter()
        .filter_map(|(_, node)| node.kind.solid_key())
        .collect()
}

/// Extend `Cube.step` with a two-instance assembly (same ABREP, distinct transforms).
fn twin_cube_assembly_step() -> Option<String> {
    let path = test_data("Cube.step");
    if !path.exists() {
        return None;
    }
    let cube = std::fs::read_to_string(&path).ok()?;
    const MARKER: &str = "ENDSEC;\nEND-ISO-10303-21;";
    const ASSEMBLY: &str = "\
#600 = PRODUCT('TwinAssembly','Two cube instances','',(#8));
#601 = PRODUCT_DEFINITION_FORMATION('','',#600);
#602 = PRODUCT_DEFINITION('design','',#601,#9);
#603 = PRODUCT_DEFINITION_SHAPE('','',#602);
#604 = SHAPE_DEFINITION_REPRESENTATION(#603,#610);
#610 = SHAPE_REPRESENTATION('',(#611,#615,#619),#345);
#611 = AXIS2_PLACEMENT_3D('',#612,#13,#14);
#612 = CARTESIAN_POINT('',(0.,0.,0.));
#615 = AXIS2_PLACEMENT_3D('',#616,#13,#14);
#616 = CARTESIAN_POINT('',(0.,0.,0.));
#619 = AXIS2_PLACEMENT_3D('',#620,#13,#14);
#620 = CARTESIAN_POINT('',(0.,0.,0.));
#630 = PRODUCT('Cube_copy','','',(#34));
#631 = PRODUCT_DEFINITION_FORMATION('','',#630);
#632 = PRODUCT_DEFINITION('design','',#631,#35);
#633 = PRODUCT_DEFINITION_SHAPE('','',#632);
#634 = SHAPE_DEFINITION_REPRESENTATION(#633,#10);
#640 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','Cube_a','',#602,#5,$);
#641 = PRODUCT_DEFINITION_SHAPE('Placement','',#640);
#650 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('2','Cube_b','',#602,#632,$);
#651 = PRODUCT_DEFINITION_SHAPE('Placement','',#650);
#654 = AXIS2_PLACEMENT_3D('',#655,#13,#14);
#655 = CARTESIAN_POINT('',(15.,0.,0.));
#660 = ITEM_DEFINED_TRANSFORMATION('','',#654,#632);
";
    if !cube.contains(MARKER) {
        return None;
    }
    Some(cube.replace(MARKER, &format!("{ASSEMBLY}\n{MARKER}")))
}

#[test]
fn p0_3_twin_cube_assembly_import_shares_mesh_slot() {
    let Some(text) = twin_cube_assembly_step() else {
        println!("SKIP: Cube.step not found or unexpected format");
        return;
    };
    let mut result =
        import_step_with_options(&text, &StepImportOptions::default()).expect("import twin assembly");
    let plan = result
        .document
        .build_emit_plan(&emit_plan_options_from_step(&StepImportOptions::default()))
        .expect("emit plan");

    println!(
        "twin cube assembly: instances={} mesh_slots={} shapes={} unique_solids={}",
        plan.instances.len(),
        plan.mesh_table.len(),
        result.document.shapes.len(),
        unique_solid_keys(&result.document).len()
    );

    assert!(
        plan.instances.len() >= 2,
        "assembly import should emit >=2 instances"
    );
    assert_eq!(
        unique_solid_keys(&result.document).len(),
        1,
        "both instances should reference the same SolidKey"
    );
    assert_shared_mesh_slots(&plan);
}

#[test]
fn p0_3_cube_twin_instances_share_one_mesh_slot() {
    let path = test_data("Cube.step");
    if !path.exists() {
        println!("SKIP: Cube.step not found");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("read");
    let mut result =
        import_step_with_options(&text, &StepImportOptions::default()).expect("import");
    let doc = &mut result.document;
    let sk = unique_solid_keys(doc)
        .into_iter()
        .next()
        .expect("Cube.step should yield one SolidKey");
    let shape_id = doc.add_solid_instance(sk, PMat4::from_translation([10., 0., 0.].into()), None);
    doc.roots.retain(|&id| id != shape_id);
    let twin_label = doc.labels.add_label(XdeLabel {
        parent: None,
        children: vec![],
        attrs: AttributeBag {
            name: Some("Cube_instance_2".into()),
            ..Default::default()
        },
        shape: Some(shape_id),
    });
    doc.labels.root_labels.push(twin_label);

    let plan = doc
        .build_emit_plan(&emit_plan_options_from_step(&StepImportOptions::default()))
        .expect("emit plan");

    assert_eq!(
        plan.instances.len(),
        2,
        "P0-3: two emit rows for two transforms of the same solid"
    );
    assert_eq!(plan.mesh_table.len(), 1, "P0-3: one tessellated mesh slot");
    assert!(
        plan.instances.iter().all(|i| i.mesh_slot == 0),
        "both instances must reference mesh slot 0"
    );
    assert_shared_mesh_slots(&plan);
}

#[test]
fn assembly_example_multi_instance_emit_plan() {
    let path = test_data("AssemblyExample-Assembly.step");
    if !path.exists() {
        println!("SKIP");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("read");
    let mut result =
        import_step_with_options(&text, &StepImportOptions::default()).expect("import");
    let plan = result
        .document
        .build_emit_plan(&emit_plan_options_from_step(&StepImportOptions::default()))
        .expect("plan");

    println!(
        "AssemblyExample instances={} mesh_slots={} shapes={} unique_solids={}",
        plan.instances.len(),
        plan.mesh_table.len(),
        result.document.shapes.len(),
        unique_solid_keys(&result.document).len()
    );

    assert!(
        plan.instances.len() >= 2,
        "assembly should emit multiple shape instances"
    );
    assert!(
        plan.mesh_table.len() <= plan.instances.len(),
        "P0-3: mesh slots should not exceed instances"
    );
    assert_eq!(
        plan.instances.len(),
        result.document.shapes.len(),
        "each shape node should produce one emit instance"
    );
    assert_eq!(
        unique_solid_keys(&result.document).len(),
        plan.mesh_table.len(),
        "AssemblyExample parts are unique; one mesh slot per distinct solid"
    );
    assert_eq!(
        plan.instances.len(),
        plan.mesh_table.len(),
        "no mesh slot sharing expected when every instance uses a distinct solid"
    );
}

#[test]
fn pmi_refs_bind_to_nearest_part_label_on_twin_cube_assembly() {
    let Some(text) = twin_cube_assembly_step() else {
        println!("SKIP: Cube.step not found or unexpected format");
        return;
    };
    let mut result =
        import_step_with_options(&text, &StepImportOptions::default()).expect("import twin assembly");
    let doc = &mut result.document;

    let shaped_pairs: Vec<(rc3d_shape::LabelId, rc3d_shape::ShapeId)> = doc
        .labels
        .labels
        .iter()
        .filter_map(|(label_id, label)| label.shape.map(|shape_id| (label_id, shape_id)))
        .collect();
    let mut shaped: Vec<(rc3d_shape::LabelId, rc3d_core::math::PVec3)> = shaped_pairs
        .iter()
        .map(|(label_id, shape_id)| {
            let world = doc.world_transform(*shape_id);
            (
                *label_id,
                rc3d_core::math::PVec3::new(world.w_axis.x, world.w_axis.y, world.w_axis.z),
            )
        })
        .collect();
    shaped.sort_by(|a, b| a.1.x.partial_cmp(&b.1.x).unwrap());
    assert_eq!(shaped.len(), 2, "twin assembly should expose two shaped labels");
    let (left_label, left_pos) = shaped[0];
    let (right_label, right_pos) = shaped[1];
    assert!(
        (right_pos.x - left_pos.x).abs() > 1.0,
        "instances should be spatially separated"
    );

    doc.pmi_pool.entries = vec![
        PmiEntry {
            id: 0,
            label: "left_pmi".into(),
            step_entity_id: None,
            origin: [left_pos.x, left_pos.y, left_pos.z],
            normal: [0.0, 1.0, 0.0],
        },
        PmiEntry {
            id: 1,
            label: "right_pmi".into(),
            step_entity_id: None,
            origin: [right_pos.x, right_pos.y, right_pos.z],
            normal: [0.0, 1.0, 0.0],
        },
    ];

    let plan = doc
        .build_emit_plan(&emit_plan_options_from_step(&StepImportOptions::default()))
        .expect("emit plan");

    assert_eq!(plan.pmi_refs.len(), 2);
    assert_eq!(plan.pmi_refs[0].label_id, left_label);
    assert_eq!(plan.pmi_refs[1].label_id, right_label);
}

fn shaped_labels_with_world(doc: &mut rc3d_shape::ShapeDocument) -> Vec<(rc3d_shape::LabelId, rc3d_core::math::PVec3)> {
    let shaped_pairs: Vec<(rc3d_shape::LabelId, rc3d_shape::ShapeId)> = doc
        .labels
        .labels
        .iter()
        .filter_map(|(label_id, label)| label.shape.map(|shape_id| (label_id, shape_id)))
        .collect();
    let mut shaped: Vec<(rc3d_shape::LabelId, rc3d_core::math::PVec3)> = shaped_pairs
        .iter()
        .map(|(label_id, shape_id)| {
            let world = doc.world_transform(*shape_id);
            (
                *label_id,
                rc3d_core::math::PVec3::new(world.w_axis.x, world.w_axis.y, world.w_axis.z),
            )
        })
        .collect();
    shaped.sort_by(|a, b| a.1.x.partial_cmp(&b.1.x).unwrap());
    shaped
}

#[test]
fn pmi_refs_prefer_semantic_label_binding() {
    let Some(text) = twin_cube_assembly_step() else {
        println!("SKIP: Cube.step not found or unexpected format");
        return;
    };
    let mut result =
        import_step_with_options(&text, &StepImportOptions::default()).expect("import twin assembly");
    let doc = &mut result.document;
    let shaped = shaped_labels_with_world(doc);
    assert_eq!(shaped.len(), 2, "expected two shaped labels");
    let (left_label, left_pos) = shaped[0];
    let (right_label, right_pos) = shaped[1];
    let left_step = 900001u64;
    let right_step = 900002u64;
    doc.labels.labels.get_mut(left_label).unwrap().attrs.step_entity_id = Some(left_step);
    doc.labels.labels.get_mut(right_label).unwrap().attrs.step_entity_id = Some(right_step);
    doc.provenance.label_to_step.insert(left_label, left_step);
    doc.provenance.label_to_step.insert(right_label, right_step);
    doc.pmi_pool.entries = vec![
        PmiEntry {
            id: 0,
            label: "semantic_left".into(),
            step_entity_id: Some(left_step),
            origin: [right_pos.x, right_pos.y, right_pos.z],
            normal: [0.0, 1.0, 0.0],
        },
        PmiEntry {
            id: 1,
            label: "semantic_right".into(),
            step_entity_id: Some(right_step),
            origin: [left_pos.x, left_pos.y, left_pos.z],
            normal: [0.0, 1.0, 0.0],
        },
    ];
    let plan = doc
        .build_emit_plan(&emit_plan_options_from_step(&StepImportOptions::default()))
        .expect("emit plan");
    assert_eq!(plan.pmi_refs.len(), 2);
    assert_eq!(plan.pmi_refs[0].label_id, left_label);
    assert_eq!(plan.pmi_refs[1].label_id, right_label);
}

#[test]
fn pmi_refs_fallback_to_nearest_when_semantic_missing() {
    let Some(text) = twin_cube_assembly_step() else {
        println!("SKIP: Cube.step not found or unexpected format");
        return;
    };
    let mut result =
        import_step_with_options(&text, &StepImportOptions::default()).expect("import twin assembly");
    let doc = &mut result.document;
    let shaped = shaped_labels_with_world(doc);
    assert_eq!(shaped.len(), 2, "expected two shaped labels");
    let (left_label, left_pos) = shaped[0];
    let (right_label, right_pos) = shaped[1];
    doc.pmi_pool.entries = vec![
        PmiEntry {
            id: 0,
            label: "fallback_left".into(),
            step_entity_id: Some(999999),
            origin: [left_pos.x, left_pos.y, left_pos.z],
            normal: [0.0, 1.0, 0.0],
        },
        PmiEntry {
            id: 1,
            label: "fallback_right".into(),
            step_entity_id: Some(999998),
            origin: [right_pos.x, right_pos.y, right_pos.z],
            normal: [0.0, 1.0, 0.0],
        },
    ];
    let plan = doc
        .build_emit_plan(&emit_plan_options_from_step(&StepImportOptions::default()))
        .expect("emit plan");
    assert_eq!(plan.pmi_refs.len(), 2);
    assert_eq!(plan.pmi_refs[0].label_id, left_label);
    assert_eq!(plan.pmi_refs[1].label_id, right_label);
}

#[test]
fn pmi_refs_support_mixed_semantic_and_fallback_binding() {
    let Some(text) = twin_cube_assembly_step() else {
        println!("SKIP: Cube.step not found or unexpected format");
        return;
    };
    let mut result =
        import_step_with_options(&text, &StepImportOptions::default()).expect("import twin assembly");
    let doc = &mut result.document;
    let shaped = shaped_labels_with_world(doc);
    assert_eq!(shaped.len(), 2, "expected two shaped labels");
    let (left_label, _left_pos) = shaped[0];
    let (right_label, right_pos) = shaped[1];
    let semantic_step = 900101u64;
    doc.labels.labels.get_mut(left_label).unwrap().attrs.step_entity_id = Some(semantic_step);
    doc.provenance.label_to_step.insert(left_label, semantic_step);
    doc.pmi_pool.entries = vec![
        PmiEntry {
            id: 0,
            label: "mixed_semantic".into(),
            step_entity_id: Some(semantic_step),
            origin: [right_pos.x, right_pos.y, right_pos.z],
            normal: [0.0, 1.0, 0.0],
        },
        PmiEntry {
            id: 1,
            label: "mixed_fallback".into(),
            step_entity_id: Some(999997),
            origin: [right_pos.x, right_pos.y, right_pos.z],
            normal: [0.0, 1.0, 0.0],
        },
    ];
    let plan = doc
        .build_emit_plan(&emit_plan_options_from_step(&StepImportOptions::default()))
        .expect("emit plan");
    assert_eq!(plan.pmi_refs.len(), 2);
    assert_eq!(plan.pmi_refs[0].label_id, left_label, "semantic should override geometry");
    assert_eq!(plan.pmi_refs[1].label_id, right_label, "missing semantic should use nearest");
}
