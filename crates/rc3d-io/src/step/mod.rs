pub mod value;
pub mod entity_types;
pub mod adapter;
pub mod primary_keyword;
pub mod model;
pub mod part21;
pub mod schema;
pub mod pmi;
pub mod parser;
pub mod entity_geom;
pub mod assembly;
pub mod nurbs;
pub mod write;
pub mod validate;
pub mod xml;
pub mod bool;
pub mod tree;
pub mod header;
pub mod lod;
pub mod brep;
pub mod topology;
pub mod curve;
pub mod mesh_result;
pub mod import_options;
mod import_pipeline;
mod caf_transfer;
mod scene_emit;
mod assembly_explode;

pub use import_options::{
    StepImportMode, StepImportOptions, StepImportReport, StepImportResult,
};
pub use caf_transfer::{CafTransferOutput, StepCafTransfer};
pub use scene_emit::{apply_plan, emit_plan_options_from_step, SceneEmitOptions};
pub use adapter::AdapterMode;

use std::path::Path;
use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_scene::node_data::{MaterialNode, SeparatorNode};

#[derive(Debug, thiserror::Error)]
pub enum StepError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("STEP parse error: {0}")]
    Parse(String),
    #[error("No geometry found in STEP file")]
    NoGeometry,
    #[error("Validation failed: {0}")]
    Validation(String),
    #[error("Import quality: {0}")]
    ImportQuality(String),
}

const LARGE_STEP_BYTES: usize = 32 * 1024 * 1024;

/// Decode STEP file bytes (UTF-8 or ISO-8859-1 / Latin-1 per ISO 10303-21).
pub fn decode_step_bytes(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => bytes.iter().map(|&b| b as char).collect(),
    }
}

/// Write a SceneGraph to a STEP file (Part 21 ASCII).
pub fn write_step_file(path: &Path, graph: &SceneGraph) -> Result<(), StepError> {
    let text = write::write_step_from_graph(graph)
        .map_err(|e| StepError::Validation(e))?;
    std::fs::write(path, &text)?;
    Ok(())
}

/// Write EntityIndex to a STEP file (cleaned pass-through).
pub fn write_step_entities_file(path: &Path, entities: &parser::EntityIndex) -> Result<(), StepError> {
    let text = write::write_step_from_entities(entities);
    std::fs::write(path, &text)?;
    Ok(())
}

pub fn parse_step_file(path: &Path) -> Result<SceneGraph, StepError> {
    parse_step_file_with_options(path, &StepImportOptions::default())
}

pub fn import_step_file_with_options(
    path: &Path,
    options: &StepImportOptions,
) -> Result<StepImportResult, StepError> {
    use std::io::Read;

    let file_len = std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0);

    let is_xml = {
        let mut file = std::fs::File::open(path)?;
        let mut head = [0u8; 64];
        let n = file.read(&mut head)?;
        let head = decode_step_bytes(&head[..n]);
        let trimmed = head.trim();
        trimmed.starts_with("<?xml") || trimmed.starts_with("<iso_10303_28")
    };

    if is_xml {
        let bytes = std::fs::read(path)?;
        let text = decode_step_bytes(&bytes);
        let exchange = parser::Exchange {
            header: None,
            entities: xml::parse_xml_step(&text).map_err(StepError::Parse)?,
            diagnostics: parser::ParseDiagnostics::default(),
            part21_stats: None,
            schema_violations: Vec::new(),
        };
        exchange_to_import_result(exchange, options)
    } else if file_len > LARGE_STEP_BYTES {
        let exchange = parser::parse_step_from_file_with_options(path, options)
            .map_err(StepError::Parse)?;
        exchange_to_import_result(exchange, options)
    } else {
        let _t_io = std::time::Instant::now();
        let bytes = std::fs::read(path)?;
        let text = decode_step_bytes(&bytes);
        eprintln!("[STEP timing] file IO: {:.1}s", _t_io.elapsed().as_secs_f32());
        import_step_with_options(&text, options)
    }
}

pub fn parse_step_file_with_options(
    path: &Path,
    options: &StepImportOptions,
) -> Result<SceneGraph, StepError> {
    Ok(import_step_file_with_options(path, options)?.graph)
}

/// Parse STEP text into a SceneGraph using the OCC-aligned B-Rep pipeline.
pub fn parse_step(input: &str) -> Result<SceneGraph, StepError> {
    parse_step_with_options(input, &StepImportOptions::default())
}

pub fn import_step_with_options(
    input: &str,
    options: &StepImportOptions,
) -> Result<StepImportResult, StepError> {
    let trimmed = input.trim();
    let exchange = if trimmed.starts_with("<?xml") || trimmed.starts_with("<iso_10303_28") {
        let entities = xml::parse_xml_step(trimmed).map_err(StepError::Parse)?;
        parser::Exchange {
            header: None,
            entities,
            diagnostics: parser::ParseDiagnostics::default(),
            part21_stats: None,
            schema_violations: Vec::new(),
        }
    } else {
        let _tp = std::time::Instant::now();
        let ex = parser::parse_exchange_with_options(trimmed, options).map_err(StepError::Parse)?;
        eprintln!("[STEP timing] parse: {:.1}s  entities: {}", _tp.elapsed().as_secs_f32(), ex.entities.len());
        ex
    };
    exchange_to_import_result(exchange, options)
}

pub fn parse_step_with_options(
    input: &str,
    options: &StepImportOptions,
) -> Result<SceneGraph, StepError> {
    Ok(import_step_with_options(input, options)?.graph)
}

fn exchange_to_import_result(
    exchange: parser::Exchange,
    options: &StepImportOptions,
) -> Result<StepImportResult, StepError> {
    let _total = std::time::Instant::now();
    let mut import_report = StepImportReport::default();
    import_report.skipped_parse_entities = exchange.diagnostics.skipped_entities.len();
    import_report.unknown_entity_count = exchange.diagnostics.unknown_entity_count;
    if let Some(ref header) = exchange.header {
        import_report.ap_schema = header.ap_schema.clone();
        if let Some(ref ap) = import_report.ap_schema {
            log::info!("[STEP] detected schema: {}", ap);
        }
    }
    if let Some(ref stats) = exchange.part21_stats {
        import_report.data_section_count = stats.data_section_count;
        import_report.complex_external_count = stats.complex_external_count;
    }
    import_report.schema_violations = exchange.schema_violations.len();
    if options.strict_schema && !exchange.schema_violations.is_empty() {
        let msg: Vec<String> = exchange
            .schema_violations
            .iter()
            .take(8)
            .map(|v| format!("#{} {} {}: {}", v.entity_id, v.entity_name, v.constraint, v.description))
            .collect();
        return Err(StepError::Validation(format!(
            "schema: {} violation(s): {}",
            exchange.schema_violations.len(),
            msg.join("; ")
        )));
    }
    if import_report.unknown_entity_count > 0 {
        log::warn!(
            "[STEP] {} unknown entity type(s) in file",
            import_report.unknown_entity_count
        );
    }

    let report = validate::validate(&exchange.entities);
    log::info!(
        "[STEP] {} entities, {} shells, {} faces",
        exchange.entities.len(),
        report.topology_info.shells,
        report.topology_info.faces,
    );
    for w in &report.warnings {
        log::warn!("[STEP] {}", w);
    }
    import_report.validation_errors = report.errors.len();
    if options.fail_on_validation_errors() && !report.errors.is_empty() {
        return Err(StepError::Validation(report.errors.join("; ")));
    }

    let build_options = brep::BRepBuildOptions::from_import(options);
    let t_caf = std::time::Instant::now();
    let transfer = StepCafTransfer::transfer(&exchange.entities, &build_options)?;
    let mut document = transfer.document;
    eprintln!("[STEP timing] B-Rep build: {:.1}s", t_caf.elapsed().as_secs_f32());
    import_report.skipped_faces = transfer.build_report.skipped_faces;
    import_report.skipped_edges = transfer.build_report.skipped_edges;
    import_report.void_shell_count = transfer.build_report.void_shell_count;
    import_report.void_shells_subtracted = transfer.build_report.void_shells_subtracted;
    import_report.oriented_forward_faces = transfer.build_report.oriented_forward_faces;
    import_report.oriented_reversed_faces = transfer.build_report.oriented_reversed_faces;

    if !options.allow_void_shells_unmeshed() && transfer.build_report.void_shell_count > 0 {
        return Err(StepError::ImportQuality(format!(
            "BREP_WITH_VOIDS: {} void shell(s) present; strict import does not mesh voids",
            transfer.build_report.void_shell_count
        )));
    }

    let root_solids = transfer.root_solids;
    let g0_tol = topology::global_tolerance(&exchange.entities).max(1e-6);
    for &sk in &root_solids {
        let shell_keys: Vec<_> = document
            .store
            .solids
            .get(sk)
            .map(|s| {
                let mut keys = Vec::with_capacity(1 + s.void_shells.len());
                keys.push(s.outer_shell);
                keys.extend(s.void_shells.iter().copied());
                keys
            })
            .unwrap_or_default();
        if !options.skip_visualization {
            let mut total_same_param = 0usize;
            for shell_key in shell_keys {
                total_same_param +=
                    brep::same_parameter::same_parameter_shell(&mut document.store, shell_key, g0_tol);
            }
            if total_same_param > 0 {
                log::debug!(
                    "[STEP] SameParameter: {} edge(s) on solid {:?}",
                    total_same_param,
                    sk
                );
            }
        }
    }

    let assembly_ctx = assembly::AssemblyContext::build(&exchange.entities);
    let shell_instances = assembly_ctx.shell_instances(&exchange.entities);
    let assembly_geom_nodes = document
        .labels
        .labels
        .iter()
        .filter(|(_, label)| label.shape.is_some())
        .count();
    import_report.assembly_node_count = assembly_geom_nodes;
    if assembly_geom_nodes > 0 {
        log::info!(
            "[STEP] assembly labels: {} shaped label(s)",
            assembly_geom_nodes,
        );
    }

    let t_heal = std::time::Instant::now();
    let heal_iters = if options.skip_visualization { 2 } else { 5 };
    let total_heal = import_pipeline::run_heal_pipeline(
        &mut document.store,
        &root_solids,
        options.heal_level,
        heal_iters,
    );
    import_report.heal_check_errors = total_heal.check_errors;
    import_report.skipped_faces += total_heal.skip_face_keys.len();
    if !options.skip_visualization {
        import_pipeline::run_continuity_checks(
            &document.store,
            &root_solids,
            g0_tol,
            &mut import_report,
        );
    }

    if options.fail_on_heal_check_errors() && total_heal.check_errors > 0 {
        return Err(StepError::ImportQuality(format!(
            "B-Rep heal reported {} check error(s)",
            total_heal.check_errors
        )));
    }

    let explode_factor = assembly_explode::effective_assembly_explode(
        options.assembly_preview_explode,
        options.mode,
        &document.store,
        &root_solids,
    );
    if explode_factor > 0.0 {
        log::info!("[STEP] assembly preview explode: {:.2}", explode_factor);
    }

    let mut plan_options = emit_plan_options_from_step(options);
    plan_options.heal_skip_faces = total_heal.skip_face_keys.clone();
    plan_options.explode_offsets = assembly_explode::compute_assembly_explode_offsets(
        &mut document,
        &root_solids,
        explode_factor,
    );
    let t_mesh_start = std::time::Instant::now();
    let plan = document
        .build_emit_plan(&plan_options)
        .map_err(|e| StepError::ImportQuality(e.to_string()))?;
    let t_mesh_elapsed = t_mesh_start.elapsed().as_secs_f32();
    let t_total = _total.elapsed().as_secs_f32();
    eprintln!("[STEP timing] B-Rep build: {:.1}s  heal: {:.1}s  mesh+plan: {:.1}s  total: {:.1}s",
        t_heal.elapsed().as_secs_f32(),
        (t_mesh_start - t_heal).as_secs_f32(),
        t_mesh_elapsed,
        t_total);

    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    let default_color = [0.9, 0.9, 0.9];
    graph.add_child(
        root,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(default_color[0], default_color[1], default_color[2]),
            base_color: Vec3::new(default_color[0], default_color[1], default_color[2]),
            roughness: 0.35,
            opacity: 1.0,
            ..Default::default()
        }),
    );

    apply_plan(
        &mut graph,
        root,
        &plan,
        &SceneEmitOptions::default(),
        &document.pmi_pool,
    )?;

    if !options.skip_visualization {
        let mut props_vertices: Vec<Vec3> = Vec::new();
        let mut props_indices: Vec<i32> = Vec::new();
        for cached in plan.mesh_table.values() {
            import_pipeline::append_props_mesh_public(&cached.mesh, &mut props_vertices, &mut props_indices);
        }

        if !props_vertices.is_empty() && !props_indices.is_empty() {
            let props = brep::compute_mesh_properties(&props_vertices, &props_indices);
            log::info!(
                "[STEP] mesh properties: volume={:.6} area={:.6} com=[{:.4}, {:.4}, {:.4}]",
                props.volume,
                props.surface_area,
                props.center_of_mass[0],
                props.center_of_mass[1],
                props.center_of_mass[2],
            );
        }

        let mut mesh_config = brep::mesh::BRepMeshConfig::default();
        mesh_config.relative_deflection = options.mesh_relative_deflection;

        brep::overlay::build_edge_curves(
            &mut graph,
            root,
            &document.store,
            &root_solids,
            &shell_instances,
            &mesh_config,
        );

        // Mesh wireframe overlay for debugging degenerate faces
        brep::overlay::build_mesh_wireframe(
            &mut graph,
            root,
            &props_vertices,
            &props_indices,
        );
    }

    if import_report.skipped_parse_entities > 0
        || import_report.skipped_faces > 0
        || import_report.skipped_edges > 0
    {
        log::warn!("[STEP] import report: {:?}", import_report);
    }

    if !document.pmi_pool.entries.is_empty() {
        log::info!(
            "[STEP] PMI: {} annotation(s) via emit plan",
            document.pmi_pool.entries.len(),
        );
    }

    if let Some(root_entry) = graph.get_mut(root) {
        root_entry.display_mode = Some(DisplayMode::Shaded);
    }

    Ok(StepImportResult {
        document,
        graph,
        report: import_report,
        entities: exchange.entities,
    })
}
