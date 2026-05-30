pub mod value;
pub mod entity_types;
pub mod adapter;
pub mod primary_keyword;
pub mod model;
pub mod part21;
pub mod schema;
pub mod pmi;
pub mod parser;
pub mod geom;
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

pub use import_options::{
    StepImportMode, StepImportOptions, StepImportReport, StepImportResult,
};
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
        let bytes = std::fs::read(path)?;
        let text = decode_step_bytes(&bytes);
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
        parser::parse_exchange_with_options(trimmed, options).map_err(StepError::Parse)?
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
    let brep_result = brep::build_brep_with_options(&exchange.entities, &build_options)?;
    import_report.skipped_faces = brep_result.build_report.skipped_faces;
    import_report.skipped_edges = brep_result.build_report.skipped_edges;
    import_report.void_shell_count = brep_result.build_report.void_shell_count;

    if !options.allow_void_shells_unmeshed() && brep_result.build_report.void_shell_count > 0 {
        return Err(StepError::ImportQuality(format!(
            "BREP_WITH_VOIDS: {} void shell(s) present; strict import does not mesh voids",
            brep_result.build_report.void_shell_count
        )));
    }

    let root_solids = brep_result.root_solids;
    let mut reg = brep_result.registry;
    let g0_tol = topology::global_tolerance(&exchange.entities).max(1e-6);
    for &sk in &root_solids {
        let shell_key = reg.solids.get(sk).map(|s| s.outer_shell);
        if let Some(shell_key) = shell_key {
            let n = brep::same_parameter::same_parameter_shell(&mut reg, shell_key, g0_tol);
            if n > 0 {
                log::debug!("[STEP] SameParameter: {} edge(s) on solid {:?}", n, sk);
            }
        }
    }

    let assembly_ctx = assembly::AssemblyContext::build(&exchange.entities);
    let shell_instances = assembly_ctx.shell_instances(&exchange.entities);
    let shell_styles = assembly::extract_shell_styles(&exchange.entities);

    let assembly_tree = assembly_ctx.assembly_tree(&exchange.entities);
    let assembly_geom_nodes = assembly_tree
        .nodes
        .iter()
        .filter(|n| !n.shells.is_empty())
        .count();
    import_report.assembly_node_count = assembly_geom_nodes;
    if !assembly_tree.nodes.is_empty() && assembly_geom_nodes > 0 {
        log::info!(
            "[STEP] assembly tree: {} nodes, {} with geometry",
            assembly_tree.nodes.len(),
            assembly_geom_nodes,
        );
    }

    let total_heal = import_pipeline::run_heal_pipeline(
        &mut reg,
        &root_solids,
        options.heal_level,
        5,
    );
    import_report.heal_check_errors = total_heal.check_errors;
    import_report.skipped_faces += total_heal.skip_face_keys.len();
    import_pipeline::run_continuity_checks(
        &reg,
        &root_solids,
        g0_tol,
        &mut import_report,
    );

    if options.fail_on_heal_check_errors() && total_heal.check_errors > 0 {
        return Err(StepError::ImportQuality(format!(
            "B-Rep heal reported {} check error(s)",
            total_heal.check_errors
        )));
    }

    let mut mesh_config = brep::mesh::BRepMeshConfig::default();
    mesh_config.relative_deflection = options.mesh_relative_deflection;
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

    let assembly_explode = import_pipeline::effective_assembly_explode(
        options.assembly_preview_explode,
        options.mode,
        &reg,
        &root_solids,
    );

    let emit = import_pipeline::emit_scene_meshes(
        &mut graph,
        root,
        &reg,
        &root_solids,
        &mesh_config,
        &total_heal,
        &assembly_tree,
        assembly_geom_nodes,
        &shell_styles,
        &shell_instances,
        assembly_explode,
    )?;
    let props_vertices = emit.props_vertices;
    let props_indices = emit.props_indices;

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

    log::info!(
        "[STEP] mesh complete: {} solid(s), {} skipped face(s) during heal",
        root_solids.len(),
        total_heal.skip_face_keys.len(),
    );

    brep::overlay::build_edge_curves(
        &mut graph,
        root,
        &reg,
        &root_solids,
        &shell_instances,
        &mesh_config,
    );

    // Mesh wireframe overlay for debugging degenerate faces
    if !props_vertices.is_empty() && !props_indices.is_empty() {
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

    // PMI annotations
    {
        let pmi_data = pmi::pmi_extract::extract_pmi(&exchange.entities);
        let has_pmi = !pmi_data.dimensions.is_empty()
            || !pmi_data.datums.is_empty()
            || !pmi_data.tolerances.is_empty();
        if has_pmi {
            log::info!(
                "[STEP] PMI: {} dims, {} datums, {} tolerances",
                pmi_data.dimensions.len(),
                pmi_data.datums.len(),
                pmi_data.tolerances.len(),
            );
            pmi::pmi_render::attach_pmi_to_scene(&mut graph, root, &pmi_data);
        }
    }

    if let Some(root_entry) = graph.get_mut(root) {
        root_entry.display_mode = Some(DisplayMode::Shaded);
    }

    Ok(StepImportResult {
        graph,
        report: import_report,
        assembly_tree,
        entities: exchange.entities,
    })
}
