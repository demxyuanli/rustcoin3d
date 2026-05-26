pub mod value;
pub mod entity_types;
#[cfg(feature = "pmi")]
pub mod pmi;
pub mod parser;
pub mod geom;
pub mod pcurve;
pub mod assembly;
pub mod nurbs;
pub mod write;
pub mod validate;
pub mod xml;
pub mod bool;
pub mod tree;
pub mod header;
pub mod lod;
pub mod fillet;
pub mod brep;
pub mod topology;
pub mod tessellate;
pub mod surface_tess;
pub mod curve;
pub mod refine;
pub mod mesh_result;
pub mod import_options;

pub use import_options::{StepImportMode, StepImportOptions, StepImportReport};

use std::path::Path;
use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_scene::node_data::{
    Coordinate3Node, IndexedFaceSetNode, MaterialNode, NormalNode, SeparatorNode,
};

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

pub fn parse_step_file_with_options(
    path: &Path,
    options: &StepImportOptions,
) -> Result<SceneGraph, StepError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() > LARGE_STEP_BYTES {
        let exchange = parser::parse_step_from_file_with_options(path, options)
            .map_err(StepError::Parse)?;
        exchange_to_scene_graph(exchange, options)
    } else {
        let text = decode_step_bytes(&bytes);
        parse_step_with_options(&text, options)
    }
}

/// Parse STEP text into a SceneGraph using the OCC-aligned B-Rep pipeline.
pub fn parse_step(input: &str) -> Result<SceneGraph, StepError> {
    parse_step_with_options(input, &StepImportOptions::default())
}

pub fn parse_step_with_options(
    input: &str,
    options: &StepImportOptions,
) -> Result<SceneGraph, StepError> {
    let exchange = parser::parse_exchange_with_options(input, options)
        .map_err(StepError::Parse)?;
    exchange_to_scene_graph(exchange, options)
}

fn exchange_to_scene_graph(
    exchange: parser::Exchange,
    options: &StepImportOptions,
) -> Result<SceneGraph, StepError> {
    let mut import_report = StepImportReport::default();
    import_report.skipped_parse_entities = exchange.diagnostics.skipped_entities.len();

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

    let mut reg = brep_result.registry;
    let shell_instances = assembly::extract_shell_instances(&exchange.entities);
    let shell_styles = assembly::extract_shell_styles(&exchange.entities);

    let assembly_tree = assembly::build_assembly_tree(&exchange.entities);
    if !assembly_tree.nodes.is_empty() && assembly_tree.nodes.iter().any(|n| !n.shells.is_empty()) {
        log::info!(
            "[STEP] assembly tree: {} nodes, {} with geometry",
            assembly_tree.nodes.len(),
            assembly_tree.nodes.iter().filter(|n| !n.shells.is_empty()).count(),
        );
    }

    let heal_config = brep::heal::HealConfig::default();
    let mut total_heal = brep::heal::HealReport::default();
    for &sk in &brep_result.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            total_heal.merge(brep::heal::heal_shell(solid.outer_shell, &mut reg, &heal_config));
        }
    }
    import_report.heal_check_errors = total_heal.check_errors;
    log::info!("[STEP] healed: {:?}", total_heal);
    if options.fail_on_heal_check_errors() && total_heal.check_errors > 0 {
        return Err(StepError::ImportQuality(format!(
            "B-Rep heal reported {} check error(s)",
            total_heal.check_errors
        )));
    }

    let mesh_config = brep::mesh::BRepMeshConfig::default();
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    fn make_material(color: [f32; 3]) -> MaterialNode {
        MaterialNode {
            diffuse_color: Vec3::new(color[0], color[1], color[2]),
            base_color: Vec3::new(color[0], color[1], color[2]),
            roughness: 0.35,
            opacity: 1.0,
            ..Default::default()
        }
    }
    let default_color = [0.9, 0.9, 0.9];
    graph.add_child(root, NodeData::Material(make_material(default_color)));

    let default_xform = assembly::AssemblyTransform::default();
    let mut any_geom = false;
    for &sk in &brep_result.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let base_mesh = brep::mesh::mesh_brep_shell(
                solid.outer_shell,
                &reg,
                &mesh_config,
                &total_heal.skip_face_keys,
            );
            if base_mesh.vertices.is_empty() || base_mesh.indices.is_empty() {
                continue;
            }

            let shell_step_id = reg
                .shells
                .get(solid.outer_shell)
                .and_then(|s| s.step_id);

            let shell_color = shell_step_id
                .and_then(|sid| shell_styles.get(&sid))
                .map(|style| [style.diffuse.x, style.diffuse.y, style.diffuse.z]);

            let mut instances: Vec<&assembly::AssemblyTransform> = shell_step_id
                .map(|sid| {
                    shell_instances
                        .iter()
                        .filter(|(id, _)| *id == sid)
                        .map(|(_, xform)| xform)
                        .collect()
                })
                .unwrap_or_default();
            if instances.is_empty() {
                instances.push(&default_xform);
            }

            for xform in instances {
                let mut mesh = base_mesh.clone();
                apply_mesh_transform(&mut mesh, xform);
                any_geom = true;

                let comp = graph.add_child(root, NodeData::Separator(SeparatorNode));
                graph.add_child(
                    comp,
                    NodeData::Material(make_material(shell_color.unwrap_or(default_color))),
                );
                graph.add_child(
                    comp,
                    NodeData::Coordinate3(Coordinate3Node {
                        point: mesh.vertices,
                    }),
                );
                if !mesh.normals.is_empty() {
                    graph.add_child(
                        comp,
                        NodeData::Normal(NormalNode::from_vectors(mesh.normals)),
                    );
                }
                graph.add_child(
                    comp,
                    NodeData::IndexedFaceSet(IndexedFaceSetNode {
                        coord_index: mesh.indices,
                    }),
                );
            }
        }
    }

    if !any_geom {
        return Err(StepError::NoGeometry);
    }

    log::info!(
        "[STEP] mesh complete: {} solid(s), {} skipped face(s) during heal",
        brep_result.root_solids.len(),
        total_heal.skip_face_keys.len(),
    );

    brep::overlay::build_edge_curves(
        &mut graph,
        root,
        &reg,
        &brep_result.root_solids,
        &shell_instances,
        &mesh_config,
    );

    if import_report.skipped_parse_entities > 0
        || import_report.skipped_faces > 0
        || import_report.skipped_edges > 0
    {
        log::warn!("[STEP] import report: {:?}", import_report);
    }

    if let Some(root_entry) = graph.get_mut(root) {
        root_entry.display_mode = Some(DisplayMode::ShadedWithEdges);
    }
    Ok(graph)
}

fn apply_mesh_transform(mesh: &mut mesh_result::MeshResult, xform: &assembly::AssemblyTransform) {
    for v in &mut mesh.vertices {
        *v = xform.transform_point(*v);
    }
    for n in &mut mesh.normals {
        let t = xform.matrix.transform_vector3(*n);
        let len = t.length();
        if len > 1e-10 {
            *n = t * (1.0 / len);
        }
    }
}
