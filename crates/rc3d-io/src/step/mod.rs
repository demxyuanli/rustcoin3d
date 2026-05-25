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
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8(bytes)
        .map_err(|e| StepError::Parse(format!("invalid UTF-8: {}", e)))?;
    parse_step(&text)
}

/// Parse STEP text into a SceneGraph using the B-Rep pipeline.
/// Builds parametric B-Rep → heals → meshes with CDT + refinement + optimization.
pub fn parse_step(input: &str) -> Result<SceneGraph, StepError> {
    let exchange = parser::parse_exchange(input)
        .map_err(|e| StepError::Parse(e))?;

    let report = validate::validate(&exchange.entities);
    log::info!(
        "[STEP] {} entities, {} shells, {} faces",
        exchange.entities.len(),
        report.topology_info.shells,
        report.topology_info.faces,
    );
    for w in &report.warnings { log::warn!("[STEP] {}", w); }

    let brep_result = brep::build_brep(&exchange.entities)?;
    let mut reg = brep_result.registry;

    // Heal
    let heal_config = brep::heal::HealConfig::default();
    let mut total_heal = brep::heal::HealReport::default();
    for &sk in &brep_result.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            total_heal.merge(brep::heal::heal_shell(solid.outer_shell, &mut reg, &heal_config));
        }
    }
    log::info!("[STEP] healed: {:?}", total_heal);

    // Mesh
    let mesh_config = brep::mesh::BRepMeshConfig::default();
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    let default_mat = MaterialNode {
        diffuse_color: Vec3::new(0.9, 0.9, 0.9),
        base_color: Vec3::new(0.94, 0.94, 0.94),
        roughness: 0.35, opacity: 1.0, ..Default::default()
    };
    graph.add_child(root, NodeData::Material(default_mat));

    let mut any_geom = false;
    for &sk in &brep_result.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let mesh = brep::mesh::mesh_brep_shell(solid.outer_shell, &reg, &mesh_config);
            if mesh.vertices.is_empty() || mesh.indices.is_empty() { continue; }
            any_geom = true;

            let comp = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(comp, NodeData::Coordinate3(Coordinate3Node { point: mesh.vertices }));
            if !mesh.normals.is_empty() {
                graph.add_child(comp, NodeData::Normal(NormalNode::from_vectors(mesh.normals)));
            }
            graph.add_child(comp, NodeData::IndexedFaceSet(IndexedFaceSetNode { coord_index: mesh.indices }));
        }
    }

    if !any_geom { return Err(StepError::NoGeometry); }

    if let Some(root_entry) = graph.get_mut(root) {
        root_entry.display_mode = Some(DisplayMode::ShadedWithEdges);
    }
    Ok(graph)
}
