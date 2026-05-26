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

/// Parse STEP text into a SceneGraph using the OCC-aligned B-Rep pipeline:
/// StepToTopoDS (build_brep) → ShapeFix (heal) → BRepMesh_IncrementalMesh (mesh_brep_shell).
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
    let shell_transforms = assembly::extract_shell_transforms(&exchange.entities);
    let shell_styles = assembly::extract_shell_styles(&exchange.entities);

    // Assembly tree (P2: hierarchical Transform nodes — currently flat transforms are
    // applied per-shell via shell_transforms, which is correct for single-product files.
    // TODO: per-product Separator→Transform→[Material→Coordinate3→IFS] using the tree.)
    let assembly_tree = assembly::build_assembly_tree(&exchange.entities);
    if !assembly_tree.nodes.is_empty() && assembly_tree.nodes.iter().any(|n| !n.shells.is_empty()) {
        log::info!(
            "[STEP] assembly tree: {} nodes, {} with geometry (flat rendering)",
            assembly_tree.nodes.len(),
            assembly_tree.nodes.iter().filter(|n| !n.shells.is_empty()).count(),
        );
    }

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

    // Default material color for shells without explicit styling
    fn make_material(color: [f32; 3]) -> MaterialNode {
        MaterialNode {
            diffuse_color: Vec3::new(color[0], color[1], color[2]),
            base_color: Vec3::new(color[0], color[1], color[2]),
            roughness: 0.35, opacity: 1.0, ..Default::default()
        }
    }
    let default_color = [0.9, 0.9, 0.9];
    graph.add_child(root, NodeData::Material(make_material(default_color)));

    let mut any_geom = false;
    for &sk in &brep_result.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let mut mesh = brep::mesh::mesh_brep_shell(solid.outer_shell, &reg, &mesh_config, &total_heal.skip_face_keys);

            // Look up per-shell color from STYLED_ITEM
            let shell_color = reg.shells.get(solid.outer_shell)
                .and_then(|s| s.step_id)
                .and_then(|sid| shell_styles.get(&sid))
                .map(|style| [style.diffuse.x, style.diffuse.y, style.diffuse.z]);

            if let Some(shell) = reg.shells.get(solid.outer_shell) {
                if let Some(step_id) = shell.step_id {
                    if let Some(xform) = shell_transforms.get(&step_id) {
                        apply_mesh_transform(&mut mesh, xform);
                    }
                }
            }
            if mesh.vertices.is_empty() || mesh.indices.is_empty() { continue; }
            any_geom = true;

            let comp = graph.add_child(root, NodeData::Separator(SeparatorNode));
            // Per-shell material with color from STEP if available
            graph.add_child(comp, NodeData::Material(make_material(
                shell_color.unwrap_or(default_color)
            )));
            graph.add_child(comp, NodeData::Coordinate3(Coordinate3Node { point: mesh.vertices }));
            if !mesh.normals.is_empty() {
                graph.add_child(comp, NodeData::Normal(NormalNode::from_vectors(mesh.normals)));
            }
            graph.add_child(comp, NodeData::IndexedFaceSet(IndexedFaceSetNode { coord_index: mesh.indices }));
        }
    }

    if !any_geom { return Err(StepError::NoGeometry); }

    brep::overlay::build_edge_curves(
        &mut graph,
        root,
        &reg,
        &brep_result.root_solids,
        &shell_transforms,
        &mesh_config,
    );

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
