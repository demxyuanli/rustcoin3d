pub mod value;
pub mod entity_types;
pub mod surface_tess;
#[cfg(feature = "pmi")]
pub mod pmi;
pub mod parser;
pub mod topology;
pub mod geom;
pub mod pcurve;
pub mod tessellate;
pub mod assembly;
pub mod nurbs;
pub mod write;
pub mod validate;
pub mod xml;
pub mod bool;
pub mod tree;
pub mod curve;
pub mod header;
pub mod topo;
pub mod refine;
pub mod lod;
pub mod fillet;

use std::path::Path;
use std::collections::HashMap;
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::DisplayMode;
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_scene::node_data::{
    Coordinate3Node, IndexedFaceSetNode, IndexedLineSetNode, MaterialNode, NormalNode, SeparatorNode, TransformNode,
};
use crate::step::tree::{AssemblyTree, ProductMetadata};
use crate::step::header::HeaderInfo;

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

/// Full STEP import result with metadata and assembly tree.
pub struct StepImportResult {
    pub graph: SceneGraph,
    pub assembly_tree: Option<AssemblyTree>,
    pub metadata: HashMap<u64, ProductMetadata>,
    pub header: Option<HeaderInfo>,
}

/// Parse STEP text with full metadata extraction.
pub fn parse_step_full(input: &str) -> Result<StepImportResult, StepError> {
    let exchange = parser::parse_exchange(input)
        .map_err(|e| StepError::Parse(e))?;
    let header = exchange.header.clone();
    let metadata = tree::extract_all_metadata(&exchange.entities);
    let assembly_tree = assembly::build_assembly_tree(&exchange.entities);
    let graph = parse_step(input)?;

    Ok(StepImportResult {
        graph,
        assembly_tree: Some(assembly_tree),
        metadata,
        header,
    })
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
    // Read entire file (reliable for STEP files up to ~500MB)
    let bytes = std::fs::read(path)?;
    let text = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("STEP file is not valid UTF-8, replacing invalid bytes");
            String::from_utf8_lossy(e.as_bytes()).into_owned()
        }
    };
    parse_step(&text)
}

pub fn parse_step(input: &str) -> Result<SceneGraph, StepError> {
    parse_step_with_options(input, false)
}

pub fn parse_step_with_shared_topology(input: &str) -> Result<SceneGraph, StepError> {
    parse_step_with_options(input, true)
}

fn parse_step_with_options(input: &str, use_shared_topology: bool) -> Result<SceneGraph, StepError> {
    let exchange = parser::parse_exchange(input)
        .map_err(|e| StepError::Parse(e))?;

    // Run validation and log issues
    let report = validate::validate(&exchange.entities);
    let entity_count = exchange.entities.len();
    log::info!(
        "[STEP] {} entities, {} shells, {} faces, {} points",
        entity_count,
        report.topology_info.shells,
        report.topology_info.faces,
        report.topology_info.points,
    );
    for w in &report.warnings {
        log::warn!("[STEP] validation: {}", w);
    }

    let shells = topology::collect_shells(&exchange.entities);
    if shells.is_empty() {
        if !report.errors.is_empty() {
            return Err(StepError::Validation(report.errors.join("; ")));
        }
        return Err(StepError::NoGeometry);
    }

    let transforms = assembly::extract_shell_transforms(&exchange.entities);
    let styles = assembly::extract_shell_styles(&exchange.entities);
    eprintln!(
        "[STEP] {} shells, {} transforms, {} styles",
        shells.len(),
        transforms.len(),
        styles.len()
    );

    if use_shared_topology {
        let topo_result = topo::build::build_shared_topology(&shells, &exchange.entities);
        log::info!(
            "[STEP] Shared topology: {} unique vertices, {} unique edges across {} shells",
            topo_result.vertices.len(), topo_result.edges.len(), topo_result.shells.len(),
        );
        build_hierarchical_scene_from_topo(
            &topo_result, &transforms, &styles, &exchange.entities
        )
    } else {
        let mut graph = build_hierarchical_scene(&shells, &transforms, &styles, &exchange.entities)?;

        // Add STEP original edge curves as lines
        let edge_count = build_step_edges_overlay(&mut graph, &shells, &exchange.entities);
        eprintln!("[STEP] {} edge curves rendered", edge_count);

        Ok(graph)
    }
}

/// Extract unique edge curves from shells and add them as IndexedLineSet geometry.
/// Renders the original B-rep edges (not tessellated triangle edges).
fn build_step_edges_overlay(
    graph: &mut SceneGraph,
    shells: &[topology::StepShell],
    entities: &parser::EntityIndex,
) -> usize {
    use std::collections::HashSet;
    let mut edge_keys: HashSet<(u32, u32, u32, u32, u32, u32)> = HashSet::new();
    let mut all_pts: Vec<Vec3> = Vec::new();
    let mut all_indices: Vec<i32> = Vec::new();

    for shell in shells {
        for face in &shell.faces {
            for bloop in &face.bounds {
                for edge in &bloop.edges {
                    let sk = rc3d_core::utils::hash::f32x3_quantized_bits([
                        edge.start.x, edge.start.y, edge.start.z,
                    ]);
                    let ek = rc3d_core::utils::hash::f32x3_quantized_bits([
                        edge.end.x, edge.end.y, edge.end.z,
                    ]);
                    let key = (sk[0], sk[1], sk[2], ek[0], ek[1], ek[2]);
                    if !edge_keys.insert(key) { continue; }

                    // Sample the edge curve
                    let pts = geom::sample_curve(
                        edge.curve_id, entities, edge.start, edge.end, edge.tolerance.max(0.05),
                    );
                    if pts.len() < 2 { continue; }

                    let base = all_pts.len() as i32;
                    for pt in &pts {
                        all_pts.push(*pt);
                        all_indices.push(base + (all_pts.len() as i32 - base - 1));
                    }
                    all_indices.push(-1); // sentinel
                }
            }
        }
    }

    if all_pts.is_empty() { return 0; }

    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    // Material must come before geometry (Coin3D traversal order)
    graph.add_child(
        root,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.6, 0.6, 0.6),
            base_color: Vec3::new(0.6, 0.6, 0.6),
            emissive_color: Vec3::new(0.8, 0.8, 0.8),
            roughness: 1.0,
            metallic: 0.0,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(root, NodeData::Coordinate3(Coordinate3Node {
        point: all_pts,
    }));
    graph.add_child(
        root,
        NodeData::IndexedLineSet(IndexedLineSetNode {
            coord_index: all_indices,
            line_width: 2.0,
        }),
    );

    edge_keys.len()
}

fn build_hierarchical_scene_from_topo(
    topo: &topo::build::TopoBuildResult,
    _transforms: &assembly::ShellTransformMap,
    _styles: &assembly::ShellStyleMap,
    entities: &parser::EntityIndex,
) -> Result<SceneGraph, StepError> {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    let default_material = MaterialNode {
        diffuse_color: Vec3::new(0.9, 0.9, 0.9),
        ambient_color: Vec3::new(0.35, 0.35, 0.35),
        specular_color: Vec3::new(0.0, 0.0, 0.0),
        shininess: 0.0,
        base_color: Vec3::new(0.94, 0.94, 0.94),
        metallic: 0.0,
        roughness: 0.35,
        opacity: 1.0,
        ..Default::default()
    };
    graph.add_child(
        root,
        NodeData::Material(MaterialNode::from_diffuse(default_material.diffuse_color)),
    );

    let mut any_geometry = false;

    for (_si, topo_shell) in topo.shells.iter().enumerate() {
        // Convert TopoShell back to flat faces for tessellation
        let flat_faces: Vec<topology::StepFace> = topo_shell.faces.iter().map(|face| {
            let mut bounds = Vec::new();
            for loop_i in std::iter::once(&face.outer_loop).chain(face.inner_loops.iter()) {
                let edges: Vec<topology::StepEdge> = loop_i.edges.iter().map(|&(edge_id, reversed)| {
                    let te = topo.edges.get(edge_id).unwrap();
                    let start = topo.vertices.get(te.start).map(|v| v.position)
                        .unwrap_or(Vec3::ZERO);
                    let end = topo.vertices.get(te.end).map(|v| v.position)
                        .unwrap_or(Vec3::ZERO);
                    topology::StepEdge {
                        start,
                        end,
                        curve_id: te.curve_entity_id,
                        curve_type: "LINE".into(),
                        reversed: if te.sense == topo::EdgeSense::Forward { reversed } else { !reversed },
                        tolerance: te.tolerance,
                    }
                }).collect();
                if !edges.is_empty() {
                    bounds.push(topology::StepLoop { edges });
                }
            }
            topology::StepFace {
                bounds,
                surface_id: face.surface_entity_id,
                same_sense: face.same_sense,
            }
        }).collect();

        let mesh = tessellate::tessellate_faces(&flat_faces, entities);
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            continue;
        }
        any_geometry = true;

        let component = graph.add_child(root, NodeData::Separator(SeparatorNode));

        let shell_material = MaterialNode {
            diffuse_color: Vec3::new(0.9, 0.9, 0.9),
            base_color: Vec3::new(0.94, 0.94, 0.94),
            roughness: 0.35,
            opacity: 1.0,
            ..Default::default()
        };
        graph.add_child(component, NodeData::Material(shell_material));

        graph.add_child(
            component,
            NodeData::Coordinate3(Coordinate3Node { point: mesh.vertices }),
        );
        if !mesh.normals.is_empty() {
            graph.add_child(
                component,
                NodeData::Normal(NormalNode::from_vectors(mesh.normals)),
            );
        }
        graph.add_child(
            component,
            NodeData::IndexedFaceSet(IndexedFaceSetNode { coord_index: mesh.indices }),
        );
    }

    if !any_geometry {
        return Err(StepError::NoGeometry);
    }

    if let Some(root_entry) = graph.get_mut(root) {
        root_entry.display_mode = Some(DisplayMode::ShadedWithEdges);
    }
    Ok(graph)
}

fn build_hierarchical_scene(
    shells: &[topology::StepShell],
    transforms: &assembly::ShellTransformMap,
    styles: &assembly::ShellStyleMap,
    entities: &parser::EntityIndex,
) -> Result<SceneGraph, StepError> {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Default material (used when no style is found)
    let default_material = MaterialNode {
        diffuse_color: Vec3::new(0.9, 0.9, 0.9),
        ambient_color: Vec3::new(0.35, 0.35, 0.35),
        specular_color: Vec3::new(0.0, 0.0, 0.0),
        shininess: 0.0,
        base_color: Vec3::new(0.94, 0.94, 0.94),
        metallic: 0.0,
        roughness: 0.35,
        opacity: 1.0,
        ..Default::default()
    };
    graph.add_child(
        root,
        NodeData::Material(MaterialNode::from_diffuse(default_material.diffuse_color)),
    );

    let mut any_geometry = false;

    for (_shell_idx, shell) in shells.iter().enumerate() {
        let mesh = tessellate::tessellate_faces(&shell.faces, entities);
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            continue;
        }
        any_geometry = true;

        let component = graph.add_child(root, NodeData::Separator(SeparatorNode));

        // Create a Material node for each shell (uses style color or default)
        let style = styles.get(&shell.id);
        let material = if let Some(s) = style {
            MaterialNode {
                diffuse_color: s.diffuse,
                base_color: s.diffuse,
                opacity: s.opacity,
                ..default_material.clone()
            }
        } else {
            default_material.clone()
        };
        graph.add_child(component, NodeData::Material(material));

        if let Some(xform) = transforms.get(&shell.id) {
            let (scale, rotation, translation) = xform.matrix.to_scale_rotation_translation();
            graph.add_child(
                component,
                NodeData::Transform(TransformNode {
                    scale,
                    rotation: Mat4::from_quat(rotation),
                    translation,
                    ..Default::default()
                }),
            );
        }

        graph.add_child(
            component,
            NodeData::Coordinate3(Coordinate3Node { point: mesh.vertices }),
        );
        if !mesh.normals.is_empty() {
            graph.add_child(
                component,
                NodeData::Normal(NormalNode::from_vectors(mesh.normals)),
            );
        }
        graph.add_child(
            component,
            NodeData::IndexedFaceSet(IndexedFaceSetNode { coord_index: mesh.indices }),
        );
    }

    if !any_geometry {
        return Err(StepError::NoGeometry);
    }

    // Attach PMI annotations if present
    #[cfg(feature = "pmi")]
    {
        let pmi_data = pmi::pmi_extract::extract_pmi(entities);
        pmi::pmi_render::attach_pmi_to_scene(&mut graph, root, &pmi_data);
    }

    if let Some(root_entry) = graph.get_mut(root) {
        root_entry.display_mode = Some(DisplayMode::ShadedWithEdges);
    }
    Ok(graph)
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use rc3d_core::NodeId;

    #[test]
    fn test_assembly_example_loads() {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_data/AssemblyExample-Assembly.step"));
        if !path.exists() {
            eprintln!("Skipping test: {} not found", path.display());
            return;
        }
        let graph = parse_step_file(path).expect("should parse AssemblyExample");
        // Count nodes with geometry by traversing from roots
        let mut mesh_count = 0;
        let mut total_verts = 0;
        let mut total_indices = 0;
        let mut stack: Vec<NodeId> = graph.roots().to_vec();
        while let Some(id) = stack.pop() {
            if let Some(entry) = graph.get(id) {
                match &entry.data {
                    NodeData::IndexedFaceSet(ifs) => {
                        mesh_count += 1;
                        total_indices += ifs.coord_index.len();
                    }
                    NodeData::Coordinate3(coord) => {
                        total_verts += coord.point.len();
                    }
                    _ => {}
                }
                stack.extend(entry.children.iter().copied());
            }
        }
        eprintln!(
            "AssemblyExample: {} meshes, {} vertices, {} indices",
            mesh_count, total_verts, total_indices
        );
        assert!(mesh_count > 0, "should have at least one mesh");
        assert!(total_verts > 100, "should have substantial vertices");
    }

    #[test]
    fn test_shape_step_loads() {
        // Test that Shape.step (a vase defined by surface of revolution) loads correctly
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_data/Shape.step"));
        if !path.exists() {
            eprintln!("Skipping test: {} not found", path.display());
            return;
        }
        let graph = parse_step_file(path).expect("should parse Shape.step");
        // Count nodes with geometry by traversing from roots
        let mut mesh_count = 0;
        let mut total_verts = 0;
        let mut total_indices = 0;
        let mut stack: Vec<NodeId> = graph.roots().to_vec();
        while let Some(id) = stack.pop() {
            if let Some(entry) = graph.get(id) {
                match &entry.data {
                    NodeData::IndexedFaceSet(ifs) => {
                        mesh_count += 1;
                        total_indices += ifs.coord_index.len();
                    }
                    NodeData::Coordinate3(coord) => {
                        total_verts += coord.point.len();
                    }
                    _ => {}
                }
                stack.extend(entry.children.iter().copied());
            }
        }
        eprintln!(
            "Shape.step: {} meshes, {} vertices, {} indices",
            mesh_count, total_verts, total_indices
        );
        // Shape.step defines a vase using surface of revolution
        // It should produce a closed mesh with proper revolution
        assert!(mesh_count > 0, "should have at least one mesh");
        assert!(total_verts > 1000, "vase should have substantial vertices, got {}", total_verts);
        assert!(total_indices > 5000, "vase should have substantial triangles, got {}", total_indices);
    }
}
