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

use std::path::Path;
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::DisplayMode;
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_scene::node_data::{
    Coordinate3Node, IndexedFaceSetNode, MaterialNode, NormalNode, SeparatorNode, TransformNode,
};

#[derive(Debug, thiserror::Error)]
pub enum StepError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("STEP parse error: {0}")]
    Parse(String),
    #[error("No geometry found in STEP file")]
    NoGeometry,
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
    let exchange = parser::parse_exchange(input)
        .map_err(|e| StepError::Parse(e))?;

    let shells = topology::collect_shells(&exchange.entities);
    if shells.is_empty() {
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

    let graph = build_hierarchical_scene(&shells, &transforms, &styles, &exchange.entities)?;
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
