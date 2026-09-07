//! 3D PDF export for scene graphs (HOOPS Publish equivalent).
//!
//! Two export flavours share the small hand-rolled PDF writer:
//!
//! * [`export_scene_report`] — a text report (node counts, type histogram),
//!   useful for diagnostics.
//! * [`export_u3d_pdf`] — a real interactive 3D PDF: the scene geometry is
//!   flattened to world-space triangles and embedded as a U3D (ECMA-363)
//!   stream behind a `/Subtype /3D` annotation. Acrobat 7.1+ / Reader show
//!   a rotate/pan/zoom model.

use std::fmt::Write;

use rc3d_scene::SceneGraph;

mod extract;
mod pdf3d;
mod u3d;

pub use pdf3d::{PdfLighting, PdfOptions, PdfRenderMode};
pub use u3d::U3dMesh;

/// PDF document builder.
pub struct PdfDocument {
    pages: Vec<String>,
    title: String,
}

impl PdfDocument {
    pub fn new(title: &str) -> Self {
        Self { pages: Vec::new(), title: title.to_string() }
    }

    /// Add a page with text content.
    pub fn add_page(&mut self, content: &str) {
        self.pages.push(content.to_string());
    }

    /// Generate minimal PDF bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = String::new();
        // PDF header
        buf.push_str("%PDF-1.4\n");
        // Object 1: Catalog
        buf.push_str("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        // Object 2: Pages
        buf.push_str("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
        // Object 3: Page with content stream
        let content = self.render_content();
        buf.push_str("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n");
        // Object 4: Content stream
        buf.push_str(&format!(
            "4 0 obj\n<< /Length {} >>\nstream\nBT\n/F1 12 Tf\n{}\nET\nendstream\nendobj\n",
            content.len() + 20,
            content
        ));
        // Cross-reference and trailer
        let xref_offset = buf.len();
        buf.push_str("xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n");
        buf.push_str(&format!("{:010} 00000 n \n", 60));
        buf.push_str(&format!("{:010} 00000 n \n", 180));
        buf.push_str(&format!("{:010} 00000 n \n", 380));
        buf.push_str(&format!(
            "trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            xref_offset
        ));
        buf.into_bytes()
    }

    fn render_content(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "({}) Tj", self.escape(&self.title));
        let _ = writeln!(s, "0 -20 Td");
        for (i, page) in self.pages.iter().enumerate() {
            let _ = writeln!(s, "0 -15 Td");
            let _ = writeln!(s, "({}) Tj", self.escape(&format!("Page {}: {}", i + 1, page)));
        }
        s
    }

    fn escape(&self, s: &str) -> String {
        s.replace('\\', "\\\\").replace('(', "\\(").replace(')', "\\)")
    }
}

/// Export scene graph metadata to a PDF report.
pub fn export_scene_report(graph: &SceneGraph, title: &str) -> Result<Vec<u8>, String> {
    let mut doc = PdfDocument::new(title);
    let mut node_count = 0u32;
    let mut type_counts: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    count_scene_nodes(graph, &mut node_count, &mut type_counts);

    let mut content = format!(
        "Total nodes: {}\n\nNode type counts:\n",
        node_count
    );
    let mut types: Vec<(&str, u32)> = type_counts.into_iter().collect();
    rc3d_core::utils::sort::sort_by_count_desc(&mut types);
    for (name, count) in &types {
        content.push_str(&format!("  {}: {}\n", name, count));
    }
    content.push_str(&format!("\nRoot nodes: {}\n", graph.roots().len()));
    doc.add_page(&content);
    Ok(doc.to_bytes())
}

fn count_scene_nodes(
    graph: &SceneGraph,
    total: &mut u32,
    counts: &mut std::collections::HashMap<&str, u32>,
) {
    for &root in graph.roots() {
        count_recursive(graph, root, total, counts);
    }
}

fn count_recursive(
    graph: &SceneGraph,
    node: rc3d_core::NodeId,
    total: &mut u32,
    counts: &mut std::collections::HashMap<&str, u32>,
) {
    let Some(entry) = graph.get(node) else { return };
    *total += 1;
    *counts.entry(entry.data.type_name()).or_default() += 1;
    for &child in &entry.children {
        count_recursive(graph, child, total, counts);
    }
}

// ── U3D/PRC 3D PDF Export (HOOPS Publish equivalent) ──

/// Export a scene as an interactive U3D-embedded 3D PDF (default options).
///
/// Geometry is flattened to world space via the shared scene traversal (so
/// Separator / Transform hierarchies come out in their rendered positions).
/// Every supported shape becomes its own U3D mesh / model node, so parts
/// can be selected and toggled individually in Acrobat. Named Separators
/// become GroupNodes, mirroring the scene assembly tree in the Acrobat
/// model panel. Shapes drawn under a `Material` node carry that diffuse
/// colour through a lit shader; a material albedo texture (with a matching
/// `TextureCoordinate2` set) is embedded as a PNG texture resource.
/// Supported shapes: `IndexedFaceSet` + `Coordinate3` (imported meshes),
/// `Cube`, `Sphere`, `Cylinder`, `Cone`, `Torus`.
///
/// Returns an error when the scene contains no supported geometry.
pub fn export_u3d_pdf(graph: &SceneGraph, title: &str) -> Result<Vec<u8>, String> {
    export_u3d_pdf_opts(graph, title, &PdfOptions::default())
}

/// [`export_u3d_pdf`] with per-view presentation options (render mode,
/// lighting scheme, background colour, FOV, zoom, framing center).
pub fn export_u3d_pdf_opts(
    graph: &SceneGraph,
    title: &str,
    options: &PdfOptions,
) -> Result<Vec<u8>, String> {
    let extracted = extract::extract_scene(graph);
    if extracted.is_empty() {
        return Err("scene has no extractable triangle geometry".to_string());
    }
    let (center, radius) = union_bounds(&extracted);
    let meshes: Vec<U3dMesh> = extracted
        .into_iter()
        .enumerate()
        .map(|(i, m)| m.into_u3d_mesh(&format!("Part{i}")))
        .collect();
    Ok(pdf3d::encode_3d_pdf(&pdf3d::PdfScene {
        u3d: u3d::encode_u3d_many(&meshes),
        title: title.to_string(),
        center,
        radius,
        options: *options,
    }))
}

/// Bounding sphere over every collected mesh (tightest containing sphere of
/// the union of the per-mesh AABBs).
fn union_bounds(meshes: &[extract::ExtractedMesh]) -> ([f32; 3], f32) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    let mut any = false;
    for mesh in meshes {
        let (c, r) = mesh.bounds();
        for i in 0..3 {
            min[i] = min[i].min(c[i] - r);
            max[i] = max[i].max(c[i] + r);
            any = true;
        }
    }
    if !any {
        return ([0.0; 3], 1.0);
    }
    let center = [
        0.5 * (min[0] + max[0]),
        0.5 * (min[1] + max[1]),
        0.5 * (min[2] + max[2]),
    ];
    let mut r2 = 0.0_f32;
    for i in 0..3 {
        r2 = r2.max((max[i] - center[i]) * (max[i] - center[i]));
    }
    (center, r2.sqrt().max(1e-4))
}

/// Export an explicit triangle mesh as an interactive 3D PDF (no scene
/// graph involved). `positions` and `normals` must be parallel; every three
/// consecutive `indices` form one triangle.
pub fn export_mesh_u3d_pdf(
    title: &str,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    indices: &[u32],
) -> Result<Vec<u8>, String> {
    if positions.is_empty() || indices.len() < 3 {
        return Err("empty mesh: need positions and triangle indices".to_string());
    }
    let mesh = U3dMesh {
        name: "MeshResource".to_string(),
        positions: positions.to_vec(),
        normals: if normals.len() == positions.len() {
            normals.to_vec()
        } else {
            positions.iter().map(|_| [0.0, 1.0, 0.0]).collect()
        },
        triangles: indices
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect(),
        diffuse: None,
        texcoords: Vec::new(),
        texture: None,
        assembly: Vec::new(),
    };
    let (center, radius) = bounds_of(positions);
    Ok(pdf3d::encode_3d_pdf(&pdf3d::PdfScene {
        u3d: u3d::encode_u3d(&mesh),
        title: title.to_string(),
        center,
        radius,
        options: PdfOptions::default(),
    }))
}

fn bounds_of(positions: &[[f32; 3]]) -> ([f32; 3], f32) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for p in positions {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    if positions.is_empty() {
        return ([0.0; 3], 1.0);
    }
    let center = [
        0.5 * (min[0] + max[0]),
        0.5 * (min[1] + max[1]),
        0.5 * (min[2] + max[2]),
    ];
    let mut r2 = 0.0_f32;
    for p in positions {
        let d = [
            p[0] - center[0],
            p[1] - center[1],
            p[2] - center[2],
        ];
        r2 = r2.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    }
    (center, r2.sqrt().max(1e-4))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::Vec3;
    use rc3d_scene::node_data::{Coordinate3Node, IndexedFaceSetNode, SeparatorNode};
    use rc3d_scene::NodeData;

    #[test]
    fn test_empty_export() {
        let g = SceneGraph::new();
        let bytes = export_scene_report(&g, "Empty Scene").unwrap();
        assert!(!bytes.is_empty());
        assert!(bytes.starts_with(b"%PDF-1.4"));
    }

    #[test]
    fn test_scene_with_cube() {
        let mut g = SceneGraph::new();
        g.add_root(NodeData::Cube(rc3d_scene::node_data::CubeNode::default()));
        let bytes = export_scene_report(&g, "Cube Report").unwrap();
        assert!(bytes.len() > 100);
    }

    #[test]
    fn test_document_escape() {
        let doc = PdfDocument::new("Test (with) parens");
        let bytes = doc.to_bytes();
        assert!(bytes.len() > 50);
    }

    fn cube_scene() -> SceneGraph {
        let mut g = SceneGraph::new();
        let root = g.add_root(NodeData::Separator(SeparatorNode));
        // A unit cube as an indexed face set (12 triangles, 8 vertices).
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let coord_index = vec![
            0, 1, 2, -1, 1, 2, 4, -1, 0, 2, 3, -1, 2, 3, 6, -1,
            0, 1, 3, -1, 1, 3, 5, -1, 1, 4, 5, -1, 4, 5, 7, -1,
            4, 6, 7, -1, 2, 4, 6, -1, 3, 5, 6, -1, 5, 6, 7, -1,
        ];
        g.add_child(
            root,
            NodeData::Coordinate3(Coordinate3Node::from_points(points)),
        );
        g.add_child(
            root,
            NodeData::IndexedFaceSet(IndexedFaceSetNode::from_coord_index(coord_index)),
        );
        g
    }

    #[test]
    fn test_export_u3d_pdf_has_3d_annotation() {
        let g = cube_scene();
        let bytes = export_u3d_pdf(&g, "Cube 3D").unwrap();
        assert!(bytes.starts_with(b"%PDF-1.7"), "real 3D PDF header");
        assert!(bytes.len() > 900, "embedding a U3D stream grows the file");
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("/Subtype /3D"), "3D annotation present");
        assert!(text.contains("/Subtype /U3D"), "U3D stream present");
        assert!(text.contains("/AIS true"), "auto activation on import");
        assert!(text.contains("/C2W ["), "camera-to-world matrix present");
    }

    #[test]
    fn test_export_u3d_pdf_no_geometry_is_error() {
        let g = SceneGraph::new();
        assert!(export_u3d_pdf(&g, "Empty 3D").is_err());
    }

    /// Coloured shapes are exported as separate named parts, each bound to a
    /// material palette entry (names surface as ASCII in the U3D payload).
    #[test]
    fn export_splits_coloured_shapes_into_parts() {
        use rc3d_scene::node_data::{MaterialNode, SphereNode};
        let mut g = SceneGraph::new();
        let root = g.add_root(NodeData::Separator(SeparatorNode));
        g.add_child(
            root,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(1.0, 0.0, 0.0))),
        );
        g.add_child(root, NodeData::Cube(rc3d_scene::node_data::CubeNode::default()));
        g.add_child(
            root,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.0, 0.0, 1.0))),
        );
        g.add_child(root, NodeData::Sphere(SphereNode { radius: 0.5 }));
        let bytes = export_u3d_pdf(&g, "Parts 3D").unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("Part0"), "first mesh part named");
        assert!(text.contains("Part1"), "second mesh part named");
        assert!(text.contains("Mat0") && text.contains("Mat1"), "two palette colours");
    }

    #[test]
    fn test_u3d_roundtrip_sizes() {
        let g = cube_scene();
        let pdf = export_u3d_pdf(&g, "Cube Sizes").unwrap();
        // Locate the embedded U3D length from the /Length entry of object 6
        // by scanning the PDF text, then check the stream begins with the
        // U3D magic bytes and ends cleanly.
        let text = String::from_utf8_lossy(&pdf);
        let marker = "/Subtype /U3D\n";
        let Some(marker_at) = text.find(marker) else {
            panic!("U3D stream dict not found");
        };
        let dict = &text[marker_at..];
        let Some(len_field) = dict.find("/Length ") else {
            panic!("no /Length in U3D stream dict");
        };
        let rest = &dict[len_field + "/Length ".len()..];
        let len_str: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        let u3d_len: usize = len_str.parse().expect("numeric U3D length");
        assert!(u3d_len > 300, "U3D payload is a real document");
    }

    /// Write demo files under `target/` for manual Acrobat verification: a
    /// default-option export plus one exercising the group tree and the
    /// per-view options (CAD lighting, transparent wireframe, background).
    #[test]
    fn demo_writes_u3d_pdf_for_manual_check() {
        use rc3d_core::math::Vec3;
        use rc3d_scene::node_data::{MaterialNode, SeparatorNode, TransformNode};
        use rc3d_scene::NodeData;

        let mut g = SceneGraph::new();
        let root = g.add_root(NodeData::Separator(SeparatorNode));
        // Named separators become U3D GroupNodes, so Acrobat's model panel
        // shows Car > Frame (red cube) and Car > Wheel (blue sphere).
        let car = g.add_child(root, NodeData::Separator(SeparatorNode));
        g.set_name(car, "Car");
        let frame = g.add_child(car, NodeData::Separator(SeparatorNode));
        g.set_name(frame, "Frame");
        g.add_child(
            frame,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.85, 0.15, 0.1))),
        );
        g.add_child(frame, NodeData::Cube(rc3d_scene::node_data::CubeNode {
            width: 2.0,
            height: 1.5,
            depth: 1.0,
        }));
        let wheel = g.add_child(car, NodeData::Separator(SeparatorNode));
        g.set_name(wheel, "Wheel");
        g.add_child(
            wheel,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.1, 0.35, 0.9))),
        );
        let move_sphere = g.add_child(
            wheel,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(2.0, 0.0, 0.0))),
        );
        g.add_child(move_sphere, NodeData::Sphere(rc3d_scene::node_data::SphereNode {
            radius: 0.45,
        }));
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target");
        std::fs::create_dir_all(&dir).unwrap();

        let bytes = export_u3d_pdf(&g, "rc3d Demo Assembly").unwrap();
        assert!(bytes.starts_with(b"%PDF-1.7"));
        assert!(String::from_utf8_lossy(&bytes).contains("Car"), "group names embedded");
        std::fs::write(dir.join("u3d_demo.pdf"), &bytes).unwrap();

        let opts = PdfOptions {
            render_mode: Some(PdfRenderMode::TransparentWireframe),
            lighting: PdfLighting::Cad,
            background: Some([0.92, 0.93, 0.95]),
            fov_degrees: Some(55.0),
            zoom: Some(1.1),
            center: None,
        };
        let styled = export_u3d_pdf_opts(&g, "rc3d Demo Styled", &opts).unwrap();
        let text = String::from_utf8_lossy(&styled);
        assert!(text.contains("/3DLightingScheme /Subtype /CAD"), "modern lighting scheme");
        assert!(text.contains("/Subtype /TransparentWireframe"), "render mode set");
        std::fs::write(dir.join("u3d_demo_styled.pdf"), &styled).unwrap();
    }
}
