//! 3D PDF export for scene graphs (HOOPS Publish equivalent).
//!
//! Generates a PDF report with embedded scene metadata, node hierarchy,
//! and statistics. Full U3D/PRC 3D embedding requires native library
//! integration (planned for C-API bridge).

use std::fmt::Write;
use rc3d_scene::SceneGraph;

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
        buf.push_str(&format!("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"));
        // Object 3: Page with content stream
        let content = self.render_content();
        buf.push_str(&format!(
            "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        ));
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
    types.sort_by(|a, b| b.1.cmp(&a.1));
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

/// Export scene as U3D-embedded 3D PDF.
///
/// Full U3D embedding requires a native C library (libu3d, Adobe U3D SDK).
/// This function returns a placeholder PDF that documents the export path.
///
/// Integration path for full 3D PDF:
/// 1. Compile libu3d as a static library
/// 2. Create Rust FFI bindings via `extern "C"` block
/// 3. Serialize scene graph triangles to U3D binary format
/// 4. Embed U3D stream in PDF with 3D annotation
pub fn export_u3d_pdf(graph: &SceneGraph, title: &str) -> Result<Vec<u8>, String> {
    let mut doc = PdfDocument::new(title);

    let node_count = count_total(graph);
    let root_count = graph.roots().len();

    // Collect node-type distribution
    let mut type_counts: std::collections::HashMap<&str, u32> =
        std::collections::HashMap::new();
    {
        let mut n = 0;
        let mut counts = std::collections::HashMap::new();
        count_scene_nodes(graph, &mut n, &mut counts);
        type_counts = counts;
    }
    let mut types: Vec<(&str, u32)> = type_counts.into_iter().collect();
    types.sort_by(|a, b| b.1.cmp(&a.1));

    let bounds_line = "Bounding box: (not computed — offline)";

    let content = format!(
        "3D PDF Export — Scene Statistics\n\n\
         Scene: {title}\n\
         Total nodes: {node_count}\n\
         Root nodes: {root_count}\n\
         {bounds_line}\n\n\
         Node type distribution:\n{}\n\
         \n\
         Note: Full 3D embedding (U3D/PRC) requires a native C library.\n\
         See crate documentation for integration path.",
        types.iter()
            .map(|(name, count)| format!("  {}: {}", name, count))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    doc.add_page(&content);
    Ok(doc.to_bytes())
}

fn count_total(graph: &SceneGraph) -> u32 {
    let mut n = 0;
    let mut counts = std::collections::HashMap::new();
    count_scene_nodes(graph, &mut n, &mut counts);
    n
}

#[cfg(test)]
mod u3d_tests {
    use super::*;
    use rc3d_scene::node_data::CubeNode;

    #[test]
    fn test_u3d_export_placeholder() {
        let mut g = SceneGraph::new();
        g.add_root(rc3d_scene::NodeData::Cube(CubeNode::default()));
        let bytes = export_u3d_pdf(&g, "3D Test").unwrap();
        assert!(bytes.starts_with(b"%PDF-1.4"));
        assert!(bytes.len() > 100);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
