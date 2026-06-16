use rc3d_core::math::Real;
//! BREP structure and geometry diff tools for OCC alignment verification.
//!
//! To use with OCC ground truth:
//! 1. Generate OCC .brep file: `DRAWEXE test.brep box b 10 10 10 ; brepsave box`
//! 2. Parse the OCC .brep sections and compare with our output.

use std::collections::HashMap;

// ── Task 8: Structure diff ──────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct TopoCounts {
    pub vertices: usize,
    pub edges: usize,
    pub wires: usize,
    pub faces: usize,
    pub shells: usize,
    pub solids: usize,
    pub compounds: usize,
    pub pcurves_total: usize,
}

impl TopoCounts {
    /// Extract topology counts from a BRepStep output.
    /// Note: currently works with our BRepStore; OCC .brep parsing to be added.
    pub fn from_brep_text(text: &str) -> Self {
        let mut counts = TopoCounts::default();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("TVertexes ") {
                counts.vertices = line.split_whitespace().nth(1)
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
            } else if line.starts_with("TEdges ") {
                counts.edges = line.split_whitespace().nth(1)
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
            } else if line.starts_with("TWires ") {
                counts.wires = line.split_whitespace().nth(1)
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
            } else if line.starts_with("TFaces ") {
                counts.faces = line.split_whitespace().nth(1)
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
            } else if line.starts_with("TShells ") {
                counts.shells = line.split_whitespace().nth(1)
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
            } else if line.starts_with("TSolids ") {
                counts.solids = line.split_whitespace().nth(1)
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
            } else if line.starts_with("TCompounds ") {
                counts.compounds = line.split_whitespace().nth(1)
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
            } else if line.starts_with("Curve2ds ") {
                counts.pcurves_total = line.split_whitespace().nth(1)
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
            }
        }
        counts
    }

    /// Diff two topology counts. Returns score 0.0–1.0 (1.0 = identical counts).
    pub fn diff_score(&self, other: &TopoCounts) -> Real {
        let fields: [(usize, usize); 8] = [
            (self.vertices, other.vertices),
            (self.edges, other.edges),
            (self.wires, other.wires),
            (self.faces, other.faces),
            (self.shells, other.shells),
            (self.solids, other.solids),
            (self.compounds, other.compounds),
            (self.pcurves_total, other.pcurves_total),
        ];
        let matched = fields.iter().filter(|(a, b)| a == b).count();
        matched as Real / fields.len() as Real
    }

    /// Format a multi-line diff report.
    pub fn diff_report(&self, other: &TopoCounts, step_name: &str) -> String {
        let mut report = format!("=== Topology diff for {} ===\n", step_name);
        let fields: [(&str, usize, usize); 8] = [
            ("vertices", self.vertices, other.vertices),
            ("edges", self.edges, other.edges),
            ("wires", self.wires, other.wires),
            ("faces", self.faces, other.faces),
            ("shells", self.shells, other.shells),
            ("solids", self.solids, other.solids),
            ("compounds", self.compounds, other.compounds),
            ("pcurves_total", self.pcurves_total, other.pcurves_total),
        ];
        for (name, a, b) in &fields {
            let marker = if a == b { "✓" } else { "✗" };
            report.push_str(&format!("  {} {}: ours={}  occ={}\n", marker, name, a, b));
        }
        report.push_str(&format!("  Score: {:.2} / 1.00\n", self.diff_score(other)));
        report
    }
}

// ── Task 10: Geometry properties diff ───────────────────────────────────

/// Geometric properties for diff comparison.
#[derive(Debug, Default, Clone)]
pub struct GeomProps {
    pub face_areas: Vec<Real>,
    pub edge_lengths: Vec<Real>,
    pub vertex_positions: Vec<[Real; 3]>,
}

/// Result of geometry diff comparison.
#[derive(Debug, Default, Clone)]
pub struct GeomDiff {
    pub face_area_score: Real,      // 0.0–1.0
    pub edge_length_score: Real,
    pub vertex_position_score: Real,
    pub overall_score: Real,
}

/// Compare two lists of floats with a relative epsilon.
fn compare_lists(a: &[Real], b: &[Real], rel_eps: Real) -> Real {
    if a.is_empty() && b.is_empty() { return 1.0; }
    let max_len = a.len().max(b.len());
    if max_len == 0 { return 1.0; }
    // Sort copies for order-independent comparison
    let mut sa: Vec<Real> = a.iter().copied().collect();
    let mut sb: Vec<Real> = b.iter().copied().collect();
    sa.sort_by(Real::total_cmp);
    sb.sort_by(Real::total_cmp);
    let n = sa.len().min(sb.len());
    let mut matched = 0;
    for i in 0..n {
        let va = sa[i];
        let vb = sb[i];
        let denom = va.abs().max(vb.abs()).max(1e-6);
        if (va - vb).abs() / denom <= rel_eps {
            matched += 1;
        }
    }
    matched as Real / max_len as Real
}

/// Compare two lists of 3D positions with absolute epsilon.
fn compare_positions(a: &[[Real; 3]], b: &[[Real; 3]], abs_eps: Real) -> Real {
    if a.is_empty() && b.is_empty() { return 1.0; }
    let max_len = a.len().max(b.len());
    if max_len == 0 { return 1.0; }
    // Compare pairwise by proximity (simplified: sort by x, then y, then z)
    let mut sa: Vec<[Real; 3]> = a.to_vec();
    let mut sb: Vec<[Real; 3]> = b.to_vec();
    sa.sort_by(|p, q| {
        p[0].total_cmp(&q[0])
            .then(p[1].total_cmp(&q[1]))
            .then(p[2].total_cmp(&q[2]))
    });
    sb.sort_by(|p, q| {
        p[0].total_cmp(&q[0])
            .then(p[1].total_cmp(&q[1]))
            .then(p[2].total_cmp(&q[2]))
    });
    let n = sa.len().min(sb.len());
    let mut matched = 0;
    for i in 0..n {
        let dx = (sa[i][0] - sb[i][0]).abs();
        let dy = (sa[i][1] - sb[i][1]).abs();
        let dz = (sa[i][2] - sb[i][2]).abs();
        if dx <= abs_eps && dy <= abs_eps && dz <= abs_eps {
            matched += 1;
        }
    }
    matched as Real / max_len as Real
}

impl GeomProps {
    /// Compute diff between two GeomProps.
    pub fn diff(&self, other: &GeomProps) -> GeomDiff {
        let face_area_score = compare_lists(&self.face_areas, &other.face_areas, 0.001);
        let edge_length_score = compare_lists(&self.edge_lengths, &other.edge_lengths, 0.001);
        let vertex_position_score = compare_positions(&self.vertex_positions, &other.vertex_positions, 1e-4);
        let overall = (face_area_score + edge_length_score + vertex_position_score) / 3.0;
        GeomDiff {
            face_area_score,
            edge_length_score,
            vertex_position_score,
            overall_score: overall,
        }
    }
}

impl GeomDiff {
    pub fn report(&self, step_name: &str) -> String {
        format!(
            "=== Geometry diff for {} ===\n  face_area: {:.3}\n  edge_length: {:.3}\n  vertex_pos: {:.3}\n  overall: {:.3}\n",
            step_name,
            self.face_area_score,
            self.edge_length_score,
            self.vertex_position_score,
            self.overall_score,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topo_counts_parse_brep_text() {
        let brep = "\
DBRep_DrawableShape

Locations 0

Curve3ds 12
...

Surfaces 6
...

Curve2ds 24
...

TVertexes 8
...
TEdges 12
...
TWires 6
...
TFaces 6
...
TShells 1
...
TSolids 1
...

TCompounds 0
...
";
        let counts = TopoCounts::from_brep_text(brep);
        assert_eq!(counts.vertices, 8);
        assert_eq!(counts.edges, 12);
        assert_eq!(counts.wires, 6);
        assert_eq!(counts.faces, 6);
        assert_eq!(counts.shells, 1);
        assert_eq!(counts.solids, 1);
        assert_eq!(counts.compounds, 0);
        assert_eq!(counts.pcurves_total, 24);
    }

    #[test]
    fn topo_counts_identical_score() {
        let a = TopoCounts { vertices: 8, edges: 12, wires: 6, faces: 6, shells: 1, solids: 1, compounds: 0, pcurves_total: 24 };
        let b = a.clone();
        assert_eq!(a.diff_score(&b), 1.0);
    }

    #[test]
    fn topo_counts_different_score() {
        let a = TopoCounts { vertices: 8, edges: 12, wires: 6, faces: 6, shells: 1, solids: 1, compounds: 0, pcurves_total: 24 };
        let b = TopoCounts { vertices: 8, edges: 12, wires: 6, faces: 5, shells: 1, solids: 1, compounds: 0, pcurves_total: 22 };
        assert!(a.diff_score(&b) < 1.0);
    }

    #[test]
    fn compare_lists_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(compare_lists(&a, &b, 0.001), 1.0);
    }

    #[test]
    fn compare_lists_different() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 999.0];
        assert!(compare_lists(&a, &b, 0.001) < 1.0);
    }

    #[test]
    fn compare_positions_identical() {
        let a = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert_eq!(compare_positions(&a, &b, 1e-4), 1.0);
    }

    #[test]
    fn empty_props_score_one() {
        let a = GeomProps::default();
        let b = GeomProps::default();
        let diff = a.diff(&b);
        assert_eq!(diff.overall_score, 1.0);
    }
}
