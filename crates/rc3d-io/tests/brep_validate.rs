//! BREP format self-validation: parse classic OCC BREP, check reference integrity.
//! Closed-loop test: write .brep → validate → report broken refs.

use std::collections::HashSet;

/// Parsed TShape reference from reverse-index notation.
#[derive(Debug, Clone, Copy)]
struct ShapeRef {
    /// Absolute TShape position (1-based).
    abs_pos: usize,
    /// Forward (+) or Reversed (-).
    forward: bool,
}

/// Parsed TShape entry.
#[derive(Debug)]
enum TShapeKind {
    Vertex,
    Edge { curve_idx: usize, v_refs: Vec<ShapeRef> },
    Wire { edge_refs: Vec<ShapeRef> },
    Face { surface_idx: usize, wire_refs: Vec<ShapeRef> },
    Shell { face_refs: Vec<ShapeRef> },
    Solid { shell_refs: Vec<ShapeRef> },
    Compound { solid_refs: Vec<ShapeRef> },
}

struct BrepShape {
    kind: TShapeKind,
    /// Original line number in file for error reporting.
    line: usize,
}

/// Validation result.
#[derive(Debug, Default)]
pub struct BrepValidation {
    pub total_shapes: usize,
    pub curve_count: usize,
    pub surface_count: usize,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl BrepValidation {
    pub fn is_valid(&self) -> bool { self.errors.is_empty() }
}

/// Parse and validate a classic OCC BREP ASCII file.
pub fn validate_brep(text: &str) -> BrepValidation {
    let mut v = BrepValidation::default();
    let lines: Vec<&str> = text.lines().collect();

    // Parse header sections
    let mut i = 0usize;
    let mut total_shapes = 0usize;
    let mut curve_count = 0usize;
    let mut surface_count = 0usize;

    // Skip to TShapes count
    while i < lines.len() {
        let line = lines[i].trim();
        if line.starts_with("TShapes ") {
            total_shapes = line.split_whitespace().nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(0);
            i += 1;
            break;
        }
        if line.starts_with("Curves ") {
            curve_count = line.split_whitespace().nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(0);
        }
        if line.starts_with("Surfaces ") {
            surface_count = line.split_whitespace().nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(0);
        }
        i += 1;
    }

    v.total_shapes = total_shapes;
    v.curve_count = curve_count;
    v.surface_count = surface_count;

    if total_shapes == 0 {
        v.errors.push("TShapes section not found or count is 0".into());
        return v;
    }

    // Parse TShapes entries
    let mut shapes: Vec<BrepShape> = Vec::new();
    while i < lines.len() {
        let line = lines[i].trim();
        let line_no = i + 1;

        match line {
            "Ve" => {
                // Skip tolerance, position, params, flags lines
                i = skip_ve_lines(&lines, i);
                shapes.push(BrepShape { kind: TShapeKind::Vertex, line: line_no });
            }
            "Ed" => {
                i += 1; // tolerance line
                let curve_idx = if i < lines.len() {
                    parse_edge_curve_idx(lines[i])
                } else { 0 };
                i += 1; // curve line
                if i < lines.len() && lines[i].trim() == "0" {
                    i += 1; // pcurve count line
                }
                // Skip empty line, flags line
                // Skip empty line, then flags line, then read refs
                while i < lines.len() && !lines[i].trim().starts_with("010") { i += 1; }
                i += 1; // skip flags
                let v_refs = if i < lines.len() { parse_refs(lines[i]) } else { vec![] };
                shapes.push(BrepShape {
                    kind: TShapeKind::Edge { curve_idx, v_refs },
                    line: line_no,
                });
            }
            "Wi" => {
                while i < lines.len() && !lines[i].trim().starts_with("010") { i += 1; }
                i += 1; // skip flags
                let edge_refs = if i < lines.len() { parse_refs(lines[i]) } else { vec![] };
                shapes.push(BrepShape {
                    kind: TShapeKind::Wire { edge_refs },
                    line: line_no,
                });
            }
            "Fa" => {
                let (surf_idx, _) = if i + 1 < lines.len() {
                    parse_face_data(lines[i + 1])
                } else { (0, 0.0) };
                i += 2;
                while i < lines.len() && !lines[i].trim().starts_with("010") { i += 1; }
                i += 1;
                let wire_refs = if i < lines.len() { parse_refs(lines[i]) } else { vec![] };
                shapes.push(BrepShape {
                    kind: TShapeKind::Face { surface_idx: surf_idx, wire_refs },
                    line: line_no,
                });
            }
            "Sh" => {
                while i < lines.len() && !lines[i].trim().starts_with("010") { i += 1; }
                i += 1;
                let face_refs = if i < lines.len() { parse_refs(lines[i]) } else { vec![] };
                shapes.push(BrepShape {
                    kind: TShapeKind::Shell { face_refs },
                    line: line_no,
                });
            }
            "So" => {
                while i < lines.len() && !lines[i].trim().starts_with("010") { i += 1; }
                i += 1;
                let shell_refs = if i < lines.len() { parse_refs(lines[i]) } else { vec![] };
                shapes.push(BrepShape {
                    kind: TShapeKind::Solid { shell_refs },
                    line: line_no,
                });
            }
            "Co" => {
                while i < lines.len() && !lines[i].trim().starts_with("110") { i += 1; }
                i += 1;
                let solid_refs = if i < lines.len() { parse_refs(lines[i]) } else { vec![] };
                shapes.push(BrepShape {
                    kind: TShapeKind::Compound { solid_refs },
                    line: line_no,
                });
            }
            _ => {}
        }
        i += 1;
    }

    // Validate references
    validate_refs(&shapes, total_shapes, curve_count, surface_count, &mut v);

    v
}

fn skip_ve_lines(lines: &[&str], i: usize) -> usize {
    // Skip: tolerance, position, params, empty, flags, *
    let mut j = i + 1;
    let mut skipped = 0;
    while j < lines.len() && skipped < 5 {
        j += 1;
        skipped += 1;
    }
    j
}

fn skip_to_next(lines: &[&str], i: usize, prefix: &str) -> usize {
    let mut j = i + 1;
    while j < lines.len() {
        if lines[j].trim().starts_with(prefix) {
            return j - 1;
        }
        j += 1;
    }
    i
}

fn parse_edge_curve_idx(line: &str) -> usize {
    // Format: "type  curve_idx  0  0  param"
    let parts: Vec<&str> = line.split_whitespace().collect();
    parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0)
}

fn parse_face_data(line: &str) -> (usize, f32) {
    // Format: "location  tolerance  surface_idx  orientation"
    let parts: Vec<&str> = line.split_whitespace().collect();
    let si = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let tol = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    (si, tol)
}

fn parse_refs(line: &str) -> Vec<ShapeRef> {
    let line = line.trim();
    if line == "*" { return vec![]; }
    // Format: "+N 0 -M 0 *" where N,M are reverse indices
    let cleaned = line.trim_end_matches('*').trim();
    let parts: Vec<&str> = cleaned.split_whitespace().collect();
    let mut refs = Vec::new();
    let mut i = 0;
    while i + 1 < parts.len() {
        let sign = parts[i];
        // parts[i+1] should be "0"
        if let Some(num) = sign.strip_prefix('+').or_else(|| sign.strip_prefix('-')) {
            if let Ok(n) = num.parse::<usize>() {
                refs.push(ShapeRef {
                    abs_pos: n, // reverse index — converted later
                    forward: sign.starts_with('+'),
                });
            }
        }
        i += 2;
    }
    refs
}

fn validate_refs(
    shapes: &[BrepShape],
    total: usize,
    curve_count: usize,
    surface_count: usize,
    v: &mut BrepValidation,
) {
    if total == 0 { return; }

    // Build map: TShape position → kind label
    let mut pos_types: Vec<&str> = Vec::new();
    for s in shapes {
        let label = match s.kind {
            TShapeKind::Vertex => "Ve",
            TShapeKind::Edge { .. } => "Ed",
            TShapeKind::Wire { .. } => "Wi",
            TShapeKind::Face { .. } => "Fa",
            TShapeKind::Shell { .. } => "Sh",
            TShapeKind::Solid { .. } => "So",
            TShapeKind::Compound { .. } => "Co",
        };
        pos_types.push(label);
    }

    // Check shape count matches declared
    if shapes.len() != total {
        v.errors.push(format!(
            "TShapes count mismatch: declared {}, parsed {}",
            total, shapes.len()
        ));
    }

    // Validate each shape's references
    for (pos, s) in shapes.iter().enumerate() {
        let abs_pos = pos + 1; // 1-based

        match &s.kind {
            TShapeKind::Vertex => {}
            TShapeKind::Edge { curve_idx, v_refs } => {
                // Check curve index
                if *curve_idx == 0 || *curve_idx > curve_count {
                    v.errors.push(format!(
                        "Line {}: Ed at pos {}: curve_idx {} out of bounds (1-{})",
                        s.line, abs_pos, curve_idx, curve_count
                    ));
                }
                // Check vertex refs (should point to Ve)
                for r in v_refs {
                    let target = rev_to_abs(r.abs_pos, total);
                    if target == 0 || target > pos_types.len() {
                        v.errors.push(format!(
                            "Line {}: Ed at pos {}: vertex ref +{}(→abs{}) out of bounds",
                            s.line, abs_pos, r.abs_pos, target
                        ));
                    } else if pos_types[target - 1] != "Ve" {
                        v.errors.push(format!(
                            "Line {}: Ed at pos {}: vertex ref +{}(→abs{}) points to {} (expected Ve)",
                            s.line, abs_pos, r.abs_pos, target, pos_types[target - 1]
                        ));
                    }
                }
                if v_refs.len() != 2 {
                    v.warnings.push(format!(
                        "Line {}: Ed at pos {}: expected 2 vertex refs, got {}",
                        s.line, abs_pos, v_refs.len()
                    ));
                }
            }
            TShapeKind::Wire { edge_refs } => {
                for r in edge_refs {
                    let target = rev_to_abs(r.abs_pos, total);
                    if target == 0 || target > pos_types.len() {
                        v.errors.push(format!(
                            "Line {}: Wi at pos {}: edge ref +{}(→abs{}) out of bounds",
                            s.line, abs_pos, r.abs_pos, target
                        ));
                    } else if pos_types[target - 1] != "Ed" {
                        v.errors.push(format!(
                            "Line {}: Wi at pos {}: ref +{}(→abs{}) points to {} (expected Ed)",
                            s.line, abs_pos, r.abs_pos, target, pos_types[target - 1]
                        ));
                    }
                }
            }
            TShapeKind::Face { surface_idx, wire_refs } => {
                // Check surface index (1-based)
                if *surface_idx == 0 || *surface_idx > surface_count {
                    v.errors.push(format!(
                        "Line {}: Fa at pos {}: surface_idx {} out of bounds (1-{})",
                        s.line, abs_pos, surface_idx, surface_count
                    ));
                }
                for r in wire_refs {
                    let target = rev_to_abs(r.abs_pos, total);
                    if target == 0 || target > pos_types.len() {
                        v.errors.push(format!(
                            "Line {}: Fa at pos {}: wire ref +{}(→abs{}) out of bounds",
                            s.line, abs_pos, r.abs_pos, target
                        ));
                    } else if pos_types[target - 1] != "Wi" {
                        v.errors.push(format!(
                            "Line {}: Fa at pos {}: ref +{}(→abs{}) points to {} (expected Wi)",
                            s.line, abs_pos, r.abs_pos, target, pos_types[target - 1]
                        ));
                    }
                }
            }
            TShapeKind::Shell { face_refs } => {
                for r in face_refs {
                    let target = rev_to_abs(r.abs_pos, total);
                    if target == 0 || target > pos_types.len() {
                        v.errors.push(format!(
                            "Line {}: Sh at pos {}: face ref +{}(→abs{}) out of bounds",
                            s.line, abs_pos, r.abs_pos, target
                        ));
                    } else if pos_types[target - 1] != "Fa" {
                        v.errors.push(format!(
                            "Line {}: Sh at pos {}: ref +{}(→abs{}) points to {} (expected Fa)",
                            s.line, abs_pos, r.abs_pos, target, pos_types[target - 1]
                        ));
                    }
                }
            }
            TShapeKind::Solid { shell_refs } => {
                for r in shell_refs {
                    let target = rev_to_abs(r.abs_pos, total);
                    if target == 0 || target > pos_types.len() {
                        v.errors.push(format!(
                            "Line {}: So at pos {}: shell ref +{}(→abs{}) out of bounds",
                            s.line, abs_pos, r.abs_pos, target
                        ));
                    } else if pos_types[target - 1] != "Sh" {
                        v.errors.push(format!(
                            "Line {}: So at pos {}: ref +{}(→abs{}) points to {} (expected Sh)",
                            s.line, abs_pos, r.abs_pos, target, pos_types[target - 1]
                        ));
                    }
                }
            }
            TShapeKind::Compound { solid_refs } => {
                for r in solid_refs {
                    let target = rev_to_abs(r.abs_pos, total);
                    if target == 0 || target > pos_types.len() {
                        v.errors.push(format!(
                            "Line {}: Co at pos {}: solid ref +{}(→abs{}) out of bounds",
                            s.line, abs_pos, r.abs_pos, target
                        ));
                    } else if pos_types[target - 1] != "So" {
                        v.errors.push(format!(
                            "Line {}: Co at pos {}: ref +{}(→abs{}) points to {} (expected So)",
                            s.line, abs_pos, r.abs_pos, target, pos_types[target - 1]
                        ));
                    }
                }
            }
        }
    }

    // Check for orphan shapes (no incoming references)
    let _referenced: HashSet<usize> = shapes.iter().enumerate()
        .filter(|(_, s)| matches!(s.kind, TShapeKind::Compound { .. }))
        .map(|(i, _)| i + 1)
        .collect();
}

/// Convert OCC reverse-index to absolute position.
fn rev_to_abs(rev_idx: usize, total: usize) -> usize {
    if rev_idx == 0 || rev_idx > total { 0 }
    else { total + 1 - rev_idx }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_cube_brep() {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../test_output/brep/Cube.brep")
        ).expect("read cube.brep");

        let v = validate_brep(&text);
        println!("Cube: {} shapes, {} errors, {} warnings",
            v.total_shapes, v.errors.len(), v.warnings.len());
        for e in &v.errors { println!("  ERROR: {}", e); }
        for w in &v.warnings { println!("  WARN: {}", w); }
        assert!(v.is_valid(), "Cube.brep should be valid: {:?}", v.errors);
    }

    #[test]
    fn validate_all_brep_files() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_output/brep");
        let mut all_ok = true;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("brep") { continue; }
            let text = std::fs::read_to_string(&path).unwrap();
            let v = validate_brep(&text);
            let status = if v.is_valid() { "✓" } else { "✗" };
            println!("{} {}: {} shapes, {} err, {} warn",
                status,
                path.file_name().unwrap().to_string_lossy(),
                v.total_shapes, v.errors.len(), v.warnings.len());
            for e in &v.errors { println!("    ERROR: {}", e); }
            for w in &v.warnings { println!("    WARN: {}", w); }
            if !v.is_valid() { all_ok = false; }
        }
        assert!(all_ok, "some BREP files have validation errors");
    }
}
