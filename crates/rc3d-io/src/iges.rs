//! Minimal IGES format reader.
//!
//! IGES (Initial Graphics Exchange Specification) is a fixed-width 80-column
//! text format with five sections: Start (S), Global (G), Directory (D),
//! Parameter (P), and Terminate (T).
//!
//! Supported entity types: 100 (Circular Arc), 110 (Line).

use std::path::Path;

use rc3d_core::math::{PVec3, Real};
use rc3d_shape::geom::curve2d::Curve2d;
use rc3d_shape::geom::CurveGeom;
use rc3d_shape::geom::SurfaceGeom;
use rc3d_shape::store::BRepStore;
use rc3d_shape::topo::Orientation;

#[derive(Debug, thiserror::Error)]
pub enum IgesError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
}

/// A parsed directory entry (two 80-char lines).
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DirEntry {
    entity_type: u32,
    param_ptr: usize,   // 1-based index into parameter data lines
    structure: i32,
    line_font: i32,
    level: i32,
    view: i32,
    transform: i32,
    label: i32,
    status: u32,
    // Second line fields (10-20)
    entity_label: String,
    entity_subscript: i32,
    // Remaining fields mostly unused for basic geometry
}

/// Parsed sections from an IGES file.
struct IgesSections {
    _start: String,
    _global: String,
    directory: Vec<DirEntry>,
    param_lines: Vec<String>,
}

/// Parse an IGES file and build a BRepStore with the contained geometry.
pub fn import_iges(path: &Path) -> Result<BRepStore, IgesError> {
    let content = std::fs::read_to_string(path)?;
    parse_iges_str(&content)
}

/// Parse an IGES string and build a BRepStore.
pub fn parse_iges_str(input: &str) -> Result<BRepStore, IgesError> {
    let sections = split_sections(input)?;
    build_store(&sections)
}

// ---------------------------------------------------------------------------
// Section splitting
// ---------------------------------------------------------------------------

fn split_sections(input: &str) -> Result<IgesSections, IgesError> {
    let lines: Vec<&str> = input.lines().collect();

    if lines.is_empty() {
        return Err(IgesError::Parse("empty input".into()));
    }

    // Validate line lengths (must be exactly 80 chars)
    for (i, line) in lines.iter().enumerate() {
        if line.len() != 80 {
            return Err(IgesError::Parse(format!(
                "line {} has {} columns, expected 80",
                i + 1,
                line.len()
            )));
        }
    }

    let mut s_lines: Vec<&str> = Vec::new();
    let mut g_lines: Vec<&str> = Vec::new();
    let mut d_lines: Vec<&str> = Vec::new();
    let mut p_lines: Vec<&str> = Vec::new();

    for line in &lines {
        let section_char = line.as_bytes().get(72).copied().unwrap_or(b' ');
        match section_char {
            b'S' => s_lines.push(*line),
            b'G' => g_lines.push(*line),
            b'D' => d_lines.push(*line),
            b'P' => p_lines.push(*line),
            b'T' => {} // Terminate — handled separately
            _ => {
                return Err(IgesError::Parse(format!(
                    "unexpected section char '{}' at column 73",
                    section_char as char
                )));
            }
        }
    }

    // Extract parameter data strings (columns 1-64 of each P line, stripped)
    let param_strings: Vec<String> = p_lines
        .iter()
        .map(|l| l[..64].trim_end().to_string())
        .collect();

    // Parse directory entries (two lines per entry)
    let directory = parse_directory(&d_lines)?;

    Ok(IgesSections {
        _start: s_lines.iter().map(|l| l[..72].to_string()).collect::<Vec<_>>().join("\n"),
        _global: g_lines.iter().map(|l| l[..72].to_string()).collect::<Vec<_>>().join("\n"),
        directory,
        param_lines: param_strings,
    })
}

// ---------------------------------------------------------------------------
// Directory entry parsing
// ---------------------------------------------------------------------------

fn parse_directory(d_lines: &[&str]) -> Result<Vec<DirEntry>, IgesError> {
    if d_lines.len() % 2 != 0 {
        return Err(IgesError::Parse(format!(
            "odd number of D lines: {} (expected pairs)",
            d_lines.len()
        )));
    }

    let mut entries = Vec::new();
    let mut i = 0;
    while i < d_lines.len() {
        let line1 = d_lines[i];
        let line2 = d_lines[i + 1];

        let entity_type = parse_i32_field(line1, 0)? as u32;
        let param_ptr = parse_i32_field(line1, 8)? as usize;
        let structure = parse_i32_field(line1, 16)?;
        let line_font = parse_i32_field(line1, 24)?;
        let level = parse_i32_field(line1, 32)?;
        let view = parse_i32_field(line1, 40)?;
        let transform = parse_i32_field(line1, 48)?;
        let label = parse_i32_field(line1, 56)?;
        let status = parse_i32_field(line1, 64)? as u32;

        // Second line fields
        let entity_label = line2[8..16].trim().to_string();
        let entity_subscript = parse_i32_field(line2, 16)?;

        entries.push(DirEntry {
            entity_type,
            param_ptr,
            structure,
            line_font,
            level,
            view,
            transform,
            label,
            status,
            entity_label,
            entity_subscript,
        });
        i += 2;
    }

    Ok(entries)
}

/// Parse an 8-character integer field from the given line at the given column offset.
fn parse_i32_field(line: &str, col: usize) -> Result<i32, IgesError> {
    let field = &line[col..col + 8];
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return Ok(0);
    }
    trimmed
        .parse::<i32>()
        .map_err(|_| IgesError::Parse(format!("invalid integer field '{}' at col {}", field, col + 1)))
}

// ---------------------------------------------------------------------------
// Parameter data parsing
// ---------------------------------------------------------------------------

/// Collect all parameter text for a directory entry, stopping at the record
/// delimiter (semicolon). Parameter records may span multiple lines.
fn collect_param_data(
    param_lines: &[String],
    start_ptr: usize, // 1-based
) -> Result<String, IgesError> {
    if start_ptr == 0 || start_ptr > param_lines.len() {
        return Err(IgesError::Parse(format!(
            "parameter pointer {} out of range (1..{})",
            start_ptr,
            param_lines.len()
        )));
    }

    let mut buf = String::new();
    for idx in (start_ptr - 1)..param_lines.len() {
        let line = &param_lines[idx];
        // Lines are split at column 64; continuation is indicated by the
        // absence of a semicolon. However, some files put the semicolon
        // mid-line or at the end.
        buf.push_str(line);
        if buf.contains(';') {
            break;
        }
    }

    if !buf.contains(';') {
        return Err(IgesError::Parse(format!(
            "no parameter terminator ';' found starting at line {}",
            start_ptr
        )));
    }

    Ok(buf)
}

/// Split parameter string into comma-separated tokens, handling 'D' exponents.
fn split_params(param_str: &str) -> Vec<String> {
    // Remove trailing semicolon
    let s = param_str.trim_end_matches(';');
    s.split(',')
        .map(|t| t.trim().to_string())
        .collect()
}

fn parse_real_token(token: &str, default: Real) -> Real {
    let t = token.trim();
    if t.is_empty() {
        return default;
    }
    t.replace('D', "E").replace('d', "E")
        .parse::<Real>()
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Entity → CurveGeom conversion
// ---------------------------------------------------------------------------

fn params_for_entry(
    sections: &IgesSections,
    entry: &DirEntry,
) -> Result<Vec<String>, IgesError> {
    let data = collect_param_data(&sections.param_lines, entry.param_ptr)?;
    Ok(split_params(&data))
}

fn build_curve_from_entry(
    entry: &DirEntry,
    sections: &IgesSections,
) -> Result<Option<(CurveGeom, PVec3, PVec3)>, IgesError> {
    // Returns (curve, start_point, end_point) or None for unsupported types
    match entry.entity_type {
        110 => {
            // Line: X1,Y1,Z1, X2,Y2,Z2
            let params = params_for_entry(sections, entry)?;
            if params.len() < 6 {
                return Err(IgesError::Parse(format!(
                    "Type 110 (Line) expects 6 parameters, got {}",
                    params.len()
                )));
            }
            let x1 = parse_real_token(&params[0], 0.0);
            let y1 = parse_real_token(&params[1], 0.0);
            let z1 = parse_real_token(&params[2], 0.0);
            let x2 = parse_real_token(&params[3], 0.0);
            let y2 = parse_real_token(&params[4], 0.0);
            let z2 = parse_real_token(&params[5], 0.0);

            let start = PVec3::new(x1, y1, z1);
            let end = PVec3::new(x2, y2, z2);
            let dir = end - start;
            let curve = CurveGeom::Line {
                origin: start,
                direction: dir,
            };
            Ok(Some((curve, start, end)))
        }
        100 => {
            // Circular Arc: ZT, XC,YC, XS,YS, XE,YE
            let params = params_for_entry(sections, entry)?;
            if params.len() < 7 {
                return Err(IgesError::Parse(format!(
                    "Type 100 (Circular Arc) expects 7 parameters, got {}",
                    params.len()
                )));
            }
            let zt = parse_real_token(&params[0], 0.0);
            let xc = parse_real_token(&params[1], 0.0);
            let yc = parse_real_token(&params[2], 0.0);
            let xs = parse_real_token(&params[3], 0.0);
            let ys = parse_real_token(&params[4], 0.0);
            let xe = parse_real_token(&params[5], 0.0);
            let ye = parse_real_token(&params[6], 0.0);

            let center = PVec3::new(xc, yc, zt);
            let start = PVec3::new(xs, ys, zt);
            let end = PVec3::new(xe, ye, zt);

            let r = (start - center).length();
            if r < 1e-12 {
                return Err(IgesError::Parse(
                    "Type 100 (Circular Arc): zero radius".into(),
                ));
            }

            // Axis is Z (arc lies in XY plane at depth ZT)
            let axis = PVec3::Z;
            let curve = CurveGeom::circle(center, axis, r);

            Ok(Some((curve, start, end)))
        }
        _ => Ok(None), // Unsupported type — skip silently
    }
}

// ---------------------------------------------------------------------------
// BRepStore construction
// ---------------------------------------------------------------------------

fn build_store(sections: &IgesSections) -> Result<BRepStore, IgesError> {
    let mut store = BRepStore::new();
    let tolerance: Real = 1e-6;

    // Create a single dummy face (plane) to host all edges.
    let dummy_surface = SurfaceGeom::Plane {
        origin: PVec3::ZERO,
        normal: PVec3::Z,
        u_dir: PVec3::X,
    };
    let face_key = store.add_face(dummy_surface, tolerance);

    for entry in &sections.directory {
        match build_curve_from_entry(entry, sections) {
            Ok(Some((curve, start_pt, end_pt))) => {
                let v_start = store.find_or_add_vertex(start_pt, tolerance);
                let v_end = store.find_or_add_vertex(end_pt, tolerance);

                if v_start == v_end {
                    // Degenerate edge — skip
                    continue;
                }

                // Build a simple 2D PCurve placeholder (line from (0,0) to (1,0))
                let pcurve = Curve2d::Line {
                    origin: (0.0, 0.0),
                    direction: (1.0, 0.0),
                };

                let ek = store.add_edge_with_pcurve(
                    v_start,
                    v_end,
                    curve,
                    tolerance,
                    face_key,
                    pcurve,
                    true, // same_sense
                );

                // Push the edge into the face's outer wire
                if let Some(face) = store.faces.get_mut(face_key) {
                    if let Some(wire) = store.wires.get_mut(face.outer_wire) {
                        wire.edges.push((ek, Orientation::Forward));
                    }
                }
            }
            Ok(None) => {} // Skip unsupported types
            Err(e) => return Err(e),
        }
    }

    Ok(store)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an 80-column IGES string from (data, section, sequence) tuples.
    fn make_iges(lines: &[(&str, &str, &str)]) -> String {
        lines
            .iter()
            .map(|&(data, sec, seq)| format!("{:<72}{}{:>7}", data, sec, seq))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    #[test]
    fn test_parse_minimal_iges() {
        // Two Type 110 lines in one IGES file
        let iges_str = make_iges(&[
            ("TEST IGES FILE - MINIMAL LINE EXAMPLE", "S", "1"),
            ("1H,,1H;,4HSLIN,,,,,,,,,,,,,,,,,1.0,2,2HIN,32767,0.0,15.0,;", "G", "1"),
            ("     110       1       0       0       0       0       000000000", "D", "1"),
            ("     110       0       0       1       0                          ", "D", "2"),
            ("     110       2       0       0       0       0       000000000", "D", "3"),
            ("     110       0       0       1       0                          ", "D", "4"),
            ("0.0,0.0,0.0,10.0,0.0,0.0;", "P", "1"),
            ("0.0,10.0,0.0,10.0,10.0,0.0;", "P", "2"),
            ("S0000001G0000001D0000004P0000002T0000001", "T", "1"),
        ]);

        let store = parse_iges_str(&iges_str).expect("parse should succeed");

        // 2 lines: (0,0,0)→(10,0,0) and (0,10,0)→(10,10,0) = 4 distinct vertices
        assert_eq!(store.vertices.len(), 4);
        assert_eq!(store.edges.len(), 2);
        assert_eq!(store.faces.len(), 1);
    }

    #[test]
    fn test_parse_arc() {
        let iges_str = make_iges(&[
            ("TEST IGES FILE - ARC EXAMPLE", "S", "1"),
            ("1H,,1H;,4HSLIN,,,,,,,,,,,,,,,,,1.0,2,2HIN,32767,0.0,15.0,;", "G", "1"),
            ("     100       1       0       0       0       0       000000000", "D", "1"),
            ("     100       0       0       1       0                          ", "D", "2"),
            ("0.0,0.0,0.0,10.0,0.0,0.0,0.0,10.0,0.0;", "P", "1"),
            ("S0000001G0000001D0000002P0000001T0000001", "T", "1"),
        ]);

        let store = parse_iges_str(&iges_str).expect("parse should succeed");
        assert_eq!(store.edges.len(), 1);
        // Arc from (10,0,0) to (0,10,0) around center (0,0,0) — 2 distinct vertices
        assert_eq!(store.vertices.len(), 2);
    }

    #[test]
    fn test_empty_file() {
        let result = parse_iges_str("");
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_line_length() {
        let result = parse_iges_str("too short\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_type_skipped() {
        let iges_str = make_iges(&[
            ("TEST IGES - UNSUPPORTED TYPE", "S", "1"),
            ("1H,,1H;,4HSLIN,,,,,,,,,,,,,,,,,1.0,2,2HIN,32767,0.0,15.0,;", "G", "1"),
            ("     999       1       0       0       0       0       000000000", "D", "1"),
            ("     999       0       0       1       0                          ", "D", "2"),
            ("1.0,2.0,3.0,4.0;", "P", "1"),
            ("S0000001G0000001D0000002P0000001T0000001", "T", "1"),
        ]);

        let store = parse_iges_str(&iges_str).expect("parse should succeed");
        assert_eq!(store.edges.len(), 0);
    }
}
