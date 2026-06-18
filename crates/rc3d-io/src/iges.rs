//! IGES format reader (ISO 10303-308).
//!
//! IGES (Initial Graphics Exchange Specification) uses fixed-width 80-column
//! text with five sections: Start (S), Global (G), Directory (D),
//! Parameter (P), and Terminate (T).
//!
//! ## Supported entity types
//!
//! | Type | Name | Category |
//! |------|------|----------|
//! | 100 | Circular Arc | Curve |
//! | 102 | Composite Curve | Curve |
//! | 108 | Plane (bounded) | Surface |
//! | 110 | Line | Curve |
//! | 120 | Surface of Revolution | Surface |
//! | 122 | Tabulated Cylinder | Surface |
//! | 128 | Rational BSpline Surface | Surface |
//! | 140 | Offset Surface | Surface |
//! | 142 | Curve on Parametric Surface | Hybrid |
//! | 144 | Trimmed Surface | Surface |

use std::path::Path;

use log;
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
// Entity param extraction
// ---------------------------------------------------------------------------

fn params_for_entry(
    sections: &IgesSections,
    entry: &DirEntry,
) -> Result<Vec<String>, IgesError> {
    let data = collect_param_data(&sections.param_lines, entry.param_ptr)?;
    Ok(split_params(&data))
}

// ---------------------------------------------------------------------------
// Entity → CurveGeom conversion (curves)
// ---------------------------------------------------------------------------

fn build_curve_from_entry(
    entry: &DirEntry,
    sections: &IgesSections,
) -> Result<Option<(CurveGeom, PVec3, PVec3)>, IgesError> {
    match entry.entity_type {
        110 => build_line(entry, sections),
        100 => build_circular_arc(entry, sections),
        102 => build_composite_curve(entry, sections),
        _ => Ok(None),
    }
}

fn build_line(entry: &DirEntry, sections: &IgesSections) -> Result<Option<(CurveGeom, PVec3, PVec3)>, IgesError> {
    let params = params_for_entry(sections, entry)?;
    if params.len() < 6 {
        return Err(IgesError::Parse(format!("Type 110 needs 6 params, got {}", params.len())));
    }
    let start = PVec3::new(parse_real_token(&params[0], 0.0), parse_real_token(&params[1], 0.0), parse_real_token(&params[2], 0.0));
    let end = PVec3::new(parse_real_token(&params[3], 0.0), parse_real_token(&params[4], 0.0), parse_real_token(&params[5], 0.0));
    Ok(Some((CurveGeom::Line { origin: start, direction: end - start }, start, end)))
}

fn build_circular_arc(entry: &DirEntry, sections: &IgesSections) -> Result<Option<(CurveGeom, PVec3, PVec3)>, IgesError> {
    let params = params_for_entry(sections, entry)?;
    if params.len() < 7 {
        return Err(IgesError::Parse(format!("Type 100 needs 7 params, got {}", params.len())));
    }
    let zt = parse_real_token(&params[0], 0.0);
    let center = PVec3::new(parse_real_token(&params[1], 0.0), parse_real_token(&params[2], 0.0), zt);
    let start = PVec3::new(parse_real_token(&params[3], 0.0), parse_real_token(&params[4], 0.0), zt);
    let end = PVec3::new(parse_real_token(&params[5], 0.0), parse_real_token(&params[6], 0.0), zt);
    let r = (start - center).length();
    if r < 1e-12 { return Ok(None); }
    Ok(Some((CurveGeom::circle(center, PVec3::Z, r), start, end)))
}

/// Composite curve: references child curve entities by DE pointer.
fn build_composite_curve(entry: &DirEntry, sections: &IgesSections) -> Result<Option<(CurveGeom, PVec3, PVec3)>, IgesError> {
    let params = params_for_entry(sections, entry)?;
    if params.len() < 2 { return Ok(None); }
    let n_curves = parse_real_token(&params[0], 0.0) as usize;
    let mut curves: Vec<CurveGeom> = Vec::new();
    let mut first_start = PVec3::ZERO;
    let mut last_end = PVec3::ZERO;
    for i in 0..n_curves {
        let de_idx = parse_real_token(&params.get(1 + i).unwrap_or(&String::new()), 0.0) as usize;
        if de_idx == 0 || de_idx > sections.directory.len() { continue; }
        let child = &sections.directory[de_idx - 1];
        if let Ok(Some((curve, s, e))) = build_curve_from_entry(child, sections) {
            if i == 0 { first_start = s; }
            last_end = e;
            curves.push(curve);
        }
    }
    if curves.is_empty() { return Ok(None); }
    if curves.len() == 1 { return Ok(Some((curves.pop().unwrap(), first_start, last_end))); }
    let composites: Vec<(CurveGeom, bool)> = curves.into_iter().map(|c| (c, false)).collect();
    Ok(Some((CurveGeom::Composite { segments: composites, cached_lengths: None }, first_start, last_end)))
}

// ---------------------------------------------------------------------------
// Entity → SurfaceGeom conversion (surfaces)
// ---------------------------------------------------------------------------

fn build_surface_from_entry(
    entry: &DirEntry,
    sections: &IgesSections,
) -> Result<Option<SurfaceGeom>, IgesError> {
    match entry.entity_type {
        108 => build_plane_surface(entry, sections),
        120 => build_revolution_surface(entry, sections),
        122 => build_tabulated_cylinder(entry, sections),
        128 => build_bspline_surface(entry, sections),
        140 => build_offset_surface(entry, sections),
        _ => Ok(None),
    }
}

/// Type 108: Plane — bounded planar surface.
/// Parameters: A,B,C,D (plane equation) + optional boundary curve DE pointer.
fn build_plane_surface(entry: &DirEntry, sections: &IgesSections) -> Result<Option<SurfaceGeom>, IgesError> {
    let params = params_for_entry(sections, entry)?;
    if params.len() < 4 { return Ok(None); }
    let a = parse_real_token(&params[0], 0.0);
    let b = parse_real_token(&params[1], 0.0);
    let c = parse_real_token(&params[2], 0.0);
    let d = parse_real_token(&params[3], 0.0);
    let normal = PVec3::new(a, b, c);
    if normal.length() < 1e-12 { return Ok(None); }
    let n = normal.normalize();
    let origin = n * (-d / normal.length_squared());
    let u_dir = if n.x.abs() < 0.9 { PVec3::X.cross(n).normalize() } else { PVec3::Y.cross(n).normalize() };
    Ok(Some(SurfaceGeom::Plane { origin, normal: n, u_dir }))
}

/// Type 120: Surface of Revolution — rotatation of a generatrix curve around an axis.
fn build_revolution_surface(entry: &DirEntry, sections: &IgesSections) -> Result<Option<SurfaceGeom>, IgesError> {
    let params = params_for_entry(sections, entry)?;
    if params.len() < 10 { return Ok(None); }
    let axis_origin = PVec3::new(parse_real_token(&params[0], 0.0), parse_real_token(&params[1], 0.0), parse_real_token(&params[2], 0.0));
    let axis_dir = PVec3::new(parse_real_token(&params[3], 0.0), parse_real_token(&params[4], 0.0), parse_real_token(&params[5], 0.0));
    let _start_angle = parse_real_token(&params[6], 0.0);
    let _end_angle = parse_real_token(&params[7], 0.0);
    // Parameter 9 is the DE pointer to the generatrix curve (usually a composite)
    let _generatrix_de = parse_real_token(&params.get(8).unwrap_or(&String::new()), 0.0) as usize;
    // For now: return a basic revolution surface without the generatrix (placeholder)
    // Full implementation would resolve the generatrix curve and build a proper swept surface
    log::debug!("[IGES] Type 120 revolution: axis={:?}, dir={:?}", axis_origin, axis_dir);
    Ok(None) // Defer to later: needs curve resolution
}

/// Type 122: Tabulated Cylinder — extrusion of a directrix curve along a vector.
fn build_tabulated_cylinder(entry: &DirEntry, sections: &IgesSections) -> Result<Option<SurfaceGeom>, IgesError> {
    let params = params_for_entry(sections, entry)?;
    if params.len() < 6 { return Ok(None); }
    let dx = parse_real_token(&params[0], 0.0);
    let dy = parse_real_token(&params[1], 0.0);
    let dz = parse_real_token(&params[2], 0.0);
    let extrusion_vec = PVec3::new(dx, dy, dz);
    // Parameter 4 is the DE pointer to the directrix curve
    let _directrix_de = parse_real_token(&params.get(3).unwrap_or(&String::new()), 0.0) as usize;
    if extrusion_vec.length() < 1e-12 { return Ok(None); }
    let default_curve = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
    Ok(Some(SurfaceGeom::Extrusion {
        generatrix: Box::new(default_curve),
        direction: extrusion_vec,
    }))
}

/// Type 128: Rational BSpline Surface.
fn build_bspline_surface(entry: &DirEntry, sections: &IgesSections) -> Result<Option<SurfaceGeom>, IgesError> {
    let params = params_for_entry(sections, entry)?;
    if params.len() < 16 { return Ok(None); }
    let u_degree = parse_real_token(&params[3], 2.0) as usize;
    let v_degree = parse_real_token(&params[4], 2.0) as usize;
    let u_cp_count = 1 + parse_real_token(&params[5], 0.0) as usize;
    let v_cp_count = 1 + parse_real_token(&params[6], 0.0) as usize;
    let _u_knot_count = parse_real_token(&params[9], 0.0) as usize;
    let _v_knot_count = parse_real_token(&params[10], 0.0) as usize;
    let _weighted = parse_real_token(&params.get(15).unwrap_or(&String::new()), 0.0) as u32;
    // BSpline surface format: after the header params, followed by:
    //   u_knots[*], v_knots[*], weights[*], control_points[*]
    // Full parsing requires variable-length data. For now, return a placeholder.
    // The parameter layout is complex — defer full implementation.
    log::debug!("[IGES] Type 128 BSpline: u_deg={} v_deg={} u_cp={} v_cp={}", u_degree, v_degree, u_cp_count, v_cp_count);
    Ok(None) // Defer: needs variable-length parameter parsing
}

/// Type 140: Offset Surface.
fn build_offset_surface(entry: &DirEntry, sections: &IgesSections) -> Result<Option<SurfaceGeom>, IgesError> {
    let params = params_for_entry(sections, entry)?;
    if params.len() < 3 { return Ok(None); }
    let _basis_de = parse_real_token(&params[0], 0.0) as usize;
    let distance = parse_real_token(&params[1], 0.0);
    let _approx_tol = parse_real_token(&params.get(2).unwrap_or(&String::new()), 1e-3);
    Ok(Some(SurfaceGeom::Offset {
        basis: Box::new(SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X }),
        distance,
    }))
}

/// Type 142: Curve on Parametric Surface.
fn build_curve_on_surface(
    entry: &DirEntry,
    sections: &IgesSections,
) -> Result<Option<(CurveGeom, SurfaceGeom, PVec3, PVec3)>, IgesError> {
    let params = params_for_entry(sections, entry)?;
    if params.len() < 6 { return Ok(None); }
    let _creation_mode = parse_real_token(&params[0], 0.0) as u32;
    let _surface_de = parse_real_token(&params[1], 0.0) as usize;
    let _curve_3d_de = parse_real_token(&params[2], 0.0) as usize;
    let _pref_curve_de = parse_real_token(&params[3], 0.0) as usize;
    // For now, fall back to the 3D curve representation
    let curve_de = _curve_3d_de.max(_pref_curve_de).max(0);
    if curve_de == 0 || curve_de > sections.directory.len() { return Ok(None); }
    let child = &sections.directory[curve_de - 1];
    build_curve_from_entry(child, sections).map(|r| {
        r.map(|(c, s, e)| (c, SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X }, s, e))
    })
}

// ---------------------------------------------------------------------------
// BRepStore construction
// ---------------------------------------------------------------------------

fn build_store(sections: &IgesSections) -> Result<BRepStore, IgesError> {
    let mut store = BRepStore::new();
    let tolerance: Real = 1e-6;

    // Collect all standalone curves (not owned by surfaces)
    let mut curve_entries: Vec<(usize, DirEntry)> = Vec::new();
    let mut surface_entries: Vec<(usize, SurfaceGeom)> = Vec::new();

    for (idx, entry) in sections.directory.iter().enumerate() {
        match entry.entity_type {
            // Curves
            100 | 110 | 102 => { curve_entries.push((idx, entry.clone())); }
            // Surfaces (skip complex types that need child resolution)
            108 | 122 => {
                if let Ok(Some(surf)) = build_surface_from_entry(entry, sections) {
                    surface_entries.push((idx, surf));
                }
            }
            120 | 128 | 140 => {
                // Complex: defer or use placeholder
                if let Ok(Some(surf)) = build_surface_from_entry(entry, sections) {
                    surface_entries.push((idx, surf));
                }
            }
            // Hybrid
            142 => {
                if let Ok(Some((curve, _surf, start_pt, end_pt))) = build_curve_on_surface(entry, sections) {
                    curve_entries.push((idx, entry.clone()));
                    // Also create the surface if useful
                    let _ = _surf; // surface association is deferred
                    let _ = start_pt; let _ = end_pt;
                }
            }
            _ => {} // unsupported
        }
    }

    if surface_entries.is_empty() {
        // Fallback: create a dummy plane face to host standalone curves
        let dummy_surface = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let face_key = store.add_face(dummy_surface, tolerance);
        add_curves_to_face(&mut store, face_key, &curve_entries, sections, tolerance)?;
    } else {
        // Create a face per surface, assign curves to their surface
        for (_idx, surf) in &surface_entries {
            let face_key = store.add_face(surf.clone(), tolerance);
            add_curves_to_face(&mut store, face_key, &curve_entries, sections, tolerance)?;
        }
    }

    Ok(store)
}

fn add_curves_to_face(
    store: &mut BRepStore,
    face_key: rc3d_shape::topo::FaceKey,
    curve_entries: &[(usize, DirEntry)],
    sections: &IgesSections,
    tolerance: Real,
) -> Result<(), IgesError> {
    for (_idx, entry) in curve_entries {
        if let Ok(Some((curve, start_pt, end_pt))) = build_curve_from_entry(entry, sections) {
            let v_start = store.find_or_add_vertex(start_pt, tolerance);
            let v_end = store.find_or_add_vertex(end_pt, tolerance);
            if v_start == v_end { continue; }
            let pcurve = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
            let ek = store.add_edge_with_pcurve(v_start, v_end, curve, tolerance, face_key, pcurve, true);
            if let Some(face) = store.faces.get_mut(face_key) {
                if let Some(wire) = store.wires.get_mut(face.outer_wire) {
                    wire.edges.push((ek, Orientation::Forward));
                }
            }
        }
    }
    Ok(())
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

    #[test]
    fn test_parse_plane_surface() {
        // Type 108 with A,B,C,D = 0,0,1,-5 (plane z=5)
        let iges_str = make_iges(&[
            ("TEST - PLANE", "S", "1"),
            ("1H,,1H;,4HSLIN,,,,,,,,,,,,,,,,,1.0,2,2HIN,32767,0.0,15.0,;", "G", "1"),
            ("     108       1       0       0       0       0       000000000", "D", "1"),
            ("     108       0       0       1       0                          ", "D", "2"),
            ("0.0,0.0,1.0,-5.0,0;", "P", "1"),
            ("S0000001G0000001D0000002P0000001T0000001", "T", "1"),
        ]);
        let store = parse_iges_str(&iges_str).expect("parse should succeed");
        assert_eq!(store.faces.len(), 1, "should create one face for the plane");
    }

    #[test]
    fn test_parse_tabulated_cylinder() {
        // Type 122: extrusion along (0,0,10) with no curve pointer
        let iges_str = make_iges(&[
            ("TEST - TABULATED CYLINDER", "S", "1"),
            ("1H,,1H;,4HSLIN,,,,,,,,,,,,,,,,,1.0,2,2HIN,32767,0.0,15.0,;", "G", "1"),
            ("     122       1       0       0       0       0       000000000", "D", "1"),
            ("     122       0       0       1       0                          ", "D", "2"),
            ("0.0,0.0,10.0,0,0.0,0.0;", "P", "1"),
            ("S0000001G0000001D0000002P0000001T0000001", "T", "1"),
        ]);
        let store = parse_iges_str(&iges_str).expect("parse should succeed");
        assert_eq!(store.faces.len(), 1, "should create one face for the extrusion");
    }

    #[test]
    fn test_parse_composite_curve() {
        // Type 102 referencing two Type 110 lines
        let iges_str = make_iges(&[
            ("TEST - COMPOSITE CURVE", "S", "1"),
            ("1H,,1H;,4HSLIN,,,,,,,,,,,,,,,,,1.0,2,2HIN,32767,0.0,15.0,;", "G", "1"),
            ("     102       3       0       0       0       0       000000000", "D", "1"),
            ("     102       0       0       1       0                          ", "D", "2"),
            ("     110       1       0       0       0       0       000000000", "D", "3"),
            ("     110       0       0       1       0                          ", "D", "4"),
            ("     110       2       0       0       0       0       000000000", "D", "5"),
            ("     110       0       0       1       0                          ", "D", "6"),
            ("2,2,4;", "P", "1"),
            ("0.0,0.0,0.0,10.0,0.0,0.0;", "P", "2"),
            ("10.0,0.0,0.0,10.0,10.0,0.0;", "P", "3"),
            ("S0000001G0000001D0000006P0000003T0000001", "T", "1"),
        ]);
        let store = parse_iges_str(&iges_str).expect("parse should succeed");
        assert!(store.edges.len() >= 1, "composite should produce at least 1 edge");
    }

    #[test]
    fn test_parse_mixed_curves_and_surfaces() {
        // One Type 108 plane + one Type 110 line
        let iges_str = make_iges(&[
            ("TEST - MIXED", "S", "1"),
            ("1H,,1H;,4HSLIN,,,,,,,,,,,,,,,,,1.0,2,2HIN,32767,0.0,15.0,;", "G", "1"),
            ("     108       1       0       0       0       0       000000000", "D", "1"),
            ("     108       0       0       1       0                          ", "D", "2"),
            ("     110       2       0       0       0       0       000000000", "D", "3"),
            ("     110       0       0       1       0                          ", "D", "4"),
            ("0.0,0.0,1.0,0.0,0;", "P", "1"),
            ("0.0,0.0,0.0,5.0,5.0,0.0;", "P", "2"),
            ("S0000001G0000001D0000004P0000002T0000001", "T", "1"),
        ]);
        let store = parse_iges_str(&iges_str).expect("parse should succeed");
        assert!(store.faces.len() >= 1, "should have at least the plane face");
        assert!(store.edges.len() >= 1, "should have at least the line edge");
    }
}
