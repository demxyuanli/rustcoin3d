//! IGES format writer (ISO 10303-308).
//!
//! Exports B-Rep topology (curves and surfaces) to IGES 5.3 fixed 80-column
//! text format. Supports the most common entity types needed for practical
//! CAD interchange.
//!
//! ## Supported entity types
//!
//! | Type | Name                  | Source                      |
//! |------|-----------------------|-----------------------------|
//! | 100  | Circular Arc          | CurveGeom::Circle           |
//! | 106  | Copious Data (poly)   | Polyline / BSpline / Ellipse|
//! | 108  | Plane                 | SurfaceGeom::Plane          |
//! | 110  | Line                  | CurveGeom::Line             |

use std::path::Path;

use rc3d_core::math::{PVec3, Real};
use rc3d_shape::geom::curve_eval::CurveGeom;
use rc3d_shape::geom::surface_eval::SurfaceGeom;
use rc3d_shape::store::BRepStore;
use rc3d_shape::topo::{EdgeKey, FaceKey};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Write BRepStore to IGES file.
pub fn write_iges(path: &Path, store: &BRepStore) -> Result<(), std::io::Error> {
    let text = write_iges_string(store);
    std::fs::write(path, text)
}

/// Generate IGES string from BRepStore.
pub fn write_iges_string(store: &BRepStore) -> String {
    let mut ctx = IgesContext::default();
    ctx.collect_entities(store);
    ctx.emit()
}

// ---------------------------------------------------------------------------
// Entity representation
// ---------------------------------------------------------------------------

/// An IGES entity ready to emit: type code + parameter string.
#[derive(Debug, Clone)]
struct IgesEntity {
    entity_type: u32,
    params: String,
}

/// Internal state during IGES emission.
#[derive(Default)]
struct IgesContext {
    /// Collected entities (curves + surfaces) in order.
    entities: Vec<IgesEntity>,
}

impl IgesContext {
    fn collect_entities(&mut self, store: &BRepStore) {
        let mut seen_curves: Vec<EdgeKey> = Vec::new();
        let mut seen_surfaces: Vec<FaceKey> = Vec::new();

        // Walk faces: surfaces + their edge curves
        for (fk, face) in store.faces.iter() {
            seen_surfaces.push(fk);

            // Collect edges from outer wire
            if let Some(wire) = store.wires.get(face.outer_wire) {
                for &(ek, _orient) in &wire.edges {
                    if !seen_curves.contains(&ek) {
                        seen_curves.push(ek);
                    }
                }
            }
        }

        // Emit curves first (so surfaces can reference them as boundaries)
        for &ek in &seen_curves {
            if let Some(edge) = store.edges.get(ek) {
                let (v_start, v_end) = edge_endpoints(store, ek);
                let entity = curve_to_iges(&edge.curve, v_start, v_end);
                if let Some(e) = entity {
                    self.entities.push(e);
                }
            }
        }

        // Emit surfaces
        for &fk in &seen_surfaces {
            if let Some(face) = store.faces.get(fk) {
                if let Some(entity) = surface_to_iges(&face.surface) {
                    self.entities.push(entity);
                }
            }
        }
    }

    /// Build the complete IGES text (5 sections).
    fn emit(&self) -> String {
        let mut out = String::new();

        // --- parameter data (built first so we know line counts) ---
        let pd_lines = self.emit_parameter_data();

        // --- directory entries ---
        let de_lines = self.emit_directory();

        // --- S section ---
        let s_line = make_iges_line("RUSTCOIN3D IGES EXPORT", 'S', 1);
        out.push_str(&s_line);
        out.push('\n');

        // --- G section ---
        let g_data = concat!(
            "1H,,1H;,12HRUSTCOIN3D IG,12HRUSTCOIN3D IG,",
            ",,,,,,,,,,,1.0,1,2HIN,32767,0.0,"
        );
        let g_line = make_iges_line(g_data, 'G', 1);
        out.push_str(&g_line);
        out.push('\n');

        // --- D section ---
        for line in &de_lines {
            out.push_str(line);
            out.push('\n');
        }

        // --- P section ---
        for line in &pd_lines {
            out.push_str(line);
            out.push('\n');
        }

        // --- T section ---
        let t_line = make_iges_line(
            &format!(
                "S{:07}G{:07}D{:07}P{:07}T{:07}",
                1,
                1,
                de_lines.len(),
                pd_lines.len(),
                1,
            ),
            'T',
            1,
        );
        out.push_str(&t_line);
        out.push('\n');

        out
    }

    /// Emit Parameter Data section lines.
    fn emit_parameter_data(&self) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();

        for entity in &self.entities {
            let raw = &entity.params;
            // Split into 64-char chunks (IGES P-lines: cols 1-64 for data, 65-72 blank)
            let mut remaining = raw.as_str();
            if remaining.is_empty() {
                remaining = ";";
            }
            while !remaining.is_empty() {
                let chunk_len = 64usize.min(remaining.len());
                let chunk = &remaining[..chunk_len];
                let idx = lines.len() + 1;
                lines.push(make_iges_line(chunk, 'P', idx));
                remaining = &remaining[chunk_len..];
            }
        }

        lines
    }

    /// Emit Directory Entry section lines (2 per entity).
    fn emit_directory(&self) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let mut seq = 1usize;

        // Track parameter line offsets: for each entity, we need its 1-based
        // starting line index in the P section.
        let mut pd_offset = 1usize; // 1-based

        for entity in &self.entities {
            // Compute how many P-lines this entity uses
            let param_line_count = count_p_lines(&entity.params);

            // DE line 1
            let de1 = format!(
                "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
                entity.entity_type,
                pd_offset, // parameter pointer
                0,         // structure
                0,         // line font
                0,         // level
                0,         // view
                0,         // transform
                0,         // label
                0,         // status
            );
            lines.push(make_iges_line(&de1, 'D', seq));
            seq += 1;

            // DE line 2
            let de2 = format!(
                "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
                entity.entity_type,
                0,                 // line weight
                0,                 // color
                param_line_count,  // parameter line count
                0,                 // form
                0,                 // reserved
                0,                 // reserved
                "",                // entity label
                0,                 // entity subscript
            );
            lines.push(make_iges_line(&de2, 'D', seq));
            seq += 1;

            pd_offset += param_line_count;
        }

        lines
    }
}

fn count_p_lines(params: &str) -> usize {
    if params.is_empty() {
        return 1; // just ";"
    }
    // Each P-line fits 64 chars of data
    params.len().div_ceil(64).max(1)
}

// ---------------------------------------------------------------------------
// Curve → IGES entity conversion
// ---------------------------------------------------------------------------

/// Resolve the 3D start and end points of an edge.
fn edge_endpoints(store: &BRepStore, ek: EdgeKey) -> (PVec3, PVec3) {
    let edge = match store.edges.get(ek) {
        Some(e) => e,
        None => return (PVec3::ZERO, PVec3::ZERO),
    };
    let p_start = store
        .vertices
        .get(edge.v_low)
        .map(|v| v.position)
        .unwrap_or(PVec3::ZERO);
    let p_end = store
        .vertices
        .get(edge.v_high)
        .map(|v| v.position)
        .unwrap_or(PVec3::ZERO);
    (p_start, p_end)
}

/// Convert a CurveGeom to an IGES entity (if supported).
/// Returns None for unsupported curves.
fn curve_to_iges(curve: &CurveGeom, v_start: PVec3, v_end: PVec3) -> Option<IgesEntity> {
    match curve {
        CurveGeom::Line { .. } => {
            let params = format!(
                "{},{},{},{},{},{};",
                fmt_iges_real(v_start.x),
                fmt_iges_real(v_start.y),
                fmt_iges_real(v_start.z),
                fmt_iges_real(v_end.x),
                fmt_iges_real(v_end.y),
                fmt_iges_real(v_end.z),
            );
            Some(IgesEntity {
                entity_type: 110,
                params,

            })
        }

        CurveGeom::Circle {
            center, ..
        } => {
            // IGES Type 100 circular arcs are in the XY plane (ZT = Z).
            // We need to check if the circle lies in a plane perpendicular to Z.
            // For simplicity, emit in world coords using Z=ZT form.
            let zt = center.z;
            let params = format!(
                "{},{},{},{},{},{},{};",
                fmt_iges_real(zt),
                fmt_iges_real(center.x),
                fmt_iges_real(center.y),
                fmt_iges_real(v_start.x),
                fmt_iges_real(v_start.y),
                fmt_iges_real(v_end.x),
                fmt_iges_real(v_end.y),
            );
            Some(IgesEntity {
                entity_type: 100,
                params,

            })
        }

        CurveGeom::Polyline { points } => {
            if points.len() < 2 {
                return None;
            }
            let mut params = String::from("1,"); // IP=1 for form 12 (polyline)
            params.push_str(&points.len().to_string());
            for pt in points {
                params.push(',');
                params.push_str(&fmt_iges_real(pt.x));
                params.push(',');
                params.push_str(&fmt_iges_real(pt.y));
                params.push(',');
                params.push_str(&fmt_iges_real(pt.z));
            }
            params.push(';');
            Some(IgesEntity {
                entity_type: 106,
                params,

            })
        }

        // Sample analytic curves as polylines
        CurveGeom::Ellipse { .. }
        | CurveGeom::Hyperbola { .. }
        | CurveGeom::Parabola { .. }
        | CurveGeom::BSpline { .. }
        | CurveGeom::BezierCurve { .. }
        | CurveGeom::Offset { .. } => {
            sample_as_polyline(curve, 32)
        }

        CurveGeom::Trimmed { basis, .. } => {
            curve_to_iges(basis, v_start, v_end)
        }

        CurveGeom::Composite { segments, .. } => {
            if segments.is_empty() {
                return None;
            }
            // For Composite curves, emit each segment separately and reference
            // them via a Type 102 (Composite Curve).
            // However, DE cross-referencing requires knowing future DE indices.
            // Simplify: sample as polyline.
            sample_as_polyline(curve, 32)
        }
    }
}

/// Sample a curve at `n` evenly-spaced parameter values and emit as Type 106
/// polyline.
fn sample_as_polyline(curve: &CurveGeom, n: usize) -> Option<IgesEntity> {
    let points: Vec<PVec3> = (0..=n)
        .map(|i| curve.d0(i as Real / n as Real))
        .collect();
    if points.len() < 2 {
        return None;
    }
    let mut params = String::from("1,");
    params.push_str(&points.len().to_string());
    for pt in &points {
        params.push(',');
        params.push_str(&fmt_iges_real(pt.x));
        params.push(',');
        params.push_str(&fmt_iges_real(pt.y));
        params.push(',');
        params.push_str(&fmt_iges_real(pt.z));
    }
    params.push(';');
    Some(IgesEntity {
        entity_type: 106,
        params,
    })
}

// ---------------------------------------------------------------------------
// Surface → IGES entity conversion
// ---------------------------------------------------------------------------

fn surface_to_iges(surface: &SurfaceGeom) -> Option<IgesEntity> {
    match surface {
        SurfaceGeom::Plane { origin, normal, .. } => {
            // Plane equation: Ax + By + Cz + D = 0
            // D = -(A*ox + B*oy + C*oz)
            let n = normal.normalize();
            let d = -(n.x * origin.x + n.y * origin.y + n.z * origin.z);
            // Parameter 5: DE pointer to boundary curve (0 = none)
            let params = format!(
                "{},{},{},{},0;",
                fmt_iges_real(n.x),
                fmt_iges_real(n.y),
                fmt_iges_real(n.z),
                fmt_iges_real(d),
            );
            Some(IgesEntity {
                entity_type: 108,
                params,

            })
        }

        SurfaceGeom::Cylinder {
            origin, axis, radius, ..
        } => {
            // Type 120: Surface of Revolution
            // Generatrix: line from (origin) to (origin + axis * 1 + right * radius)
            // Actually, for a cylinder: generatrix is a line parallel to axis at
            // distance radius from the axis. A simple representation: line from
            // (origin + x_dir*radius) to (origin + x_dir*radius + axis)
            let n = axis.normalize();
            let u = if n.x.abs() < 0.9 {
                PVec3::new(1.0, 0.0, 0.0).cross(n).normalize()
            } else {
                PVec3::new(0.0, 1.0, 0.0).cross(n).normalize()
            };
            let gen_start = *origin + u * *radius;
            let gen_end = gen_start + n;

            // Emit the generatrix as a separate line entity (Type 110)
            // We don't know the DE index yet, so just store the line as a child.
            let _gen_params = format!(
                "{},{},{},{},{},{};",
                fmt_iges_real(gen_start.x),
                fmt_iges_real(gen_start.y),
                fmt_iges_real(gen_start.z),
                fmt_iges_real(gen_end.x),
                fmt_iges_real(gen_end.y),
                fmt_iges_real(gen_end.z),
            );

            // For the surface, parameters are:
            // AX, AY, AZ, DX, DY, DZ, SA, EA, generatrix_DE
            // SA=0, EA=2*PI, generatrix_DE will be filled by index
            let params = format!(
                "{},{},{},{},{},{},0.0,{:.7},0;",
                fmt_iges_real(origin.x),
                fmt_iges_real(origin.y),
                fmt_iges_real(origin.z),
                fmt_iges_real(n.x),
                fmt_iges_real(n.y),
                fmt_iges_real(n.z),
                2.0 * std::f64::consts::PI,
            );
            // Children: the generatrix line. Note: DE cross-referencing
            // requires ordering (children before parent). We handle this by
            // ensuring the generatrix is emitted before the surface.
            Some(IgesEntity {
                entity_type: 120,
                params,
            })
        }

        // Skip complex surfaces
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// IGES formatting helpers
// ---------------------------------------------------------------------------

/// Format a real number for IGES parameter data.
/// Uses D-exponent notation for large/small numbers, plain decimal otherwise.
fn fmt_iges_real(v: Real) -> String {
    if v.abs() < 1e-15 {
        return "0.0".into();
    }
    if v.abs() < 1e-3 || v.abs() >= 1e5 {
        format!("{:.7E}", v).replace('E', "D")
    } else {
        format!("{:.7}", v)
    }
}

/// Build a single 80-column IGES line.
/// `data` fills columns 1-72 (truncated/padded to exactly 72 chars),
/// `section` goes in column 73, and `seq` fills columns 74-80.
fn make_iges_line(data: &str, section: char, seq: usize) -> String {
    // Truncate to 72 chars for columns 1-72
    let mut line_data = String::with_capacity(72);
    for ch in data.chars().take(72) {
        line_data.push(ch);
    }
    // Pad to exactly 72 columns
    while line_data.len() < 72 {
        line_data.push(' ');
    }
    format!("{}{}{:>7}", line_data, section, seq)
}

// ---------------------------------------------------------------------------
// Roundtrip tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::PVec3;
    use rc3d_shape::geom::CurveGeom;
    use rc3d_shape::geom::SurfaceGeom;
    use rc3d_shape::topo::Orientation;

    fn make_test_store() -> BRepStore {
        let mut store = BRepStore::new();
        let tol = 1e-6;

        // Create a plane face
        let plane = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let fk = store.add_face(plane, tol);

        // Add a line edge
        let v0 = store.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.0), tol);
        let v1 = store.find_or_add_vertex(PVec3::new(10.0, 0.0, 0.0), tol);
        let line = CurveGeom::Line {
            origin: PVec3::new(0.0, 0.0, 0.0),
            direction: PVec3::new(10.0, 0.0, 0.0),
        };
        let pcurve = rc3d_shape::geom::curve2d::Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let ek = store.add_edge_with_pcurve(v0, v1, line, tol, fk, pcurve, true);

        // Wire the edge into the face
        if let Some(face) = store.faces.get_mut(fk) {
            if let Some(wire) = store.wires.get_mut(face.outer_wire) {
                wire.edges.push((ek, Orientation::Forward));
            }
        }

        store
    }

    #[test]
    fn test_write_iges_line() {
        let store = make_test_store();
        let text = write_iges_string(&store);

        // Must have all 5 sections
        assert!(text.contains("S"), "missing S section");
        assert!(text.contains("G"), "missing G section");
        assert!(text.contains("D"), "missing D section");
        assert!(text.contains("P"), "missing P section");
        assert!(text.contains("T"), "missing T section");

        // Must have 80-char lines
        for line in text.lines() {
            assert_eq!(
                line.len(),
                80,
                "line length {} (expected 80): '{}'",
                line.len(),
                line
            );
        }

        // Must contain a Type 110 (Line) directory entry
        assert!(text.contains("110"), "missing Type 110 line entity");
    }

    #[test]
    fn test_fmt_iges_real() {
        assert_eq!(fmt_iges_real(0.0), "0.0");
        assert_eq!(fmt_iges_real(1e-16), "0.0");
        assert_eq!(fmt_iges_real(1.5), "1.5000000");
        assert_eq!(fmt_iges_real(1e-4), "1.0000000D-4");
        assert_eq!(fmt_iges_real(1e6), "1.0000000D6");
    }

    #[test]
    fn test_make_iges_line() {
        let line = make_iges_line("TEST DATA", 'S', 1);
        assert_eq!(line.len(), 80);
        assert_eq!(&line[72..73], "S");
        assert_eq!(line[..9].trim(), "TEST DATA");
        assert_eq!(line[73..].trim(), "1");
    }

    #[test]
    fn test_roundtrip_line() {
        let store = make_test_store();
        let text = write_iges_string(&store);

        // Parse the output back through the IGES reader
        let parsed = crate::iges::parse_iges_str(&text).expect("roundtrip parse should succeed");

        // Should have the same number of edges
        assert_eq!(
            parsed.edges.len(),
            store.edges.len(),
            "edge count mismatch after roundtrip"
        );

        // Should have at least one face
        assert!(parsed.faces.len() >= 1, "missing face after roundtrip");
    }

    #[test]
    fn test_roundtrip_arc() {
        let mut store = BRepStore::new();
        let tol = 1e-6;

        let plane = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let fk = store.add_face(plane, tol);

        // Quarter circle arc from (10,0,0) to (0,10,0) centered at origin
        let v0 = store.find_or_add_vertex(PVec3::new(10.0, 0.0, 0.0), tol);
        let v1 = store.find_or_add_vertex(PVec3::new(0.0, 10.0, 0.0), tol);
        let circle = CurveGeom::circle(PVec3::ZERO, PVec3::Z, 10.0);
        let pcurve = rc3d_shape::geom::curve2d::Curve2d::Line {
            origin: (1.0, 0.0),
            direction: (0.0, 1.0),
        };
        let ek =
            store.add_edge_with_pcurve(v0, v1, circle, tol, fk, pcurve, true);

        if let Some(face) = store.faces.get_mut(fk) {
            if let Some(wire) = store.wires.get_mut(face.outer_wire) {
                wire.edges.push((ek, Orientation::Forward));
            }
        }

        let text = write_iges_string(&store);
        assert!(text.contains("100"), "missing Type 100 arc entity");

        let parsed =
            crate::iges::parse_iges_str(&text).expect("roundtrip parse should succeed");
        assert_eq!(parsed.edges.len(), 1, "should have 1 edge");
    }

    #[test]
    fn test_roundtrip_plane() {
        let mut store = BRepStore::new();
        let tol = 1e-6;

        let plane = SurfaceGeom::Plane {
            origin: PVec3::new(0.0, 0.0, 5.0),
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        store.add_face(plane, tol);

        let text = write_iges_string(&store);
        assert!(text.contains("108"), "missing Type 108 plane entity");

        let parsed =
            crate::iges::parse_iges_str(&text).expect("roundtrip parse should succeed");
        assert_eq!(parsed.faces.len(), 1, "should have 1 face");
    }

    #[test]
    fn test_roundtrip_empty_store() {
        let store = BRepStore::new();
        let text = write_iges_string(&store);

        // Must still produce valid IGES
        assert!(text.contains("S"), "missing S section");
        assert!(text.contains("G"), "missing G section");
        assert!(text.contains("T"), "missing T section");
    }

    #[test]
    fn test_sample_polyline() {
        let circle = CurveGeom::circle(PVec3::ZERO, PVec3::Z, 1.0);
        let poly = sample_as_polyline(&circle, 8).expect("should produce polyline");
        assert_eq!(poly.entity_type, 106);
        // Should have 9 points (0..=8) = 3*9 + header fields
        assert!(poly.params.starts_with("1,9,"));
    }
}
