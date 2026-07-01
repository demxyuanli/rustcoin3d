//! OCC BREP ASCII format writer — classic CASCADE Topology V1 format.
//!
//! Format reference: Open CASCADE Technology BRepTools::Write()
//! This is the format FreeCAD and other OCC-based tools read.
//!
//! Key differences from DrawableShape:
//! - Header: "CASCADE Topology V1, (c) Matra-Datavision"
//! - Section order: Locations, Curve2ds, Curves, Polygon3D, PolygonOnTriangulations,
//!   Surfaces, Triangulations, TShapes
//! - Curves has 7-number lines (includes edge parameter range)
//! - Surfaces has 13-number Plane lines (includes v_dir)
//! - TShapes is a unified section with Ve/Ed/Wi/Fa/Sh/So/Co markers
//! - Sub-shape references count backwards from end: +N = shape at total-N+1

use std::io::{self, Write};
use rc3d_core::math::{Real, PVec3};
use crate::geom::curve2d::Curve2d;
use crate::geom::{CurveGeom, SurfaceGeom};
use crate::store::BRepStore;
use crate::topo::*;

/// Round a float for BREP output to avoid tiny FP artifacts that confuse OCC.
/// Values below 1e-12 are snapped to zero.
fn rnd(x: Real) -> Real {
    if x.abs() < 1e-12 { 0.0 } else { (x * 1e12).round() / 1e12 }
}

/// Extract unique knot values and their multiplicities from a full knot vector.
/// OCC BREP V1 format requires unique knots followed by multiplicities.
fn extract_unique_and_mults(knots: &[Real]) -> (Vec<Real>, Vec<usize>) {
    if knots.is_empty() {
        return (vec![], vec![]);
    }
    let mut unique = vec![knots[0]];
    let mut mults = vec![1usize];
    for &k in &knots[1..] {
        let last = *unique.last().unwrap();
        if (k - last).abs() < 1e-12 {
            *mults.last_mut().unwrap() += 1;
        } else {
            unique.push(k);
            mults.push(1);
        }
    }
    (unique, mults)
}

/// Write the full BRepStore as OCC BREP ASCII (classic format).
pub fn write_brep(store: &BRepStore, output: &mut impl Write) -> io::Result<()> {
    let mut w = BrepWriter::new(store);
    w.write_all(output)
}

struct PCurveEntry {
    edge_abs_pos: usize,  // 1-based edge position in TShapes
    face_abs_pos: usize,  // 1-based face position in TShapes
    curve: Curve2d,
}

struct BrepWriter<'a> {
    store: &'a BRepStore,
    shapes: Vec<ShapeEntry>,
    total_shapes: usize,
    pcurve_entries: Vec<PCurveEntry>,
    pcurve_count: usize,
}

/// Compact representation of each TShape for ordering.
enum ShapeEntry {
    Vertex(VertexKey),
    Edge(EdgeKey),
    Wire(WireKey),
    Face(FaceKey),
    Shell(ShellKey),
    Solid(SolidKey),
    Compound(CompoundKey),
}

impl<'a> BrepWriter<'a> {
    fn new(store: &'a BRepStore) -> Self {
        let mut shapes = Vec::new();

        // Build ordered shape list: all compounds, solids, shells, faces, wires, edges, vertices.
        // OCC writes shapes in dependency order: vertices first, then edges, wires, faces, etc.
        for (vk, _) in &store.vertices {
            shapes.push(ShapeEntry::Vertex(vk));
        }
        for (ek, _) in &store.edges {
            shapes.push(ShapeEntry::Edge(ek));
        }
        for (wk, wire) in &store.wires {
            // Skip empty wires (placeholders from face construction)
            if wire.edges.is_empty() {
                continue;
            }
            shapes.push(ShapeEntry::Wire(wk));
        }
        for (fk, _) in &store.faces {
            shapes.push(ShapeEntry::Face(fk));
        }
        for (sk, _) in &store.shells {
            shapes.push(ShapeEntry::Shell(sk));
        }
        for (sk, _) in &store.solids {
            shapes.push(ShapeEntry::Solid(sk));
        }
        for (ck, _) in &store.compounds {
            shapes.push(ShapeEntry::Compound(ck));
        }

        // Pre-collect all PCurve entries from edges.
        // Stored PCurves were validated and normalized during STEP import.
        let mut pcurve_entries = Vec::new();
        for (i, entry) in shapes.iter().enumerate() {
            if let ShapeEntry::Edge(ek) = entry {
                if let Some(edge) = store.edges.get(*ek) {
                    for (fk, pc) in &edge.pcurves {
                        let face_pos = shapes.iter()
                            .position(|s| matches!(s, ShapeEntry::Face(k) if k == fk))
                            .map(|p| p + 1).unwrap_or(0);
                        if face_pos > 0 {
                            pcurve_entries.push(PCurveEntry {
                                edge_abs_pos: i + 1,
                                face_abs_pos: face_pos,
                                curve: pc.clone(),
                            });
                        }
                    }
                }
            }
        }
        let pcurve_count = pcurve_entries.len();

        let total = shapes.len();
        Self { store, shapes, total_shapes: total, pcurve_entries, pcurve_count }
    }

    fn write_all(&mut self, output: &mut impl Write) -> io::Result<()> {
        writeln!(output, "DBRep_DrawableShape")?;
        writeln!(output)?;
        writeln!(output, "CASCADE Topology V1, (c) Matra-Datavision")?;
        self.write_locations(output)?;
        self.write_curve2ds(output)?;
        self.write_curves(output)?;
        writeln!(output, "Polygon3D 0")?;
        writeln!(output, "PolygonOnTriangulations 0")?;
        self.write_surfaces(output)?;
        writeln!(output, "Triangulations 0")?;
        writeln!(output)?;
        self.write_tshapes(output)?;
        Ok(())
    }

    /// Calculate reverse index: +N means shape at absolute position (total - N + 1).
    fn rev_idx(&self, abs_pos: usize) -> usize {
        if abs_pos == 0 { return 0; }
        self.total_shapes + 1 - abs_pos
    }

    /// Find absolute position of a vertex in shapes list.
    fn vertex_pos(&self, vk: VertexKey) -> usize {
        self.shapes.iter().position(|s| matches!(s, ShapeEntry::Vertex(k) if *k == vk))
            .map(|p| p + 1).unwrap_or(0)
    }

    fn edge_pos(&self, ek: EdgeKey) -> usize {
        self.shapes.iter().position(|s| matches!(s, ShapeEntry::Edge(k) if *k == ek))
            .map(|p| p + 1).unwrap_or(0)
    }

    fn wire_pos(&self, wk: WireKey) -> usize {
        self.shapes.iter().position(|s| matches!(s, ShapeEntry::Wire(k) if *k == wk))
            .map(|p| p + 1).unwrap_or(0)
    }

    fn face_pos(&self, fk: FaceKey) -> usize {
        self.shapes.iter().position(|s| matches!(s, ShapeEntry::Face(k) if *k == fk))
            .map(|p| p + 1).unwrap_or(0)
    }

    fn shell_pos(&self, sk: ShellKey) -> usize {
        self.shapes.iter().position(|s| matches!(s, ShapeEntry::Shell(k) if *k == sk))
            .map(|p| p + 1).unwrap_or(0)
    }

    fn solid_pos(&self, sk: SolidKey) -> usize {
        self.shapes.iter().position(|s| matches!(s, ShapeEntry::Solid(k) if *k == sk))
            .map(|p| p + 1).unwrap_or(0)
    }

    #[allow(dead_code)]
    fn compound_pos(&self, ck: CompoundKey) -> usize {
        self.shapes.iter().position(|s| matches!(s, ShapeEntry::Compound(k) if *k == ck))
            .map(|p| p + 1).unwrap_or(0)
    }

    // ── Sections ──────────────────────────────────────────────────

    fn write_locations(&self, output: &mut impl Write) -> io::Result<()> {
        let n = self.store.locations.len();
        if n == 0 {
            writeln!(output, "Locations 0")?;
            return Ok(());
        }
        writeln!(output, "Locations {}", n)?;
        for (i, loc) in self.store.locations.iter().enumerate() {
            let idx = i + 1; // 1-based location index
            // Full 3×4 affine matrix form: 3 rows of (r11,r12,r13,t)
            writeln!(output, "{}", idx)?;
            writeln!(output, "{} {} {} {}",
                rnd(loc[0]), rnd(loc[1]), rnd(loc[2]), rnd(loc[3]))?;
            writeln!(output, "{} {} {} {}",
                rnd(loc[4]), rnd(loc[5]), rnd(loc[6]), rnd(loc[7]))?;
            writeln!(output, "{} {} {} {}",
                rnd(loc[8]), rnd(loc[9]), rnd(loc[10]), rnd(loc[11]))?;
        }
        Ok(())
    }

    /// Write PCurves section — raw 2D curves in OCC format (same types as 3D Curves).
    fn write_curve2ds(&self, output: &mut impl Write) -> io::Result<()> {
        if self.pcurve_count == 0 {
            writeln!(output, "Curve2ds 0")?;
            return Ok(());
        }
        writeln!(output, "Curve2ds {}", self.pcurve_count)?;
        for entry in &self.pcurve_entries {
            match &entry.curve {
                Curve2d::Line { origin, direction } => {
                    writeln!(output, "1 {} {} {} {}",
                        rnd(origin.0), rnd(origin.1),
                        rnd(direction.0), rnd(direction.1))?;
                }
                Curve2d::Circle { center, radius } => {
                    let r = radius.max(1e-6);
                    let tau = std::f64::consts::TAU;
                    writeln!(output, "2 0 {} {} {} {} {} {}", tau * r, center.0, center.1, 1.0, 0.0, r)?;
                }
                Curve2d::Ellipse { center, semi_major, semi_minor } => {
                    writeln!(output, "3 0 {} {} {} {} {} {} {}", std::f64::consts::TAU * semi_major.max(*semi_minor),
                        center.0, center.1, *semi_major, *semi_minor, 1.0, 0.0)?;
                }
                Curve2d::BSpline { control_points, .. } => {
                    if let (Some(f), Some(l)) = (control_points.first(), control_points.last()) {
                        writeln!(output, "1 {} {} {} {}",
                            rnd(f.0), rnd(f.1),
                            rnd(l.0 - f.0), rnd(l.1 - f.1))?;
                    } else { writeln!(output, "1 0 1 1 0")?; }
                }
                Curve2d::Trimmed { basis, .. } => {
                    match basis.as_ref() {
                        Curve2d::Line { origin, direction } => {
                            writeln!(output, "1 {} {} {} {}",
                                rnd(origin.0), rnd(origin.1),
                                rnd(direction.0), rnd(direction.1))?;
                        }
                        Curve2d::Circle { center, radius } => {
                            let r = radius.max(1e-6);
                            writeln!(output, "2 0 {} {} {} {} {} {}", std::f64::consts::TAU * r, center.0, center.1, 1.0, 0.0, r)?;
                        }
                        _ => writeln!(output, "1 0 1 1 0")?,
                    }
                }
                Curve2d::Polyline { points } => {
                    // Simplify collinear polylines to Line for compact OCC-compatible output
                    if let Some(line) = crate::geom::curve2d::simplify_polyline_to_line(points) {
                        match &line {
                            Curve2d::Line { origin, direction } => {
                                writeln!(output, "1 {} {} {} {}",
                                    rnd(origin.0), rnd(origin.1),
                                    rnd(direction.0), rnd(direction.1))?;
                            }
                            _ => write_polyline2d_as_bspline(output, points)?,
                        }
                    } else {
                        write_polyline2d_as_bspline(output, points)?;
                    }
                }
                Curve2d::Composite { .. } => {
                    writeln!(output, "1 0 1 1 0")?;
                }
            }
        }
        Ok(())
    }

    fn write_curves(&self, output: &mut impl Write) -> io::Result<()> {
        let mut curves: Vec<&CurveGeom> = Vec::new();
        for entry in &self.shapes {
            if let ShapeEntry::Edge(ek) = entry {
                if let Some(edge) = self.store.edges.get(*ek) {
                    curves.push(&edge.curve);
                }
            }
        }
        writeln!(output, "Curves {}", curves.len())?;

        for curve in &curves {
            let expanded = expand_curve(curve);
            match expanded {
                CurveGeom::Line { origin, direction } => {
                    let len = direction.length();
                    let dir = if len > 1e-12 { *direction / len } else { PVec3::X };
                    writeln!(output, "1 {} {} {} {} {} {}",
                        rnd(origin.x), rnd(origin.y), rnd(origin.z),
                        rnd(dir.x), rnd(dir.y), rnd(dir.z))?;
                }
                CurveGeom::Circle { center, axis, radius, x_dir, y_dir } => {
                    writeln!(output, "2 {} {} {}  {} {} {}  {} {} {}  {} {} {}  {}",
                        rnd(center.x), rnd(center.y), rnd(center.z),
                        rnd(axis.x), rnd(axis.y), rnd(axis.z),
                        rnd(x_dir.x), rnd(x_dir.y), rnd(x_dir.z),
                        rnd(y_dir.x), rnd(y_dir.y), rnd(y_dir.z), rnd(*radius))?;
                }
                CurveGeom::Ellipse { center, axis, semi_major, semi_minor, x_dir, y_dir } => {
                    writeln!(output, "3 {} {} {}  {} {} {}  {} {} {}  {} {} {}  {} {}",
                        rnd(center.x), rnd(center.y), rnd(center.z),
                        rnd(axis.x), rnd(axis.y), rnd(axis.z),
                        rnd(x_dir.x), rnd(x_dir.y), rnd(x_dir.z),
                        rnd(y_dir.x), rnd(y_dir.y), rnd(y_dir.z),
                        rnd(*semi_major), rnd(*semi_minor))?;
                }
                CurveGeom::Hyperbola { center, axis, semi_major, semi_minor, x_dir, y_dir } => {
                    writeln!(output, "4 {} {} {}  {} {} {}  {} {} {}  {} {} {}  {} {}",
                        rnd(center.x), rnd(center.y), rnd(center.z),
                        rnd(axis.x), rnd(axis.y), rnd(axis.z),
                        rnd(x_dir.x), rnd(x_dir.y), rnd(x_dir.z),
                        rnd(y_dir.x), rnd(y_dir.y), rnd(y_dir.z),
                        rnd(*semi_major), rnd(*semi_minor))?;
                }
                CurveGeom::Parabola { center, axis, focal_dist, x_dir, y_dir } => {
                    writeln!(output, "5 {} {} {}  {} {} {}  {} {} {}  {} {} {}  {}",
                        rnd(center.x), rnd(center.y), rnd(center.z),
                        rnd(axis.x), rnd(axis.y), rnd(axis.z),
                        rnd(x_dir.x), rnd(x_dir.y), rnd(x_dir.z),
                        y_dir.x, y_dir.y, y_dir.z, focal_dist)?;
                }
                CurveGeom::BezierCurve { degree, control_points, weights } => {
                    write!(output, "6 {}  {}  {}",
                        degree, control_points.len(),
                        if weights.is_some() { 1 } else { 0 })?;
                    for cp in control_points {
                        write!(output, "  {} {} {}", cp.x, cp.y, cp.z)?;
                    }
                    if let Some(w) = weights {
                        for wt in w { write!(output, " {}", wt)?; }
                    }
                    writeln!(output)?;
                }
                CurveGeom::BSpline { degree, control_points, knots, weights } => {
                    let (unique_knots, mults) = extract_unique_and_mults(knots);
                    write!(output, "7 {}  {}  {}  {}",
                        degree, control_points.len(), unique_knots.len(),
                        if weights.is_some() { 1 } else { 0 })?;
                    for cp in control_points {
                        write!(output, "  {} {} {}", cp.x, cp.y, cp.z)?;
                    }
                    for k in &unique_knots { write!(output, " {}", k)?; }
                    for m in &mults { write!(output, " {}", m)?; }
                    if let Some(w) = weights {
                        for wt in w { write!(output, " {}", wt)?; }
                    }
                    writeln!(output)?;
                }
                CurveGeom::Polyline { points } => {
                    // Write as BSpline degree 1
                    write_polyline_as_bspline(output, points)?;
                }
                CurveGeom::Trimmed { basis, .. } => {
                    // Expand: write the inner basis in its full form.
                    // The edge t_min/t_max in TShapes already encode the trim.
                    match basis.as_ref() {
                        CurveGeom::Line { origin, direction } => {
                            let len = direction.length();
                            let dir = if len > 1e-12 { *direction / len } else { PVec3::X };
                            writeln!(output, "1 {} {} {} {} {} {}",
                                rnd(origin.x), rnd(origin.y), rnd(origin.z),
                                rnd(dir.x), rnd(dir.y), rnd(dir.z))?;
                        }
                        CurveGeom::Circle { center, axis, radius, x_dir, y_dir } => {
                            writeln!(output, "2 {} {} {}  {} {} {}  {} {} {}  {} {} {}  {}",
                                center.x, center.y, center.z,
                                axis.x, axis.y, axis.z,
                                x_dir.x, x_dir.y, x_dir.z,
                                y_dir.x, y_dir.y, y_dir.z, radius)?;
                        }
                        CurveGeom::Ellipse { center, axis, semi_major, semi_minor, x_dir, y_dir } => {
                            writeln!(output, "3 {} {} {}  {} {} {}  {} {} {}  {} {} {}  {} {}",
                                center.x, center.y, center.z,
                                axis.x, axis.y, axis.z,
                                x_dir.x, x_dir.y, x_dir.z,
                                y_dir.x, y_dir.y, y_dir.z,
                                semi_major, semi_minor)?;
                        }
                        CurveGeom::BSpline { degree, control_points, knots, weights } => {
                            let (uk, um) = extract_unique_and_mults(knots);
                            write!(output, "7 {}  {}  {}  {}",
                                degree, control_points.len(), uk.len(),
                                if weights.is_some() { 1 } else { 0 })?;
                            for cp in control_points {
                                write!(output, "  {} {} {}", cp.x, cp.y, cp.z)?;
                            }
                            for k in &uk { write!(output, " {}", k)?; }
                            for m in &um { write!(output, " {}", m)?; }
                            if let Some(w) = weights {
                                for wt in w { write!(output, " {}", wt)?; }
                            }
                            writeln!(output)?;
                        }
                        CurveGeom::BezierCurve { degree, control_points, weights } => {
                            write!(output, "6 {}  {}  {}",
                                degree, control_points.len(),
                                if weights.is_some() { 1 } else { 0 })?;
                            for cp in control_points {
                                write!(output, "  {} {} {}", cp.x, cp.y, cp.z)?;
                            }
                            if let Some(w) = weights {
                                for wt in w { write!(output, " {}", wt)?; }
                            }
                            writeln!(output)?;
                        }
                        _ => writeln!(output, "1 0 0 0  1 0 0")?,
                    }
                }
                _ => {
                    // Fallback for composite/offset etc.
                    writeln!(output, "1 0 0 0  1 0 0")?;
                }
            }
        }
        Ok(())
    }

    fn write_surfaces(&self, output: &mut impl Write) -> io::Result<()> {
        let mut surfaces: Vec<&SurfaceGeom> = Vec::new();
        for entry in &self.shapes {
            if let ShapeEntry::Face(fk) = entry {
                if let Some(face) = self.store.faces.get(*fk) {
                    surfaces.push(&face.surface);
                }
            }
        }
        writeln!(output, "Surfaces {}", surfaces.len())?;

        for surface in &surfaces {
            match surface {
                SurfaceGeom::Plane { origin, normal, u_dir } => {
                    let n = if normal.length_squared() > 1e-12 {
                        *normal / normal.length()
                    } else {
                        *normal
                    };
                    let u = if u_dir.length_squared() > 1e-12 {
                        *u_dir / u_dir.length()
                    } else {
                        *u_dir
                    };
                    let v = n.cross(u);
                    // OCC Plane: 1 ox oy oz nx ny nz ux uy uz vx vy vz (13 numbers)
                    writeln!(output, "1 {} {} {} {} {} {} {} {} {} {} {} {}",
                        rnd(origin.x), rnd(origin.y), rnd(origin.z),
                        rnd(n.x), rnd(n.y), rnd(n.z),
                        rnd(u.x), rnd(u.y), rnd(u.z),
                        rnd(v.x), rnd(v.y), rnd(v.z))?;
                }
                SurfaceGeom::Cylinder { origin, axis, radius, x_dir, y_dir } => {
                    writeln!(output, "2 {} {} {} {} {} {} {} {} {} {} {} {} {}",
                        rnd(origin.x), rnd(origin.y), rnd(origin.z),
                        rnd(axis.x), rnd(axis.y), rnd(axis.z),
                        rnd(x_dir.x), rnd(x_dir.y), rnd(x_dir.z),
                        rnd(y_dir.x), rnd(y_dir.y), rnd(y_dir.z), rnd(*radius))?;
                }
                SurfaceGeom::Cone { apex, axis, semi_angle: _, radius_at_apex, x_dir, y_dir } => {
                    writeln!(output, "3 {} {} {} {} {} {} {} {} {} {} {} {} {}",
                        rnd(apex.x), rnd(apex.y), rnd(apex.z),
                        rnd(axis.x), rnd(axis.y), rnd(axis.z),
                        rnd(x_dir.x), rnd(x_dir.y), rnd(x_dir.z),
                        rnd(y_dir.x), rnd(y_dir.y), rnd(y_dir.z),
                        rnd(*radius_at_apex))?;
                }
                SurfaceGeom::Sphere { center, radius } => {
                    // OCC Sphere format: 4 cx cy cz nx ny nz ux uy uz vx vy vz r
                    // nx,ny,nz = polar axis (Z), u_dir = (1,0,0), v_dir = (0,1,0)
                    let u = PVec3::X;
                    let v = PVec3::Y;
                    let n = PVec3::Z;
                    writeln!(output, "4 {} {} {} {} {} {} {} {} {} {} {} {} {}",
                        rnd(center.x), rnd(center.y), rnd(center.z),
                        rnd(n.x), rnd(n.y), rnd(n.z),
                        rnd(u.x), rnd(u.y), rnd(u.z),
                        rnd(v.x), rnd(v.y), rnd(v.z),
                        radius)?;
                }
                SurfaceGeom::Torus { center, axis, major_r, minor_r, x_dir, y_dir } => {
                    writeln!(output, "5 {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                        rnd(center.x), rnd(center.y), rnd(center.z),
                        rnd(axis.x), rnd(axis.y), rnd(axis.z),
                        rnd(x_dir.x), rnd(x_dir.y), rnd(x_dir.z),
                        rnd(y_dir.x), rnd(y_dir.y), rnd(y_dir.z),
                        rnd(*major_r), rnd(*minor_r))?;
                }
                SurfaceGeom::BSpline(ns) => {
                    let rational = nurbs_is_rational(&ns.weights);
                    let (u_knots, u_mults) = extract_unique_and_mults(&ns.knots_u);
                    let (v_knots, v_mults) = extract_unique_and_mults(&ns.knots_v);
                    // OCC BREP V1: 8 du dv uc vc ru rv (7 ints), where
                    //   uc = u_knot_count (unique), vc = v_knot_count (unique)
                    //   ru = 0 (polynomial) or 2 (rational non-periodic) or 3 (rational periodic)
                    //   rv = same as ru
                    let mode = if rational { 2 } else { 0 };
                    write!(output, "8 {} {} {} {} {} {}",
                        ns.degree_u, ns.degree_v,
                        u_knots.len(), v_knots.len(),
                        mode, mode)?;
                    // Control points: for rational surfaces, write x y z w (actual weight)
                    for (row_idx, row) in ns.control_points.iter().enumerate() {
                        for (col_idx, cp) in row.iter().enumerate() {
                            if rational {
                                let w: Real = ns.weights
                                    .get(row_idx)
                                    .and_then(|weights_row: &Vec<Real>| weights_row.get(col_idx))
                                    .copied()
                                    .unwrap_or(1.0);
                                write!(output, " {} {} {} {}", cp.x, cp.y, cp.z, w)?;
                            } else {
                                write!(output, " {} {} {}", cp.x, cp.y, cp.z)?;
                            }
                        }
                    }
                    // Unique knots + multiplicities (OCC V1 format)
                    for k in &u_knots { write!(output, " {}", k)?; }
                    for m in &u_mults { write!(output, " {}", m)?; }
                    for k in &v_knots { write!(output, " {}", k)?; }
                    for m in &v_mults { write!(output, " {}", m)?; }
                    writeln!(output)?;
                }
                SurfaceGeom::Extrusion { generatrix, direction } => {
                    // Write as-is with direction; generatrix is handled by OCC internally
                    let base_pt = generatrix.d0(0.0);
                    writeln!(output, "6 {} {} {}  {} {} {}",
                        direction.x, direction.y, direction.z,
                        base_pt.x, base_pt.y, base_pt.z)?;
                }
                SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir, .. } => {
                    writeln!(output, "7 {} {} {} {} {} {}",
                        axis_origin.x, axis_origin.y, axis_origin.z,
                        axis_dir.x, axis_dir.y, axis_dir.z)?;
                    let gen_pts: Vec<PVec3> = (0..=16).map(|i| generatrix.d0(i as Real / 16.0)).collect();
                    let n = gen_pts.len();
                    write!(output, "7 0 0  1 {} {}", n, n + 2)?;
                    for p in &gen_pts { write!(output, "  {} {} {}", p.x, p.y, p.z)?; }
                    writeln!(output)?;
                    writeln!(output, " 0 {} 1 {}", 2, 2)?;
                }
                SurfaceGeom::Offset { basis, distance } => {
                    // Expand offset surface: write the actual geometry
                    write_expanded_offset_surface(output, basis, *distance)?;
                }
            }
        }
        Ok(())
    }

    // ── TShapes section ───────────────────────────────────────────

    fn write_tshapes(&self, output: &mut impl Write) -> io::Result<()> {
        writeln!(output, "TShapes {}", self.total_shapes)?;

        let mut edge_idx = 0usize;
        for entry in &self.shapes {
            match entry {
                ShapeEntry::Vertex(vk) => self.write_ve(output, *vk)?,
                ShapeEntry::Edge(ek) => {
                    edge_idx += 1;
                    self.write_ed(output, *ek, edge_idx)?;
                }
                ShapeEntry::Wire(wk) => self.write_wi(output, *wk)?,
                ShapeEntry::Face(fk) => self.write_fa(output, *fk)?,
                ShapeEntry::Shell(sk) => self.write_sh(output, *sk)?,
                ShapeEntry::Solid(sk) => self.write_so(output, *sk)?,
                ShapeEntry::Compound(ck) => {
                    let compound = self.store.compounds.get(*ck);
                    let solid_keys: Vec<SolidKey> = compound.map(|c| c.solids.clone()).unwrap_or_default();
                    self.write_co(output, &solid_keys)?;
                }
            }
        }

        // Final references line: +1 0
        writeln!(output)?;
        writeln!(output, "+1 0")?;
        Ok(())
    }

    fn write_ve(&self, output: &mut impl Write, vk: VertexKey) -> io::Result<()> {
        let v = self.store.vertices.get(vk).map(|v| v.position)
            .unwrap_or(PVec3::ZERO);
        writeln!(output, "Ve")?;
        writeln!(output, "1e-07")?;
        writeln!(output, "{} {} {}", rnd(v.x), rnd(v.y), rnd(v.z))?;
        writeln!(output, "0 0")?;
        writeln!(output)?;
        writeln!(output, "0101101")?; // free, modified, checked, orientable, closed, infinite, convex
        writeln!(output, "*")?;
        Ok(())
    }

    fn write_ed(&self, output: &mut impl Write, ek: EdgeKey, curve_idx: usize) -> io::Result<()> {
        let edge = match self.store.edges.get(ek) {
            Some(e) => e,
            None => {
                writeln!(output, "Ed")?;
                writeln!(output, " 1e-07 1 1 0")?;
                writeln!(output, "1  0 0 0 0")?;
                writeln!(output, "0")?;
                writeln!(output)?;
                writeln!(output, "0101000")?;
                writeln!(output, "*")?;
                return Ok(());
            }
        };

        let param_range = (edge.t_max - edge.t_min).abs();

        let curve_type = occ_curve_type(expand_curve(&edge.curve));
        let v1_pos = self.vertex_pos(edge.v_low);
        let v2_pos = self.vertex_pos(edge.v_high);
        let v1_loc = self.store.vertex_locations.get(&edge.v_low).copied().unwrap_or(0);
        let v2_loc = self.store.vertex_locations.get(&edge.v_high).copied().unwrap_or(0);
        let rv1 = self.rev_idx(v1_pos);
        let rv2 = self.rev_idx(v2_pos);

        // OCC convention: edges use 1e-07 unless the tolerance is already
        // inflated (e.g. by same-parameter healing). Clamp large tolerances.
        let tol_str = if edge.tolerance > 0.1 {
            "1e-04".to_string()
        } else {
            "1e-07".to_string()
        };
        // Degenerated: self-loop with zero-length curve, OR listed in
        // face.degenerated_edges (sphere poles, cone apex). Self-loop
        // circles on non-singular surfaces (torus) are NOT degenerated.
        let is_degen = edge.v_low == edge.v_high && (
            matches!(&edge.curve, CurveGeom::Line { direction, .. } if direction.length() < 1e-12)
            || self.store.faces.iter().any(|(_fk, f)| f.degenerated_edges.contains(&ek))
        );
        let same_range = if is_degen { 0 } else { 1 };
        writeln!(output, "Ed")?;
        let degen_flag: u8 = if is_degen { 1 } else { 0 };
        writeln!(output, " {} 1 {} {}", tol_str, same_range, degen_flag)?;
        // OCC format (V1): curve_type curve_idx orient flag t_min t_max
        // orient: 1 = same parameter direction as edge, 0 = opposite
        // flag: 0 = polynomial, 2 = rational, etc.
        writeln!(output, "{}  {} 1 0 {} {}",
            curve_type, curve_idx,
            rnd(edge.t_min), rnd(edge.t_max))?;

        // PCurve references — OCC V1 format varies by curve2d type.
        // Degenerated edges have no PCurves (OCC writes just "0").
        if is_degen {
            writeln!(output, "0")?;
        } else if self.pcurve_count == 0 {
            writeln!(output, "0")?;
        } else {
            let edge_abs = self.edge_pos(ek);
            let mut pc_lines: Vec<(usize, usize)> = Vec::new(); // (1-based curve2d index, 0-based entry index)
            for (idx, entry) in self.pcurve_entries.iter().enumerate() {
                if entry.edge_abs_pos == edge_abs {
                    pc_lines.push((idx + 1, idx));
                }
            }
            if pc_lines.is_empty() {
                writeln!(output, "0")?;
            } else {
                for (pc_idx, entry_idx) in &pc_lines {
                    write_pcurve_ref(output, pc_idx, &self.pcurve_entries[*entry_idx].curve)?;
                }
                writeln!(output, "0")?;
            }
        }
        writeln!(output)?;
        let tshape_flags = if is_degen { "0101100" } else { "0101000" };
        writeln!(output, "{}", tshape_flags)?;
        writeln!(output, "+{} {} -{} {} *", rv1, v1_loc, rv2, v2_loc)?;
        Ok(())
    }

    fn write_wi(&self, output: &mut impl Write, wk: WireKey) -> io::Result<()> {
        let wire = match self.store.wires.get(wk) {
            Some(w) => w,
            None => {
                writeln!(output, "Wi")?;
                writeln!(output)?;
                writeln!(output, "0101100")?;
                writeln!(output, "*")?;
                return Ok(());
            }
        };

        writeln!(output, "Wi")?;
        writeln!(output)?;
        // Flags
        writeln!(output, "0101100")?;
        // Edge references (reverse-indexed)
        for (ek, orient) in &wire.edges {
            let ep = self.edge_pos(*ek);
            let rp = self.rev_idx(ep);
            match orient {
                Orientation::Forward => write!(output, "+{} 0 ", rp)?,
                Orientation::Reversed => write!(output, "-{} 0 ", rp)?,
                _ => write!(output, "+{} 0 ", rp)?,
            }
        }
        writeln!(output, "*")?;
        Ok(())
    }

    fn write_fa(&self, output: &mut impl Write, fk: FaceKey) -> io::Result<()> {
        let face = match self.store.faces.get(fk) {
            Some(f) => f,
            None => {
                writeln!(output, "Fa")?;
                writeln!(output, "0  1e-07 1 0")?;
                writeln!(output)?;
                writeln!(output, "0101000")?;
                writeln!(output, "*")?;
                return Ok(());
            }
        };

        let wp = self.wire_pos(face.outer_wire);
        // Find surface index (1-based) by scanning shapes for Face entries
        let surf_idx = self.shapes.iter()
            .filter(|s| matches!(s, ShapeEntry::Face(_)))
            .position(|s| matches!(s, ShapeEntry::Face(k) if *k == fk))
            .map(|p| p + 1).unwrap_or(0);

        writeln!(output, "Fa")?;
        // OCC format: location_index  tolerance  surface_index  natural_restriction
        // natural_restriction = 0: use wire boundaries (standard)
        let natural = 0i32;
        writeln!(output, "0  {}  {} {}", face.tolerance, surf_idx, natural)?;
        writeln!(output)?;
        writeln!(output, "0101000")?;
        // Outer wire reference (reverse-indexed, forward orientation)
        let rwp = self.rev_idx(wp);
        write!(output, "+{} 0 ", rwp)?;
        // Inner wire references (reverse-indexed, reversed orientation)
        for iw in &face.inner_wires {
            let iwp = self.wire_pos(*iw);
            if iwp > 0 {
                let riwp = self.rev_idx(iwp);
                write!(output, "-{} 0 ", riwp)?;
            }
        }
        writeln!(output, "*")?;
        Ok(())
    }

    fn write_sh(&self, output: &mut impl Write, sk: ShellKey) -> io::Result<()> {
        let shell = match self.store.shells.get(sk) {
            Some(s) => s,
            None => {
                writeln!(output, "Sh")?;
                writeln!(output)?;
                writeln!(output, "0101100")?;
                writeln!(output, "*")?;
                return Ok(());
            }
        };

        writeln!(output, "Sh")?;
        writeln!(output)?;
        writeln!(output, "0101100")?;
        // Compute solid center for face-orientation heuristic
        let center = shell_center(&shell.faces, self.store);
        for (fk, _orient) in &shell.faces {
            let fp = self.face_pos(*fk);
            let rp = self.rev_idx(fp);
            // Determine if face normal (with same_sense) points outward from solid center
            let outward = face_normal_points_outward(*fk, center, self.store);
            if outward {
                write!(output, "+{} 0 ", rp)?;
            } else {
                write!(output, "-{} 0 ", rp)?;
            }
        }
        writeln!(output, "*")?;
        Ok(())
    }

    fn write_so(&self, output: &mut impl Write, sk: SolidKey) -> io::Result<()> {
        let solid = match self.store.solids.get(sk) {
            Some(s) => s,
            None => {
                writeln!(output, "So")?;
                writeln!(output)?;
                writeln!(output, "0100000")?;
                writeln!(output, "*")?;
                return Ok(());
            }
        };

        let sp = self.shell_pos(solid.outer_shell);
        let rsp = self.rev_idx(sp);

        writeln!(output, "So")?;
        writeln!(output)?;
        writeln!(output, "0100000")?;
        writeln!(output, "+{} 0 *", rsp)?;
        Ok(())
    }

    fn write_co(&self, output: &mut impl Write, solid_keys: &[SolidKey]) -> io::Result<()> {
        let mut refs = Vec::new();
        for sk in solid_keys {
            let sp = self.solid_pos(*sk);
            refs.push(self.rev_idx(sp));
        }

        writeln!(output, "Co")?;
        writeln!(output)?;
        writeln!(output, "1100000")?;
        for r in &refs {
            write!(output, "+{} 0 ", r)?;
        }
        writeln!(output, "*")?;
        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Map CurveGeom to OCC curve type number.
fn occ_curve_type(curve: &CurveGeom) -> usize {
    match curve {
        CurveGeom::Line { .. } => 1,
        CurveGeom::Circle { .. } => 2,
        CurveGeom::Ellipse { .. } => 3,
        CurveGeom::Hyperbola { .. } => 4,
        CurveGeom::Parabola { .. } => 5,
        CurveGeom::BezierCurve { .. } => 6,
        CurveGeom::BSpline { .. } | CurveGeom::Polyline { .. } => 7,
        CurveGeom::Offset { .. } => 8,
        CurveGeom::Trimmed { basis, .. } => occ_curve_type(basis),
        CurveGeom::Composite { .. } => 7,
    }
}


fn nurbs_is_rational(weights: &[Vec<Real>]) -> bool {
    weights.iter().any(|row| row.iter().any(|&w| (w - 1.0).abs() > 1e-6))
}

/// Expand an offset surface to its actual geometry and write it.
fn write_expanded_offset_surface(
    output: &mut impl Write,
    basis: &SurfaceGeom,
    distance: Real,
) -> io::Result<()> {
    match basis {
        SurfaceGeom::Plane { origin, normal, u_dir } => {
            let n = if normal.length_squared() > 1e-12 {
                *normal / normal.length()
            } else {
                *normal
            };
            let new_origin = *origin + n * distance;
            let u = if u_dir.length_squared() > 1e-12 {
                *u_dir / u_dir.length()
            } else {
                *u_dir
            };
            let v = n.cross(u);
            writeln!(output, "1 {} {} {} {} {} {} {} {} {} {} {} {}",
                new_origin.x, new_origin.y, new_origin.z,
                n.x, n.y, n.z,
                u.x, u.y, u.z,
                v.x, v.y, v.z)?;
        }
        SurfaceGeom::Cylinder { origin, axis, radius, x_dir, y_dir } => {
            let new_radius = radius + distance;
            writeln!(output, "2 {} {} {} {} {} {} {} {} {} {} {} {} {}",
                origin.x, origin.y, origin.z,
                axis.x, axis.y, axis.z,
                new_radius,
                x_dir.x, x_dir.y, x_dir.z,
                y_dir.x, y_dir.y, y_dir.z)?;
        }
        SurfaceGeom::Sphere { center, radius } => {
            let n = PVec3::Z;
            let u = PVec3::X;
            let v = PVec3::Y;
            writeln!(output, "4 {} {} {} {} {} {} {} {} {} {} {} {} {}",
                center.x, center.y, center.z,
                n.x, n.y, n.z,
                u.x, u.y, u.z,
                v.x, v.y, v.z,
                radius + distance)?;
        }
        _ => {
            // Fallback: write basis surface as-is, offset info lost
            writeln!(output, "1 0 0 0  0 0 1  1 0 0  0 1 0")?;
        }
    }
    Ok(())
}
/// Estimate the center of a shell by averaging all vertex positions of its faces.
fn shell_center(faces: &[(FaceKey, Orientation)], store: &BRepStore) -> PVec3 {
    let mut sum = PVec3::ZERO;
    let mut count = 0usize;
    for &(fk, _) in faces {
        if let Some(face) = store.faces.get(fk) {
            if let Some(wire) = store.wires.get(face.outer_wire) {
                for &(ek, _) in &wire.edges {
                    if let Some(edge) = store.edges.get(ek) {
                        if let Some(v) = store.vertices.get(edge.v_low) {
                            sum = sum + v.position;
                            count += 1;
                        }
                        if let Some(v) = store.vertices.get(edge.v_high) {
                            sum = sum + v.position;
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    if count > 0 { sum / count as Real } else { PVec3::ZERO }
}

/// Check whether the face normal (accounting for same_sense) points away
/// from the given reference point (typically the solid center).
fn face_normal_points_outward(fk: FaceKey, ref_point: PVec3, store: &BRepStore) -> bool {
    if let Some(face) = store.faces.get(fk) {
        // Sample a point near the face center from the outer wire
        if let Some(wire) = store.wires.get(face.outer_wire) {
            if let Some(&(ek, _)) = wire.edges.first() {
                if let Some(edge) = store.edges.get(ek) {
                    let mid_t = (edge.t_min + edge.t_max) * 0.5;
                    let p3d = edge.curve.d0(mid_t);
                    // Get surface normal at the projected UV of this point
                    let uv = if let Some(pc) = edge.pcurves.get(&fk) {
                        pc.d0(mid_t)
                    } else {
                        return true; // no PCurve, assume outward
                    };
                    let (su, sv) = face.surface.d1_native(uv.0, uv.1);
                    let n = su.cross(sv);
                    let n = if n.length_squared() > 1e-12 { n.normalize() } else { n };
                    // Use surface normal (not face normal with same_sense).
                    // The shell orientation sign encodes the face flip for outward pointing.
                    let to_center = ref_point - p3d;
                    return n.dot(to_center) < 0.0;
                }
            }
        }
    }
    true // default: assume outward
}

fn expand_curve(curve: &CurveGeom) -> &CurveGeom {
    match curve {
        CurveGeom::Trimmed { basis, .. } => basis.as_ref(),
        other => other,
    }
}

/// Write a 2D polyline as a 2D BSpline degree 1.
/// Return the effective OCC curve2d type after applying the same simplification
/// as write_curve2ds. Used by write_ed for consistent PCurve reference types.
fn occ_pcurve_type(curve: &Curve2d) -> u8 {
    match curve {
        Curve2d::Line { .. } => 1,
        Curve2d::Circle { .. } => 2,
        Curve2d::Ellipse { .. } => 3,
        Curve2d::BSpline { .. } => 7,
        Curve2d::Polyline { points } => {
            // Polyline simplifies to Line if collinear, else BSpline
            if crate::geom::curve2d::simplify_polyline_to_line(points).is_some() { 1 } else { 7 }
        }
        Curve2d::Trimmed { basis, .. } => occ_pcurve_type(basis),
        Curve2d::Composite { .. } => 7,
    }
}

/// Write a single PCurve reference line in OCC V1 format, consistent with
/// the simplified form used by write_curve2ds.
fn write_pcurve_ref(output: &mut impl Write, pc_idx: &usize, curve: &Curve2d) -> io::Result<()> {
    match occ_pcurve_type(curve) {
        1 => {
            // Line pcurve: type 1, idx, orient=0, t0, t1
            // Approximate t1 from stored curve direction/length
            let (t0, t1) = match curve {
                Curve2d::Line { direction, .. } => {
                    let len = (direction.0 * direction.0 + direction.1 * direction.1).sqrt();
                    (0.0, len)
                }
                Curve2d::Polyline { points } => {
                    // Simplified to line: use first-to-last length as approx
                    if points.len() >= 2 {
                        let dx = points[points.len()-1].0 - points[0].0;
                        let dy = points[points.len()-1].1 - points[0].1;
                        (0.0, (dx * dx + dy * dy).sqrt())
                    } else { (0.0, 1.0) }
                }
                _ => (0.0, 1.0),
            };
            writeln!(output, "1 {} 0 {} {}", pc_idx, rnd(t0), rnd(t1))
        }
        2 => {
            // Circle pcurve: 2 idx 1 0 0 t1
            let t1 = match curve {
                Curve2d::Circle { radius, .. } => std::f64::consts::TAU * radius.max(1e-6),
                Curve2d::Polyline { points } => {
                    if points.len() >= 2 {
                        let dx = points[points.len()-1].0 - points[0].0;
                        let dy = points[points.len()-1].1 - points[0].1;
                        (dx * dx + dy * dy).sqrt()
                    } else { 1.0 }
                }
                _ => 1.0,
            };
            writeln!(output, "2 {} 1 0 0 {}", pc_idx, rnd(t1))
        }
        _ => {
            // BSpline/other: 7 idx num_poles degree flags
            let (n_poles, deg) = match curve {
                Curve2d::BSpline { degree, control_points, .. } => (control_points.len(), *degree),
                Curve2d::Polyline { points } => (points.len(), 1),
                _ => (2, 1),
            };
            writeln!(output, "7 {} {} {} 0", pc_idx, n_poles, deg)
        }
    }
}

fn write_polyline2d_as_bspline(output: &mut impl Write, points: &[(Real, Real)]) -> io::Result<()> {
    if points.len() < 2 {
        return writeln!(output, "1 0 1 1 0");
    }
    // OCC-compatible: degree-1 2-pole BSpline. num_knots=1 means
    // implicit knot vector; OCC auto-computes multiplicities.
    let first = points[0];
    let last = points[points.len() - 1];
    writeln!(output, "7 1 2 1 0 {} {} {} {}",
        rnd(first.0), rnd(first.1), rnd(last.0), rnd(last.1))
}

fn write_polyline_as_bspline(output: &mut impl Write, points: &[PVec3]) -> io::Result<()> {
    if points.len() < 2 {
        return writeln!(output, "1 0 0 0  1 0 0");
    }
    let n = points.len();
    let knot_len = n + 2;
    write!(output, "7 1  {}  {}  0", n, knot_len)?;
    for p in points {
        write!(output, "  {} {} {}", p.x, p.y, p.z)?;
    }
    write!(output, " 0 0")?;
    for i in 1..n-1 { write!(output, " {}", i)?; }
    writeln!(output, " {} {}", n-1, n-1)?;
    Ok(())
}
