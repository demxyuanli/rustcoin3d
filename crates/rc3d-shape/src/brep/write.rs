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
use rc3d_core::math::Vec3;
use crate::geom::curve2d::Curve2d;
use crate::geom::{CurveGeom, SurfaceGeom};
use crate::store::BRepStore;
use crate::topo::*;

/// Write the full BRepStore as OCC BREP ASCII (classic format).
pub fn write_brep(store: &BRepStore, output: &mut impl Write) -> io::Result<()> {
    let mut w = BrepWriter::new(store);
    w.write_all(output)
}

struct BrepWriter<'a> {
    store: &'a BRepStore,
    shapes: Vec<ShapeEntry>,
    total_shapes: usize,
    needs_default_compound: bool,
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
        let has_compounds = store.compounds.len() > 0;
        for (ck, _) in &store.compounds {
            shapes.push(ShapeEntry::Compound(ck));
        }
        // If no compounds exist but we have solids, create a single default compound
        // referencing all solids (matching OCC convention).
        let needs_default_compound = !has_compounds && store.solids.len() > 0;

        let total = shapes.len() + if needs_default_compound { 1 } else { 0 };
        Self { store, shapes, total_shapes: total, needs_default_compound }
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

    fn compound_pos(&self, ck: CompoundKey) -> usize {
        self.shapes.iter().position(|s| matches!(s, ShapeEntry::Compound(k) if *k == ck))
            .map(|p| p + 1).unwrap_or(0)
    }

    // ── Sections ──────────────────────────────────────────────────

    fn write_locations(&self, output: &mut impl Write) -> io::Result<()> {
        writeln!(output, "Locations 0")?;
        Ok(())
    }

    /// Curve2ds — always 0 in classic format (PCurves stored on edges inline).
    fn write_curve2ds(&self, _output: &mut impl Write) -> io::Result<()> {
        // Classic format embeds PCurves in edge data, so we write 0 here.
        // We'll re-visit this when PCurve support is complete.
        writeln!(_output, "Curve2ds 0")?;
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
                    let dir = if len > 1e-12 { *direction / len } else { *direction };
                    // OCC format: 1 ox oy oz dx dy dz  (unit direction)
                    writeln!(output, "1 {} {} {} {} {} {}",
                        origin.x, origin.y, origin.z,
                        dir.x, dir.y, dir.z)?;
                }
                CurveGeom::Circle { center, axis, radius, x_dir, y_dir } => {
                    writeln!(output, "2 {} {} {}  {} {} {}  {}  {} {} {}  {} {} {}",
                        center.x, center.y, center.z,
                        axis.x, axis.y, axis.z, radius,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z)?;
                }
                CurveGeom::Ellipse { center, axis, semi_major, semi_minor, x_dir, y_dir } => {
                    writeln!(output, "3 {} {} {}  {} {} {}  {} {}  {} {} {}  {} {} {}",
                        center.x, center.y, center.z,
                        axis.x, axis.y, axis.z,
                        semi_major, semi_minor,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z)?;
                }
                CurveGeom::Hyperbola { center, axis, semi_major, semi_minor, x_dir, y_dir } => {
                    writeln!(output, "4 {} {} {}  {} {} {}  {} {}  {} {} {}  {} {} {}",
                        center.x, center.y, center.z,
                        axis.x, axis.y, axis.z,
                        semi_major, semi_minor,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z)?;
                }
                CurveGeom::Parabola { center, axis, focal_dist, x_dir, y_dir } => {
                    writeln!(output, "5 {} {} {}  {} {} {}  {}  {} {} {}  {} {} {}",
                        center.x, center.y, center.z,
                        axis.x, axis.y, axis.z,
                        focal_dist,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z)?;
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
                    write!(output, "7 {}  {}  {}  {}",
                        degree, control_points.len(), knots.len(),
                        if weights.is_some() { 1 } else { 0 })?;
                    for cp in control_points {
                        write!(output, "  {} {} {}", cp.x, cp.y, cp.z)?;
                    }
                    for k in knots { write!(output, " {}", k)?; }
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
                    // Expand: write inner basis
                    match basis.as_ref() {
                        CurveGeom::Line { origin, direction } => {
                            let len = direction.length();
                            let dir = if len > 1e-12 { *direction / len } else { *direction };
                            writeln!(output, "1 {} {} {} {} {} {}",
                                origin.x, origin.y, origin.z,
                                dir.x, dir.y, dir.z)?;
                        }
                        _ => {
                            // Fallback
                            writeln!(output, "1 0 0 0  1 0 0")?;
                        }
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
                        origin.x, origin.y, origin.z,
                        n.x, n.y, n.z,
                        u.x, u.y, u.z,
                        v.x, v.y, v.z)?;
                }
                SurfaceGeom::Cylinder { origin, axis, radius, x_dir, y_dir } => {
                    writeln!(output, "2 {} {} {} {} {} {} {} {} {} {} {} {} {}",
                        origin.x, origin.y, origin.z,
                        axis.x, axis.y, axis.z,
                        radius,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z)?;
                }
                SurfaceGeom::Cone { apex, axis, semi_angle, radius_at_apex, x_dir, y_dir } => {
                    writeln!(output, "3 {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                        apex.x, apex.y, apex.z,
                        axis.x, axis.y, axis.z,
                        semi_angle, radius_at_apex,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z)?;
                }
                SurfaceGeom::Sphere { center, radius } => {
                    writeln!(output, "4 {} {} {}  {}", center.x, center.y, center.z, radius)?;
                }
                SurfaceGeom::Torus { center, axis, major_r, minor_r, x_dir, y_dir } => {
                    writeln!(output, "5 {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                        center.x, center.y, center.z,
                        axis.x, axis.y, axis.z,
                        major_r, minor_r,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z)?;
                }
                SurfaceGeom::BSpline(ns) => {
                    writeln!(output, "8 {} {} {} {} {} {} {} {} {}",
                        ns.degree_u, ns.degree_v,
                        ns.u_count(), ns.v_count(),
                        ns.knots_u.len(), ns.knots_v.len(),
                        if nurbs_is_rational(&ns.weights) { 1 } else { 0 },
                        0, 0)?;
                    for row in &ns.control_points {
                        for cp in row {
                            writeln!(output, "{} {} {}", cp.x, cp.y, cp.z)?;
                        }
                    }
                    for k in &ns.knots_u { write!(output, " {}", k)?; }
                    writeln!(output)?;
                    for k in &ns.knots_v { write!(output, " {}", k)?; }
                    writeln!(output)?;
                }
                SurfaceGeom::Extrusion { direction, .. } => {
                    writeln!(output, "6 {} {} {}  0 0 0  0 0 1  0 1 0",
                        direction.x, direction.y, direction.z)?;
                }
                SurfaceGeom::Revolution { axis_origin, axis_dir, .. } => {
                    let (x_dir, y_dir) = crate::geom::build_ortho_axes(*axis_dir);
                    writeln!(output, "7 {} {} {} {} {} {} {} {} {} {} {} {}",
                        axis_origin.x, axis_origin.y, axis_origin.z,
                        axis_dir.x, axis_dir.y, axis_dir.z,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z)?;
                }
                SurfaceGeom::Offset { distance, .. } => {
                    writeln!(output, "9 {}", distance)?;
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

        // Write default compound if none existed (wraps all solids)
        if self.needs_default_compound {
            let solid_keys: Vec<SolidKey> = self.store.solids.keys().collect();
            self.write_co(output, &solid_keys)?;
        }

        // Final references line: +1 0
        writeln!(output)?;
        writeln!(output, "+1 0")?;
        Ok(())
    }

    fn write_ve(&self, output: &mut impl Write, vk: VertexKey) -> io::Result<()> {
        let v = self.store.vertices.get(vk).map(|v| (v.position, v.tolerance.max(1e-7)))
            .unwrap_or((Vec3::ZERO, 1e-7));
        writeln!(output, "Ve")?;
        writeln!(output, "{}", v.1)?;
        writeln!(output, "{} {} {}", v.0.x, v.0.y, v.0.z)?;
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

        let curve_len = edge.curve.d0(edge.t_max).distance(edge.curve.d0(edge.t_min));
        let param_range = if curve_len > 1e-12 { curve_len } else { 1.0 };

        let curve_type = occ_curve_type(&expand_curve(&edge.curve));
        let v1_pos = self.vertex_pos(edge.v_low);
        let v2_pos = self.vertex_pos(edge.v_high);
        let rv1 = self.rev_idx(v1_pos);
        let rv2 = self.rev_idx(v2_pos);

        let tol = edge.tolerance.max(1e-7);
        writeln!(output, "Ed")?;
        writeln!(output, " {} {} 1 0", tol, 1)?;
        // curve_type curve_idx 0 0 param_range
        writeln!(output, "{}  {} 0 0 {}", curve_type, curve_idx, param_range)?;
        writeln!(output, "0")?;
        writeln!(output)?;
        writeln!(output, "0101000")?;
        writeln!(output, "+{} 0 -{} 0 *", rv1, rv2)?;
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
        // Face format: location_index  tolerance  surface_index(1-based)  orientation
        writeln!(output, "0  {}  {} 1", face.tolerance, surf_idx)?;
        writeln!(output)?;
        writeln!(output, "0101000")?;
        // Wire reference (reverse-indexed)
        let rwp = self.rev_idx(wp);
        writeln!(output, "+{} 0 *", rwp)?;
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
        for (fk, orient) in &shell.faces {
            let fp = self.face_pos(*fk);
            let rp = self.rev_idx(fp);
            match orient {
                Orientation::Forward => write!(output, "+{} 0 ", rp)?,
                Orientation::Reversed => write!(output, "-{} 0 ", rp)?,
                _ => write!(output, "+{} 0 ", rp)?,
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

fn nurbs_is_rational(weights: &[Vec<f32>]) -> bool {
    weights.iter().any(|row| row.iter().any(|&w| (w - 1.0).abs() > 1e-6))
}

fn expand_curve(curve: &CurveGeom) -> &CurveGeom {
    match curve {
        CurveGeom::Trimmed { basis, .. } => basis.as_ref(),
        other => other,
    }
}

fn write_polyline_as_bspline(output: &mut impl Write, points: &[Vec3]) -> io::Result<()> {
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
    for i in 1..n-1 {
        write!(output, " {}", i)?;
    }
    write!(output, " {} {}", n-1, n-1)?;
    writeln!(output)?;
    Ok(())
}
