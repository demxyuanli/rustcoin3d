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

struct PCurveEntry {
    edge_abs_pos: usize,  // 1-based edge position in TShapes
    face_abs_pos: usize,  // 1-based face position in TShapes
    curve: Curve2d,
    same_sense: bool,
}

struct BrepWriter<'a> {
    store: &'a BRepStore,
    shapes: Vec<ShapeEntry>,
    total_shapes: usize,
    needs_default_compound: bool,
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
        let has_compounds = store.compounds.len() > 0;
        for (ck, _) in &store.compounds {
            shapes.push(ShapeEntry::Compound(ck));
        }
        // If no compounds exist but we have solids, create a single default compound
        // referencing all solids (matching OCC convention).
        let needs_default_compound = !has_compounds && store.solids.len() > 0;

        // Pre-collect PCurve entries only for non-planar faces (planar faces don't need them)
        let mut pcurve_entries = Vec::new();
        for (i, entry) in shapes.iter().enumerate() {
            if let ShapeEntry::Edge(ek) = entry {
                if let Some(edge) = store.edges.get(*ek) {
                    for (fk, (pc, same_sense)) in &edge.pcurves {
                        // Only include PCurves for non-planar faces (expand Offset to check)
                        let is_non_planar = store.faces.get(*fk)
                            .map(|f| !is_planar_surface(&f.surface))
                            .unwrap_or(false);
                        if is_non_planar {
                            let face_pos = shapes.iter()
                                .position(|s| matches!(s, ShapeEntry::Face(k) if k == fk))
                                .map(|p| p + 1).unwrap_or(0);
                            pcurve_entries.push(PCurveEntry {
                                edge_abs_pos: i + 1,
                                face_abs_pos: face_pos,
                                curve: pc.clone(),
                                same_sense: *same_sense,
                            });
                        }
                    }
                }
            }
        }
        let pcurve_count = pcurve_entries.len();

        let total = shapes.len() + if needs_default_compound { 1 } else { 0 };
        Self { store, shapes, total_shapes: total, needs_default_compound, pcurve_entries, pcurve_count }
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

    /// Write PCurves section with actual 2D curve data.
    fn write_curve2ds(&self, output: &mut impl Write) -> io::Result<()> {
        writeln!(output, "Curve2ds {}", self.pcurve_count)?;
        // Count edges before this one to compute curve_index
        let mut edge_count = 0usize;
        let mut edge_to_curve: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for entry in &self.shapes {
            if matches!(entry, ShapeEntry::Edge(_)) {
                edge_count += 1;
                // Map edge TShape position -> curve index
                // We need to find positions of edges...
            }
        }
        // Build edge TShape position -> curve index map
        let mut curve_idx_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut ci = 0usize;
        for (pos, entry) in self.shapes.iter().enumerate() {
            if matches!(entry, ShapeEntry::Edge(_)) {
                ci += 1;
                curve_idx_map.insert(pos + 1, ci);
            }
        }
        for entry in &self.pcurve_entries {
            let ori = if entry.same_sense { 1 } else { 0 };
            let curve_idx = curve_idx_map.get(&entry.edge_abs_pos).copied().unwrap_or(0);
            write!(output, "{} {} {} ", curve_idx, entry.face_abs_pos, ori)?;
            match &entry.curve {
                Curve2d::Line { origin, direction } => {
                    writeln!(output, "1 {} {} {} {}", origin.0, origin.1, direction.0, direction.1)?;
                }
                Curve2d::Circle { center, radius } => {
                    writeln!(output, "2 {} {} {}", center.0, center.1, radius)?;
                }
                Curve2d::Ellipse { center, semi_major, semi_minor } => {
                    writeln!(output, "3 {} {} {} {}", center.0, center.1, semi_major, semi_minor)?;
                }
                Curve2d::BSpline { degree, control_points, knots, weights } => {
                    write!(output, "7 {} {} {} {}", degree, control_points.len(), knots.len(),
                        if weights.is_some() { 1 } else { 0 })?;
                    for cp in control_points { write!(output, " {} {}", cp.0, cp.1)?; }
                    for k in knots { write!(output, " {}", k)?; }
                    if let Some(w) = weights { for wt in w { write!(output, " {}", wt)?; } }
                    writeln!(output)?;
                }
                Curve2d::Trimmed { basis, t_min, t_max } => {
                    match basis.as_ref() {
                        Curve2d::Line { origin, direction } => {
                            writeln!(output, "1 {} {} {} {}", origin.0, origin.1, direction.0, direction.1)?;
                        }
                        Curve2d::Circle { center, radius } => {
                            writeln!(output, "2 {} {} {}", center.0, center.1, radius)?;
                        }
                        _ => writeln!(output, "1 0 0 1 0")?,
                    }
                }
                _ => writeln!(output, "1 0 0 1 0")?,
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
                    let dir = if len > 1e-12 { *direction / len } else { Vec3::X };
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
                    let rational = nurbs_is_rational(&ns.weights);
                    writeln!(output, "8 {} {} {} {} {} {} {} {} {}",
                        ns.degree_u, ns.degree_v,
                        ns.u_count(), ns.v_count(),
                        ns.knots_u.len(), ns.knots_v.len(),
                        if rational { 1 } else { 0 },
                        0, 0)?;
                    // Control points: include weight inline if rational
                    if rational {
                        for (i, row) in ns.control_points.iter().enumerate() {
                            for (j, cp) in row.iter().enumerate() {
                                let w = ns.weights.get(i).and_then(|rw| rw.get(j)).copied().unwrap_or(1.0);
                                writeln!(output, "{} {} {} {}", cp.x, cp.y, cp.z, w)?;
                            }
                        }
                    } else {
                        for row in &ns.control_points {
                            for cp in row {
                                writeln!(output, "{} {} {}", cp.x, cp.y, cp.z)?;
                            }
                        }
                    }
                    for k in &ns.knots_u { write!(output, " {}", k)?; }
                    writeln!(output)?;
                    for k in &ns.knots_v { write!(output, " {}", k)?; }
                    writeln!(output)?;
                }
                SurfaceGeom::Extrusion { generatrix, direction } => {
                    // Write as-is with direction; generatrix is handled by OCC internally
                    let base_pt = generatrix.d0(0.0);
                    writeln!(output, "6 {} {} {}  {} {} {}",
                        direction.x, direction.y, direction.z,
                        base_pt.x, base_pt.y, base_pt.z)?;
                }
                SurfaceGeom::Revolution { axis_origin, axis_dir, .. } => {
                    let (x_dir, y_dir) = crate::geom::build_ortho_axes(*axis_dir);
                    writeln!(output, "7 {} {} {} {} {} {} {} {} {} {} {} {}",
                        axis_origin.x, axis_origin.y, axis_origin.z,
                        axis_dir.x, axis_dir.y, axis_dir.z,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z)?;
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

        let param_range = curve_param_range(&expand_curve(&edge.curve), edge.t_min, edge.t_max);

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

        // PCurve references for this edge (use curve index, not edge position)
        let edge_abs = self.edge_pos(ek);
        let curve_idx_for_edge = curve_idx;
        if self.pcurve_count == 0 {
            writeln!(output, "0")?;
        } else {
            let mut pc_indices: Vec<usize> = Vec::new();
            for (idx, entry) in self.pcurve_entries.iter().enumerate() {
                if entry.edge_abs_pos == edge_abs {
                    pc_indices.push(idx + 1); // 1-based
                }
            }
            if pc_indices.is_empty() {
                writeln!(output, "0")?;
            } else {
                write!(output, "{} C0", pc_indices.len())?;
                for pi in &pc_indices {
                    write!(output, " {} 0", pi)?;
                }
                writeln!(output)?;
            }
        }
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

/// Compute the natural parameter range for a curve between t_min and t_max.
/// OCC uses the curve's natural parameterization, not chord distance.
fn curve_param_range(curve: &CurveGeom, t_min: f32, t_max: f32) -> f32 {
    let span = (t_max - t_min).abs();
    if span < 1e-12 { return 1.0; }
    match curve {
        CurveGeom::Line { direction, .. } => {
            let len = direction.length();
            if len > 1e-12 { len * span } else { 1.0 }
        }
        CurveGeom::Circle { radius, .. } => {
            std::f32::consts::TAU * radius * span
        }
        CurveGeom::Ellipse { semi_major, semi_minor, .. } => {
            // Approximate with Ramanujan's formula for perimeter
            let a = semi_major.max(*semi_minor);
            let b = semi_minor.min(*semi_major);
            let h = ((a - b) / (a + b)).powi(2);
            let perimeter = std::f32::consts::PI * (a + b) * (1.0 + 3.0 * h / (10.0 + (4.0 - 3.0 * h).sqrt()));
            perimeter * span
        }
        CurveGeom::BSpline { control_points, knots, degree, .. } => {
            // Approximate: chord length of control polygon
            let mut total = 0.0f32;
            for w in control_points.windows(2) {
                total += w[0].distance(w[1]);
            }
            let domain_start = knots.get(*degree).copied().unwrap_or(0.0);
            let domain_end = knots.get(control_points.len()).copied().unwrap_or(1.0);
            let domain = (domain_end - domain_start).abs().max(1e-6);
            total * span / domain
        }
        CurveGeom::Polyline { points } => {
            let mut total = 0.0f32;
            for w in points.windows(2) {
                total += w[0].distance(w[1]);
            }
            total.max(1.0)
        }
        _ => {
            // Fallback: chord distance
            let d = curve.d0(t_max).distance(curve.d0(t_min));
            if d > 1e-12 { d } else { 1.0 }
        }
    }
}

fn nurbs_is_rational(weights: &[Vec<f32>]) -> bool {
    weights.iter().any(|row| row.iter().any(|&w| (w - 1.0).abs() > 1e-6))
}

/// Expand an offset surface to its actual geometry and write it.
fn write_expanded_offset_surface(
    output: &mut impl Write,
    basis: &SurfaceGeom,
    distance: f32,
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
            writeln!(output, "4 {} {} {}  {}",
                center.x, center.y, center.z,
                radius + distance)?;
        }
        _ => {
            // Fallback: write basis surface as-is, offset info lost
            writeln!(output, "1 0 0 0  0 0 1  1 0 0  0 1 0")?;
        }
    }
    Ok(())
}

/// Sample generatrix curve and write extrusion as BSpline surface.
fn write_extrusion_as_bspline(
    output: &mut impl Write,
    generatrix: &CurveGeom,
    direction: Vec3,
) -> io::Result<()> {
    // Sample the generatrix at N points
    let n = 8usize;
    let mut pts: Vec<Vec3> = (0..=n).map(|i| {
        generatrix.d0(i as f32 / n as f32)
    }).collect();
    // Top row = generatrix + direction
    let top: Vec<Vec3> = pts.iter().map(|p| *p + direction).collect();
    pts.extend(top);

    let u_count = 2usize; // 2 rows (bottom, top)
    let v_count = n + 1;  // control points per row
    let degree_u = 1usize;
    let degree_v = 1usize;
    let knots_u = vec![0.0, 0.0, 1.0, 1.0];
    let knots_v: Vec<f32> = {
        let mut k = vec![0.0, 0.0];
        for i in 1..v_count-1 { k.push(i as f32); }
        k.push((v_count - 1) as f32);
        k.push((v_count - 1) as f32);
        k
    };

    writeln!(output, "8 {} {} {} {} {} {} 0 0 0",
        degree_u, degree_v, u_count, v_count,
        knots_u.len(), knots_v.len())?;
    for p in &pts {
        writeln!(output, "{} {} {}", p.x, p.y, p.z)?;
    }
    for k in &knots_u { write!(output, " {}", k)?; }
    writeln!(output)?;
    for k in &knots_v { write!(output, " {}", k)?; }
    writeln!(output)?;
    Ok(())
}

/// Sample generatrix and revolve around axis to write as BSpline surface.
fn write_revolution_as_bspline(
    output: &mut impl Write,
    generatrix: &CurveGeom,
    axis_origin: Vec3,
    axis_dir: Vec3,
) -> io::Result<()> {
    let n_u = 16usize;
    let n_v = 8usize;

    let gen_pts: Vec<Vec3> = (0..=n_v).map(|i| {
        generatrix.d0(i as f32 / n_v as f32)
    }).collect();

    let axis = axis_dir.normalize();
    let mut all_pts = Vec::new();
    for i in 0..=n_u {
        let angle = (i as f32 / n_u as f32) * std::f32::consts::TAU;
        for p in &gen_pts {
            // Rodrigues rotation: p' = origin + R * (p - origin)
            let rel = *p - axis_origin;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let rotated = rel * cos_a + axis.cross(rel) * sin_a + axis * axis.dot(rel) * (1.0 - cos_a);
            all_pts.push(axis_origin + rotated);
        }
    }

    let u_count = n_u + 1;
    let v_count = n_v + 1;
    let degree_u = 2usize;
    let degree_v = 1usize;
    let ku_count = u_count + degree_u + 1;
    let kv_count = v_count + degree_v + 1;

    writeln!(output, "8 {} {} {} {} {} {} 0 0 0",
        degree_u, degree_v, u_count, v_count, ku_count, kv_count)?;
    for p in &all_pts {
        writeln!(output, "{} {} {}", p.x, p.y, p.z)?;
    }
    // u-knots (periodic-like for full revolution)
    for i in 0..ku_count {
        write!(output, " {}", i as f32)?;
    }
    writeln!(output)?;
    // v-knots (clamped)
    for i in 0..kv_count {
        if i <= degree_v { write!(output, " 0")?; }
        else if i >= kv_count - degree_v - 1 { write!(output, " {}", (v_count - degree_v) as f32)?; }
        else { write!(output, " {}", (i - degree_v) as f32)?; }
    }
    writeln!(output)?;
    Ok(())
}

fn is_planar_surface(surface: &SurfaceGeom) -> bool {
    match surface {
        SurfaceGeom::Plane { .. } => true,
        SurfaceGeom::Offset { basis, .. } => is_planar_surface(basis),
        _ => false,
    }
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
    for i in 1..n-1 { write!(output, " {}", i)?; }
    writeln!(output, " {} {}", n-1, n-1)?;
    Ok(())
}
