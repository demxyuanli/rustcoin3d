//! OCC BREP ASCII format writer.
//! Format: Open CASCADE Technology BRepTools::Write() / DBRep_Write.
//! Sections appear in strict order -- out-of-order sections crash OCC's reader.

use std::collections::HashMap;
use std::io::{self, Write};
use rc3d_core::math::Vec3;
use crate::geom::curve2d::Curve2d;
use crate::geom::{CurveGeom, SurfaceGeom};
use crate::store::BRepStore;
use crate::topo::*;

/// Write the full BRepStore as OCC BREP ASCII.
pub fn write_brep(store: &BRepStore, output: &mut impl Write) -> io::Result<()> {
    let mut w = BrepWriter::new(store);
    w.write_all(output)
}

struct BrepWriter<'a> {
    store: &'a BRepStore,
    curve_indices: HashMap<EdgeKey, usize>,
    surface_indices: HashMap<FaceKey, usize>,
    curve_count: usize,
    surface_count: usize,
}

impl<'a> BrepWriter<'a> {
    fn new(store: &'a BRepStore) -> Self {
        Self {
            store,
            curve_indices: HashMap::new(),
            surface_indices: HashMap::new(),
            curve_count: 0,
            surface_count: 0,
        }
    }

    fn write_all(&mut self, output: &mut impl Write) -> io::Result<()> {
        writeln!(output, "DBRep_DrawableShape")?;
        writeln!(output)?;
        self.write_locations(output)?;
        self.write_curves3d(output)?;
        self.write_surfaces(output)?;
        self.write_pcurves(output)?;
        self.write_vertices(output)?;
        self.write_edges(output)?;
        self.write_wires(output)?;
        self.write_faces(output)?;
        self.write_shells(output)?;
        self.write_solids(output)?;
        self.write_compounds(output)?;
        Ok(())
    }

    // ── Location section ─────────────────────────────────────────

    fn write_locations(&self, output: &mut impl Write) -> io::Result<()> {
        writeln!(output, "Locations 0")?;
        writeln!(output)?;
        Ok(())
    }

    // ── Curve3D section ──────────────────────────────────────────

    fn write_curves3d(&mut self, output: &mut impl Write) -> io::Result<()> {
        let mut curves: Vec<(usize, &CurveGeom)> = Vec::new();
        for (ek, edge) in &self.store.edges {
            if !self.curve_indices.contains_key(&ek) {
                self.curve_count += 1;
                self.curve_indices.insert(ek, self.curve_count);
                curves.push((self.curve_count, &edge.curve));
            }
        }
        writeln!(output, "Curve3ds {}", self.curve_count)?;

        for (_ci, curve) in &curves {
            let (expanded, _trim) = expand_curve(curve);
            match expanded {
                CurveGeom::Line { origin, direction } => {
                    writeln!(
                        output, "1 {} {} {}  {} {} {}",
                        origin.x, origin.y, origin.z,
                        direction.x, direction.y, direction.z
                    )?;
                }
                CurveGeom::Circle { center, axis, radius, x_dir, y_dir } => {
                    writeln!(
                        output,
                        "2 {} {} {}  {} {} {}  {}  {} {} {}  {} {} {}",
                        center.x, center.y, center.z,
                        axis.x, axis.y, axis.z,
                        radius,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z
                    )?;
                }
                CurveGeom::Ellipse { center, axis, semi_major, semi_minor, x_dir, y_dir } => {
                    writeln!(
                        output,
                        "3 {} {} {}  {} {} {}  {} {}  {} {} {}  {} {} {}",
                        center.x, center.y, center.z,
                        axis.x, axis.y, axis.z,
                        semi_major, semi_minor,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z
                    )?;
                }
                CurveGeom::Hyperbola { center, axis, semi_major, semi_minor, x_dir, y_dir } => {
                    writeln!(
                        output,
                        "4 {} {} {}  {} {} {}  {} {}  {} {} {}  {} {} {}",
                        center.x, center.y, center.z,
                        axis.x, axis.y, axis.z,
                        semi_major, semi_minor,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z
                    )?;
                }
                CurveGeom::Parabola { center, axis, focal_dist, x_dir, y_dir } => {
                    writeln!(
                        output,
                        "5 {} {} {}  {} {} {}  {}  {} {} {}  {} {} {}",
                        center.x, center.y, center.z,
                        axis.x, axis.y, axis.z,
                        focal_dist,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z
                    )?;
                }
                CurveGeom::BezierCurve { degree, control_points, weights } => {
                    write!(
                        output, "6 {}  {}  {}",
                        degree,
                        control_points.len(),
                        if weights.is_some() { 1 } else { 0 }
                    )?;
                    for cp in control_points {
                        write!(output, "  {} {} {}", cp.x, cp.y, cp.z)?;
                    }
                    if let Some(w) = weights {
                        for wt in w {
                            write!(output, " {}", wt)?;
                        }
                    }
                    writeln!(output)?;
                }
                CurveGeom::BSpline { degree, control_points, knots, weights } => {
                    write!(
                        output, "7 {}  {}  {}  {}",
                        degree,
                        control_points.len(),
                        knots.len(),
                        if weights.is_some() { 1 } else { 0 }
                    )?;
                    for cp in control_points {
                        write!(output, "  {} {} {}", cp.x, cp.y, cp.z)?;
                    }
                    for k in knots {
                        write!(output, " {}", k)?;
                    }
                    if let Some(w) = weights {
                        for wt in w {
                            write!(output, " {}", wt)?;
                        }
                    }
                    writeln!(output)?;
                }
                CurveGeom::Offset { basis, .. } => {
                    // Fallback: approximate offset curve as a polyline and write as BSpline
                    let pts: Vec<Vec3> = (0..=16).map(|i| {
                        let t = i as f32 / 16.0;
                        basis.d0(t)
                    }).collect();
                    write_polyline_as_bspline(output, &pts)?;
                }
                CurveGeom::Polyline { points } => {
                    write_polyline_as_bspline(output, points)?;
                }
                CurveGeom::Composite { segments, .. } => {
                    // Expand composite: concatenate points from all segments
                    let mut pts: Vec<Vec3> = Vec::new();
                    for (seg, reversed) in segments {
                        let seg_pts: Vec<Vec3> = (0..=8).map(|i| {
                            let t = i as f32 / 8.0;
                            if *reversed { seg.d0(1.0 - t) } else { seg.d0(t) }
                        }).collect();
                        if pts.is_empty() {
                            pts = seg_pts;
                        } else {
                            pts.extend(seg_pts.into_iter().skip(1));
                        }
                    }
                    write_polyline_as_bspline(output, &pts)?;
                }
                CurveGeom::Trimmed { .. } => {
                    // Should not reach here — expand_curve unwraps Trimmed
                    writeln!(output, "1 0 0 0  1 0 0")?;
                }
            }
        }
        writeln!(output)?;
        Ok(())
    }

    // ── Surface section ──────────────────────────────────────────

    fn write_surfaces(&mut self, output: &mut impl Write) -> io::Result<()> {
        let mut surfaces: Vec<(usize, &SurfaceGeom)> = Vec::new();
        for (fk, face) in &self.store.faces {
            if !self.surface_indices.contains_key(&fk) {
                self.surface_count += 1;
                self.surface_indices.insert(fk, self.surface_count);
                surfaces.push((self.surface_count, &face.surface));
            }
        }
        writeln!(output, "Surfaces {}", self.surface_count)?;

        for (_si, surface) in &surfaces {
            match surface {
                SurfaceGeom::Plane { origin, normal, u_dir } => {
                    writeln!(
                        output, "1 {} {} {}  {} {} {}  {} {} {}",
                        origin.x, origin.y, origin.z,
                        normal.x, normal.y, normal.z,
                        u_dir.x, u_dir.y, u_dir.z
                    )?;
                }
                SurfaceGeom::Cylinder { origin, axis, radius, x_dir, y_dir } => {
                    writeln!(
                        output,
                        "2 {} {} {}  {} {} {}  {}  {} {} {}  {} {} {}",
                        origin.x, origin.y, origin.z,
                        axis.x, axis.y, axis.z,
                        radius,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z
                    )?;
                }
                SurfaceGeom::Cone { apex, axis, semi_angle, radius_at_apex, x_dir, y_dir } => {
                    writeln!(
                        output,
                        "3 {} {} {}  {} {} {}  {} {}  {} {} {}  {} {} {}",
                        apex.x, apex.y, apex.z,
                        axis.x, axis.y, axis.z,
                        semi_angle, radius_at_apex,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z
                    )?;
                }
                SurfaceGeom::Sphere { center, radius } => {
                    writeln!(output, "4 {} {} {}  {}", center.x, center.y, center.z, radius)?;
                }
                SurfaceGeom::Torus { center, axis, major_r, minor_r, x_dir, y_dir } => {
                    writeln!(
                        output,
                        "5 {} {} {}  {} {} {}  {} {}  {} {} {}  {} {} {}",
                        center.x, center.y, center.z,
                        axis.x, axis.y, axis.z,
                        major_r, minor_r,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z
                    )?;
                }
                SurfaceGeom::BSpline(ns) => {
                    let rational = nurbs_is_rational(&ns.weights);
                    let periodic_u = if ns.is_u_closed(1e-3) { 1 } else { 0 };
                    let periodic_v = if ns.is_v_closed(1e-3) { 1 } else { 0 };
                    let u_count = ns.u_count();
                    let v_count = ns.v_count();
                    writeln!(
                        output,
                        "8 {} {} {} {} {} {} {} {} {}",
                        ns.degree_u,
                        ns.degree_v,
                        u_count,
                        v_count,
                        ns.knots_u.len(),
                        ns.knots_v.len(),
                        if rational { 1 } else { 0 },
                        periodic_u,
                        periodic_v,
                    )?;
                    // Control points: one per row (u-major order)
                    for row in &ns.control_points {
                        for cp in row {
                            if rational {
                                // Need corresponding weight
                                let w = 1.0;
                                writeln!(output, "{} {} {} {}", cp.x, cp.y, cp.z, w)?;
                            } else {
                                writeln!(output, "{} {} {}", cp.x, cp.y, cp.z)?;
                            }
                        }
                    }
                    if rational {
                        for row in &ns.weights {
                            for w in row {
                                write!(output, " {}", w)?;
                            }
                        }
                        writeln!(output)?;
                    }
                    // Knot values
                    for k in &ns.knots_u {
                        write!(output, " {}", k)?;
                    }
                    writeln!(output)?;
                    for k in &ns.knots_v {
                        write!(output, " {}", k)?;
                    }
                    writeln!(output)?;
                }
                SurfaceGeom::Extrusion { direction, .. } => {
                    writeln!(
                        output,
                        "6 {} {} {}  0 0 0  0 0 1  0 1 0",
                        direction.x, direction.y, direction.z
                    )?;
                }
                SurfaceGeom::Revolution { axis_origin, axis_dir, .. } => {
                    let (x_dir, y_dir) = crate::geom::build_ortho_axes(*axis_dir);
                    writeln!(
                        output,
                        "7 {} {} {}  {} {} {}  {} {} {}  {} {} {}",
                        axis_origin.x, axis_origin.y, axis_origin.z,
                        axis_dir.x, axis_dir.y, axis_dir.z,
                        x_dir.x, x_dir.y, x_dir.z,
                        y_dir.x, y_dir.y, y_dir.z
                    )?;
                }
                SurfaceGeom::Offset { distance, .. } => {
                    writeln!(output, "9 {}", distance)?;
                }
            }
        }
        writeln!(output)?;
        Ok(())
    }

    // ── PCurve (Curve2d) section ────────────────────────────────

    fn write_pcurves(&mut self, output: &mut impl Write) -> io::Result<()> {
        let mut entries: Vec<(usize, usize, &Curve2d, bool)> = Vec::new();
        for (ek, edge) in &self.store.edges {
            let ci = self.curve_indices.get(&ek).copied().unwrap_or(0);
            for (fk, (pc, same_sense)) in &edge.pcurves {
                let fi = self.surface_indices.get(&fk).copied().unwrap_or(0);
                entries.push((ci, fi, pc, *same_sense));
            }
        }
        writeln!(output, "Curve2ds {}", entries.len())?;

        for (ci, fi, pc, same_sense) in &entries {
            let orientation = if *same_sense { 1 } else { 0 };
            write!(output, "{} {} {}  ", ci, fi, orientation)?;
            match pc {
                Curve2d::Line { origin, direction } => {
                    writeln!(output, "1 {} {}  {} {}", origin.0, origin.1, direction.0, direction.1)?;
                }
                Curve2d::Circle { center, radius } => {
                    writeln!(output, "2 {} {}  {}", center.0, center.1, radius)?;
                }
                Curve2d::Ellipse { center, semi_major, semi_minor } => {
                    writeln!(output, "3 {} {}  {} {}", center.0, center.1, semi_major, semi_minor)?;
                }
                Curve2d::BSpline { degree, control_points, knots, weights } => {
                    write!(
                        output, "7 {}  {}  {}  {}",
                        degree,
                        control_points.len(),
                        knots.len(),
                        if weights.is_some() { 1 } else { 0 }
                    )?;
                    for cp in control_points {
                        write!(output, "  {} {}", cp.0, cp.1)?;
                    }
                    for k in knots {
                        write!(output, " {}", k)?;
                    }
                    if let Some(w) = weights {
                        for wt in w {
                            write!(output, " {}", wt)?;
                        }
                    }
                    writeln!(output)?;
                }
                Curve2d::Trimmed { basis, t_min, t_max } => {
                    // Expand trimmed: write inner basis curve data inline, no comments
                    match basis.as_ref() {
                        Curve2d::Line { origin, direction } => {
                            writeln!(
                                output, "1 {} {}  {} {}",
                                origin.0, origin.1, direction.0, direction.1
                            )?;
                        }
                        Curve2d::Circle { center, radius } => {
                            writeln!(
                                output, "2 {} {}  {}",
                                center.0, center.1, radius
                            )?;
                        }
                        other => {
                            // Fallback: write as 2D polyline via sampling
                            let pts: Vec<(f32, f32)> = (0..=8).map(|i| {
                                let t = *t_min + (*t_max - *t_min) * i as f32 / 8.0;
                                // Approximate: use d0 if available, else skip
                                let p = other.d0((i as f32) / 8.0);
                                p
                            }).collect();
                            write_polyline2d_as_bspline(output, &pts)?;
                        }
                    }
                }
                Curve2d::Polyline { points } => {
                    write_polyline2d_as_bspline(output, points)?;
                }
                Curve2d::Composite { .. } => {
                    // Write as degenerate: single-segment line at origin
                    writeln!(output, "1 0 0  1 0")?;
                }
            }
        }
        writeln!(output)?;
        Ok(())
    }

    // ── Topology sections ────────────────────────────────────────

    fn write_vertices(&self, output: &mut impl Write) -> io::Result<()> {
        let count = self.store.vertices.len();
        writeln!(output, "TVertexes {}", count)?;
        for (_vk, v) in &self.store.vertices {
            // Format: tolerance  location_index(0)  1(used)  X  Y  Z
            writeln!(
                output, "{}  0  1  {} {} {}",
                v.tolerance, v.position.x, v.position.y, v.position.z
            )?;
        }
        writeln!(output)?;
        Ok(())
    }

    fn write_edges(&self, output: &mut impl Write) -> io::Result<()> {
        let count = self.store.edges.len();
        writeln!(output, "TEdges {}", count)?;
        for (_ek, edge) in &self.store.edges {
            let ci = self.curve_indices.get(&_ek).copied().unwrap_or(0);
            let vi_low = vk_index(self.store, edge.v_low);
            let vi_high = vk_index(self.store, edge.v_high);
            // Format: tolerance  location(0)  same_sense(1)  curve_index  v_low_index  v_high_index  t_min  t_max
            writeln!(
                output, "{}  0  1  {}  {}  {}  {}  {}",
                edge.tolerance, ci, vi_low, vi_high, edge.t_min, edge.t_max
            )?;
        }
        writeln!(output)?;
        Ok(())
    }

    fn write_wires(&self, output: &mut impl Write) -> io::Result<()> {
        let count = self.store.wires.len();
        writeln!(output, "TWires {}", count)?;
        for (wk, wire) in &self.store.wires {
            writeln!(output, "{}", wire.edges.len())?;
            for (ek, orient) in &wire.edges {
                let ei = edge_index(self.store, *ek);
                let ori_val = match orient {
                    Orientation::Forward => 1,
                    Orientation::Reversed => 0,
                    Orientation::Internal | Orientation::External => 0,
                };
                writeln!(output, "{} {}", ei, ori_val)?;
            }
            // Chain check
            for i in 0..wire.edges.len() {
                let j = (i + 1) % wire.edges.len();
                let (ek_i, ori_i) = &wire.edges[i];
                let (ek_j, ori_j) = &wire.edges[j];
                if let (Some(ei), Some(ej)) = (self.store.edges.get(*ek_i), self.store.edges.get(*ek_j)) {
                    let end_i = match ori_i {
                        Orientation::Forward => ei.v_high,
                        Orientation::Reversed => ei.v_low,
                        Orientation::Internal | Orientation::External => ei.v_high,
                    };
                    let start_j = match ori_j {
                        Orientation::Forward => ej.v_low,
                        Orientation::Reversed => ej.v_high,
                        Orientation::Internal | Orientation::External => ej.v_low,
                    };
                    if end_i != start_j {
                        writeln!(
                            output, "-- WARNING: broken wire chain at edge {}->{} in wire {:?}", i, j, wk
                        )?;
                    }
                }
            }
        }
        writeln!(output)?;
        Ok(())
    }

    fn write_faces(&self, output: &mut impl Write) -> io::Result<()> {
        let count = self.store.faces.len();
        writeln!(output, "TFaces {}", count)?;
        for (fk, face) in &self.store.faces {
            let si = self.surface_indices.get(&fk).copied().unwrap_or(0);
            let owi = wire_index(self.store, face.outer_wire);
            let sense = if face.same_sense { 1 } else { 0 };
            // Format: surface_index  same_sense  location(0)  tolerance  outer_wire_index  inner_wire_count  [inner_wire_indices...]
            write!(
                output, "{} {}  0  {}  {}  {}",
                si, sense, face.tolerance, owi, face.inner_wires.len()
            )?;
            for iw in &face.inner_wires {
                write!(output, " {}", wire_index(self.store, *iw))?;
            }
            writeln!(output)?;
            // Color comment if present
            if let Some(c) = &face.color {
                writeln!(output, "-- color: {} {} {}", c[0], c[1], c[2])?;
            }
        }
        writeln!(output)?;
        Ok(())
    }

    fn write_shells(&self, output: &mut impl Write) -> io::Result<()> {
        let count = self.store.shells.len();
        writeln!(output, "TShells {}", count)?;
        for (_sk, shell) in &self.store.shells {
            writeln!(output, "{}", shell.faces.len())?;
            for (fk, orient) in &shell.faces {
                let fi = face_index(self.store, *fk);
                let ori_val = match orient {
                    Orientation::Forward => 1,
                    Orientation::Reversed => 0,
                    Orientation::Internal | Orientation::External => 0,
                };
                writeln!(output, "{} {}", fi, ori_val)?;
            }
            writeln!(output, "{}", if shell.closed { 1 } else { 0 })?;
        }
        writeln!(output)?;
        Ok(())
    }

    fn write_solids(&self, output: &mut impl Write) -> io::Result<()> {
        let count = self.store.solids.len();
        writeln!(output, "TSolids {}", count)?;
        for (_sk, solid) in &self.store.solids {
            let si = shell_index(self.store, solid.outer_shell);
            writeln!(output, "{}", si)?;
            writeln!(output, "{}", solid.void_shells.len())?;
            for vs in &solid.void_shells {
                writeln!(output, "{}", shell_index(self.store, *vs))?;
            }
        }
        writeln!(output)?;
        Ok(())
    }

    fn write_compounds(&self, output: &mut impl Write) -> io::Result<()> {
        let count = self.store.compounds.len();
        writeln!(output, "TCompounds {}", count)?;
        for (_ck, compound) in &self.store.compounds {
            writeln!(output, "{}", compound.solids.len())?;
            for sk in &compound.solids {
                writeln!(output, "{}", solid_index(self.store, *sk))?;
            }
        }
        writeln!(output)?;
        Ok(())
    }
}

// ── Polyline → BSpline conversion helpers ──────────────────────

/// Write a 3D polyline as a BSpline degree 1 with proper knot vector.
fn write_polyline_as_bspline(output: &mut impl Write, points: &[rc3d_core::math::Vec3]) -> io::Result<()> {
    if points.len() < 2 {
        return writeln!(output, "1 0 0 0  1 0 0"); // degenerate fallback
    }
    let n = points.len();
    let degree = 1usize;
    let knot_len = n + degree + 1;
    let mut knots = Vec::with_capacity(knot_len);
    knots.push(0.0);
    knots.push(0.0);
    for i in 1..n-1 {
        knots.push(i as f32);
    }
    knots.push((n - 1) as f32);
    knots.push((n - 1) as f32);
    // BSpline type 7: degree, cp_count, knot_count, rational(0)
    write!(output, "7 {}  {}  {}  0", degree, n, knot_len)?;
    for p in points {
        write!(output, "  {} {} {}", p.x, p.y, p.z)?;
    }
    for k in &knots {
        write!(output, " {}", k)?;
    }
    writeln!(output)?;
    Ok(())
}

/// Write a 2D polyline as a 2D BSpline degree 1.
fn write_polyline2d_as_bspline(output: &mut impl Write, points: &[(f32, f32)]) -> io::Result<()> {
    if points.len() < 2 {
        return writeln!(output, "1 0 0  1 0");
    }
    let n = points.len();
    let degree = 1usize;
    let knot_len = n + degree + 1;
    let mut knots = Vec::with_capacity(knot_len);
    knots.push(0.0);
    knots.push(0.0);
    for i in 1..n-1 {
        knots.push(i as f32);
    }
    knots.push((n - 1) as f32);
    knots.push((n - 1) as f32);
    write!(output, "7 {}  {}  {}  0", degree, n, knot_len)?;
    for p in points {
        write!(output, "  {} {}", p.0, p.1)?;
    }
    for k in &knots {
        write!(output, " {}", k)?;
    }
    writeln!(output)?;
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────

/// Check if NURBS weights are non-uniform (rational surface).
fn nurbs_is_rational(weights: &[Vec<f32>]) -> bool {
    weights.iter().any(|row| row.iter().any(|&w| (w - 1.0).abs() > 1e-6))
}

/// Expand Trimmed curves to their inner basis (trimming handled by Edge t_min/t_max).
fn expand_curve<'c>(curve: &'c CurveGeom) -> (&'c CurveGeom, (f32, f32)) {
    match curve {
        CurveGeom::Trimmed { basis, t_min, t_max } => {
            (basis.as_ref(), (*t_min, *t_max))
        }
        other => (other, (0.0, 1.0)),
    }
}

/// Human-readable curve type name for comments.
fn curve_type_name(curve: &CurveGeom) -> &'static str {
    match curve {
        CurveGeom::Line { .. } => "Line",
        CurveGeom::Circle { .. } => "Circle",
        CurveGeom::Ellipse { .. } => "Ellipse",
        CurveGeom::Hyperbola { .. } => "Hyperbola",
        CurveGeom::Parabola { .. } => "Parabola",
        CurveGeom::BezierCurve { .. } => "BezierCurve",
        CurveGeom::BSpline { .. } => "BSpline",
        CurveGeom::Trimmed { .. } => "Trimmed",
        CurveGeom::Composite { .. } => "Composite",
        CurveGeom::Polyline { .. } => "Polyline",
        CurveGeom::Offset { .. } => "Offset",
    }
}

/// Get 1-based vertex index.
fn vk_index(store: &BRepStore, vk: VertexKey) -> usize {
    store.vertices.iter().position(|(k, _)| k == vk)
        .map(|p| p + 1).unwrap_or(0)
}

/// Get 1-based edge index.
fn edge_index(store: &BRepStore, ek: EdgeKey) -> usize {
    store.edges.iter().position(|(k, _)| k == ek)
        .map(|p| p + 1).unwrap_or(0)
}

/// Get 1-based wire index.
fn wire_index(store: &BRepStore, wk: WireKey) -> usize {
    store.wires.iter().position(|(k, _)| k == wk)
        .map(|p| p + 1).unwrap_or(0)
}

/// Get 1-based face index.
fn face_index(store: &BRepStore, fk: FaceKey) -> usize {
    store.faces.iter().position(|(k, _)| k == fk)
        .map(|p| p + 1).unwrap_or(0)
}

/// Get 1-based shell index.
fn shell_index(store: &BRepStore, sk: ShellKey) -> usize {
    store.shells.iter().position(|(k, _)| k == sk)
        .map(|p| p + 1).unwrap_or(0)
}

/// Get 1-based solid index.
fn solid_index(store: &BRepStore, sk: SolidKey) -> usize {
    store.solids.iter().position(|(k, _)| k == sk)
        .map(|p| p + 1).unwrap_or(0)
}
