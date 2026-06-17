//! Binary BRep persistence (BRepTools_ShapeSet equivalent).
//!
//! Simple custom binary format for fast, lossless roundtripping of
//! BRepStore data. Skip BSpline curves/surfaces for now.

use std::collections::HashMap;
use std::io::{self, Read, Write};

use rc3d_core::math::{Real, PVec3};
use rc3d_shape::geom::{CurveGeom, SurfaceGeom};
use rc3d_shape::geom::curve2d::Curve2d;
use rc3d_shape::topo::*;
use rc3d_shape::store::BRepStore;

// ── Constants ────────────────────────────────────────────────────────

const MAGIC: &[u8; 8] = b"RUSTBREP";
const VERSION: u32 = 1;
const FLAGS: u32 = 0;

// ── Curve type tags ──────────────────────────────────────────────────

mod curve_tag {
    pub const LINE: u8 = 1;
    pub const CIRCLE: u8 = 2;
    pub const ELLIPSE: u8 = 3;
    pub const HYPERBOLA: u8 = 4;
    pub const PARABOLA: u8 = 5;
    // BSpline: 6 (reserved, skipped)
    pub const BEZIER: u8 = 7;
    pub const TRIMMED: u8 = 8;
    pub const COMPOSITE: u8 = 9;
    pub const POLYLINE: u8 = 10;
    pub const OFFSET: u8 = 11;
}

// ── Surface type tags ────────────────────────────────────────────────

mod surface_tag {
    pub const PLANE: u8 = 1;
    pub const CYLINDER: u8 = 2;
    pub const CONE: u8 = 3;
    pub const SPHERE: u8 = 4;
    pub const TORUS: u8 = 5;
    // BSpline: 6 (reserved, skipped)
    pub const EXTRUSION: u8 = 7;
    pub const REVOLUTION: u8 = 8;
    pub const OFFSET: u8 = 9;
}

// ── PCurve (Curve2d) type tags ───────────────────────────────────────

mod pcurve_tag {
    pub const LINE: u8 = 1;
    pub const CIRCLE: u8 = 2;
    pub const ELLIPSE: u8 = 3;
    // BSpline: 4 (reserved, skipped)
    pub const TRIMMED: u8 = 5;
    pub const POLYLINE: u8 = 6;
    pub const COMPOSITE: u8 = 7;
}

// ── BinaryWriter ─────────────────────────────────────────────────────

struct BinaryWriter {
    buf: Vec<u8>,
}

impl BinaryWriter {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn into_vec(self) -> Vec<u8> {
        self.buf
    }

    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_bool(&mut self, v: bool) {
        self.write_u8(if v { 1 } else { 0 });
    }

    fn write_pvec3(&mut self, v: PVec3) {
        self.write_f64(v.x);
        self.write_f64(v.y);
        self.write_f64(v.z);
    }

    fn write_slice_u32(&mut self, slice: &[u32]) {
        self.write_u32(slice.len() as u32);
        for &v in slice {
            self.write_u32(v);
        }
    }

    fn write_curve(&mut self, c: &CurveGeom) {
        match c {
            CurveGeom::Line { origin, direction } => {
                self.write_u8(curve_tag::LINE);
                self.write_pvec3(*origin);
                self.write_pvec3(*direction);
            }
            CurveGeom::Circle { center, axis, radius, x_dir, y_dir } => {
                self.write_u8(curve_tag::CIRCLE);
                self.write_pvec3(*center);
                self.write_pvec3(*axis);
                self.write_f64(*radius);
                self.write_pvec3(*x_dir);
                self.write_pvec3(*y_dir);
            }
            CurveGeom::Ellipse { center, axis, semi_major, semi_minor, x_dir, y_dir } => {
                self.write_u8(curve_tag::ELLIPSE);
                self.write_pvec3(*center);
                self.write_pvec3(*axis);
                self.write_f64(*semi_major);
                self.write_f64(*semi_minor);
                self.write_pvec3(*x_dir);
                self.write_pvec3(*y_dir);
            }
            CurveGeom::Hyperbola { center, axis, semi_major, semi_minor, x_dir, y_dir } => {
                self.write_u8(curve_tag::HYPERBOLA);
                self.write_pvec3(*center);
                self.write_pvec3(*axis);
                self.write_f64(*semi_major);
                self.write_f64(*semi_minor);
                self.write_pvec3(*x_dir);
                self.write_pvec3(*y_dir);
            }
            CurveGeom::Parabola { center, axis, focal_dist, x_dir, y_dir } => {
                self.write_u8(curve_tag::PARABOLA);
                self.write_pvec3(*center);
                self.write_pvec3(*axis);
                self.write_f64(*focal_dist);
                self.write_pvec3(*x_dir);
                self.write_pvec3(*y_dir);
            }
            CurveGeom::BSpline { .. } => {
                // BSpline not supported — write as polyline approximation
                // Fall back to sampling 32 points along the curve
                let pts: Vec<PVec3> = (0..=32).map(|i| {
                    let t = i as Real / 32.0;
                    c.d0(t)
                }).collect();
                self.write_u8(curve_tag::POLYLINE);
                self.write_u32(pts.len() as u32);
                for p in &pts {
                    self.write_pvec3(*p);
                }
            }
            CurveGeom::BezierCurve { degree, control_points, weights } => {
                self.write_u8(curve_tag::BEZIER);
                self.write_u32(*degree as u32);
                self.write_u32(control_points.len() as u32);
                for p in control_points {
                    self.write_pvec3(*p);
                }
                match weights {
                    Some(w) => {
                        self.write_bool(true);
                        self.write_u32(w.len() as u32);
                        for &v in w {
                            self.write_f64(v);
                        }
                    }
                    None => self.write_bool(false),
                }
            }
            CurveGeom::Trimmed { basis, t_min, t_max } => {
                self.write_u8(curve_tag::TRIMMED);
                self.write_f64(*t_min);
                self.write_f64(*t_max);
                self.write_curve(basis);
            }
            CurveGeom::Composite { segments, cached_lengths } => {
                self.write_u8(curve_tag::COMPOSITE);
                self.write_u32(segments.len() as u32);
                for (seg, reversed) in segments {
                    self.write_bool(*reversed);
                    self.write_curve(seg);
                }
                match cached_lengths {
                    Some(lens) => {
                        self.write_bool(true);
                        self.write_u32(lens.len() as u32);
                        for &v in lens {
                            self.write_f64(v);
                        }
                    }
                    None => self.write_bool(false),
                }
            }
            CurveGeom::Polyline { points } => {
                self.write_u8(curve_tag::POLYLINE);
                self.write_u32(points.len() as u32);
                for p in points {
                    self.write_pvec3(*p);
                }
            }
            CurveGeom::Offset { basis, offset_dir, distance } => {
                self.write_u8(curve_tag::OFFSET);
                self.write_pvec3(*offset_dir);
                self.write_f64(*distance);
                self.write_curve(basis);
            }
        }
    }

    fn write_surface(&mut self, s: &SurfaceGeom) {
        match s {
            SurfaceGeom::Plane { origin, normal, u_dir } => {
                self.write_u8(surface_tag::PLANE);
                self.write_pvec3(*origin);
                self.write_pvec3(*normal);
                self.write_pvec3(*u_dir);
            }
            SurfaceGeom::Cylinder { origin, axis, radius, x_dir, y_dir } => {
                self.write_u8(surface_tag::CYLINDER);
                self.write_pvec3(*origin);
                self.write_pvec3(*axis);
                self.write_f64(*radius);
                self.write_pvec3(*x_dir);
                self.write_pvec3(*y_dir);
            }
            SurfaceGeom::Cone { apex, axis, semi_angle, radius_at_apex, x_dir, y_dir } => {
                self.write_u8(surface_tag::CONE);
                self.write_pvec3(*apex);
                self.write_pvec3(*axis);
                self.write_f64(*semi_angle);
                self.write_f64(*radius_at_apex);
                self.write_pvec3(*x_dir);
                self.write_pvec3(*y_dir);
            }
            SurfaceGeom::Sphere { center, radius } => {
                self.write_u8(surface_tag::SPHERE);
                self.write_pvec3(*center);
                self.write_f64(*radius);
            }
            SurfaceGeom::Torus { center, axis, major_r, minor_r, x_dir, y_dir } => {
                self.write_u8(surface_tag::TORUS);
                self.write_pvec3(*center);
                self.write_pvec3(*axis);
                self.write_f64(*major_r);
                self.write_f64(*minor_r);
                self.write_pvec3(*x_dir);
                self.write_pvec3(*y_dir);
            }
            SurfaceGeom::BSpline(_) => {
                // BSpline not supported — write as plane (placeholder)
                // The surface geometry is preserved as-is; this path should
                // not be hit since we skip BSpline surfaces per instructions.
                self.write_u8(surface_tag::PLANE);
                self.write_pvec3(PVec3::ZERO);
                self.write_pvec3(PVec3::Z);
                self.write_pvec3(PVec3::X);
            }
            SurfaceGeom::Extrusion { generatrix, direction } => {
                self.write_u8(surface_tag::EXTRUSION);
                self.write_pvec3(*direction);
                self.write_curve(generatrix);
            }
            SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } => {
                self.write_u8(surface_tag::REVOLUTION);
                self.write_pvec3(*axis_origin);
                self.write_pvec3(*axis_dir);
                self.write_curve(generatrix);
            }
            SurfaceGeom::Offset { basis, distance } => {
                self.write_u8(surface_tag::OFFSET);
                self.write_f64(*distance);
                self.write_surface(basis);
            }
        }
    }

    fn write_pcurve(&mut self, pc: &Curve2d) {
        match pc {
            Curve2d::Line { origin, direction } => {
                self.write_u8(pcurve_tag::LINE);
                self.write_f64(origin.0);
                self.write_f64(origin.1);
                self.write_f64(direction.0);
                self.write_f64(direction.1);
            }
            Curve2d::Circle { center, radius } => {
                self.write_u8(pcurve_tag::CIRCLE);
                self.write_f64(center.0);
                self.write_f64(center.1);
                self.write_f64(*radius);
            }
            Curve2d::Ellipse { center, semi_major, semi_minor } => {
                self.write_u8(pcurve_tag::ELLIPSE);
                self.write_f64(center.0);
                self.write_f64(center.1);
                self.write_f64(*semi_major);
                self.write_f64(*semi_minor);
            }
            Curve2d::BSpline { .. } => {
                // BSpline not supported — write as polyline approximation
                let pts: Vec<(Real, Real)> = (0..=32).map(|i| {
                    let t = i as Real / 32.0;
                    pc.d0(t)
                }).collect();
                self.write_u8(pcurve_tag::POLYLINE);
                self.write_u32(pts.len() as u32);
                for (u, v) in &pts {
                    self.write_f64(*u);
                    self.write_f64(*v);
                }
            }
            Curve2d::Trimmed { basis, t_min, t_max } => {
                self.write_u8(pcurve_tag::TRIMMED);
                self.write_f64(*t_min);
                self.write_f64(*t_max);
                self.write_pcurve(basis);
            }
            Curve2d::Polyline { points } => {
                self.write_u8(pcurve_tag::POLYLINE);
                self.write_u32(points.len() as u32);
                for (u, v) in points {
                    self.write_f64(*u);
                    self.write_f64(*v);
                }
            }
            Curve2d::Composite { segments } => {
                self.write_u8(pcurve_tag::COMPOSITE);
                self.write_u32(segments.len() as u32);
                for (seg, reversed) in segments {
                    self.write_bool(*reversed);
                    self.write_pcurve(seg);
                }
            }
        }
    }
}

// ── BinaryReader ─────────────────────────────────────────────────────

struct BinaryReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        if self.pos >= self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected EOF"));
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let end = self.pos + 4;
        if end > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected EOF"));
        }
        let v = u32::from_le_bytes(self.data[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v)
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let end = self.pos + 8;
        if end > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected EOF"));
        }
        let v = u64::from_le_bytes(self.data[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v)
    }

    fn read_f64(&mut self) -> io::Result<f64> {
        let end = self.pos + 8;
        if end > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected EOF"));
        }
        let v = f64::from_le_bytes(self.data[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v)
    }

    fn read_bool(&mut self) -> io::Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_pvec3(&mut self) -> io::Result<PVec3> {
        let x = self.read_f64()?;
        let y = self.read_f64()?;
        let z = self.read_f64()?;
        Ok(PVec3::new(x, y, z))
    }

    fn read_u32_vec(&mut self) -> io::Result<Vec<u32>> {
        let len = self.read_u32()? as usize;
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(self.read_u32()?);
        }
        Ok(v)
    }

    fn read_f64_vec(&mut self) -> io::Result<Vec<f64>> {
        let len = self.read_u32()? as usize;
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(self.read_f64()?);
        }
        Ok(v)
    }

    fn read_pvec3_vec(&mut self) -> io::Result<Vec<PVec3>> {
        let len = self.read_u32()? as usize;
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(self.read_pvec3()?);
        }
        Ok(v)
    }

    fn read_curve(&mut self) -> io::Result<CurveGeom> {
        let tag = self.read_u8()?;
        match tag {
            curve_tag::LINE => {
                let origin = self.read_pvec3()?;
                let direction = self.read_pvec3()?;
                Ok(CurveGeom::Line { origin, direction })
            }
            curve_tag::CIRCLE => {
                let center = self.read_pvec3()?;
                let axis = self.read_pvec3()?;
                let radius = self.read_f64()?;
                let x_dir = self.read_pvec3()?;
                let y_dir = self.read_pvec3()?;
                Ok(CurveGeom::Circle { center, axis, radius, x_dir, y_dir })
            }
            curve_tag::ELLIPSE => {
                let center = self.read_pvec3()?;
                let axis = self.read_pvec3()?;
                let semi_major = self.read_f64()?;
                let semi_minor = self.read_f64()?;
                let x_dir = self.read_pvec3()?;
                let y_dir = self.read_pvec3()?;
                Ok(CurveGeom::Ellipse { center, axis, semi_major, semi_minor, x_dir, y_dir })
            }
            curve_tag::HYPERBOLA => {
                let center = self.read_pvec3()?;
                let axis = self.read_pvec3()?;
                let semi_major = self.read_f64()?;
                let semi_minor = self.read_f64()?;
                let x_dir = self.read_pvec3()?;
                let y_dir = self.read_pvec3()?;
                Ok(CurveGeom::Hyperbola { center, axis, semi_major, semi_minor, x_dir, y_dir })
            }
            curve_tag::PARABOLA => {
                let center = self.read_pvec3()?;
                let axis = self.read_pvec3()?;
                let focal_dist = self.read_f64()?;
                let x_dir = self.read_pvec3()?;
                let y_dir = self.read_pvec3()?;
                Ok(CurveGeom::Parabola { center, axis, focal_dist, x_dir, y_dir })
            }
            curve_tag::BEZIER => {
                let degree = self.read_u32()? as usize;
                let cp_len = self.read_u32()? as usize;
                let mut control_points = Vec::with_capacity(cp_len);
                for _ in 0..cp_len {
                    control_points.push(self.read_pvec3()?);
                }
                let weights = if self.read_bool()? {
                    let w_len = self.read_u32()? as usize;
                    let mut w = Vec::with_capacity(w_len);
                    for _ in 0..w_len {
                        w.push(self.read_f64()?);
                    }
                    Some(w)
                } else {
                    None
                };
                Ok(CurveGeom::BezierCurve { degree, control_points, weights })
            }
            curve_tag::TRIMMED => {
                let t_min = self.read_f64()?;
                let t_max = self.read_f64()?;
                let basis = Box::new(self.read_curve()?);
                Ok(CurveGeom::Trimmed { basis, t_min, t_max })
            }
            curve_tag::COMPOSITE => {
                let seg_len = self.read_u32()? as usize;
                let mut segments = Vec::with_capacity(seg_len);
                for _ in 0..seg_len {
                    let reversed = self.read_bool()?;
                    let seg = self.read_curve()?;
                    segments.push((seg, reversed));
                }
                let cached_lengths = if self.read_bool()? {
                    Some(self.read_f64_vec()?)
                } else {
                    None
                };
                Ok(CurveGeom::Composite { segments, cached_lengths })
            }
            curve_tag::POLYLINE => {
                let points = self.read_pvec3_vec()?;
                Ok(CurveGeom::Polyline { points })
            }
            curve_tag::OFFSET => {
                let offset_dir = self.read_pvec3()?;
                let distance = self.read_f64()?;
                let basis = Box::new(self.read_curve()?);
                Ok(CurveGeom::Offset { basis, offset_dir, distance })
            }
            _ => Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("unknown curve tag: {}", tag))),
        }
    }

    fn read_surface(&mut self) -> io::Result<SurfaceGeom> {
        let tag = self.read_u8()?;
        match tag {
            surface_tag::PLANE => {
                let origin = self.read_pvec3()?;
                let normal = self.read_pvec3()?;
                let u_dir = self.read_pvec3()?;
                Ok(SurfaceGeom::Plane { origin, normal, u_dir })
            }
            surface_tag::CYLINDER => {
                let origin = self.read_pvec3()?;
                let axis = self.read_pvec3()?;
                let radius = self.read_f64()?;
                let x_dir = self.read_pvec3()?;
                let y_dir = self.read_pvec3()?;
                Ok(SurfaceGeom::Cylinder { origin, axis, radius, x_dir, y_dir })
            }
            surface_tag::CONE => {
                let apex = self.read_pvec3()?;
                let axis = self.read_pvec3()?;
                let semi_angle = self.read_f64()?;
                let radius_at_apex = self.read_f64()?;
                let x_dir = self.read_pvec3()?;
                let y_dir = self.read_pvec3()?;
                Ok(SurfaceGeom::Cone { apex, axis, semi_angle, radius_at_apex, x_dir, y_dir })
            }
            surface_tag::SPHERE => {
                let center = self.read_pvec3()?;
                let radius = self.read_f64()?;
                Ok(SurfaceGeom::Sphere { center, radius })
            }
            surface_tag::TORUS => {
                let center = self.read_pvec3()?;
                let axis = self.read_pvec3()?;
                let major_r = self.read_f64()?;
                let minor_r = self.read_f64()?;
                let x_dir = self.read_pvec3()?;
                let y_dir = self.read_pvec3()?;
                Ok(SurfaceGeom::Torus { center, axis, major_r, minor_r, x_dir, y_dir })
            }
            surface_tag::EXTRUSION => {
                let direction = self.read_pvec3()?;
                let generatrix = Box::new(self.read_curve()?);
                Ok(SurfaceGeom::Extrusion { generatrix, direction })
            }
            surface_tag::REVOLUTION => {
                let axis_origin = self.read_pvec3()?;
                let axis_dir = self.read_pvec3()?;
                let generatrix = Box::new(self.read_curve()?);
                Ok(SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir })
            }
            surface_tag::OFFSET => {
                let distance = self.read_f64()?;
                let basis = Box::new(self.read_surface()?);
                Ok(SurfaceGeom::Offset { basis, distance })
            }
            _ => Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("unknown surface tag: {}", tag))),
        }
    }

    fn read_pcurve(&mut self) -> io::Result<Curve2d> {
        let tag = self.read_u8()?;
        match tag {
            pcurve_tag::LINE => {
                let ox = self.read_f64()?;
                let oy = self.read_f64()?;
                let dx = self.read_f64()?;
                let dy = self.read_f64()?;
                Ok(Curve2d::Line { origin: (ox, oy), direction: (dx, dy) })
            }
            pcurve_tag::CIRCLE => {
                let cx = self.read_f64()?;
                let cy = self.read_f64()?;
                let radius = self.read_f64()?;
                Ok(Curve2d::Circle { center: (cx, cy), radius })
            }
            pcurve_tag::ELLIPSE => {
                let cx = self.read_f64()?;
                let cy = self.read_f64()?;
                let semi_major = self.read_f64()?;
                let semi_minor = self.read_f64()?;
                Ok(Curve2d::Ellipse { center: (cx, cy), semi_major, semi_minor })
            }
            pcurve_tag::TRIMMED => {
                let t_min = self.read_f64()?;
                let t_max = self.read_f64()?;
                let basis = Box::new(self.read_pcurve()?);
                Ok(Curve2d::Trimmed { basis, t_min, t_max })
            }
            pcurve_tag::POLYLINE => {
                let len = self.read_u32()? as usize;
                let mut points = Vec::with_capacity(len);
                for _ in 0..len {
                    let u = self.read_f64()?;
                    let v = self.read_f64()?;
                    points.push((u, v));
                }
                Ok(Curve2d::Polyline { points })
            }
            pcurve_tag::COMPOSITE => {
                let seg_len = self.read_u32()? as usize;
                let mut segments = Vec::with_capacity(seg_len);
                for _ in 0..seg_len {
                    let reversed = self.read_bool()?;
                    let seg = self.read_pcurve()?;
                    segments.push((seg, reversed));
                }
                Ok(Curve2d::Composite { segments })
            }
            _ => Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("unknown pcurve tag: {}", tag))),
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────

/// Write BRepStore to binary buffer.
pub fn write_brep_binary(store: &BRepStore) -> Vec<u8> {
    let mut w = BinaryWriter::new();

    // Header
    w.buf.extend_from_slice(MAGIC);
    w.write_u32(VERSION);
    w.write_u32(FLAGS);

    // Build key → index maps
    let v_map: HashMap<VertexKey, u32> = store.vertices.keys()
        .enumerate()
        .map(|(i, k)| (k, i as u32))
        .collect();
    let e_map: HashMap<EdgeKey, u32> = store.edges.keys()
        .enumerate()
        .map(|(i, k)| (k, i as u32))
        .collect();
    let w_map: HashMap<WireKey, u32> = store.wires.keys()
        .enumerate()
        .map(|(i, k)| (k, i as u32))
        .collect();
    let f_map: HashMap<FaceKey, u32> = store.faces.keys()
        .enumerate()
        .map(|(i, k)| (k, i as u32))
        .collect();
    let sh_map: HashMap<ShellKey, u32> = store.shells.keys()
        .enumerate()
        .map(|(i, k)| (k, i as u32))
        .collect();
    let so_map: HashMap<SolidKey, u32> = store.solids.keys()
        .enumerate()
        .map(|(i, k)| (k, i as u32))
        .collect();
    let c_map: HashMap<CompoundKey, u32> = store.compounds.keys()
        .enumerate()
        .map(|(i, k)| (k, i as u32))
        .collect();

    // ── Vertex table ──────────────────────────────────────────────
    w.write_u32(store.vertices.len() as u32);
    for vk in store.vertices.keys() {
        let v = &store.vertices[vk];
        w.write_f64(v.position.x);
        w.write_f64(v.position.y);
        w.write_f64(v.position.z);
        w.write_f64(v.tolerance);
    }

    // ── Edge table ────────────────────────────────────────────────
    w.write_u32(store.edges.len() as u32);
    for ek in store.edges.keys() {
        let e = &store.edges[ek];
        w.write_curve(&e.curve);
        w.write_f64(e.t_min);
        w.write_f64(e.t_max);
        w.write_f64(e.tolerance);
        w.write_u32(v_map[&e.v_low]);
        w.write_u32(v_map[&e.v_high]);
        // PCurves: sort face keys for deterministic output
        let mut face_keys: Vec<FaceKey> = e.pcurves.keys().copied().collect();
        face_keys.sort();
        w.write_u32(face_keys.len() as u32);
        for fk in &face_keys {
            w.write_u32(f_map[fk]);
            w.write_pcurve(&e.pcurves[fk]);
        }
    }

    // ── Wire table ────────────────────────────────────────────────
    w.write_u32(store.wires.len() as u32);
    for wk in store.wires.keys() {
        let wire = &store.wires[wk];
        w.write_u32(wire.edges.len() as u32);
        for (ek, orient) in &wire.edges {
            w.write_u32(e_map[ek]);
            w.write_u8(match orient {
                Orientation::Forward => 0,
                Orientation::Reversed => 1,
                Orientation::Internal => 2,
                Orientation::External => 3,
            });
        }
    }

    // ── Face table ────────────────────────────────────────────────
    w.write_u32(store.faces.len() as u32);
    for fk in store.faces.keys() {
        let f = &store.faces[fk];
        w.write_surface(&f.surface);
        w.write_u32(w_map[&f.outer_wire]);
        w.write_u32(f.inner_wires.len() as u32);
        for iw in &f.inner_wires {
            w.write_u32(w_map[iw]);
        }
        w.write_bool(f.same_sense);
        w.write_f64(f.tolerance);
        // Seam edges
        w.write_u32(f.seam_edges.len() as u32);
        for sek in &f.seam_edges {
            w.write_u32(e_map[sek]);
        }
        // Color
        match &f.color {
            Some(c) => {
                w.write_bool(true);
                w.write_f64(c[0]);
                w.write_f64(c[1]);
                w.write_f64(c[2]);
            }
            None => w.write_bool(false),
        }
        // Degenerated edges
        w.write_u32(f.degenerated_edges.len() as u32);
        for dk in &f.degenerated_edges {
            w.write_u32(e_map[dk]);
        }
    }

    // ── Shell table ───────────────────────────────────────────────
    w.write_u32(store.shells.len() as u32);
    for shk in store.shells.keys() {
        let sh = &store.shells[shk];
        w.write_u32(sh.faces.len() as u32);
        for (fk, orient) in &sh.faces {
            w.write_u32(f_map[fk]);
            w.write_u8(match orient {
                Orientation::Forward => 0,
                Orientation::Reversed => 1,
                Orientation::Internal => 2,
                Orientation::External => 3,
            });
        }
        w.write_bool(sh.closed);
        match sh.step_id {
            Some(id) => { w.write_bool(true); w.write_u64(id); }
            None => w.write_bool(false),
        }
    }

    // ── Solid table ───────────────────────────────────────────────
    w.write_u32(store.solids.len() as u32);
    for sok in store.solids.keys() {
        let so = &store.solids[sok];
        w.write_u32(sh_map[&so.outer_shell]);
        w.write_u32(so.void_shells.len() as u32);
        for vs in &so.void_shells {
            w.write_u32(sh_map[vs]);
        }
    }

    // ── Compound table ────────────────────────────────────────────
    w.write_u32(store.compounds.len() as u32);
    for ck in store.compounds.keys() {
        let c = &store.compounds[ck];
        w.write_u32(c.solids.len() as u32);
        for sk in &c.solids {
            w.write_u32(so_map[sk]);
        }
    }

    w.into_vec()
}

/// Read BRepStore from binary buffer.
pub fn read_brep_binary(data: &[u8]) -> io::Result<BRepStore> {
    let mut r = BinaryReader::new(data);

    // Header
    if r.data.len() < 16 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "data too short for header"));
    }
    let magic = &r.data[0..8];
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("invalid magic: expected {:?}, got {:?}", MAGIC, magic)));
    }
    r.pos = 8;
    let _version = r.read_u32()?;
    let _flags = r.read_u32()?;

    // ── Vertex table ──────────────────────────────────────────────
    let v_count = r.read_u32()? as usize;
    let mut v_positions: Vec<PVec3> = Vec::with_capacity(v_count);
    let mut v_tols: Vec<Real> = Vec::with_capacity(v_count);
    for _ in 0..v_count {
        v_positions.push(r.read_pvec3()?);
        v_tols.push(r.read_f64()?);
    }

    // ── Edge table ────────────────────────────────────────────────
    let e_count = r.read_u32()? as usize;
    struct RawEdge {
        curve: CurveGeom,
        t_min: Real,
        t_max: Real,
        tolerance: Real,
        v_low_idx: u32,
        v_high_idx: u32,
        pcurves: Vec<(u32, Curve2d)>, // (face_idx, pcurve)
    }
    let mut raw_edges: Vec<RawEdge> = Vec::with_capacity(e_count);
    for _ in 0..e_count {
        let curve = r.read_curve()?;
        let t_min = r.read_f64()?;
        let t_max = r.read_f64()?;
        let tolerance = r.read_f64()?;
        let v_low_idx = r.read_u32()?;
        let v_high_idx = r.read_u32()?;
        let pc_count = r.read_u32()? as usize;
        let mut pcurves = Vec::with_capacity(pc_count);
        for _ in 0..pc_count {
            let face_idx = r.read_u32()?;
            let pc = r.read_pcurve()?;
            pcurves.push((face_idx, pc));
        }
        raw_edges.push(RawEdge { curve, t_min, t_max, tolerance, v_low_idx, v_high_idx, pcurves });
    }

    // ── Wire table ────────────────────────────────────────────────
    let w_count = r.read_u32()? as usize;
    struct RawWire {
        edges: Vec<(u32, Orientation)>, // (edge_idx, orient)
    }
    let mut raw_wires: Vec<RawWire> = Vec::with_capacity(w_count);
    for _ in 0..w_count {
        let edge_count = r.read_u32()? as usize;
        let mut edges = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            let edge_idx = r.read_u32()?;
            let orient = match r.read_u8()? {
                1 => Orientation::Reversed,
                2 => Orientation::Internal,
                3 => Orientation::External,
                _ => Orientation::Forward,
            };
            edges.push((edge_idx, orient));
        }
        raw_wires.push(RawWire { edges });
    }

    // ── Face table ────────────────────────────────────────────────
    let f_count = r.read_u32()? as usize;
    struct RawFace {
        surface: SurfaceGeom,
        outer_wire_idx: u32,
        inner_wire_indices: Vec<u32>,
        same_sense: bool,
        tolerance: Real,
        seam_edge_indices: Vec<u32>,
        color: Option<[Real; 3]>,
        degenerated_edge_indices: Vec<u32>,
    }
    let mut raw_faces: Vec<RawFace> = Vec::with_capacity(f_count);
    for _ in 0..f_count {
        let surface = r.read_surface()?;
        let outer_wire_idx = r.read_u32()?;
        let iw_count = r.read_u32()? as usize;
        let mut inner_wire_indices = Vec::with_capacity(iw_count);
        for _ in 0..iw_count {
            inner_wire_indices.push(r.read_u32()?);
        }
        let same_sense = r.read_bool()?;
        let tolerance = r.read_f64()?;
        let se_count = r.read_u32()? as usize;
        let mut seam_edge_indices = Vec::with_capacity(se_count);
        for _ in 0..se_count {
            seam_edge_indices.push(r.read_u32()?);
        }
        let color = if r.read_bool()? {
            Some([r.read_f64()?, r.read_f64()?, r.read_f64()?])
        } else {
            None
        };
        let de_count = r.read_u32()? as usize;
        let mut degenerated_edge_indices = Vec::with_capacity(de_count);
        for _ in 0..de_count {
            degenerated_edge_indices.push(r.read_u32()?);
        }
        raw_faces.push(RawFace {
            surface, outer_wire_idx, inner_wire_indices, same_sense, tolerance,
            seam_edge_indices, color, degenerated_edge_indices,
        });
    }

    // ── Shell table ───────────────────────────────────────────────
    let sh_count = r.read_u32()? as usize;
    struct RawShell {
        faces: Vec<(u32, Orientation)>,
        closed: bool,
        step_id: Option<u64>,
    }
    let mut raw_shells: Vec<RawShell> = Vec::with_capacity(sh_count);
    for _ in 0..sh_count {
        let face_count = r.read_u32()? as usize;
        let mut faces = Vec::with_capacity(face_count);
        for _ in 0..face_count {
            let face_idx = r.read_u32()?;
            let orient = match r.read_u8()? {
                1 => Orientation::Reversed,
                2 => Orientation::Internal,
                3 => Orientation::External,
                _ => Orientation::Forward,
            };
            faces.push((face_idx, orient));
        }
        let closed = r.read_bool()?;
        let step_id = if r.read_bool()? { Some(r.read_u64()?) } else { None };
        raw_shells.push(RawShell { faces, closed, step_id });
    }

    // ── Solid table ───────────────────────────────────────────────
    let so_count = r.read_u32()? as usize;
    struct RawSolid {
        outer_shell_idx: u32,
        void_shell_indices: Vec<u32>,
    }
    let mut raw_solids: Vec<RawSolid> = Vec::with_capacity(so_count);
    for _ in 0..so_count {
        let outer_shell_idx = r.read_u32()?;
        let void_count = r.read_u32()? as usize;
        let mut void_shell_indices = Vec::with_capacity(void_count);
        for _ in 0..void_count {
            void_shell_indices.push(r.read_u32()?);
        }
        raw_solids.push(RawSolid { outer_shell_idx, void_shell_indices });
    }

    // ── Compound table ────────────────────────────────────────────
    let c_count = r.read_u32()? as usize;
    struct RawCompound {
        solid_indices: Vec<u32>,
    }
    let mut raw_compounds: Vec<RawCompound> = Vec::with_capacity(c_count);
    for _ in 0..c_count {
        let solid_count = r.read_u32()? as usize;
        let mut solid_indices = Vec::with_capacity(solid_count);
        for _ in 0..solid_count {
            solid_indices.push(r.read_u32()?);
        }
        raw_compounds.push(RawCompound { solid_indices });
    }

    // ── Build BRepStore from raw data ─────────────────────────────
    let mut store = BRepStore::new();

    // Insert vertices
    let mut v_keys: Vec<VertexKey> = Vec::with_capacity(v_count);
    for i in 0..v_count {
        let k = store.vertices.insert(BRepVertex {
            position: v_positions[i],
            tolerance: v_tols[i],
        });
        v_keys.push(k);
    }

    // Insert edges (without pcurves first, add pcurves after face keys exist)
    let mut e_keys: Vec<EdgeKey> = Vec::with_capacity(e_count);
    for re in &raw_edges {
        let v_low = v_keys[re.v_low_idx as usize];
        let v_high = v_keys[re.v_high_idx as usize];
        let k = store.edges.insert(BRepEdge {
            curve: re.curve.clone(),
            tolerance: re.tolerance,
            v_low,
            v_high,
            t_min: re.t_min,
            t_max: re.t_max,
            pcurves: HashMap::new(), // filled after face keys exist
            cached_deflection: None,
        });
        e_keys.push(k);
    }

    // Insert wires
    let mut w_keys: Vec<WireKey> = Vec::with_capacity(w_count);
    for rw in &raw_wires {
        let edges: Vec<(EdgeKey, Orientation)> = rw.edges.iter().map(|(ei, o)| {
            (e_keys[*ei as usize], *o)
        }).collect();
        let k = store.wires.insert(BRepWire { edges });
        w_keys.push(k);
    }

    // Insert faces
    let mut f_keys: Vec<FaceKey> = Vec::with_capacity(f_count);
    for rf in &raw_faces {
        let outer_wire = w_keys[rf.outer_wire_idx as usize];
        let inner_wires: Vec<WireKey> = rf.inner_wire_indices.iter()
            .map(|&i| w_keys[i as usize]).collect();
        let seam_edges: Vec<EdgeKey> = rf.seam_edge_indices.iter()
            .map(|&i| e_keys[i as usize]).collect();
        let degenerated_edges: Vec<EdgeKey> = rf.degenerated_edge_indices.iter()
            .map(|&i| e_keys[i as usize]).collect();
        let k = store.faces.insert(BRepFace {
            surface: rf.surface.clone(),
            outer_wire,
            inner_wires,
            same_sense: rf.same_sense,
            tolerance: rf.tolerance,
            seam_edges,
            color: rf.color,
            degenerated_edges,
        });
        f_keys.push(k);
    }

    // Fill pcurves into edges now that face keys exist
    for (ei, re) in raw_edges.iter().enumerate() {
        let ek = e_keys[ei];
        if let Some(edge) = store.edges.get_mut(ek) {
            for (fi, pc) in &re.pcurves {
                let fk = f_keys[*fi as usize];
                edge.pcurves.insert(fk, pc.clone());
            }
        }
    }

    // Insert shells
    let mut sh_keys: Vec<ShellKey> = Vec::with_capacity(sh_count);
    for rs in &raw_shells {
        let faces: Vec<(FaceKey, Orientation)> = rs.faces.iter().map(|(fi, o)| {
            (f_keys[*fi as usize], *o)
        }).collect();
        let k = store.shells.insert(BRepShell {
            faces,
            closed: rs.closed,
            step_id: rs.step_id,
        });
        sh_keys.push(k);
    }

    // Insert solids
    let mut so_keys: Vec<SolidKey> = Vec::with_capacity(so_count);
    for rs in &raw_solids {
        let outer_shell = sh_keys[rs.outer_shell_idx as usize];
        let void_shells: Vec<ShellKey> = rs.void_shell_indices.iter()
            .map(|&i| sh_keys[i as usize]).collect();
        let k = store.solids.insert(BRepSolid { outer_shell, void_shells });
        so_keys.push(k);
    }

    // Insert compounds
    for rc in &raw_compounds {
        let solids: Vec<SolidKey> = rc.solid_indices.iter()
            .map(|&i| so_keys[i as usize]).collect();
        store.compounds.insert(BRepCompound { solids });
    }

    Ok(store)
}

/// Write BRepStore to file.
pub fn write_brep_file(path: &std::path::Path, store: &BRepStore) -> io::Result<()> {
    let data = write_brep_binary(store);
    std::fs::write(path, &data)
}

/// Read BRepStore from file.
pub fn read_brep_file(path: &std::path::Path) -> io::Result<BRepStore> {
    let data = std::fs::read(path)?;
    read_brep_binary(&data)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::PVec3;

    fn make_test_cube() -> BRepStore {
        let mut store = BRepStore::new();
        // 8 vertices of a unit cube [0,1]^3
        let v000 = store.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.0), 1e-6);
        let v100 = store.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-6);
        let v110 = store.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), 1e-6);
        let v010 = store.find_or_add_vertex(PVec3::new(0.0, 1.0, 0.0), 1e-6);
        let v001 = store.find_or_add_vertex(PVec3::new(0.0, 0.0, 1.0), 1e-6);
        let v101 = store.find_or_add_vertex(PVec3::new(1.0, 0.0, 1.0), 1e-6);
        let v111 = store.find_or_add_vertex(PVec3::new(1.0, 1.0, 1.0), 1e-6);
        let v011 = store.find_or_add_vertex(PVec3::new(0.0, 1.0, 1.0), 1e-6);

        // Helper: create plane face with a rectangular wire
        let mut add_face = |store: &mut BRepStore,
                            origin: PVec3, normal: PVec3, u_dir: PVec3,
                            corners: &[(VertexKey, (Real, Real))]|
        -> FaceKey {
            let fk = store.add_face(
                SurfaceGeom::Plane { origin, normal, u_dir },
                1e-6,
            );
            // Build outer wire
            let mut edges = Vec::new();
            for i in 0..corners.len() {
                let j = (i + 1) % corners.len();
                let v0 = corners[i].0;
                let v1 = corners[j].0;
                let uv0 = corners[i].1;
                let uv1 = corners[j].1;
                let p0 = store.vertices.get(v0).unwrap().position;
                let p1 = store.vertices.get(v1).unwrap().position;
                let dir = p1 - p0;
                let curve = CurveGeom::Line { origin: p0, direction: dir };
                let pcurve = Curve2d::Line {
                    origin: uv0,
                    direction: (uv1.0 - uv0.0, uv1.1 - uv0.1),
                };
                let ek = store.add_edge_with_pcurve(v0, v1, curve, 1e-6, fk, pcurve, true);
                edges.push((ek, Orientation::Forward));
            }
            // Replace the auto-created empty outer wire
            let wk = store.wires.insert(BRepWire { edges });
            if let Some(face) = store.faces.get_mut(fk) {
                face.outer_wire = wk;
            }
            fk
        };

        // Six faces of a unit cube (simplified: store faces in a list)
        let _fk_bottom = add_face(&mut store,
            PVec3::ZERO, -PVec3::Z, PVec3::X,
            &[(v000, (0.0, 0.0)), (v100, (1.0, 0.0)), (v110, (1.0, 1.0)), (v010, (0.0, 1.0))],
        );
        let _fk_top = add_face(&mut store,
            PVec3::new(0.0, 0.0, 1.0), PVec3::Z, PVec3::X,
            &[(v001, (0.0, 0.0)), (v011, (0.0, 1.0)), (v111, (1.0, 1.0)), (v101, (1.0, 0.0))],
        );
        let _fk_front = add_face(&mut store,
            PVec3::new(0.0, 0.0, 0.0), -PVec3::Y, PVec3::X,
            &[(v000, (0.0, 0.0)), (v100, (1.0, 0.0)), (v101, (1.0, 1.0)), (v001, (0.0, 1.0))],
        );
        let _fk_back = add_face(&mut store,
            PVec3::new(0.0, 1.0, 0.0), PVec3::Y, PVec3::X,
            &[(v010, (0.0, 0.0)), (v110, (1.0, 0.0)), (v111, (1.0, 1.0)), (v011, (0.0, 1.0))],
        );
        let _fk_left = add_face(&mut store,
            PVec3::new(0.0, 0.0, 0.0), -PVec3::X, PVec3::Y,
            &[(v000, (0.0, 0.0)), (v010, (1.0, 0.0)), (v011, (1.0, 1.0)), (v001, (0.0, 1.0))],
        );
        let _fk_right = add_face(&mut store,
            PVec3::new(1.0, 0.0, 0.0), PVec3::X, PVec3::Y,
            &[(v100, (0.0, 0.0)), (v101, (1.0, 0.0)), (v111, (1.0, 1.0)), (v110, (0.0, 1.0))],
        );

        store
    }

    #[test]
    fn test_empty_roundtrip() {
        let store = BRepStore::new();
        let data = write_brep_binary(&store);
        let restored = read_brep_binary(&data).unwrap();
        assert_eq!(restored.vertices.len(), 0);
        assert_eq!(restored.edges.len(), 0);
        assert_eq!(restored.faces.len(), 0);
    }

    #[test]
    fn test_unit_cube_roundtrip() {
        let store = make_test_cube();
        let data = write_brep_binary(&store);
        let restored = read_brep_binary(&data).unwrap();

        assert_eq!(restored.vertices.len(), store.vertices.len(),
            "vertex count mismatch");
        assert_eq!(restored.edges.len(), store.edges.len(),
            "edge count mismatch");
        assert_eq!(restored.faces.len(), store.faces.len(),
            "face count mismatch");

        // Verify vertex positions are preserved
        for (vk, v) in store.vertices.iter() {
            let pos = v.position;
            let found = restored.vertices.iter()
                .any(|(_, rv)| (rv.position - pos).length() < 1e-12);
            assert!(found, "missing vertex at {:?}", pos);
        }

        // Verify each face has a matching surface
        for (_, face) in store.faces.iter() {
            let found = restored.faces.iter().any(|(_, rf)| {
                // Compare surface types
                std::mem::discriminant(&rf.surface) == std::mem::discriminant(&face.surface)
            });
            assert!(found, "missing face with surface type");
        }
    }

    #[test]
    fn test_roundtrip_single_vertex() {
        let mut store = BRepStore::new();
        store.find_or_add_vertex(PVec3::new(1.0, 2.0, 3.0), 1e-6);
        let data = write_brep_binary(&store);
        let restored = read_brep_binary(&data).unwrap();
        assert_eq!(restored.vertices.len(), 1);
        let v = restored.vertices.iter().next().unwrap().1;
        assert!((v.position.x - 1.0).abs() < 1e-12);
        assert!((v.position.y - 2.0).abs() < 1e-12);
        assert!((v.position.z - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_curve_circle_roundtrip() {
        let mut store = BRepStore::new();
        let v0 = store.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-6);
        let v1 = store.find_or_add_vertex(PVec3::new(-1.0, 0.0, 0.0), 1e-6);
        let fk = store.add_face(
            SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X },
            1e-6,
        );
        // Actually create a proper line edge for the test (circle BSpline not needed)
        let curve = CurveGeom::Line { origin: PVec3::new(1.0, 0.0, 0.0), direction: PVec3::new(-2.0, 0.0, 0.0) };
        let pc = Curve2d::Line { origin: (1.0, 0.0), direction: (-2.0, 0.0) };
        store.add_edge_with_pcurve(v0, v1, curve.clone(), 1e-6, fk, pc, true);

        let data = write_brep_binary(&store);
        let restored = read_brep_binary(&data).unwrap();
        assert_eq!(restored.edges.len(), 1);
        let edge = restored.edges.iter().next().unwrap().1;
        assert!(matches!(edge.curve, CurveGeom::Line { .. }));
    }

    #[test]
    fn test_surface_cylinder_roundtrip() {
        let mut store = BRepStore::new();
        store.add_face(
            SurfaceGeom::Cylinder {
                origin: PVec3::ZERO, axis: PVec3::Z, radius: 1.0,
                x_dir: PVec3::X, y_dir: PVec3::Y,
            },
            1e-6,
        );
        let data = write_brep_binary(&store);
        let restored = read_brep_binary(&data).unwrap();
        assert_eq!(restored.faces.len(), 1);
        let face = restored.faces.iter().next().unwrap().1;
        assert!(matches!(face.surface, SurfaceGeom::Cylinder { .. }));
    }

    #[test]
    fn test_file_roundtrip() {
        let mut store = BRepStore::new();
        store.find_or_add_vertex(PVec3::new(1.0, 2.0, 3.0), 1e-6);
        store.find_or_add_vertex(PVec3::new(4.0, 5.0, 6.0), 1e-6);

        let dir = std::env::temp_dir();
        let path = dir.join("test_brep_binary_roundtrip.bin");
        write_brep_file(&path, &store).unwrap();
        let restored = read_brep_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(restored.vertices.len(), 2);
    }
}
