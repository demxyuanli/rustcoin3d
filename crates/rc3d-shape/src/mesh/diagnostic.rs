//! Optional B-Rep mesh coordinate dump for debugging bad triangulation.
//! Enable: set env `RC3D_BREP_MESH_LOG=1` (stderr) or `RC3D_BREP_MESH_LOG=path/to/dump.txt`.

use std::io::Write;

use rc3d_core::math::{Real, PVec3};

use super::edge_pool::FaceEdgeBoundaryIdx;
use super::face_fill::FaceMeshRange;
use super::face_uv::FaceUvLoops;
use super::report::ShellMeshReport;
use crate::geom::SurfaceGeom;
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, ShellKey};

pub fn surface_kind(surface: &SurfaceGeom) -> &'static str {
    match surface {
        SurfaceGeom::Plane { .. } => "Plane",
        SurfaceGeom::Cylinder { .. } => "Cylinder",
        SurfaceGeom::Cone { .. } => "Cone",
        SurfaceGeom::Sphere { .. } => "Sphere",
        SurfaceGeom::Torus { .. } => "Torus",
        SurfaceGeom::BSpline(_) => "BSpline",
        SurfaceGeom::Extrusion { .. } => "Extrusion",
        SurfaceGeom::Revolution { .. } => "Revolution",
        SurfaceGeom::Offset { .. } => "Offset",
    }
}

fn fmt_v(p: PVec3) -> String {
    format!("({:.6},{:.6},{:.6})", p.x, p.y, p.z)
}

fn tri_edges(a: PVec3, b: PVec3, c: PVec3) -> (Real, Real, Real) {
    ((b - a).length(), (c - b).length(), (a - c).length())
}

fn median(mut xs: Vec<Real>) -> Real {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

pub fn diag_enabled() -> bool {
    match std::env::var("RC3D_BREP_MESH_LOG") {
        Ok(s) if !s.is_empty() && s != "0" => true,
        _ => false,
    }
}

/// Wire-loop diagnostics: consecutive 3D edge lengths, z range, jump edges.
pub fn format_wire_loop_lines(
    face_key: FaceKey,
    loops: &FaceUvLoops,
    wire_edges: &[(EdgeKey, crate::topo::Orientation, Vec<usize>)],
    edge_boundary_idx: &FaceEdgeBoundaryIdx,
    vertices: &[PVec3],
) -> Vec<String> {
    let mut lines = Vec::new();
    let boundary = &loops.outer.boundary;
    if boundary.is_empty() {
        return lines;
    }

    let mut z_min = f64::MAX;
    let mut z_max = f64::MIN;
    for v in boundary {
        if let Some(p) = vertices.get(v.global_idx) {
            z_min = z_min.min(p.z);
            z_max = z_max.max(p.z);
        }
    }

    let mut edge_lens = Vec::new();
    for i in 0..boundary.len() {
        let j = (i + 1) % boundary.len();
        let len = if boundary[i].global_idx < vertices.len()
            && boundary[j].global_idx < vertices.len()
        {
            (vertices[boundary[j].global_idx] - vertices[boundary[i].global_idx]).length()
        } else {
            0.0
        };
        edge_lens.push(len);
    }
    let med = median(edge_lens.clone());
    let jump_thresh = (med * 6.0).max(med + 0.5).max(1.0);

    lines.push(format!(
        "[BRep mesh diag] wire face {face_key:?} boundary_pts={} z_range=({z_min:.6},{z_max:.6}) median_edge={med:.6} jump_thresh={jump_thresh:.6}",
        boundary.len(),
    ));

    for (ek, _orient, pis) in wire_edges {
        let mut seg_lens = Vec::new();
        for w in pis.windows(2) {
            let gi0 = edge_boundary_idx.get(&(face_key, *ek, w[0])).copied();
            let gi1 = edge_boundary_idx.get(&(face_key, *ek, w[1])).copied();
            if let (Some(g0), Some(g1)) = (gi0, gi1) {
                if g0 < vertices.len() && g1 < vertices.len() {
                    seg_lens.push((vertices[g1] - vertices[g0]).length());
                }
            }
        }
        let seg_max = seg_lens.iter().copied().fold(0.0_f64, Real::max);
        lines.push(format!(
            "[BRep mesh diag]   wire edge {ek:?} pts={} seg_max={seg_max:.6}",
            pis.len(),
        ));
    }

    for i in 0..boundary.len() {
        let j = (i + 1) % boundary.len();
        let gi0 = boundary[i].global_idx;
        let gi1 = boundary[j].global_idx;
        let (p0, p1) = match (vertices.get(gi0), vertices.get(gi1)) {
            (Some(a), Some(b)) => (*a, *b),
            _ => continue,
        };
        let len = edge_lens[i];
        let jump = len > jump_thresh;
        if jump || i < 8 || i + 8 >= boundary.len() {
            lines.push(format!(
                "[BRep mesh diag]   boundary[{i}->{j}] gi=({gi0},{gi1}) len={len:.6} jump={jump} z=({:.6},{:.6})",
                p0.z, p1.z,
            ));
        }
    }

    lines
}

struct DiagWriter {
    file: Option<std::fs::File>,
}

impl DiagWriter {
    fn open() -> Option<Self> {
        let spec = std::env::var("RC3D_BREP_MESH_LOG").ok()?;
        if spec.is_empty() || spec == "0" {
            return None;
        }
        if spec == "1" || spec.eq_ignore_ascii_case("true") {
            return Some(Self { file: None });
        }
        std::fs::File::create(&spec).ok().map(|file| Self {
            file: Some(file),
        })
    }

    fn writeln(&mut self, line: &str) {
        if let Some(f) = &mut self.file {
            let _ = writeln!(f, "{line}");
        } else {
            eprintln!("{line}");
        }
    }
}

/// Dump all vertex coordinates and per-triangle vertex coordinates when env is set.
pub fn log_mesh_coordinates_if_requested(
    shell_key: ShellKey,
    reg: &BRepStore,
    vertices: &[PVec3],
    indices: &[i32],
    face_ranges: &[FaceMeshRange],
    report: &ShellMeshReport,
    wire_diag: &[String],
) {
    let Some(mut out) = DiagWriter::open() else {
        return;
    };

    if !wire_diag.is_empty() {
        out.writeln("[BRep mesh diag] --- wire loops ---");
        for line in wire_diag {
            out.writeln(line);
        }
    }

    let tri_total = indices.len() / 4;
    out.writeln(&format!(
        "[BRep mesh diag] shell {shell_key:?} vertices={} triangles={} meshed_faces={}/{}",
        vertices.len(),
        tri_total,
        report.meshed_faces,
        report.face_count,
    ));

    out.writeln("[BRep mesh diag] --- all vertices ---");
    for (i, p) in vertices.iter().enumerate() {
        out.writeln(&format!("[BRep mesh diag] vertex[{i}]: {}", fmt_v(*p)));
    }

    out.writeln("[BRep mesh diag] --- per-face triangle coordinates ---");
    for range in face_ranges {
        let face = match reg.faces.get(range.face_key) {
            Some(f) => f,
            None => continue,
        };
        let kind = surface_kind(&face.surface);
        let start = range.first_tri * 4;
        let end = start + range.tri_count * 4;
        if end > indices.len() {
            continue;
        }

        let mut edge_lens = Vec::new();
        let mut max_edge = 0.0_f64;
        for chunk in indices[start..end].chunks(4) {
            if chunk.len() < 4 {
                continue;
            }
            let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
            if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
                continue;
            }
            let (e0, e1, e2) = tri_edges(vertices[i0], vertices[i1], vertices[i2]);
            max_edge = max_edge.max(e0).max(e1).max(e2);
            edge_lens.push(e0);
            edge_lens.push(e1);
            edge_lens.push(e2);
        }
        let med = median(edge_lens);
        let needle = med > 1e-6 && max_edge > med * 8.0;

        out.writeln(&format!(
            "[BRep mesh diag] face {:?} surface={kind} tris={} max_edge={:.6} median_edge={:.6} needle_like={needle}",
            range.face_key,
            range.tri_count,
            max_edge,
            med,
        ));

        let mut local_tri = 0usize;
        for chunk in indices[start..end].chunks(4) {
            if chunk.len() < 4 {
                continue;
            }
            let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
            if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
                continue;
            }
            let p0 = vertices[i0];
            let p1 = vertices[i1];
            let p2 = vertices[i2];
            let (e0, e1, e2) = tri_edges(p0, p1, p2);
            out.writeln(&format!(
                "[BRep mesh diag]   face {:?} tri[{local_tri}]: idx=({i0},{i1},{i2}) v0={} v1={} v2={} edges=({e0:.6},{e1:.6},{e2:.6})",
                range.face_key,
                fmt_v(p0),
                fmt_v(p1),
                fmt_v(p2),
            ));
            local_tri += 1;
        }
    }

    out.writeln("[BRep mesh diag] --- flat triangle list ---");
    for t in 0..tri_total {
        let base = t * 4;
        let Some(chunk) = indices.get(base..base + 4) else {
            break;
        };
        if chunk.len() < 4 {
            break;
        }
        let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        out.writeln(&format!(
            "[BRep mesh diag] tri[{t}]: ({i0},{i1},{i2}) {} {} {}",
            fmt_v(vertices[i0]),
            fmt_v(vertices[i1]),
            fmt_v(vertices[i2]),
        ));
    }

    out.writeln("[BRep mesh diag] --- end dump ---");
}
