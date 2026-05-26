//! BRepMesh_Face param-domain fill — OCC BRepMesh_Delaun stand-in (earcut P0).
//! Maps trim loops in UV -> Geom_Surface::Value(u,v).

use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;

use super::face_cdt::triangulate_uv_cdt_with_steiner;
use super::face_uv::FaceUvLoops;
use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::topo::{BRepFace, FaceKey};

#[derive(Debug, Clone)]
pub struct FaceFillConfig {
    pub enable_interior: bool,
    /// IMeshTools_Parameters::DeflectionInterior — max chord-to-surface gap (3D).
    pub deflection_interior: f32,
    /// IMeshTools_Parameters::MinSize — absolute minimum triangle edge length (3D).
    pub min_size: f32,
    /// Fraction of shell bbox diagonal used as MinSize floor when `relative_deflection > 0`.
    pub min_size_relative: f32,
    /// Set by shell mesher: shell_diag * min_size_relative when relative mode active.
    pub shell_min_size: f32,
    /// Max deflection-driven split iterations during fill (OCC node-insertion loop).
    pub max_adapt_iterations: usize,
}

impl Default for FaceFillConfig {
    fn default() -> Self {
        Self {
            enable_interior: true,
            deflection_interior: 0.05,
            min_size: 1e-3,
            min_size_relative: 0.001,
            shell_min_size: 0.0,
            max_adapt_iterations: 6,
        }
    }
}

pub struct FaceMeshRange {
    pub face_key: FaceKey,
    pub first_tri: usize,
    pub tri_count: usize,
    pub boundary_global: HashSet<usize>,
    pub max_chord_error: f32,
}

pub fn default_grid_for_surface(surface: &SurfaceGeom) -> (u32, u32) {
    match surface {
        SurfaceGeom::Plane { .. } => (1, 1),
        _ => (4, 4),
    }
}

/// Effective MinSize for a face (absolute floor + relative to boundary bbox diagonal).
pub fn effective_min_size(config: &FaceFillConfig) -> f32 {
    // MinSize floor tied to DeflectionInterior (OCC: min edge ~ few × deflection).
    let def_floor = config.deflection_interior * 2.0;
    let mut min = config.min_size.max(def_floor);
    if config.shell_min_size > 0.0 {
        min = min.max(config.shell_min_size);
    }
    min
}

/// CDT-first triangulation with Steiner refinement, earcut as ultimate fallback.
#[allow(clippy::too_many_arguments)]
pub fn fill_trimmed(
    face_key: FaceKey,
    loops: &FaceUvLoops,
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    config: &FaceFillConfig,
) -> FaceMeshRange {
    let first_tri = all_indices.len() / 4;
    let mut boundary_global = HashSet::new();
    for v in &loops.outer.boundary {
        boundary_global.insert(v.global_idx);
    }
    for inner in &loops.inners {
        for v in &inner.boundary {
            boundary_global.insert(v.global_idx);
        }
    }

    let mut tris: Vec<(i32, i32, i32)> = Vec::new();

    // --- Primary: CDT + insert-time Steiner refinement ---
    let (tris_flat, max_chord_error) = triangulate_uv_cdt_with_steiner(
        loops, face, global_vertices, global_normals, pos_to_idx, config,
    );
    for chunk in tris_flat.chunks(3) {
        if chunk.len() != 3 {
            continue;
        }
        tris.push((chunk[0] as i32, chunk[1] as i32, chunk[2] as i32));
    }

    // --- Fallback: earcut if CDT produced no triangles ---
    if tris.is_empty() {
        let mut earcut_verts: Vec<(usize, (f32, f32))> = Vec::new();
        for v in &loops.outer.boundary {
            earcut_verts.push((v.global_idx, v.uv));
        }
        let mut hole_indices = Vec::new();
        for inner in &loops.inners {
            hole_indices.push(earcut_verts.len());
            for v in &inner.boundary {
                earcut_verts.push((v.global_idx, v.uv));
            }
        }

        if earcut_verts.len() >= 3 {
            let flat: Vec<f64> = earcut_verts
                .iter()
                .flat_map(|&(_, (u, v))| [u as f64, v as f64])
                .collect();

            let ear_indices = match earcutr::earcut(&flat, &hole_indices, 2) {
                Ok(indices) if !indices.is_empty() => indices,
                _ => {
                    let g0 = earcut_verts[0].0 as i32;
                    for i in 1..earcut_verts.len() - 1 {
                        tris.push((g0, earcut_verts[i].0 as i32, earcut_verts[i + 1].0 as i32));
                    }
                    vec![]
                }
            };

            if !ear_indices.is_empty() {
                for chunk in ear_indices.chunks(3) {
                    if chunk.len() != 3 {
                        continue;
                    }
                    let (i0, i1, i2) = (chunk[0], chunk[1], chunk[2]);
                    if i0 >= earcut_verts.len()
                        || i1 >= earcut_verts.len()
                        || i2 >= earcut_verts.len()
                    {
                        continue;
                    }
                    tris.push((
                        earcut_verts[i0].0 as i32,
                        earcut_verts[i1].0 as i32,
                        earcut_verts[i2].0 as i32,
                    ));
                }
            }
        }

        if tris.is_empty() {
            log::debug!("[BRep mesh] face fill: triangulation failed, skipping");
            return FaceMeshRange {
                face_key,
                first_tri,
                tri_count: 0,
                boundary_global,
                max_chord_error: 0.0,
            };
        }
    }

    for (mut i0, mut i1, mut i2) in tris {
        if i0 == i1 || i1 == i2 || i2 == i0 {
            continue;
        }
        fix_winding(
            &mut i0,
            &mut i1,
            &mut i2,
            global_vertices,
            &face.surface,
            face.same_sense,
        );
        all_indices.extend_from_slice(&[i0, i1, i2, -1]);
        accumulate_normals(i0, i1, i2, global_vertices, global_normals);
    }

    let tri_count = all_indices.len() / 4 - first_tri;
    FaceMeshRange {
        face_key,
        first_tri,
        tri_count,
        boundary_global,
        max_chord_error,
    }
}

pub fn grid_fallback_3d(
    face_key: FaceKey,
    boundary_global: &[usize],
    _face: &BRepFace,
    global_vertices: &[Vec3],
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
) -> FaceMeshRange {
    let first_tri = all_indices.len() / 4;
    let boundary_set: HashSet<usize> = boundary_global.iter().copied().collect();

    if boundary_global.len() < 3 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: boundary_set,
            max_chord_error: 0.0,
        };
    }

    log::debug!(
        "[BRep mesh] face {:?}: grid fallback 3D ({} boundary verts)",
        face_key,
        boundary_global.len()
    );

    let points: Vec<Vec3> = boundary_global
        .iter()
        .map(|&gi| global_vertices[gi])
        .collect();

    let mut normal = Vec3::ZERO;
    for i in 2..points.len() {
        let cross = (points[1] - points[0]).cross(points[i] - points[0]);
        if cross.length() > 1e-10 {
            normal = cross;
            break;
        }
    }
    if normal.length() < 1e-10 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: boundary_set,
            max_chord_error: 0.0,
        };
    }
    normal = normal.normalize();
    let u_axis = if normal.dot(Vec3::Z).abs() < 0.9 {
        normal.cross(Vec3::Z).normalize()
    } else {
        normal.cross(Vec3::Y).normalize()
    };
    let v_axis = normal.cross(u_axis).normalize();
    let origin = points[0];

    let flat: Vec<f64> = points
        .iter()
        .flat_map(|p| {
            let rel = *p - origin;
            [rel.dot(u_axis) as f64, rel.dot(v_axis) as f64]
        })
        .collect();

    let ear_indices = match earcutr::earcut(&flat, &[], 2) {
        Ok(indices) if !indices.is_empty() => indices,
        _ => {
            let g0 = boundary_global[0] as i32;
            for i in 1..boundary_global.len() - 1 {
                all_indices.extend_from_slice(&[
                    g0,
                    boundary_global[i] as i32,
                    boundary_global[i + 1] as i32,
                    -1,
                ]);
            }
            let tri_count = all_indices.len() / 4 - first_tri;
            return FaceMeshRange {
                face_key,
                first_tri,
                tri_count,
                boundary_global: boundary_set,
                max_chord_error: 0.0,
            };
        }
    };

    for chunk in ear_indices.chunks(3) {
        if chunk.len() != 3 {
            continue;
        }
        let (i0, i1, i2) = (
            boundary_global[chunk[0]] as i32,
            boundary_global[chunk[1]] as i32,
            boundary_global[chunk[2]] as i32,
        );
        all_indices.extend_from_slice(&[i0, i1, i2, -1]);
        accumulate_normals(i0, i1, i2, global_vertices, global_normals);
    }

    let tri_count = all_indices.len() / 4 - first_tri;
    FaceMeshRange {
        face_key,
        first_tri,
        tri_count,
        boundary_global: boundary_set,
        max_chord_error: 0.0,
    }
}

pub fn measure_face_chord_error(
    face: &BRepFace,
    global_vertices: &[Vec3],
    all_indices: &[i32],
    range: &FaceMeshRange,
) -> f32 {
    let inv_tol = face.tolerance.max(1e-3);
    let start = range.first_tri * 4;
    let end = start + range.tri_count * 4;
    let mut max_chord = 0.0f32;
    let slice = all_indices.get(start..end.min(all_indices.len()));
    if let Some(tris) = slice {
        for chunk in tris.chunks(4) {
            if chunk.len() < 3 {
                continue;
            }
            max_chord = max_chord.max(tri_max_chord_error(
                chunk[0],
                chunk[1],
                chunk[2],
                global_vertices,
                &face.surface,
                inv_tol,
            ));
        }
    }
    max_chord
}

fn surface_point_deviation(p: Vec3, surface: &SurfaceGeom, inv_tol: f32) -> f32 {
    let snap_tol = inv_tol.max(0.1);
    if let Some((u, v)) = surface.inverse_native_uv(p, snap_tol) {
        let d = (p - surface.d0_at_native_uv(u, v)).length();
        if d <= snap_tol {
            return d;
        }
    }
    if let Some((u, v)) = surface.project(p) {
        let d = (p - surface.d0_at_native_uv(u, v)).length();
        if d <= snap_tol * 4.0 {
            return d;
        }
    }
    if let Some((u, v)) = surface.inverse_native_uv_build(p, snap_tol * 4.0) {
        let d = (p - surface.d0_at_native_uv(u, v)).length();
        if d <= snap_tol * 4.0 {
            return d;
        }
    }
    0.0
}

fn tri_max_chord_error(
    i0: i32,
    i1: i32,
    i2: i32,
    verts: &[Vec3],
    surface: &SurfaceGeom,
    inv_tol: f32,
) -> f32 {
    let p0 = verts[i0 as usize];
    let p1 = verts[i1 as usize];
    let p2 = verts[i2 as usize];
    let mut max_dev = 0.0f32;
    for p in [p0, p1, p2] {
        max_dev = max_dev.max(surface_point_deviation(p, surface, inv_tol));
    }
    for (a, b) in [(p0, p1), (p1, p2), (p2, p0)] {
        let mid = (a + b) * 0.5;
        max_dev = max_dev.max(surface_point_deviation(mid, surface, inv_tol));
    }
    let centroid = (p0 + p1 + p2) * (1.0 / 3.0);
    max_dev = max_dev.max(surface_point_deviation(centroid, surface, inv_tol));
    max_dev
}

fn fix_winding(
    i0: &mut i32,
    i1: &mut i32,
    i2: &mut i32,
    global_vertices: &[Vec3],
    surface: &SurfaceGeom,
    same_sense: bool,
) {
    let p0 = global_vertices[*i0 as usize];
    let p1 = global_vertices[*i1 as usize];
    let p2 = global_vertices[*i2 as usize];
    let tri_n = (p1 - p0).cross(p2 - p0);
    if tri_n.length() <= 1e-10 {
        return;
    }
    let centroid = (p0 + p1 + p2) * (1.0 / 3.0);
    if let Some((uc, vc)) = surface.project(centroid) {
        let mut face_n = surface.normal_native(uc, vc);
        if !same_sense {
            face_n = -face_n;
        }
        if tri_n.dot(face_n) < 0.0 {
            std::mem::swap(i1, i2);
        }
    }
}

fn accumulate_normals(
    i0: i32,
    i1: i32,
    i2: i32,
    global_vertices: &[Vec3],
    global_normals: &mut [Vec3],
) {
    let p0 = global_vertices[i0 as usize];
    let p1 = global_vertices[i1 as usize];
    let p2 = global_vertices[i2 as usize];
    let tri_n = (p1 - p0).cross(p2 - p0);
    if tri_n.length() > 1e-10 {
        let n = tri_n.normalize();
        global_normals[i0 as usize] = global_normals[i0 as usize] + n;
        global_normals[i1 as usize] = global_normals[i1 as usize] + n;
        global_normals[i2 as usize] = global_normals[i2 as usize] + n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::mesh::face_uv::{point_in_trim, signed_area_2d, UvLoop, UvSource, UvVertex};
    use crate::step::brep::topo::{BRepFace, FaceKey};

    #[test]
    fn fill_trimmed_plane_no_tris_inside_hole() {
        let outer = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 0, uv: (0.0, 0.0) },
                UvVertex { global_idx: 1, uv: (4.0, 0.0) },
                UvVertex { global_idx: 2, uv: (4.0, 4.0) },
                UvVertex { global_idx: 3, uv: (0.0, 4.0) },
            ],
        };
        let inner = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 4, uv: (1.0, 1.0) },
                UvVertex { global_idx: 5, uv: (3.0, 1.0) },
                UvVertex { global_idx: 6, uv: (3.0, 3.0) },
                UvVertex { global_idx: 7, uv: (1.0, 3.0) },
            ],
        };
        let loops = FaceUvLoops {
            outer,
            inners: vec![inner],
            uv_source: UvSource::Pcurve,
        };
        let face = BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: Default::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
        };
        let mut verts: Vec<Vec3> = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(4.0, 4.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
            Vec3::new(3.0, 3.0, 0.0),
            Vec3::new(1.0, 3.0, 0.0),
        ];
        let mut norms = vec![Vec3::Z; 8];
        let mut indices = Vec::new();
        let mut pos_map = HashMap::new();
        let fk = FaceKey::default();
        let range = fill_trimmed(
            fk,
            &loops,
            &face,
            &mut verts,
            &mut norms,
            &mut indices,
            &mut pos_map,
            &FaceFillConfig {
                enable_interior: false,
                ..Default::default()
            },
        );
        assert!(range.tri_count > 0);
        let hole = vec![(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)];
        let outer_poly = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let uv_map: HashMap<usize, (f32, f32)> = loops
            .outer
            .boundary
            .iter()
            .chain(loops.inners.iter().flat_map(|l| l.boundary.iter()))
            .map(|v| (v.global_idx, v.uv))
            .collect();
        for chunk in indices.chunks(4) {
            if chunk.len() < 4 {
                continue;
            }
            let uv = |gi: i32| uv_map[&(gi as usize)];
            let cu = (uv(chunk[0]).0 + uv(chunk[1]).0 + uv(chunk[2]).0) / 3.0;
            let cv = (uv(chunk[0]).1 + uv(chunk[1]).1 + uv(chunk[2]).1) / 3.0;
            assert!(
                point_in_trim(cu, cv, &outer_poly, &[hole.clone()]),
                "triangle centroid ({cu},{cv}) inside hole"
            );
        }
    }

    #[test]
    fn signed_area_used_for_orientation() {
        let cw = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
        assert!(signed_area_2d(&cw) < 0.0);
    }
}
