//! Constrained Delaunay triangulation for trimmed faces (spade CDT, earcut fallback).
//! OCC BRepMesh_Delaun stand-in: CDT-first + insert-time Steiner refinement.

use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;
use rc3d_core::utils::hash::f32x3_quantized_bits;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use super::face_fill::{effective_min_size, FaceFillConfig};
use super::face_uv::{point_in_trim, FaceUvLoops, UvSource};
use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::BRepFace;

/// Hard cap on CDT vertices per face (boundary + Steiner).
const MAX_CDT_VERTICES: usize = 4096;

fn uv_quant_key(uv: (f32, f32)) -> (u64, u64) {
    ((uv.0 * 1e6).round() as u64, (uv.1 * 1e6).round() as u64)
}

fn insert_uv(
    cdt: &mut ConstrainedDelaunayTriangulation<Point2<f64>>,
    mut uv: (f32, f32),
    gi: usize,
    handles: &mut Vec<spade::handles::FixedVertexHandle>,
    handle_gi: &mut Vec<usize>,
    uv_to_handle: &mut HashMap<(u64, u64), usize>,
) -> Option<spade::handles::FixedVertexHandle> {
    for bump in 0..32usize {
        let key = uv_quant_key(uv);
        if let Some(&hi) = uv_to_handle.get(&key) {
            if handle_gi[hi] == gi {
                return Some(handles[hi]);
            }
            uv.0 += 1e-5 * (bump as f32 + 1.0);
            continue;
        }
        let pt = Point2::new(uv.0 as f64, uv.1 as f64);
        let Ok(h) = cdt.insert(pt) else {
            return None;
        };
        let hi = handles.len();
        handles.push(h);
        handle_gi.push(gi);
        uv_to_handle.insert(key, hi);
        return Some(h);
    }
    None
}

fn extract_cdt_triangles(
    cdt: &ConstrainedDelaunayTriangulation<Point2<f64>>,
    handles: &[spade::handles::FixedVertexHandle],
    handle_gi: &[usize],
    filter: impl Fn(f32, f32) -> bool,
) -> Vec<usize> {
    let mut tris = Vec::new();
    for face_h in cdt.inner_faces() {
        let verts: Vec<_> = face_h
            .vertices()
            .iter()
            .map(|v| v.fix().index())
            .collect();
        if verts.len() != 3 {
            continue;
        }
        let i0 = verts[0];
        let i1 = verts[1];
        let i2 = verts[2];
        if i0 >= handle_gi.len() || i1 >= handle_gi.len() || i2 >= handle_gi.len() {
            continue;
        }
        let uv0 = (
            cdt.vertex(handles[i0]).position().x as f32,
            cdt.vertex(handles[i0]).position().y as f32,
        );
        let uv1 = (
            cdt.vertex(handles[i1]).position().x as f32,
            cdt.vertex(handles[i1]).position().y as f32,
        );
        let uv2 = (
            cdt.vertex(handles[i2]).position().x as f32,
            cdt.vertex(handles[i2]).position().y as f32,
        );
        let cu = (uv0.0 + uv1.0 + uv2.0) / 3.0;
        let cv = (uv0.1 + uv1.1 + uv2.1) / 3.0;
        if !filter(cu, cv) {
            continue;
        }
        tris.push(handle_gi[i0]);
        tris.push(handle_gi[i1]);
        tris.push(handle_gi[i2]);
    }
    tris
}

/// CDT-first triangulation with insert-time Steiner refinement.
/// Returns (flat list of global vertex indices per triangle, max_chord_error).
pub fn triangulate_uv_cdt_with_steiner(
    loops: &FaceUvLoops,
    face: &BRepFace,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    config: &FaceFillConfig,
    reg: Option<&BRepRegistry>,
) -> (Vec<usize>, f32) {
    if loops.outer.boundary.len() < 3 {
        return (Vec::new(), 0.0);
    }

    let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
        ConstrainedDelaunayTriangulation::new();
    let mut handles: Vec<spade::handles::FixedVertexHandle> = Vec::new();
    let mut handle_gi: Vec<usize> = Vec::new();
    let mut uv_to_handle: HashMap<(u64, u64), usize> = HashMap::new();

    let mut outer_handles = Vec::new();
    for v in &loops.outer.boundary {
        let Some(h) = insert_uv(
            &mut cdt,
            v.uv,
            v.global_idx,
            &mut handles,
            &mut handle_gi,
            &mut uv_to_handle,
        ) else {
            return (Vec::new(), 0.0);
        };
        outer_handles.push(h);
    }
    if outer_handles.len() < 3 {
        return (Vec::new(), 0.0);
    }
    let n_outer = outer_handles.len();
    for i in 0..n_outer {
        let a = outer_handles[i];
        let b = outer_handles[(i + 1) % n_outer];
        let _ = cdt.try_add_constraint(a, b);
    }

    for inner in &loops.inners {
        let mut inner_handles = Vec::new();
        for v in &inner.boundary {
            let Some(h) = insert_uv(
                &mut cdt,
                v.uv,
                v.global_idx,
                &mut handles,
                &mut handle_gi,
                &mut uv_to_handle,
            ) else {
                return (Vec::new(), 0.0);
            };
            inner_handles.push(h);
        }
        let n = inner_handles.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = inner_handles[i];
            let b = inner_handles[(i + 1) % n];
            let before = cdt.num_constraints();
            let added = cdt.try_add_constraint(a, b);
            if added.is_empty() && cdt.num_constraints() == before && !cdt.exists_constraint(a, b) {
                return (Vec::new(), 0.0);
            }
        }
    }

    // Degenerated edges (from surface singularities like sphere poles / cone apex).
    // Insert degenerated edges as CDT constraints when PCurve UV extent is non-zero.
    if let Some(reg) = reg {
        for &dek in &face.degenerated_edges {
            let edge = match reg.edges.get(dek) {
                Some(e) => e,
                None => continue,
            };
            // Degenerated edges are created for this face; use the first available PCurve
            let pc = match edge.pcurves.values().next() {
                Some(p) => p,
                None => continue,
            };
            let uv0 = pc.d0(0.0);
            let uv1 = pc.d0(1.0);
            // Skip zero-length PCurves (legacy degenerated edges)
            if (uv0.x - uv1.x).abs() < 1e-10 && (uv0.y - uv1.y).abs() < 1e-10 {
                continue;
            }
            // Create a global vertex for this degenerated edge's UV endpoints if not already present
            let pt0 = face.surface.d0_native(uv0.x, uv0.y);
            let pt1 = face.surface.d0_native(uv1.x, uv1.y);
            let hash0 = f32x3_quantized_bits([pt0.x, pt0.y, pt0.z]);
            let hash1 = f32x3_quantized_bits([pt1.x, pt1.y, pt1.z]);
            let gi0 = *pos_to_idx.entry(hash0).or_insert_with(|| {
                let i = global_vertices.len();
                global_vertices.push(pt0);
                let mut n = face.surface.normal_native(uv0.x, uv0.y);
                if !face.same_sense { n = -n; }
                global_normals.push(n);
                i
            });
            let gi1 = *pos_to_idx.entry(hash1).or_insert_with(|| {
                let i = global_vertices.len();
                global_vertices.push(pt1);
                let mut n = face.surface.normal_native(uv1.x, uv1.y);
                if !face.same_sense { n = -n; }
                global_normals.push(n);
                i
            });
            let h0 = match insert_uv(&mut cdt, (uv0.x, uv0.y), gi0, &mut handles, &mut handle_gi, &mut uv_to_handle) {
                Some(h) => h,
                None => continue,
            };
            let h1 = match insert_uv(&mut cdt, (uv1.x, uv1.y), gi1, &mut handles, &mut handle_gi, &mut uv_to_handle) {
                Some(h) => h,
                None => continue,
            };
            let _ = cdt.try_add_constraint(h0, h1);
        }
    }

    let mut max_chord = 0.0f32;
    if config.enable_interior && config.deflection_interior > 0.0 {
        // Structured interior grid (Truck-style parameter_division) replaces edge-midpoint Steiner.
        let u_min = loops.outer.boundary.iter().map(|v| v.uv.0).fold(f32::INFINITY, f32::min);
        let u_max = loops.outer.boundary.iter().map(|v| v.uv.0).fold(f32::MIN, f32::max);
        let v_min = loops.outer.boundary.iter().map(|v| v.uv.1).fold(f32::INFINITY, f32::min);
        let v_max = loops.outer.boundary.iter().map(|v| v.uv.1).fold(f32::MIN, f32::max);
        let range = (u_min, u_max, v_min, v_max);
        let (u_divs, v_divs) = face.surface.parameter_division(range, config.deflection_interior);

        let outer_uv: Vec<(f32, f32)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
        let inner_uv: Vec<Vec<(f32, f32)>> = loops
            .inners
            .iter()
            .map(|l| l.boundary.iter().map(|v| v.uv).collect())
            .collect();

        let u_span = (u_max - u_min).abs();
        let v_span = (v_max - v_min).abs();
        let u_eps = (u_span * 1e-7).max(1e-6);
        let v_eps = (v_span * 1e-7).max(1e-6);

        let mut inserted = 0usize;
        for u in &u_divs {
            for v in &v_divs {
                if handles.len() >= MAX_CDT_VERTICES {
                    break;
                }
                // Skip boundary-adjacent points that would duplicate existing vertices
                let on_boundary = loops.outer.boundary.iter().any(|vb| {
                    (vb.uv.0 - u).abs() < u_eps && (vb.uv.1 - v).abs() < v_eps
                }) || loops.inners.iter().any(|inner| {
                    inner.boundary.iter().any(|vb| {
                        (vb.uv.0 - u).abs() < u_eps && (vb.uv.1 - v).abs() < v_eps
                    })
                });
                if on_boundary {
                    continue;
                }
                if point_in_trim(*u, *v, &outer_uv, &inner_uv) {
                    let pt_3d = face.surface.d0_native(*u, *v);
                    let hash = f32x3_quantized_bits([pt_3d.x, pt_3d.y, pt_3d.z]);
                    let gi = *pos_to_idx.entry(hash).or_insert_with(|| {
                        let i = global_vertices.len();
                        global_vertices.push(pt_3d);
                        let mut n = face.surface.normal_native(*u, *v);
                        if !face.same_sense {
                            n = -n;
                        }
                        global_normals.push(n);
                        i
                    });
                    if insert_uv(
                        &mut cdt,
                        (*u, *v),
                        gi,
                        &mut handles,
                        &mut handle_gi,
                        &mut uv_to_handle,
                    )
                    .is_some()
                    {
                        inserted += 1;
                    }
                }
            }
        }
        log::debug!(
            "[BRep mesh] structured grid inserted {} interior points (uv_divs={}x{})",
            inserted,
            u_divs.len(),
            v_divs.len()
        );

        // Light-weight post-check: if any interior triangle edge still deviates
        // significantly from the surface, insert the surface-projected midpoint.
        // Uses cheap d0_native(uv_mid) for deviation pre-check; only calls the
        // expensive project(mid_3d) when a split is actually needed.
        let min_sz = effective_min_size(config);
        for _iter in 0..1 {
            if handles.len() >= MAX_CDT_VERTICES {
                break;
            }
            let mut splits: Vec<(f64, f64)> = Vec::new();
            for face_h in cdt.inner_faces() {
                let verts: Vec<_> = face_h
                    .vertices()
                    .iter()
                    .map(|v| v.fix().index())
                    .collect();
                if verts.len() != 3 {
                    continue;
                }
                let i0 = verts[0];
                let i1 = verts[1];
                let i2 = verts[2];

                let uv0 = (
                    cdt.vertex(handles[i0]).position().x as f32,
                    cdt.vertex(handles[i0]).position().y as f32,
                );
                let uv1 = (
                    cdt.vertex(handles[i1]).position().x as f32,
                    cdt.vertex(handles[i1]).position().y as f32,
                );
                let uv2 = (
                    cdt.vertex(handles[i2]).position().x as f32,
                    cdt.vertex(handles[i2]).position().y as f32,
                );

                let p0 = face.surface.d0_native(uv0.0, uv0.1);
                let p1 = face.surface.d0_native(uv1.0, uv1.1);
                let p2 = face.surface.d0_native(uv2.0, uv2.1);
                let area_3d = (p1 - p0).cross(p2 - p0).length();
                if area_3d < 1e-12 {
                    continue;
                }

                for (a, b, uva, uvb) in [
                    (&p0, &p1, uv0, uv1),
                    (&p1, &p2, uv1, uv2),
                    (&p2, &p0, uv2, uv0),
                ] {
                    let edge_len = (*a - *b).length();
                    if edge_len <= min_sz {
                        continue;
                    }
                    let mid_3d = (*a + *b) * 0.5;
                    let uv_mid = ((uva.0 + uvb.0) * 0.5, (uva.1 + uvb.1) * 0.5);
                    let surf_mid = face.surface.d0_native(uv_mid.0, uv_mid.1);
                    let dev = (mid_3d - surf_mid).length();
                    max_chord = max_chord.max(dev);
                    let mut needs_split = false;
                    if dev > config.deflection_interior {
                        needs_split = true;
                    }
                    let na = face.surface.normal_native(uva.0, uva.1);
                    let nb = face.surface.normal_native(uvb.0, uvb.1);
                    let angle = na.normalize().dot(nb.normalize()).max(-1.0).min(1.0).acos();
                    if angle > config.angular_deflection {
                        needs_split = true;
                    }
                    if needs_split {
                        if let Some((u, v)) = face.surface.project(mid_3d) {
                            if point_in_trim(u, v, &outer_uv, &inner_uv) {
                                splits.push((u as f64, v as f64));
                            }
                        }
                    }
                }
            }
            if splits.is_empty() {
                break;
            }
            let mut dedup = HashSet::new();
            for (u, v) in splits {
                let key = ((u * 1e4) as u64, (v * 1e4) as u64);
                if dedup.contains(&key) {
                    continue;
                }
                dedup.insert(key);
                let pt_3d = face.surface.d0_native(u as f32, v as f32);
                let hash = f32x3_quantized_bits([pt_3d.x, pt_3d.y, pt_3d.z]);
                let gi = *pos_to_idx.entry(hash).or_insert_with(|| {
                    let i = global_vertices.len();
                    global_vertices.push(pt_3d);
                    let mut n = face.surface.normal_native(u as f32, v as f32);
                    if !face.same_sense {
                        n = -n;
                    }
                    global_normals.push(n);
                    i
                });
                insert_uv(
                    &mut cdt,
                    (u as f32, v as f32),
                    gi,
                    &mut handles,
                    &mut handle_gi,
                    &mut uv_to_handle,
                );
            }
        }
    }

    let outer_uv: Vec<(f32, f32)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
    let inner_uv: Vec<Vec<(f32, f32)>> = loops
        .inners
        .iter()
        .map(|l| l.boundary.iter().map(|v| v.uv).collect())
        .collect();

    let use_trim = loops.uv_source == UvSource::Pcurve;
    let mut tris = if use_trim {
        extract_cdt_triangles(&cdt, &handles, &handle_gi, |cu, cv| {
            point_in_trim(cu, cv, &outer_uv, &inner_uv)
        })
    } else {
        extract_cdt_triangles(&cdt, &handles, &handle_gi, |_, _| true)
    };
    if tris.is_empty() && use_trim {
        tris = extract_cdt_triangles(&cdt, &handles, &handle_gi, |_, _| true);
    }

    (tris, max_chord)
}

/// Backward-compat: UV-only CDT without surface/refinement (no Steiner points).
/// Use `triangulate_uv_cdt_with_steiner` for the full pipeline.
pub fn triangulate_uv_cdt(loops: &FaceUvLoops) -> Option<Vec<usize>> {
    if loops.outer.boundary.len() < 3 {
        return None;
    }
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
        color: None,
            degenerated_edges: vec![],
    };
    let mut verts = Vec::new();
    let mut norms = Vec::new();
    let mut pmap = HashMap::new();
    let config = FaceFillConfig {
        enable_interior: false,
        ..Default::default()
    };
    let (tris, _) = triangulate_uv_cdt_with_steiner(
        loops, &face, &mut verts, &mut norms, &mut pmap, &config, None,
    );
    if tris.is_empty() {
        None
    } else {
        Some(tris)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::face_uv::{UvLoop, UvSource, UvVertex};
    use crate::step::brep::geom::CurveGeom;

    #[test]
    fn steiner_split_on_curved_surface() {
        let surface = SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 10.0 };
        let face = BRepFace {
            surface,
            outer_wire: Default::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        };
        let outer = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 0, uv: (0.0, 1.5708) },
                UvVertex { global_idx: 1, uv: (1.5708, 1.5708) },
                UvVertex { global_idx: 2, uv: (1.5708, 2.5708) },
                UvVertex { global_idx: 3, uv: (0.0, 2.5708) },
            ],
        };
        let loops = FaceUvLoops { outer, inners: vec![], uv_source: UvSource::Pcurve };

        let mut verts = vec![
            face.surface.d0_native(0.0, 1.5708),
            face.surface.d0_native(1.5708, 1.5708),
            face.surface.d0_native(1.5708, 2.5708),
            face.surface.d0_native(0.0, 2.5708),
        ];
        let mut norms = vec![Vec3::Z; 4];
        let mut pos_map = HashMap::new();
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(f32x3_quantized_bits([v.x, v.y, v.z]), i);
        }

        let config_loose = FaceFillConfig {
            enable_interior: true,
            deflection_interior: 0.5,
            min_size: 0.1,
            min_size_relative: 0.0,
            shell_min_size: 0.0,
            max_adapt_iterations: 4,
            angular_deflection: std::f32::consts::PI,
        };

        let (tris_loose, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, &mut verts, &mut norms, &mut pos_map, &config_loose, None,
        );
        let verts_before = verts.len();

        let config_tight = FaceFillConfig {
            deflection_interior: 0.01,
            max_adapt_iterations: 4,
            ..config_loose.clone()
        };
        let (tris_tight, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, &mut verts, &mut norms, &mut pos_map, &config_tight, None,
        );
        assert!(verts.len() > verts_before,
            "tight deflection should insert Steiner points on sphere, verts {} -> {}",
            verts_before, verts.len());
        assert!(tris_tight.len() > tris_loose.len(),
            "Steiner points should produce more triangles");
    }

    #[test]
    fn cdt_holed_rect_no_triangles_inside_hole() {
        let outer = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 0, uv: (0.0, 0.0) },
                UvVertex { global_idx: 1, uv: (10.0, 0.0) },
                UvVertex { global_idx: 2, uv: (10.0, 10.0) },
                UvVertex { global_idx: 3, uv: (0.0, 10.0) },
            ],
        };
        let inner = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 4, uv: (3.0, 3.0) },
                UvVertex { global_idx: 5, uv: (7.0, 3.0) },
                UvVertex { global_idx: 6, uv: (7.0, 7.0) },
                UvVertex { global_idx: 7, uv: (3.0, 7.0) },
            ],
        };
        let loops = FaceUvLoops {
            outer,
            inners: vec![inner],
            uv_source: UvSource::Pcurve,
        };

        let tris = triangulate_uv_cdt(&loops).expect("CDT should succeed");
        assert!(tris.len() >= 3);

        let gi_to_uv: std::collections::HashMap<usize, (f32, f32)> = loops
            .outer
            .boundary
            .iter()
            .chain(loops.inners[0].boundary.iter())
            .map(|v| (v.global_idx, v.uv))
            .collect();

        let outer_uv: Vec<(f32, f32)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
        let inner_uv: Vec<Vec<(f32, f32)>> = loops
            .inners
            .iter()
            .map(|l| l.boundary.iter().map(|v| v.uv).collect())
            .collect();

        for chunk in tris.chunks(3) {
            let c = (
                (gi_to_uv[&chunk[0]].0 + gi_to_uv[&chunk[1]].0 + gi_to_uv[&chunk[2]].0) / 3.0,
                (gi_to_uv[&chunk[0]].1 + gi_to_uv[&chunk[1]].1 + gi_to_uv[&chunk[2]].1) / 3.0,
            );
            let in_hole = c.0 > 3.5 && c.0 < 6.5 && c.1 > 3.5 && c.1 < 6.5;
            assert!(!in_hole, "triangle centroid {:?} should not lie in hole", c);
            assert!(point_in_trim(c.0, c.1, &outer_uv, &inner_uv));
        }
    }

    #[test]
    fn distinct_3d_points_with_same_uv_get_distinct_handles() {
        let outer = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 0, uv: (1.0, 1.0) },
                UvVertex { global_idx: 1, uv: (1.0, 1.0) },
                UvVertex { global_idx: 2, uv: (2.0, 1.0) },
                UvVertex { global_idx: 3, uv: (2.0, 2.0) },
            ],
        };
        let loops = FaceUvLoops { outer, inners: vec![], uv_source: UvSource::Synthetic };
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
            color: None,
            degenerated_edges: vec![],
        };
        let mut verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
        ];
        let mut norms = vec![Vec3::Z; 4];
        let mut pos_map = HashMap::new();
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(f32x3_quantized_bits([v.x, v.y, v.z]), i);
        }
        let config = FaceFillConfig {
            enable_interior: false,
            ..Default::default()
        };
        let (tris, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, &mut verts, &mut norms, &mut pos_map, &config, None,
        );
        assert!(!tris.is_empty());
        for chunk in tris.chunks(3) {
            assert_ne!(chunk[0], chunk[1]);
            assert_ne!(chunk[1], chunk[2]);
            assert_ne!(chunk[2], chunk[0]);
        }
    }

    #[test]
    fn test_steiner_edge_midpoint() {
        let surface = SurfaceGeom::Sphere {
            center: Vec3::ZERO,
            radius: 10.0,
        };
        let face = BRepFace {
            surface: surface.clone(),
            outer_wire: Default::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        };
        let outer = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 0, uv: (0.0, 1.5708) },
                UvVertex { global_idx: 1, uv: (1.5708, 1.5708) },
                UvVertex { global_idx: 2, uv: (1.5708, 2.5708) },
                UvVertex { global_idx: 3, uv: (0.0, 2.5708) },
            ],
        };
        let loops = FaceUvLoops {
            outer,
            inners: vec![],
            uv_source: UvSource::Pcurve,
        };
        let mut verts = vec![
            surface.d0_native(0.0, 1.5708),
            surface.d0_native(1.5708, 1.5708),
            surface.d0_native(1.5708, 2.5708),
            surface.d0_native(0.0, 2.5708),
        ];
        let mut norms = vec![Vec3::Z; 4];
        let mut pos_map = HashMap::new();
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(f32x3_quantized_bits([v.x, v.y, v.z]), i);
        }
        let config = FaceFillConfig {
            enable_interior: true,
            deflection_interior: 0.05,
            min_size: 0.01,
            max_adapt_iterations: 4,
            ..Default::default()
        };
        let before = verts.len();
        let (tris, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, &mut verts, &mut norms, &mut pos_map, &config, None,
        );
        assert!(verts.len() > before, "Steiner should add interior vertices");
        assert!(!tris.is_empty());
    }

    #[test]
    fn test_cdt_degenerated_sphere_constraint() {
        use crate::step::brep::registry::BRepRegistry;
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let pole = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 1.0), 1e-4);
        let equator_key = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let equator_pos = reg.vertices[equator_key].position;
        let fk = reg.faces.insert(BRepFace {
            surface: surface.clone(),
            outer_wire: Default::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let degen_pc = CurveGeom::Line {
            origin: Vec3::new(0.0, 1.5, 0.0),
            direction: Vec3::new(0.0, -0.5, 0.0),
        };
        let degen_curve = CurveGeom::Line {
            origin: Vec3::new(1.0, 0.0, 0.0),
            direction: Vec3::new(-1.0, 0.0, 1.0),
        };
        let dek = reg.add_seam_edge(pole, pole, degen_curve, 1e-4, fk, degen_pc);
        if let Some(face) = reg.faces.get_mut(fk) {
            face.degenerated_edges.push(dek);
        }
        let outer = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 0, uv: (0.0, 1.4) },
                UvVertex { global_idx: 1, uv: (1.0, 1.4) },
                UvVertex { global_idx: 2, uv: (1.0, 1.6) },
                UvVertex { global_idx: 3, uv: (0.0, 1.6) },
            ],
        };
        let loops = FaceUvLoops {
            outer,
            inners: vec![],
            uv_source: UvSource::Pcurve,
        };
        let face = reg.faces.get(fk).unwrap().clone();
        let mut verts = vec![
            surface.d0_native(0.0, 1.4),
            surface.d0_native(1.0, 1.4),
            surface.d0_native(1.0, 1.6),
            surface.d0_native(0.0, 1.6),
            equator_pos,
        ];
        let mut norms = vec![Vec3::Z; verts.len()];
        let mut pos_map = HashMap::new();
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(f32x3_quantized_bits([v.x, v.y, v.z]), i);
        }
        let (tris, _) = triangulate_uv_cdt_with_steiner(
            &loops,
            &face,
            &mut verts,
            &mut norms,
            &mut pos_map,
            &FaceFillConfig::default(),
            Some(&reg),
        );
        assert!(!tris.is_empty(), "CDT with degenerated constraint should produce tris");
    }

    #[test]
    fn test_cdt_degenerated_cone() {
        use crate::step::brep::registry::BRepRegistry;
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Cone {
            apex: Vec3::ZERO,
            axis: Vec3::Z,
            semi_angle: 0.463648f32,
            radius_at_apex: 0.0,
        };
        let apex = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let fk = reg.faces.insert(BRepFace {
            surface: surface.clone(),
            outer_wire: Default::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let degen_pc = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.5, 0.0),
            direction: Vec3::new(0.0, 0.5, 0.0),
        };
        let degen_curve = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::Z,
        };
        let dek = reg.add_seam_edge(apex, apex, degen_curve, 1e-4, fk, degen_pc);
        if let Some(face) = reg.faces.get_mut(fk) {
            face.degenerated_edges.push(dek);
        }
        let outer = UvLoop {
            boundary: vec![
                UvVertex { global_idx: 0, uv: (0.0, 0.2) },
                UvVertex { global_idx: 1, uv: (1.0, 0.2) },
                UvVertex { global_idx: 2, uv: (1.0, 0.4) },
                UvVertex { global_idx: 3, uv: (0.0, 0.4) },
            ],
        };
        let loops = FaceUvLoops {
            outer,
            inners: vec![],
            uv_source: UvSource::Pcurve,
        };
        let face = reg.faces.get(fk).unwrap().clone();
        let mut verts = vec![
            surface.d0_native(0.0, 0.2),
            surface.d0_native(1.0, 0.2),
            surface.d0_native(1.0, 0.4),
            surface.d0_native(0.0, 0.4),
        ];
        let mut norms = vec![Vec3::Z; verts.len()];
        let mut pos_map = HashMap::new();
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(f32x3_quantized_bits([v.x, v.y, v.z]), i);
        }
        let (tris, _) = triangulate_uv_cdt_with_steiner(
            &loops,
            &face,
            &mut verts,
            &mut norms,
            &mut pos_map,
            &FaceFillConfig::default(),
            Some(&reg),
        );
        assert!(!tris.is_empty(), "cone CDT with apex degenerated edge should produce tris");
    }

}
