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

/// Cap Steiner splits per iteration to avoid CDT blow-up on bad trim domains.
const MAX_SPLITS_PER_ITER: usize = 256;

fn uv_quant_key(uv: (f32, f32)) -> (u64, u64) {
    ((uv.0 * 1e6).round() as u64, (uv.1 * 1e6).round() as u64)
}

/// Estimate chord deviation at a UV point by sampling the surface.
fn chord_dev_at(u: f32, v: f32, surface: &SurfaceGeom) -> f32 {
    let p = surface.d0_native(u, v);
    let eps = 1e-5;
    let pu = surface.d0_native(u + eps, v);
    let pv = surface.d0_native(u, v + eps);
    // Approximate deviation as the distance from point to tangent plane at midpoint
    let du = pu - p;
    let dv = pv - p;
    let normal = du.cross(dv).normalize();
    let mid = surface.d0_native(u + eps * 0.5, v + eps * 0.5);
    (mid - p).dot(normal).abs()
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

    // Degenerated edges (from surface singularities like sphere poles / cone apex)
    // have zero-length UV PCurves (direction == Vec3::ZERO), making CDT constraint
    // insertion meaningless. Skipped for now — future Phase 3 work will carry non-zero
    // Insert degenerated edges as CDT constraints
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
        let min_sz = effective_min_size(config);
        for _iter in 0..config.max_adapt_iterations {
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

                // Skip triangles with zero 3D area (degenerated-edge vertices)
                let area_3d = (p1 - p0).cross(p2 - p0).length();
                if area_3d < 1e-12 {
                    continue;
                }

                let mut tri_split = false;
                for (a, b, uva, uvb) in [
                    (&p0, &p1, uv0, uv1),
                    (&p1, &p2, uv1, uv2),
                    (&p2, &p0, uv2, uv0),
                ] {
                    let edge_len = (*a - *b).length();
                    let mid_uv = ((uva.0 + uvb.0) * 0.5, (uva.1 + uvb.1) * 0.5);
                    let on_surf = face.surface.d0_native(mid_uv.0, mid_uv.1);
                    let dev = ((*a + *b) * 0.5 - on_surf).length();
                    max_chord = max_chord.max(dev);
                    if dev > config.deflection_interior && edge_len > min_sz {
                        tri_split = true;
                    }
                    let na = face.surface.normal_native(uva.0, uva.1);
                    let nb = face.surface.normal_native(uvb.0, uvb.1);
                    let angle = na.normalize().dot(nb.normalize()).max(-1.0).min(1.0).acos();
                    if angle > config.angular_deflection && edge_len > min_sz {
                        tri_split = true;
                    }
                }
                if tri_split {
                    // Compute edge-midpoint deflections; split at the worst-deviated edge
                    let devs = [
                        ((p0 + p1) * 0.5
                            - face
                                .surface
                                .d0_native((uv0.0 + uv1.0) * 0.5, (uv0.1 + uv1.1) * 0.5))
                            .length(),
                        ((p1 + p2) * 0.5
                            - face
                                .surface
                                .d0_native((uv1.0 + uv2.0) * 0.5, (uv1.1 + uv2.1) * 0.5))
                            .length(),
                        ((p2 + p0) * 0.5
                            - face
                                .surface
                                .d0_native((uv2.0 + uv0.0) * 0.5, (uv2.1 + uv0.1) * 0.5))
                            .length(),
                    ];
                    let worst = devs
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap()
                        .0;
                    let (cu, cv) = match worst {
                        0 => ((uv0.0 + uv1.0) * 0.5, (uv0.1 + uv1.1) * 0.5),
                        1 => ((uv1.0 + uv2.0) * 0.5, (uv1.1 + uv2.1) * 0.5),
                        _ => ((uv2.0 + uv0.0) * 0.5, (uv2.1 + uv0.1) * 0.5),
                    };
                    splits.push((cu as f64, cv as f64));
                }
            }
            if splits.is_empty() {
                break;
            }
            // Sort by descending chord error: refine worst triangles first
            splits.sort_unstable_by(|(u1, v1), (u2, v2)| {
                let d1 = chord_dev_at(*u1 as f32, *v1 as f32, &face.surface);
                let d2 = chord_dev_at(*u2 as f32, *v2 as f32, &face.surface);
                d2.partial_cmp(&d1).unwrap_or(std::cmp::Ordering::Equal)
            });
            if splits.len() > MAX_SPLITS_PER_ITER {
                splits.truncate(MAX_SPLITS_PER_ITER);
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
}
