//! Constrained Delaunay triangulation for trimmed faces (native Bowyer-Watson CDT).
//! OCC BRepMesh_Delaun stand-in: CDT-first + insert-time Steiner refinement.

use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;
use rc3d_core::utils::hash::f32x3_quantized_bits;

use super::delaunay2d::{insert_uv_native, uv_bbox_from_loops, NativeCdt};
use super::face_fill::{effective_min_size, FaceFillConfig};
use super::face_uv::{point_in_trim, FaceUvLoops, UvSource};
use crate::geom::SurfaceGeom;
use crate::store::BRepStore;
use crate::topo::{BRepFace, FaceKey};

const MAX_CDT_VERTICES: usize = 4096;

fn uv_quant_key(uv: (f32, f32)) -> (u64, u64) {
    ((uv.0 * 1e6).round() as u64, (uv.1 * 1e6).round() as u64)
}

fn uv_quant_key_relative(uv: (f32, f32), uv_span: (f32, f32)) -> (u64, u64) {
    let su = uv_span.0.max(1e-6);
    let sv = uv_span.1.max(1e-6);
    ((uv.0 / su * 1e6).round() as u64, (uv.1 / sv * 1e6).round() as u64)
}

fn quant_key(uv: (f32, f32), span: Option<(f32, f32)>) -> (u64, u64) {
    match span {
        Some(s) => uv_quant_key_relative(uv, s),
        None => uv_quant_key(uv),
    }
}

fn compute_uv_span(loops: &FaceUvLoops) -> (f32, f32) {
    let mut u_min = f32::MAX;
    let mut u_max = f32::MIN;
    let mut v_min = f32::MAX;
    let mut v_max = f32::MIN;
    for v in &loops.outer.boundary {
        u_min = u_min.min(v.uv.0);
        u_max = u_max.max(v.uv.0);
        v_min = v_min.min(v.uv.1);
        v_max = v_max.max(v.uv.1);
    }
    for inner in &loops.inners {
        for v in &inner.boundary {
            u_min = u_min.min(v.uv.0);
            u_max = u_max.max(v.uv.0);
            v_min = v_min.min(v.uv.1);
            v_max = v_max.max(v.uv.1);
        }
    }
    ((u_max - u_min).abs(), (v_max - v_min).abs())
}

/// Native Bowyer-Watson CDT path (OCC BRepMesh_Delaun-style).
/// Returns (flat list of global vertex indices per triangle, max_chord_error).
#[allow(clippy::too_many_arguments)]
pub fn triangulate_uv_cdt_with_steiner(
    loops: &FaceUvLoops,
    face: &BRepFace,
    face_key: Option<FaceKey>,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
    config: &FaceFillConfig,
    reg: Option<&BRepStore>,
) -> (Vec<usize>, f32) {
    if loops.outer.boundary.len() < 3 {
        return (Vec::new(), 0.0);
    }

    let outer_uv: Vec<(f32, f32)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
    let inner_uv: Vec<Vec<(f32, f32)>> = loops
        .inners
        .iter()
        .map(|l| l.boundary.iter().map(|v| v.uv).collect())
        .collect();

    let (u_min, v_min, u_max, v_max) = uv_bbox_from_loops(&outer_uv, &inner_uv);
    let mut cdt = NativeCdt::from_uv_bbox(u_min, v_min, u_max, v_max);
    let mut uv_to_handle: HashMap<(u64, u64), super::delaunay2d::CdtVertHandle> = HashMap::new();
    let uv_span = compute_uv_span(loops);
    let span_opt = Some(uv_span);

    let mut outer_handles = Vec::new();
    for v in &loops.outer.boundary {
        let Some(h) = insert_uv_native(
            &mut cdt,
            v.uv,
            v.global_idx,
            &mut uv_to_handle,
            span_opt,
            quant_key,
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
            let Some(h) = insert_uv_native(
                &mut cdt,
                v.uv,
                v.global_idx,
                &mut uv_to_handle,
                span_opt,
                quant_key,
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
            let ok = cdt.try_add_constraint(a, b);
            if !ok && cdt.num_constraints() == before && !cdt.exists_constraint(a, b) {
                return (Vec::new(), 0.0);
            }
        }
    }

    if let Some(reg) = reg {
        for &dek in &face.degenerated_edges {
            let edge = match reg.edges.get(dek) {
                Some(e) => e,
                None => continue,
            };
            let pc = match face_key.and_then(|fk| edge.pcurves.get(&fk)) {
                Some(p) => p,
                None => match edge.pcurves.values().next() {
                    Some(p) => p,
                    None => continue,
                },
            };
            let uv0 = pc.d0(0.0);
            let uv1 = pc.d0(1.0);
            if (uv0.x - uv1.x).abs() < 1e-10 && (uv0.y - uv1.y).abs() < 1e-10 {
                continue;
            }
            let pt0 = face.surface.d0_native(uv0.x, uv0.y);
            let pt1 = face.surface.d0_native(uv1.x, uv1.y);
            let hash0 = f32x3_quantized_bits([pt0.x, pt0.y, pt0.z]);
            let hash1 = f32x3_quantized_bits([pt1.x, pt1.y, pt1.z]);
            let gi0 = *pos_to_idx.entry(hash0).or_insert_with(|| {
                let i = global_vertices.len();
                global_vertices.push(pt0);
                let mut n = face.surface.normal_native(uv0.x, uv0.y);
                if !face.same_sense {
                    n = -n;
                }
                global_normals.push(n);
                i
            });
            let gi1 = *pos_to_idx.entry(hash1).or_insert_with(|| {
                let i = global_vertices.len();
                global_vertices.push(pt1);
                let mut n = face.surface.normal_native(uv1.x, uv1.y);
                if !face.same_sense {
                    n = -n;
                }
                global_normals.push(n);
                i
            });
            let h0 = match insert_uv_native(
                &mut cdt,
                (uv0.x, uv0.y),
                gi0,
                &mut uv_to_handle,
                span_opt,
                quant_key,
            ) {
                Some(h) => h,
                None => continue,
            };
            let h1 = match insert_uv_native(
                &mut cdt,
                (uv1.x, uv1.y),
                gi1,
                &mut uv_to_handle,
                span_opt,
                quant_key,
            ) {
                Some(h) => h,
                None => continue,
            };
            let _ = cdt.try_add_constraint(h0, h1);
        }
    }

    let mut max_chord = 0.0f32;
    if config.enable_interior && config.deflection_interior > 0.0 {
        // Reuse UV bbox already computed above (outer_uv/inner_uv).
        let mut u_min = outer_uv.iter().map(|&(u, _)| u).fold(f32::INFINITY, f32::min);
        let mut u_max = outer_uv.iter().map(|&(u, _)| u).fold(f32::MIN, f32::max);
        let mut v_min = outer_uv.iter().map(|&(_, v)| v).fold(f32::INFINITY, f32::min);
        let mut v_max = outer_uv.iter().map(|&(_, v)| v).fold(f32::MIN, f32::max);
        for inner in &inner_uv {
            for &(u, v) in inner {
                u_min = u_min.min(u); u_max = u_max.max(u);
                v_min = v_min.min(v); v_max = v_max.max(v);
            }
        }
        let range = if matches!(face.surface, SurfaceGeom::Revolution { .. }) {
            loops
                .revolution_native_uv_bounds_from_boundary(face, global_vertices)
                .unwrap_or((u_min, u_max, v_min, v_max))
        } else {
            (u_min, u_max, v_min, v_max)
        };
        let (u_min, u_max, v_min, v_max) = range;
        let (u_divs, v_divs) = face.surface.parameter_division(
            range,
            config.deflection_interior,
            config.parameter_division_max_depth,
        );

        // Pre-build HashSet of quantized boundary UVs for O(1) on_boundary check.
        let u_span = (u_max - u_min).abs();
        let v_span = (v_max - v_min).abs();
        let u_eps = (u_span * 1e-7).max(1e-6);
        let v_eps = (v_span * 1e-7).max(1e-6);
        let boundary_uv_set: HashSet<(u64, u64)> = loops
            .outer
            .boundary
            .iter()
            .map(|vb| ((vb.uv.0 / u_eps).round() as u64, (vb.uv.1 / v_eps).round() as u64))
            .chain(loops.inners.iter().flat_map(|inner| {
                inner.boundary.iter().map(|vb| {
                    ((vb.uv.0 / u_eps).round() as u64, (vb.uv.1 / v_eps).round() as u64)
                })
            }))
            .collect();

        let mut inserted = 0usize;
        for u in &u_divs {
            for v in &v_divs {
                if cdt.vertex_count() >= MAX_CDT_VERTICES {
                    break;
                }
                let qkey = ((u / u_eps).round() as u64, (v / v_eps).round() as u64);
                if boundary_uv_set.contains(&qkey) {
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
                    if insert_uv_native(
                        &mut cdt,
                        (*u, *v),
                        gi,
                        &mut uv_to_handle,
                        span_opt,
                        quant_key,
                    )
                    .is_some()
                    {
                        inserted += 1;
                    }
                }
            }
        }
        log::debug!(
            "[BRep mesh] native CDT grid inserted {} interior points (uv_divs={}x{})",
            inserted,
            u_divs.len(),
            v_divs.len()
        );

        let min_sz = effective_min_size(config);
        if !config.skip_interior_edge_split {
            for _iter in 0..1 {
                if cdt.vertex_count() >= MAX_CDT_VERTICES {
                    break;
                }
                let mut splits: Vec<(f64, f64)> = Vec::new();
                for (_gids, uvs) in cdt.inner_faces_detail() {
                    let uv0 = uvs[0];
                    let uv1 = uvs[1];
                    let uv2 = uvs[2];
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
                    insert_uv_native(
                        &mut cdt,
                        (u as f32, v as f32),
                        gi,
                        &mut uv_to_handle,
                        span_opt,
                        quant_key,
                    );
                }
            }
        }
    }

    let use_trim =
        loops.uv_source == UvSource::Pcurve || !loops.inners.is_empty();
    cdt.finalize();
    let mut tris = if use_trim {
        cdt.extract_triangles(|cu, cv| point_in_trim(cu, cv, &outer_uv, &inner_uv))
    } else {
        cdt.extract_triangles(|_, _| true)
    };
    if tris.is_empty() && use_trim {
        tris = cdt.extract_triangles(|_, _| true);
    }

    (tris, max_chord)
}

/// Backward-compat: UV-only CDT without surface/refinement (no Steiner points).
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
        loops, &face, None, &mut verts, &mut norms, &mut pmap, &config, None,
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
    use super::super::face_uv::{point_in_trim, UvLoop, UvSource, UvVertex};
    use crate::geom::CurveGeom;
    use rc3d_core::utils::hash::f32x3_quantized_bits;

    #[test]
    fn steiner_split_on_curved_surface() {
        let surface = SurfaceGeom::Sphere {
            center: Vec3::ZERO,
            radius: 10.0,
        };
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
                UvVertex {
                    global_idx: 0,
                    uv: (0.0, 1.5708),
                },
                UvVertex {
                    global_idx: 1,
                    uv: (1.5708, 1.5708),
                },
                UvVertex {
                    global_idx: 2,
                    uv: (1.5708, 2.5708),
                },
                UvVertex {
                    global_idx: 3,
                    uv: (0.0, 2.5708),
                },
            ],
        };
        let loops = FaceUvLoops {
            outer,
            inners: vec![],
            uv_source: UvSource::Pcurve,
        };

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
            ..Default::default()
        };

        let (tris_loose, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, None, &mut verts, &mut norms, &mut pos_map, &config_loose, None,
        );
        let verts_before = verts.len();

        let config_tight = FaceFillConfig {
            deflection_interior: 0.01,
            max_adapt_iterations: 4,
            ..config_loose.clone()
        };
        let (tris_tight, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, None, &mut verts, &mut norms, &mut pos_map, &config_tight, None,
        );
        assert!(
            verts.len() > verts_before,
            "tight deflection should insert Steiner points on sphere, verts {} -> {}",
            verts_before,
            verts.len()
        );
        assert!(
            tris_tight.len() > tris_loose.len(),
            "Steiner points should produce more triangles"
        );
    }

    #[test]
    fn cdt_holed_rect_no_triangles_inside_hole() {
        let outer = UvLoop {
            boundary: vec![
                UvVertex {
                    global_idx: 0,
                    uv: (0.0, 0.0),
                },
                UvVertex {
                    global_idx: 1,
                    uv: (10.0, 0.0),
                },
                UvVertex {
                    global_idx: 2,
                    uv: (10.0, 10.0),
                },
                UvVertex {
                    global_idx: 3,
                    uv: (0.0, 10.0),
                },
            ],
        };
        let inner = UvLoop {
            boundary: vec![
                UvVertex {
                    global_idx: 4,
                    uv: (3.0, 3.0),
                },
                UvVertex {
                    global_idx: 5,
                    uv: (7.0, 3.0),
                },
                UvVertex {
                    global_idx: 6,
                    uv: (7.0, 7.0),
                },
                UvVertex {
                    global_idx: 7,
                    uv: (3.0, 7.0),
                },
            ],
        };
        let loops = FaceUvLoops {
            outer,
            inners: vec![inner],
            uv_source: UvSource::Pcurve,
        };

        let tris = triangulate_uv_cdt(&loops).expect("CDT should succeed");
        assert!(tris.len() >= 3);

        let gi_to_uv: HashMap<usize, (f32, f32)> = loops
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
                UvVertex {
                    global_idx: 0,
                    uv: (1.0, 1.0),
                },
                UvVertex {
                    global_idx: 1,
                    uv: (1.0, 1.0),
                },
                UvVertex {
                    global_idx: 2,
                    uv: (2.0, 1.0),
                },
                UvVertex {
                    global_idx: 3,
                    uv: (2.0, 2.0),
                },
            ],
        };
        let loops = FaceUvLoops {
            outer,
            inners: vec![],
            uv_source: UvSource::Synthetic,
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
            &loops, &face, None, &mut verts, &mut norms, &mut pos_map, &config, None,
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
                UvVertex {
                    global_idx: 0,
                    uv: (0.0, 1.5708),
                },
                UvVertex {
                    global_idx: 1,
                    uv: (1.5708, 1.5708),
                },
                UvVertex {
                    global_idx: 2,
                    uv: (1.5708, 2.5708),
                },
                UvVertex {
                    global_idx: 3,
                    uv: (0.0, 2.5708),
                },
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
            &loops, &face, None, &mut verts, &mut norms, &mut pos_map, &config, None,
        );
        assert!(verts.len() > before, "Steiner should add interior vertices");
        assert!(!tris.is_empty());
    }

    #[test]
    fn test_cdt_degenerated_sphere_constraint() {
        use crate::store::BRepStore;
        let mut reg = BRepStore::new();
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
                UvVertex {
                    global_idx: 0,
                    uv: (0.0, 1.4),
                },
                UvVertex {
                    global_idx: 1,
                    uv: (1.0, 1.4),
                },
                UvVertex {
                    global_idx: 2,
                    uv: (1.0, 1.6),
                },
                UvVertex {
                    global_idx: 3,
                    uv: (0.0, 1.6),
                },
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
            Some(fk),
            &mut verts,
            &mut norms,
            &mut pos_map,
            &FaceFillConfig::default(),
            Some(&reg),
        );
        assert!(
            !tris.is_empty(),
            "CDT with degenerated constraint should produce tris"
        );
    }

    #[test]
    fn test_cdt_degenerated_cone() {
        use crate::store::BRepStore;
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::cone(Vec3::ZERO, Vec3::Z, 0.463648f32, 0.0);
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
                UvVertex {
                    global_idx: 0,
                    uv: (0.0, 0.2),
                },
                UvVertex {
                    global_idx: 1,
                    uv: (1.0, 0.2),
                },
                UvVertex {
                    global_idx: 2,
                    uv: (1.0, 0.4),
                },
                UvVertex {
                    global_idx: 3,
                    uv: (0.0, 0.4),
                },
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
            Some(fk),
            &mut verts,
            &mut norms,
            &mut pos_map,
            &FaceFillConfig::default(),
            Some(&reg),
        );
        assert!(
            !tris.is_empty(),
            "cone CDT with apex degenerated edge should produce tris"
        );
    }
}
