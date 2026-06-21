//! Constrained Delaunay triangulation for trimmed faces (native Bowyer-Watson CDT).
//! OCC BRepMesh_Delaun stand-in: CDT-first + insert-time Steiner refinement.

use std::collections::{HashMap, HashSet};

use rc3d_core::math::{Real, PVec3};

use super::boundary::{
    register_boundary_point_with_normal_indexed_shared, BoundaryPosIndex, SharedBoundaryPool,
};
use super::delaunay2d::{insert_uv_native, uv_bbox_from_loops, NativeCdt};
use super::face_fill::{effective_min_size, FaceFillConfig};
use super::face_uv::{point_in_trim, FaceUvLoops, UvSource};
use crate::geom::SurfaceGeom;
use crate::store::BRepStore;
use crate::topo::{BRepFace, FaceKey};

/// Result of constrained Delaunay edge enforcement for one face.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CdtConstraintReport {
    pub failed: usize,
    pub total: usize,
}

impl CdtConstraintReport {
    pub fn has_failures(&self) -> bool {
        self.failed > 0
    }
}

/// Safe f64→u64 conversion — clamps to [0, u64::MAX] to avoid overflow
/// panics in debug mode and silent wraps in release.
fn f64_to_u64(v: f64) -> u64 {
    if v.is_nan() || v.is_infinite() { return 0; }
    if v < 0.0 { 0 } else if v > u64::MAX as f64 { u64::MAX } else { v as u64 }
}

fn uv_quant_key(uv: (Real, Real)) -> (u64, u64) {
    (f64_to_u64((uv.0 * 1e6).round()), f64_to_u64((uv.1 * 1e6).round()))
}

fn uv_quant_key_relative(uv: (Real, Real), uv_span: (Real, Real)) -> (u64, u64) {
    let su = uv_span.0.max(1e-6);
    let sv = uv_span.1.max(1e-6);
    (f64_to_u64((uv.0 / su * 1e6).round()), f64_to_u64((uv.1 / sv * 1e6).round()))
}

fn quant_key(uv: (Real, Real), span: Option<(Real, Real)>) -> (u64, u64) {
    match span {
        Some(s) => uv_quant_key_relative(uv, s),
        None => uv_quant_key(uv),
    }
}

fn compute_uv_span(loops: &FaceUvLoops) -> (Real, Real) {
    let mut u_min = f64::MAX;
    let mut u_max = f64::MIN;
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;
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
/// Returns (flat triangle global indices, max chord error, constraint report).
#[allow(clippy::too_many_arguments)]
pub fn triangulate_uv_cdt_with_steiner(
    loops: &FaceUvLoops,
    face: &BRepFace,
    face_key: Option<FaceKey>,
    global_vertices: &mut Vec<PVec3>,
    global_normals: &mut Vec<PVec3>,
    pos_to_idx: &mut BoundaryPosIndex,
    config: &FaceFillConfig,
    reg: Option<&BRepStore>,
    shared_boundary: Option<&SharedBoundaryPool>,
) -> (Vec<usize>, Real, CdtConstraintReport) {
    if loops.outer.boundary.len() < 3 {
        return (Vec::new(), 0.0, CdtConstraintReport::default());
    }

    let outer_uv: Vec<(Real, Real)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
    let inner_uv: Vec<Vec<(Real, Real)>> = loops
        .inners
        .iter()
        .map(|l| l.boundary.iter().map(|v| v.uv).collect())
        .collect();

    let (mut u_min, mut v_min, mut u_max, mut v_max) = uv_bbox_from_loops(&outer_uv, &inner_uv);

    // Expand bbox to cover degenerate edge UVs (e.g., sphere pole edges may extend
    // beyond the outer loop range). The super-triangle must enclose all insertable points.
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
            u_min = u_min.min(uv0.0).min(uv1.0);
            u_max = u_max.max(uv0.0).max(uv1.0);
            v_min = v_min.min(uv0.1).min(uv1.1);
            v_max = v_max.max(uv0.1).max(uv1.1);
        }
    }

    let mut cdt = NativeCdt::from_uv_bbox_with_backend(
        u_min, v_min, u_max, v_max, config.delaunay_backend,
    );
    let mut uv_to_handle: HashMap<(u64, u64), super::delaunay2d::CdtVertHandle> = HashMap::new();
    let uv_span = compute_uv_span(loops);
    let span_opt = Some(uv_span);

    let native_uv = |u: Real, v: Real| -> (Real, Real) {
        if let (Some(r), Some(fk)) = (reg, face_key) {
            r.face_native_uv(fk, u, v)
        } else {
            (u, v)
        }
    };

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
            return (Vec::new(), 0.0, CdtConstraintReport::default());
        };
        outer_handles.push(h);
    }
    if outer_handles.len() < 3 {
        return (Vec::new(), 0.0, CdtConstraintReport::default());
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
                return (Vec::new(), 0.0, CdtConstraintReport::default());
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
                // Log constraint failure but continue — don't kill the entire face.
                // BRepMesh_Delaun continues with failures; ModelHealer fixes gaps later.
                log::debug!(
                    "[CDT] constraint failure: inner loop edge {}-{} could not be added, continuing",
                    a, b
                );
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
            if (uv0.0 - uv1.0).abs() < 1e-10 && (uv0.1 - uv1.1).abs() < 1e-10 {
                continue;
            }
            let (nu0, nv0) = native_uv(uv0.0, uv0.1);
            let pt0 = face.surface.d0_native(nu0, nv0);
            let (nu1, nv1) = native_uv(uv1.0, uv1.1);
            let pt1 = face.surface.d0_native(nu1, nv1);
            let gi0 = register_boundary_point_with_normal_indexed_shared(
                pt0,
                global_vertices,
                global_normals,
                pos_to_idx,
                shared_boundary,
                || {
                    let mut n = face.surface.normal_native(nu0, nv0);
                    if !face.same_sense {
                        n = -n;
                    }
                    n
                },
            );
            let gi1 = register_boundary_point_with_normal_indexed_shared(
                pt1,
                global_vertices,
                global_normals,
                pos_to_idx,
                shared_boundary,
                || {
                    let mut n = face.surface.normal_native(nu1, nv1);
                    if !face.same_sense {
                        n = -n;
                    }
                    n
                },
            );
            let h0 = match insert_uv_native(
                &mut cdt,
                (uv0.0, uv0.1),
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
                (uv1.0, uv1.1),
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

    let mut max_chord = 0.0_f64;
    if config.enable_interior && config.deflection_interior > 0.0 {
        // Reuse UV bbox already computed above (outer_uv/inner_uv).
        let mut u_min = outer_uv.iter().map(|&(u, _)| u).fold(f64::INFINITY, Real::min);
        let mut u_max = outer_uv.iter().map(|&(u, _)| u).fold(f64::MIN, Real::max);
        let mut v_min = outer_uv.iter().map(|&(_, v)| v).fold(f64::INFINITY, Real::min);
        let mut v_max = outer_uv.iter().map(|&(_, v)| v).fold(f64::MIN, Real::max);
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
        let (u_divs, v_divs) = super::curvature_driven_divisions(
            &face.surface,
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
            .map(|vb| (f64_to_u64((vb.uv.0 / u_eps).round()), f64_to_u64((vb.uv.1 / v_eps).round())))
            .chain(loops.inners.iter().flat_map(|inner| {
                inner.boundary.iter().map(|vb| {
                    (f64_to_u64((vb.uv.0 / u_eps).round()), f64_to_u64((vb.uv.1 / v_eps).round()))
                })
            }))
            .collect();

        let mut inserted = 0usize;
        for u in &u_divs {
            for v in &v_divs {
                if cdt.vertex_count() >= config.max_cdt_vertices.max(1) {
                    break;
                }
                let qkey = (f64_to_u64((u / u_eps).round()), f64_to_u64((v / v_eps).round()));
                if boundary_uv_set.contains(&qkey) {
                    continue;
                }
                if point_in_trim(*u, *v, &outer_uv, &inner_uv) {
                    let (nu, nv) = native_uv(*u, *v);
                    let pt_3d = face.surface.d0_native(nu, nv);
                    let gi = register_boundary_point_with_normal_indexed_shared(
                        pt_3d,
                        global_vertices,
                        global_normals,
                        pos_to_idx,
                        shared_boundary,
                        || {
                            let mut n = face.surface.normal_native(nu, nv);
                            if !face.same_sense {
                                n = -n;
                            }
                            n
                        },
                    );
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
            let max_iter = config.max_adapt_iterations.max(1).min(8);
            for _iter in 0..max_iter {
                if cdt.vertex_count() >= config.max_cdt_vertices.max(1) {
                    break;
                }
                // Re-triangulate for DelaBella backend (no-op for BowyerWatson).
                // This ensures inner_faces_detail() returns valid data for chord checking.
                cdt.retriangulate();
                // OCC BRepMesh_NodeInsertionMeshAlgo: per-triangle split strategy.
                // - 1 failing edge → insert at its UV midpoint
                // - 2+ failing edges → insert at triangle UV centroid
                // - Sort by descending chord error → worst triangles refined first.
                let mut splits: Vec<((f64, f64), Real)> = Vec::new(); // ((u,v), chord_error)
                for (_gids, uvs) in cdt.inner_faces_detail() {
                    let uv0 = uvs[0];
                    let uv1 = uvs[1];
                    let uv2 = uvs[2];
                    let (nu0, nv0) = native_uv(uv0.0, uv0.1);
                    let p0 = face.surface.d0_native(nu0, nv0);
                    let (nu1, nv1) = native_uv(uv1.0, uv1.1);
                    let p1 = face.surface.d0_native(nu1, nv1);
                    let (nu2, nv2) = native_uv(uv2.0, uv2.1);
                    let p2 = face.surface.d0_native(nu2, nv2);
                    let area_3d = (p1 - p0).cross(p2 - p0).length();
                    if area_3d < 1e-12 {
                        continue;
                    }

                    // Collect which edges fail and their chord errors.
                    let mut failed_edges: Vec<(usize, Real, (Real, Real))> = Vec::new(); // (edge_idx, chord_err, uv_mid)
                    let edges: [(&PVec3, &PVec3, (Real, Real), (Real, Real)); 3] = [
                        (&p0, &p1, uv0, uv1),
                        (&p1, &p2, uv1, uv2),
                        (&p2, &p0, uv2, uv0),
                    ];
                    for (ei, &(a, b, uva, uvb)) in edges.iter().enumerate() {
                        let edge_len = (*a - *b).length();
                        if edge_len <= min_sz {
                            continue;
                        }
                        let mid_3d = (*a + *b) * 0.5;
                        let uv_mid = ((uva.0 + uvb.0) * 0.5, (uva.1 + uvb.1) * 0.5);
                        let (nu_mid, nv_mid) = native_uv(uv_mid.0, uv_mid.1);
                        let surf_mid = face.surface.d0_native(nu_mid, nv_mid);
                        let dev = (mid_3d - surf_mid).length();
                        max_chord = max_chord.max(dev);
                        let mut needs = dev > config.deflection_interior;
                        if !needs {
                            let (nu_a, nv_a) = native_uv(uva.0, uva.1);
                            let na = face.surface.normal_native(nu_a, nv_a);
                            let (nu_b, nv_b) = native_uv(uvb.0, uvb.1);
                            let nb = face.surface.normal_native(nu_b, nv_b);
                            let angle = na.normalize().dot(nb.normalize()).max(-1.0).min(1.0).acos();
                            needs = angle > config.angular_deflection;
                        }
                        if needs {
                            failed_edges.push((ei, dev, uv_mid));
                        }
                    }

                    if failed_edges.is_empty() {
                        continue;
                    }

                    // OCC strategy: single-edge fail → midpoint; multi-edge → centroid.
                    let split_uv = if failed_edges.len() == 1 {
                        let (_, _, uv_mid) = failed_edges[0];
                        uv_mid
                    } else {
                        // Centroid of the triangle in UV space.
                        let cu = (uv0.0 + uv1.0 + uv2.0) / 3.0;
                        let cv = (uv0.1 + uv1.1 + uv2.1) / 3.0;
                        (cu, cv)
                    };
                    if point_in_trim(split_uv.0, split_uv.1, &outer_uv, &inner_uv) {
                        let max_dev = failed_edges.iter().map(|&(_, d, _)| d).fold(0.0_f64, Real::max);
                        splits.push(((split_uv.0 as f64, split_uv.1 as f64), max_dev));
                    }
                }
                if splits.is_empty() {
                    break;
                }
                // Sort by descending chord error — worst triangles refined first.
                splits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut dedup = HashSet::new();
                for ((u, v), _dev) in splits {
                    let key = ((u * 1e4) as u64, (v * 1e4) as u64);
                    if dedup.contains(&key) {
                        continue;
                    }
                    dedup.insert(key);
                    let (nu, nv) = native_uv(u as Real, v as Real);
                    let pt_3d = face.surface.d0_native(nu, nv);
                    let gi = register_boundary_point_with_normal_indexed_shared(
                        pt_3d,
                        global_vertices,
                        global_normals,
                        pos_to_idx,
                        shared_boundary,
                        || {
                            let mut n = face.surface.normal_native(nu, nv);
                            if !face.same_sense {
                                n = -n;
                            }
                            n
                        },
                    );
                    insert_uv_native(
                        &mut cdt,
                        (u as Real, v as Real),
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
    if tris.is_empty() && use_trim && loops.inners.is_empty() {
        tris = cdt.extract_triangles(|_, _| true);
    }

    let constraint_report = CdtConstraintReport {
        failed: cdt.constraint_failure_count(),
        total: cdt.num_constraints(),
    };
    if constraint_report.has_failures() {
        log::debug!(
            "[BRep mesh] CDT constraint enforcement failed {}/{} edges",
            constraint_report.failed,
            constraint_report.total
        );
    }

    (tris, max_chord, constraint_report)
}

/// Backward-compat: UV-only CDT without surface/refinement (no Steiner points).
pub fn triangulate_uv_cdt(loops: &FaceUvLoops) -> Option<Vec<usize>> {
    if loops.outer.boundary.len() < 3 {
        return None;
    }
    let face = BRepFace {
        surface: SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
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
    let mut pmap = BoundaryPosIndex::with_cell_size(crate::mesh::boundary::BOUNDARY_DEDUP_TOLERANCE);
    let config = FaceFillConfig {
        enable_interior: false,
        ..Default::default()
    };
    let (tris, _, _) = triangulate_uv_cdt_with_steiner(
        loops, &face, None, &mut verts, &mut norms, &mut pmap, &config, None, None,
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
    use crate::geom::curve2d::Curve2d;
    use super::super::face_uv::{point_in_trim, UvLoop, UvSource, UvVertex};
    use crate::geom::CurveGeom;

    #[test]
    fn steiner_split_on_curved_surface() {
        let surface = SurfaceGeom::Sphere {
            center: PVec3::ZERO,
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
        let mut norms = vec![PVec3::Z; 4];
        let mut pos_map =
            BoundaryPosIndex::with_cell_size(crate::mesh::boundary::BOUNDARY_DEDUP_TOLERANCE);
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(i, [v.x, v.y, v.z]);
        }

        let config_loose = FaceFillConfig {
            enable_interior: true,
            deflection_interior: 0.5,
            min_size: 0.1,
            min_size_relative: 0.0,
            shell_min_size: 0.0,
            max_adapt_iterations: 4,
            angular_deflection: std::f64::consts::PI,
            parameter_division_max_depth: 0, // disable grid; test Steiner splitting only
            skip_interior_edge_split: false, // enable Steiner edge splits
            ..Default::default()
        };

        let (tris_loose, _, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, None, &mut verts, &mut norms, &mut pos_map, &config_loose, None, None,
        );
        let verts_before = verts.len();

        let config_tight = FaceFillConfig {
            deflection_interior: 0.01,
            max_adapt_iterations: 4,
            parameter_division_max_depth: 0, // disable grid; test Steiner splitting only
            skip_interior_edge_split: false, // enable Steiner edge splits
            ..config_loose.clone()
        };
        let (tris_tight, _, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, None, &mut verts, &mut norms, &mut pos_map, &config_tight, None, None,
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

        let gi_to_uv: HashMap<usize, (Real, Real)> = loops
            .outer
            .boundary
            .iter()
            .chain(loops.inners[0].boundary.iter())
            .map(|v| (v.global_idx, v.uv))
            .collect();

        let outer_uv: Vec<(Real, Real)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
        let inner_uv: Vec<Vec<(Real, Real)>> = loops
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
                origin: PVec3::ZERO,
                normal: PVec3::Z,
                u_dir: PVec3::X,
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
            PVec3::new(0.0, 0.0, 0.0),
            PVec3::new(1.0, 0.0, 0.0),
            PVec3::new(2.0, 0.0, 0.0),
            PVec3::new(2.0, 1.0, 0.0),
        ];
        let mut norms = vec![PVec3::Z; 4];
        let mut pos_map =
            BoundaryPosIndex::with_cell_size(crate::mesh::boundary::BOUNDARY_DEDUP_TOLERANCE);
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(i, [v.x, v.y, v.z]);
        }
        let config = FaceFillConfig {
            enable_interior: false,
            ..Default::default()
        };
        let (tris, _, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, None, &mut verts, &mut norms, &mut pos_map, &config, None, None,
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
            center: PVec3::ZERO,
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
        let mut norms = vec![PVec3::Z; 4];
        let mut pos_map =
            BoundaryPosIndex::with_cell_size(crate::mesh::boundary::BOUNDARY_DEDUP_TOLERANCE);
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(i, [v.x, v.y, v.z]);
        }
        let config = FaceFillConfig {
            enable_interior: true,
            deflection_interior: 0.05,
            min_size: 0.01,
            max_adapt_iterations: 4,
            skip_interior_edge_split: false, // enable Steiner edge splits
            ..Default::default()
        };
        let before = verts.len();
        let (tris, _, _) = triangulate_uv_cdt_with_steiner(
            &loops, &face, None, &mut verts, &mut norms, &mut pos_map, &config, None, None,
        );
        assert!(verts.len() > before, "Steiner should add interior vertices");
        assert!(!tris.is_empty());
    }

    #[test]
    fn test_cdt_degenerated_sphere_constraint() {
        use crate::store::BRepStore;
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Sphere {
            center: PVec3::ZERO,
            radius: 1.0,
        };
        let pole = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 1.0), 1e-4);
        let equator_key = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-4);
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
        let degen_pc = Curve2d::Line {
            origin: (0.0, 1.5),
            direction: (0.0, -0.5),
        };
        let degen_curve = CurveGeom::Line {
            origin: PVec3::new(1.0, 0.0, 0.0),
            direction: PVec3::new(-1.0, 0.0, 1.0),
        };
        let dek = reg.add_seam_edge(pole, pole, degen_curve, 1e-4, fk, degen_pc, true);
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
        let mut norms = vec![PVec3::Z; verts.len()];
        let mut pos_map =
            BoundaryPosIndex::with_cell_size(crate::mesh::boundary::BOUNDARY_DEDUP_TOLERANCE);
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(i, [v.x, v.y, v.z]);
        }
        let (tris, _, _) = triangulate_uv_cdt_with_steiner(
            &loops,
            &face,
            Some(fk),
            &mut verts,
            &mut norms,
            &mut pos_map,
            &FaceFillConfig::default(),
            Some(&reg),
            None,
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
        let surface = SurfaceGeom::cone(PVec3::ZERO, PVec3::Z, 0.463648_f64, 0.0);
        let apex = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
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
        let degen_pc = Curve2d::Line {
            origin: (0.0, 0.5),
            direction: (0.0, 0.5),
        };
        let degen_curve = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::Z,
        };
        let dek = reg.add_seam_edge(apex, apex, degen_curve, 1e-4, fk, degen_pc, true);
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
        let mut norms = vec![PVec3::Z; verts.len()];
        let mut pos_map =
            BoundaryPosIndex::with_cell_size(crate::mesh::boundary::BOUNDARY_DEDUP_TOLERANCE);
        for (i, v) in verts.iter().enumerate() {
            pos_map.insert(i, [v.x, v.y, v.z]);
        }
        let (tris, _, _) = triangulate_uv_cdt_with_steiner(
            &loops,
            &face,
            Some(fk),
            &mut verts,
            &mut norms,
            &mut pos_map,
            &FaceFillConfig::default(),
            Some(&reg),
            None,
        );
        assert!(
            !tris.is_empty(),
            "cone CDT with apex degenerated edge should produce tris"
        );
    }
}
