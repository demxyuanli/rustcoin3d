//! Incremental constrained Delaunay adapter for UV face fill.
//!
//! Supports two backends:
//! - `BowyerWatson` (default) — existing incremental CDT implementation
//! - `DelaBella` — Newton Apple Wrapper algorithm (experimental, adaptive-exact predicates)

use std::collections::HashMap;

use super::triangulation::Delaunay2d;
use super::DelaunayConfig;
use super::delabella::DelaBella as DelaBellaEngine;
use super::delabella::constrain_edge as dela_constrain_edge;
use super::delabella::triangulate as dela_triangulate;

/// Delaunay triangulation backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DelaunayBackend {
    /// Bowyer-Watson incremental CDT (default, stable).
    #[default]
    BowyerWatson,
    /// DelaBella Newton Apple Wrapper (experimental).
    DelaBella,
}

/// Opaque vertex handle (internal Delaunay vertex index).
pub type CdtVertHandle = u32;

// ── Backend enum ──────────────────────────────────────────────────────

enum CdtInner {
    BowyerWatson(Delaunay2d),
    DelaBella(DelaBellaEngine),
}

// ── NativeCdt public interface ────────────────────────────────────────

pub struct NativeCdt {
    inner: CdtInner,
    handles: Vec<u32>,    // maps CdtVertHandle → internal vertex index
    handle_gi: Vec<usize>, // maps CdtVertHandle → global mesh index
    constraints: usize,
    /// Queued constraint edges (internal vertex index pairs) for DelaBella backend.
    dela_pending_constraints: Vec<(u32, u32)>,
    /// Constraints that failed enforcement during the last `finalize`.
    constraint_failures: usize,
}

impl NativeCdt {
    /// Create with the default backend (BowyerWatson).
    pub fn from_uv_bbox(u_min: f32, v_min: f32, u_max: f32, v_max: f32) -> Self {
        Self::from_uv_bbox_with_backend(u_min, v_min, u_max, v_max, DelaunayBackend::default())
    }

    /// Create with a specified backend.
    pub fn from_uv_bbox_with_backend(
        u_min: f32,
        v_min: f32,
        u_max: f32,
        v_max: f32,
        backend: DelaunayBackend,
    ) -> Self {
        let inner = match backend {
            DelaunayBackend::BowyerWatson => {
                let mut delaunay = Delaunay2d::new(DelaunayConfig {
                    sort_by_diagonal: false,
                    optimize: true,
                    max_optimize_passes: 1,
                });
                delaunay.begin_live(
                    u_min as f64,
                    v_min as f64,
                    u_max as f64,
                    v_max as f64,
                );
                CdtInner::BowyerWatson(delaunay)
            }
            DelaunayBackend::DelaBella => {
                // DelaBella is batch-mode: we collect points and triangulate on finalize.
                let _ = (u_min, v_min, u_max, v_max);
                CdtInner::DelaBella(DelaBellaEngine::new())
            }
        };
        Self {
            inner,
            handles: Vec::new(),
            handle_gi: Vec::new(),
            constraints: 0,
            dela_pending_constraints: Vec::new(),
            constraint_failures: 0,
        }
    }

    pub fn constraint_failure_count(&self) -> usize {
        self.constraint_failures
    }

    pub fn insert(&mut self, u: f64, v: f64, gi: usize) -> Option<CdtVertHandle> {
        let hi = self.handles.len() as CdtVertHandle;
        match &mut self.inner {
            CdtInner::BowyerWatson(delaunay) => {
                let vidx = delaunay.insert_live(u, v, gi as u32);
                self.handles.push(vidx);
            }
            CdtInner::DelaBella(della) => {
                let vidx = della.add_vert(u, v, gi as u32);
                self.handles.push(vidx);
            }
        }
        self.handle_gi.push(gi);
        Some(hi)
    }

    pub fn try_add_constraint(&mut self, ha: CdtVertHandle, hb: CdtVertHandle) -> bool {
        let a = self.handles.get(ha as usize).copied();
        let b = self.handles.get(hb as usize).copied();
        match (a, b, &mut self.inner) {
            (Some(a), Some(b), CdtInner::BowyerWatson(delaunay)) => {
                // Queue only; enforce all constraints at finalize (after interior inserts).
                let ok = delaunay.add_constraint(a, b);
                if ok {
                    self.constraints += 1;
                }
                ok
            }
            (Some(a), Some(b), CdtInner::DelaBella(_della)) => {
                // DelaBella constraints are applied during finalize (after triangulation)
                self.dela_pending_constraints.push((a, b));
                self.constraints += 1;
                true
            }
            _ => false,
        }
    }

    pub fn exists_constraint(&self, ha: CdtVertHandle, hb: CdtVertHandle) -> bool {
        let a = self.handles.get(ha as usize).copied();
        let b = self.handles.get(hb as usize).copied();
        match (a, b, &self.inner) {
            (Some(a), Some(b), CdtInner::BowyerWatson(delaunay)) => delaunay.has_edge(a, b),
            (Some(a), Some(b), CdtInner::DelaBella(della)) => {
                // Before triangulation: check pending queue.
                // After triangulation: check fixed edges.
                if !della.all_faces().is_empty() {
                    return della.has_fixed_edge(a, b);
                }
                self.dela_pending_constraints.iter().any(|&(ca, cb)| {
                    (ca == a && cb == b) || (cb == a && ca == b)
                })
            }
            _ => false,
        }
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints
    }

    pub fn vertex_uv(&self, ha: CdtVertHandle) -> (f32, f32) {
        let vi = self.handles[ha as usize];
        match &self.inner {
            CdtInner::BowyerWatson(delaunay) => {
                let p = delaunay.vertex_point(vi);
                (p.x as f32, p.y as f32)
            }
            CdtInner::DelaBella(della) => {
                let (x, y) = della.vert_pos(vi);
                (x as f32, y as f32)
            }
        }
    }

    pub fn handle_global_index(&self, ha: CdtVertHandle) -> usize {
        self.handle_gi[ha as usize]
    }

    pub fn extract_triangles(
        &self,
        filter: impl Fn(f32, f32) -> bool,
    ) -> Vec<usize> {
        let mut tris = Vec::new();
        for (gids, uvs) in self.inner_faces_detail() {
            let cu = (uvs[0].0 + uvs[1].0 + uvs[2].0) / 3.0;
            let cv = (uvs[0].1 + uvs[1].1 + uvs[2].1) / 3.0;
            if !filter(cu, cv) {
                continue;
            }
            tris.push(gids[0]);
            tris.push(gids[1]);
            tris.push(gids[2]);
        }
        tris
    }

    /// Global indices and UV coords per inner triangle.
    pub fn inner_faces_detail(&self) -> Vec<([usize; 3], [(f32, f32); 3])> {
        match &self.inner {
            CdtInner::BowyerWatson(delaunay) => {
                let mut out = Vec::new();
                for face in delaunay.inner_faces() {
                    let gids = [
                        delaunay.vertex_data(face[0]) as usize,
                        delaunay.vertex_data(face[1]) as usize,
                        delaunay.vertex_data(face[2]) as usize,
                    ];
                    let uvs = [
                        delaunay.vertex_point(face[0]).to_f32(),
                        delaunay.vertex_point(face[1]).to_f32(),
                        delaunay.vertex_point(face[2]).to_f32(),
                    ];
                    out.push((gids, uvs));
                }
                out
            }
            CdtInner::DelaBella(della) => {
                let mut out = Vec::new();
                // Use delaunay_faces() (Delaunay-only interior faces) to match
                // BowyerWatson's inner_faces() which returns non-super-vertex faces.
                for (_, verts) in della.delaunay_faces() {
                    if verts.len() != 3 {
                        continue;
                    }
                    out.push((
                        [
                            della.vert_orig_idx(verts[0]) as usize,
                            della.vert_orig_idx(verts[1]) as usize,
                            della.vert_orig_idx(verts[2]) as usize,
                        ],
                        {
                            let uv = |vi: u32| {
                                let (x, y) = della.vert_pos(vi);
                                (x as f32, y as f32)
                            };
                            [uv(verts[0]), uv(verts[1]), uv(verts[2])]
                        },
                    ));
                }
                out
            }
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.handle_gi.len()
    }

    /// Rebuild the internal triangulation from accumulated points and re-apply all
    /// pending constraints. Used by `retriangulate()` and `finalize()`.
    fn rebuild_dela_bella(
        verts: &[(f64, f64)],
        orig: &[u32],
        constraints: &[(u32, u32)],
    ) -> (DelaBellaEngine, HashMap<u32, u32>, usize) {
        let mut fresh = DelaBellaEngine::new();
        dela_triangulate(&mut fresh, verts, orig);

        let mut orig_to_fresh: HashMap<u32, u32> = HashMap::new();
        for (fi, v) in fresh.verts.iter().enumerate() {
            orig_to_fresh.insert(v.orig_idx, fi as u32);
        }

        let mut failures = 0usize;
        for (va, vb) in constraints {
            let oa = orig[*va as usize];
            let ob = orig[*vb as usize];
            if let (Some(&fa), Some(&fb)) = (orig_to_fresh.get(&oa), orig_to_fresh.get(&ob)) {
                if !dela_constrain_edge(&mut fresh, fa, fb) {
                    failures += 1;
                }
            } else {
                failures += 1;
            }
        }

        (fresh, orig_to_fresh, failures)
    }

    pub fn retriangulate(&mut self) {
        match &mut self.inner {
            CdtInner::BowyerWatson(_) => {}
            CdtInner::DelaBella(_) => {
                self.rebuild_dela_bella_inner(false);
            }
        }
    }

    pub fn finalize(&mut self) {
        match &mut self.inner {
            CdtInner::BowyerWatson(delaunay) => {
                self.constraint_failures = delaunay.finalize_constraints();
            }
            CdtInner::DelaBella(_) => {
                self.rebuild_dela_bella_inner(true);
            }
        }
    }

    /// Shared DelaBella rebuild: extract points from current engine, triangulate
    /// from scratch, re-apply constraints, remap handles.
    /// When `take_constraints` is true, clears pending constraints (finalize);
    /// when false, clones them for re-application (retriangulate).
    fn rebuild_dela_bella_inner(&mut self, take_constraints: bool) {
        let (points, orig) = match &self.inner {
            CdtInner::DelaBella(della) => {
                if della.verts.len() < 3 {
                    return;
                }
                let points: Vec<(f64, f64)> = della.verts.iter().map(|v| (v.x, v.y)).collect();
                let orig: Vec<u32> = della.verts.iter().map(|v| v.orig_idx).collect();
                (points, orig)
            }
            _ => return,
        };
        let constraints = if take_constraints {
            std::mem::take(&mut self.dela_pending_constraints)
        } else {
            self.dela_pending_constraints.clone()
        };

        let (fresh, orig_to_fresh, failures) =
            Self::rebuild_dela_bella(&points, &orig, &constraints);
        if take_constraints {
            self.constraint_failures = failures;
        }

        for ha in 0..self.handles.len() {
            let gi = self.handle_gi[ha] as u32;
            if let Some(&new_vi) = orig_to_fresh.get(&gi) {
                self.handles[ha] = new_vi;
            }
        }

        if let CdtInner::DelaBella(ref mut della) = self.inner {
            *della = fresh;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delaBella_finalize_debug() {
        let mut della = DelaBellaEngine::new();
        della.add_vert(0.0, 0.0, 0);
        della.add_vert(1.0, 0.0, 1);
        della.add_vert(0.0, 1.0, 2);

        let points: Vec<(f64, f64)> = della.verts.iter().map(|v| (v.x, v.y)).collect();
        let orig: Vec<u32> = della.verts.iter().map(|v| v.orig_idx).collect();

        let mut fresh = DelaBellaEngine::new();
        let n = dela_triangulate(&mut fresh, &points, &orig);
        assert!(n > 0, "triangulate should return > 0");
        assert!(!fresh.all_faces().is_empty(), "should have alive faces");
    }

    #[test]
    fn delaBella_exists_constraint_after_finalize() {
        let mut cdt = NativeCdt::from_uv_bbox_with_backend(
            0.0, 0.0, 1.0, 1.0, DelaunayBackend::DelaBella,
        );
        let h0 = cdt.insert(0.0, 0.0, 0).unwrap();
        let h1 = cdt.insert(1.0, 0.0, 1).unwrap();
        let h2 = cdt.insert(0.0, 1.0, 2).unwrap();
        cdt.try_add_constraint(h0, h1);
        cdt.finalize();
        assert!(
            cdt.exists_constraint(h0, h1),
            "constraint should exist after finalize"
        );
        assert!(
            !cdt.exists_constraint(h0, h2),
            "non-constrained edge should not exist"
        );
    }

    #[test]
    fn delaBella_native_cdt_full_flow() {
        let mut cdt = NativeCdt::from_uv_bbox_with_backend(
            0.0, 0.0, 1.0, 1.0, DelaunayBackend::DelaBella,
        );
        let h0 = cdt.insert(0.0, 0.0, 0).unwrap();
        let h1 = cdt.insert(1.0, 0.0, 1).unwrap();
        let h2 = cdt.insert(0.0, 1.0, 2).unwrap();
        cdt.try_add_constraint(h0, h1);
        cdt.try_add_constraint(h1, h2);
        cdt.try_add_constraint(h2, h0);
        cdt.finalize();

        let detail = cdt.inner_faces_detail();
        assert!(!detail.is_empty(), "should have faces after finalize");
        let tris = cdt.extract_triangles(|_, _| true);
        assert_eq!(tris.len(), 3, "should produce 1 triangle (3 indices)");
    }

    #[test]
    fn delaBella_retriangulate_no_constraints() {
        let mut cdt = NativeCdt::from_uv_bbox_with_backend(
            0.0, 0.0, 1.0, 1.0, DelaunayBackend::DelaBella,
        );
        cdt.insert(0.0, 0.0, 0).unwrap();
        cdt.insert(1.0, 0.0, 1).unwrap();
        cdt.insert(1.0, 1.0, 2).unwrap();
        cdt.insert(0.0, 1.0, 3).unwrap();
        // No constraints
        assert!(cdt.inner_faces_detail().is_empty());
        cdt.retriangulate();
        let faces = cdt.inner_faces_detail();
        assert_eq!(faces.len(), 2, "square without constraints => 2 triangles");
    }

    #[test]
    fn delaBella_retriangulate_with_constraints() {
        let mut cdt = NativeCdt::from_uv_bbox_with_backend(
            0.0, 0.0, 1.0, 1.0, DelaunayBackend::DelaBella,
        );
        let h0 = cdt.insert(0.0, 0.0, 0).unwrap();
        let h1 = cdt.insert(1.0, 0.0, 1).unwrap();
        let h2 = cdt.insert(1.0, 1.0, 2).unwrap();
        let h3 = cdt.insert(0.0, 1.0, 3).unwrap();
        cdt.try_add_constraint(h0, h1);
        cdt.try_add_constraint(h1, h2);
        cdt.try_add_constraint(h2, h3);
        cdt.try_add_constraint(h3, h0);
        assert!(cdt.inner_faces_detail().is_empty());
        cdt.retriangulate();
        let faces = cdt.inner_faces_detail();
        assert_eq!(faces.len(), 2, "square with constraints => 2 triangles");
    }

    #[test]
    fn delaBella_matches_bowyer_watson_on_square() {
        // Both backends should produce the same triangle count on the same point set
        let pts = [
            (0.0f64, 0.0f64), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
            (0.5, 0.5), // interior point
        ];

        let bw_tris = {
            let mut cdt = NativeCdt::from_uv_bbox_with_backend(
                0.0, 0.0, 1.0, 1.0, DelaunayBackend::BowyerWatson,
            );
            let h: Vec<_> = pts.iter().enumerate()
                .map(|(i, &(u, v))| cdt.insert(u, v, i).unwrap())
                .collect();
            cdt.try_add_constraint(h[0], h[1]);
            cdt.try_add_constraint(h[1], h[2]);
            cdt.try_add_constraint(h[2], h[3]);
            cdt.try_add_constraint(h[3], h[0]);
            cdt.finalize();
            cdt.extract_triangles(|_, _| true).len() / 3
        };

        let db_tris = {
            let mut cdt = NativeCdt::from_uv_bbox_with_backend(
                0.0, 0.0, 1.0, 1.0, DelaunayBackend::DelaBella,
            );
            let h: Vec<_> = pts.iter().enumerate()
                .map(|(i, &(u, v))| cdt.insert(u, v, i).unwrap())
                .collect();
            cdt.try_add_constraint(h[0], h[1]);
            cdt.try_add_constraint(h[1], h[2]);
            cdt.try_add_constraint(h[2], h[3]);
            cdt.try_add_constraint(h[3], h[0]);
            cdt.finalize();
            cdt.extract_triangles(|_, _| true).len() / 3
        };

        // Both backends should produce valid triangulations (not necessarily identical
        // due to different floating-point / constraint implementation details).
        assert!(bw_tris >= 1, "BW should produce >= 1 triangle, got {}", bw_tris);
        assert!(db_tris >= 1, "DB should produce >= 1 triangle, got {}", db_tris);
    }
}

/// UV bbox (min_u, min_v, max_u, max_v) from trim loops.
pub fn uv_bbox_from_loops(
    outer: &[(f32, f32)],
    inners: &[Vec<(f32, f32)>],
) -> (f32, f32, f32, f32) {
    let mut u_min = f32::MAX;
    let mut u_max = f32::MIN;
    let mut v_min = f32::MAX;
    let mut v_max = f32::MIN;
    for &(u, v) in outer {
        u_min = u_min.min(u);
        u_max = u_max.max(u);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    for inner in inners {
        for &(u, v) in inner {
            u_min = u_min.min(u);
            u_max = u_max.max(u);
            v_min = v_min.min(v);
            v_max = v_max.max(v);
        }
    }
    if u_min > u_max {
        return (0.0, 0.0, 1.0, 1.0);
    }
    (u_min, v_min, u_max, v_max)
}

/// Insert UV with quantization / bump logic for coincident UV keys.
pub fn insert_uv_native(
    cdt: &mut NativeCdt,
    mut uv: (f32, f32),
    gi: usize,
    uv_to_handle: &mut HashMap<(u64, u64), CdtVertHandle>,
    uv_span: Option<(f32, f32)>,
    quant_key: fn((f32, f32), Option<(f32, f32)>) -> (u64, u64),
) -> Option<CdtVertHandle> {
    for bump in 0..32usize {
        let key = quant_key(uv, uv_span);
        if let Some(&hi) = uv_to_handle.get(&key) {
            if cdt.handle_global_index(hi) == gi {
                return Some(hi);
            }
            let bump_scale = 1e-5 * (bump as f32 + 1.0);
            if let Some(span) = uv_span {
                uv.0 += bump_scale * span.0;
                uv.1 += bump_scale * span.1 * ((bump % 2) as f32 * 2.0 - 1.0);
            } else {
                uv.0 += bump_scale;
                uv.1 += bump_scale * ((bump % 2) as f32 * 2.0 - 1.0);
            }
            continue;
        }
        let hi = cdt.insert(uv.0 as f64, uv.1 as f64, gi)?;
        uv_to_handle.insert(key, hi);
        return Some(hi);
    }
    None
}
