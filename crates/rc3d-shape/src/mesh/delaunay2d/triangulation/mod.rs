//! Bowyer-Watson Delaunay triangulation with constrained edges (CDT).
//!
//! Follows the OCC BRepMesh_Delaun workflow:
//! super-triangle -> sorted incremental insertion -> constraint enforcement
//! -> remove auxiliary triangles.

mod constraints;
mod insertion;

use std::collections::{HashMap, HashSet};

use super::circle_index::CircleIndex;
use super::geom::{adaptive_in_circle, circumcircle, robust_orient2d, Point2d};
use super::half_edge::VertIdx;

pub(crate) const EPS: f64 = 1e-12;

/// Output triangle referencing user vertex data indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Triangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct DelaunayConfig {
    /// Sort insertion order by (x + y), OCC ComparatorOfVertexOfDelaun style.
    pub sort_by_diagonal: bool,
    /// Run Lawson edge-flip optimization after insertion.
    pub optimize: bool,
    pub max_optimize_passes: usize,
}

impl Default for DelaunayConfig {
    fn default() -> Self {
        Self {
            sort_by_diagonal: true,
            optimize: true,
            max_optimize_passes: 2,
        }
    }
}

/// 2D Delaunay / constrained Delaunay triangulation engine.
pub struct Delaunay2d {
    pub(super) config: DelaunayConfig,
    pub(super) points: Vec<Point2d>,
    /// External payload per vertex (e.g. global mesh index).
    pub(super) vertex_data: Vec<u32>,
    pub(super) tris: Vec<[VertIdx; 3]>,
    pub(super) tri_alive: Vec<bool>,
    pub(super) tri_circle_slot: Vec<Option<usize>>,
    pub(super) circles: CircleIndex,
    pub(super) super_verts: [VertIdx; 3],
    pub(super) constraints: Vec<(VertIdx, VertIdx)>,
    /// Enforced frontier / boundary edges.
    pub(super) constrained_edges: HashSet<(VertIdx, VertIdx)>,
    pub(super) built: bool,
    pub(super) pending: Vec<VertIdx>,
    /// Incremental mode: super mesh initialized, points inserted one-by-one.
    pub(super) live: bool,
    /// Cached live triangle indices (avoids O(n) scan of tri_alive on every query).
    pub(super) alive_tris: Vec<u32>,
    /// Reverse index: tri_id → position in alive_tris (for O(1) swap-remove on kill).
    pub(super) tri_alive_slot: Vec<Option<usize>>,
    /// Incrementally-maintained edge→triangle adjacency (avoids O(T) rebuild per query).
    pub(super) edge_adj_cache: HashMap<(VertIdx, VertIdx), Vec<u32>>,
    /// Count from the most recent `process_constraints` / `finalize_constraints` call.
    pub(super) last_constraint_failures: usize,
}

impl Default for Delaunay2d {
    fn default() -> Self {
        Self::new(DelaunayConfig::default())
    }
}

impl Delaunay2d {
    pub fn new(config: DelaunayConfig) -> Self {
        Self {
            config,
            points: Vec::new(),
            vertex_data: Vec::new(),
            tris: Vec::new(),
            tri_alive: Vec::new(),
            tri_circle_slot: Vec::new(),
            circles: CircleIndex::new(),
            super_verts: [0, 0, 0],
            constraints: Vec::new(),
            constrained_edges: HashSet::new(),
            built: false,
            pending: Vec::new(),
            live: false,
            alive_tris: Vec::new(),
            tri_alive_slot: Vec::new(),
            edge_adj_cache: HashMap::new(),
            last_constraint_failures: 0,
        }
    }

    /// Constraint edges that failed enforcement in the last finalize/build pass.
    pub fn constraint_failure_count(&self) -> usize {
        self.last_constraint_failures
    }

    pub fn with_capacity(n: usize, config: DelaunayConfig) -> Self {
        let mut s = Self::new(config);
        s.points.reserve(n + 3);
        s.vertex_data.reserve(n + 3);
        s.tris.reserve(n * 2);
        s
    }

    /// Insert a vertex before `build`. Returns internal vertex index.
    pub fn insert(&mut self, x: f64, y: f64, data: u32) -> VertIdx {
        self.built = false;
        let idx = self.points.len() as VertIdx;
        self.points.push(Point2d::new(x, y));
        self.vertex_data.push(data);
        self.pending.push(idx);
        idx
    }

    pub fn insert_f32(&mut self, x: f32, y: f32, data: u32) -> VertIdx {
        self.insert(x as f64, y as f64, data)
    }

    pub fn vertex_count(&self) -> usize {
        self.points.len()
    }

    pub fn vertex_data(&self, v: VertIdx) -> u32 {
        self.vertex_data[v as usize]
    }

    pub fn vertex_point(&self, v: VertIdx) -> Point2d {
        self.points[v as usize]
    }

    /// Queue a constrained edge (call before or after `build`; enforced at build time).
    pub fn add_constraint(&mut self, a: VertIdx, b: VertIdx) -> bool {
        if a as usize >= self.points.len() || b as usize >= self.points.len() || a == b {
            return false;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if !self.constraints.iter().any(|&(x, y)| x == lo && y == hi) {
            self.constraints.push((lo, hi));
        }
        true
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Start incremental triangulation over a known UV bounding box.
    pub fn begin_live(&mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) {
        self.points.clear();
        self.vertex_data.clear();
        self.tris.clear();
        self.tri_alive.clear();
        self.tri_circle_slot.clear();
        self.circles = CircleIndex::new();
        self.constraints.clear();
        self.constrained_edges.clear();
        self.pending.clear();
        self.alive_tris.clear();
        self.tri_alive_slot.clear();
        self.edge_adj_cache.clear();
        self.built = false;
        self.live = true;
        self.init_super_mesh_from_bbox(min_x, min_y, max_x, max_y);
        self.circles
            .set_bounds(min_x, min_y, max_x, max_y, 16);
    }

    /// Insert one vertex in incremental mode (Bowyer-Watson step).
    pub fn insert_live(&mut self, x: f64, y: f64, data: u32) -> VertIdx {
        let idx = self.points.len() as VertIdx;
        self.points.push(Point2d::new(x, y));
        self.vertex_data.push(data);
        self.insert_vertex_internal(idx);
        idx
    }

    /// Enforce a constrained edge in incremental mode.
    pub fn constrain_live(&mut self, a: VertIdx, b: VertIdx) -> bool {
        self.add_constraint(a, b);
        self.enforce_constraint_full(a, b)
    }

    pub fn has_edge(&self, a: VertIdx, b: VertIdx) -> bool {
        self.has_mesh_edge(a, b)
    }

    /// Enforce queued constraints and run frontier cleanup.
    pub fn finalize_constraints(&mut self) -> usize {
        if self.live && !self.constraints.is_empty() {
            self.process_constraints()
        } else {
            self.frontier_adjust();
            self.last_constraint_failures = 0;
            0
        }
    }

    /// Run triangulation on all pending vertices.
    pub fn build(&mut self) {
        if self.live {
            self.built = true;
            return;
        }
        if self.pending.is_empty() {
            self.built = true;
            return;
        }

        let mut order: Vec<VertIdx> = self.pending.clone();
        if self.config.sort_by_diagonal {
            order.sort_by(|&a, &b| {
                let pa = self.points[a as usize];
                let pb = self.points[b as usize];
                pa.x
                    .partial_cmp(&pb.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        pa.y
                            .partial_cmp(&pb.y)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| {
                        (pa.x + pa.y)
                            .partial_cmp(&(pb.x + pb.y))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
        }

        self.init_super_mesh();
        self.circles.set_bounds_from_points(&self.points);

        for &v in &order {
            self.insert_vertex_internal(v);
        }

        self.remove_super_triangles();

        let _ = self.process_constraints();

        if self.config.optimize {
            for _ in 0..self.config.max_optimize_passes {
                if self.lawson_flip_pass() == 0 {
                    break;
                }
            }
        }

        self.pending.clear();
        self.built = true;
    }

    /// Triangles as user-data indices (skips super-triangle vertices).
    pub fn triangles(&self) -> Vec<Triangle> {
        let mut out = Vec::new();
        for (ti, tri) in self.tris.iter().enumerate() {
            if !self.tri_alive[ti] {
                continue;
            }
            if self.uses_super_vertex(*tri) {
                continue;
            }
            out.push(Triangle {
                v0: self.vertex_data[tri[0] as usize],
                v1: self.vertex_data[tri[1] as usize],
                v2: self.vertex_data[tri[2] as usize],
            });
        }
        out
    }

    /// Flat index buffer [i0,i1,i2, ...] using user-data indices.
    pub fn triangle_indices(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for t in self.triangles() {
            out.push(t.v0);
            out.push(t.v1);
            out.push(t.v2);
        }
        out
    }

    /// Internal triangles referencing vertex indices (for CDT cavity tests).
    pub fn inner_faces(&self) -> Vec<[VertIdx; 3]> {
        self.tris
            .iter()
            .enumerate()
            .filter(|(ti, _)| self.tri_alive[*ti] && !self.uses_super_vertex(self.tris[*ti]))
            .map(|(_, t)| *t)
            .collect()
    }

    // ── Triangle storage (shared by insertion + constraints) ──

    /// Core triangle bookkeeping: storage + alive-list + edge adjacency.
    /// CircleIndex registration is handled separately by `add_triangle`.
    fn add_triangle_inner(&mut self, verts: [VertIdx; 3]) -> u32 {
        let ti = self.tris.len() as u32;
        self.tris.push(verts);
        self.tri_alive.push(true);
        self.tri_circle_slot.push(None);

        let slot = self.alive_tris.len();
        self.alive_tris.push(ti);
        if ti as usize >= self.tri_alive_slot.len() {
            self.tri_alive_slot.resize(ti as usize + 1, None);
        }
        self.tri_alive_slot[ti as usize] = Some(slot);
        for k in 0..3 {
            let ek = edge_key(verts[k], verts[(k + 1) % 3]);
            self.edge_adj_cache.entry(ek).or_default().push(ti);
        }
        ti
    }

    pub(super) fn add_triangle(&mut self, verts: [VertIdx; 3]) -> u32 {
        let ti = self.add_triangle_inner(verts);

        let a = self.points[verts[0] as usize];
        let b = self.points[verts[1] as usize];
        let c = self.points[verts[2] as usize];
        if let Some((cc, r_sq)) = circumcircle(a, b, c) {
            let slot = self.circles.insert(ti, cc, r_sq);
            self.tri_circle_slot[ti as usize] = Some(slot);
        }

        ti
    }

    /// Add triangle without updating CircleIndex (for constraint flips).
    /// CircleIndex is only needed for Delaunay vertex insertion queries.
    pub(super) fn add_triangle_no_circle(&mut self, verts: [VertIdx; 3]) -> u32 {
        self.add_triangle_inner(verts)
    }

    pub(super) fn kill_triangle(&mut self, ti: u32) {
        let idx = ti as usize;
        if !self.tri_alive[idx] {
            return;
        }
        self.tri_alive[idx] = false;
        if let Some(slot) = self.tri_circle_slot[idx] {
            self.circles.remove(slot);
            self.tri_circle_slot[idx] = None;
        }
        // O(1) swap-remove using the reverse-index from tri→alive_tris slot.
        // Maintained by add_triangle (push) and kill_triangle (swap-remove).
        if let Some(Some(pos)) = self.tri_alive_slot.get(idx).copied() {
            let last = *self.alive_tris.last().unwrap_or(&0);
            self.alive_tris.swap_remove(pos);
            if pos < self.alive_tris.len() {
                // The element that was swapped into position `pos` needs its slot updated.
                if let Some(s) = self.tri_alive_slot.get_mut(last as usize).and_then(|o| o.as_mut()) {
                    *s = pos;
                }
            }
            self.tri_alive_slot[idx] = None;
        }
        // Remove from edge adjacency cache
        let verts = self.tris[idx];
        for k in 0..3 {
            let ek = edge_key(verts[k], verts[(k + 1) % 3]);
            if let Some(vec) = self.edge_adj_cache.get_mut(&ek) {
                vec.retain(|&t| t != ti);
                if vec.is_empty() {
                    self.edge_adj_cache.remove(&ek);
                }
            }
        }
    }

    pub(super) fn tri_cc_contains(&self, ti: u32, p: Point2d) -> bool {
        let idx = ti as usize;
        if !self.tri_alive[idx] {
            return false;
        }
        let v = self.tris[idx];
        let a = self.points[v[0] as usize];
        let b = self.points[v[1] as usize];
        let c = self.points[v[2] as usize];
        adaptive_in_circle(a, b, c, p) > EPS
    }

    pub(super) fn neighbor_triangles(&self, ti: u32) -> Vec<u32> {
        let tri = self.tris[ti as usize];
        let mut nbrs = Vec::new();
        let map = self.edge_adjacency();
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            let key = edge_key(a, b);
            if let Some(tris) = map.get(&key) {
                for &t in tris {
                    if t != ti {
                        nbrs.push(t);
                    }
                }
            }
        }
        nbrs
    }

    /// Incrementally-maintained edge→triangle adjacency (O(1) lookup, no rebuild).
    pub(super) fn edge_adjacency(&self) -> &HashMap<(VertIdx, VertIdx), Vec<u32>> {
        &self.edge_adj_cache
    }

    pub(super) fn has_mesh_edge(&self, a: VertIdx, b: VertIdx) -> bool {
        let key = edge_key(a, b);
        self.edge_adj_cache.get(&key).is_some_and(|v| !v.is_empty())
    }
}

impl CircleIndex {
    fn set_bounds_from_points(&mut self, points: &[Point2d]) {
        if points.is_empty() {
            self.set_bounds(0.0, 0.0, 1.0, 1.0, 1);
            return;
        }
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        for p in points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        self.set_bounds(min_x, min_y, max_x, max_y, points.len());
    }
}

#[inline]
pub(super) fn edge_key(a: VertIdx, b: VertIdx) -> (VertIdx, VertIdx) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

pub(super) fn opposite_vertex(tri: [VertIdx; 3], e0: VertIdx, e1: VertIdx) -> VertIdx {
    tri.into_iter()
        .find(|&v| v != e0 && v != e1)
        .unwrap_or(e0)
}

pub(super) fn bbox_of_vertices(points: &[Point2d]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for p in points {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    if min_x > max_x {
        return (0.0, 0.0, 1.0, 1.0);
    }
    (min_x, min_y, max_x, max_y)
}

/// Fast f64-only orient2d for preliminary filtering (no exact arithmetic fallback).
/// Returns the 2D cross product (b-a)×(c-a). Not robust for near-degenerate cases.
#[inline]
pub(super) fn fast_orient2d(a: Point2d, b: Point2d, c: Point2d) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Adaptive orient2d: fast f64 when safe, robust exact when UV span is large or ambiguous.
#[inline]
pub(super) fn adaptive_orient2d(a: Point2d, b: Point2d, c: Point2d) -> f64 {
    let min_x = a.x.min(b.x).min(c.x);
    let max_x = a.x.max(b.x).max(c.x);
    let min_y = a.y.min(b.y).min(c.y);
    let max_y = a.y.max(b.y).max(c.y);
    let span = (max_x - min_x).max(max_y - min_y);
    if span > 1e4 {
        return robust_orient2d(a, b, c);
    }
    let fast = fast_orient2d(a, b, c);
    let safety = (span * span * 1e-15).max(1e-12);
    if fast.abs() > safety {
        fast
    } else {
        robust_orient2d(a, b, c)
    }
}

/// Orient test with coordinates relative to `origin` (stable for large UV values).
/// Equivalent to `fast_orient2d(origin, a, b)`.
#[inline]
pub(super) fn orient_rel(origin: Point2d, a: Point2d, b: Point2d) -> f64 {
    fast_orient2d(origin, a, b)
}

/// Fast segment-segment proper intersection test (f64 orient only).
/// Returns true when segments (a,b) and (c,d) cross properly (endpoints excluded).
#[inline]
pub(super) fn segments_cross_fast(a: Point2d, b: Point2d, c: Point2d, d: Point2d) -> bool {
    let o1 = adaptive_orient2d(a, b, c);
    let o2 = adaptive_orient2d(a, b, d);
    let o3 = adaptive_orient2d(c, d, a);
    let o4 = adaptive_orient2d(c, d, b);
    o1 * o2 < 0.0 && o3 * o4 < 0.0
}

pub(super) fn point_in_triangle(p: Point2d, a: Point2d, b: Point2d, c: Point2d) -> bool {
    // Use adaptive orient for correctness on near-boundary points (e.g., sphere poles).
    // Argument order: orient(a,b,p) = (b-a)×(p-a), so orientation is relative to triangle.
    let o1 = adaptive_orient2d(a, b, p);
    let o2 = adaptive_orient2d(b, c, p);
    let o3 = adaptive_orient2d(c, a, p);
    let has_neg = o1 < -EPS || o2 < -EPS || o3 < -EPS;
    let has_pos = o1 > EPS || o2 > EPS || o3 > EPS;
    !has_neg || !has_pos
}

pub(super) fn segments_cross_proper(a: Point2d, b: Point2d, c: Point2d, d: Point2d) -> bool {
    let o1 = robust_orient2d(a, b, c);
    let o2 = robust_orient2d(a, b, d);
    let o3 = robust_orient2d(c, d, a);
    let o4 = robust_orient2d(c, d, b);
    o1 * o2 < -EPS && o3 * o4 < -EPS
}

pub(super) fn segment_crosses_triangle(
    mesh: &Delaunay2d,
    ti: u32,
    a: VertIdx,
    b: VertIdx,
    pa: Point2d,
    pb: Point2d,
) -> bool {
    let tri = mesh.tris[ti as usize];
    if tri.contains(&a) && tri.contains(&b) {
        return true;
    }
    for k in 0..3 {
        let e0 = tri[k];
        let e1 = tri[(k + 1) % 3];
        if (e0 == a && e1 == b) || (e0 == b && e1 == a) {
            continue;
        }
        let p0 = mesh.points[e0 as usize];
        let p1 = mesh.points[e1 as usize];
        // Use fast orient for constraint recovery — avoids expensive expansion arithmetic
        if segments_cross_fast(pa, pb, p0, p1) {
            return true;
        }
    }
    if tri.contains(&a) {
        let pb_in = point_in_triangle(
            pb,
            mesh.points[tri[0] as usize],
            mesh.points[tri[1] as usize],
            mesh.points[tri[2] as usize],
        );
        if pb_in {
            return true;
        }
    }
    if tri.contains(&b) {
        let pa_in = point_in_triangle(
            pa,
            mesh.points[tri[0] as usize],
            mesh.points[tri[1] as usize],
            mesh.points[tri[2] as usize],
        );
        if pa_in {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_square() -> Delaunay2d {
        let mut d = Delaunay2d::default();
        d.insert(0.0, 0.0, 0);
        d.insert(10.0, 0.0, 1);
        d.insert(10.0, 10.0, 2);
        d.insert(0.0, 10.0, 3);
        d.build();
        d
    }

    #[test]
    fn square_delaunay_two_tris() {
        let d = build_square();
        let tris = d.triangles();
        assert_eq!(tris.len(), 2);
    }

    #[test]
    fn cdt_hole_constraints_present() {
        let mut d = Delaunay2d::new(DelaunayConfig {
            sort_by_diagonal: false,
            optimize: false,
            max_optimize_passes: 0,
        });
        d.begin_live(0.0, 0.0, 10.0, 10.0);
        let v0 = d.insert_live(0.0, 0.0, 0);
        let v1 = d.insert_live(10.0, 0.0, 1);
        let v2 = d.insert_live(10.0, 10.0, 2);
        let v3 = d.insert_live(0.0, 10.0, 3);
        let v4 = d.insert_live(3.0, 3.0, 4);
        let v5 = d.insert_live(7.0, 3.0, 5);
        let v6 = d.insert_live(7.0, 7.0, 6);
        let v7 = d.insert_live(3.0, 7.0, 7);
        let _ = (v0, v1, v2, v3);
        d.constrain_live(v4, v5);
        d.constrain_live(v5, v6);
        d.constrain_live(v6, v7);
        d.constrain_live(v7, v4);
        d.finalize_constraints();
        assert!(d.has_edge(v4, v5));
        assert!(d.has_edge(v7, v4));
    }

    #[test]
    fn cdt_hole() {
        let mut d = Delaunay2d::new(DelaunayConfig {
            sort_by_diagonal: false,
            optimize: true,
            max_optimize_passes: 1,
        });
        d.begin_live(0.0, 0.0, 10.0, 10.0);
        d.insert_live(0.0, 0.0, 0);
        d.insert_live(10.0, 0.0, 1);
        d.insert_live(10.0, 10.0, 2);
        d.insert_live(0.0, 10.0, 3);
        let v4 = d.insert_live(3.0, 3.0, 4);
        let v5 = d.insert_live(7.0, 3.0, 5);
        let v6 = d.insert_live(7.0, 7.0, 6);
        let v7 = d.insert_live(3.0, 7.0, 7);
        d.constrain_live(v4, v5);
        d.constrain_live(v5, v6);
        d.constrain_live(v6, v7);
        d.constrain_live(v7, v4);
        d.finalize_constraints();
        let tris = d.triangles();
        assert!(tris.len() >= 4);
    }

    #[test]
    fn interior_point() {
        let mut d = Delaunay2d::default();
        d.insert(0.0, 0.0, 0);
        d.insert(4.0, 0.0, 1);
        d.insert(2.0, 3.0, 2);
        d.insert(2.0, 1.0, 3);
        d.build();
        assert!(d.triangles().len() >= 2);
    }
}
