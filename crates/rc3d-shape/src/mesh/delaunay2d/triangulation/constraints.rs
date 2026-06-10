//! Constrained Delaunay: edge flips, cavity recovery, frontier adjustment.

use std::collections::{HashMap, HashSet, VecDeque};

use super::super::geom::Point2d;
use super::super::half_edge::VertIdx;
use super::super::polygon_mesh::{
    boundary_adjacency, earcut_polygon, mesh_left_polygon_loop, order_boundary_loop,
};
use super::{
    adaptive_orient2d, edge_key, opposite_vertex, segment_crosses_triangle, segments_cross_fast,
    Delaunay2d,
};

impl Delaunay2d {
    /// Enforce queued constraints. Returns the number that could not be recovered.
    pub(super) fn process_constraints(&mut self) -> usize {
        let list: Vec<_> = self.constraints.drain(..).collect();
        let mut failed = 0usize;
        for (a, b) in list {
            if !self.enforce_constraint_full(a, b) {
                failed += 1;
            }
        }
        self.frontier_adjust();
        self.last_constraint_failures = failed;
        failed
    }

    pub(super) fn enforce_constraint_full(&mut self, a: VertIdx, b: VertIdx) -> bool {
        if self.has_mesh_edge(a, b) {
            self.mark_constrained(a, b);
            return true;
        }
        for _ in 0..128 {
            if self.has_mesh_edge(a, b) {
                self.mark_constrained(a, b);
                return true;
            }
            let Some((tri0, tri1, e0, e1, opp0, opp1)) = self.find_crossing_quad(a, b) else {
                break;
            };
            if !self.try_flip_quad(tri0, tri1, e0, e1, opp0, opp1) {
                break;
            }
        }
        if self.has_mesh_edge(a, b) {
            self.mark_constrained(a, b);
            return true;
        }
        let recovered = self.recover_constraint_cavity(a, b);
        if recovered {
            self.mark_constrained(a, b);
            return true;
        }
        false
    }

    pub(super) fn mark_constrained(&mut self, a: VertIdx, b: VertIdx) {
        self.constrained_edges.insert(edge_key(a, b));
    }

    pub(super) fn frontier_adjust(&mut self) {
        let edges: Vec<_> = self.constrained_edges.iter().copied().collect();
        let mut skipped = HashSet::new();
        for (a, b) in edges {
            self.remove_exterior_on_frontier(a, b);
            if !self.has_triangle_on_left(a, b) {
                let _ = self.mesh_left_polygon_of(a, b, &mut skipped);
            }
        }
    }

    fn has_triangle_on_left(&self, a: VertIdx, b: VertIdx) -> bool {
        let pa = self.points[a as usize];
        let pb = self.points[b as usize];
        let key = edge_key(a, b);
        let adj = self.edge_adjacency();
        let Some(tris) = adj.get(&key) else {
            return false;
        };
        for &ti in tris {
            if !self.tri_alive[ti as usize] {
                continue;
            }
            let tri = self.tris[ti as usize];
            let opp = opposite_vertex(tri, a, b);
            let po = self.points[opp as usize];
            if adaptive_orient2d(pa, pb, po) > 0.0 {
                return true;
            }
        }
        false
    }

    /// OCC meshLeftPolygonOf: retriangulate material side of directed edge (a -> b).
    ///
    /// Iterative BFS worklist: each directed edge is processed at most once
    /// (guaranteed by the `skipped` set).
    pub(super) fn mesh_left_polygon_of(
        &mut self,
        a: VertIdx,
        b: VertIdx,
        skipped: &mut HashSet<(VertIdx, VertIdx)>,
    ) -> bool {
        let mut worklist = VecDeque::new();
        worklist.push_back((a, b));

        let mut any_progress = false;

        while let Some((va, vb)) = worklist.pop_front() {
            let key = edge_key(va, vb);
            if skipped.contains(&key) {
                continue;
            }
            skipped.insert(key);

            let adj = boundary_adjacency(self.edge_adjacency(), &self.tri_alive);
            let poly = mesh_left_polygon_loop(&self.points, va, vb, &adj, skipped);
            if poly.len() < 3 {
                continue;
            }

            for tri in earcut_polygon(&self.points, &poly) {
                self.add_triangle_no_circle(tri);
            }
            any_progress = true;

            let pn = poly.len();
            for i in 0..pn {
                let ea = poly[i];
                let eb = poly[(i + 1) % pn];
                if !self.has_triangle_on_left(ea, eb) {
                    worklist.push_back((ea, eb));
                }
            }
        }

        any_progress
    }

    fn remove_exterior_on_frontier(&mut self, a: VertIdx, b: VertIdx) {
        let pa = self.points[a as usize];
        let pb = self.points[b as usize];
        let map = self.edge_adjacency();
        let key = edge_key(a, b);
        let Some(tris) = map.get(&key).cloned() else {
            return;
        };
        for ti in tris {
            if !self.tri_alive[ti as usize] {
                continue;
            }
            let tri = self.tris[ti as usize];
            let opp = opposite_vertex(tri, a, b);
            let po = self.points[opp as usize];
            let side = adaptive_orient2d(pa, pb, po);
            if side < 0.0 {
                self.kill_triangle(ti);
            }
        }
    }

    fn recover_constraint_cavity(&mut self, a: VertIdx, b: VertIdx) -> bool {
        let cavity = self.triangles_intersected_by_segment(a, b);
        if cavity.is_empty() {
            return false;
        }

        let mut boundary: HashMap<(VertIdx, VertIdx), (VertIdx, VertIdx)> = HashMap::new();
        for &ti in &cavity {
            if !self.tri_alive[ti as usize] {
                continue;
            }
            let tri = self.tris[ti as usize];
            for k in 0..3 {
                let v0 = tri[k];
                let v1 = tri[(k + 1) % 3];
                let ek = edge_key(v0, v1);
                if let std::collections::hash_map::Entry::Vacant(e) = boundary.entry(ek) {
                    e.insert((v0, v1));
                } else {
                    boundary.remove(&ek);
                }
            }
            self.kill_triangle(ti);
        }

        let loop_verts = order_boundary_loop(&self.points, a, b, &boundary);
        if loop_verts.len() >= 3 {
            self.mesh_polygon(&loop_verts);
        }
        self.has_mesh_edge(a, b)
    }

    fn triangles_intersected_by_segment(&self, a: VertIdx, b: VertIdx) -> HashSet<u32> {
        let pa = self.points[a as usize];
        let pb = self.points[b as usize];
        let mid = Point2d::new((pa.x + pb.x) * 0.5, (pa.y + pb.y) * 0.5);
        let Some(seed) = self
            .find_seed_triangle(pa)
            .or_else(|| self.find_seed_triangle(pb))
            .or_else(|| self.find_seed_triangle(mid))
        else {
            return HashSet::new();
        };

        let mut cavity = HashSet::new();
        let mut stack = vec![seed];
        while let Some(ti) = stack.pop() {
            if !self.tri_alive[ti as usize] || cavity.contains(&ti) {
                continue;
            }
            if !segment_crosses_triangle(self, ti, a, b, pa, pb) {
                continue;
            }
            cavity.insert(ti);
            for nbr in self.neighbor_triangles(ti) {
                if !cavity.contains(&nbr) {
                    stack.push(nbr);
                }
            }
        }
        cavity
    }

    fn mesh_polygon(&mut self, verts: &[VertIdx]) {
        for tri in earcut_polygon(&self.points, verts) {
            self.add_triangle_no_circle(tri);
        }
    }

    fn find_crossing_quad(
        &self,
        a: VertIdx,
        b: VertIdx,
    ) -> Option<(u32, u32, VertIdx, VertIdx, VertIdx, VertIdx)> {
        let pa = self.points[a as usize];
        let pb = self.points[b as usize];
        let adj = self.edge_adjacency();

        let s_min_x = pa.x.min(pb.x);
        let s_max_x = pa.x.max(pb.x);
        let s_min_y = pa.y.min(pb.y);
        let s_max_y = pa.y.max(pb.y);

        for &ti in &self.alive_tris {
            let tri = self.tris[ti as usize];
            for k in 0..3 {
                let e0 = tri[k];
                let e1 = tri[(k + 1) % 3];
                if (e0 == a && e1 == b) || (e0 == b && e1 == a) {
                    continue;
                }
                let p0 = self.points[e0 as usize];
                let p1 = self.points[e1 as usize];
                if p0.x.max(p1.x) < s_min_x || p0.x.min(p1.x) > s_max_x ||
                    p0.y.max(p1.y) < s_min_y || p0.y.min(p1.y) > s_max_y
                {
                    continue;
                }
                if segments_cross_fast(pa, pb, p0, p1) {
                    let key = edge_key(e0, e1);
                    let tris = adj.get(&key)?;
                    if tris.len() != 2 {
                        continue;
                    }
                    let tri0 = tris[0];
                    let tri1 = tris[1];
                    let t0 = self.tris[tri0 as usize];
                    let opp0 = opposite_vertex(t0, e0, e1);
                    let t1 = self.tris[tri1 as usize];
                    let opp1 = opposite_vertex(t1, e0, e1);
                    return Some((tri0, tri1, e0, e1, opp0, opp1));
                }
            }
        }
        None
    }

    pub(super) fn try_flip_quad(
        &mut self,
        tri0: u32,
        tri1: u32,
        e0: VertIdx,
        e1: VertIdx,
        opp0: VertIdx,
        opp1: VertIdx,
    ) -> bool {
        let pa = self.points[opp0 as usize];
        let pb = self.points[opp1 as usize];
        let pc = self.points[e0 as usize];
        let pd = self.points[e1 as usize];

        let o1 = adaptive_orient2d(pa, pb, pc);
        let o2 = adaptive_orient2d(pa, pb, pd);
        if !segments_cross_fast(pa, pb, pc, pd) {
            return false;
        }
        if o1 == 0.0 || o2 == 0.0 {
            return false;
        }

        self.kill_triangle(tri0);
        self.kill_triangle(tri1);
        self.add_triangle_no_circle([opp0, opp1, e0]);
        self.add_triangle_no_circle([opp0, e1, opp1]);
        true
    }
}
