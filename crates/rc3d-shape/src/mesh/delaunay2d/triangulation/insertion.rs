//! Super-triangle setup, Bowyer-Watson insertion, and Lawson optimization.

use std::collections::{HashMap, HashSet, VecDeque};

use super::super::geom::{adaptive_in_circle, Point2d};
use super::super::half_edge::VertIdx;
use super::{bbox_of_vertices, edge_key, point_in_triangle, Delaunay2d, EPS};

impl Delaunay2d {
    pub(super) fn init_super_mesh(&mut self) {
        let (min_x, min_y, max_x, max_y) = bbox_of_vertices(&self.points);
        self.init_super_mesh_from_bbox(min_x, min_y, max_x, max_y);
    }

    pub(super) fn init_super_mesh_from_bbox(
        &mut self,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) {
        let dx = max_x - min_x;
        let dy = max_y - min_y;
        let delta_min = dx.min(dy).max(1e-6);
        let delta_max = dx.max(dy).max(1e-6);
        let delta = dx + dy + delta_max;

        let v0 = self.points.len() as VertIdx;
        self.points
            .push(Point2d::new((min_x + max_x) * 0.5, max_y + delta_max));
        self.vertex_data.push(u32::MAX);

        let v1 = self.points.len() as VertIdx;
        self.points
            .push(Point2d::new(min_x - delta, min_y - delta_min));
        self.vertex_data.push(u32::MAX);

        let v2 = self.points.len() as VertIdx;
        self.points
            .push(Point2d::new(max_x + delta, min_y - delta_min));
        self.vertex_data.push(u32::MAX);

        self.super_verts = [v0, v1, v2];
        self.add_triangle([v0, v1, v2]);
    }

    pub(super) fn uses_super_vertex(&self, tri: [VertIdx; 3]) -> bool {
        tri.iter().any(|&v| {
            v == self.super_verts[0] || v == self.super_verts[1] || v == self.super_verts[2]
        })
    }

    pub(super) fn remove_super_triangles(&mut self) {
        let supers: HashSet<VertIdx> = self.super_verts.iter().copied().collect();
        // Iterate over a snapshot — kill_triangle mutates alive_tris.
        let snapshot = self.alive_tris.clone();
        for &ti in &snapshot {
            if self.tris[ti as usize].iter().any(|v| supers.contains(v)) {
                self.kill_triangle(ti);
            }
        }
    }

    pub(super) fn insert_vertex_internal(&mut self, v: VertIdx) {
        let p = self.points[v as usize];

        let seed = match self.find_seed_triangle(p) {
            Some(s) => s,
            None => {
                log::debug!("Delaunay2d: no seed triangle for vertex {}, degenerate input", v);
                return;
            }
        };

        let mut cavity = HashSet::new();
        let mut queue = VecDeque::new();
        cavity.insert(seed);
        queue.push_back(seed);

        while let Some(ti) = queue.pop_front() {
            for nbr in self.neighbor_triangles(ti) {
                if cavity.contains(&nbr) {
                    continue;
                }
                if self.tri_cc_contains(nbr, p) {
                    cavity.insert(nbr);
                    queue.push_back(nbr);
                }
            }
        }

        let mut edge_count: HashMap<(VertIdx, VertIdx), (VertIdx, VertIdx)> = HashMap::new();
        for &ti in &cavity {
            let tri = self.tris[ti as usize];
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                let key = edge_key(a, b);
                if let std::collections::hash_map::Entry::Vacant(e) = edge_count.entry(key) {
                    e.insert((a, b));
                } else {
                    edge_count.remove(&key);
                }
            }
        }

        for &ti in &cavity {
            self.kill_triangle(ti);
        }

        for (_, (a, b)) in edge_count {
            self.add_triangle([a, b, v]);
        }
    }

    pub(super) fn find_seed_triangle(&self, p: Point2d) -> Option<u32> {
        use super::orient_rel;

        // Try the spatial index (CircleIndex) first — O(1) lookup in the common case.
        if let Some(ti) = self.circles.query_containing(p).into_iter().next() {
            if self.tri_alive[ti as usize] {
                let tri = self.tris[ti as usize];
                let a = self.points[tri[0] as usize];
                let b = self.points[tri[1] as usize];
                let c = self.points[tri[2] as usize];
                if point_in_triangle(p, a, b, c) {
                    return Some(ti);
                }
            }
        }
        // Linear scan over alive triangles.
        for &ti in &self.alive_tris {
            let tri = self.tris[ti as usize];
            let a = self.points[tri[0] as usize];
            let b = self.points[tri[1] as usize];
            let c = self.points[tri[2] as usize];
            if point_in_triangle(p, a, b, c) {
                return Some(ti);
            }
        }
        // Walk across adjacent triangles toward p (handles CircleIndex gaps).
        let mut ti = *self.alive_tris.first()?;
        let adj = self.edge_adjacency();
        let max_steps = self.alive_tris.len().saturating_mul(4).max(16);
        for _ in 0..max_steps {
            let tri = self.tris[ti as usize];
            let verts = [tri[0], tri[1], tri[2]];
            let pts = [
                self.points[verts[0] as usize],
                self.points[verts[1] as usize],
                self.points[verts[2] as usize],
            ];
            if point_in_triangle(p, pts[0], pts[1], pts[2]) {
                return Some(ti);
            }
            let mut moved = false;
            for k in 0..3 {
                if orient_rel(p, pts[k], pts[(k + 1) % 3]) < 0.0 {
                    let key = edge_key(verts[k], verts[(k + 1) % 3]);
                    if let Some(nbrs) = adj.get(&key) {
                        for &nbr in nbrs {
                            if nbr != ti && self.tri_alive[nbr as usize] {
                                ti = nbr;
                                moved = true;
                                break;
                            }
                        }
                    }
                    if moved {
                        break;
                    }
                }
            }
            if !moved {
                break;
            }
        }
        None
    }

    pub(super) fn lawson_flip_pass(&mut self) -> usize {
        use super::opposite_vertex;

        // Collect edge keys only — avoids cloning every Vec<(u32,u32)>→Vec<u32> entry.
        let keys: Vec<_> = self.edge_adj_cache.keys().copied().collect();
        let mut flipped = 0usize;
        let mut done: HashSet<(VertIdx, VertIdx)> = HashSet::new();

        for key in keys {
            if done.contains(&key) {
                continue;
            }
            let tris = match self.edge_adj_cache.get(&key) {
                Some(t) if t.len() == 2 => t,
                _ => continue,
            };
            let tri0 = tris[0];
            let tri1 = tris[1];
            if !self.tri_alive[tri0 as usize] || !self.tri_alive[tri1 as usize] {
                continue;
            }
            let t0 = self.tris[tri0 as usize];
            let e0 = key.0;
            let e1 = key.1;
            let opp0 = opposite_vertex(t0, e0, e1);
            let t1 = self.tris[tri1 as usize];
            let opp1 = opposite_vertex(t1, e0, e1);

            let pa = self.points[opp0 as usize];
            let pb = self.points[opp1 as usize];
            let pc = self.points[e0 as usize];
            let pd = self.points[e1 as usize];

            if (adaptive_in_circle(pa, pb, pc, pd) > EPS || adaptive_in_circle(pa, pb, pd, pc) > EPS)
                && self.try_flip_quad(tri0, tri1, e0, e1, opp0, opp1) {
                    flipped += 1;
                    done.insert(key);
                }
        }
        flipped
    }
}
