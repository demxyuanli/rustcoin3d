//! Core data structures for DelaBella Newton Apple Wrapper triangulation.
//!
//! Index-based storage arrays with free-list recycling. Triangle adjacency
//! uses the `f[3]` convention: `f[i]` is the face across the edge opposite `v[i]`,
//! i.e. sharing edge `v[(i+1)%3] — v[(i+2)%3]`.

use super::predicates::{adaptive_incircle, adaptive_orient2d};

// ── Constants ─────────────────────────────────────────────────────────

/// Invalid index sentinel (u32::MAX).
pub const INVALID: u32 = u32::MAX;

// Face flags — bits 6, 7, and a dead bit that uses a non-edge bit.
// Edge bits occupy bits 0–2 (1 bit per edge for fixed/not-fixed).
// This avoids the collision that existed when edge 2's fixed bit (0x20)
// overlapped with the old FLAG_DEAD.
pub const FLAG_DELAUNAY: u8 = 0x08; // bit 3
pub const FLAG_HULL: u8 = 0x10;     // bit 4
pub const FLAG_DEAD: u8 = 0x20;     // bit 5

// Edge-bit layout: 3 edges × 1 bit each (fixed/not-fixed) in bits 0–2.
const EDGE_BITS_PER: u8 = 1;
const EDGE_FIXED: u8 = 0x01;

// ── Vertex ────────────────────────────────────────────────────────────

/// A 2D vertex in the triangulation.
#[derive(Clone, Debug)]
pub struct Vert {
    pub x: f64,
    pub y: f64,
    /// Original input index (user data).
    pub orig_idx: u32,
    /// Multi-purpose intrusive link: free list, hull chain, boundary walk.
    pub next: u32,
}

impl Vert {
    pub fn new(x: f64, y: f64, orig_idx: u32) -> Self {
        Self {
            x,
            y,
            orig_idx,
            next: INVALID,
        }
    }
}

// ── Face (triangle) ──────────────────────────────────────────────────

/// A triangular face with adjacency links.
///
/// Convention: `f[i]` is the neighbor face across the edge opposite `v[i]`,
/// meaning it shares edge `v[(i+1)%3] — v[(i+2)%3]`.
#[derive(Clone, Debug)]
pub struct Face {
    /// Vertex indices.
    pub v: [u32; 3],
    /// Neighbor face indices. `f[i]` is opposite `v[i]`.
    pub f: [u32; 3],
    /// Bit flags: FLAG_DELAUNAY | FLAG_HULL | FLAG_DEAD | edge bits.
    pub flags: u8,
    /// Multi-purpose intrusive link: free list, hull list, etc.
    pub next: u32,
}

impl Face {
    pub fn new(v0: u32, v1: u32, v2: u32) -> Self {
        Self {
            v: [v0, v1, v2],
            f: [INVALID; 3],
            flags: 0,
            next: INVALID,
        }
    }

    /// Find which vertex slot (0, 1, 2) contains `vi`.
    #[inline]
    pub fn find_vert(&self, vi: u32) -> Option<usize> {
        if self.v[0] == vi {
            Some(0)
        } else if self.v[1] == vi {
            Some(1)
        } else if self.v[2] == vi {
            Some(2)
        } else {
            None
        }
    }

    /// Find which neighbor slot corresponds to edge (a, b).
    /// Returns the slot `i` such that the edge is `v[(i+1)%3] — v[(i+2)%3]`.
    #[inline]
    pub fn find_edge_slot(&self, a: u32, b: u32) -> Option<usize> {
        for i in 0..3 {
            let j = (i + 1) % 3;
            let k = (i + 2) % 3;
            if (self.v[j] == a && self.v[k] == b) || (self.v[j] == b && self.v[k] == a) {
                return Some(i);
            }
        }
        None
    }

    /// The vertex opposite edge slot `i`.
    #[inline]
    pub fn opp_vert(&self, edge_slot: usize) -> u32 {
        self.v[edge_slot]
    }

    /// The two vertices forming edge slot `i`.
    #[inline]
    pub fn edge_verts(&self, edge_slot: usize) -> (u32, u32) {
        let j = (edge_slot + 1) % 3;
        let k = (edge_slot + 2) % 3;
        (self.v[j], self.v[k])
    }

    // ── Edge-bit manipulation ─────────────────────────────────────────

    /// Get whether edge at position `at` is fixed (constrained).
    #[inline]
    pub fn get_edge_fixed(&self, at: usize) -> bool {
        let shift = at as u8 * EDGE_BITS_PER;
        (self.flags >> shift) & EDGE_FIXED != 0
    }

    /// Set whether edge at position `at` is fixed.
    #[inline]
    pub fn set_edge_fixed(&mut self, at: usize, fixed: bool) {
        let shift = at as u8 * EDGE_BITS_PER;
        if fixed {
            self.flags |= EDGE_FIXED << shift;
        } else {
            self.flags &= !(EDGE_FIXED << shift);
        }
    }

    /// Toggle the "fixed" bit for edge at position `at`.
    #[inline]
    pub fn toggle_edge_fixed(&mut self, at: usize) {
        let shift = at as u8 * EDGE_BITS_PER;
        self.flags ^= EDGE_FIXED << shift;
    }

    /// Mark an edge as fixed (constrained).
    #[inline]
    pub fn mark_edge_fixed(&mut self, at: usize) {
        let shift = at as u8 * EDGE_BITS_PER;
        self.flags |= EDGE_FIXED << shift;
    }

    /// Is the edge at position `at` constrained (fixed)?
    #[inline]
    pub fn is_edge_fixed(&self, at: usize) -> bool {
        self.get_edge_fixed(at)
    }

    /// Rotate edge bits CCW (shift by one position). Used after vertex rotation.
    pub fn rotate_edge_flags_ccw(&mut self) {
        let b0 = self.get_edge_fixed(0);
        let b1 = self.get_edge_fixed(1);
        let b2 = self.get_edge_fixed(2);
        self.set_edge_fixed(0, b1);
        self.set_edge_fixed(1, b2);
        self.set_edge_fixed(2, b0);
    }

    /// Rotate edge bits CW. Used after vertex rotation in the other direction.
    pub fn rotate_edge_flags_cw(&mut self) {
        let b0 = self.get_edge_fixed(0);
        let b1 = self.get_edge_fixed(1);
        let b2 = self.get_edge_fixed(2);
        self.set_edge_fixed(0, b2);
        self.set_edge_fixed(1, b0);
        self.set_edge_fixed(2, b1);
    }

    #[inline]
    pub fn is_dead(&self) -> bool {
        (self.flags & FLAG_DEAD) != 0
    }

    #[inline]
    pub fn is_delaunay(&self) -> bool {
        (self.flags & FLAG_DELAUNAY) != 0
    }

    #[inline]
    pub fn is_hull(&self) -> bool {
        (self.flags & FLAG_HULL) != 0
    }
}

// ── DelaBella triangulation struct ────────────────────────────────────

/// Main DelaBella triangulation state.
pub struct DelaBella {
    /// All vertices (including duplicates, indexed by internal u32).
    pub verts: Vec<Vert>,
    /// All faces (alive and dead, indexed by internal u32).
    pub faces: Vec<Face>,
    /// Head of the free face list (intrusive via Face::next).
    pub face_free: u32,
    /// Head of Delaunay face list.
    pub dela_first: u32,
    /// Head of hull face list.
    pub hull_first: u32,
    /// Head of boundary vertex circular list.
    pub boundary_first: u32,
    /// Number of distinct input points (after dedup).
    pub n_points: usize,
    /// Original input points (before sort/dedup).
    pub input_points: Vec<(f64, f64)>,
}

impl DelaBella {
    pub fn new() -> Self {
        Self {
            verts: Vec::new(),
            faces: Vec::new(),
            face_free: INVALID,
            dela_first: INVALID,
            hull_first: INVALID,
            boundary_first: INVALID,
            n_points: 0,
            input_points: Vec::new(),
        }
    }

    // ── Allocation ────────────────────────────────────────────────────

    /// Add a vertex and return its index.
    pub fn add_vert(&mut self, x: f64, y: f64, orig_idx: u32) -> u32 {
        let idx = self.verts.len() as u32;
        self.verts.push(Vert::new(x, y, orig_idx));
        idx
    }

    /// Allocate a face from the free list or grow the array.
    pub fn alloc_face(&mut self) -> u32 {
        if self.face_free != INVALID {
            let idx = self.face_free;
            self.face_free = self.faces[idx as usize].next;
            self.faces[idx as usize].flags = 0;
            self.faces[idx as usize].next = INVALID;
            self.faces[idx as usize].f = [INVALID; 3];
            return idx;
        }
        let idx = self.faces.len() as u32;
        self.faces.push(Face::new(INVALID, INVALID, INVALID));
        idx
    }

    /// Return a face to the free list.
    pub fn free_face(&mut self, idx: u32) {
        self.faces[idx as usize].flags = FLAG_DEAD;
        self.faces[idx as usize].next = self.face_free;
        self.face_free = idx;
    }

    /// Create a face with given vertices and neighbor indices.
    /// Snapshots neighbor data before mutation to satisfy borrow checker.
    pub fn make_face(
        &mut self,
        v0: u32,
        v1: u32,
        v2: u32,
        f0: u32,
        f1: u32,
        f2: u32,
    ) -> u32 {
        let fi = self.alloc_face();
        {
            let face = &mut self.faces[fi as usize];
            face.v = [v0, v1, v2];
            face.f = [f0, f1, f2];
            face.flags = 0;
            face.next = INVALID;
        }

        // Collect neighbor updates into a vec to avoid borrowing faces while mutating
        let mut updates: Vec<(u32, u32, usize)> = Vec::new(); // (neighbor_idx, new_fi, slot_in_neighbor)
        for slot in 0..3 {
            let ni = self.faces[fi as usize].f[slot];
            if ni != INVALID {
                let (va, vb) = {
                    let f = &self.faces[fi as usize];
                    f.edge_verts(slot)
                };
                if let Some(ns) = self.faces[ni as usize].find_edge_slot(va, vb) {
                    updates.push((ni, fi, ns));
                }
            }
        }
        for (ni, new_fi, ns) in updates {
            self.faces[ni as usize].f[ns] = new_fi;
        }

        fi
    }

    /// Kill a face: mark dead, detach from neighbor adjacency, free it.
    /// Snapshots all data first to avoid borrow conflicts.
    pub fn kill_face(&mut self, fi: u32) {
        // Snapshot neighbor info before mutation
        let neighbors;
        let edge_info: [(u32, u32); 3];
        {
            let face = &self.faces[fi as usize];
            neighbors = face.f;
            edge_info = [
                face.edge_verts(0),
                face.edge_verts(1),
                face.edge_verts(2),
            ];
        }

        // Detach from neighbors
        for slot in 0..3 {
            let ni = neighbors[slot];
            if ni != INVALID && (self.faces[ni as usize].flags & FLAG_DEAD) == 0 {
                let (va, vb) = edge_info[slot];
                if let Some(ns) = self.faces[ni as usize].find_edge_slot(va, vb) {
                    self.faces[ni as usize].f[ns] = INVALID;
                }
            }
        }

        self.free_face(fi);
    }

    // ── Queries ───────────────────────────────────────────────────────

    /// Test if a face's circumcircle contains point (px, py).
    pub fn face_incircle(&self, fi: u32, px: f64, py: f64) -> f64 {
        let face = &self.faces[fi as usize];
        let v0 = &self.verts[face.v[0] as usize];
        let v1 = &self.verts[face.v[1] as usize];
        let v2 = &self.verts[face.v[2] as usize];
        adaptive_incircle(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, px, py)
    }

    /// Orientation test for face: positive = CCW.
    pub fn face_orient(&self, fi: u32) -> f64 {
        let face = &self.faces[fi as usize];
        let v0 = &self.verts[face.v[0] as usize];
        let v1 = &self.verts[face.v[1] as usize];
        let v2 = &self.verts[face.v[2] as usize];
        adaptive_orient2d(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y)
    }

    /// Is point (px, py) on the positive (left) side of the directed edge
    /// at `edge_slot` of face `fi`?
    pub fn is_left_of_edge(&self, fi: u32, edge_slot: usize, px: f64, py: f64) -> bool {
        let face = &self.faces[fi as usize];
        let j = (edge_slot + 1) % 3;
        let k = (edge_slot + 2) % 3;
        let va = &self.verts[face.v[j] as usize];
        let vb = &self.verts[face.v[k] as usize];
        adaptive_orient2d(va.x, va.y, vb.x, vb.y, px, py) > 0.0
    }

    /// Iterate all non-dead Delaunay faces, returning (face_idx, [v0, v1, v2]).
    pub fn delaunay_faces(&self) -> Vec<(u32, [u32; 3])> {
        let mut result = Vec::new();
        let mut fi = self.dela_first;
        while fi != INVALID {
            let face = &self.faces[fi as usize];
            if !face.is_dead() {
                result.push((fi, face.v));
            }
            fi = face.next;
        }
        result
    }

    /// Iterate all non-dead faces (Delaunay + hull), returning (face_idx, [v0, v1, v2]).
    pub fn all_faces(&self) -> Vec<(u32, [u32; 3])> {
        let mut result = Vec::new();
        for (i, face) in self.faces.iter().enumerate() {
            if !face.is_dead() {
                result.push((i as u32, face.v));
            }
        }
        result
    }

    /// Count non-dead faces.
    pub fn alive_face_count(&self) -> usize {
        self.faces.iter().filter(|f| !f.is_dead()).count()
    }

    /// Get vertex position.
    pub fn vert_pos(&self, vi: u32) -> (f64, f64) {
        let v = &self.verts[vi as usize];
        (v.x, v.y)
    }

    /// Get vertex original index (user data).
    pub fn vert_orig_idx(&self, vi: u32) -> u32 {
        self.verts[vi as usize].orig_idx
    }

    /// Check if a fixed (constrained) edge exists between vertices va and vb.
    pub fn has_fixed_edge(&self, va: u32, vb: u32) -> bool {
        for (fi, _) in self.all_faces() {
            let face = &self.faces[fi as usize];
            if let Some(slot) = face.find_edge_slot(va, vb) {
                if face.is_edge_fixed(slot) {
                    return true;
                }
            }
        }
        false
    }

    /// Pre-allocate face storage for `n` points (max faces = 2n - 4).
    pub fn reserve_faces(&mut self, n: usize) {
        let max_faces = 2 * n + 4;
        if max_faces > self.faces.len() {
            let old_len = self.faces.len();
            self.faces.resize_with(max_faces, || {
                let mut f = Face::new(INVALID, INVALID, INVALID);
                f.flags = FLAG_DEAD;
                f
            });
            // Build free list from new slots
            for i in (old_len..max_faces).rev() {
                self.faces[i].next = self.face_free;
                self.face_free = i as u32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_free_cycle() {
        let mut d = DelaBella::new();
        d.reserve_faces(10);
        let f0 = d.alloc_face();
        let f1 = d.alloc_face();
        assert_ne!(f0, f1);
        d.free_face(f0);
        let f2 = d.alloc_face();
        assert_eq!(f2, f0, "should reuse freed face");
    }

    #[test]
    fn test_face_find_edge_slot() {
        let f = Face::new(0, 1, 2);
        assert_eq!(f.find_edge_slot(1, 2), Some(0)); // opposite v0
        assert_eq!(f.find_edge_slot(0, 2), Some(1)); // opposite v1
        assert_eq!(f.find_edge_slot(0, 1), Some(2)); // opposite v2
        assert_eq!(f.find_edge_slot(5, 6), None);
    }

    #[test]
    fn test_edge_bits_roundtrip() {
        let mut f = Face::new(0, 1, 2);
        assert!(!f.is_edge_fixed(0));
        f.mark_edge_fixed(1);
        assert!(f.is_edge_fixed(1));
        assert!(!f.is_edge_fixed(0));
        assert!(!f.is_edge_fixed(2));
    }

    #[test]
    fn test_edge_flags_rotation() {
        let mut f = Face::new(0, 1, 2);
        f.mark_edge_fixed(0);
        assert!(f.is_edge_fixed(0));
        f.rotate_edge_flags_ccw();
        assert!(f.is_edge_fixed(2), "CCW rotation: fixed bit 0 -> 2");
        assert!(!f.is_edge_fixed(0));
        f.rotate_edge_flags_cw();
        assert!(f.is_edge_fixed(0), "CW rotation: fixed bit 2 -> 0");
    }
}
