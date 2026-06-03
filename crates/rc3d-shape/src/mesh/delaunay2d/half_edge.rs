//! Half-edge data structure for 2D triangulation.
//!
//! Uses index-based storage for cache-friendly access and stable references.

/// Index type for vertices.
pub type VertIdx = u32;
/// Index type for half-edges.
pub type EdgeIdx = u32;
/// Index type for triangles (faces).
pub type TriIdx = u32;

pub const INVALID_VERT: VertIdx = u32::MAX;
pub const INVALID_EDGE: EdgeIdx = u32::MAX;
pub const INVALID_TRI: TriIdx = u32::MAX;

/// A half-edge in the triangulation.
/// Stores source vertex, opposite half-edge, and adjacent face.
#[derive(Clone, Copy, Debug)]
pub struct HalfEdge {
    /// Source vertex of this half-edge.
    pub vert: VertIdx,
    /// Index of the opposite (twin) half-edge.
    pub twin: EdgeIdx,
    /// Index of the next half-edge in the same triangle.
    pub next: EdgeIdx,
    /// Index of the previous half-edge in the same triangle.
    pub prev: EdgeIdx,
    /// Adjacent triangle. INVALID_TRI if boundary.
    pub tri: TriIdx,
    /// Whether this edge is a constrained (boundary/frontier) edge.
    pub constrained: bool,
}

impl Default for HalfEdge {
    fn default() -> Self {
        Self {
            vert: INVALID_VERT,
            twin: INVALID_EDGE,
            next: INVALID_EDGE,
            prev: INVALID_EDGE,
            tri: INVALID_TRI,
            constrained: false,
        }
    }
}

/// A triangle face in the triangulation.
/// Stores the three half-edges and circumcircle data.
#[derive(Clone, Copy, Debug)]
pub struct TriFace {
    /// One of the three half-edges (the other two are reachable via .next).
    pub edge: EdgeIdx,
    /// Whether this triangle is alive.
    pub alive: bool,
    /// Circumcircle center X (f64 for precision).
    pub cc_x: f64,
    /// Circumcircle center Y.
    pub cc_y: f64,
    /// Circumcircle squared radius.
    pub cc_r_sq: f64,
}

impl Default for TriFace {
    fn default() -> Self {
        Self {
            edge: INVALID_EDGE,
            alive: false,
            cc_x: 0.0,
            cc_y: 0.0,
            cc_r_sq: -1.0,
        }
    }
}

/// Half-edge mesh storage.
pub struct HalfEdgeMesh {
    pub vertices: Vec<super::geom::Point2d>,
    pub edges: Vec<HalfEdge>,
    pub triangles: Vec<TriFace>,
    /// User data index per vertex (maps to external index).
    pub vertex_data: Vec<u32>,
    /// Free list for edge recycling.
    free_edges: Vec<EdgeIdx>,
    /// Free list for triangle recycling.
    free_tris: Vec<TriIdx>,
    /// Number of super-triangle vertices (to be removed after triangulation).
    pub(super) super_vert_count: u32,
}

impl HalfEdgeMesh {
    pub fn new(reserve_verts: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(reserve_verts),
            edges: Vec::with_capacity(reserve_verts * 6),
            triangles: Vec::with_capacity(reserve_verts * 2),
            vertex_data: Vec::with_capacity(reserve_verts),
            free_edges: Vec::new(),
            free_tris: Vec::new(),
            super_vert_count: 0,
        }
    }

    // -- Vertex operations --

    #[inline]
    pub fn add_vertex(&mut self, p: super::geom::Point2d, data: u32) -> VertIdx {
        let idx = self.vertices.len() as VertIdx;
        self.vertices.push(p);
        self.vertex_data.push(data);
        idx
    }

    #[inline]
    pub fn vert_count(&self) -> usize {
        self.vertices.len()
    }

    #[inline]
    pub fn vertex(&self, v: VertIdx) -> super::geom::Point2d {
        self.vertices[v as usize]
    }

    // -- Edge operations --

    pub fn alloc_edge(&mut self) -> EdgeIdx {
        if let Some(idx) = self.free_edges.pop() {
            self.edges[idx as usize] = HalfEdge::default();
            return idx;
        }
        let idx = self.edges.len() as EdgeIdx;
        self.edges.push(HalfEdge::default());
        idx
    }

    pub fn alloc_edge_pair(&mut self) -> (EdgeIdx, EdgeIdx) {
        // When we need a pair, allocate them consecutively for cache locality
        if self.free_edges.len() >= 2 {
            let b = self.free_edges.pop().unwrap();
            let a = self.free_edges.pop().unwrap();
            self.edges[a as usize] = HalfEdge::default();
            self.edges[b as usize] = HalfEdge::default();
            return (a, b);
        }
        let a = self.edges.len() as EdgeIdx;
        self.edges.push(HalfEdge::default());
        self.edges.push(HalfEdge::default());
        let b = a + 1;
        (a, b)
    }

    #[inline]
    pub fn edge(&self, e: EdgeIdx) -> &HalfEdge {
        &self.edges[e as usize]
    }

    #[inline]
    pub fn edge_mut(&mut self, e: EdgeIdx) -> &mut HalfEdge {
        &mut self.edges[e as usize]
    }

    /// Get the three vertex indices of a triangle.
    #[inline]
    pub fn tri_verts(&self, tri: TriIdx) -> [VertIdx; 3] {
        let e0 = self.triangles[tri as usize].edge;
        let e1 = self.edges[e0 as usize].next;
        let e2 = self.edges[e1 as usize].next;
        [
            self.edges[e0 as usize].vert,
            self.edges[e1 as usize].vert,
            self.edges[e2 as usize].vert,
        ]
    }

    /// Get the three edge indices of a triangle.
    #[inline]
    pub fn tri_edges(&self, tri: TriIdx) -> [EdgeIdx; 3] {
        let e0 = self.triangles[tri as usize].edge;
        let e1 = self.edges[e0 as usize].next;
        let e2 = self.edges[e1 as usize].next;
        [e0, e1, e2]
    }

    // -- Triangle operations --

    pub fn alloc_tri(&mut self) -> TriIdx {
        if let Some(idx) = self.free_tris.pop() {
            self.triangles[idx as usize] = TriFace::default();
            return idx;
        }
        let idx = self.triangles.len() as TriIdx;
        self.triangles.push(TriFace::default());
        idx
    }

    #[inline]
    pub fn tri(&self, t: TriIdx) -> &TriFace {
        &self.triangles[t as usize]
    }

    #[inline]
    pub fn tri_mut(&mut self, t: TriIdx) -> &mut TriFace {
        &mut self.triangles[t as usize]
    }

    /// Create a triangle from three half-edge indices.
    /// Sets up the next/prev links and circumcircle.
    pub fn make_triangle(
        &mut self,
        e0: EdgeIdx,
        e1: EdgeIdx,
        e2: EdgeIdx,
        constrained: [bool; 3],
    ) -> TriIdx {
        let ti = self.alloc_tri();
        self.triangles[ti as usize].alive = true;
        self.triangles[ti as usize].edge = e0;

        // Wire next/prev
        self.edges[e0 as usize].next = e1;
        self.edges[e0 as usize].prev = e2;
        self.edges[e0 as usize].tri = ti;
        self.edges[e0 as usize].constrained = constrained[0];

        self.edges[e1 as usize].next = e2;
        self.edges[e1 as usize].prev = e0;
        self.edges[e1 as usize].tri = ti;
        self.edges[e1 as usize].constrained = constrained[1];

        self.edges[e2 as usize].next = e0;
        self.edges[e2 as usize].prev = e1;
        self.edges[e2 as usize].tri = ti;
        self.edges[e2 as usize].constrained = constrained[2];

        // Compute circumcircle
        let v = self.tri_verts(ti);
        let a = self.vertices[v[0] as usize];
        let b = self.vertices[v[1] as usize];
        let c = self.vertices[v[2] as usize];

        if let Some((cc, r_sq)) = super::geom::circumcircle(a, b, c) {
            self.triangles[ti as usize].cc_x = cc.x;
            self.triangles[ti as usize].cc_y = cc.y;
            self.triangles[ti as usize].cc_r_sq = r_sq;
        } else {
            // Degenerate: very large radius
            self.triangles[ti as usize].cc_x = 0.0;
            self.triangles[ti as usize].cc_y = 0.0;
            self.triangles[ti as usize].cc_r_sq = f64::MAX;
        }

        ti
    }

    /// Kill a triangle (mark dead, return its boundary edges).
    /// Returns the boundary half-edges (edges whose twin is not in a dead triangle).
    pub fn kill_triangle(&mut self, tri: TriIdx) {
        self.triangles[tri as usize].alive = false;
        self.free_tris.push(tri);

        let edges = self.tri_edges(tri);
        for &e in &edges {
            self.edges[e as usize].tri = INVALID_TRI;
            let twin = self.edges[e as usize].twin;
            if twin != INVALID_EDGE {
                // Detach from neighbor's twin link (keep twin link valid for boundary walking)
            }
        }
    }

    /// Create a boundary edge pair (two half-edges pointing at each other).
    /// Returns (edge_from_a_to_b, edge_from_b_to_a).
    pub fn make_boundary_edge_pair(
        &mut self,
        va: VertIdx,
        vb: VertIdx,
        constrained: bool,
    ) -> (EdgeIdx, EdgeIdx) {
        let (ea, eb) = self.alloc_edge_pair();
        self.edges[ea as usize].vert = va;
        self.edges[ea as usize].twin = eb;
        self.edges[ea as usize].tri = INVALID_TRI;
        self.edges[ea as usize].constrained = constrained;

        self.edges[eb as usize].vert = vb;
        self.edges[eb as usize].twin = ea;
        self.edges[eb as usize].tri = INVALID_TRI;
        self.edges[eb as usize].constrained = constrained;

        (ea, eb)
    }

    /// Iterate over alive triangle indices.
    pub fn alive_tri_indices(&self) -> impl Iterator<Item = TriIdx> + '_ {
        self.triangles
            .iter()
            .enumerate()
            .filter(|(_, t)| t.alive)
            .map(|(i, _)| i as TriIdx)
    }

    /// Count alive triangles.
    pub fn alive_tri_count(&self) -> usize {
        self.triangles.iter().filter(|t| t.alive).count()
    }
}
