//! CPU BVH over triangles for ray casts and optional frustum queries.

use rc3d_core::math::Vec3;
use rc3d_core::Aabb;

const DEFAULT_MAX_LEAF: usize = 4;
const DEFAULT_MAX_DEPTH: u32 = 64;

#[derive(Clone, Copy, Debug)]
pub struct BvhRayHit {
    pub t: f32,
    pub tri_id: u32,
    pub bary: Vec3,
    pub normal: Vec3,
}

#[derive(Clone, Debug)]
pub struct BvhTriangle {
    pub vertices: [Vec3; 3],
    /// Caller-defined id (e.g. triangle pick index).
    pub id: u32,
}

#[derive(Clone, Debug)]
struct BvhNode {
    bounds: Aabb,
    /// Left child node index, or unused in leaf.
    left: u32,
    /// Right child index if internal; end (exclusive) of triangle range if leaf.
    right_or_tri_end: u32,
    /// Start of triangle range in `tri_order` if leaf.
    tri_begin: u32,
    is_leaf: bool,
}

/// Bounding volume hierarchy over [`BvhTriangle`] entries.
#[derive(Clone, Debug)]
pub struct Bvh {
    nodes: Vec<BvhNode>,
    /// Permutation: triangle indices in build order for leaves.
    tri_order: Vec<u32>,
    triangles: Vec<BvhTriangle>,
}

impl Bvh {
    /// Build from mesh positions and 3*n triangle indices.
    pub fn from_indexed_mesh(positions: &[Vec3], indices: &[u32]) -> Option<Self> {
        if indices.len() < 3 || indices.len() % 3 != 0 {
            return None;
        }
        let mut tris = Vec::with_capacity(indices.len() / 3);
        for (ti, chunk) in indices.chunks_exact(3).enumerate() {
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
                return None;
            }
            tris.push(BvhTriangle {
                vertices: [positions[i0], positions[i1], positions[i2]],
                id: ti as u32,
            });
        }
        Some(Self::from_triangles(tris))
    }

    pub fn from_triangles(mut triangles: Vec<BvhTriangle>) -> Self {
        if triangles.is_empty() {
            return Self {
                nodes: Vec::new(),
                tri_order: Vec::new(),
                triangles: Vec::new(),
            };
        }
        let n = triangles.len();
        let mut tri_order: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::new();
        Self::build_recursive(
            &mut triangles,
            &mut tri_order,
            &mut nodes,
            0..n,
            0,
            DEFAULT_MAX_LEAF,
            DEFAULT_MAX_DEPTH,
        );
        Self {
            nodes,
            tri_order,
            triangles,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Closest hit along the ray (t > `t_min`).
    pub fn intersect_ray(&self, origin: Vec3, direction: Vec3, t_min: f32) -> Option<BvhRayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let dir = direction.normalize();
        let mut best_t = f32::MAX;
        let mut best_id = 0u32;
        let mut best_bary = Vec3::ZERO;
        let mut best_n = Vec3::Y;
        self.intersect_ray_node(0, origin, dir, t_min, &mut best_t, &mut best_id, &mut best_bary, &mut best_n);
        if best_t < f32::MAX {
            Some(BvhRayHit {
                t: best_t,
                tri_id: best_id,
                bary: best_bary,
                normal: best_n,
            })
        } else {
            None
        }
    }

    fn intersect_ray_node(
        &self,
        node_idx: usize,
        origin: Vec3,
        dir: Vec3,
        t_min: f32,
        best_t: &mut f32,
        best_id: &mut u32,
        best_bary: &mut Vec3,
        best_n: &mut Vec3,
    ) {
        let node = &self.nodes[node_idx];
        if !Self::ray_aabb_intersect(origin, dir, &node.bounds, t_min, *best_t) {
            return;
        }
        if node.is_leaf {
            let begin = node.tri_begin as usize;
            let end = node.right_or_tri_end as usize;
            for ord in begin..end {
                let ti = self.tri_order[ord] as usize;
                let tri = &self.triangles[ti];
                if let Some((t, bary)) = intersect_ray_triangle(origin, dir, tri.vertices[0], tri.vertices[1], tri.vertices[2], t_min) {
                    if t < *best_t {
                        *best_t = t;
                        *best_id = tri.id;
                        *best_bary = bary;
                        let e1 = tri.vertices[1] - tri.vertices[0];
                        let e2 = tri.vertices[2] - tri.vertices[0];
                        let c = e1.cross(e2);
                        *best_n = if c.length_squared() > 1e-20 {
                            c.normalize()
                        } else {
                            Vec3::Y
                        };
                    }
                }
            }
            return;
        }
        self.intersect_ray_node(node.left as usize, origin, dir, t_min, best_t, best_id, best_bary, best_n);
        self.intersect_ray_node(node.right_or_tri_end as usize, origin, dir, t_min, best_t, best_id, best_bary, best_n);
    }

    fn ray_aabb_intersect(origin: Vec3, dir: Vec3, b: &Aabb, t_min: f32, t_max: f32) -> bool {
        let mut t0 = t_min;
        let mut t1 = t_max;
        for a in 0..3 {
            let o = origin[a];
            let d = dir[a];
            if d.abs() < 1e-12 {
                if o < b.min[a] || o > b.max[a] {
                    return false;
                }
                continue;
            }
            let inv_d = 1.0 / d;
            let mut t_near = (b.min[a] - o) * inv_d;
            let mut t_far = (b.max[a] - o) * inv_d;
            if t_near > t_far {
                std::mem::swap(&mut t_near, &mut t_far);
            }
            t0 = t0.max(t_near);
            t1 = t1.min(t_far);
            if t0 > t1 {
                return false;
            }
        }
        true
    }

    /// Collect triangle ids whose AABB intersects all frustum half-spaces (inward normals).
    pub fn triangles_touching_frustum(&self, clip_planes: &[[f32; 4]]) -> Vec<u32> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        self.frustum_node(0, clip_planes, &mut out);
        out
    }

    fn frustum_node(&self, node_idx: usize, planes: &[[f32; 4]], out: &mut Vec<u32>) {
        let node = &self.nodes[node_idx];
        if !aabb_inside_frustum(&node.bounds, planes) {
            return;
        }
        if node.is_leaf {
            let begin = node.tri_begin as usize;
            let end = node.right_or_tri_end as usize;
            for ord in begin..end {
                let ti = self.tri_order[ord] as usize;
                out.push(self.triangles[ti].id);
            }
            return;
        }
        self.frustum_node(node.left as usize, planes, out);
        self.frustum_node(node.right_or_tri_end as usize, planes, out);
    }

    fn build_recursive(
        triangles: &mut [BvhTriangle],
        tri_order: &mut [u32],
        nodes: &mut Vec<BvhNode>,
        range: std::ops::Range<usize>,
        depth: u32,
        max_leaf: usize,
        max_depth: u32,
    ) -> u32 {
        let bounds = bounds_of_range(triangles, tri_order, range.clone());
        let count = range.end - range.start;
        if count <= max_leaf || depth >= max_depth {
            let node_idx = nodes.len() as u32;
            nodes.push(BvhNode {
                bounds,
                left: 0,
                right_or_tri_end: range.end as u32,
                tri_begin: range.start as u32,
                is_leaf: true,
            });
            return node_idx;
        }

        let axis = longest_axis(&bounds);
        let mid = range.start + count / 2;
        tri_order[range.start..range.end].sort_by(|&a, &b| {
            let ca = triangle_centroid(&triangles[a as usize])[axis];
            let cb = triangle_centroid(&triangles[b as usize])[axis];
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let left_child = Self::build_recursive(
            triangles,
            tri_order,
            nodes,
            range.start..mid,
            depth + 1,
            max_leaf,
            max_depth,
        );
        let right_child = Self::build_recursive(
            triangles,
            tri_order,
            nodes,
            mid..range.end,
            depth + 1,
            max_leaf,
            max_depth,
        );
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode {
            bounds,
            left: left_child,
            right_or_tri_end: right_child,
            tri_begin: 0,
            is_leaf: false,
        });
        node_idx
    }
}

fn triangle_centroid(t: &BvhTriangle) -> Vec3 {
    (t.vertices[0] + t.vertices[1] + t.vertices[2]) / 3.0
}

fn bounds_of_range(tris: &[BvhTriangle], order: &[u32], range: std::ops::Range<usize>) -> Aabb {
    let mut b = Aabb::empty();
    for i in range.start..range.end {
        let t = &tris[order[i] as usize];
        for v in &t.vertices {
            b = b.union(&Aabb::from_point(*v));
        }
    }
    b
}

fn longest_axis(b: &Aabb) -> usize {
    let e = b.size();
    if e.x >= e.y && e.x >= e.z {
        0
    } else if e.y >= e.z {
        1
    } else {
        2
    }
}

fn intersect_ray_triangle(
    origin: Vec3,
    dir: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    t_min: f32,
) -> Option<(f32, Vec3)> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = dir.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-8 {
        return None;
    }
    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(q);
    if t > t_min {
        Some((t, Vec3::new(1.0 - u - v, u, v)))
    } else {
        None
    }
}

fn aabb_inside_frustum(b: &Aabb, planes: &[[f32; 4]]) -> bool {
    let corners = [
        Vec3::new(b.min.x, b.min.y, b.min.z),
        Vec3::new(b.max.x, b.min.y, b.min.z),
        Vec3::new(b.min.x, b.max.y, b.min.z),
        Vec3::new(b.max.x, b.max.y, b.min.z),
        Vec3::new(b.min.x, b.min.y, b.max.z),
        Vec3::new(b.max.x, b.min.y, b.max.z),
        Vec3::new(b.min.x, b.max.y, b.max.z),
        Vec3::new(b.max.x, b.max.y, b.max.z),
    ];
    for plane in planes {
        let n = Vec3::new(plane[0], plane[1], plane[2]);
        let d = plane[3];
        let mut any_inside = false;
        for c in &corners {
            if n.dot(*c) + d >= 0.0 {
                any_inside = true;
                break;
            }
        }
        if !any_inside {
            return false;
        }
    }
    true
}
