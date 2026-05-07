//! Bounding Volume Hierarchy for spatial acceleration.
//!
//! Builds a balanced binary BVH from a list of AABBs.
//! Supports frustum culling (via AABB intersection) and ray queries.

use crate::aabb::Aabb;
use crate::math::Vec3;

/// A node in the BVH flat array.
#[derive(Clone, Debug)]
struct BvhNode {
    /// Union AABB of all items in this subtree.
    aabb: Aabb,
    /// For internal nodes: [left_child_idx, right_child_idx].
    /// For leaf nodes: [first_item, item_count].
    data: [u32; 2],
    /// Whether this is a leaf node.
    is_leaf: bool,
    /// Parent node index (None for root).
    parent: Option<usize>,
}

/// Spatial acceleration structure built from axis-aligned bounding boxes.
pub struct Bvh {
    nodes: Vec<BvhNode>,
    /// Item indices stored in leaf nodes (depth-first order).
    item_indices: Vec<u32>,
    /// Original item IDs (external indices).
    item_ids: Vec<u32>,
    /// Per-item AABBs (indexed by item_indices).
    item_aabbs: Vec<Aabb>,
    /// Maps original item index -> BVH leaf node index.
    item_to_leaf: Vec<usize>,
}

impl Bvh {
    /// Build a BVH from a list of (Aabb, external_id) pairs.
    pub fn build(items: &[(Aabb, u32)]) -> Self {
        if items.is_empty() {
            return Self {
                nodes: Vec::new(),
                item_indices: Vec::new(),
                item_ids: Vec::new(),
                item_aabbs: Vec::new(),
                item_to_leaf: Vec::new(),
            };
        }
        let item_aabbs: Vec<Aabb> = items.iter().map(|(a, _)| a.clone()).collect();
        let item_ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
        let mut nodes = Vec::new();
        let mut item_indices = Vec::new();
        let work: Vec<u32> = (0..items.len() as u32).collect();
        Self::build_recursive(&work, items, &mut nodes, &mut item_indices, None);

        // Build item_to_leaf map by scanning all leaf nodes
        let mut item_to_leaf = vec![usize::MAX; items.len()];
        for (node_idx, node) in nodes.iter().enumerate() {
            if node.is_leaf {
                let first = node.data[0] as usize;
                let count = node.data[1] as usize;
                for j in first..first + count {
                    let internal_idx = item_indices[j] as usize;
                    if internal_idx < item_to_leaf.len() {
                        item_to_leaf[internal_idx] = node_idx;
                    }
                }
            }
        }

        Self { nodes, item_indices, item_ids, item_aabbs, item_to_leaf }
    }

    /// Build from a slice of AABBs (IDs are 0..N).
    pub fn build_from_aabbs(aabbs: &[Aabb]) -> Self {
        let items: Vec<(Aabb, u32)> = aabbs.iter().enumerate().map(|(i, a)| (a.clone(), i as u32)).collect();
        Self::build(&items)
    }

    fn build_recursive(
        idxs: &[u32],
        items: &[(Aabb, u32)],
        nodes: &mut Vec<BvhNode>,
        item_indices: &mut Vec<u32>,
        parent_idx: Option<usize>,
    ) -> usize {
        // Compute union AABB
        let mut union_aabb = items[idxs[0] as usize].0.clone();
        for &i in &idxs[1..] {
            union_aabb = union_aabb.union(&items[i as usize].0);
        }

        let leaf_max = 4usize;
        if idxs.len() <= leaf_max {
            let first = item_indices.len() as u32;
            let count = idxs.len() as u32;
            for &i in idxs {
                item_indices.push(i);
            }
            let idx = nodes.len();
            nodes.push(BvhNode {
                aabb: union_aabb,
                data: [first, count],
                is_leaf: true,
                parent: parent_idx,
            });
            return idx;
        }

        // Find longest axis
        let size = union_aabb.size();
        let axis = if size.x >= size.y && size.x >= size.z { 0 }
            else if size.y >= size.z { 1 }
            else { 2 };

        // Sort by centroid on longest axis
        let mut sorted: Vec<u32> = idxs.to_vec();
        sorted.sort_by(|&a, &b| {
            let ca = items[a as usize].0.center();
            let cb = items[b as usize].0.center();
            let va = [ca.x, ca.y, ca.z][axis];
            let vb = [cb.x, cb.y, cb.z][axis];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = sorted.len() / 2;
        let left_slice = &sorted[..mid];
        let right_slice = &sorted[mid..];

        // Reserve slot for internal node (will be filled after children)
        let node_idx = nodes.len();
        nodes.push(BvhNode {
            aabb: union_aabb,
            data: [0, 0],
            is_leaf: false,
            parent: parent_idx,
        });

        let left_idx = Self::build_recursive(left_slice, items, nodes, item_indices, Some(node_idx));
        let right_idx = Self::build_recursive(right_slice, items, nodes, item_indices, Some(node_idx));

        nodes[node_idx].data = [left_idx as u32, right_idx as u32];
        node_idx
    }

    /// Query all external item IDs whose AABBs pass a predicate against the query volume.
    /// The predicate receives each BVH node AABB; returns true if the node may contain
    /// visible items. Leaves that pass are fully collected.
    pub fn query_filter(&self, mut pred: impl FnMut(&Aabb) -> bool, out: &mut Vec<u32>) {
        if self.nodes.is_empty() {
            return;
        }
        self.query_filter_recursive(0, &mut pred, out);
    }

    fn query_filter_recursive(&self, node_idx: usize, pred: &mut impl FnMut(&Aabb) -> bool, out: &mut Vec<u32>) {
        let node = &self.nodes[node_idx];
        if !pred(&node.aabb) {
            return;
        }
        if node.is_leaf {
            let first = node.data[0] as usize;
            let count = node.data[1] as usize;
            for i in first..first + count {
                out.push(self.item_ids[self.item_indices[i] as usize]);
            }
        } else {
            self.query_filter_recursive(node.data[0] as usize, pred, out);
            self.query_filter_recursive(node.data[1] as usize, pred, out);
        }
    }

    /// Query all external item IDs whose AABBs intersect `query_aabb`.
    pub fn query_aabb(&self, query_aabb: &Aabb, out: &mut Vec<u32>) {
        if self.nodes.is_empty() {
            return;
        }
        self.query_aabb_recursive(0, query_aabb, out);
    }

    fn query_aabb_recursive(&self, node_idx: usize, query_aabb: &Aabb, out: &mut Vec<u32>) {
        let node = &self.nodes[node_idx];
        if !node.aabb.intersects(query_aabb) {
            return;
        }
        if node.is_leaf {
            let first = node.data[0] as usize;
            let count = node.data[1] as usize;
            for i in first..first + count {
                let internal_idx = self.item_indices[i] as usize;
                if self.item_aabbs[internal_idx].intersects(query_aabb) {
                    out.push(self.item_ids[internal_idx]);
                }
            }
        } else {
            self.query_aabb_recursive(node.data[0] as usize, query_aabb, out);
            self.query_aabb_recursive(node.data[1] as usize, query_aabb, out);
        }
    }

    /// Query external item IDs whose AABBs intersect the given ray.
    /// Returns (item_id, entry_distance) pairs.
    pub fn query_ray(&self, origin: Vec3, dir: Vec3, out: &mut Vec<(u32, f32)>) {
        if self.nodes.is_empty() {
            return;
        }
        self.query_ray_recursive(0, origin, dir, out);
    }

    fn query_ray_recursive(&self, node_idx: usize, origin: Vec3, dir: Vec3, out: &mut Vec<(u32, f32)>) {
        let node = &self.nodes[node_idx];
        if ray_intersects_aabb(origin, dir, &node.aabb).is_none() {
            return;
        }
        if node.is_leaf {
            let first = node.data[0] as usize;
            let count = node.data[1] as usize;
            for i in first..first + count {
                let internal_idx = self.item_indices[i] as usize;
                if let Some(d) = ray_intersects_aabb(origin, dir, &self.item_aabbs[internal_idx]) {
                    out.push((self.item_ids[internal_idx], d));
                }
            }
        } else {
            self.query_ray_recursive(node.data[0] as usize, origin, dir, out);
            self.query_ray_recursive(node.data[1] as usize, origin, dir, out);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Update specific item AABBs without full rebuild.
    ///
    /// `updates` contains (item_index, new_aabb) pairs for items whose AABB has changed.
    /// `items` is the full list of (Aabb, u32) pairs with current AABBs, used for fallback.
    ///
    /// If more than 50% of items are dirty, falls back to full rebuild.
    pub fn incremental_update(&mut self, updates: &[(usize, Aabb)], items: &[(Aabb, u32)]) {
        let total = items.len();
        if total == 0 || updates.is_empty() {
            return;
        }

        // Fall back to full rebuild if > 50% of items changed
        if updates.len() as f64 > total as f64 * 0.5 {
            *self = Bvh::build(items);
            return;
        }

        // Update per-item AABBs so query_aabb individual checks work correctly
        for &(item_idx, ref new_aabb) in updates {
            if item_idx < self.item_aabbs.len() {
                self.item_aabbs[item_idx] = new_aabb.clone();
            }
        }

        // Find affected leaf nodes and recompute their union AABBs from all contained items
        let mut dirty = std::collections::HashSet::new();
        for &(item_idx, _) in updates {
            if item_idx < self.item_to_leaf.len() {
                let leaf_idx = self.item_to_leaf[item_idx];
                let node = &self.nodes[leaf_idx];
                let first = node.data[0] as usize;
                let count = node.data[1] as usize;
                let mut union_aabb = self.item_aabbs[self.item_indices[first] as usize].clone();
                for j in first + 1..first + count {
                    let internal_idx = self.item_indices[j] as usize;
                    union_aabb = union_aabb.union(&self.item_aabbs[internal_idx]);
                }
                self.nodes[leaf_idx].aabb = union_aabb;
                if let Some(parent) = self.nodes[leaf_idx].parent {
                    if parent < self.nodes.len() {
                        dirty.insert(parent);
                    }
                }
            }
        }

        // Bottom-up refit: propagate AABB changes upward through internal nodes
        while !dirty.is_empty() {
            let mut next = std::collections::HashSet::new();
            for &node_idx in &dirty {
                if !self.nodes[node_idx].is_leaf {
                    let left = self.nodes[node_idx].data[0] as usize;
                    let right = self.nodes[node_idx].data[1] as usize;
                    if left < self.nodes.len() && right < self.nodes.len() {
                        let new_aabb = self.nodes[left].aabb.union(&self.nodes[right].aabb);
                        let changed = self.nodes[node_idx].aabb.min != new_aabb.min
                            || self.nodes[node_idx].aabb.max != new_aabb.max;
                        if changed {
                            self.nodes[node_idx].aabb = new_aabb;
                            if let Some(parent) = self.nodes[node_idx].parent {
                                if parent < self.nodes.len() {
                                    next.insert(parent);
                                }
                            }
                        }
                    }
                }
            }
            dirty = next;
        }
    }
}

/// Ray-AABB intersection test (slab method). Returns entry distance or None.
pub fn ray_intersects_aabb(origin: Vec3, dir: Vec3, aabb: &Aabb) -> Option<f32> {
    let inv_dir = Vec3::new(
        if dir.x.abs() > 1e-12 { 1.0 / dir.x } else { 1e12 },
        if dir.y.abs() > 1e-12 { 1.0 / dir.y } else { 1e12 },
        if dir.z.abs() > 1e-12 { 1.0 / dir.z } else { 1e12 },
    );
    let t0 = (aabb.min - origin) * inv_dir;
    let t1 = (aabb.max - origin) * inv_dir;
    let tmin = t0.min(t1);
    let tmax = t0.max(t1);
    let t_enter = tmin.x.max(tmin.y).max(tmin.z);
    let t_exit = tmax.x.min(tmax.y).min(tmax.z);
    if t_enter <= t_exit && t_exit >= 0.0 {
        Some(t_enter.max(0.0))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_bvh() {
        let bvh = Bvh::build(&[]);
        assert!(bvh.is_empty());
    }

    #[test]
    fn test_single_item() {
        let aabb = Aabb { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(1.0, 1.0, 1.0) };
        let bvh = Bvh::build(&[(aabb.clone(), 42)]);
        let mut out = Vec::new();
        bvh.query_aabb(&aabb, &mut out);
        assert!(out.contains(&42));
    }

    #[test]
    fn test_miss() {
        let aabb = Aabb { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(1.0, 1.0, 1.0) };
        let bvh = Bvh::build(&[(aabb, 0)]);
        let mut out = Vec::new();
        let far = Aabb { min: Vec3::new(10.0, 10.0, 10.0), max: Vec3::new(20.0, 20.0, 20.0) };
        bvh.query_aabb(&far, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_ray_hit() {
        let aabb = Aabb { min: Vec3::new(-1.0, -1.0, 4.0), max: Vec3::new(1.0, 1.0, 6.0) };
        let bvh = Bvh::build(&[(aabb, 7)]);
        let mut out = Vec::new();
        bvh.query_ray(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 7);
    }

    #[test]
    fn test_ray_miss() {
        let aabb = Aabb { min: Vec3::new(-1.0, -1.0, 4.0), max: Vec3::new(1.0, 1.0, 6.0) };
        let bvh = Bvh::build(&[(aabb, 0)]);
        let mut out = Vec::new();
        bvh.query_ray(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn incremental_update_vs_full_rebuild() {
        let items: Vec<(Aabb, u32)> = (0..10)
            .map(|i| {
                let a = Aabb {
                    min: Vec3::new(i as f32, 0.0, 0.0),
                    max: Vec3::new(i as f32 + 0.5, 0.5, 0.5),
                };
                (a, i)
            })
            .collect();

        let mut bvh = Bvh::build(&items);

        // Move item 3 far away
        let new_aabb = Aabb {
            min: Vec3::new(0.0, 5.0, 0.0),
            max: Vec3::new(0.5, 5.5, 0.5),
        };
        let updates = vec![(3usize, new_aabb)];
        bvh.incremental_update(&updates, &items);

        // Verify the moved item is found by query_aabb
        let query_aabb = Aabb {
            min: Vec3::new(0.0, 5.0, 0.0),
            max: Vec3::new(0.5, 5.5, 0.5),
        };
        let mut results = Vec::new();
        bvh.query_aabb(&query_aabb, &mut results);
        assert!(
            results.contains(&3),
            "Should find the moved AABB, got: {:?}",
            results
        );
    }
}
