//! Dirty flag propagation and collection for incremental render cache updates.
//!
//! Works with `NodeEntry::dirty_flags` (defined in `rc3d_scene::node_entry::dirty_flags`).
//! When a node is modified, its dirty flag is set and propagated up to ancestors via CHILDREN.
//! Before traversal, dirty roots are collected — nodes that are dirty but whose parent is clean.

use rc3d_core::NodeId;
use rc3d_scene::node_entry::dirty_flags::*;
use rc3d_scene::SceneGraph;

/// Mark a node and all its ancestors as CHILDREN-dirty.
pub fn mark_node_dirty(graph: &mut SceneGraph, node: NodeId, flag: u8) {
    let Some(entry) = graph.get_mut(node) else { return };
    if entry.dirty_flags & FROZEN != 0 {
        return;
    }
    entry.dirty_flags |= flag;

    if flag & CHILDREN != 0 {
        let mut current = entry.parent;
        while let Some(parent_id) = current {
            if let Some(parent_entry) = graph.get_mut(parent_id) {
                if parent_entry.dirty_flags & FROZEN != 0 {
                    break;
                }
                parent_entry.dirty_flags |= CHILDREN;
                current = parent_entry.parent;
            } else {
                break;
            }
        }
    }
}

/// Collect dirty root nodes — nodes that are dirty but whose parent is clean.
pub fn collect_dirty_roots(graph: &SceneGraph) -> Vec<NodeId> {
    let mut roots = Vec::new();
    if graph.roots().is_empty() {
        return roots;
    }
    for &root in graph.roots() {
        collect_dirty_subtree(graph, root, &mut roots);
    }
    roots
}

fn collect_dirty_subtree(graph: &SceneGraph, node: NodeId, dirty_roots: &mut Vec<NodeId>) {
    let Some(entry) = graph.get(node) else { return };
    if entry.dirty_flags & FROZEN != 0 {
        return;
    }

    let parent_dirty = entry
        .parent
        .and_then(|p| graph.get(p))
        .map(|p| p.dirty_flags != 0)
        .unwrap_or(false);

    if entry.dirty_flags != 0 && !parent_dirty {
        dirty_roots.push(node);
    }
    for &child in &entry.children {
        collect_dirty_subtree(graph, child, dirty_roots);
    }
}

/// Clear all dirty flags after a full frame has been processed.
/// Delegates to `SceneGraph::clear_all_dirty_flags` which iterates the
/// SlotMap directly — O(N) with zero heap allocation.
pub fn clear_all_dirty_flags(graph: &mut SceneGraph) {
    graph.clear_all_dirty_flags();
}
