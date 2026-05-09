//! Graph algorithms: topological sort (Kahn, 1962), BFS propagation.

use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

/// Kahn's algorithm: topological sort into parallel layers.
///
/// `edges` are `(from, to)` pairs. `node_count` is the total number of nodes
/// (indices must be in `0..node_count`). Returns groups of node indices that
/// can be processed in parallel — each group depends only on previous groups.
///
/// Nodes with no incoming edges appear in the first group. Unreachable nodes
/// (no edges at all) are NOT included.
///
/// # Examples
///
/// ```
/// use rc3d_core::utils::graph::toposort_layered;
/// // 0 → 1 → 3,  0 → 2 → 3
/// let groups = toposort_layered(&[(0, 2), (2, 3), (0, 1), (1, 3)], 4);
/// assert_eq!(groups[0], vec![0]);
/// assert!(groups[1].contains(&1) && groups[1].contains(&2));
/// assert_eq!(groups[2], vec![3]);
/// ```
pub fn toposort_layered(edges: &[(usize, usize)], node_count: usize) -> Vec<Vec<usize>> {
    let mut in_degree = vec![0u32; node_count];
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); node_count];

    for &(from, to) in edges {
        if from < node_count && to < node_count {
            adjacency[from].push(to);
            in_degree[to] += 1;
        }
    }

    let mut groups = Vec::new();
    let mut queue: Vec<usize> = (0..node_count)
        .filter(|&i| in_degree[i] == 0)
        .collect();

    while !queue.is_empty() {
        let group = std::mem::take(&mut queue);
        for &u in &group {
            for &v in &adjacency[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push(v);
                }
            }
        }
        groups.push(group);
    }
    groups
}

/// Kahn's algorithm: topological sort returning a single linear order.
///
/// Returns `Err(())` if the graph contains a cycle.
///
/// # Examples
///
/// ```
/// use rc3d_core::utils::graph::toposort_linear;
/// // 0 → 1 → 2
/// let order = toposort_linear(&[(0, 1), (1, 2)], 3).unwrap();
/// assert_eq!(order, vec![0, 1, 2]);
/// assert!(toposort_linear(&[(0, 1), (1, 0)], 2).is_err()); // cycle
/// ```
pub fn toposort_linear(edges: &[(usize, usize)], node_count: usize) -> Result<Vec<usize>, ()> {
    let mut in_degree = vec![0u32; node_count];
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); node_count];

    for &(from, to) in edges {
        if from < node_count && to < node_count {
            adjacency[from].push(to);
            in_degree[to] += 1;
        }
    }

    let mut queue: VecDeque<usize> = (0..node_count)
        .filter(|&i| in_degree[i] == 0)
        .collect();
    let mut out = Vec::with_capacity(node_count);

    while let Some(u) = queue.pop_front() {
        out.push(u);
        for &v in &adjacency[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    if out.len() != node_count {
        return Err(());
    }
    Ok(out)
}

/// Breadth-first search from `start`, calling `visit` for each discovered node.
///
/// `neighbors(id)` returns the immediate neighbors of `id`. Each node is
/// visited at most once (guarded by an internal visited set).
///
/// `Id` must be `Copy + Eq + Hash` (e.g. `usize`, `NodeId`, `FieldId`).
pub fn bfs_visit<Id: Copy + Eq + Hash>(
    start: Id,
    mut neighbors: impl FnMut(Id) -> Vec<Id>,
    mut visit: impl FnMut(Id),
) {
    let mut q = VecDeque::new();
    let mut seen = HashSet::new();
    q.push_back(start);
    seen.insert(start);
    while let Some(id) = q.pop_front() {
        visit(id);
        for neighbor in neighbors(id) {
            if seen.insert(neighbor) {
                q.push_back(neighbor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layered_simple_chain() {
        // 0 → 1 → 2
        let groups = toposort_layered(&[(0, 1), (1, 2)], 3);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec![0]);
        assert_eq!(groups[1], vec![1]);
        assert_eq!(groups[2], vec![2]);
    }

    #[test]
    fn layered_parallel_branches() {
        // 0 → 1, 0 → 2
        let groups = toposort_layered(&[(0, 1), (0, 2)], 3);
        assert_eq!(groups[0], vec![0]);
        assert_eq!(groups[1].len(), 2);
        assert!(groups[1].contains(&1));
        assert!(groups[1].contains(&2));
    }

    #[test]
    fn layered_isolated_nodes_ignored() {
        let groups = toposort_layered(&[(0, 1)], 4);
        // node 2 and 3 have no edges → appear in first group
        assert_eq!(groups[0].len(), 3); // 0, 2, 3
        assert_eq!(groups[1], vec![1]);
    }

    #[test]
    fn linear_simple() {
        let order = toposort_linear(&[(0, 1), (1, 2)], 3).unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn linear_cycle_detected() {
        assert!(toposort_linear(&[(0, 1), (1, 0)], 2).is_err());
    }

    #[test]
    fn bfs_visit_order() {
        // 0 → [1, 2], 1 → [3]
        let adj: Vec<Vec<usize>> = vec![vec![1, 2], vec![3], vec![], vec![]];
        let mut visited = Vec::new();
        bfs_visit(
            0,
            |id| adj[id].clone(),
            |id| visited.push(id),
        );
        // BFS from 0: 0, then 1,2, then 3
        assert_eq!(visited[0], 0);
        assert!(visited[1..3].contains(&1));
        assert!(visited[1..3].contains(&2));
        assert_eq!(visited[3], 3);
    }
}
