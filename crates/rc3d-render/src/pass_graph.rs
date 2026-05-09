/// A node in the render pass dependency graph.
#[derive(Clone, Debug)]
pub struct PassNode {
    pub name: &'static str,
    /// Indices of passes that must complete before this one.
    pub depends_on: Vec<usize>,
}

/// Directed acyclic graph of render passes.
/// Topological sort groups independent passes for parallel recording.
pub struct PassDag {
    nodes: Vec<PassNode>,
    /// Groups of pass indices that can be recorded in parallel.
    /// Each group depends on all previous groups.
    pub parallel_groups: Vec<Vec<usize>>,
}

impl PassDag {
    pub fn new(nodes: Vec<PassNode>) -> Self {
        let parallel_groups = Self::toposort_parallel(&nodes);
        Self { nodes, parallel_groups }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn node_name(&self, idx: usize) -> &'static str {
        self.nodes.get(idx).map(|n| n.name).unwrap_or("?")
    }

    /// Kahn's algorithm grouping independent nodes into parallel layers.
    fn toposort_parallel(nodes: &[PassNode]) -> Vec<Vec<usize>> {
        let n = nodes.len();
        let edge_count: usize = nodes.iter().map(|nd| nd.depends_on.len()).sum();
        let mut edges = Vec::with_capacity(edge_count);
        for (i, node) in nodes.iter().enumerate() {
            for &dep in &node.depends_on {
                if dep < n {
                    edges.push((dep, i));
                }
            }
        }
        rc3d_core::utils::graph::toposort_layered(&edges, n)
    }

    /// Build the default render pass DAG for this engine.
    /// Pass indices:
    ///   0=Shadow, 1=DepthPrepass, 2=Solid, 3=Decal, 4=Volume,
    ///   5=Transparent, 6=SSR, 7=SSAO, 8=Fog, 9=DoF,
    ///   10=Bloom, 11=TAA, 12=TonemapFXAA, 13=HUD
    pub fn default_render_dag() -> Self {
        let nodes = vec![
            PassNode { name: "Shadow",        depends_on: vec![] },
            PassNode { name: "DepthPrepass",  depends_on: vec![] },
            PassNode { name: "Solid",         depends_on: vec![0, 1] },
            PassNode { name: "Decal",         depends_on: vec![2] },
            PassNode { name: "Volume",        depends_on: vec![2] },
            PassNode { name: "Transparent",   depends_on: vec![4] },
            PassNode { name: "SSR",           depends_on: vec![5] },
            PassNode { name: "SSAO",          depends_on: vec![3] },
            PassNode { name: "Fog",           depends_on: vec![0] },
            PassNode { name: "DoF",           depends_on: vec![0] },
            PassNode { name: "Bloom",         depends_on: vec![6, 7, 8, 9] },
            PassNode { name: "TAA",           depends_on: vec![10] },
            PassNode { name: "Tonemap+FXAA",  depends_on: vec![11] },
            PassNode { name: "HUD",           depends_on: vec![12] },
        ];
        Self::new(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_dag_is_valid() {
        let dag = PassDag::default_render_dag();
        assert!(dag.node_count() > 0);
        // Verify all nodes appear in exactly one group
        let total_in_groups: usize = dag.parallel_groups.iter().map(|g| g.len()).sum();
        assert_eq!(total_in_groups, dag.node_count());
    }

    #[test]
    fn shadow_and_depth_independent() {
        let dag = PassDag::default_render_dag();
        // Shadow (0) and DepthPrepass (1) should be in the same group (both depend on nothing)
        let first_group = &dag.parallel_groups[0];
        assert!(first_group.contains(&0), "Shadow should be in first group");
        assert!(first_group.contains(&1), "DepthPrepass should be in first group");
    }

    #[test]
    fn solid_comes_after_shadow() {
        let dag = PassDag::default_render_dag();
        // Solid (2) depends on Shadow (0) and Depth (1), so it should be in a later group
        let shadow_is_first = dag.parallel_groups[0].contains(&0);
        let solid_group = dag.parallel_groups.iter().position(|g| g.contains(&2));
        assert!(shadow_is_first);
        assert!(solid_group.unwrap() > 0, "Solid should come after Shadow");
    }
}
