//! Frame render graph: pass nodes, resource IDs, topological ordering, transient markers.
//!
//! Execution still lives in `render_passes`; this module provides the DAG contract and sort.
//!
//! **Meshlet + HZB:** the encoder may run compute cull once before the depth prepass and again
//! after HZB (`execute_passes`). The single `meshlet_cull_post` node here is the dependency
//! contract (cull after HZB); the optional pre-prepass cull is not modeled as a separate edge.
//!
//! **HDR default graph:** `rc3d_forward_default` documents **HDR-on** color flow
//! `RES_SCENE_HDR -> RES_POST_LDR -> RES_SWAPCHAIN_COLOR` (linear `Rgba8Unorm` post buffer, then
//! blit to the swapchain). When `hdr_post_processing` is off, shaded passes target the swapchain
//! directly (no HDR or post-LDR textures).

use std::collections::{HashMap, HashSet, VecDeque};

/// Logical resource produced/consumed by passes (textures, depth, swapchain target).
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ResourceId(pub u32);

pub const RES_SHADOW_DEPTH: ResourceId = ResourceId(0);
pub const RES_SWAPCHAIN_COLOR: ResourceId = ResourceId(1);
pub const RES_MAIN_DEPTH: ResourceId = ResourceId(2);
pub const RES_HZB_MAX: ResourceId = ResourceId(3);
pub const RES_HZB_MIN: ResourceId = ResourceId(4);
pub const RES_SCENE_HDR: ResourceId = ResourceId(5);
pub const RES_POST_LDR: ResourceId = ResourceId(6);

/// Synthetic chain resources so the documented pass sequence is acyclic for topo sort (RMW on shared RTs is not modeled).
pub const CHAIN_0: ResourceId = ResourceId(32);
pub const CHAIN_1: ResourceId = ResourceId(33);
pub const CHAIN_2: ResourceId = ResourceId(34);
pub const CHAIN_3: ResourceId = ResourceId(35);
pub const CHAIN_4: ResourceId = ResourceId(36);
pub const CHAIN_5: ResourceId = ResourceId(37);
pub const CHAIN_6: ResourceId = ResourceId(38);
pub const CHAIN_7: ResourceId = ResourceId(39);
pub const CHAIN_8: ResourceId = ResourceId(40);

#[derive(Clone, Copy, Debug)]
pub enum ResourceLifetime {
    /// Recreated or rebound every frame (shadow map, HDR scene target).
    FrameTransient,
    /// Persists until resize or quality change.
    Swapchain,
    /// Internal pyramid; same lifetime as main depth / frame.
    Internal,
}

#[derive(Clone, Copy, Debug)]
pub struct ResourceSpec {
    pub id: ResourceId,
    pub label: &'static str,
    pub lifetime: ResourceLifetime,
}

/// Pass node: name and read/write resource sets for dependency edges.
#[derive(Clone, Debug)]
pub struct RenderPassNode {
    pub name: &'static str,
    pub reads: &'static [ResourceId],
    pub writes: &'static [ResourceId],
}

/// Full graph description (hand-authored for rc3d forward frame).
pub struct RenderGraph {
    pub passes: Vec<RenderPassNode>,
    pub resources: Vec<ResourceSpec>,
}

impl RenderGraph {
    /// Default forward frame (HDR-on documentation): shadow, meshlet/HZB chain, shading into
    /// `RES_SCENE_HDR`, tonemap/FXAA into `RES_POST_LDR`, HUD into `RES_SWAPCHAIN_COLOR`.
    pub fn rc3d_forward_default() -> Self {
        Self {
            resources: vec![
                ResourceSpec {
                    id: RES_SHADOW_DEPTH,
                    label: "shadow_depth",
                    lifetime: ResourceLifetime::FrameTransient,
                },
                ResourceSpec {
                    id: RES_SWAPCHAIN_COLOR,
                    label: "swapchain_color",
                    lifetime: ResourceLifetime::Swapchain,
                },
                ResourceSpec {
                    id: RES_MAIN_DEPTH,
                    label: "main_depth_stencil",
                    lifetime: ResourceLifetime::FrameTransient,
                },
                ResourceSpec {
                    id: RES_HZB_MAX,
                    label: "hzb_max",
                    lifetime: ResourceLifetime::Internal,
                },
                ResourceSpec {
                    id: RES_HZB_MIN,
                    label: "hzb_min",
                    lifetime: ResourceLifetime::Internal,
                },
                ResourceSpec {
                    id: RES_SCENE_HDR,
                    label: "scene_hdr",
                    lifetime: ResourceLifetime::FrameTransient,
                },
                ResourceSpec {
                    id: RES_POST_LDR,
                    label: "post_ldr",
                    lifetime: ResourceLifetime::FrameTransient,
                },
            ],
            // Pass order edges use CHAIN_* so load/store on the same physical texture does not create cycles.
            passes: vec![
                RenderPassNode {
                    name: "shadow_depth",
                    reads: &[],
                    writes: &[RES_SHADOW_DEPTH, CHAIN_0],
                },
                RenderPassNode {
                    name: "depth_prepass_hzb",
                    reads: &[CHAIN_0],
                    writes: &[RES_SCENE_HDR, RES_MAIN_DEPTH, CHAIN_1],
                },
                RenderPassNode {
                    name: "hzb_build",
                    reads: &[CHAIN_1],
                    writes: &[RES_HZB_MAX, RES_HZB_MIN, CHAIN_2],
                },
                RenderPassNode {
                    name: "meshlet_cull_post",
                    reads: &[CHAIN_2, RES_HZB_MAX, RES_HZB_MIN],
                    writes: &[CHAIN_3],
                },
                RenderPassNode {
                    name: "solid_outline",
                    reads: &[CHAIN_3, RES_SHADOW_DEPTH],
                    writes: &[RES_SCENE_HDR, RES_MAIN_DEPTH, CHAIN_4],
                },
                RenderPassNode {
                    name: "wireframe",
                    reads: &[CHAIN_4],
                    writes: &[RES_SCENE_HDR, RES_MAIN_DEPTH, CHAIN_5],
                },
                RenderPassNode {
                    name: "edge_overlay",
                    reads: &[CHAIN_5],
                    writes: &[RES_SCENE_HDR, CHAIN_6],
                },
                RenderPassNode {
                    name: "selection",
                    reads: &[CHAIN_6],
                    writes: &[RES_SCENE_HDR, CHAIN_7],
                },
                RenderPassNode {
                    name: "post_tonemap_fxaa",
                    reads: &[CHAIN_7, RES_SCENE_HDR],
                    writes: &[RES_POST_LDR, CHAIN_8],
                },
                RenderPassNode {
                    name: "hud",
                    reads: &[CHAIN_8, RES_POST_LDR],
                    writes: &[RES_SWAPCHAIN_COLOR],
                },
            ],
        }
    }

    /// Topological order of pass indices. Edge: writer -> reader if reader reads a resource the writer writes.
    pub fn topological_sort(&self) -> Result<Vec<usize>, &'static str> {
        topological_sort_passes(&self.passes)
    }

    pub fn transient_resource_ids(&self) -> Vec<ResourceId> {
        self.resources
            .iter()
            .filter(|s| matches!(s.lifetime, ResourceLifetime::FrameTransient))
            .map(|s| s.id)
            .collect()
    }
}

/// Kahn topological sort. `passes[i]` depends on `passes[j]` if any read of i is written by j.
pub fn topological_sort_passes(passes: &[RenderPassNode]) -> Result<Vec<usize>, &'static str> {
    let n = passes.len();
    let mut writers: HashMap<ResourceId, Vec<usize>> = HashMap::new();
    for (i, p) in passes.iter().enumerate() {
        for &w in p.writes {
            writers.entry(w).or_default().push(i);
        }
    }
    let mut indeg = vec![0u32; n];
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (j, p) in passes.iter().enumerate() {
        for &r in p.reads {
            if let Some(ws) = writers.get(&r) {
                for &i in ws {
                    if i != j {
                        adj[i].push(j);
                        indeg[j] += 1;
                    }
                }
            }
        }
    }
    let mut q: VecDeque<usize> = indeg.iter().enumerate().filter(|(_, d)| **d == 0).map(|(i, _)| i).collect();
    let mut out = Vec::with_capacity(n);
    while let Some(u) = q.pop_front() {
        out.push(u);
        for &v in &adj[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                q.push_back(v);
            }
        }
    }
    if out.len() != n {
        return Err("render graph cycle");
    }
    Ok(out)
}

/// True if executing `passes` in vector order (`0..len`) respects all producer→consumer edges.
#[inline]
pub fn declaration_order_is_valid(passes: &[RenderPassNode]) -> bool {
    let order: Vec<usize> = (0..passes.len()).collect();
    order_respects_edges(passes, &order)
}

/// Debug helper: for each pass, every read is satisfied by an earlier writer in this graph or is external.
pub fn order_respects_edges(passes: &[RenderPassNode], order: &[usize]) -> bool {
    let mut completed_writes: HashSet<ResourceId> = HashSet::new();
    let mut order_pos: HashMap<usize, usize> = HashMap::new();
    for (pos, &idx) in order.iter().enumerate() {
        order_pos.insert(idx, pos);
    }
    for &idx in order {
        let p = &passes[idx];
        for &r in p.reads {
            let producers: Vec<usize> = passes
                .iter()
                .enumerate()
                .filter(|(_, q)| q.writes.contains(&r))
                .map(|(i, _)| i)
                .collect();
            if producers.is_empty() {
                continue;
            }
            let my_pos = order_pos[&idx];
            if !producers.iter().any(|pi| order_pos.get(pi).map_or(false, |pp| *pp < my_pos)) {
                return false;
            }
        }
        for &w in p.writes {
            completed_writes.insert(w);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rc3d_default_graph_sorts_acyclic() {
        let g = RenderGraph::rc3d_forward_default();
        let order = g.topological_sort().expect("acyclic");
        assert_eq!(order.len(), g.passes.len());
        assert!(order_respects_edges(&g.passes, &order));
        assert!(declaration_order_is_valid(&g.passes));
    }
}
