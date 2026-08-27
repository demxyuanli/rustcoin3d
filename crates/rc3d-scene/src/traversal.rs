//! Scene-graph traversal helpers.
//!
//! [`DfsPreOrder`] walks every node ID without interpreting Switch/LOD semantics.
//! [`scene_traverse`] + [`SceneVisitor`] is the deep module Actions should use:
//! structural navigation (Separator, Switch, LOD, Billboard, …) lives in one place;
//! visitors only supply matrix state and per-node visit hooks.

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::{Appearance, DisplayMode, NodeId};

use crate::node_data::{BillboardNode, NodeData};
use crate::node_entry::NodeEntry;
use crate::scene_graph::SceneGraph;

/// Depth-first pre-order traversal yielding node IDs.
pub struct DfsPreOrder<'a> {
    graph: &'a SceneGraph,
    stack: Vec<NodeId>,
}

impl<'a> DfsPreOrder<'a> {
    pub fn new(graph: &'a SceneGraph, roots: &[NodeId]) -> Self {
        let mut stack = roots.to_vec();
        stack.reverse();
        Self { graph, stack }
    }

    /// Parallel traversal: process each root subtree in its own rayon thread.
    ///
    /// `action_factory` is called per root to create a thread-local action.
    /// The produced action must be `Send` so it can be moved into the worker thread.
    pub fn traverse_parallel<F, T>(graph: &SceneGraph, roots: &[NodeId], action_factory: F)
    where
        F: Fn() -> T + Sync,
        T: FnMut(&SceneGraph, NodeId) + Send + Sync,
    {
        let roots = roots.to_vec();
        rayon::scope(|s| {
            for &root in &roots {
                let mut action = action_factory();
                s.spawn(move |_| {
                    action(graph, root);
                });
            }
        });
    }
}

impl Iterator for DfsPreOrder<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.stack.pop()?;
        if let Some(entry) = self.graph.get(id) {
            // Push children in reverse so leftmost child is visited first
            for child in entry.children.iter().rev() {
                self.stack.push(*child);
            }
        }
        Some(id)
    }
}

impl SceneGraph {
    pub fn traverse_dfs(&self, root: NodeId) -> DfsPreOrder<'_> {
        DfsPreOrder::new(self, &[root])
    }

    pub fn traverse_all(&self) -> DfsPreOrder<'_> {
        DfsPreOrder::new(self, self.roots())
    }

    /// Parallel traversal over all roots of this scene graph.
    pub fn traverse_parallel_all<F, T>(&self, action_factory: F)
    where
        F: Fn() -> T + Sync,
        T: FnMut(&SceneGraph, NodeId) + Send + Sync,
    {
        DfsPreOrder::traverse_parallel(self, self.roots(), action_factory);
    }
}

/// Whether the traversal kernel should walk `entry.children` after [`SceneVisitor::visit_node`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildPolicy {
    /// Recurse into the node's graph children (Group-like).
    Recurse,
    /// Do not recurse (leaf, or the visitor already walked children).
    Skip,
}

/// How [`scene_traverse`] walks children of a Separator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeparatorPolicy {
    /// Coin3D: `enter` / walk each child with normal Transform sibling semantics / `leave`.
    ScopedPush,
    /// RenderCollector: flatten *direct* Transform children into an accumulating model matrix
    /// (only Transform grandchildren are visited; matrix persists for later Separator siblings).
    FlattenDirectTransforms,
}

/// Model/view matrices required by structural nodes (Billboard, Transform, copies).
pub trait TraversalMatrices {
    fn model_matrix(&self) -> Mat4;
    fn set_model_matrix(&mut self, matrix: Mat4);
    fn view_matrix(&self) -> Mat4;
}

/// Visit hooks for non-structural nodes. Structural navigation is owned by [`scene_traverse`].
pub trait SceneVisitor: TraversalMatrices {
    /// Called before Separator children are traversed (Coin3D state push).
    fn enter_separator(&mut self) {}

    /// Called after Separator children are traversed (Coin3D state pop).
    fn leave_separator(&mut self) {}

    /// Separator child-walk strategy. Default is Coin3D [`SeparatorPolicy::ScopedPush`].
    fn separator_policy(&self) -> SeparatorPolicy {
        SeparatorPolicy::ScopedPush
    }

    /// Return false to skip this node and its subtree (e.g. hidden nodes).
    fn should_visit(&self, _node: NodeId) -> bool {
        true
    }

    /// Called when an `InstancedMeshNode` is encountered. The visitor should store
    /// the transforms so they can be applied to the next emitted draw call.
    fn set_instance_transforms(&mut self, _transforms: &[rc3d_core::math::Mat4]) {}

    /// Orthogonal fill + edges (HOOPS-style). Separator push/pop isolates this.
    fn appearance(&self) -> Appearance {
        Appearance::from_display_mode(DisplayMode::default())
    }

    fn set_appearance(&mut self, _app: Appearance) {}

    /// Coin3D `SoDrawStyle` preset (nearest enum for HUD / legacy).
    fn display_mode(&self) -> DisplayMode {
        self.appearance().to_display_mode()
    }

    /// Override both axes from a preset.
    fn set_display_mode(&mut self, mode: DisplayMode) {
        self.set_appearance(Appearance::from_display_mode(mode));
    }

    /// Handle a node that is not driven by the structural kernel.
    ///
    /// Return [`ChildPolicy::Recurse`] to walk `entry.children`, or [`ChildPolicy::Skip`]
    /// if this is a leaf / the visitor already recursed via [`scene_traverse`].
    fn visit_node(
        &mut self,
        graph: &SceneGraph,
        node: NodeId,
        entry: &NodeEntry,
    ) -> ChildPolicy;
}

/// Billboard facing matrix matching Action traversal (axis-aligned or full camera facing).
pub fn billboard_facing(billboard: &BillboardNode, view: Mat4) -> Mat4 {
    let inv = view.inverse();
    if billboard.axis_aligned {
        let mut fwd = Vec3::new(inv.w_axis.x, 0.0, inv.w_axis.z);
        if fwd.length_squared() < 1e-12 {
            return Mat4::IDENTITY;
        }
        fwd = fwd.normalize();
        Mat4::look_at_rh(Vec3::ZERO, fwd, Vec3::Y)
    } else {
        Mat4::from_cols(inv.x_axis, inv.y_axis, inv.z_axis, Mat4::IDENTITY.w_axis)
    }
}

/// Apply `NodeEntry` fill/edge overrides for a nested scope, then restore.
fn with_node_appearance<V: SceneVisitor>(
    visitor: &mut V,
    entry: &NodeEntry,
    f: impl FnOnce(&mut V),
) {
    let inherited = visitor.appearance();
    let next = entry.appearance(inherited);
    if next != inherited {
        visitor.set_appearance(next);
        f(visitor);
        visitor.set_appearance(inherited);
    } else {
        f(visitor);
    }
}

/// Action-style scene traversal with unified Separator / Switch / LOD / Billboard semantics.
pub fn scene_traverse<V: SceneVisitor>(visitor: &mut V, graph: &SceneGraph, node: NodeId) {
    if !visitor.should_visit(node) {
        return;
    }
    let Some(entry) = graph.get(node) else {
        return;
    };

    match &entry.data {
        NodeData::Separator(_) => {
            visitor.enter_separator();
            visitor.set_appearance(entry.appearance(visitor.appearance()));
            match visitor.separator_policy() {
                SeparatorPolicy::ScopedPush => {
                    for &child in &entry.children {
                        scene_traverse(visitor, graph, child);
                    }
                }
                SeparatorPolicy::FlattenDirectTransforms => {
                    // Preserve RenderCollector semantics: accumulate Transform / Rotation
                    // matrices along Separator siblings; only walk grandchildren for those kids.
                    let base = visitor.model_matrix();
                    let mut accum = base;
                    for &child in &entry.children {
                        let Some(ce) = graph.get(child) else {
                            continue;
                        };
                        if let Some(lm) = ce.data.local_matrix() {
                            accum *= lm;
                            visitor.set_model_matrix(accum);
                            with_node_appearance(visitor, ce, |visitor| {
                                for &gc in &ce.children {
                                    scene_traverse(visitor, graph, gc);
                                }
                            });
                        } else {
                            visitor.set_model_matrix(accum);
                            scene_traverse(visitor, graph, child);
                        }
                    }
                }
            }
            visitor.leave_separator();
        }
        NodeData::Billboard(b) => {
            with_node_appearance(visitor, entry, |visitor| {
                let current = visitor.model_matrix();
                let facing = billboard_facing(b, visitor.view_matrix());
                visitor.set_model_matrix(current * facing);
                for &child in &entry.children {
                    scene_traverse(visitor, graph, child);
                }
                visitor.set_model_matrix(current);
            });
        }
        NodeData::ResetTransform(_) => {
            with_node_appearance(visitor, entry, |visitor| {
                let saved = visitor.model_matrix();
                visitor.set_model_matrix(Mat4::IDENTITY);
                for &child in &entry.children {
                    scene_traverse(visitor, graph, child);
                }
                visitor.set_model_matrix(saved);
            });
        }
        NodeData::ExplodedView(ev) => {
            with_node_appearance(visitor, entry, |visitor| {
                let base = visitor.model_matrix();
                for &child in &entry.children {
                    visitor.set_model_matrix(base * Mat4::from_translation(ev.direction * ev.factor));
                    scene_traverse(visitor, graph, child);
                }
                visitor.set_model_matrix(base);
            });
        }
        NodeData::Switch(sw) => {
            with_node_appearance(visitor, entry, |visitor| {
                match sw.which_child {
                    -2 => {}
                    -1 => {
                        for &child in &sw.children {
                            scene_traverse(visitor, graph, child);
                        }
                    }
                    idx if idx >= 0 => {
                        let i = idx as usize;
                        if i < sw.children.len() {
                            scene_traverse(visitor, graph, sw.children[i]);
                        }
                    }
                    _ => {}
                }
            });
        }
        NodeData::MultipleCopy(mc) => {
            with_node_appearance(visitor, entry, |visitor| {
                let base = visitor.model_matrix();
                for &copy_mat in &mc.copies {
                    visitor.set_model_matrix(base * copy_mat);
                    for &child in &mc.children {
                        scene_traverse(visitor, graph, child);
                    }
                }
                visitor.set_model_matrix(base);
            });
        }
        NodeData::Lod(lod) => {
            with_node_appearance(visitor, entry, |visitor| {
                let level = lod.current_level.min(lod.levels.len().saturating_sub(1));
                if let Some(level_data) = lod.levels.get(level) {
                    for &child in &level_data.children {
                        scene_traverse(visitor, graph, child);
                    }
                }
            });
        }
        NodeData::HandlerNode(h) => {
            with_node_appearance(visitor, entry, |visitor| {
                h.traverse(graph, node, &entry.children, &mut |id| {
                    scene_traverse(visitor, graph, id);
                });
            });
        }
        NodeData::Transform(_) | NodeData::Rotation(_) | NodeData::RotationXYZ(_) => {
            with_node_appearance(visitor, entry, |visitor| {
                let current = visitor.model_matrix();
                if let Some(lm) = entry.data.local_matrix() {
                    visitor.set_model_matrix(current * lm);
                }
                for &child in &entry.children {
                    scene_traverse(visitor, graph, child);
                }
            });
        }
        NodeData::InstancedMesh(im) => {
            with_node_appearance(visitor, entry, |visitor| {
                let transforms: Vec<rc3d_core::math::Mat4> = im.transforms.iter()
                    .map(rc3d_core::math::Mat4::from_cols_array_2d)
                    .collect();
                visitor.set_instance_transforms(&transforms);
                for &child in &entry.children {
                    scene_traverse(visitor, graph, child);
                }
                visitor.set_instance_transforms(&[]);
            });
        }
        _ => {
            with_node_appearance(visitor, entry, |visitor| {
                let policy = visitor.visit_node(graph, node, entry);
                if policy == ChildPolicy::Recurse {
                    for &child in &entry.children {
                        scene_traverse(visitor, graph, child);
                    }
                }
            });
        }
    }
}
