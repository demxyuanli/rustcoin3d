use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};
use crate::action::{Action, ActionKind};

/// Collects all enabled section planes from the scene graph.
///
/// ## Section Cap Rendering (planned)
///
/// When geometry is clipped, the interior appears hollow. To fill the cut surface:
///
/// 1. **Stencil pass**: Render back-facing triangles of the clipped mesh with stencil write.
///    The stencil buffer marks which pixels are inside the solid volume at the clip plane.
///
/// 2. **Cap fill pass**: Render a full-screen quad where the clip plane intersects geometry,
///    using the stencil buffer to fill only the marked pixels with the cap color.
///
/// 3. **Shader approach**: In the fragment shader, compute the clip-plane distance and
///    discard fragments on the clipped side. Use derivative operations to detect the
///    clip boundary and fill it.
///
/// 4. **Hatch pattern**: Cap surface is rendered with a screen-space hatch pattern
///    via a procedural shader (cross-hatching based on gl_FragCoord).
pub struct SectionPlaneAction {
    pub planes: Vec<[f32; 4]>,
}

impl SectionPlaneAction {
    pub fn new() -> Self {
        Self { planes: Vec::new() }
    }
}

impl Action for SectionPlaneAction {
    fn kind(&self) -> ActionKind { ActionKind::Search }

    fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        Self::collect(graph, root, &mut self.planes);
    }
}

impl SectionPlaneAction {
    fn collect(graph: &SceneGraph, node: NodeId, planes: &mut Vec<[f32; 4]>) {
        let Some(entry) = graph.get(node) else { return };
        if let NodeData::SectionPlane(sp) = &entry.data {
            if sp.enabled {
                planes.push(sp.plane);
            }
        }
        for &child in &entry.children {
            Self::collect(graph, child, planes);
        }
    }
}
