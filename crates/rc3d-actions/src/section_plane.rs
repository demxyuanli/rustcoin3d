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
    /// Cap colors for planes with cap_enabled (same order as planes).
    pub cap_colors: Vec<[f32; 4]>,
    pub has_caps: bool,
}

impl SectionPlaneAction {
    pub fn new() -> Self {
        Self { planes: Vec::new(), cap_colors: Vec::new(), has_caps: false }
    }
}

impl Action for SectionPlaneAction {
    fn kind(&self) -> ActionKind { ActionKind::Search }

    fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        self.planes.clear();
        self.cap_colors.clear();
        self.has_caps = false;
        Self::collect(graph, root, &mut self.planes, &mut self.cap_colors, &mut self.has_caps);
    }
}

impl SectionPlaneAction {
    fn collect(graph: &SceneGraph, node: NodeId, planes: &mut Vec<[f32; 4]>, caps: &mut Vec<[f32; 4]>, has_caps: &mut bool) {
        let Some(entry) = graph.get(node) else { return };
        if let NodeData::SectionPlane(sp) = &entry.data {
            if sp.enabled {
                planes.push(sp.plane);
                caps.push(sp.cap_color);
                *has_caps = *has_caps || sp.cap_enabled;
            }
        }
        for &child in &entry.children {
            Self::collect(graph, child, planes, caps, has_caps);
        }
    }
}
