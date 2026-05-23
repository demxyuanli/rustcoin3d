//! Convert PMI data to scene graph markup nodes.
//! Reuses existing Markup/Annotation rendering pipeline (future integration).

use super::pmi_extract::PmiData;

/// Attach PMI annotations to the scene graph.
/// MVP stub — no-op until AnnotationSet/AnnotationElement node types are available.
pub fn attach_pmi_to_scene(
    _graph: &mut rc3d_scene::SceneGraph,
    _parent: rc3d_core::NodeId,
    _pmi: &PmiData,
) {
    // MVP: no-op. Will create AnnotationSet + AnnotationElement nodes in a follow-up.
}
