//! Split collected draws so HOOPS-style face / edge tints render as extra ranges.

use std::sync::Arc;

use rc3d_core::math::Vec3;
use rc3d_core::{EdgeStyle, FillStyle};
use rc3d_scene::{EdgeTint, FaceMaterialGroup, FaceTint};

use crate::render_action::{DrawCall, RenderCollector};
use crate::vertex::Vertex;

impl RenderCollector {
    /// Split the last draw into index-buffer groups; tinted faces override `base_color`.
    pub(crate) fn apply_face_tints_to_last(
        &mut self,
        tints: &[FaceTint],
        face_of_tri: impl Fn(u32) -> u32,
    ) {
        if tints.is_empty() {
            return;
        }
        let Some(base) = self.draw_calls.last() else {
            return;
        };
        let Some(indices) = base.indices.clone() else {
            return;
        };
        let n_tris = (indices.len() / 3) as u32;
        if n_tris == 0 {
            return;
        }

        let mut slots = vec![0u32; n_tris as usize];
        for (i, tint) in tints.iter().enumerate() {
            let slot = (i + 1) as u32;
            for tri in 0..n_tris {
                if face_of_tri(tri) == tint.id {
                    slots[tri as usize] = slot;
                }
            }
        }
        if slots.iter().all(|&s| s == 0) {
            return;
        }

        let groups = FaceMaterialGroup::compact_from_triangle_slots(&slots);
        let base = self.draw_calls.pop().expect("last draw");
        let empty: Arc<Vec<[f32; 3]>> = Arc::new(Vec::new());
        for (i, group) in groups.iter().enumerate() {
            let mut dc = base.clone();
            if i > 0 {
                dc.edge_positions = empty.clone();
                dc.wireframe_edge_positions = empty.clone();
                dc.meshlet_data = None;
            }
            dc.index_first = group.start;
            dc.index_draw_count = group.count;
            if group.material_index > 0 {
                if let Some(tint) = tints.get(group.material_index as usize - 1) {
                    let rgb = Vec3::new(tint.color[0], tint.color[1], tint.color[2]);
                    dc.base_color = rgb;
                    dc.diffuse_color = rgb;
                    dc.emissive_color = rgb * 0.18;
                    dc.albedo_path = None;
                }
            }
            self.draw_calls.push(dc);
        }
    }

    /// Overlay colored line segments for [`EdgeTint`] (FillStyle::None + Hard edges).
    pub(crate) fn emit_edge_tint_overlays(
        &mut self,
        vertices: &Arc<Vec<Vertex>>,
        indices: Option<&Arc<Vec<u32>>>,
        tints: &[EdgeTint],
        model_aabb: Option<rc3d_core::Aabb>,
    ) {
        if tints.is_empty() {
            return;
        }
        let Some(idx) = indices else {
            return;
        };
        for tint in tints {
            let start = tint.triangle as usize * 3;
            if start + 2 >= idx.len() {
                continue;
            }
            let i0 = idx[start] as usize;
            let i1 = idx[start + 1] as usize;
            let i2 = idx[start + 2] as usize;
            let (a, b) = match tint.edge {
                0 => (i0, i1),
                1 => (i1, i2),
                _ => (i2, i0),
            };
            let Some(va) = vertices.get(a) else {
                continue;
            };
            let Some(vb) = vertices.get(b) else {
                continue;
            };
            let mut dc = DrawCall {
                vertices: vertices.clone(),
                indices: Some(idx.clone()),
                edge_positions: Arc::new(vec![va.position, vb.position]),
                fill_style: FillStyle::None,
                edge_style: EdgeStyle::Hard,
                overlay_color: Some(tint.color),
                selected: false,
                aabb: model_aabb.clone(),
                index_first: 0,
                index_draw_count: 0,
                ..DrawCall::default()
            };
            dc.mvp = self.state.projection_matrix()
                * self.state.view_matrix()
                * self.state.model_matrix();
            dc.model_matrix = self.state.model_matrix();
            dc.camera_pos = self.camera_pos;
            dc.display_mode = self.state.appearance().to_display_mode();
            dc.depth_reversed_z =
                rc3d_core::depth_reversed_z_from_projection(self.state.projection_matrix());
            dc.projection_orthographic = self.projection_orthographic;
            dc.node_type_label = Arc::from("EdgeTint");
            self.draw_calls.push(dc);
        }
    }

    pub(crate) fn apply_node_sub_entity(
        &mut self,
        graph: &rc3d_scene::SceneGraph,
        node: rc3d_core::NodeId,
        face_of_tri: impl Fn(u32) -> u32,
    ) {
        let Some(dc) = self.draw_calls.last() else {
            return;
        };
        let verts = dc.vertices.clone();
        let indices = dc.indices.clone();
        let aabb = dc.aabb.clone();
        self.apply_face_tints_to_last(graph.face_tints(node), face_of_tri);
        self.emit_edge_tint_overlays(&verts, indices.as_ref(), graph.edge_tints(node), aabb);
    }
}
