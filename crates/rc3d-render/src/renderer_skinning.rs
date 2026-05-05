use crate::gpu_resource::MeshId;
use crate::render_action::DrawCall;

impl super::Renderer {
    pub(crate) fn encode_skinning_compute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        visible: &[&DrawCall],
        mesh_handles: &[Option<MeshId>],
    ) {
        if self.gpu.skinned_mesh_resources.is_empty() {
            return;
        }
        let Some(pass) = self.gpu.gpu_skinning_pass.as_ref() else {
            return;
        };
        for (i, dc) in visible.iter().enumerate() {
            let Some(skin) = dc.skinning.as_ref() else {
                continue;
            };
            let Some(mesh_id) = mesh_handles.get(i).copied().flatten() else {
                continue;
            };
            let Some(res) = self.gpu.skinned_mesh_resources.get(&mesh_id) else {
                continue;
            };
            let jc = skin.skeleton.joint_count();
            if jc == 0 {
                continue;
            }
            let local: Vec<glam::Mat4> = if let Some(ref clip) = skin.clip {
                clip.sample_all(self.frame.animation_time_sec, &skin.skeleton)
            } else {
                skin.skeleton.joints.iter().map(|j| j.bind_transform).collect()
            };
            let mats = skin.skeleton.skinning_matrices(&local);
            pass.skin_mesh(&self.queue, encoder, res, &mats);
        }
    }
}
