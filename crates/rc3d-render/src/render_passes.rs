use crate::adaptive_quality::AdaptiveQuality;
use crate::render_action::DrawCall;
use crate::vertex::CSM_CASCADE_COUNT;
use glam::Mat4;
use rc3d_core::DisplayMode;

mod pass_edge;
mod pass_hidden;
pub(crate) mod pass_effects;
mod pass_grid;
pub(crate) mod pass_markup;
mod meshlet_cull;
mod pass_post;
mod pass_shadow;
mod pass_solid;
mod pass_transparent;
pub(crate) mod pass_text;
pub(crate) mod pass_viewport;
mod ss_edge;
pub(crate) mod pass_wireframe;
mod pass_hud;
mod pass_shared;

pub(crate) mod draw_opaque;

mod film;
mod overlay;
mod overlay_only;
mod execute;

#[cfg(test)]
mod pass_markup_tests;
use draw_opaque::draw_opaque_triangle_batches;
pub(crate) use meshlet_cull::submit_meshlet_cull;

pub(crate) use execute::execute_passes;
pub(crate) use overlay_only::render_overlay_only_frame;

pub(crate) struct PassContext<'a> {
    pub visible: &'a [&'a DrawCall],
    pub solid_order: &'a [usize],
    pub edge_order: &'a [usize],
    pub selected_order: &'a [usize],
    pub transparent_order: &'a [usize],
    pub mesh_handles: &'a [Option<crate::gpu_resource::MeshId>],
    /// Global display mode after perf/adaptive remap. Per-draw style uses `DrawCall.display_mode`.
    #[allow(dead_code)]
    pub mode: DisplayMode,
    pub bg_color: wgpu::Color,
    pub performance_mode_active: bool,
    pub wireframe_supported: bool,
    pub adaptive_quality: AdaptiveQuality,
    pub outline_width: f32,
    pub outline_color: [f32; 4],
    pub meshlet_indices: &'a [usize],
    pub camera_pos: [f32; 3],
    pub depth_reversed_z: bool,
    /// CSM cascade view-projection matrices (one per cascade)
    pub csm_view_proj: [Mat4; CSM_CASCADE_COUNT],
    pub run_shadow_pass: bool,
    /// Camera projection matrix (for SSAO depth reconstruction)
    pub camera_proj: Mat4,
    /// Camera inverse projection matrix
    pub camera_inv_proj: Mat4,
    /// Combined view-projection matrix (proj * view).
    pub scene_vp: Mat4,
    /// Previous frame's view-projection matrix (for velocity buffer).
    pub prev_vp: Mat4,
    pub effect_commands: &'a pass_effects::EffectCommands,
    pub light_sets: &'a crate::light_set::LightSetTable,
}

/// Final color target for the frame (swapchain or an application-owned render target).
pub(crate) enum FramePresentation<'a> {
    Swapchain,
    /// Same render path as the swapchain (`hdr_off`, `ldr_fxaa_off` enforced by callers for correct resolve).
    OffscreenSurface {
        output_texture: &'a wgpu::Texture,
        output_view: &'a wgpu::TextureView,
        width_px: u32,
        height_px: u32,
    },
}
