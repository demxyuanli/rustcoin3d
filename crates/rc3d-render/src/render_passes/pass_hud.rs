//! HUD and annotation overlay rendering.
//!
//! Encodes the HUD chrome and plane annotation glyphs onto the current
//! swapchain / offscreen target. Called from both the full render path
//! (`execute_passes`) and the overlay-only path (`render_overlay_only_frame`).

/// Encode HUD overlay passes (annotation text + HUD chrome).
///
/// `depth_view` is required only when the HUD has plane annotation glyphs;
/// it may be `None` otherwise (the depth-stencil attachment is conditionally added).
pub(crate) fn encode_hud_overlay(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    depth_reversed_z: bool,
) {
    if !renderer.hud_enabled {
        return;
    }
    renderer.prepare_hud_overlay_for_render();
    let film = renderer.current_pass_viewport();
    let Some(hud) = renderer.gpu.hud.as_ref() else {
        return;
    };

    // Annotation text pass (needs depth for correct occlusion)
    if hud.has_plane_annotation_glyphs() {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Annotation Text Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None, multiview_mask: None,
        });
        renderer.apply_scene_viewport(&mut pass);
        hud.render_plane_annotations(&mut pass, depth_reversed_z);
    }

    // HUD chrome (no depth). Keep the full-target viewport so glyphon pixel
    // coords match the swapchain; scissor to the 3D film so egui chrome cannot
    // cover the stats, and draw after overlay tiles so the nav cube cannot either.
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("HUD Overlay Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None, multiview_mask: None,
    });
    pass.set_scissor_rect(film.x, film.y, film.width.max(1), film.height.max(1));
    hud.render_hud_chrome(&mut pass);
}
