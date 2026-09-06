//! Overlay-only frame: markup/HUD with no 3D geometry.
use super::pass_effects;
use super::{pass_hud, pass_markup, pass_shared, FramePresentation};
use crate::FrameStats;

/// Render a frame with only overlay elements (markup, HUD) — no 3D geometry.
/// Used when draw_calls is empty but markup vertices exist.
pub(crate) fn render_overlay_only_frame(
    renderer: &mut crate::renderer::Renderer,
    presentation: FramePresentation<'_>,
    post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    _frame_counter: u64,
    effect_commands: &pass_effects::EffectCommands,
) -> FrameStats {
    let t_entry = std::time::Instant::now();
    let (_tex_ptr, mut acquired_swapchain, eff_width, eff_height) = pass_shared::acquire_surface(
        renderer,
        &presentation,
        |_tex_ptr, w, h| (w, h),
    );

    let view: &wgpu::TextureView = match &acquired_swapchain {
        Some((_s, vw)) => vw,
        None => match &presentation {
            FramePresentation::OffscreenSurface { output_view, .. } => output_view,
            FramePresentation::Swapchain => return FrameStats::default(),
        },
    };

    let ew = eff_width.max(1);
    let eh = eff_height.max(1);
    renderer.pass_target_size = (ew, eh);

    let bg_color = wgpu::Color {
        r: 0.02,
        g: 0.02,
        b: 0.02,
        a: 1.0,
    };

    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Overlay-only Encoder"),
        });

    // Clear color to bg_color (no depth needed for overlays)
    {
        let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Overlay Clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(bg_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None, multiview_mask: None,
        });
    }

    renderer.encode_compositor(&mut encoder, view);

    // Markup overlay — skip occlusion when geometry pass is skipped (no depth data).
    let occlusion: Option<(Vec<f32>, u32, u32)> = None;
    let scene_vp = renderer.frame.scene_vp;
    let depth_reversed_z = renderer
        .frame
        .scene_depth_reversed_z;
    if renderer.gpu.depth_texture.is_none() {
        renderer.create_depth_texture();
    }
    let depth_view = renderer
        .gpu
        .depth_texture
        .as_ref()
        .map(|(_, v, _)| v.clone());

    // Static frame: reuse cached projection output.
    let is_static = renderer.frame.bvh_fully_static && renderer.frame.static_frame_count >= 2;
    if !is_static {
        let (projected, wl) = pass_markup::compute_projected_markup(
            renderer,
            effect_commands,
            scene_vp,
            ew as f32,
            eh as f32,
            depth_reversed_z,
            occlusion.as_ref().map(|(d, w, h)| (&d[..], *w, *h)),
        );
        renderer.frame.cached_projected_markup = projected;
        renderer.frame.cached_projected_labels = wl;
    }
    // Take cached fields to avoid borrow conflict with pass_markup's &mut renderer.
    let cache_markup = std::mem::take(&mut renderer.frame.cached_projected_markup);
    let cache_labels = std::mem::take(&mut renderer.frame.cached_projected_labels);
    pass_markup::pass_markup(
        renderer,
        &mut encoder,
        view,
        depth_view.as_ref().expect("depth texture"),
        ew,
        eh,
        scene_vp,
        depth_reversed_z,
        &cache_markup,
        &cache_labels,
    );
    renderer.frame.cached_projected_markup = cache_markup;
    renderer.frame.cached_projected_labels = cache_labels;
    renderer.gpu.flat_pool.flush(&renderer.queue);
    renderer.gpu.line_pool.flush(&renderer.queue);

    renderer.encode_overlay_composite(&mut encoder, view);
    if let Some(depth_view) = depth_view.as_ref() {
        pass_hud::encode_hud_overlay(
            renderer,
            &mut encoder,
            view,
            depth_view,
            depth_reversed_z,
        );
    }
    // Scene-only retain before egui (same contract as execute_passes).
    if let Some((ref surface_tex, _)) = acquired_swapchain {
        renderer.capture_ui_retain(&mut encoder, &surface_tex.texture);
    }
    if let Some(cb) = post_swapchain_overlay {
        cb(&mut encoder, view);
    }

    renderer
        .queue
        .submit(std::iter::once(encoder.finish()));

    if let Some((surface_tex, vw)) = acquired_swapchain.take() {
        drop(vw);
        renderer.queue.present(surface_tex);
    }

    let t_total = t_entry.elapsed().as_secs_f64() * 1000.0;
    FrameStats {
        frame_time_ms: t_total,
        cpu_sections: Vec::new(),
        gpu_sections: Vec::new(),
        ..FrameStats::default()
    }
}
