//! Frame orchestration: acquire surface, film, overlays, submit.
use crate::render_action::DrawCall;
use super::film::{encode_film, FilmEncodeResult};
use super::overlay::encode_swapchain_overlays;
use super::{pass_shared, FramePresentation, PassContext};
use crate::FrameStats;

pub(crate) fn execute_passes(
    renderer: &mut crate::renderer::Renderer,
    ctx: &PassContext<'_>,
    draw_calls: &[DrawCall],
    _frame_counter: u64,
    post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    presentation: FramePresentation<'_>,
) -> FrameStats {
    let t_entry = std::time::Instant::now();

    let t_surface_start = std::time::Instant::now();
    // Texture pointer is unused: outline composites onto shade_view; FXAA/HDR use shade_view/view.
    let (_, mut acquired_swapchain, eff_width, eff_height) = pass_shared::acquire_surface(
        renderer,
        &presentation,
        |_, _, _| (),
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
    let pass_vp = renderer.current_pass_viewport();

    if renderer.gpu.depth_texture.is_none() {
        renderer.create_depth_texture();
    }
    let Some((_, depth_view, depth_read_view)) = renderer.gpu.depth_texture.as_ref() else {
        return FrameStats::default();
    };
    let depth_view = depth_view.clone();
    let depth_read_view = depth_read_view.clone();

    if renderer.hdr_post_processing {
        renderer.ensure_post_fx_targets();
    }
    let on_swapchain = matches!(presentation, FramePresentation::Swapchain);
    let use_ldr_fxaa = !renderer.hdr_post_processing && renderer.enable_ldr_fxaa;
    let need_comp_film = on_swapchain && renderer.compositor_needs_ldr_film();
    let use_ldr_film = !renderer.hdr_post_processing && on_swapchain && (use_ldr_fxaa || need_comp_film);
    if use_ldr_film {
        renderer.ensure_ldr_shade_target();
    }
    let scene_pl = renderer
        .gpu.pipelines
        .for_shaded_target(ctx.depth_reversed_z, renderer.hdr_post_processing)
        .clone();
    let shade_color_view = if renderer.hdr_post_processing {
        if let Some(fx) = renderer.gpu.post_fx.as_ref() {
            fx.hdr_view.clone()
        } else {
            log::warn!("hdr_post_processing is enabled but post_fx is missing; falling back to swapchain view");
            view.clone()
        }
    } else if use_ldr_film {
        renderer
            .gpu.ldr_shade_view
            .as_ref()
            .expect("LDR shade film")
            .clone()
    } else {
        view.clone()
    };
    let shade_view: &wgpu::TextureView = &shade_color_view;

    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });


    let film = encode_film(
        renderer,
        ctx,
        &mut encoder,
        view,
        shade_view,
        &depth_view,
        &depth_read_view,
        ew,
        eh,
        pass_vp,
        scene_pl,
        use_ldr_fxaa,
        use_ldr_film,
        need_comp_film,
    );
    let FilmEncodeResult {
        scene_pl,
        defer_line_overlays,
        run_hidden,
        run_wireframe,
        edge_worthy,
        has_overlay,
    } = film;
    let retain_src = acquired_swapchain
        .as_ref()
        .map(|(surface_tex, _)| &surface_tex.texture);
    let occlusion_captured = encode_swapchain_overlays(
        renderer,
        ctx,
        &mut encoder,
        view,
        &depth_view,
        &depth_read_view,
        ew,
        eh,
        &scene_pl,
        defer_line_overlays,
        run_hidden,
        run_wireframe,
        edge_worthy,
        has_overlay,
        post_swapchain_overlay,
        retain_src,
    );
    renderer.prune_mesh_cache();
    renderer.gpu_timer.resolve(&mut encoder);
    let t0 = std::time::Instant::now();
    renderer.queue.submit(std::iter::once(encoder.finish()));
    let t_submit = t0.elapsed().as_secs_f64() * 1000.0;

    // Kick off the async occlusion-depth readback (harvested non-blockingly
    // at the start of a later frame's markup pass — no GPU stall here).
    if occlusion_captured {
        if let Some(ref buf) = renderer.frame.occlusion_capture_buf {
            let status = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
            let status_cb = std::sync::Arc::clone(&status);
            buf.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                let v = if result.is_ok() { 1 } else { 2 };
                status_cb.store(v, std::sync::atomic::Ordering::Release);
            });
            renderer.frame.occlusion_map_pending = Some(status);
            renderer.device.poll(wgpu::PollType::Poll).unwrap();
        }
    }

    if let Some((surface_tex, vw)) = acquired_swapchain.take() {
        drop(vw);
        renderer.queue.present(surface_tex);
    }
    let t_present = t0.elapsed().as_secs_f64() * 1000.0 - t_submit;

    let t_collect = std::time::Instant::now();
    renderer.gpu_timer.collect(&renderer.device);
    let t_collect = t_collect.elapsed().as_secs_f64() * 1000.0;

    if renderer.frame.frame_counter % 10 == 0 {
        log::debug!(
            "CPU submit/present/collect: submit={:.1}ms present={:.1}ms collect={:.1}ms",
            t_submit, t_present, t_collect
        );
    }
    let gpu_timestamps = &renderer.gpu_timer.last_timestamps;

    let mut gpu_sections_data = Vec::new();
    for i in (0..gpu_timestamps.len()).step_by(2) {
        if i + 1 < gpu_timestamps.len() {
            let label = renderer.gpu_timer.labels.get(i / 2).copied().unwrap_or("?");
            let dur_ticks = gpu_timestamps[i + 1].saturating_sub(gpu_timestamps[i]);
            let dur_us = dur_ticks as f64 * renderer.gpu_timer.timestamp_period_ns as f64 / 1_000.0;
            gpu_sections_data.push((label, dur_us));
        }
    }

    let stats = build_frame_stats(
        ctx.visible,
        draw_calls.len(),
        gpu_sections_data,
        renderer.cpu_span.total_ms(),
        renderer.cpu_span.spans().to_vec(),
    );

    log_frame_stats(&stats, renderer.frame.frame_counter);

    let t_total = t_entry.elapsed().as_secs_f64() * 1000.0;
    let t_surface = t_surface_start.elapsed().as_secs_f64() * 1000.0;
    if renderer.frame.frame_counter % 10 == 0 {
        log::debug!(
            "execute_passes wall={:.1}ms (surface_acquire={:.1}ms) draws={}",
            t_total, t_surface, draw_calls.len()
        );
    }

    stats

}

/// Build frame statistics from rendering results.
fn build_frame_stats(
    visible: &[&DrawCall],
    total_draw_calls: usize,
    gpu_sections: Vec<(&'static str, f64)>,
    frame_time_ms: f64,
    cpu_sections: Vec<(&'static str, f64)>,
) -> FrameStats {
    let total_visible_triangles: u64 = visible
        .iter()
        .map(|dc| {
            if let Some(md) = dc.meshlet_data.as_ref() {
                md.total_triangles as u64
            } else if let Some(indices) = dc.indices.as_ref() {
                (indices.len() / 3) as u64
            } else {
                (dc.vertices.len() / 3) as u64
            }
        })
        .sum();
    FrameStats {
        visible_triangles: total_visible_triangles,
        visible_draw_calls: visible.len(),
        culled_draw_calls: total_draw_calls.saturating_sub(visible.len()),
        gpu_pass_times_us: None,
        diagnostics: None,
        frame_time_ms,
        cpu_sections,
        gpu_sections,
    }
}

/// Periodic per-pass GPU/CPU timing report (every 10 frames).
fn log_frame_stats(stats: &FrameStats, frame_counter: u64) {
    if frame_counter % 10 != 0 || stats.gpu_sections.is_empty() {
        return;
    }
    let parts: Vec<String> = stats.gpu_sections.iter()
        .map(|(label, us)| format!("{}={:.0}us", label, us))
        .collect();
    let cpu_parts: Vec<String> = stats.cpu_sections.iter()
        .map(|(label, ms)| format!("{}={:.2}ms", label, ms))
        .collect();
    log::debug!(
        "GPU: {} | CPU: {} | frame={:.2}ms draws={} tris={}",
        parts.join(" "),
        cpu_parts.join(" "),
        stats.frame_time_ms,
        stats.visible_draw_calls,
        stats.visible_triangles,
    );
}
