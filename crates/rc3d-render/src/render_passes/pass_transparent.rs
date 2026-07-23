use super::PassContext;
use crate::vertex::SceneUniforms;
use rc3d_scene::AlphaMode;

fn alpha_mode_to_f32(mode: AlphaMode) -> f32 {
    match mode {
        AlphaMode::Opaque => 0.0,
        AlphaMode::Mask => 1.0,
        AlphaMode::Blend => 2.0,
    }
}

/// Camera-space Z depth for sorting (larger = farther from camera).
fn camera_depth(dc: &crate::render_action::DrawCall) -> f32 {
    dc.mvp.z_axis.w
}

/// Emits draw calls for transparent objects into the given render pass.
/// Shared between basic alpha-blend and WBOIT modes.
fn emit_transparent_draws(
    renderer: &mut crate::renderer::Renderer,
    pass: &mut wgpu::RenderPass<'_>,
    ctx: &PassContext<'_>,
    sorted: &[usize],
) {
    let mut clip_arr = [[0.0f32; 4]; 6];
    for (i, cp) in renderer.frame.clip_planes.iter().enumerate() {
        if i < 6 {
            clip_arr[i] = *cp;
        }
    }
    let clip_count = [
        renderer.frame.clip_planes.len().min(6) as f32,
        0.0,
        0.0,
        0.0,
    ];

    let mut last_bound_mesh = None;

    for &i in sorted {
        let dc = ctx.visible[i];

        let uniforms = SceneUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            model: dc.model_matrix.to_cols_array_2d(),
            camera_pos: [dc.camera_pos.x, dc.camera_pos.y, dc.camera_pos.z, 1.0],
            diffuse_color: [
                dc.diffuse_color.x,
                dc.diffuse_color.y,
                dc.diffuse_color.z,
                1.0,
            ],
            ambient_color: [
                dc.ambient_color.x,
                dc.ambient_color.y,
                dc.ambient_color.z,
                1.0,
            ],
            specular_color: [
                dc.specular_color.x,
                dc.specular_color.y,
                dc.specular_color.z,
                1.0,
            ],
            shininess: [dc.shininess, 0.0, 0.0, 0.0],
            clip_planes: clip_arr,
            clip_count,
            pbr_base_color: [
                dc.base_color.x,
                dc.base_color.y,
                dc.base_color.z,
                dc.opacity,
            ],
            pbr_metallic_roughness: [dc.metallic, dc.roughness, dc.anisotropic, 0.0],
            pbr_emissive_alpha: [
                dc.emissive_color.x,
                dc.emissive_color.y,
                dc.emissive_color.z,
                dc.alpha_cutoff,
            ],
            pbr_alpha_flags: [
                alpha_mode_to_f32(dc.alpha_mode),
                dc.opacity,
                if dc.double_sided { 1.0 } else { 0.0 },
                0.0,
            ],
            pbr_clearcoat: [dc.clearcoat_factor, dc.clearcoat_roughness, 0.0, 0.0],
            pbr_sheen: [0.0, 0.0, 0.0, 0.0],
            pbr_specular: [dc.specular_color_factor.x, dc.specular_color_factor.y, dc.specular_color_factor.z, dc.specular_factor],
            light_set_index: [dc.light_set_id as f32, 0.0, 0.0, 0.0],
        };

        if let Some(offset) = renderer.gpu.phong_pool.push_scene(&uniforms) {
            let mat_key = {
                use std::hash::Hasher;
                let mut h = twox_hash::XxHash64::with_seed(0);
                h.write(dc.albedo_path.as_deref().unwrap_or("").as_bytes());
                h.write(dc.normal_path.as_deref().unwrap_or("").as_bytes());
                h.write(
                    dc.metallic_roughness_path
                        .as_deref()
                        .unwrap_or("")
                        .as_bytes(),
                );
                h.write(dc.emissive_path.as_deref().unwrap_or("").as_bytes());
                h.write(dc.occlusion_path.as_deref().unwrap_or("").as_bytes());
                h.write_u64(dc.base_color.x.to_bits() as u64);
                h.write_u64(dc.base_color.y.to_bits() as u64);
                h.write_u64(dc.base_color.z.to_bits() as u64);
                h.write_u64(dc.metallic.to_bits() as u64);
                h.write_u64(dc.roughness.to_bits() as u64);
                h.write_u64(dc.opacity.to_bits() as u64);
                h.write_u64(dc.alpha_cutoff.to_bits() as u64);
                h.write_u64(alpha_mode_to_f32(dc.alpha_mode).to_bits() as u64);
                h.finish()
            };

            if renderer.last_material_bg_key != mat_key || renderer.last_material_bg.is_none() {
                let bg = super::draw_opaque::albedo_material_bind_group(
                    &mut renderer.gpu.texture_cache,
                    &renderer.device,
                    &renderer.gpu.pipelines.pbr_material_bgl,
                    &renderer.queue,
                    dc,
                );
                renderer.last_material_bg_key = mat_key;
                renderer.last_material_bg = Some(bg.clone());
            }

            pass.set_bind_group(0, renderer.gpu.phong_pool.bind_group(), &[offset]);
            if let Some(ref mat_bg) = renderer.last_material_bg {
                pass.set_bind_group(1, mat_bg, &[]);
            }

            if let Some(mesh_id) = ctx.mesh_handles[i] {
                let inst_count = match &dc.instance_transforms {
                    Some(t) if !t.is_empty() => t.len() as u32,
                    _ => 1,
                };
                renderer.draw_mesh_instanced(pass, mesh_id, 0, inst_count, &mut last_bound_mesh);
            }
        }
    }
}

/// Renders transparent (alpha-blended) draw calls back-to-front (painter's algorithm).
///
/// Uses the `solid_alpha` pipeline (PBR with `ALPHA_BLENDING`, depth-write off,
/// double-sided). Draw calls are sorted far-to-near by camera distance.
pub(super) fn pass_transparent(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    if ctx.transparent_order.is_empty() {
        return;
    }

    // Sort transparent objects back-to-front by camera-space depth.
    let mut sorted: Vec<usize> = ctx.transparent_order.to_vec();
    sorted.sort_by(|&a, &b| {
        let da = camera_depth(ctx.visible[a]);
        let db = camera_depth(ctx.visible[b]);
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Transparent Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
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
        occlusion_query_set: None,
    });

    pass.set_pipeline(&scene_pl.solid_alpha);
    pass.set_stencil_reference(1);

    if let Some(csm) = &renderer.gpu.csm_shadow {
        pass.set_bind_group(2, &csm.bind_group, &[]);
    }
    pass.set_bind_group(3, &renderer.gpu.ibl_instance_bind_group, &[]);

    emit_transparent_draws(renderer, &mut pass, ctx, &sorted);
}

/// Renders transparent draw calls using WBOIT (Weighted Blended OIT).
///
/// Instead of painter's algorithm, this uses two MRT targets:
/// - accum (Rgba16Float, additive blend): accumulates weighted premultiplied color
/// - revealage (R8Unorm, multiplicative blend): tracks how much background shows through
///
/// After this pass, call `pass_wboit_composite` to blend the result onto the scene.
pub(super) fn pass_transparent_wboit(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    accum_view: &wgpu::TextureView,
    revealage_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    if ctx.transparent_order.is_empty() {
        return;
    }

    // Clear accum to black (0,0,0,0) and revealage to white (1,1,1,1)
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("WBOIT Accumulate"),
        color_attachments: &[
            Some(wgpu::RenderPassColorAttachment {
                view: accum_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: revealage_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_pipeline(&scene_pl.wboit_accum);
    pass.set_stencil_reference(1);

    if let Some(csm) = &renderer.gpu.csm_shadow {
        pass.set_bind_group(2, &csm.bind_group, &[]);
    }
    pass.set_bind_group(3, &renderer.gpu.ibl_instance_bind_group, &[]);

    // WBOIT is order-independent, so no sorting needed
    let order: Vec<usize> = ctx.transparent_order.to_vec();
    emit_transparent_draws(renderer, &mut pass, ctx, &order);
}

/// Composites WBOIT accum/revealage buffers onto the scene using
/// hardware premultiplied-alpha blending (One, OneMinusSrcAlpha).
/// No scene texture read-back needed — avoids read-write hazard.
pub(super) fn pass_wboit_composite(
    renderer: &crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    target_view: &wgpu::TextureView,
    accum_view: &wgpu::TextureView,
    revealage_view: &wgpu::TextureView,
) {
    let _fx = match &renderer.gpu.post_fx {
        Some(fx) => fx,
        None => return,
    };
    let post_pl = &renderer.gpu.post_fx_pipelines;

    // Bind group: accum + revealage textures + sampler (shader group 1)
    let accum_bg = renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("WBOIT Accum BG"),
            layout: &post_pl.wboit_accum_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(revealage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&post_pl.tonemap_sampler),
                },
            ],
        });

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("WBOIT Composite"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_pipeline(&post_pl.wboit_composite_pipeline);
    pass.set_bind_group(0, &accum_bg, &[]);
    pass.draw(0..3, 0..1);
}
