use crate::render_action::DrawCall;
use crate::vertex::{
    FlatUniforms, InstanceData, SceneUniforms, CSM_CASCADE_COUNT, MAX_MORPH_WEIGHTS,
};
use rc3d_scene::AlphaMode;
use slotmap::Key;

fn alpha_mode_to_f32(mode: AlphaMode) -> f32 {
    match mode {
        AlphaMode::Opaque => 0.0,
        AlphaMode::Mask => 1.0,
        AlphaMode::Blend => 2.0,
    }
}

fn pack_morph_weights(weights: &[f32]) -> [f32; MAX_MORPH_WEIGHTS] {
    let mut packed = [0.0f32; MAX_MORPH_WEIGHTS];
    let count = weights.len().min(MAX_MORPH_WEIGHTS);
    packed[..count].copy_from_slice(&weights[..count]);
    packed
}

pub(super) fn instance_data_from_draw(dc: &DrawCall) -> InstanceData {
    let diffuse = if dc.fill_style == rc3d_core::FillStyle::HiddenLine {
        [0.08, 0.08, 0.08, 1.0]
    } else {
        [
            dc.diffuse_color.x,
            dc.diffuse_color.y,
            dc.diffuse_color.z,
            1.0,
        ]
    };
    InstanceData {
        model: dc.model_matrix.to_cols_array_2d(),
        mvp: dc.mvp.to_cols_array_2d(),
        diffuse_color: diffuse,
        base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, 1.0],
        metallic_roughness: [dc.metallic, dc.roughness, 0.0, 0.0],
        emissive_alpha: [
            dc.emissive_color.x,
            dc.emissive_color.y,
            dc.emissive_color.z,
            dc.alpha_cutoff,
        ],
        morph_weights: pack_morph_weights(&dc.morph_weights),
        morph_count: [
            dc.morph_weights.len().min(MAX_MORPH_WEIGHTS) as f32,
            0.0,
            0.0,
            0.0,
        ],
    }
}

/// Fast material identity hash from texture paths + PBR parameters.
/// Pure-color materials with different base_color/metallic/roughness
/// produce different keys, avoiding stale bind group cache hits.
fn material_key(dc: &DrawCall) -> u64 {
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
}

pub(super) fn albedo_material_bind_group<'a>(
    texture_cache: &'a mut crate::texture_cache::TextureCache,
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    queue: &wgpu::Queue,
    dc: &DrawCall,
) -> &'a wgpu::BindGroup {
    let albedo_handle = match &dc.albedo_path {
        None => texture_cache.white_handle(),
        Some(p) => texture_cache.load_path(device, queue, p.as_ref()),
    };
    let normal_handle = match &dc.normal_path {
        None => texture_cache.default_normal_handle(),
        Some(p) => texture_cache.load_path(device, queue, p.as_ref()),
    };
    let mr_handle = match &dc.metallic_roughness_path {
        None => texture_cache.white_handle(),
        Some(p) => texture_cache.load_path(device, queue, p.as_ref()),
    };
    let emissive_handle = match &dc.emissive_path {
        None => texture_cache.white_handle(),
        Some(p) => texture_cache.load_path(device, queue, p.as_ref()),
    };
    let occlusion_handle = match &dc.occlusion_path {
        None => texture_cache.white_handle(),
        Some(p) => texture_cache.load_path(device, queue, p.as_ref()),
    };
    texture_cache.pbr_material_bind_group(
        device,
        layout,
        albedo_handle,
        normal_handle,
        mr_handle,
        emissive_handle,
        occlusion_handle,
    )
}

fn draw_flat_triangle_batches(
    renderer: &mut crate::renderer::Renderer,
    pass: &mut wgpu::RenderPass<'_>,
    ctx: &super::PassContext<'_>,
    flat_solid_pipeline: &wgpu::RenderPipeline,
) {
    pass.set_pipeline(flat_solid_pipeline);

    let mut last_bound_mesh = None;
    for &i in ctx.solid_order {
        let dc = ctx.visible[i];
        if !dc.appearance().is_flat_fill() {
            continue;
        }
        let face_color = if dc.fill_style == rc3d_core::FillStyle::HiddenLine {
            renderer.hidden_line_fill_color
        } else {
            renderer.flat_face_color.unwrap_or([
                dc.diffuse_color.x,
                dc.diffuse_color.y,
                dc.diffuse_color.z,
                1.0,
            ])
        };
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: face_color,
            model: dc.model_matrix.to_cols_array_2d(),
            clip_planes: [[0.0; 4]; 6],
            clip_count: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
            if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_mesh_instanced(
                    pass,
                    mesh_id,
                    0,
                    1,
                    dc.index_draw_range(),
                    &mut last_bound_mesh,
                );
            }
        }
    }
}

/// Records opaque triangle draws. May be called twice per frame on the same
/// encoder (depth prepass + solid pass). Both calls MUST produce the identical
/// instance/indirect buffer layout: `queue.write_buffer` calls are applied
/// before `queue.submit`, so the GPU executes BOTH passes against the LAST
/// written contents. `upload_buffers=false` skips the redundant second upload
/// when the prepass already wrote identical data this frame.
pub(super) fn draw_opaque_triangle_batches(
    renderer: &mut crate::renderer::Renderer,
    pass: &mut wgpu::RenderPass<'_>,
    ctx: &super::PassContext<'_>,
    solid_pipeline: &wgpu::RenderPipeline,
    flat_solid_pipeline: &wgpu::RenderPipeline,
    upload_buffers: bool,
) {
    renderer.apply_scene_viewport(pass);
    let any_flat = ctx
        .solid_order
        .iter()
        .any(|&i| ctx.visible[i].appearance().is_flat_fill());
    let any_lit = ctx
        .solid_order
        .iter()
        .any(|&i| ctx.visible[i].appearance().wants_lit_solid());
    if any_flat && !any_lit {
        draw_flat_triangle_batches(renderer, pass, ctx, flat_solid_pipeline);
        return;
    }

    // Pre-load textures for all visible draws (avoids synchronous I/O in draw loop).
    for dc in ctx.visible {
        if let Some(ref p) = dc.albedo_path {
            renderer
                .gpu
                .texture_cache
                .load_path(&renderer.device, &renderer.queue, p);
        }
        if let Some(ref p) = dc.normal_path {
            renderer
                .gpu
                .texture_cache
                .load_path(&renderer.device, &renderer.queue, p);
        }
        if let Some(ref p) = dc.metallic_roughness_path {
            renderer
                .gpu
                .texture_cache
                .load_path(&renderer.device, &renderer.queue, p);
        }
        if let Some(ref p) = dc.emissive_path {
            renderer
                .gpu
                .texture_cache
                .load_path(&renderer.device, &renderer.queue, p);
        }
        if let Some(ref p) = dc.occlusion_path {
            renderer
                .gpu
                .texture_cache
                .load_path(&renderer.device, &renderer.queue, p);
        }
    }

    pass.set_pipeline(solid_pipeline);
    pass.set_stencil_reference(1);

    // Set static bind groups once for the entire pass (same for all draws)
    match &renderer.gpu.csm_shadow {
        Some(csm) => pass.set_bind_group(2, &csm.bind_group, &[]),
        None => log::error!("CSM shadow missing; shadow bind group not set"),
    }
    pass.set_bind_group(3, &renderer.gpu.ibl_instance_bind_group, &[]);

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

    {
        let mb = &mut renderer.gpu.draw_bufs.meshlet_bitmask;
        mb.resize(ctx.visible.len(), false);
        mb.fill(false);
        for &i in ctx.meshlet_indices {
            mb[i] = true;
        }
    }
    {
        let mk = &mut renderer.gpu.draw_bufs.mat_keys;
        mk.clear();
        mk.extend(ctx.visible.iter().map(|dc| material_key(dc)));
    }

    let mut last_bound_mesh = None;
    let instance_stride = std::mem::size_of::<InstanceData>() as u64;

    // Phase 1: collect instance data for all subgroups into a single buffer.
    // Phase 2: issue draw calls (avoiding many small write_buffer calls).
    let mut draw_batches: Vec<(u32, u32, Option<crate::gpu_resource::MeshId>, u64, usize)> =
        Vec::new();
    {
        let all_instances = &mut renderer.gpu.draw_bufs.instances;
        all_instances.clear();

        // Pre-insert meshlet InstanceData at slot 0 so the Phase 2 batch write
        // puts it there. The meshlet draw reads instances[0] via the indirect
        // buffer's first_instance=0. Phase 2 write goes to queue before encoder
        // submit, so the meshlet data is at slot 0 when the render pass starts.
        // NOTE: this condition must not depend on which pass we record for, so
        // prepass and solid produce identical instance layouts (see fn doc).
        let mut instance_cursor: u64 = 0;
        if !ctx.meshlet_indices.is_empty() {
            let dc = ctx.visible[ctx.meshlet_indices[0]];
            all_instances.push(instance_data_from_draw(dc));
            instance_cursor = instance_stride; // standard draws start at slot 1
        }
        let mut start = 0usize;
        while start < ctx.solid_order.len() {
            let head_idx = ctx.solid_order[start];
            let head_dc = ctx.visible[head_idx];
            let light_key = head_dc.light_key;
            let mut end = start + 1;
            while end < ctx.solid_order.len() {
                let idx = ctx.solid_order[end];
                if ctx.visible[idx].light_key != light_key {
                    break;
                }
                end += 1;
            }

            {
                let md = &mut renderer.gpu.draw_bufs.meshlet_draws;
                let sd = &mut renderer.gpu.draw_bufs.standard_draws;
                md.clear();
                sd.clear();
                // Meshlet draw path requires HZB (even for frustum-only cull, the
                // bind group layout binds HZB texture views). Without HZB, fall
                // back to standard instanced draws for meshlet-indexed geometry.
                // Must be pass-independent so prepass/solid layouts match.
                let can_meshlet_draw = renderer.gpu.hzb.is_some();
                for &i in &ctx.solid_order[start..end] {
                    if !ctx.visible[i].appearance().wants_lit_solid() {
                        continue;
                    }
                    if renderer.gpu.draw_bufs.meshlet_bitmask[i] {
                        if can_meshlet_draw {
                            md.push(i);
                        } else {
                            sd.push(i);
                        }
                    } else {
                        sd.push(i);
                    }
                }
            }

            // Meshlet path: per-draw (no instance batching needed).
            // Also recorded in the depth prepass: draw_clustered{,_basic} use
            // the currently bound pipeline, so the prepass renders meshlet
            // depth with its own prepass pipeline (keeps HZB occluder coverage).
            for &i in renderer.gpu.draw_bufs.meshlet_draws.iter() {
                let dc = ctx.visible[i];
                let md = match dc.meshlet_data.as_ref() {
                    Some(md) => md,
                    None => continue,
                };
                let ptr = std::sync::Arc::as_ptr(md) as u64;
                if !(renderer.gpu.cluster_renderer.is_some()
                    && renderer.gpu.assets.cluster_contains(&ptr))
                {
                    continue;
                }
                let diffuse_color = if dc.fill_style == rc3d_core::FillStyle::HiddenLine {
                    [0.08, 0.08, 0.08, 1.0]
                } else {
                    [
                        dc.diffuse_color.x,
                        dc.diffuse_color.y,
                        dc.diffuse_color.z,
                        1.0,
                    ]
                };
                let uniforms = SceneUniforms {
                    mvp: dc.mvp.to_cols_array_2d(),
                    model: dc.model_matrix.to_cols_array_2d(),
                    camera_pos: [dc.camera_pos.x, dc.camera_pos.y, dc.camera_pos.z, 1.0],
                    diffuse_color,
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
                    pbr_base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, 1.0],
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
                        dc.shade_mode_w(),
                    ],
                    pbr_clearcoat: [dc.clearcoat_factor, dc.clearcoat_roughness, dc.iridescence_factor, dc.iridescence_ior],
                    pbr_sheen: [dc.sheen_color.x, dc.sheen_color.y, dc.sheen_color.z, dc.sheen_roughness],
                    pbr_specular: [dc.specular_color_factor.x, dc.specular_color_factor.y, dc.specular_color_factor.z, dc.specular_factor],
                    pbr_transmission: [dc.transmission_factor, dc.ior, dc.iridescence_thickness_min, dc.iridescence_thickness_max],
                    light_set_index: [head_dc.light_set_id as f32, 0.0, 0.0, 0.0],
                };
                if let Some(offset) = renderer.gpu.phong_pool.push_scene(&uniforms) {
                    let key = renderer.gpu.draw_bufs.mat_keys[i];
                    if renderer.last_material_bg_key != key || renderer.last_material_bg.is_none() {
                        let bg = albedo_material_bind_group(
                            &mut renderer.gpu.texture_cache,
                            &renderer.device,
                            &renderer.gpu.pipelines.pbr_material_bgl,
                            &renderer.queue,
                            dc,
                        );
                        renderer.last_material_bg_key = key;
                        renderer.last_material_bg = Some(bg.clone());
                    }
                    pass.set_bind_group(0, renderer.gpu.phong_pool.bind_group(), &[offset]);
                    if let Some(ref mat_bg) = renderer.last_material_bg {
                        pass.set_bind_group(1, mat_bg, &[]);
                    }
                    // InstanceData at slot 0 written by pre-insert at Phase 1 start.
                    // Meshlet draw reads instances[0] via indirect buffer first_instance=0.
                    if let Some(cluster_set) = renderer.gpu.assets.cluster_get(&ptr) {
                        if cluster_set.total_triangles > 0 {
                            if let Some(cluster_renderer) = renderer.gpu.cluster_renderer.as_ref() {
                                if renderer.gpu.gpu_capability.meshlet_gpu_cull_enabled {
                                    cluster_renderer.draw_clustered(pass, cluster_set);
                                } else {
                                    cluster_renderer.draw_clustered_basic(pass, cluster_set);
                                }
                                last_bound_mesh = None;
                            }
                        }
                    }
                }
            }

            // Standard path: accumulate instance data for batched write
            if !renderer.gpu.draw_bufs.standard_draws.is_empty() {
                let sd = &mut renderer.gpu.draw_bufs.standard_draws;
                sd.sort_by_key(|&i| {
                    let dc = ctx.visible[i];
                    (
                        ctx.mesh_handles[i].map(|m| m.data().as_ffi()),
                        renderer.gpu.draw_bufs.mat_keys[i],
                        dc.index_first,
                        dc.index_draw_count,
                    )
                });
                let mut sub_start = 0usize;
                while sub_start < sd.len() {
                    let first_mesh = ctx.mesh_handles[sd[sub_start]];
                    let first_mat = renderer.gpu.draw_bufs.mat_keys[sd[sub_start]];
                    let first_range = {
                        let dc = ctx.visible[sd[sub_start]];
                        (dc.index_first, dc.index_draw_count)
                    };
                    let mut sub_end = sub_start + 1;
                    while sub_end < sd.len() {
                        let idx = sd[sub_end];
                        let dc = ctx.visible[idx];
                        if ctx.mesh_handles[idx] != first_mesh
                            || renderer.gpu.draw_bufs.mat_keys[idx] != first_mat
                            || dc.index_first != first_range.0
                            || dc.index_draw_count != first_range.1
                        {
                            break;
                        }
                        sub_end += 1;
                    }
                    let subgroup = &sd[sub_start..sub_end];
                    sub_start = sub_end;

                    if let Some(_mesh_id) = first_mesh {
                        if !subgroup.is_empty() {
                            let first_instance = (instance_cursor / instance_stride) as u32;
                            for &i in subgroup {
                                let dc = ctx.visible[i];
                                all_instances.push(instance_data_from_draw(dc));
                                instance_cursor += instance_stride;
                            }
                            let n = subgroup.len() as u32;
                            let rep_dc_idx = subgroup[0];
                            draw_batches.push((
                                first_instance,
                                n,
                                first_mesh,
                                renderer.gpu.draw_bufs.mat_keys[rep_dc_idx],
                                rep_dc_idx,
                            ));
                        }
                    }
                }
            }
            start = end;
        }

        // Phase 2: write all instance data (meshlet at slot 0 + standard at slots 1+).
        // queue.write_buffer goes to queue before encoder submit, so slot 0
        // has meshlet data when the render pass starts.
        if upload_buffers && !renderer.gpu.draw_bufs.instances.is_empty() {
            renderer.queue.write_buffer(
                &renderer.gpu.instance_buffer,
                0,
                bytemuck::cast_slice(&renderer.gpu.draw_bufs.instances),
            );
        }

        if renderer.gpu.multi_draw_indirect_supported {
            // Multi-draw indirect path: build DrawIndexedIndirectArgs args, write to GPU buffer,
            // then batch by (mesh, mat_key, light_set_id) to minimise state changes.
            // Collect mesh index_counts before mutable borrow of standard_indirect_args.
            let mesh_index_counts: Vec<(u32, u32)> = draw_batches
                .iter()
                .map(|&(_, _n, mesh_id, _, rep_dc_idx)| {
                    let dc = ctx.visible[rep_dc_idx];
                    match mesh_id.and_then(|m| renderer.get_mesh(m)) {
                        Some(mesh) if mesh.index_buffer.is_some() => {
                            dc.resolved_index_range(mesh.index_count)
                        }
                        _ => (0, 0),
                    }
                })
                .collect();

            let indirect_args = &mut renderer.gpu.draw_bufs.standard_indirect_args;
            indirect_args.clear();

            for (bi, &(first_instance, n, _mesh_id, _first_mat_key, _rep_dc_idx)) in
                draw_batches.iter().enumerate()
            {
                // Build 1:1 with draw_batches so offsets align in the batching loop.
                // Zero-instance draws are skipped by the GPU and only exist for alignment.
                let (first_index, ic) = mesh_index_counts[bi];
                let icount = if ic > 0 { n } else { 0 };
                indirect_args.push(wgpu::util::DrawIndexedIndirectArgs {
                    index_count: ic,
                    instance_count: icount,
                    first_index,
                    base_vertex: 0,
                    first_instance,
                });
            }

            if !indirect_args.is_empty() {
                let indirect_buf = renderer
                    .gpu
                    .draw_bufs
                    .standard_indirect_buf
                    .as_ref()
                    .unwrap();
                let arg_stride = std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>();
                if upload_buffers {
                    let mut indirect_bytes: Vec<u8> =
                        Vec::with_capacity(indirect_args.len() * arg_stride);
                    for arg in indirect_args.iter() {
                        indirect_bytes.extend_from_slice(bytemuck::bytes_of(&arg.index_count));
                        indirect_bytes.extend_from_slice(bytemuck::bytes_of(&arg.instance_count));
                        indirect_bytes.extend_from_slice(bytemuck::bytes_of(&arg.first_index));
                        indirect_bytes.extend_from_slice(bytemuck::bytes_of(&arg.base_vertex));
                        indirect_bytes.extend_from_slice(bytemuck::bytes_of(&arg.first_instance));
                    }
                    renderer
                        .queue
                        .write_buffer(indirect_buf, 0, &indirect_bytes);
                }
                let mut batch_start = 0u32;
                let mut current_mesh: Option<crate::gpu_resource::MeshId> = None;
                let mut current_mat_key: Option<u64> = None;
                let mut current_light_set: Option<u32> = None;

                for (i, &(_first_instance, n, mesh_id, first_mat_key, rep_dc_idx)) in
                    draw_batches.iter().enumerate()
                {
                    // Flush previous batch on invalid entry or state change.
                    let mesh_valid = mesh_id.is_some() && n > 0;
                    if !mesh_valid {
                        // Flush batch before gap, then reset.
                        let count = i as u32 - batch_start;
                        if count > 0 {
                            if let Some(prev_mesh) = current_mesh {
                                renderer.draw_mesh_multi_indirect(
                                    pass,
                                    prev_mesh,
                                    indirect_buf,
                                    batch_start as u64 * arg_stride as u64,
                                    count,
                                    &mut last_bound_mesh,
                                );
                            }
                        }
                        batch_start = i as u32 + 1;
                        current_mesh = None;
                        current_mat_key = None;
                        current_light_set = None;
                        continue;
                    }
                    let mesh_id = mesh_id.unwrap();
                    let rep_dc = ctx.visible[rep_dc_idx];
                    let light_set_changed = current_light_set != Some(rep_dc.light_set_id);
                    let mesh_changed = current_mesh != Some(mesh_id);
                    let mat_changed = current_mat_key != Some(first_mat_key);

                    if (mesh_changed || mat_changed || light_set_changed) && i > 0 {
                        let count = i as u32 - batch_start;
                        if count > 0 {
                            if let Some(prev_mesh) = current_mesh {
                                renderer.draw_mesh_multi_indirect(
                                    pass,
                                    prev_mesh,
                                    indirect_buf,
                                    batch_start as u64 * arg_stride as u64,
                                    count,
                                    &mut last_bound_mesh,
                                );
                            }
                        }
                        batch_start = i as u32;
                    }

                    if mat_changed || light_set_changed || current_mesh.is_none() {
                        let diffuse_color = if rep_dc.fill_style == rc3d_core::FillStyle::HiddenLine {
                            [0.08, 0.08, 0.08, 1.0]
                        } else {
                            [
                                rep_dc.diffuse_color.x,
                                rep_dc.diffuse_color.y,
                                rep_dc.diffuse_color.z,
                                1.0,
                            ]
                        };
                        let uniforms = SceneUniforms {
                            mvp: rep_dc.mvp.to_cols_array_2d(),
                            model: rep_dc.model_matrix.to_cols_array_2d(),
                            camera_pos: [
                                rep_dc.camera_pos.x,
                                rep_dc.camera_pos.y,
                                rep_dc.camera_pos.z,
                                1.0,
                            ],
                            diffuse_color,
                            ambient_color: [
                                rep_dc.ambient_color.x,
                                rep_dc.ambient_color.y,
                                rep_dc.ambient_color.z,
                                1.0,
                            ],
                            specular_color: [
                                rep_dc.specular_color.x,
                                rep_dc.specular_color.y,
                                rep_dc.specular_color.z,
                                1.0,
                            ],
                            shininess: [rep_dc.shininess, 0.0, 0.0, 0.0],
                            clip_planes: clip_arr,
                            clip_count,
                            pbr_base_color: [
                                rep_dc.base_color.x,
                                rep_dc.base_color.y,
                                rep_dc.base_color.z,
                                1.0,
                            ],
                            pbr_metallic_roughness: [
                                rep_dc.metallic,
                                rep_dc.roughness,
                                rep_dc.anisotropic,
                                0.0,
                            ],
                            pbr_emissive_alpha: [
                                rep_dc.emissive_color.x,
                                rep_dc.emissive_color.y,
                                rep_dc.emissive_color.z,
                                rep_dc.alpha_cutoff,
                            ],
                            pbr_alpha_flags: [
                                alpha_mode_to_f32(rep_dc.alpha_mode),
                                rep_dc.opacity,
                                if rep_dc.double_sided { 1.0 } else { 0.0 },
                                rep_dc.shade_mode_w(),
                            ],
                            pbr_clearcoat: [rep_dc.clearcoat_factor, rep_dc.clearcoat_roughness, rep_dc.iridescence_factor, rep_dc.iridescence_ior],
                            pbr_sheen: [rep_dc.sheen_color.x, rep_dc.sheen_color.y, rep_dc.sheen_color.z, rep_dc.sheen_roughness],
                            pbr_specular: [rep_dc.specular_color_factor.x, rep_dc.specular_color_factor.y, rep_dc.specular_color_factor.z, rep_dc.specular_factor],
                            pbr_transmission: [rep_dc.transmission_factor, rep_dc.ior, rep_dc.iridescence_thickness_min, rep_dc.iridescence_thickness_max],
                            light_set_index: [rep_dc.light_set_id as f32, 0.0, 0.0, 0.0],
                        };
                        let Some(phong_offset) = renderer.gpu.phong_pool.push_scene(&uniforms)
                        else {
                            // Uniform pool full: flush batch, skip remaining.
                            let count = i as u32 - batch_start;
                            if count > 0 {
                                if let Some(prev_mesh) = current_mesh {
                                    renderer.draw_mesh_multi_indirect(
                                        pass,
                                        prev_mesh,
                                        indirect_buf,
                                        batch_start as u64 * arg_stride as u64,
                                        count,
                                        &mut last_bound_mesh,
                                    );
                                }
                            }
                            batch_start = i as u32 + 1;
                            current_mesh = None;
                            current_mat_key = None;
                            current_light_set = None;
                            continue;
                        };
                        pass.set_bind_group(
                            0,
                            renderer.gpu.phong_pool.bind_group(),
                            &[phong_offset],
                        );
                        if renderer.last_material_bg_key != first_mat_key
                            || renderer.last_material_bg.is_none()
                        {
                            let mat_bg = albedo_material_bind_group(
                                &mut renderer.gpu.texture_cache,
                                &renderer.device,
                                &renderer.gpu.pipelines.pbr_material_bgl,
                                &renderer.queue,
                                rep_dc,
                            );
                            renderer.last_material_bg_key = first_mat_key;
                            renderer.last_material_bg = Some(mat_bg.clone());
                        }
                        if let Some(ref mat_bg) = renderer.last_material_bg {
                            pass.set_bind_group(1, mat_bg, &[]);
                        }
                    }

                    current_mesh = Some(mesh_id);
                    current_mat_key = Some(first_mat_key);
                    current_light_set = Some(rep_dc.light_set_id);
                }

                // Flush final batch
                let final_count = draw_batches.len() as u32 - batch_start;
                if final_count > 0 {
                    if let Some(mesh_id) = current_mesh {
                        renderer.draw_mesh_multi_indirect(
                            pass,
                            mesh_id,
                            indirect_buf,
                            batch_start as u64 * arg_stride as u64,
                            final_count,
                            &mut last_bound_mesh,
                        );
                    }
                }
            }
        } else {
            // Fallback: per-draw draw_indexed when multi-draw indirect is not supported.
            for &(first_instance, n, mesh_id, first_mat_key, rep_dc_idx) in &draw_batches {
                let Some(mesh_id) = mesh_id else {
                    continue;
                };
                if n == 0 {
                    continue;
                }
                let rep_dc = ctx.visible[rep_dc_idx];
                let diffuse_color = if rep_dc.fill_style == rc3d_core::FillStyle::HiddenLine {
                    [0.08, 0.08, 0.08, 1.0]
                } else {
                    [
                        rep_dc.diffuse_color.x,
                        rep_dc.diffuse_color.y,
                        rep_dc.diffuse_color.z,
                        1.0,
                    ]
                };
                let uniforms = SceneUniforms {
                    mvp: rep_dc.mvp.to_cols_array_2d(),
                    model: rep_dc.model_matrix.to_cols_array_2d(),
                    camera_pos: [
                        rep_dc.camera_pos.x,
                        rep_dc.camera_pos.y,
                        rep_dc.camera_pos.z,
                        1.0,
                    ],
                    diffuse_color,
                    ambient_color: [
                        rep_dc.ambient_color.x,
                        rep_dc.ambient_color.y,
                        rep_dc.ambient_color.z,
                        1.0,
                    ],
                    specular_color: [
                        rep_dc.specular_color.x,
                        rep_dc.specular_color.y,
                        rep_dc.specular_color.z,
                        1.0,
                    ],
                    shininess: [rep_dc.shininess, 0.0, 0.0, 0.0],
                    clip_planes: clip_arr,
                    clip_count,
                    pbr_base_color: [
                        rep_dc.base_color.x,
                        rep_dc.base_color.y,
                        rep_dc.base_color.z,
                        1.0,
                    ],
                    pbr_metallic_roughness: [
                        rep_dc.metallic,
                        rep_dc.roughness,
                        rep_dc.anisotropic,
                        0.0,
                    ],
                    pbr_emissive_alpha: [
                        rep_dc.emissive_color.x,
                        rep_dc.emissive_color.y,
                        rep_dc.emissive_color.z,
                        rep_dc.alpha_cutoff,
                    ],
                    pbr_alpha_flags: [
                        alpha_mode_to_f32(rep_dc.alpha_mode),
                        rep_dc.opacity,
                        if rep_dc.double_sided { 1.0 } else { 0.0 },
                        rep_dc.shade_mode_w(),
                    ],
                    pbr_clearcoat: [rep_dc.clearcoat_factor, rep_dc.clearcoat_roughness, rep_dc.iridescence_factor, rep_dc.iridescence_ior],
                    pbr_sheen: [rep_dc.sheen_color.x, rep_dc.sheen_color.y, rep_dc.sheen_color.z, rep_dc.sheen_roughness],
                    pbr_specular: [rep_dc.specular_color_factor.x, rep_dc.specular_color_factor.y, rep_dc.specular_color_factor.z, rep_dc.specular_factor],
                    pbr_transmission: [rep_dc.transmission_factor, rep_dc.ior, rep_dc.iridescence_thickness_min, rep_dc.iridescence_thickness_max],
                    light_set_index: [rep_dc.light_set_id as f32, 0.0, 0.0, 0.0],
                };
                let Some(phong_offset) = renderer.gpu.phong_pool.push_scene(&uniforms) else {
                    continue;
                };
                pass.set_bind_group(0, renderer.gpu.phong_pool.bind_group(), &[phong_offset]);
                if renderer.last_material_bg_key != first_mat_key
                    || renderer.last_material_bg.is_none()
                {
                    let mat_bg = albedo_material_bind_group(
                        &mut renderer.gpu.texture_cache,
                        &renderer.device,
                        &renderer.gpu.pipelines.pbr_material_bgl,
                        &renderer.queue,
                        rep_dc,
                    );
                    renderer.last_material_bg_key = first_mat_key;
                    renderer.last_material_bg = Some(mat_bg.clone());
                }
                if let Some(ref mat_bg) = renderer.last_material_bg {
                    pass.set_bind_group(1, mat_bg, &[]);
                }
                renderer.draw_mesh_instanced(
                    pass,
                    mesh_id,
                    first_instance,
                    n,
                    rep_dc.index_draw_range(),
                    &mut last_bound_mesh,
                );
            }
        }
    }

    if any_flat {
        draw_flat_triangle_batches(renderer, pass, ctx, flat_solid_pipeline);
    }
}

pub(crate) fn csm_to_uniform(vps: &[glam::Mat4; CSM_CASCADE_COUNT]) -> [[f32; 4]; 16] {
    let mut arr = [[0.0f32; 4]; 16];
    for (i, vp) in vps.iter().enumerate() {
        let cols = vp.to_cols_array_2d();
        for r in 0..4 {
            arr[i * 4 + r] = cols[r];
        }
    }
    arr
}
