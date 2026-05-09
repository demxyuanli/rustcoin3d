use crate::render_action::DrawCall;
use crate::vertex::{FlatUniforms, InstanceData, SceneUniforms, MAX_MORPH_WEIGHTS, CSM_CASCADE_COUNT};
use rc3d_core::DisplayMode;
use rc3d_scene::AlphaMode;

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

/// Fast material identity hash from texture paths + PBR parameters.
/// Pure-color materials with different base_color/metallic/roughness
/// produce different keys, avoiding stale bind group cache hits.
fn material_key(dc: &DrawCall) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    dc.albedo_path.hash(&mut h);
    dc.normal_path.hash(&mut h);
    dc.metallic_roughness_path.hash(&mut h);
    dc.emissive_path.hash(&mut h);
    dc.occlusion_path.hash(&mut h);
    dc.base_color.x.to_bits().hash(&mut h);
    dc.base_color.y.to_bits().hash(&mut h);
    dc.base_color.z.to_bits().hash(&mut h);
    dc.metallic.to_bits().hash(&mut h);
    dc.roughness.to_bits().hash(&mut h);
    dc.opacity.to_bits().hash(&mut h);
    dc.alpha_cutoff.to_bits().hash(&mut h);
    alpha_mode_to_f32(dc.alpha_mode).to_bits().hash(&mut h);
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
        device, layout,
        albedo_handle, normal_handle, mr_handle, emissive_handle, occlusion_handle,
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
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: [dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0],
        };
        if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
            if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_mesh_instanced(pass, mesh_id, 0, 1, &mut last_bound_mesh);
            }
        }
    }
}

pub(super) fn draw_opaque_triangle_batches(
    renderer: &mut crate::renderer::Renderer,
    pass: &mut wgpu::RenderPass<'_>,
    ctx: &super::PassContext<'_>,
    solid_pipeline: &wgpu::RenderPipeline,
    flat_solid_pipeline: &wgpu::RenderPipeline,
    draw_meshlets: bool,
) {
    if ctx.mode == DisplayMode::Flat {
        draw_flat_triangle_batches(renderer, pass, ctx, flat_solid_pipeline);
        return;
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
    let clip_count = [renderer.frame.clip_planes.len().min(6) as f32, 0.0, 0.0, 0.0];

    let meshlet_set: std::collections::HashSet<usize> = ctx.meshlet_indices.iter().copied().collect();

    // Pre-compute material keys for all visible draws to avoid repeated hashing.
    let mat_keys: Vec<u64> = ctx.visible.iter().map(|dc| material_key(dc)).collect();

    let mut last_bound_mesh = None;
    // Track cumulative offset into instance_buffer so each subgroup writes to a
    // non-overlapping region and draws with the correct first_instance.
    let instance_stride = std::mem::size_of::<InstanceData>() as u64;
    let mut instance_cursor: u64 = 0;

    let mut start = 0usize;
    while start < ctx.solid_order.len() {
        let head_idx = ctx.solid_order[start];
        let head_dc = ctx.visible[head_idx];
        let light_key = head_dc.light_key;
        let mut end = start + 1;
        while end < ctx.solid_order.len() {
            let idx = ctx.solid_order[end];
            let dc = ctx.visible[idx];
            if dc.light_key != light_key {
                break;
            }
            end += 1;
        }

        let mut meshlet_draws: Vec<usize> = Vec::new();
        let mut standard_draws: Vec<usize> = Vec::new();
        for &i in &ctx.solid_order[start..end] {
            if meshlet_set.contains(&i) {
                if draw_meshlets {
                    meshlet_draws.push(i);
                } else {
                    standard_draws.push(i);
                }
            } else {
                standard_draws.push(i);
            }
        }

        for &i in &meshlet_draws {
            if !draw_meshlets { continue; }
            let dc = ctx.visible[i];
            let md = match dc.meshlet_data.as_ref() { Some(md) => md, None => continue };
            let ptr = std::sync::Arc::as_ptr(md) as u64;
            if !(renderer.gpu.cluster_renderer.is_some() && renderer.gpu.assets.cluster_contains(&ptr)) { continue; }
            let diffuse_color = if ctx.mode == DisplayMode::HiddenLine {
                [0.08, 0.08, 0.08, 1.0]
            } else {
                [dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]
            };
            let uniforms = SceneUniforms {
                mvp: dc.mvp.to_cols_array_2d(),
                model: dc.model_matrix.to_cols_array_2d(),
                camera_pos: [dc.camera_pos.x, dc.camera_pos.y, dc.camera_pos.z, 1.0],
                light_dirs: head_dc.light_dirs, light_colors: head_dc.light_colors,
                light_types: head_dc.light_types, light_positions: head_dc.light_positions,
                spot_params: head_dc.spot_params,
                light_count: [head_dc.light_count as f32, 0.0, 0.0, 0.0],
                diffuse_color,
                ambient_color: [dc.ambient_color.x, dc.ambient_color.y, dc.ambient_color.z, 1.0],
                specular_color: [dc.specular_color.x, dc.specular_color.y, dc.specular_color.z, 1.0],
                shininess: [dc.shininess, 0.0, 0.0, 0.0],
                clip_planes: clip_arr, clip_count,
                pbr_base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, 1.0],
                pbr_metallic_roughness: [dc.metallic, dc.roughness, dc.anisotropic, 0.0],
                pbr_emissive_alpha: [dc.emissive_color.x, dc.emissive_color.y, dc.emissive_color.z, dc.alpha_cutoff],
                pbr_alpha_flags: [alpha_mode_to_f32(dc.alpha_mode), dc.opacity, if dc.double_sided { 1.0 } else { 0.0 }, 0.0],
                ibl_diffuse: renderer.gpu.ibl_diffuse, ibl_specular: renderer.gpu.ibl_specular,
                csm_view_proj: csm_to_uniform(&ctx.csm_view_proj),
                csm_split_depths: ctx.csm_split_depths,
                shadow_params: ctx.shadow_params,
            };
            if let Some(offset) = renderer.gpu.phong_pool.push_scene(&uniforms) {
                let key = mat_keys[i];
                if renderer.last_material_bg_key != key || renderer.last_material_bg.is_none() {
                    let bg = albedo_material_bind_group(
                        &mut renderer.gpu.texture_cache, &renderer.device,
                        &renderer.gpu.pipelines.pbr_material_bgl, &renderer.queue, dc,
                    );
                    renderer.last_material_bg_key = key;
                    renderer.last_material_bg = Some(bg.clone());
                }
                pass.set_bind_group(0, renderer.gpu.phong_pool.bind_group(), &[offset]);
                if let Some(ref mat_bg) = renderer.last_material_bg {
                    pass.set_bind_group(1, mat_bg, &[]);
                }
                if let Some(cluster_set) = renderer.gpu.assets.cluster_get(&ptr) {
                    if let Some(cluster_renderer) = renderer.gpu.cluster_renderer.as_ref() {
                        cluster_renderer.draw_clustered(pass, cluster_set);
                    }
                }
            }
        }

        if !standard_draws.is_empty() {
            // O(n) HashMap grouping by (mesh_id, material_key) — no sort needed.
            let mut groups: std::collections::HashMap<
                (Option<crate::gpu_resource::MeshId>, u64),
                Vec<usize>,
            > = std::collections::HashMap::new();
            for &i in &standard_draws {
                let key = (ctx.mesh_handles[i], mat_keys[i]);
                groups.entry(key).or_default().push(i);
            }

            for ((mesh_id, _mat_key), subgroup) in &groups {
                if let Some(mesh_id) = mesh_id {
                    let first_dc = ctx.visible[subgroup[0]];
                    let mut instances: Vec<InstanceData> = Vec::with_capacity(subgroup.len());
                    for &i in subgroup {
                        let dc = ctx.visible[i];
                        let diffuse = if ctx.mode == DisplayMode::HiddenLine {
                            [0.08, 0.08, 0.08, 1.0]
                        } else {
                            [dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]
                        };
                        instances.push(InstanceData {
                            model: dc.model_matrix.to_cols_array_2d(),
                            mvp: dc.mvp.to_cols_array_2d(),
                            diffuse_color: diffuse,
                            base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, 1.0],
                            metallic_roughness: [dc.metallic, dc.roughness, 0.0, 0.0],
                            emissive_alpha: [dc.emissive_color.x, dc.emissive_color.y, dc.emissive_color.z, dc.alpha_cutoff],
                            morph_weights: pack_morph_weights(&dc.morph_weights),
                            morph_count: [dc.morph_weights.len().min(crate::vertex::MAX_MORPH_WEIGHTS) as f32, 0.0, 0.0, 0.0],
                        });
                    }

                    // Shared SceneUniforms: light data from head_dc, material from first_dc.
                    let diffuse = if ctx.mode == DisplayMode::HiddenLine {
                        [0.08, 0.08, 0.08, 1.0]
                    } else {
                        [first_dc.diffuse_color.x, first_dc.diffuse_color.y, first_dc.diffuse_color.z, 1.0]
                    };
                    let uniforms = SceneUniforms {
                        mvp: first_dc.mvp.to_cols_array_2d(),
                        model: first_dc.model_matrix.to_cols_array_2d(),
                        camera_pos: [first_dc.camera_pos.x, first_dc.camera_pos.y, first_dc.camera_pos.z, 1.0],
                        light_dirs: head_dc.light_dirs, light_colors: head_dc.light_colors,
                        light_types: head_dc.light_types, light_positions: head_dc.light_positions,
                        spot_params: head_dc.spot_params,
                        light_count: [head_dc.light_count as f32, 0.0, 0.0, 0.0],
                        diffuse_color: diffuse,
                        ambient_color: [first_dc.ambient_color.x, first_dc.ambient_color.y, first_dc.ambient_color.z, 1.0],
                        specular_color: [first_dc.specular_color.x, first_dc.specular_color.y, first_dc.specular_color.z, 1.0],
                        shininess: [first_dc.shininess, 0.0, 0.0, 0.0],
                        clip_planes: clip_arr, clip_count,
                        pbr_base_color: [first_dc.base_color.x, first_dc.base_color.y, first_dc.base_color.z, 1.0],
                        pbr_metallic_roughness: [first_dc.metallic, first_dc.roughness, first_dc.anisotropic, 0.0],
                        pbr_emissive_alpha: [first_dc.emissive_color.x, first_dc.emissive_color.y, first_dc.emissive_color.z, first_dc.alpha_cutoff],
                        pbr_alpha_flags: [alpha_mode_to_f32(first_dc.alpha_mode), first_dc.opacity, if first_dc.double_sided { 1.0 } else { 0.0 }, 0.0],
                        ibl_diffuse: renderer.gpu.ibl_diffuse, ibl_specular: renderer.gpu.ibl_specular,
                        csm_view_proj: csm_to_uniform(&ctx.csm_view_proj),
                        csm_split_depths: ctx.csm_split_depths,
                        shadow_params: ctx.shadow_params,
                    };
                    let Some(offset) = renderer.gpu.phong_pool.push_scene(&uniforms) else {
                        continue;
                    };

                    let first_instance = (instance_cursor / instance_stride) as u32;
                    let n = subgroup.len() as u32;
                    renderer.queue.write_buffer(
                        &renderer.gpu.instance_buffer, instance_cursor,
                        bytemuck::cast_slice(&instances),
                    );
                    instance_cursor += n as u64 * instance_stride;

                    let first_mat_key = mat_keys[subgroup[0]];
                    if renderer.last_material_bg_key != first_mat_key || renderer.last_material_bg.is_none() {
                        let mat_bg = albedo_material_bind_group(
                            &mut renderer.gpu.texture_cache, &renderer.device,
                            &renderer.gpu.pipelines.pbr_material_bgl, &renderer.queue, first_dc,
                        );
                        renderer.last_material_bg_key = first_mat_key;
                        renderer.last_material_bg = Some(mat_bg.clone());
                    }
                    pass.set_bind_group(0, renderer.gpu.phong_pool.bind_group(), &[offset]);
                    if let Some(ref mat_bg) = renderer.last_material_bg {
                        pass.set_bind_group(1, mat_bg, &[]);
                    }

                    renderer.draw_mesh_instanced(pass, *mesh_id, first_instance, n, &mut last_bound_mesh);
                }
            }
        }
        start = end;
    }
}

pub(super) fn csm_to_uniform(vps: &[glam::Mat4; CSM_CASCADE_COUNT]) -> [[f32; 4]; 16] {
    let mut arr = [[0.0f32; 4]; 16];
    for (i, vp) in vps.iter().enumerate() {
        let cols = vp.to_cols_array_2d();
        for r in 0..4 {
            arr[i * 4 + r] = cols[r];
        }
    }
    arr
}
