use crate::render_action::DrawCall;
use crate::vertex::{InstanceData, SceneUniforms, MAX_MORPH_WEIGHTS, CSM_CASCADE_COUNT};
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

pub(super) fn draw_opaque_triangle_batches(
    renderer: &mut crate::renderer::Renderer,
    pass: &mut wgpu::RenderPass<'_>,
    ctx: &super::PassContext<'_>,
    solid_pipeline: &wgpu::RenderPipeline,
    draw_meshlets: bool,
) {
    pass.set_pipeline(solid_pipeline);
    pass.set_stencil_reference(1);

    let mut clip_arr = [[0.0f32; 4]; 6];
    for (i, cp) in renderer.frame.clip_planes.iter().enumerate() {
        if i < 6 {
            clip_arr[i] = *cp;
        }
    }
    let clip_count = [renderer.frame.clip_planes.len().min(6) as f32, 0.0, 0.0, 0.0];

    let meshlet_set: std::collections::HashSet<usize> = ctx.meshlet_indices.iter().copied().collect();

    let mut last_bound_mesh = None;
    let mut start = 0usize;
    while start < ctx.solid_order.len() {
        let head_idx = ctx.solid_order[start];
        let head_dc = ctx.visible[head_idx];
        let light_key = crate::sort_keys::light_sort_key(head_dc);
        let mut end = start + 1;
        while end < ctx.solid_order.len() {
            let idx = ctx.solid_order[end];
            let dc = ctx.visible[idx];
            if crate::sort_keys::light_sort_key(dc) != light_key {
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
                let mat_bg = albedo_material_bind_group(
                    &mut renderer.gpu.texture_cache, &renderer.device,
                    &renderer.gpu.pipelines.pbr_material_bgl, &renderer.queue, dc,
                );
                pass.set_bind_group(0, renderer.gpu.phong_pool.bind_group(), &[offset]);
                pass.set_bind_group(1, mat_bg, &[]);
                match &renderer.gpu.csm_shadow {
                    Some(csm) => pass.set_bind_group(2, &csm.bind_group, &[]),
                    None => log::error!("CSM shadow missing; shadow bind group not set"),
                }
                pass.set_bind_group(3, &renderer.gpu.ibl_instance_bind_group, &[]);
                if let Some(cluster_set) = renderer.gpu.assets.cluster_get(&ptr) {
                    if let Some(cluster_renderer) = renderer.gpu.cluster_renderer.as_ref() {
                        cluster_renderer.draw_clustered(pass, cluster_set);
                    }
                }
            }
        }

        if !standard_draws.is_empty() {
            for &i in &standard_draws {
                let dc = ctx.visible[i];
                let diffuse = if ctx.mode == DisplayMode::HiddenLine {
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
                    diffuse_color: diffuse,
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
                let Some(offset) = renderer.gpu.phong_pool.push_scene(&uniforms) else { continue };

                let one = [InstanceData {
                    model: dc.model_matrix.to_cols_array_2d(),
                    mvp: dc.mvp.to_cols_array_2d(),
                    diffuse_color: diffuse,
                    base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, 1.0],
                    metallic_roughness: [dc.metallic, dc.roughness, 0.0, 0.0],
                    emissive_alpha: [dc.emissive_color.x, dc.emissive_color.y, dc.emissive_color.z, dc.alpha_cutoff],
                    morph_weights: pack_morph_weights(&dc.morph_weights),
                    morph_count: [dc.morph_weights.len().min(crate::vertex::MAX_MORPH_WEIGHTS) as f32, 0.0, 0.0, 0.0],
                }];
                renderer.queue.write_buffer(&renderer.gpu.instance_buffer, 0, bytemuck::cast_slice(&one));

                let mat_bg = albedo_material_bind_group(
                    &mut renderer.gpu.texture_cache, &renderer.device,
                    &renderer.gpu.pipelines.pbr_material_bgl, &renderer.queue, dc,
                );
                pass.set_bind_group(0, renderer.gpu.phong_pool.bind_group(), &[offset]);
                pass.set_bind_group(1, mat_bg, &[]);
                match &renderer.gpu.csm_shadow {
                    Some(csm) => pass.set_bind_group(2, &csm.bind_group, &[]),
                    None => log::error!("CSM shadow missing; shadow bind group not set"),
                }
                pass.set_bind_group(3, &renderer.gpu.ibl_instance_bind_group, &[]);

                if let Some(mesh_id) = ctx.mesh_handles[i] {
                    renderer.draw_mesh_batched(pass, mesh_id, &mut last_bound_mesh);
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
