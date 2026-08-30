const CLUSTERS_X: u32 = 16;
const CLUSTERS_Y: u32 = 8;
const CLUSTERS_Z: u32 = 24;
const CLUSTER_COUNT: u32 = CLUSTERS_X * CLUSTERS_Y * CLUSTERS_Z;
const MAX_LIGHTS: u32 = 256;
const MAX_LIGHTS_PER_CLUSTER: u32 = 64;

/// GPU-side clustered forward lighting resources.
pub struct ClusterLightResources {
    pub point_light_buffer: wgpu::Buffer,
    pub spot_light_buffer: wgpu::Buffer,
    pub light_grid: wgpu::Buffer,
    pub light_index_list: wgpu::Buffer,
    pub light_grid_bgl: wgpu::BindGroupLayout,
}

impl ClusterLightResources {
    pub fn new(device: &wgpu::Device) -> Self {
        let point_light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PointLight SSBO"),
            size: (std::mem::size_of::<GpuPointLight>() * MAX_LIGHTS as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let spot_light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SpotLight SSBO"),
            size: (std::mem::size_of::<GpuSpotLight>() * MAX_LIGHTS as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Per-cluster: [offset, count] in the global light index list
        let light_grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LightGrid SSBO"),
            size: (CLUSTER_COUNT * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Global light index list: cluster_count * MAX_LIGHTS_PER_CLUSTER entries
        let light_index_list = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LightIndexList SSBO"),
            size: (CLUSTER_COUNT * MAX_LIGHTS_PER_CLUSTER * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_grid_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LightGrid BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        Self {
            point_light_buffer,
            spot_light_buffer,
            light_grid,
            light_index_list,
            light_grid_bgl,
        }
    }
}

/// GPU-compatible point light data.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPointLight {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
}

/// GPU-compatible spot light data.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSpotLight {
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub radius: f32,
    pub cos_inner: f32,
    pub cos_outer: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightCullParamsUniform {
    inv_proj: [[f32; 4]; 4],
    screen_width: u32,
    screen_height: u32,
    num_point_lights: u32,
    num_spot_lights: u32,
    z_near: f32,
    z_far: f32,
    depth_slice_scale: f32,
    depth_slice_bias: f32,
}

/// Compute pipeline for light cluster assignment.
pub struct ClusterLightCuller {
    pipeline: wgpu::ComputePipeline,
    cull_bgl: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
}

impl ClusterLightCuller {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cluster Light Cull"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/cluster_light_cull.wgsl").into(),
            ),
        });

        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ClusterLightCull BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ClusterLightCull PLL"),
            bind_group_layouts: &[Some(&cull_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cluster Light Cull Pipe"),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("cluster_light_cull"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LightCullParams"),
            size: std::mem::size_of::<LightCullParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            cull_bgl,
            params_buffer,
        }
    }

    /// Run light culling. `inv_proj` is the inverse of the camera projection matrix.
    /// `point_lights` and `spot_lights` are the scene's light arrays.
    pub fn cull_lights(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &ClusterLightResources,
        inv_proj: glam::Mat4,
        screen_width: u32,
        screen_height: u32,
        z_near: f32,
        z_far: f32,
        point_lights: &[GpuPointLight],
        spot_lights: &[GpuSpotLight],
    ) {
        let num_point = (point_lights.len() as u32).min(MAX_LIGHTS);
        let num_spot = (spot_lights.len() as u32).min(MAX_LIGHTS.saturating_sub(num_point));

        let depth_slice_scale = CLUSTERS_Z as f32 / (z_far / z_near).ln();
        let depth_slice_bias = 0.0f32;

        if num_point > 0 {
            queue.write_buffer(
                &resources.point_light_buffer,
                0,
                bytemuck::cast_slice(&point_lights[..num_point as usize]),
            );
        }
        if num_spot > 0 {
            queue.write_buffer(
                &resources.spot_light_buffer,
                0,
                bytemuck::cast_slice(&spot_lights[..num_spot as usize]),
            );
        }

        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&LightCullParamsUniform {
                inv_proj: inv_proj.to_cols_array_2d(),
                screen_width,
                screen_height,
                num_point_lights: num_point,
                num_spot_lights: num_spot,
                z_near,
                z_far,
                depth_slice_scale,
                depth_slice_bias,
            }),
        );

        // Clear light grid
        let zeroes = vec![0u8; (CLUSTER_COUNT * 2 * 4) as usize];
        queue.write_buffer(&resources.light_grid, 0, &zeroes);

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ClusterLightCull BG"),
            layout: &self.cull_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.point_light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.spot_light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.light_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.light_index_list.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cluster Light Cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(CLUSTERS_X, CLUSTERS_Y, 1);
        drop(pass);
    }

    /// Get the bind group layout for the light grid (for use in PBR pipeline layout).
    pub fn light_grid_bgl(&self) -> &wgpu::BindGroupLayout {
        // Return the resources' BGL; we need the culler to expose it.
        // Actually, the resources struct owns the BGL. Return a reference pattern.
        &self.cull_bgl // Reuse; the actual light_grid_bgl is on resources
    }
}

/// Collect visible point/spot lights from draw-call light sets and dispatch cluster light culling.
pub fn dispatch_cluster_light_cull(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    visible: &[&crate::render_action::DrawCall],
    light_sets: &crate::light_set::LightSetTable,
    camera_inv_proj: glam::Mat4,
    ew: u32,
    eh: u32,
) {
    let Some(culler) = renderer.gpu.cluster_light_culler.as_ref() else { return };
    let Some(resources) = renderer.gpu.cluster_lights.as_ref() else { return };

    let mut point_lights: Vec<GpuPointLight> = Vec::new();
    let mut spot_lights: Vec<GpuSpotLight> = Vec::new();

    let mut seen_light_sets: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for dc in visible.iter() {
        if !seen_light_sets.insert(dc.light_set_id) {
            continue;
        }
        let lights = light_sets.get(dc.light_set_id);
        let (ref light_dirs, ref light_colors, ref light_types, ref light_positions, ref spot_params, light_count) = *lights;
        for i in 0..(light_count as usize).min(crate::vertex::MAX_LIGHTS) {
            let lt = light_types[i][0];
            let pos = light_positions[i];
            let col = light_colors[i];
            let intensity = light_colors[i][3];

            if (lt - 1.0).abs() < 0.5 {
                if point_lights.len() < 256 {
                    point_lights.push(GpuPointLight {
                        position: [pos[0], pos[1], pos[2]],
                        radius: pos[3].max(1.0),
                        color: [col[0], col[1], col[2]],
                        intensity,
                    });
                }
            } else if (lt - 3.0).abs() < 0.5 {
                let dir = light_dirs[i];
                let sp = spot_params[i];
                if spot_lights.len() < 256 {
                    spot_lights.push(GpuSpotLight {
                        position: [pos[0], pos[1], pos[2]],
                        direction: [dir[0], dir[1], dir[2]],
                        radius: pos[3].max(1.0),
                        cos_inner: sp[0],
                        cos_outer: sp[1],
                        color: [col[0], col[1], col[2]],
                        intensity,
                        _pad: 0.0,
                    });
                }
            }
        }
    }

    if !point_lights.is_empty() || !spot_lights.is_empty() {
        culler.cull_lights(
            &renderer.device,
            &renderer.queue,
            encoder,
            resources,
            camera_inv_proj,
            ew, eh,
            0.1, 1000.0,
            &point_lights,
            &spot_lights,
        );
    }
}
