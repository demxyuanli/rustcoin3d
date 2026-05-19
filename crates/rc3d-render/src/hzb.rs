pub struct HzbPyramid {
    pub texture: wgpu::Texture,
    pub mip_views: Vec<wgpu::TextureView>,
    pub full_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    /// Pre-built downsample bind groups for each mip transition (mip-1 → mip), index 0 = mip0→mip1
    pub downsample_max_bgs: Vec<wgpu::BindGroup>,
    pub downsample_min_bgs: Vec<wgpu::BindGroup>,
}

/// Two HZB chains from the same depth buffer: **max** mips for reverse-Z (larger = closer),
/// **min** mips for forward-Z (smaller = closer). Mip0 of each is the same linearized depth.
pub struct HzbPyramids {
    pub max_pyramid: HzbPyramid,
    pub min_pyramid: HzbPyramid,
}

impl HzbPyramids {
    pub fn new(
        device: &wgpu::Device,
        downsample_bgl: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            max_pyramid: HzbPyramid::new(device, downsample_bgl, width, height, true),
            min_pyramid: HzbPyramid::new(device, downsample_bgl, width, height, true),
        }
    }
}

impl HzbPyramid {
    pub fn new(
        device: &wgpu::Device,
        downsample_bgl: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
        build_sets: bool,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let mip_count = ((width.max(height) as f32).log2().floor() as u32 + 1).max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HZB Pyramid"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let full_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("HZB Full"),
            ..Default::default()
        });

        let mut mip_views = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            mip_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("HZB Mip View"),
                format: Some(wgpu::TextureFormat::R32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                usage: Some(
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                ),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            }));
        }

        let mut downsample_max_bgs = Vec::new();
        let mut downsample_min_bgs = Vec::new();
        if build_sets {
            for mip in 1..mip_count as usize {
                downsample_max_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("HZB Downsample Max BG (cached)"),
                    layout: downsample_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&mip_views[mip - 1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&mip_views[mip]),
                        },
                    ],
                }));
                downsample_min_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("HZB Downsample Min BG (cached)"),
                    layout: downsample_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&mip_views[mip - 1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&mip_views[mip]),
                        },
                    ],
                }));
            }
        }

        Self {
            texture,
            mip_views,
            full_view,
            width,
            height,
            mip_count,
            downsample_max_bgs,
            downsample_min_bgs,
        }
    }
}

pub struct HzbBaker {
    depth_to_mip0_pipeline: wgpu::ComputePipeline,
    depth_to_mip0_bgl: wgpu::BindGroupLayout,
    downsample_max_pipeline: wgpu::ComputePipeline,
    downsample_min_pipeline: wgpu::ComputePipeline,
    pub downsample_bgl: wgpu::BindGroupLayout,
    /// Generation counter incremented on resize to invalidate depth-to-mip0 caches.
    depth_view_generation: u64,
}

impl HzbBaker {
    pub fn new(device: &wgpu::Device) -> Self {
        let depth_to_mip0_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HZB Depth To Mip0"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hzb_depth_to_mip0.wgsl").into()),
        });

        let depth_to_mip0_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HZB Depth To Mip0 BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let depth_to_mip0_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HZB Depth To Mip0 PLL"),
            bind_group_layouts: &[&depth_to_mip0_bgl],
            push_constant_ranges: &[],
        });

        let depth_to_mip0_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HZB Depth To Mip0"),
            layout: Some(&depth_to_mip0_pll),
            module: &depth_to_mip0_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let downsample_max_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HZB Downsample Max"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hzb_downsample_max.wgsl").into()),
        });
        let downsample_min_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HZB Downsample Min"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hzb_downsample_min.wgsl").into()),
        });

        let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HZB Downsample BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let downsample_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HZB Downsample PLL"),
            bind_group_layouts: &[&downsample_bgl],
            push_constant_ranges: &[],
        });

        let downsample_max_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HZB Downsample Max"),
            layout: Some(&downsample_pll),
            module: &downsample_max_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let downsample_min_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HZB Downsample Min"),
            layout: Some(&downsample_pll),
            module: &downsample_min_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            depth_to_mip0_pipeline,
            depth_to_mip0_bgl,
            downsample_max_pipeline,
            downsample_min_pipeline,
            downsample_bgl,
            depth_view_generation: 0,
        }
    }

    pub fn increment_generation(&mut self) {
        self.depth_view_generation = self.depth_view_generation.wrapping_add(1);
    }

    pub fn build_from_depth(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        build_max: bool,
        build_min: bool,
        max_pyramid: &HzbPyramid,
        min_pyramid: &HzbPyramid,
    ) {
        if build_max {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HZB depth to max mip0"),
                layout: &self.depth_to_mip0_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&max_pyramid.mip_views[0]),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("HZB depth to max mip0"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.depth_to_mip0_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                let wg_x = max_pyramid.width.div_ceil(8);
                let wg_y = max_pyramid.height.div_ceil(8);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            for mip in 1..max_pyramid.mip_count as usize {
                let dst_w = (max_pyramid.width >> mip as u32).max(1);
                let dst_h = (max_pyramid.height >> mip as u32).max(1);
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("HZB Downsample Max"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.downsample_max_pipeline);
                    pass.set_bind_group(0, &max_pyramid.downsample_max_bgs[mip - 1], &[]);
                    let wg_x = dst_w.div_ceil(8);
                    let wg_y = dst_h.div_ceil(8);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
            }
        }

        if build_min {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HZB depth to min mip0"),
                layout: &self.depth_to_mip0_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&min_pyramid.mip_views[0]),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("HZB depth to min mip0"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.depth_to_mip0_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                let wg_x = min_pyramid.width.div_ceil(8);
                let wg_y = min_pyramid.height.div_ceil(8);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            for mip in 1..min_pyramid.mip_count as usize {
                let dst_w = (min_pyramid.width >> mip as u32).max(1);
                let dst_h = (min_pyramid.height >> mip as u32).max(1);
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("HZB Downsample Min"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.downsample_min_pipeline);
                    pass.set_bind_group(0, &min_pyramid.downsample_min_bgs[mip - 1], &[]);
                    let wg_x = dst_w.div_ceil(8);
                    let wg_y = dst_h.div_ceil(8);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
            }
        }
    }
}
