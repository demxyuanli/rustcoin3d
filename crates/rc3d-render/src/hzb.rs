pub struct HzbPyramid {
    pub texture: wgpu::Texture,
    pub mip_views: Vec<wgpu::TextureView>,
    pub full_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
}

/// Two HZB chains from the same depth buffer: **max** mips for reverse-Z (larger = closer),
/// **min** mips for forward-Z (smaller = closer). Mip0 of each is the same linearized depth.
pub struct HzbPyramids {
    pub max_pyramid: HzbPyramid,
    pub min_pyramid: HzbPyramid,
}

impl HzbPyramids {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        Self {
            max_pyramid: HzbPyramid::new(device, width, height),
            min_pyramid: HzbPyramid::new(device, width, height),
        }
    }
}

impl HzbPyramid {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
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

        Self {
            texture,
            mip_views,
            full_view,
            width,
            height,
            mip_count,
        }
    }
}

pub struct HzbBaker {
    depth_to_mip0_pipeline: wgpu::ComputePipeline,
    depth_to_mip0_bgl: wgpu::BindGroupLayout,
    downsample_max_pipeline: wgpu::ComputePipeline,
    downsample_min_pipeline: wgpu::ComputePipeline,
    downsample_bgl: wgpu::BindGroupLayout,
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
        }
    }

    pub fn build_from_depth(
        &self,
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
                let wg_x = (max_pyramid.width + 7) / 8;
                let wg_y = (max_pyramid.height + 7) / 8;
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            for mip in 1..max_pyramid.mip_count as usize {
                let dst_w = (max_pyramid.width >> mip as u32).max(1);
                let dst_h = (max_pyramid.height >> mip as u32).max(1);
                let bg_max = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("HZB Downsample Max BG"),
                    layout: &self.downsample_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&max_pyramid.mip_views[mip - 1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&max_pyramid.mip_views[mip]),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("HZB Downsample Max"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.downsample_max_pipeline);
                    pass.set_bind_group(0, &bg_max, &[]);
                    let wg_x = (dst_w + 7) / 8;
                    let wg_y = (dst_h + 7) / 8;
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
                let wg_x = (min_pyramid.width + 7) / 8;
                let wg_y = (min_pyramid.height + 7) / 8;
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            for mip in 1..min_pyramid.mip_count as usize {
                let dst_w = (min_pyramid.width >> mip as u32).max(1);
                let dst_h = (min_pyramid.height >> mip as u32).max(1);
                let bg_min = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("HZB Downsample Min BG"),
                    layout: &self.downsample_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&min_pyramid.mip_views[mip - 1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&min_pyramid.mip_views[mip]),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("HZB Downsample Min"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.downsample_min_pipeline);
                    pass.set_bind_group(0, &bg_min, &[]);
                    let wg_x = (dst_w + 7) / 8;
                    let wg_y = (dst_h + 7) / 8;
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
            }
        }
    }
}
