use std::collections::HashMap;
use std::path::Path;

use glam::Vec3;
use slotmap::{new_key_type, SlotMap};

new_key_type! {
    pub struct TextureHandle;
}

struct GpuTexture2d {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    mip_level_count: u32,
}

/// CPU-loaded RGBA8 before GPU upload.
pub struct RgbaImageData {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl RgbaImageData {
    pub fn from_path(path: &Path) -> Result<Self, String> {
        let img = image::open(path).map_err(|e| e.to_string())?;
        Self::from_dynamic(img)
    }

    pub fn from_dynamic(img: image::DynamicImage) -> Result<Self, String> {
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Ok(Self {
            width,
            height,
            rgba: rgba.into_raw(),
        })
    }

    pub fn white_1x1() -> Self {
        Self {
            width: 1,
            height: 1,
            rgba: vec![255, 255, 255, 255],
        }
    }
}

/// Average HDR / LDR environment radiance for cheap IBL (diffuse + specular approximation).
pub fn ibl_from_image_path(path: &Path) -> Option<(Vec3, Vec3)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb32f();
    let mut sum = Vec3::ZERO;
    let mut n = 0u32;
    for p in rgb.pixels() {
        let c = p.0;
        sum += Vec3::new(c[0], c[1], c[2]);
        n += 1;
    }
    if n == 0 {
        return None;
    }
    let avg = sum / n as f32;
    let diffuse = avg * 0.22;
    let spec = avg * 0.55;
    Some((diffuse, spec))
}

fn compute_mip_count(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height);
    if max_dim <= 1 {
        1
    } else {
        (max_dim as f32).log2().floor() as u32 + 1
    }
}

pub struct TextureCache {
    sampler: wgpu::Sampler,
    white: TextureHandle,
    flat_normal: TextureHandle,
    textures: SlotMap<TextureHandle, GpuTexture2d>,
    albedo_bind_groups: HashMap<TextureHandle, wgpu::BindGroup>,
    pbr_bind_groups: HashMap<(TextureHandle, TextureHandle, TextureHandle, TextureHandle, TextureHandle), wgpu::BindGroup>,
    path_to_handle: HashMap<String, TextureHandle>,
    mip_pipeline: Option<wgpu::ComputePipeline>,
    mip_bgl: Option<wgpu::BindGroupLayout>,
}

impl TextureCache {
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PBR albedo sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mut textures: SlotMap<TextureHandle, GpuTexture2d> = SlotMap::with_key();
        let white_data = RgbaImageData::white_1x1();
        let white = Self::insert_rgba(device, _queue, &white_data, &mut textures, None, None);
        let flat_normal_data = RgbaImageData {
            width: 1, height: 1,
            rgba: vec![128, 128, 255, 255], // tangent-space Z+ (0,0,1) encoded as RGB
        };
        let flat_normal = Self::insert_rgba(device, _queue, &flat_normal_data, &mut textures, None, None);

        Self {
            sampler,
            white,
            flat_normal,
            textures,
            albedo_bind_groups: HashMap::new(),
            pbr_bind_groups: HashMap::new(),
            path_to_handle: HashMap::new(),
            mip_pipeline: None,
            mip_bgl: None,
        }
    }

    fn ensure_mip_pipeline(&mut self, device: &wgpu::Device) {
        if self.mip_pipeline.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mip Downsample BGL"),
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
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mip Downsample"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/mip_downsample.wgsl").into(),
            ),
        });

        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mip Downsample PLL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mip Downsample Pipe"),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        self.mip_bgl = Some(bgl);
        self.mip_pipeline = Some(pipeline);
    }

    fn generate_mip_chain(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        mip_count: u32,
    ) {
        let mip_pipeline = match &self.mip_pipeline {
            Some(p) => p,
            None => return,
        };
        let mip_bgl = match &self.mip_bgl {
            Some(bgl) => bgl,
            None => return,
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mipmap Generator"),
        });

        let w = texture.width();
        let h = texture.height();

        for mip in 1..mip_count {
            let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Mip Src"),
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: mip - 1,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                ..Default::default()
            });
            let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Mip Dst"),
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                ..Default::default()
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mip Downsample BG"),
                layout: mip_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&dst_view),
                    },
                ],
            });

            let dst_w = (w >> mip).max(1);
            let dst_h = (h >> mip).max(1);
            let wg_x = dst_w.div_ceil(8);
            let wg_y = dst_h.div_ceil(8);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Mip Downsample"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(mip_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    fn insert_rgba(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &RgbaImageData,
        textures: &mut SlotMap<TextureHandle, GpuTexture2d>,
        _mip_pipeline: Option<&wgpu::ComputePipeline>,
        _mip_bgl: Option<&wgpu::BindGroupLayout>,
    ) -> TextureHandle {
        let w = data.width.max(1);
        let h = data.height.max(1);
        let mip_count = compute_mip_count(w, h);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Albedo"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Albedo View"),
            ..Default::default()
        });

        // Generate mip chain if pipeline is available
        // This is done separately after insert to avoid borrow issues
        
        textures.insert(GpuTexture2d {
            _texture: texture,
            view,
            mip_level_count: mip_count,
        })
    }

    pub fn white_handle(&self) -> TextureHandle {
        self.white
    }

    pub fn default_normal_handle(&self) -> TextureHandle {
        self.flat_normal
    }

    pub fn white_handle_ref(&self) -> TextureHandle {
        self.white
    }

    pub fn pbr_material_bind_group(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        albedo_handle: TextureHandle,
        normal_handle: TextureHandle,
        metallic_roughness_handle: TextureHandle,
        emissive_handle: TextureHandle,
        occlusion_handle: TextureHandle,
    ) -> &wgpu::BindGroup {
        let key = (albedo_handle, normal_handle, metallic_roughness_handle, emissive_handle, occlusion_handle);
        self.pbr_bind_groups
            .entry(key)
            .or_insert_with(|| {
                let albedo_view = self.textures.get(albedo_handle).map(|t| &t.view).expect("albedo handle");
                let normal_view = self.textures.get(normal_handle).map(|t| &t.view).expect("normal handle");
                let mr_view = self.textures.get(metallic_roughness_handle).map(|t| &t.view).expect("mr handle");
                let emissive_view = self.textures.get(emissive_handle).map(|t| &t.view).expect("emissive handle");
                let occ_view = self.textures.get(occlusion_handle).map(|t| &t.view).expect("occlusion handle");
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PBR material BG"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(albedo_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(normal_view) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(mr_view) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(emissive_view) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(occ_view) },
                    ],
                })
            })
    }

    pub fn black_placeholder(&mut self, device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Black 1x1"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    pub fn load_path(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
    ) -> TextureHandle {
        if let Some(&h) = self.path_to_handle.get(path) {
            return h;
        }
        let p = Path::new(path);
        let data = match RgbaImageData::from_path(p) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("Texture load failed ({path}): {e}, using white");
                RgbaImageData::white_1x1()
            }
        };
        self.ensure_mip_pipeline(device);
        let h = Self::insert_rgba(
            device, queue, &data, &mut self.textures,
            self.mip_pipeline.as_ref(),
            self.mip_bgl.as_ref(),
        );
        self.path_to_handle.insert(path.to_string(), h);

        // Generate mip chain for this texture (submits its own command buffer)
        if let Some(gpu_tex) = self.textures.get(h) {
            self.generate_mip_chain(device, queue, &gpu_tex._texture, gpu_tex.mip_level_count);
        }

        h
    }

    pub fn albedo_bind_group(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        handle: TextureHandle,
    ) -> &wgpu::BindGroup {
        self.albedo_bind_groups.entry(handle).or_insert_with(|| {
            let view = self
                .textures
                .get(handle)
                .map(|t| &t.view)
                .expect("texture handle");
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PBR albedo BG"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        })
    }
}
