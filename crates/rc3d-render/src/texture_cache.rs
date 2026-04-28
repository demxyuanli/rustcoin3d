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

pub struct TextureCache {
    sampler: wgpu::Sampler,
    white: TextureHandle,
    textures: SlotMap<TextureHandle, GpuTexture2d>,
    albedo_bind_groups: HashMap<TextureHandle, wgpu::BindGroup>,
    path_to_handle: HashMap<String, TextureHandle>,
}

impl TextureCache {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
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
        let white = Self::insert_rgba(device, queue, &white_data, &mut textures);

        Self {
            sampler,
            white,
            textures,
            albedo_bind_groups: HashMap::new(),
            path_to_handle: HashMap::new(),
        }
    }

    fn insert_rgba(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &RgbaImageData,
        textures: &mut SlotMap<TextureHandle, GpuTexture2d>,
    ) -> TextureHandle {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Albedo"),
            size: wgpu::Extent3d {
                width: data.width.max(1),
                height: data.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
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
                bytes_per_row: Some(4 * data.width.max(1)),
                rows_per_image: Some(data.height.max(1)),
            },
            wgpu::Extent3d {
                width: data.width.max(1),
                height: data.height.max(1),
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        textures.insert(GpuTexture2d {
            _texture: texture,
            view,
        })
    }

    pub fn white_handle(&self) -> TextureHandle {
        self.white
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
        let h = Self::insert_rgba(device, queue, &data, &mut self.textures);
        self.path_to_handle.insert(path.to_string(), h);
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
