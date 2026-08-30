//! Headless offscreen rendering for screenshots, thumbnails, and print output.

use wgpu::{Extent3d, TextureDescriptor, TextureFormat, TextureUsages, TextureViewDescriptor};

pub struct OffscreenTarget {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl OffscreenTarget {
    pub async fn new(width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .expect("offscreen adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("offscreen device");

        let texture = device.create_texture(&TextureDescriptor {
            label: Some("Offscreen target"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());

        Self {
            device,
            queue,
            texture,
            view,
            width,
            height,
        }
    }

    /// Reuses a [`Device`](wgpu::Device) / [`Queue`](wgpu::Queue) (e.g. from [`Renderer::device`](crate::Renderer::device)) so
    /// the texture is compatible for copy/resolve with the main pass.
    pub fn new_with_device(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("Offscreen target (shared device)"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            device: device.clone(),
            queue: queue.clone(),
            texture,
            view,
            width,
            height,
        }
    }

    /// Read rendered pixels into a CPU buffer.
    pub fn read_pixels(&self) -> Vec<u8> {
        let bpr = ((self.width * 4).saturating_add(255)) & !255;
        let size = (bpr as u64) * (self.height as u64);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Offscreen readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(self.height),
                },
            },
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        if rx.recv().is_ok() {
            let data = buffer.slice(..).get_mapped_range().expect("offscreen map").to_vec();
            buffer.unmap();
            data
        } else {
            Vec::new()
        }
    }
}
