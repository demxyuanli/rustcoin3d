//! Full-window color target feeding the egui viewport `Image` widget.

pub struct CliViewportRt {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub extent: [u32; 2],
}

impl CliViewportRt {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cli editor viewport"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            extent: [width, height],
        }
    }
}

/// Recreate the RT only when the layout size moves by more than this (per axis).
/// Egui layout rounding often toggles ±1px each frame; reallocating every frame causes visible flicker.
const VIEWPORT_RT_SIZE_HYSTERESIS_PX: u32 = 4;

pub fn ensure_viewport_rt(
    slot: &mut Option<CliViewportRt>,
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> bool {
    let width = width.max(1);
    let height = height.max(1);
    let recreated = match slot.as_ref() {
        None => true,
        Some(r) => {
            width.abs_diff(r.extent[0]) > VIEWPORT_RT_SIZE_HYSTERESIS_PX
                || height.abs_diff(r.extent[1]) > VIEWPORT_RT_SIZE_HYSTERESIS_PX
        }
    };
    if recreated {
        *slot = Some(CliViewportRt::new(device, format, width, height));
    }
    recreated
}
