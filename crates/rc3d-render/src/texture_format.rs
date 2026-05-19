/// GPU texture format negotiation and mipmap generation utilities.
///
/// Provides:
/// - Mipmap auto-generation from loaded image data
/// - Compressed format detection (BC7/BC5/BC4) for supported platforms
/// - Optimal format selection based on texture type and device capabilities
///
/// GPU-ready compressed texture format variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressedFormat {
    /// BC7 RGBA (high quality, 8bpp)
    Bc7RgbaUnorm,
    /// BC5 RG (normal maps, 8bpp)
    Bc5RgUnorm,
    /// BC4 R (single channel, 4bpp)
    Bc4RUnorm,
    /// BC3 RGBA (DXT5, 8bpp)
    Bc3RgbaUnorm,
    /// Uncompressed fallback
    Rgba8Unorm,
}

impl CompressedFormat {
    /// Convert to wgpu TextureFormat.
    pub fn to_wgpu(&self) -> wgpu::TextureFormat {
        match self {
            Self::Bc7RgbaUnorm => wgpu::TextureFormat::Bc7RgbaUnorm,
            Self::Bc5RgUnorm => wgpu::TextureFormat::Bc5RgUnorm,
            Self::Bc4RUnorm => wgpu::TextureFormat::Bc4RUnorm,
            Self::Bc3RgbaUnorm => wgpu::TextureFormat::Bc3RgbaUnorm,
            Self::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        }
    }

    /// Check if the device supports this compressed format.
    pub fn supported(&self, adapter: &wgpu::Adapter) -> bool {
        let features = adapter.features();
        match self {
            Self::Bc7RgbaUnorm | Self::Bc5RgUnorm | Self::Bc4RUnorm | Self::Bc3RgbaUnorm => {
                features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
            }
            Self::Rgba8Unorm => true,
        }
    }

    /// Select the best format for a given texture type.
    pub fn best_for_albedo(adapter: &wgpu::Adapter) -> Self {
        if Self::Bc7RgbaUnorm.supported(adapter) {
            Self::Bc7RgbaUnorm
        } else if Self::Bc3RgbaUnorm.supported(adapter) {
            Self::Bc3RgbaUnorm
        } else {
            Self::Rgba8Unorm
        }
    }

    pub fn best_for_normal_map(adapter: &wgpu::Adapter) -> Self {
        if Self::Bc5RgUnorm.supported(adapter) {
            Self::Bc5RgUnorm
        } else {
            Self::Rgba8Unorm
        }
    }
}

/// Compute the number of mip levels for given dimensions.
pub fn compute_mip_levels(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height);
    if max_dim <= 1 {
        1
    } else {
        (max_dim as f32).log2().floor() as u32 + 1
    }
}

/// Downsample an RGBA8 image to half resolution for mipmap generation.
/// Simple 2x2 box filter.
pub fn downsample_rgba8(
    src: &[u8],
    src_width: u32,
    src_height: u32,
) -> Vec<u8> {
    let dst_width = (src_width / 2).max(1);
    let dst_height = (src_height / 2).max(1);
    let mut dst = vec![0u8; (dst_width * dst_height * 4) as usize];

    for y in 0..dst_height {
        for x in 0..dst_width {
            let sx = (x * 2) as usize;
            let sy = (y * 2) as usize;
            let sx1 = (sx + 1).min(src_width as usize - 1);
            let sy1 = (sy + 1).min(src_height as usize - 1);

            let idx00 = (sy * src_width as usize + sx) * 4;
            let idx10 = (sy * src_width as usize + sx1) * 4;
            let idx01 = (sy1 * src_width as usize + sx) * 4;
            let idx11 = (sy1 * src_width as usize + sx1) * 4;

            let di = ((y * dst_width + x) * 4) as usize;
            for c in 0..4 {
                let sum = src[idx00 + c] as u32
                    + src[idx10 + c] as u32
                    + src[idx01 + c] as u32
                    + src[idx11 + c] as u32;
                dst[di + c] = (sum / 4) as u8;
            }
        }
    }

    dst
}

/// Generate a full mipmap chain for an RGBA8 image.
/// Returns (Vec<Vec<u8>>, Vec<(u32, u32)>) where each entry is (data, (width, height)).
pub fn generate_mip_chain_rgba8(
    src: &[u8],
    width: u32,
    height: u32,
    max_levels: u32,
) -> Vec<(Vec<u8>, u32, u32)> {
    let levels = max_levels.min(compute_mip_levels(width, height));
    let mut chain = Vec::with_capacity(levels as usize);
    chain.push((src.to_vec(), width, height));

    let mut current_data = src.to_vec();
    let mut current_w = width;
    let mut current_h = height;

    for _ in 1..levels {
        let next = downsample_rgba8(&current_data, current_w, current_h);
        current_w = (current_w / 2).max(1);
        current_h = (current_h / 2).max(1);
        chain.push((next.clone(), current_w, current_h));
        current_data = next;
    }

    chain
}

/// Upload a texture with full mipmap chain to the GPU.
/// If `generate_mips` is true and the image has only 1 level, mipmaps are auto-generated.
pub fn upload_texture_with_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba_data: &[u8],
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    generate_mips: bool,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let mip_levels = if generate_mips {
        compute_mip_levels(width, height)
    } else {
        1
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    // Upload level 0
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    // Generate and upload mipmaps
    if generate_mips && mip_levels > 1 {
        let chain = generate_mip_chain_rgba8(rgba_data, width, height, mip_levels);
        for (level, (data, w, h)) in chain.iter().enumerate().skip(1) {
            let bytes_per_row = 4 * w;
            // Pad to 256-byte alignment if needed
            let padded_bytes_per_row = bytes_per_row.div_ceil(256) * 256;
            let mut padded = vec![0u8; (padded_bytes_per_row * h) as usize];
            for row in 0..*h as usize {
                let src_start = row * bytes_per_row as usize;
                let dst_start = row * padded_bytes_per_row as usize;
                padded[dst_start..dst_start + bytes_per_row as usize]
                    .copy_from_slice(&data[src_start..src_start + bytes_per_row as usize]);
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &padded,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(*h),
                },
                wgpu::Extent3d {
                    width: *w,
                    height: *h,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mip_levels() {
        assert_eq!(compute_mip_levels(256, 256), 9);
        assert_eq!(compute_mip_levels(1, 1), 1);
        assert_eq!(compute_mip_levels(4, 4), 3);
    }

    #[test]
    fn test_downsample() {
        let src = vec![
            255u8, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 255, 255, 0, 255,
        ];
        let dst = downsample_rgba8(&src, 2, 2);
        assert_eq!(dst.len(), 4);
        // Average of (255+0+0+255)/4 = 127 for each channel
    }
}
