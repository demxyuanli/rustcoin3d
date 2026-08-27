//! Shared utilities for render pass helpers.

use crate::renderer::Renderer;
use crate::render_passes::FramePresentation;

/// Result of acquiring a surface for rendering.
///
/// The closure `f` receives the raw texture pointer at the moment of acquisition.
/// Returns whatever the closure produces along with the acquired swapchain (if any)
/// and the effective rendering dimensions.
pub(super) fn acquire_surface<R>(
    renderer: &Renderer,
    presentation: &FramePresentation<'_>,
    f: impl FnOnce(*const wgpu::Texture, u32, u32) -> R,
) -> (R, Option<(wgpu::SurfaceTexture, wgpu::TextureView)>, u32, u32) {
    match presentation {
        FramePresentation::Swapchain => {
            let acquired = renderer.surface.get_current_texture();
            match acquired {
                Ok(output) => {
                    let tex_ptr = std::ptr::from_ref(&output.texture);
                    let v = output
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let result = f(tex_ptr, renderer.config.width, renderer.config.height);
                    (result, Some((output, v)), renderer.config.width, renderer.config.height)
                }
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    renderer.surface.configure(&renderer.device, &renderer.config);
                    // Caller must handle the None return — typically by returning FrameStats::default()
                    let result = f(std::ptr::null(), 0, 0);
                    (result, None, 0, 0)
                }
                Err(_) => {
                    let result = f(std::ptr::null(), 0, 0);
                    (result, None, 0, 0)
                }
            }
        }
        FramePresentation::OffscreenSurface {
            output_texture,
            width_px,
            height_px,
            ..
        } => {
            let tex_ptr = (*output_texture) as *const wgpu::Texture;
            let result = f(tex_ptr, *width_px, *height_px);
            (result, None, *width_px, *height_px)
        }
    }
}
