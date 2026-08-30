//! Rasterize annotation labels on their plane tangent (rotated screen baseline).

use std::collections::HashMap;

use glyphon::{
    Attrs, Buffer, Color, ContentType, CustomGlyph, Family, FontSystem, Metrics, RasterizedCustomGlyph,
    Shaping, SwashCache, SwashContent,
};

use crate::render_passes::pass_text::TextDrawCommand;

/// Stable raster cache: reuse glyph IDs and masks across frames (avoids atlas flicker).
pub struct PlaneLabelRasterCache {
    entries: HashMap<PlaneLabelKey, CachedPlaneLabelRaster>,
    next_id: u16,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct PlaneLabelKey {
    text: String,
    size_tenths: u16,
    /// Baseline angle in tenths of a degree (coarse bucket avoids per-frame re-rasterize).
    angle_deci_deg: i16,
}

#[derive(Clone)]
struct CachedPlaneLabelRaster {
    glyph_id: u16,
    width: u16,
    height: u16,
    anchor_x: f32,
    anchor_y: f32,
    raster: PlaneGlyphCacheEntry,
}

impl PlaneLabelRasterCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            next_id: 1,
        }
    }

    fn alloc_id(&mut self) -> u16 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);
        id
    }
}

fn snap_screen_half_pixel(pos: [f32; 2]) -> [f32; 2] {
    [(pos[0] * 2.0).round() / 2.0, (pos[1] * 2.0).round() / 2.0]
}

fn label_cache_key(cmd: &TextDrawCommand) -> PlaneLabelKey {
    PlaneLabelKey {
        text: cmd.string.clone(),
        size_tenths: (cmd.size * 10.0).round().clamp(1.0, u16::MAX as f32) as u16,
        angle_deci_deg: (cmd.baseline_angle_rad.to_degrees() * 10.0).round() as i16,
    }
}

pub(crate) struct PlaneGlyphRaster {
    pub data: Vec<u8>,
    pub width: u16,
    pub height: u16,
    /// Pixel in the raster that maps to `TextDrawCommand::screen_pos` (pre-rotation center).
    pub anchor_x: f32,
    pub anchor_y: f32,
}

/// Cached raster plus dimensions (must match `data.len() == width * height`).
pub(crate) type PlaneGlyphCacheEntry = (u16, u16, RasterizedCustomGlyph);

fn canonical_mask_dimensions(data: &[u8], width: u16, height: u16) -> (u16, u16) {
    let h = height.max(1);
    let expected = width as usize * h as usize;
    if data.len() == expected {
        return (width, h);
    }
    let w = (data.len() / h as usize).max(1).min(u16::MAX as usize) as u16;
    (w, h)
}

fn resize_mask(
    src: &[u8],
    sw: u16,
    sh: u16,
    dw: u16,
    dh: u16,
) -> Vec<u8> {
    let sw = sw.max(1);
    let sh = sh.max(1);
    let mut dst = vec![0u8; dw as usize * dh as usize];
    let copy_w = sw.min(dw) as usize;
    let copy_h = sh.min(dh) as usize;
    for y in 0..copy_h {
        let src_row = &src[y * sw as usize..y * sw as usize + copy_w];
        let dst_row = &mut dst[y * dw as usize..y * dw as usize + copy_w];
        dst_row.copy_from_slice(src_row);
    }
    dst
}

/// Return raster data sized exactly for glyphon's `RasterizeCustomGlyphRequest`.
pub(crate) fn glyph_for_request(
    (sw, sh, glyph): &PlaneGlyphCacheEntry,
    req: &glyphon::RasterizeCustomGlyphRequest,
) -> RasterizedCustomGlyph {
    if req.width == *sw && req.height == *sh {
        return glyph.clone();
    }
    RasterizedCustomGlyph {
        data: resize_mask(&glyph.data, *sw, *sh, req.width, req.height),
        content_type: glyph.content_type,
    }
}

fn blit_swash_mask(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    x: i32,
    y: i32,
    image: &glyphon::SwashImage,
) {
    let gw = image.placement.width as i32;
    let gh = image.placement.height as i32;
    for row in 0..gh {
        for col in 0..gw {
            let src_i = (row * gw + col) as usize;
            if src_i >= image.data.len() {
                continue;
            }
            let alpha = image.data[src_i];
            if alpha == 0 {
                continue;
            }
            let dx = x + col + image.placement.left;
            let dy = y + row + image.placement.top;
            if dx < 0 || dy < 0 {
                continue;
            }
            let (dx, dy) = (dx as u32, dy as u32);
            if dx >= dst_w || dy >= dst_h {
                continue;
            }
            let di = (dy * dst_w + dx) as usize;
            dst[di] = dst[di].saturating_add(alpha);
        }
    }
}

fn rasterize_horizontal_mask(
    font_system: &mut FontSystem,
    swash_cache: &mut SwashCache,
    text: &str,
    size: f32,
) -> Option<PlaneGlyphRaster> {
    let mut buffer = Buffer::new(font_system, Metrics::new(size, size * 1.2));
    buffer.set_size(Some(4096.0), Some(4096.0));
    let attrs = Attrs::new().family(Family::SansSerif);
    buffer.set_text(text, &attrs, Shaping::Advanced, None);
    buffer.shape_until_scroll(font_system, false);

    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for run in buffer.layout_runs() {
        for glyph in run.glyphs.iter() {
            let pg = glyph.physical((0.0, 0.0), 1.0);
            let Some(image) = swash_cache.get_image(font_system, pg.cache_key).as_ref() else {
                continue;
            };
            if image.content != SwashContent::Mask {
                continue;
            }
            let left = pg.x + image.placement.left;
            let top = pg.y + image.placement.top;
            let right = left + image.placement.width as i32;
            let bottom = top + image.placement.height as i32;
            min_x = min_x.min(left);
            min_y = min_y.min(top);
            max_x = max_x.max(right);
            max_y = max_y.max(bottom);
        }
    }
    if min_x == i32::MAX {
        return None;
    }

    let w = (max_x - min_x).max(1) as u32;
    let h = (max_y - min_y).max(1) as u32;
    let mut mask = vec![0u8; (w * h) as usize];
    let offset = (-min_x as f32, -min_y as f32);

    for run in buffer.layout_runs() {
        for glyph in run.glyphs.iter() {
            let pg = glyph.physical(offset, 1.0);
            let Some(image) = swash_cache.get_image(font_system, pg.cache_key).as_ref() else {
                continue;
            };
            if image.content != SwashContent::Mask {
                continue;
            }
            blit_swash_mask(&mut mask, w, h, pg.x, pg.y, image);
        }
    }

    let (width, height) = canonical_mask_dimensions(&mask, w as u16, h as u16);
    Some(PlaneGlyphRaster {
        data: mask,
        width,
        height,
        anchor_x: f32::from(width) * 0.5,
        anchor_y: f32::from(height) * 0.5,
    })
}

fn rotate_mask(
    src: &[u8],
    sw: u32,
    sh: u32,
    angle: f32,
) -> (Vec<u8>, u32, u32, f32, f32) {
    if sw == 0 || sh == 0 {
        return (Vec::new(), 0, 0, 0.0, 0.0);
    }
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let hw = sw as f32 * 0.5;
    let hh = sh as f32 * 0.5;
    let corners = [
        rotate_corner(-hw, -hh, cos_a, sin_a),
        rotate_corner(hw, -hh, cos_a, sin_a),
        rotate_corner(hw, hh, cos_a, sin_a),
        rotate_corner(-hw, hh, cos_a, sin_a),
    ];
    let min_x = corners.iter().map(|c| c.0).fold(f32::INFINITY, f32::min);
    let max_x = corners.iter().map(|c| c.0).fold(f32::NEG_INFINITY, f32::max);
    let min_y = corners.iter().map(|c| c.1).fold(f32::INFINITY, f32::min);
    let max_y = corners.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
    let dw = (max_x - min_x).ceil().max(1.0) as u32;
    let dh = (max_y - min_y).ceil().max(1.0) as u32;
    let mut dst = vec![0u8; (dw * dh) as usize];
    let dst_cx = min_x + (max_x - min_x) * 0.5;
    let dst_cy = min_y + (max_y - min_y) * 0.5;

    for dy in 0..dh {
        for dx in 0..dw {
            let wx = min_x + dx as f32;
            let wy = min_y + dy as f32;
            let lx = (wx - dst_cx) * cos_a + (wy - dst_cy) * sin_a + hw;
            let ly = -(wx - dst_cx) * sin_a + (wy - dst_cy) * cos_a + hh;
            let sx = lx.floor() as i32;
            let sy = ly.floor() as i32;
            if sx < 0 || sy < 0 || sx >= sw as i32 || sy >= sh as i32 {
                continue;
            }
            let si = (sy as u32 * sw + sx as u32) as usize;
            let a = src[si];
            if a == 0 {
                continue;
            }
            let di = (dy * dw + dx) as usize;
            dst[di] = dst[di].saturating_add(a);
        }
    }
    // Anchor: where the unrotated glyph center lands inside the output bitmap (top-left origin).
    let anchor_x = dst_cx - min_x;
    let anchor_y = dst_cy - min_y;
    (dst, dw, dh, anchor_x, anchor_y)
}

fn rotate_corner(x: f32, y: f32, cos_a: f32, sin_a: f32) -> (f32, f32) {
    (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
}

/// Rasterize label text aligned to the annotation plane tangent (`baseline_angle_rad`).
pub fn rasterize_plane_aligned_label(
    font_system: &mut FontSystem,
    swash_cache: &mut SwashCache,
    cmd: &TextDrawCommand,
) -> Option<PlaneGlyphRaster> {
    if !cmd.plane_aligned || cmd.string.is_empty() {
        return None;
    }
    let horizontal = rasterize_horizontal_mask(font_system, swash_cache, &cmd.string, cmd.size)?;
    let (data, w, h, anchor_x, anchor_y) = rotate_mask(
        &horizontal.data,
        horizontal.width as u32,
        horizontal.height as u32,
        cmd.baseline_angle_rad,
    );
    if w == 0 || h == 0 {
        return None;
    }
    let (width, height) = canonical_mask_dimensions(&data, w as u16, h as u16);
    Some(PlaneGlyphRaster {
        data,
        width,
        height,
        anchor_x,
        anchor_y,
    })
}

/// Build custom glyphs; raster is cached by text/size/angle, only screen position updates each frame.
/// `raster_by_id` is append-only: glyphon requires `Some` for every id it has seen before.
pub fn build_plane_label_custom_glyphs(
    cache: &mut PlaneLabelRasterCache,
    raster_by_id: &mut HashMap<u16, PlaneGlyphCacheEntry>,
    font_system: &mut FontSystem,
    swash_cache: &mut SwashCache,
    labels: &[TextDrawCommand],
) -> Vec<CustomGlyph> {
    let mut glyphs = Vec::new();

    for cmd in labels.iter().filter(|c| c.plane_aligned) {
        let key = label_cache_key(cmd);
        let cached = if let Some(c) = cache.entries.get(&key) {
            c.clone()
        } else {
            let Some(img) = rasterize_plane_aligned_label(font_system, swash_cache, cmd) else {
                continue;
            };
            let glyph_id = cache.alloc_id();
            let entry = CachedPlaneLabelRaster {
                glyph_id,
                width: img.width,
                height: img.height,
                anchor_x: img.anchor_x,
                anchor_y: img.anchor_y,
                raster: (
                    img.width,
                    img.height,
                    RasterizedCustomGlyph {
                        data: img.data,
                        content_type: ContentType::Mask,
                    },
                ),
            };
            cache.entries.insert(key, entry.clone());
            entry
        };

        let screen = snap_screen_half_pixel(cmd.screen_pos);
        glyphs.push(CustomGlyph {
            id: cached.glyph_id,
            left: screen[0] - cached.anchor_x,
            top: screen[1] - cached.anchor_y,
            width: f32::from(cached.width),
            height: f32::from(cached.height),
            color: Some(Color::rgba(
                (cmd.color[0] * 255.0) as u8,
                (cmd.color[1] * 255.0) as u8,
                (cmd.color[2] * 255.0) as u8,
                (cmd.color[3] * 255.0) as u8,
            )),
            snap_to_physical_pixel: false,
            metadata: cmd.clip_depth_ndc.to_bits() as usize,
        });
        raster_by_id.insert(cached.glyph_id, cached.raster);
    }

    glyphs
}

/// Fallback when the atlas asks for an id not in the current frame list (must not return `None`).
pub fn fallback_custom_glyph_raster(req: &glyphon::RasterizeCustomGlyphRequest) -> RasterizedCustomGlyph {
    RasterizedCustomGlyph {
        data: vec![0u8; req.width as usize * req.height as usize],
        content_type: ContentType::Mask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_mask_zero_angle_anchor_is_bitmap_center() {
        let w = 100u32;
        let h = 20u32;
        let src = vec![255u8; (w * h) as usize];
        let (_, dw, dh, ax, ay) = rotate_mask(&src, w, h, 0.0);
        assert_eq!(dw, w);
        assert_eq!(dh, h);
        assert!((ax - w as f32 * 0.5).abs() < 0.01);
        assert!((ay - h as f32 * 0.5).abs() < 0.01);
    }
}
