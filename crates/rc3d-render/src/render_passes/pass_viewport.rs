//! Viewport border overlay rendering.
//!
//! Draws viewport split borders and the active-viewport highlight border
//! using the existing edge overlay pipeline.

use crate::vertex::LineVertex;

/// Generate line vertices for a viewport border rectangle.
fn border_lines(x: u32, y: u32, w: u32, h: u32) -> Vec<LineVertex> {
    let x0 = x as f32;
    let y0 = y as f32;
    let x1 = (x + w - 1) as f32;
    let y1 = (y + h - 1) as f32;
    let z = 0.0;
    vec![
        // Top edge
        LineVertex { position: [x0, y0, z] },
        LineVertex { position: [x1, y0, z] },
        // Right edge
        LineVertex { position: [x1, y0, z] },
        LineVertex { position: [x1, y1, z] },
        // Bottom edge
        LineVertex { position: [x1, y1, z] },
        LineVertex { position: [x0, y1, z] },
        // Left edge
        LineVertex { position: [x0, y1, z] },
        LineVertex { position: [x0, y0, z] },
    ]
}

/// Line vertices for all viewport borders + active-viewport highlight.
pub struct ViewportBorderGeometry {
    pub split_lines: Vec<LineVertex>,
    pub active_lines: Vec<LineVertex>,
}

impl ViewportBorderGeometry {
    /// Build border geometry from the current viewport layout.
    pub fn build(layout: &crate::viewport::ViewportLayout, surface_w: u32, surface_h: u32) -> Self {
        let mut split_lines = Vec::new();
        let mut active_lines = Vec::new();

        // Draw borders between viewports (internal split lines)
        let vp_count = layout.viewports.len();
        if vp_count > 1 {
            for vp in &layout.viewports {
                // Right border (except rightmost viewports)
                let right = vp.rect.x + vp.rect.width;
                if right < surface_w && !layout.viewports.iter().any(|v| v.rect.x == right) {
                    split_lines.extend(border_lines(right.saturating_sub(1), 0, 4, surface_h));
                }
                // Bottom border (except bottommost viewports)
                let bottom = vp.rect.y + vp.rect.height;
                if bottom < surface_h && !layout.viewports.iter().any(|v| v.rect.y == bottom) {
                    split_lines.extend(border_lines(0, bottom.saturating_sub(1), surface_w, 4));
                }
            }
        }

        // Active viewport highlight
        if let Some(active) = layout.active() {
            active_lines.extend(border_lines(
                active.rect.x, active.rect.y,
                active.rect.width, active.rect.height,
            ));
        }

        ViewportBorderGeometry { split_lines, active_lines }
    }
}
