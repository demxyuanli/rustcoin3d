//! Multi-viewport layout management for industrial visualization editors.
//!
//! Supports Single, Quad, LeftRight, and TopBottom layouts.
//! Each viewport binds a camera node and a screen-space rectangle.

use rc3d_core::NodeId;

/// Unique identifier for a viewport within a layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewportId(pub u32);

/// Pixel-space rectangle for a viewport within the surface.
#[derive(Clone, Copy, Debug)]
pub struct ViewportRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ViewportRect {
    pub fn contains(&self, px: f32, py: f32) -> bool {
        if px < 0.0 || py < 0.0 {
            return false;
        }
        let x = px as u32;
        let y = py as u32;
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }
}

/// Projection type for a viewport camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionType {
    Perspective,
    Orthographic,
}

/// A single viewport: screen rect + camera binding.
#[derive(Clone, Debug)]
pub struct Viewport {
    pub id: ViewportId,
    pub name: String,
    pub rect: ViewportRect,
    pub camera_node: Option<NodeId>,
    pub projection_type: ProjectionType,
    pub is_active: bool,
}

/// Layout mode determines the viewport split arrangement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutMode {
    Single,
    Quad,
    LeftRight,
    TopBottom,
}

/// Which split parameter is adjusted when dragging a layout divider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewportSplitAxis {
    /// `quad_h_split`: vertical bar (Quad, LeftRight).
    HorizontalFraction,
    /// `quad_v_split`: horizontal bar (Quad, TopBottom).
    VerticalFraction,
}

/// Pixel half-width for splitter hit-testing (total grab strip ≈ 2× this).
pub const VIEWPORT_SPLITTER_HIT_PX: f32 = 5.0;

/// Manages viewport rect computation and active viewport tracking.
pub struct ViewportLayout {
    pub viewports: Vec<Viewport>,
    pub layout_mode: LayoutMode,
    pub active_id: ViewportId,
    /// Quad layout: vertical split position as fraction of width (0.1 - 0.9).
    pub quad_h_split: f32,
    /// Quad layout: horizontal split position as fraction of height (0.1 - 0.9).
    pub quad_v_split: f32,
    next_id: u32,
}

impl ViewportLayout {
    pub fn new() -> Self {
        Self {
            viewports: Vec::new(),
            layout_mode: LayoutMode::Single,
            active_id: ViewportId(0),
            quad_h_split: 0.5,
            quad_v_split: 0.5,
            next_id: 0,
        }
    }

    pub fn set_quad_splits(&mut self, h: f32, v: f32) {
        self.quad_h_split = h.clamp(0.1, 0.9);
        self.quad_v_split = v.clamp(0.1, 0.9);
    }

    fn alloc_id(&mut self) -> ViewportId {
        let id = ViewportId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Rebuild viewports for the current layout mode at the given surface size.
    pub fn rebuild(&mut self, surface_width: u32, surface_height: u32) {
        let preserved_active_idx = self
            .viewports
            .iter()
            .position(|v| v.id == self.active_id);
        self.next_id = 0;
        self.viewports.clear();

        match self.layout_mode {
            LayoutMode::Single => {
                let id = self.alloc_id();
                self.viewports.push(Viewport {
                    id,
                    name: "Perspective".into(),
                    rect: ViewportRect { x: 0, y: 0, width: surface_width, height: surface_height },
                    camera_node: None,
                    projection_type: ProjectionType::Perspective,
                    is_active: true,
                });
            }
            LayoutMode::Quad => {
                let hw = ((surface_width as f32) * self.quad_h_split) as u32;
                let hh = ((surface_height as f32) * self.quad_v_split) as u32;
                let rw = (surface_width.saturating_sub(hw)).max(1);
                let rh = (surface_height.saturating_sub(hh)).max(1);
                let vps = [
                    ("Top",     hw, 0,  rw, hh, ProjectionType::Orthographic),
                    ("Front",   0,  hh, hw, rh, ProjectionType::Orthographic),
                    ("Right",   hw, hh, rw, rh, ProjectionType::Orthographic),
                    ("Persp",   0,  0,  hw, hh, ProjectionType::Perspective),
                ];
                for &(name, x, y, w, h, pt) in &vps {
                    let id = self.alloc_id();
                    let is_active = name == "Persp";
                    self.viewports.push(Viewport {
                        id,
                        name: name.into(),
                        rect: ViewportRect { x, y, width: w.max(1), height: h.max(1) },
                        camera_node: None,
                        projection_type: pt,
                        is_active,
                    });
                }
                // Ensure at least one is active
                if !self.viewports.iter().any(|v| v.is_active) {
                    self.viewports[0].is_active = true;
                    self.active_id = self.viewports[0].id;
                }
            }
            LayoutMode::LeftRight => {
                let hw = ((surface_width as f32) * self.quad_h_split) as u32;
                let rw = surface_width.saturating_sub(hw).max(1);
                let id0 = self.alloc_id();
                self.viewports.push(Viewport {
                    id: id0,
                    name: "Left".into(),
                    rect: ViewportRect { x: 0, y: 0, width: hw.max(1), height: surface_height },
                    camera_node: None,
                    projection_type: ProjectionType::Perspective,
                    is_active: true,
                });
                let id1 = self.alloc_id();
                self.viewports.push(Viewport {
                    id: id1,
                    name: "Right".into(),
                    rect: ViewportRect {
                        x: hw,
                        y: 0,
                        width: rw,
                        height: surface_height,
                    },
                    camera_node: None,
                    projection_type: ProjectionType::Perspective,
                    is_active: false,
                });
            }
            LayoutMode::TopBottom => {
                let hh = ((surface_height as f32) * self.quad_v_split) as u32;
                let bh = surface_height.saturating_sub(hh).max(1);
                let id0 = self.alloc_id();
                self.viewports.push(Viewport {
                    id: id0,
                    name: "Top".into(),
                    rect: ViewportRect { x: 0, y: 0, width: surface_width, height: hh.max(1) },
                    camera_node: None,
                    projection_type: ProjectionType::Perspective,
                    is_active: true,
                });
                let id1 = self.alloc_id();
                self.viewports.push(Viewport {
                    id: id1,
                    name: "Bottom".into(),
                    rect: ViewportRect {
                        x: 0,
                        y: hh,
                        width: surface_width,
                        height: bh,
                    },
                    camera_node: None,
                    projection_type: ProjectionType::Perspective,
                    is_active: false,
                });
            }
        }

        if let Some(i) = preserved_active_idx {
            if i < self.viewports.len() {
                self.set_active(self.viewports[i].id);
            } else if let Some(first) = self.viewports.first() {
                self.set_active(first.id);
            }
        } else {
            let active_exists = self.viewports.iter().any(|v| v.id == self.active_id);
            if !active_exists {
                if let Some(first) = self.viewports.first() {
                    self.set_active(first.id);
                }
            }
        }
    }

    /// Hit-test a layout splitter near `(px, py)` in surface pixels.
    pub fn splitter_hit(
        &self,
        px: f32,
        py: f32,
        surface_width: u32,
        surface_height: u32,
    ) -> Option<ViewportSplitAxis> {
        let w = surface_width as f32;
        let h = surface_height as f32;
        if w <= 0.0 || h <= 0.0 {
            return None;
        }
        match self.layout_mode {
            LayoutMode::Single => None,
            LayoutMode::Quad => {
                let vx = w * self.quad_h_split;
                let hy = h * self.quad_v_split;
                let dx = (px - vx).abs();
                let dy = (py - hy).abs();
                let on_v = dx <= VIEWPORT_SPLITTER_HIT_PX;
                let on_h = dy <= VIEWPORT_SPLITTER_HIT_PX;
                match (on_v, on_h) {
                    (true, true) => Some(if dx <= dy {
                        ViewportSplitAxis::HorizontalFraction
                    } else {
                        ViewportSplitAxis::VerticalFraction
                    }),
                    (true, false) => Some(ViewportSplitAxis::HorizontalFraction),
                    (false, true) => Some(ViewportSplitAxis::VerticalFraction),
                    _ => None,
                }
            }
            LayoutMode::LeftRight => {
                let vx = w * self.quad_h_split;
                if (px - vx).abs() <= VIEWPORT_SPLITTER_HIT_PX {
                    Some(ViewportSplitAxis::HorizontalFraction)
                } else {
                    None
                }
            }
            LayoutMode::TopBottom => {
                let hy = h * self.quad_v_split;
                if (py - hy).abs() <= VIEWPORT_SPLITTER_HIT_PX {
                    Some(ViewportSplitAxis::VerticalFraction)
                } else {
                    None
                }
            }
        }
    }

    /// Update split fractions from cursor position during a drag.
    pub fn apply_split_drag(
        &mut self,
        axis: ViewportSplitAxis,
        px: f32,
        py: f32,
        surface_width: u32,
        surface_height: u32,
    ) {
        let w = surface_width.max(1) as f32;
        let h = surface_height.max(1) as f32;
        match axis {
            ViewportSplitAxis::HorizontalFraction => {
                self.quad_h_split = (px / w).clamp(0.1, 0.9);
            }
            ViewportSplitAxis::VerticalFraction => {
                self.quad_v_split = (py / h).clamp(0.1, 0.9);
            }
        }
    }

    /// Set the active viewport by id.
    pub fn set_active(&mut self, id: ViewportId) {
        for vp in &mut self.viewports {
            vp.is_active = vp.id == id;
        }
        self.active_id = id;
    }

    /// Cycle to the next layout mode.
    pub fn cycle_layout(&mut self, surface_width: u32, surface_height: u32) {
        self.layout_mode = match self.layout_mode {
            LayoutMode::Single => LayoutMode::Quad,
            LayoutMode::Quad => LayoutMode::LeftRight,
            LayoutMode::LeftRight => LayoutMode::TopBottom,
            LayoutMode::TopBottom => LayoutMode::Single,
        };
        self.rebuild(surface_width, surface_height);
    }

    /// Cycle the active viewport to the next one.
    pub fn cycle_active(&mut self) {
        if self.viewports.is_empty() {
            return;
        }
        let pos = self.viewports.iter().position(|v| v.id == self.active_id).unwrap_or(0);
        let next = (pos + 1) % self.viewports.len();
        self.set_active(self.viewports[next].id);
    }

    /// Find which viewport contains a surface pixel coordinate.
    pub fn viewport_at(&self, x: f32, y: f32) -> Option<&Viewport> {
        self.viewports.iter().find(|v| v.rect.contains(x, y))
    }

    /// The currently active viewport.
    pub fn active(&self) -> Option<&Viewport> {
        self.viewports.iter().find(|v| v.is_active)
    }
}

impl Default for ViewportLayout {
    fn default() -> Self {
        let mut layout = Self::new();
        layout.layout_mode = LayoutMode::Single;
        layout
    }
}
