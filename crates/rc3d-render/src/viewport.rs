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

/// Manages viewport rect computation and active viewport tracking.
pub struct ViewportLayout {
    pub viewports: Vec<Viewport>,
    pub layout_mode: LayoutMode,
    pub active_id: ViewportId,
    next_id: u32,
}

impl ViewportLayout {
    pub fn new() -> Self {
        Self {
            viewports: Vec::new(),
            layout_mode: LayoutMode::Single,
            active_id: ViewportId(0),
            next_id: 0,
        }
    }

    fn alloc_id(&mut self) -> ViewportId {
        let id = ViewportId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Rebuild viewports for the current layout mode at the given surface size.
    pub fn rebuild(&mut self, surface_width: u32, surface_height: u32) {
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
                let hw = surface_width / 2;
                let hh = surface_height / 2;
                let rw = (surface_width - hw).max(1);
                let rh = (surface_height - hh).max(1);
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
                let hw = surface_width / 2;
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
                    rect: ViewportRect { x: hw, y: 0, width: (surface_width - hw).max(1), height: surface_height },
                    camera_node: None,
                    projection_type: ProjectionType::Perspective,
                    is_active: false,
                });
            }
            LayoutMode::TopBottom => {
                let hh = surface_height / 2;
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
                    rect: ViewportRect { x: 0, y: hh, width: surface_width, height: (surface_height - hh).max(1) },
                    camera_node: None,
                    projection_type: ProjectionType::Perspective,
                    is_active: false,
                });
            }
        }

        // Ensure active_id points to a real viewport
        let active_exists = self.viewports.iter().any(|v| v.id == self.active_id);
        if !active_exists {
            if let Some(first) = self.viewports.first() {
                self.active_id = first.id;
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
