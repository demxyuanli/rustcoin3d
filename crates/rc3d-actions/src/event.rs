//! Scene-graph event types for industrial visualization.
//!
//! Platform-agnostic event types decoupled from winit,
//! allowing event routing through the scene graph via HandleEventAction.

use rc3d_core::math::Mat4;

/// Platform-agnostic event for scene-graph routing.
#[derive(Clone, Debug)]
pub enum Event {
    MouseMove {
        x: f32,
        y: f32,
        dx: f32,
        dy: f32,
    },
    ButtonPress {
        button: u8, // 0=left, 1=middle, 2=right
        x: f32,
        y: f32,
    },
    ButtonRelease {
        button: u8,
        x: f32,
        y: f32,
    },
    KeyPress {
        key: String,
    },
    KeyRelease {
        key: String,
    },
    Scroll {
        dx: f32,
        dy: f32,
    },
    /// Touch / trackpad contact (platform-agnostic).
    Touch {
        x: f32,
        y: f32,
        /// 0=down, 1=move, 2=up
        phase: u8,
    },
}

/// Context passed to HandleEventAction during scene-graph traversal.
#[derive(Clone, Debug)]
pub struct EventContext {
    pub event: Event,
    /// (view_matrix, projection_matrix) for the target viewport.
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
    /// Whether the event has been consumed (stops further propagation).
    pub consumed: bool,
    /// Viewport size in **physical pixels** `(width, height)` for pointer pick rays.
    /// Must match `Event` pointer coordinates, which are **viewport-local** (0..width, 0..height)
    /// relative to the hit viewport rectangle in surface pixels (winit physical space).
    pub pointer_pick_viewport: Option<(f32, f32)>,
}

impl EventContext {
    pub fn new(event: Event, view: Mat4, proj: Mat4) -> Self {
        Self {
            event,
            view_matrix: view,
            projection_matrix: proj,
            consumed: false,
            pointer_pick_viewport: None,
        }
    }

    /// Mark this event as consumed, stopping further propagation.
    pub fn consume(&mut self) {
        self.consumed = true;
    }

    pub fn is_consumed(&self) -> bool {
        self.consumed
    }

    /// Build a world-space ray from cursor position within this event context.
    pub fn pick_ray(&self, screen_x: f32, screen_y: f32, vp_width: f32, vp_height: f32) -> crate::ray_pick::Ray {
        crate::ray_pick::Ray::from_screen_point(
            screen_x, screen_y,
            vp_width, vp_height,
            self.view_matrix,
            self.projection_matrix,
        )
    }
}
