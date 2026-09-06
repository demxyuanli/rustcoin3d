//! Independent overlay viewports: each owns a [`World`] and a screen rectangle.

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_render::viewport::ViewportRect;
use rc3d_scene::node_data::NodeData;
use rc3d_scene::SceneGraph;

use crate::world::World;

/// Default overlay name for the navigation cube widget.
pub const OVERLAY_NAV_CUBE: &str = "nav_cube";

/// A secondary 3D world composited onto the swapchain after the main film.
pub struct OverlayViewport {
    pub name: String,
    pub world: World,
    pub rect: ViewportRect,
    pub view: Mat4,
    pub projection: Mat4,
    pub camera_pos: Vec3,
    pub orthographic: bool,
    pub ortho_half: f32,
    pub eye_distance: f32,
    pub follow_main_camera: bool,
    pub enabled: bool,
    pub clear_color: [f32; 4],
    pub light_id: Option<NodeId>,
}

impl OverlayViewport {
    pub fn new(name: impl Into<String>, graph: SceneGraph) -> Self {
        Self {
            name: name.into(),
            world: World::new(graph),
            rect: ViewportRect::default(),
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
            camera_pos: Vec3::new(0.0, 0.0, 4.0),
            orthographic: true,
            ortho_half: 1.55,
            eye_distance: 4.0,
            follow_main_camera: true,
            enabled: true,
            clear_color: [0.0, 0.0, 0.0, 0.0],
            light_id: None,
        }
    }

    /// Place this overlay at the top-right of `region` (window pixels, Y-down).
    pub fn layout_top_right_of(&mut self, region: ViewportRect, size_px: u32, margin_px: u32) {
        let size = size_px.max(1);
        let margin = margin_px;
        let x = region.x + region.width.saturating_sub(margin.saturating_add(size));
        let y = region.y.saturating_add(margin);
        self.rect = ViewportRect {
            x,
            y,
            width: size,
            height: size,
        }
        .clamped_to(
            region.x.saturating_add(region.width).max(1),
            region.y.saturating_add(region.height).max(1),
        );
    }

    /// Point the overlay camera along `from` (world direction from origin toward the eye).
    pub fn aim(&mut self, from: Vec3, up: Vec3) {
        let mut dir = from.normalize_or_zero();
        if dir.length_squared() < 1.0e-8 {
            dir = Vec3::Z;
        }
        let mut up = up.normalize_or_zero();
        if up.length_squared() < 1.0e-8 || dir.dot(up).abs() > 0.999 {
            up = if dir.y.abs() > 0.5 { -Vec3::Z } else { Vec3::Y };
        }
        self.camera_pos = dir * self.eye_distance;
        self.view = Mat4::look_at_rh(self.camera_pos, Vec3::ZERO, up);
        let h = self.ortho_half.max(0.1);
        if self.orthographic {
            self.projection = Mat4::orthographic_rh(-h, h, -h, h, 0.5, 32.0);
        } else {
            self.projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.5, 32.0);
        }
        if let Some(id) = self.light_id {
            if let Some(entry) = self.world.graph.get_mut(id) {
                if let NodeData::DirectionalLight(light) = &mut entry.data {
                    light.direction = -dir;
                }
            }
        }
    }
}
