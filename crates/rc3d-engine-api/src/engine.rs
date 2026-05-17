use std::path::Path;

use rc3d_core::math::Vec3;
use rc3d_core::{DisplayMode, EngineResult, NodeId};
use rc3d_render::render_action::DrawCall;
use rc3d_render::viewport::{LayoutMode, ViewportLayout};
use rc3d_render::{AdaptiveControl, FrameStats, Renderer};
use rc3d_scene::SceneGraph;

use crate::background::BackgroundSettings;
use crate::camera::CameraController;
use crate::fps_tracker::FpsTracker;
use crate::viewport::ViewportCameraSet;
use crate::world::World;

/// The main runtime facade: creates and manages all engine subsystems.
///
/// Users create an `Engine` with a window, load scenes, adjust settings,
/// and call `render()` each frame. The engine owns a [`World`], an optional
/// [`Renderer`], a [`CameraController`], and per-frame state trackers.
pub struct Engine {
    pub world: World,
    pub renderer: Option<Renderer>,
    pub controller: CameraController,
    pub viewport_cameras: ViewportCameraSet,
    pub fps: FpsTracker,

    /// When true, requests a redraw on every `AboutToWait` event, even when
    /// the scene is idle (no user interaction). Set to `true` for animated
    /// scenes or when polling external data. Default: `false`.
    pub continuous_redraw: bool,

    /// Adaptive quality control mode (Disabled, Locked, Dynamic).
    pub adaptive_control: AdaptiveControl,

    /// Optional callback that returns a string to display as an on-screen HUD
    /// overlay. The string is rendered at the top-left of each viewport.
    pub hud_text_hook: Option<Box<dyn Fn() -> String>>,

    /// Optional callback invoked by [`render`] just before the draw calls are
    /// submitted to the GPU. Use this to modify renderer state each frame
    /// (e.g. to inject custom uniforms or toggle debug overlays).
    pub pre_render_hook: Option<Box<dyn FnMut(&mut Renderer)>>,

    /// Optional callback invoked when the user clicks on geometry in the
    /// scene. Receives the scene graph, the picked NodeId, and the
    /// world-space intersection point.
    pub on_pick: Option<Box<dyn FnMut(&mut SceneGraph, NodeId, Vec3)>>,

    /// Optional keyboard event hook. Receives the physical key. Return `true`
    /// to request a redraw after handling the keypress.
    pub panel_overlay_key_hook: Option<Box<dyn Fn(winit::keyboard::PhysicalKey) -> bool>>,

    /// Optional mouse click hook for HUD overlay interaction.
    /// Receives (x, y, window_width, window_height). Return `true` to request
    /// a redraw after handling the click.
    pub panel_overlay_mouse_hook: Option<Box<dyn Fn(f32, f32, u32, u32) -> bool>>,
}

impl Engine {
    /// Create a new engine for the given window.
    ///
    /// Initializes the wgpu renderer (async via `pollster::block_on`),
    /// creates an empty scene graph, and sets up default camera and
    /// viewport state.
    pub fn new(window: &winit::window::Window) -> Self {
        let renderer = pollster::block_on(Renderer::new(window));
        let graph = SceneGraph::new();
        let world = World::new(graph);
        let controller = CameraController::new(Vec3::new(0.0, 0.0, 0.0), 5.0);
        Self {
            world,
            renderer: Some(renderer),
            controller,
            viewport_cameras: ViewportCameraSet::default(),
            fps: FpsTracker::new(120),
            continuous_redraw: false,
            adaptive_control: AdaptiveControl::Disabled,
            hud_text_hook: None,
            pre_render_hook: None,
            on_pick: None,
            panel_overlay_key_hook: None,
            panel_overlay_mouse_hook: None,
        }
    }

    /// Set the global display mode (Shaded, Flat, Wireframe, etc.).
    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        if let Some(ref mut r) = self.renderer {
            r.set_display_mode(mode);
        }
    }

    /// Configure the background (solid color, gradient, or HDR image).
    pub fn set_background(&mut self, bg: BackgroundSettings) {
        if let Some(ref mut r) = self.renderer {
            r.set_background(bg.into());
        }
    }

    /// Update post-processing effect parameters (vignette, chromatic aberration,
    /// bloom strength, film grain).
    pub fn set_post_effects(&mut self, vignette: f32, chromatic: f32, bloom: f32, grain: f32) {
        if let Some(ref mut r) = self.renderer {
            r.set_post_effect_params(vignette, chromatic, bloom, grain);
        }
    }

    /// Set the adaptive quality control mode.
    ///
    /// `AdaptiveControl::Disabled` locks quality at High.
    /// `AdaptiveControl::Dynamic { allow_downgrade: true }` enables automatic
    /// quality downgrades based on frame time.
    pub fn set_adaptive_quality(&mut self, mode: AdaptiveControl) {
        self.adaptive_control = mode;
    }

    /// Replace the current scene graph.
    ///
    /// This replaces the entire scene. To merge an imported scene into the
    /// existing graph, use [`import`](Self::import) instead.
    pub fn load_scene(&mut self, graph: SceneGraph) {
        self.world.graph = graph;
        self.world.collector.invalidate_mesh_cache();
        self.world.invalidate_caches(self.renderer.as_mut().unwrap());
    }

    /// Mutable access to the scene graph.
    pub fn scene_mut(&mut self) -> &mut SceneGraph {
        &mut self.world.graph
    }

    /// Immutable access to the scene graph.
    pub fn scene(&self) -> &SceneGraph {
        &self.world.graph
    }

    /// Import a 3D file (STL, OBJ, glTF/GLB, FBX, IV) into the scene graph.
    ///
    /// The imported geometry is wrapped in a new `Separator` node at the root.
    /// Returns the [`NodeId`] of that separator.
    pub fn import(&mut self, path: impl AsRef<Path>) -> EngineResult<NodeId> {
        crate::import::import_file(&mut self.world.graph, path)
    }

    /// Mutable access to the camera controller (orbit/pan/zoom).
    pub fn camera_mut(&mut self) -> &mut CameraController {
        &mut self.controller
    }

    /// Immutable access to the camera controller.
    pub fn camera(&self) -> &CameraController {
        &self.controller
    }

    /// Set the viewport layout mode (Single, Quad, LeftRight, TopBottom).
    ///
    /// Rebuilds the viewports to match the surface size.
    pub fn set_layout_mode(&mut self, mode: LayoutMode) {
        let renderer = self.renderer.as_mut().expect("renderer not initialized");
        let (w, h) = (renderer.config.width, renderer.config.height);
        let layout = renderer.viewport_layout_mut();
        layout.layout_mode = mode;
        layout.rebuild(w, h);
        // Re-sync camera viewport ids after the layout rebuild re-allocates them
        self.viewport_cameras
            .remap_viewport_ids_from_layout(layout);
    }

    /// Mutable access to the viewport layout (for splitter drag, quad splits, etc.).
    pub fn viewport_layout_mut(&mut self) -> &mut ViewportLayout {
        self.renderer
            .as_mut()
            .expect("renderer not initialized")
            .viewport_layout_mut()
    }

    /// Render a single frame.
    ///
    /// Evaluates engines, resets the collector, updates camera nodes from
    /// controller state, traverses the scene graph, and submits draw calls
    /// to the GPU.
    pub fn render(&mut self) -> FrameStats {
        // Extract pre-render hook to avoid borrow conflict with renderer
        let mut pre_hook = self.pre_render_hook.take();
        let renderer = self.renderer.as_mut().expect("renderer not initialized");

        // 1. Reapply CAD tier constraints
        renderer.reapply_cad_tier_constraints();

        // 2. Evaluate engines (animation, simulation, etc.)
        self.world.evaluate_engines();

        // 3. Reset collector for this frame
        let dm = renderer.global_display_mode;
        self.world.reset_collector(dm);

        // 4. Update camera nodes from controller state
        let layout = renderer.viewport_layout();
        self.viewport_cameras.update_all(&mut self.world.graph, &layout);

        // 5. Traverse scene graph to populate draw calls
        self.world.traverse_all_roots();

        // 6. Send effect commands to renderer
        let effect_cmds = std::mem::replace(
            &mut self.world.collector.effect_commands,
            Default::default(),
        );
        renderer.set_effect_commands(effect_cmds);

        // 7. Call pre-render hook (before GPU submission)
        if let Some(ref mut h) = pre_hook {
            h(renderer);
        }

        // 8. Render
        let cache = &self.world.cached_draw_calls;
        let dc: &[DrawCall] = if cache.is_empty() {
            &self.world.collector.draw_calls
        } else {
            cache
        };
        let stats = renderer.render_draw_calls(dc, &self.world.graph);

        // 9. Report frame time for adaptive quality controller
        renderer.report_frame_time_ms(stats.frame_time_ms as f32, self.adaptive_control);

        // 10. Restore pre-render hook and track FPS
        self.pre_render_hook = pre_hook;
        self.fps.push(stats.frame_time_ms as f32);

        stats
    }

    /// Handle window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        if let Some(ref mut r) = self.renderer {
            r.resize(width, height);
        }
    }

    /// Access the wgpu device (for external rendering integration).
    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.renderer.as_ref().expect("renderer not initialized").device
    }

    /// Access the wgpu queue.
    pub fn wgpu_queue(&self) -> &wgpu::Queue {
        &self.renderer.as_ref().expect("renderer not initialized").queue
    }

    /// Current surface texture format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.renderer
            .as_ref()
            .expect("renderer not initialized")
            .surface_format()
    }

    /// Return the currently collected draw calls (for inspection or external rendering).
    pub fn draw_calls(&self) -> &[DrawCall] {
        if self.world.cached_draw_calls.is_empty() {
            &self.world.collector.draw_calls
        } else {
            &self.world.cached_draw_calls
        }
    }

    /// Mutable access to the world for advanced use.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

#[cfg(test)]
mod tests {
    use super::World;
    use crate::camera::CameraController;
    use crate::fps_tracker::FpsTracker;
    use crate::import;
    use rc3d_core::math::Vec3;
    use rc3d_core::EngineError;
    use rc3d_scene::node_data::{NodeData, SeparatorNode};
    use rc3d_scene::SceneGraph;
    use std::path::Path;

    // ----------------------------------------------------------------
    // World tests
    // ----------------------------------------------------------------

    #[test]
    fn world_construction_empty_graph() {
        let graph = SceneGraph::new();
        let world = World::new(graph);
        assert!(world.graph.roots().is_empty());
        assert!(world.engines.is_none());
    }

    #[test]
    fn world_load_scene_replaces_graph() {
        let graph1 = SceneGraph::new();
        let mut world = World::new(graph1);

        let mut graph2 = SceneGraph::new();
        graph2.add_root(NodeData::Separator(SeparatorNode));

        // Simulates the effect of Engine::load_scene
        world.graph = graph2;
        assert_eq!(world.graph.roots().len(), 1);
    }

    #[test]
    fn world_scene_mut_returns_mutable_graph() {
        let graph = SceneGraph::new();
        let mut world = World::new(graph);
        let g: &mut SceneGraph = &mut world.graph;

        let root = g.add_root(NodeData::Separator(SeparatorNode));
        assert_eq!(g.roots().len(), 1);
        assert!(g.get(root).is_some());
    }

    #[test]
    fn world_scene_returns_immutable_graph() {
        let graph = SceneGraph::new();
        let world = World::new(graph);
        let g: &SceneGraph = &world.graph;
        assert!(g.roots().is_empty());
    }

    // ----------------------------------------------------------------
    // Import error-path tests
    // ----------------------------------------------------------------

    #[test]
    fn import_unknown_extension_returns_error() {
        let mut graph = SceneGraph::new();
        let result = import::import_file(&mut graph, Path::new("model.xyz"));
        assert!(result.is_err());
        match result.unwrap_err() {
            EngineError::Parse(msg) => {
                assert!(
                    msg.contains("Unknown format") || msg.contains("xyz"),
                    "expected unknown-format error, got: {msg}"
                );
            }
            other => panic!("expected EngineError::Parse, got: {other}"),
        }
    }

    #[test]
    fn import_empty_path_returns_error() {
        let mut graph = SceneGraph::new();
        let result = import::import_file(&mut graph, Path::new(""));
        assert!(result.is_err());
    }

    // ----------------------------------------------------------------
    // CameraController defaults
    // ----------------------------------------------------------------

    #[test]
    fn camera_controller_defaults() {
        let target = Vec3::new(1.0, 2.0, 3.0);
        let distance = 10.0;
        let cam = CameraController::new(target, distance);

        assert_eq!(cam.target, target);
        assert_eq!(cam.distance, distance);
        assert_eq!(cam.yaw, 0.0);
        assert_eq!(cam.pitch, 0.4);
        assert_eq!(cam.up, Vec3::Y);
        assert!(!cam.middle_orbit_held);
        assert!(!cam.left_orbit_held);
        assert!(!cam.panning);
        assert!(!cam.walk_mode);
        assert!(cam.bookmarks.iter().all(|b| b.is_none()));
        // position_changed starts true so the first frame always traverses
        assert!(cam.position_changed.get());
    }

    // ----------------------------------------------------------------
    // FpsTracker tests
    // ----------------------------------------------------------------

    #[test]
    fn fps_tracker_initial_state() {
        let tracker = FpsTracker::new(120);
        assert_eq!(tracker.average_frame_ms(), 0.0);
        assert_eq!(tracker.fps(), 0.0);
        assert_eq!(tracker.smoothed_fps(), 0.0);
    }

    #[test]
    fn fps_tracker_push_updates_metrics() {
        let mut tracker = FpsTracker::new(120);
        let frame_ms: f32 = 16.67; // ~60 FPS
        tracker.push(frame_ms);
        assert!(tracker.average_frame_ms() > 0.0);
        assert!(tracker.fps() > 0.0);
        assert!(tracker.smoothed_fps() > 0.0);
    }
}
