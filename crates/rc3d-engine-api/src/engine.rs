use std::path::Path;

use rc3d_core::math::Vec3;
use rc3d_core::{DisplayMode, EngineResult, NodeId};
use rc3d_render::render_action::DrawCall;
use rc3d_render::viewport::{LayoutMode, ViewportLayout};
use rc3d_render::{FrameStats, Renderer};
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

        // 7. Render
        let cache = &self.world.cached_draw_calls;
        let dc: &[DrawCall] = if cache.is_empty() {
            &self.world.collector.draw_calls
        } else {
            cache
        };
        let stats = renderer.render_draw_calls(dc, &self.world.graph);

        // 8. Track FPS
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
