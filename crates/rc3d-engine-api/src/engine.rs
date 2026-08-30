use std::collections::HashSet;
use std::path::Path;

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::{DisplayMode, EngineResult, NodeId, VisualStyle, VisualStyleLibrary};
use rc3d_gizmo::Gizmo;
use rc3d_actions::Action;
use rc3d_actions::SectionPlaneAction;
use rc3d_render::render_action::DrawCall;
use rc3d_render::viewport::{LayoutMode, ViewportLayout};
use rc3d_render::{AdaptiveControl, FrameStats, Renderer};
use rc3d_scene::SceneGraph;

type PreRenderHook = Box<dyn FnMut(&mut Renderer)>;
type PickCallback = Box<dyn FnMut(&mut SceneGraph, NodeId, Vec3)>;
type PickHitCallback = Box<dyn FnMut(&mut SceneGraph, &rc3d_actions::PickHit)>;
type PanelMouseHook = Box<dyn Fn(f32, f32, u32, u32) -> bool>;

use crate::background::BackgroundSettings;
use crate::camera::{CameraController, ViewPreset};
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
    pub pre_render_hook: Option<PreRenderHook>,

    /// Optional callback invoked when the user clicks on geometry in the
    /// scene. Receives the scene graph, the picked NodeId, and the
    /// world-space intersection point.
    pub on_pick: Option<PickCallback>,
    /// Called with the full pick hit (face / edge indices) after [`Self::pick_at`].
    pub on_pick_hit: Option<PickHitCallback>,
    /// Ray pick granularity. Default [`rc3d_actions::PickMode::Node`].
    pub pick_mode: rc3d_actions::PickMode,

    /// Optional keyboard event hook. Receives the physical key. Return `true`
    /// to request a redraw after handling the keypress.
    pub panel_overlay_key_hook: Option<Box<dyn Fn(winit::keyboard::PhysicalKey) -> bool>>,

    /// Optional mouse click hook for HUD overlay interaction.
    /// Receives (x, y, window_width, window_height). Return `true` to request
    /// a redraw after handling the click.
    pub panel_overlay_mouse_hook: Option<PanelMouseHook>,

    /// Nodes hidden from rendering (e.g. via editor Hide command).
    pub hidden_nodes: HashSet<NodeId>,

    /// Named visual styles (HOOPS-style catalog) applied to subtrees.
    pub visual_styles: VisualStyleLibrary,

    /// Transform manipulator overlay (drawn from selection each frame).
    pub gizmo: Gizmo,

    /// Extra overlay line batches (section-plane widget, etc.) drawn with the gizmo.
    pub overlay_line_batches: Vec<(Vec<rc3d_render::LineVertex>, [f32; 4])>,

    /// Shared pointer / modifier / click-vs-drag state for [`Self::handle_window_event`].
    pub input: crate::input_state::InputState,
    /// Last 3D film rect applied from the host (egui central hole).
    scene_region: rc3d_render::viewport::ViewportRect,
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
            on_pick_hit: None,
            pick_mode: rc3d_actions::PickMode::Node,
            panel_overlay_key_hook: None,
            panel_overlay_mouse_hook: None,
            hidden_nodes: HashSet::new(),
            visual_styles: VisualStyleLibrary::builtin(),
            gizmo: Gizmo::new(),
            overlay_line_batches: Vec::new(),
            input: {
                let s = window.inner_size();
                crate::input_state::InputState {
                    window_size: (s.width, s.height),
                    ..Default::default()
                }
            },
            scene_region: rc3d_render::viewport::ViewportRect {
                x: 0,
                y: 0,
                width: window.inner_size().width.max(1),
                height: window.inner_size().height.max(1),
            },
        }
    }

    pub fn visual_styles(&self) -> &VisualStyleLibrary {
        &self.visual_styles
    }

    pub fn visual_styles_mut(&mut self) -> &mut VisualStyleLibrary {
        &mut self.visual_styles
    }

    /// Apply a catalog style to a Separator (or any node). Returns false if unknown.
    pub fn apply_visual_style(&mut self, id: NodeId, name: &str) -> bool {
        let Some(style) = self.visual_styles.get(name).cloned() else {
            return false;
        };
        self.world.graph.apply_visual_style(id, &style);
        true
    }

    pub fn register_visual_style(&mut self, style: VisualStyle) {
        self.visual_styles.register(style);
    }

    /// Resolve PMI names on every `AnnotationSet` and stamp unbound points.
    pub fn bind_scene_pmi(&mut self) -> usize {
        self.world.graph.bind_pmi()
    }

    /// Append a JSON PMI sidecar to `set_id` and apply bindings.
    pub fn apply_pmi_json(&mut self, set_id: NodeId, json: &str) -> Result<usize, String> {
        let doc = rc3d_scene::PmiDocument::from_json(json).map_err(|e| e.to_string())?;
        Ok(rc3d_scene::apply_pmi_document(&mut self.world.graph, set_id, &doc))
    }

    /// Apply `name` to the current selection, or to scene roots if nothing is selected.
    pub fn apply_visual_style_to_selected(&mut self, name: &str) -> usize {
        let selected: Vec<NodeId> = self.world.graph.selected_nodes().iter().copied().collect();
        let targets = if selected.is_empty() {
            self.world.graph.roots().to_vec()
        } else {
            selected
        };
        let mut n = 0;
        for id in targets {
            if self.apply_visual_style(id, name) {
                n += 1;
            }
        }
        n
    }

    /// Set the global display mode (Shaded, Flat, Wireframe, etc.).
    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        if let Some(ref mut r) = self.renderer {
            r.set_display_mode(mode);
        }
    }

    /// HOOPS Isolate/Ghost: selected stays shaded; unselected filled draws
    /// become translucent. No-op when the selection is empty.
    pub fn set_ghost_unselected(&mut self, enabled: bool) {
        if let Some(ref mut r) = self.renderer {
            r.set_ghost_unselected(enabled);
        }
    }

    pub fn set_ghost_opacity(&mut self, opacity: f32) {
        if let Some(ref mut r) = self.renderer {
            r.set_ghost_opacity(opacity);
        }
    }

    /// Weighted blended OIT for overlapping transparent / ghost / transmission draws.
    /// Default is on; disable to fall back to painter's algorithm.
    pub fn set_wboit(&mut self, enabled: bool) {
        if let Some(ref mut r) = self.renderer {
            r.set_wboit(enabled);
        }
    }

    /// HOOPS X-ray: all filled geometry translucent with crease edges kept.
    pub fn set_xray_mode(&mut self, enabled: bool) {
        if let Some(ref mut r) = self.renderer {
            r.set_xray_mode(enabled);
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

    /// Halftone / glitch stylize (three.js HalftonePass / GlitchPass analog).
    pub fn set_post_stylize(&mut self, halftone: f32, glitch: f32) {
        if let Some(ref mut r) = self.renderer {
            r.set_post_stylize(halftone, glitch);
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
        crate::import::resolve_file_nodes(&mut self.world.graph);
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
    /// `Quad` installs the HOOPS-style Front/Right/Top/Persp camera pack.
    /// Other modes drop viewport-camera bindings (camera nodes stay in the graph).
    pub fn set_layout_mode(&mut self, mode: LayoutMode) {
        match mode {
            LayoutMode::Quad => self.apply_standard_quad_views(),
            other => {
                self.viewport_cameras.cameras.clear();
                self.rebuild_layout(other);
            }
        }
    }

    fn rebuild_layout(&mut self, mode: LayoutMode) {
        let renderer = self.renderer.as_mut().expect("renderer not initialized");
        let region = self.scene_region;
        {
            let layout = renderer.viewport_layout_mut();
            layout.layout_mode = mode;
            layout.rebuild_in_rect(region);
        }
        renderer.set_scene_region(region);
        let layout = renderer.viewport_layout_mut();
        self.viewport_cameras
            .remap_viewport_ids_from_layout(layout);
        self.viewport_cameras.bind_camera_nodes(layout);
    }

    /// Size the 3D film to `rect` (window pixels). Call each frame after egui layout.
    pub fn apply_scene_region(&mut self, rect: rc3d_render::viewport::ViewportRect) {
        let renderer = match self.renderer.as_mut() {
            Some(r) => r,
            None => return,
        };
        let rect = rect.clamped_to(renderer.config.width, renderer.config.height);
        let same = self.scene_region == rect;
        self.scene_region = rect;
        renderer.set_scene_region(rect);
        if same {
            return;
        }
        renderer.viewport_layout_mut().rebuild_in_rect(rect);
        let layout = renderer.viewport_layout_mut();
        self.viewport_cameras
            .remap_viewport_ids_from_layout(layout);
        self.viewport_cameras.bind_camera_nodes(layout);
    }

    pub fn scene_region(&self) -> rc3d_render::viewport::ViewportRect {
        self.scene_region
    }

    /// True when the last cursor position lies inside the 3D film rectangle.
    pub fn pointer_in_scene_region(&self) -> bool {
        let (cx, cy) = (self.input.cursor_pos.0 as f32, self.input.cursor_pos.1 as f32);
        self.scene_region.contains(cx, cy)
    }

    /// Default four-view pack: Persp (Iso) + Top/Front/Right orthographic cameras.
    pub fn apply_standard_quad_views(&mut self) {
        self.rebuild_layout(LayoutMode::Quad);
        let aabb = rc3d_scene::GetBoundingBoxAction::compute_scene_aabb(&self.world.graph);
        let renderer = self.renderer.as_mut().expect("renderer not initialized");
        let layout = renderer.viewport_layout_mut();
        self.viewport_cameras
            .install_standard_quad(&mut self.world.graph, layout, aabb);
    }

    pub fn cycle_layout_mode(&mut self) {
        let current = self
            .renderer
            .as_ref()
            .expect("renderer not initialized")
            .viewport_layout()
            .layout_mode;
        let next = match current {
            LayoutMode::Single => LayoutMode::Quad,
            LayoutMode::Quad => LayoutMode::LeftRight,
            LayoutMode::LeftRight => LayoutMode::TopBottom,
            LayoutMode::TopBottom => LayoutMode::Single,
        };
        self.set_layout_mode(next);
    }

    pub fn cycle_active_viewport(&mut self) {
        let renderer = self.renderer.as_mut().expect("renderer not initialized");
        let layout = renderer.viewport_layout_mut();
        layout.cycle_active();
        let id = layout.active_id;
        self.viewport_cameras.set_active(id, layout);
    }

    /// Apply a named view preset to the active viewport camera, or the main orbit camera.
    pub fn set_view_preset(&mut self, preset: ViewPreset) {
        let aabb = rc3d_scene::GetBoundingBoxAction::compute_scene_aabb(&self.world.graph);
        if let Some(vc) = self.viewport_cameras.active_mut() {
            vc.controller.set_view_preset(preset);
            if let Some(ref box_) = aabb {
                vc.controller.fit_bounds(box_, std::f32::consts::FRAC_PI_4);
            }
            return;
        }
        self.controller.set_view_preset(preset);
        if let Some(ref box_) = aabb {
            self.controller.fit_bounds(box_, std::f32::consts::FRAC_PI_4);
        }
    }

    /// Orient the active camera along `from` (target toward eye). Does not fit bounds.
    pub fn set_view_from_direction(&mut self, from: rc3d_core::math::Vec3) {
        if let Some(vc) = self.viewport_cameras.active_mut() {
            vc.controller.set_view_from_direction(from);
            return;
        }
        self.controller.set_view_from_direction(from);
    }

    /// Orbit the active camera by yaw/pitch deltas (radians-scale, same as mouse orbit).
    pub fn orbit_view(&mut self, dx: f32, dy: f32) {
        if let Some(vc) = self.viewport_cameras.active_mut() {
            vc.controller.orbit(dx, dy);
            return;
        }
        self.controller.orbit(dx, dy);
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
        self.render_with_overlay(None)
    }

    /// Render a frame and optionally composite a post-swapchain overlay (egui).
    pub fn render_with_overlay(
        &mut self,
        post_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    ) -> FrameStats {
        // Extract pre-render hook to avoid borrow conflict with renderer
        let mut pre_hook = self.pre_render_hook.take();
        let renderer = self.renderer.as_mut().expect("renderer not initialized");

        // 1. Reapply CAD tier constraints
        renderer.reapply_cad_tier_constraints();

        // 2. Evaluate engines (animation, simulation, etc.)
        self.world.evaluate_engines();

        if crate::import::resolve_file_nodes(&mut self.world.graph) > 0 {
            self.world.collector.invalidate_mesh_cache();
        }

        // 3. Set materials on renderer (so shader lookups work during draw)
        renderer.set_materials(self.world.materials.clone());

        // 4. Reset collector for this frame
        let dm = renderer.display_mode();
        self.world.reset_collector(dm);
        self.world.collector.set_hidden_nodes(&self.hidden_nodes);

        // 5. Update camera nodes from controller state
        let aspect = renderer
            .viewport_layout()
            .active()
            .map(|v| v.rect.aspect())
            .filter(|a| a.is_finite() && *a > 1.0e-4)
            .unwrap_or_else(|| {
                renderer.config.width as f32 / renderer.config.height.max(1) as f32
            });
        let quad_pack = renderer.viewport_layout().layout_mode == LayoutMode::Quad
            && self.viewport_cameras.cameras.len() >= 4;

        // Legacy path: copy the main orbit onto every camera node.
        // Skip when the four-view pack owns distinct cameras.
        if !quad_pack {
            let roots: Vec<NodeId> = self.world.graph.roots().to_vec();
            for &root in &roots {
                self.controller
                    .update_camera_recursive(&mut self.world.graph, root, aspect);
            }
        }

        // Viewport-camera path: update cameras bound to specific viewports
        let layout = renderer.viewport_layout();
        self.viewport_cameras.update_all(&mut self.world.graph, layout);

        // 5b. Update LOD levels based on camera distance
        let lod_cam_pos = self.controller.eye_position();
        self.world.graph.update_lod_levels(lod_cam_pos);

        // 6. Traverse scene graph to populate draw calls
        self.world.traverse_all_roots();

        // 7. (reserved for future use)

        // 8. Collect markup overlay vertices (annotations, text, dimensions)
        let markup_root = self.world.graph.roots().first().copied().unwrap_or_default();
        renderer.collect_markup_vertices(&self.world.graph, markup_root);

        // 9. Transfer light sets from collector to renderer
        let light_sets = std::mem::take(&mut self.world.collector.light_sets);
        renderer.set_light_sets(light_sets);
        renderer.set_light_probe(
            self.world.collector.light_probe_sh,
            self.world.collector.light_probe_intensity,
        );

        // 9b. Collect section planes from scene graph
        let mut section_action = SectionPlaneAction::new();
        let roots: Vec<NodeId> = self.world.graph.roots().to_vec();
        for &root in &roots {
            section_action.apply(&self.world.graph, root);
        }
        renderer.set_clip_planes(section_action.planes, section_action.cap_tints);

        // 10. Fallback: if traversal found no camera, apply a default projection.
        // Without this, the collector stays at IDENTITY and nothing renders.
        if self.world.collector.projection_matrix == Mat4::IDENTITY {
            self.world.collector.projection_matrix =
                Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0);
            self.world.collector.view_matrix =
                Mat4::look_at_rh(Vec3::new(0.0, 2.0, 8.0), Vec3::ZERO, Vec3::Y);
            self.world.collector.camera_pos = Vec3::new(0.0, 2.0, 8.0);
        }

        // 11. Apply camera matrices to draw calls (single-view path only)
        let (sw, sh) = (renderer.config.width, renderer.config.height);
        let stereo_eyes = if quad_pack {
            None
        } else {
            crate::viewport::collect_stereo_eyes(&self.world.graph, sw, sh)
        };
        let stereo_pack = stereo_eyes.is_some();
        if !quad_pack && !stereo_pack {
            rc3d_render::render_action::apply_world_camera(
                &mut self.world.collector.draw_calls,
                self.world.collector.view_matrix,
                self.world.collector.projection_matrix,
                self.world.collector.camera_pos,
            );
        }

        rc3d_render::apply_ghost_unselected(
            &mut self.world.collector.draw_calls,
            renderer.ghost_unselected,
            renderer.ghost_opacity,
        );
        rc3d_render::apply_xray(
            &mut self.world.collector.draw_calls,
            renderer.xray_mode,
            rc3d_render::XRAY_FILL_OPACITY,
        );

        // 10. Cache draw calls for static-frame fast path
        self.world.cached_draw_calls = self.world.collector.draw_calls.clone();
        // Clear stale dirty flags after full traversal
        rc3d_render::dirty_flags::clear_all_dirty_flags(&mut self.world.graph);

        // 11. Send effect commands to renderer (annotation labels are projected in render pass)
        let effect_cmds = std::mem::take(&mut self.world.collector.effect_commands);
        renderer.set_effect_commands(effect_cmds);
        renderer.set_scene_view_projection(
            self.world.collector.view_matrix,
            self.world.collector.projection_matrix,
        );

        // 12. Call pre-render hook (before GPU submission)
        if let Some(ref mut h) = pre_hook {
            h(renderer);
        }

        crate::gizmo_bind::sync_gizmo_from_selection(&mut self.gizmo, &self.world.graph);
        renderer.gizmo_line_batches.clear();
        if self.gizmo.visible {
            renderer.gizmo_line_batches = self.gizmo.generate_lines();
        }
        renderer
            .gizmo_line_batches
            .extend(self.overlay_line_batches.iter().cloned());

        renderer.update_cube_cameras(&self.world.graph, &self.world.cached_draw_calls);

        // 15. Render using cached draw calls for consistent state
        let stats = if quad_pack {
            let eyes = self
                .viewport_cameras
                .collect_quad_eyes(&self.world.graph, renderer.viewport_layout());
            if let Some(eye) = eyes.iter().find(|e| !e.orthographic).or(eyes.first()) {
                renderer.set_scene_view_projection(eye.view, eye.projection);
            }
            renderer.render_standard_quad_views_with_overlay(
                &mut self.world.cached_draw_calls,
                &self.world.graph,
                &eyes,
                post_overlay,
            )
        } else if let Some((mode, eyes)) = stereo_eyes {
            if let Some(eye) = eyes.first() {
                renderer.set_scene_view_projection(eye.view, eye.projection);
            }
            let stats = renderer.render_stereo_views(
                &mut self.world.cached_draw_calls,
                &self.world.graph,
                &eyes,
                mode,
            );
            let _ = post_overlay;
            stats
        } else {
            renderer.render_draw_calls_with_overlay(
                &self.world.cached_draw_calls,
                &self.world.graph,
                post_overlay,
            )
        };

        // 16. Update HUD overlay (renders FPS + markup text in top-left corner)
        let mut mode_name = format!(
            "{:?} | IBL:{}",
            renderer.display_mode(),
            renderer.ibl_preset_name(),
        );
        if quad_pack {
            mode_name.push_str(" | Quad: Persp/Top/Front/Right");
        }
        if stereo_pack {
            mode_name.push_str(" | Stereo");
        }
        if renderer.enable_wboit {
            mode_name.push_str(" | WBOIT");
        }
        let markup_text = renderer.collect_markup_text(&self.world.graph);
        if !markup_text.is_empty() {
            mode_name.push('\n');
            mode_name.push_str(&markup_text.join("\n"));
        }
        if let Some(ref text_hook) = self.hud_text_hook {
            let overlay = text_hook();
            if !overlay.is_empty() {
                mode_name.push('\n');
                mode_name.push_str(&overlay);
            }
        }
        renderer.update_hud(
            self.fps.smoothed_fps(),
            self.fps.average_frame_ms(),
            &stats,
            &mode_name,
        );

        // 14. Report frame time for adaptive quality controller
        renderer.report_frame_time_ms(stats.frame_time_ms as f32, self.adaptive_control);

        // 14. Restore pre-render hook and track FPS
        self.pre_render_hook = pre_hook;
        self.fps.push(stats.frame_time_ms as f32);

        stats
    }

    /// Composite the four-view pack to an RGBA image (layout is restored afterwards).
    pub fn render_quad_pack_image(&mut self, width: u32, height: u32) -> (u32, u32, Vec<u8>) {
        let _ = self.render();
        let renderer = self.renderer.as_mut().expect("renderer not initialized");
        let (ow, oh) = (renderer.config.width, renderer.config.height);
        let saved_mode = renderer.viewport_layout().layout_mode;
        renderer.viewport_layout_mut().layout_mode = LayoutMode::Quad;
        renderer.viewport_layout_mut().rebuild(width.max(1), height.max(1));
        self.viewport_cameras
            .remap_viewport_ids_from_layout(renderer.viewport_layout());
        self.viewport_cameras
            .bind_camera_nodes(renderer.viewport_layout_mut());
        self.viewport_cameras
            .update_all(&mut self.world.graph, renderer.viewport_layout());
        let eyes = self
            .viewport_cameras
            .collect_quad_eyes(&self.world.graph, renderer.viewport_layout());
        let mut dcs = self.world.cached_draw_calls.clone();
        let image = renderer.render_standard_quad_to_image(
            &mut dcs,
            &self.world.graph,
            &eyes,
            width.max(1),
            height.max(1),
        );
        let renderer = self.renderer.as_mut().expect("renderer not initialized");
        renderer.viewport_layout_mut().layout_mode = saved_mode;
        renderer.viewport_layout_mut().rebuild(ow, oh);
        self.viewport_cameras
            .remap_viewport_ids_from_layout(renderer.viewport_layout());
        self.viewport_cameras
            .bind_camera_nodes(renderer.viewport_layout_mut());
        image
    }

    /// Handle window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.input.window_size = (width, height);
        if let Some(ref mut r) = self.renderer {
            r.resize(width, height);
            self.scene_region = rc3d_render::viewport::ViewportRect {
                x: 0,
                y: 0,
                width: width.max(1),
                height: height.max(1),
            };
            r.set_scene_region(self.scene_region);
        }
        if let Some(ref mut r) = self.renderer {
            let layout = r.viewport_layout_mut();
            self.viewport_cameras
                .remap_viewport_ids_from_layout(layout);
            self.viewport_cameras.bind_camera_nodes(layout);
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

    /// Ray-pick at a window pixel and invoke [`Self::on_pick`] on the nearest hit.
    /// Empty space clears the scene-graph selection.
    pub fn pick_at(&mut self, screen_x: f32, screen_y: f32, width: u32, height: u32) {
        let Some((lx, ly, vw, vh, view, proj)) =
            self.pointer_pick_frame(screen_x, screen_y, width, height)
        else {
            return;
        };
        let ray = rc3d_actions::Ray::from_screen_point(lx, ly, vw, vh, view, proj);
        let mut picker = rc3d_actions::RayPickAction::with_mode(ray, self.pick_mode);
        rc3d_actions::apply_to_all_roots(&mut picker, &self.world.graph);
        if let Some(hit) = picker.hits.first().cloned() {
            if let Some(mut cb) = self.on_pick_hit.take() {
                cb(&mut self.world.graph, &hit);
                self.on_pick_hit = Some(cb);
            }
            if let Some(mut cb) = self.on_pick.take() {
                cb(&mut self.world.graph, hit.node, hit.point);
                self.on_pick = Some(cb);
            }
        } else {
            self.world.graph.clear_selection();
        }
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
