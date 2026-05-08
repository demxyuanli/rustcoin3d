use rc3d_actions::{LightSubsystem, State};
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_engine::{EngineRegistry, SimulationScheduler, TimeManager};
use rc3d_render::{MaterialLibrary, RenderCollector, Renderer};
use rc3d_scene::{SceneGraph, SensorRegistry};

pub struct World {
    pub graph: SceneGraph,
    pub engines: Option<EngineRegistry>,
    pub scheduler: SimulationScheduler,
    pub lights: LightSubsystem,
    pub time: TimeManager,
    pub materials: MaterialLibrary,
    pub collector: RenderCollector,
    pub sensor_registry: SensorRegistry,
    /// Cached draw calls from previous frame for static-scene skip.
    pub cached_draw_calls: Vec<rc3d_render::render_action::DrawCall>,
}

impl World {
    pub fn new(graph: SceneGraph) -> Self {
        Self {
            graph,
            engines: None,
            scheduler: SimulationScheduler::new(),
            lights: LightSubsystem,
            time: TimeManager::new(),
            materials: MaterialLibrary::new(),
            collector: RenderCollector::new(),
            sensor_registry: SensorRegistry::new(),
            cached_draw_calls: Vec::new(),
        }
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.time.secs()
    }

    pub fn evaluate_engines(&mut self) {
        let time = self.elapsed_secs();
        self.scheduler.run_before_engines(&mut self.graph, time);
        if let Some(engines) = self.engines.as_mut() {
            engines.evaluate_all(&mut self.graph, time);
        }
        self.scheduler.run_after_engines(&mut self.graph, time);
    }

    pub fn reset_collector(&mut self, display_mode: rc3d_core::DisplayMode) {
        self.collector.draw_calls.clear();
        self.collector.effect_commands = rc3d_render::EffectCommands::default();
        self.collector.state = State::new();
        self.collector.camera_pos = Vec3::new(0.0, 0.0, 5.0);
        self.collector.view_matrix = Mat4::IDENTITY;
        self.collector.projection_matrix = Mat4::IDENTITY;
        self.collector.projection_orthographic = false;
        self.collector.global_display_mode = display_mode;
    }

    pub fn traverse_all_roots(&mut self) {
        let roots: Vec<NodeId> = self.graph.roots().to_vec();
        for root in roots {
            self.collector.traverse(&self.graph, root);
        }
    }

    pub fn invalidate_caches(&mut self, renderer: &mut Renderer) {
        renderer.invalidate_mesh_cache();
        self.collector.invalidate_mesh_cache();
    }
}
