use std::collections::{HashSet, VecDeque};
use rc3d_core::NodeId;
use rc3d_core::math::{Vec3, Mat4};
use rc3d_scene::SceneGraph;

#[derive(Debug, Clone)]
pub struct CameraState {
    pub target: Vec3,
    pub distance: f32,
    pub phi: f32,       // azimuth (radians), 0 = +Z
    pub theta: f32,     // elevation (radians), 0 = horizon
    pub pan_offset: (f32, f32),
    pub fov: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 8.0,
            phi: std::f32::consts::FRAC_PI_4,
            theta: 0.5,
            pan_offset: (0.0, 0.0),
            fov: std::f32::consts::FRAC_PI_4,
        }
    }
}

impl CameraState {
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.phi += dx;
        self.theta = (self.theta + dy).clamp(-1.5, 1.5);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let s = self.distance * 0.001;
        self.pan_offset.0 += dx * s;
        self.pan_offset.1 += dy * s;
    }

    pub fn zoom(&mut self, amount: f32) {
        self.distance = (self.distance - amount).clamp(0.5, 500.0);
    }

    pub fn fit(&mut self) {
        self.target = Vec3::ZERO;
        self.distance = 8.0;
        self.phi = std::f32::consts::FRAC_PI_4;
        self.theta = 0.5;
        self.pan_offset = (0.0, 0.0);
    }

    pub fn position(&self) -> Vec3 {
        let x = self.distance * self.theta.cos() * self.phi.sin();
        let y = self.distance * self.theta.sin();
        let z = self.distance * self.theta.cos() * self.phi.cos();
        self.target + Vec3::new(x, y, z) + Vec3::new(self.pan_offset.0, self.pan_offset.1, 0.0)
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position(), self.target, Vec3::Y)
    }
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct TestRun {
    pub name: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub running: bool,
}

#[derive(Debug, Clone)]
pub enum EngineEvent {
    SceneLoaded,
    SceneReset,
    TestStarted(String),
    TestPassed(String),
    TestFailed(String),
    TestStopped,
    SelectionChanged,
    PropertyChanged { node: NodeId, field: String },
    LogAppended,
    LogCleared,
    DisplayModeChanged(String),
    Quit,
}

pub struct EngineState {
    pub scene: SceneGraph,
    pub selection: HashSet<NodeId>,
    pub camera: CameraState,
    pub active_test: Option<TestRun>,
    pub log_entries: VecDeque<LogEntry>,
    pub display_mode: String,
    pub events: VecDeque<EngineEvent>,
}

impl EngineState {
    pub fn new() -> Self {
        Self {
            scene: SceneGraph::new(),
            selection: HashSet::new(),
            camera: CameraState::default(),
            active_test: None,
            log_entries: VecDeque::with_capacity(256),
            display_mode: "shaded".into(),
            events: VecDeque::new(),
        }
    }

    pub fn push_event(&mut self, event: EngineEvent) {
        self.events.push_back(event);
    }

    pub fn take_events(&mut self) -> Vec<EngineEvent> {
        std::mem::take(&mut self.events).into_iter().collect()
    }

    pub fn add_log(&mut self, level: &str, message: &str) {
        if self.log_entries.len() >= 256 {
            self.log_entries.pop_front();
        }
        self.log_entries.push_back(LogEntry {
            level: level.to_string(),
            message: message.to_string(),
        });
        self.push_event(EngineEvent::LogAppended);
    }
}
