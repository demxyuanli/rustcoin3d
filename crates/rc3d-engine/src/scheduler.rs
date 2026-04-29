use rc3d_scene::SceneGraph;

pub type FrameTimeCallback = Box<dyn FnMut(&mut SceneGraph, f64)>;

/// Ordered simulation phases before/after engine evaluation (`TickGroup`-style hooks).
#[derive(Default)]
pub struct SimulationScheduler {
    pub before_engines: Vec<FrameTimeCallback>,
    pub after_engines: Vec<FrameTimeCallback>,
}

impl SimulationScheduler {
    pub fn new() -> Self {
        Self {
            before_engines: Vec::new(),
            after_engines: Vec::new(),
        }
    }

    pub fn add_before_engines(&mut self, f: impl FnMut(&mut SceneGraph, f64) + 'static) {
        self.before_engines.push(Box::new(f));
    }

    pub fn add_after_engines(&mut self, f: impl FnMut(&mut SceneGraph, f64) + 'static) {
        self.after_engines.push(Box::new(f));
    }

    pub fn run_before_engines(&mut self, graph: &mut SceneGraph, time_secs: f64) {
        for f in &mut self.before_engines {
            f(graph, time_secs);
        }
    }

    pub fn run_after_engines(&mut self, graph: &mut SceneGraph, time_secs: f64) {
        for f in &mut self.after_engines {
            f(graph, time_secs);
        }
    }
}
