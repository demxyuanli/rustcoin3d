use std::time::Instant;

use rc3d_scene::SceneGraph;

/// A sensor fires a callback when triggered (by time or field change).
pub trait Sensor: std::fmt::Debug {
    fn should_fire(&self, time: f64) -> bool;
    fn fire(&mut self, graph: &mut SceneGraph, time: f64);
}

/// Fires once after a delay.
pub struct AlarmSensor {
    pub fire_time: f64,
    pub callback: Option<Box<dyn FnMut(&mut SceneGraph)>>,
    pub fired: bool,
}

impl std::fmt::Debug for AlarmSensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlarmSensor")
            .field("fire_time", &self.fire_time)
            .field("fired", &self.fired)
            .finish()
    }
}

impl AlarmSensor {
    pub fn new(delay_secs: f64, callback: impl FnMut(&mut SceneGraph) + 'static) -> Self {
        Self {
            fire_time: delay_secs,
            callback: Some(Box::new(callback)),
            fired: false,
        }
    }
}

impl Sensor for AlarmSensor {
    fn should_fire(&self, time: f64) -> bool {
        !self.fired && time >= self.fire_time
    }
    fn fire(&mut self, graph: &mut SceneGraph, _time: f64) {
        self.fired = true;
        if let Some(cb) = &mut self.callback {
            cb(graph);
        }
    }
}

/// Fires at regular intervals.
pub struct TimerSensor {
    pub interval: f64,
    pub next_fire: f64,
    pub callback: Option<Box<dyn FnMut(&mut SceneGraph, f64)>>,
}

impl std::fmt::Debug for TimerSensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimerSensor")
            .field("interval", &self.interval)
            .field("next_fire", &self.next_fire)
            .finish()
    }
}

impl TimerSensor {
    pub fn new(interval_secs: f64, callback: impl FnMut(&mut SceneGraph, f64) + 'static) -> Self {
        Self {
            interval: interval_secs,
            next_fire: interval_secs,
            callback: Some(Box::new(callback)),
        }
    }
}

impl Sensor for TimerSensor {
    fn should_fire(&self, time: f64) -> bool {
        time >= self.next_fire
    }
    fn fire(&mut self, graph: &mut SceneGraph, time: f64) {
        self.next_fire = time + self.interval;
        if let Some(cb) = &mut self.callback {
            cb(graph, time);
        }
    }
}

/// Registry of all sensors.
pub struct SensorQueue {
    pub sensors: Vec<Box<dyn Sensor>>,
    pub start: Instant,
}

impl SensorQueue {
    pub fn new() -> Self {
        Self {
            sensors: Vec::new(),
            start: Instant::now(),
        }
    }

    pub fn add(&mut self, sensor: impl Sensor + 'static) {
        self.sensors.push(Box::new(sensor));
    }

    pub fn process(&mut self, graph: &mut SceneGraph) {
        let time = self.start.elapsed().as_secs_f64();
        for sensor in &mut self.sensors {
            if sensor.should_fire(time) {
                sensor.fire(graph, time);
            }
        }
    }
}

impl Default for SensorQueue {
    fn default() -> Self {
        Self::new()
    }
}
