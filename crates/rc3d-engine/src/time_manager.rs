use std::time::Instant;

/// Monotonic simulation clock with optional time scaling (for pause/slow-mo via scale = 0 or < 1).
#[derive(Debug)]
pub struct TimeManager {
    start: Instant,
    pub time_scale: f64,
}

impl TimeManager {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            time_scale: 1.0,
        }
    }

    pub fn secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * self.time_scale
    }

    pub fn reset_origin(&mut self) {
        self.start = Instant::now();
    }
}

impl Default for TimeManager {
    fn default() -> Self {
        Self::new()
    }
}
