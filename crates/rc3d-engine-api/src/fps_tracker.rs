use std::collections::VecDeque;
use std::time::Instant;

use rc3d_render::FrameStats;

pub struct FpsTracker {
    samples: VecDeque<f32>,
    sum: f32,
    capacity: usize,
    ema_fps: f32,
    last_log: Instant,
}

impl FpsTracker {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            sum: 0.0,
            capacity,
            ema_fps: 0.0,
            last_log: Instant::now(),
        }
    }

    pub fn push(&mut self, frame_time_ms: f32) {
        if self.samples.len() == self.capacity {
            if let Some(old) = self.samples.pop_front() {
                self.sum -= old;
            }
        }
        self.samples.push_back(frame_time_ms);
        self.sum += frame_time_ms;
        let inst_fps = if frame_time_ms > 0.0 {
            1000.0 / frame_time_ms
        } else {
            0.0
        };
        if self.ema_fps <= 0.0 {
            self.ema_fps = inst_fps;
        } else {
            let alpha = 0.08_f32;
            self.ema_fps += (inst_fps - self.ema_fps) * alpha;
        }
    }

    pub fn average_frame_ms(&self) -> f32 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.sum / self.samples.len() as f32
        }
    }

    pub fn fps(&self) -> f32 {
        let avg = self.average_frame_ms();
        if avg > 0.0 {
            1000.0 / avg
        } else {
            0.0
        }
    }

    pub fn smoothed_fps(&self) -> f32 {
        if self.ema_fps > 0.0 {
            self.ema_fps
        } else {
            self.fps()
        }
    }

    pub fn maybe_log(&mut self, stats: &FrameStats, quality: &str) {
        if self.last_log.elapsed().as_secs() >= 1 {
            self.last_log = Instant::now();
            log::debug!(
                "FPS: {:.1} | frame: {:.2}ms | tris: {} | draws: {} | culled: {} | quality: {}",
                self.fps(),
                self.average_frame_ms(),
                stats.visible_triangles,
                stats.visible_draw_calls,
                stats.culled_draw_calls,
                quality,
            );
        }
    }
}
