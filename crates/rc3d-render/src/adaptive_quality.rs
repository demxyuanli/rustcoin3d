/// Five quality levels for adaptive rendering.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd)]
pub enum QualityLevel {
    Ultra,   // >=60fps — full effects, 4 cascades @ 2048^2
    High,    // 45-60fps — full effects, 4 cascades @ 2048^2
    Medium,  // 30-45fps — reduced SSAO, 2 cascades @ 1024^2
    Low,     // 20-30fps — no SSAO/Bloom, 1 cascade @ 512^2
    Minimal, // <20fps — no post, flat shading, 1 cascade @ 256^2
}

impl QualityLevel {
    pub fn name(self) -> &'static str {
        match self {
            QualityLevel::Ultra => "Ultra",
            QualityLevel::High => "High",
            QualityLevel::Medium => "Medium",
            QualityLevel::Low => "Low",
            QualityLevel::Minimal => "Minimal",
        }
    }

    pub fn is_low(self) -> bool {
        matches!(self, QualityLevel::Low | QualityLevel::Minimal)
    }

    fn downgrade(self) -> Self {
        match self {
            QualityLevel::Ultra => QualityLevel::High,
            QualityLevel::High => QualityLevel::Medium,
            QualityLevel::Medium => QualityLevel::Low,
            QualityLevel::Low => QualityLevel::Minimal,
            QualityLevel::Minimal => QualityLevel::Minimal,
        }
    }

    fn upgrade(self) -> Self {
        match self {
            QualityLevel::Ultra => QualityLevel::Ultra,
            QualityLevel::High => QualityLevel::Ultra,
            QualityLevel::Medium => QualityLevel::High,
            QualityLevel::Low => QualityLevel::Medium,
            QualityLevel::Minimal => QualityLevel::Low,
        }
    }
}

/// Controls adaptive quality with EMA smoothing and hysteresis.
pub struct AdaptiveController {
    pub current: QualityLevel,
    frame_time_ema: f32,
    consecutive_over: u32,
    consecutive_under: u32,
}

impl AdaptiveController {
    pub fn new() -> Self {
        Self {
            current: QualityLevel::Ultra,
            frame_time_ema: 16.6, // Assume 60fps initially
            consecutive_over: 0,
            consecutive_under: 0,
        }
    }

    /// Update quality based on latest frame time (ms).
    /// Returns the (possibly unchanged) quality level.
    pub fn update(&mut self, frame_time_ms: f32) -> QualityLevel {
        // EMA smoothing
        self.frame_time_ema = self.frame_time_ema * 0.9 + frame_time_ms * 0.1;

        let target_fps = self.fps_for_level(self.current);
        let target_frame_time = 1000.0 / target_fps;

        // Check if frame time exceeds threshold (over budget)
        if self.frame_time_ema > target_frame_time * 1.2 {
            self.consecutive_over += 1;
            self.consecutive_under = 0;
        } else if self.frame_time_ema < target_frame_time * 0.8 {
            self.consecutive_under += 1;
            self.consecutive_over = 0;
        } else {
            self.consecutive_over = 0;
            self.consecutive_under = 0;
        }

        // Downgrade after 5 consecutive slow frames (fast reaction)
        if self.consecutive_over >= 5 {
            self.current = self.current.downgrade();
            self.consecutive_over = 0;
        }
        // Upgrade after 30 consecutive fast frames (slow recovery — hysteresis)
        if self.consecutive_under >= 30 {
            self.current = self.current.upgrade();
            self.consecutive_under = 0;
        }

        self.current
    }

    fn fps_for_level(&self, level: QualityLevel) -> f32 {
        match level {
            QualityLevel::Ultra => 60.0,
            QualityLevel::High => 45.0,
            QualityLevel::Medium => 30.0,
            QualityLevel::Low => 20.0,
            QualityLevel::Minimal => 15.0,
        }
    }
}

// -- Keep backward compat: old 3-level AdaptiveQuality as a type alias --
/// Backward-compatible 3-level quality (maps Ultra/High->High, Medium->Medium, Low/Minimal->Low).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveQuality {
    High,
    Medium,
    Low,
}

impl AdaptiveQuality {
    pub fn name(self) -> &'static str {
        match self {
            AdaptiveQuality::High => "High",
            AdaptiveQuality::Medium => "Medium",
            AdaptiveQuality::Low => "Low",
        }
    }

    pub fn is_low(self) -> bool {
        self == AdaptiveQuality::Low
    }

    pub fn from_quality_level(ql: QualityLevel) -> Self {
        match ql {
            QualityLevel::Ultra | QualityLevel::High => AdaptiveQuality::High,
            QualityLevel::Medium => AdaptiveQuality::Medium,
            QualityLevel::Low | QualityLevel::Minimal => AdaptiveQuality::Low,
        }
    }

    pub fn update(self, frame_time_ms: f32) -> Self {
        match self {
            AdaptiveQuality::High => {
                if frame_time_ms >= ADAPTIVE_LOW_ENTER_MS { AdaptiveQuality::Low }
                else if frame_time_ms >= ADAPTIVE_MEDIUM_ENTER_MS { AdaptiveQuality::Medium }
                else { AdaptiveQuality::High }
            }
            AdaptiveQuality::Medium => {
                if frame_time_ms >= ADAPTIVE_LOW_ENTER_MS { AdaptiveQuality::Low }
                else if frame_time_ms <= ADAPTIVE_MEDIUM_EXIT_MS { AdaptiveQuality::High }
                else { AdaptiveQuality::Medium }
            }
            AdaptiveQuality::Low => {
                if frame_time_ms <= ADAPTIVE_LOW_EXIT_MS {
                    if frame_time_ms <= ADAPTIVE_MEDIUM_EXIT_MS { AdaptiveQuality::High }
                    else { AdaptiveQuality::Medium }
                } else { AdaptiveQuality::Low }
            }
        }
    }
}

// Keep existing threshold constants
const ADAPTIVE_MEDIUM_ENTER_MS: f32 = 26.0;
const ADAPTIVE_LOW_ENTER_MS: f32 = 40.0;
const ADAPTIVE_MEDIUM_EXIT_MS: f32 = 22.0;
const ADAPTIVE_LOW_EXIT_MS: f32 = 33.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_downgrades_on_sustained_slow_frames() {
        let mut ctrl = AdaptiveController::new();
        // With EMA smoothing (alpha=0.1), the EMA takes ~3 iterations to rise
        // above the Ultra over-threshold (20ms at 60fps). Then 5 consecutive_overs
        // trigger a downgrade. Total: ~7 iterations of 30ms.
        for _ in 0..7 {
            ctrl.update(30.0);
        }
        assert_eq!(ctrl.current, QualityLevel::High); // Ultra -> High
    }

    #[test]
    fn quality_recovers_slowly() {
        let mut ctrl = AdaptiveController::new();
        // Force downgrade Ultra -> High
        for _ in 0..7 { ctrl.update(30.0); }
        assert_eq!(ctrl.current, QualityLevel::High);

        // Recovery requires 30 consecutive frames below the High under-threshold
        // (17.78ms at 45fps). With 0ms input and EMA decay, the EMA drops below
        // 17.78ms after ~2 OK-zone iterations, then needs 30 consecutive_unders.
        // Total after downgrade: ~32 iterations.
        for _ in 0..35 {
            ctrl.update(0.0);
        }
        assert_eq!(ctrl.current, QualityLevel::Ultra);
    }

    #[test]
    fn old_api_still_works() {
        let q = AdaptiveQuality::High;
        let q = q.update(30.0);
        assert_eq!(q, AdaptiveQuality::Medium);
    }
}
