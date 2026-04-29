#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AdaptiveQuality {
    High,
    Medium,
    Low,
}

const ADAPTIVE_MEDIUM_ENTER_MS: f32 = 26.0;
const ADAPTIVE_LOW_ENTER_MS: f32 = 40.0;
const ADAPTIVE_MEDIUM_EXIT_MS: f32 = 22.0;
const ADAPTIVE_LOW_EXIT_MS: f32 = 33.0;

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

    pub fn update(self, frame_time_ms: f32) -> Self {
        match self {
            AdaptiveQuality::High => {
                if frame_time_ms >= ADAPTIVE_LOW_ENTER_MS {
                    AdaptiveQuality::Low
                } else if frame_time_ms >= ADAPTIVE_MEDIUM_ENTER_MS {
                    AdaptiveQuality::Medium
                } else {
                    AdaptiveQuality::High
                }
            }
            AdaptiveQuality::Medium => {
                if frame_time_ms >= ADAPTIVE_LOW_ENTER_MS {
                    AdaptiveQuality::Low
                } else if frame_time_ms <= ADAPTIVE_MEDIUM_EXIT_MS {
                    AdaptiveQuality::High
                } else {
                    AdaptiveQuality::Medium
                }
            }
            AdaptiveQuality::Low => {
                if frame_time_ms <= ADAPTIVE_LOW_EXIT_MS {
                    if frame_time_ms <= ADAPTIVE_MEDIUM_EXIT_MS {
                        AdaptiveQuality::High
                    } else {
                        AdaptiveQuality::Medium
                    }
                } else {
                    AdaptiveQuality::Low
                }
            }
        }
    }
}
