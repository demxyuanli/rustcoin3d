use std::time::Instant;

use crate::streaming_lod::FullResPatch;

pub struct LODState {
    pub full_res_patches: Vec<FullResPatch>,
    pub preview_mode_active: bool,
    pub stream_next_tick: Option<Instant>,
}
