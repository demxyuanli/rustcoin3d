use std::time::Instant;

use super::streaming_lod::FullResPatch;

pub struct LODState {
    pub(crate) full_res_patches: Vec<FullResPatch>,
    pub preview_mode_active: bool,
    pub stream_next_tick: Option<Instant>,
}
