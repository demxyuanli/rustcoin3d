#[derive(Clone, Copy, Debug)]
pub struct InputState {
    pub cursor_pos: (f64, f64),
    /// Cursor position before the current `CursorMoved` event.
    pub cursor_prev: (f64, f64),
    pub shift_pressed: bool,
    pub ctrl_pressed: bool,
    pub alt_pressed: bool,
    pub window_size: (u32, u32),
    pub pending_pick: Option<(f32, f32)>,
    pub left_dragged: bool,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            cursor_pos: (0.0, 0.0),
            cursor_prev: (0.0, 0.0),
            shift_pressed: false,
            ctrl_pressed: false,
            alt_pressed: false,
            window_size: (800, 600),
            pending_pick: None,
            left_dragged: false,
        }
    }
}
