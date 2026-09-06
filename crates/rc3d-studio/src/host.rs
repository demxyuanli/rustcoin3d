//! Host-owned mutable state shared by the event loop and frame presenter.

use rc3d_editor::{EditorInteractionState, EditorSession};
use winit::dpi::PhysicalPosition;

use crate::cases::ActiveCase;

use crate::redraw::RedrawKind;

/// Mutable host state. Grouped so the frame presenter can take one `&mut`
/// bundle instead of a dozen loose parameters.
pub(crate) struct HostState {
    pub(crate) interaction: EditorInteractionState,
    pub(crate) session: EditorSession,
    pub(crate) cursor_pos: Option<PhysicalPosition<f64>>,
    pub(crate) recent: Vec<std::path::PathBuf>,
    pub(crate) last_autosave: std::time::Instant,
    pub(crate) last_prefs_save: std::time::Instant,
    pub(crate) pending_redraw: RedrawKind,
    pub(crate) scene_presented: bool,
    pub(crate) last_ui_pointer_redraw: std::time::Instant,
    pub(crate) active_case: Option<ActiveCase>,
    /// True while a dock resize handle is dragged (egui chrome reported it last frame).
    pub(crate) docks_resizing: bool,
    /// True while a viewport splitter is being dragged.
    pub(crate) viewport_split_resizing: bool,
}
