use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::window::{CursorIcon, ResizeDirection, Theme, Window, WindowAttributes};

pub fn window_attributes() -> WindowAttributes {
    // Hidden at creation: the client area only gets real pixels on the first
    // present, until then DWM would show a white un-presented swapchain plus
    // the dark Mica backdrop edge (white-top / black-strip startup flash).
    // The window is revealed in `present.rs` right after the first full frame.
    let attrs = WindowAttributes::default()
        .with_title("rustcoin3d Studio")
        .with_decorations(false)
        .with_resizable(true)
        .with_min_inner_size(LogicalSize::new(960.0, 640.0))
        .with_inner_size(LogicalSize::new(1440.0, 900.0))
        .with_visible(false);
    #[cfg(windows)]
    {
        use winit::platform::windows::{
            BackdropType, CornerPreference, WindowAttributesExtWindows,
        };
        attrs
            .with_undecorated_shadow(true)
            .with_system_backdrop(BackdropType::MainWindow)
            .with_corner_preference(CornerPreference::Round)
    }
    #[cfg(not(windows))]
    {
        attrs
    }
}

pub fn apply_after_create(window: &Window) {
    window.set_theme(Some(Theme::Dark));
    center_on_primary_monitor(window);
}

/// Center the undecorated window on the primary monitor (winit 0.30 has no
/// `Position::Centered`; the OS would otherwise cascade it arbitrarily).
fn center_on_primary_monitor(window: &Window) {
    let Some(monitor) = window.current_monitor().or_else(|| window.primary_monitor())
    else {
        return;
    };
    let mon_pos = monitor.position();
    let mon_size = monitor.size();
    let win = window.outer_size();
    let x = mon_pos.x + ((mon_size.width as i32 - win.width as i32) / 2).max(0);
    let y = mon_pos.y + ((mon_size.height as i32 - win.height as i32) / 2).max(0);
    window.set_outer_position(winit::dpi::PhysicalPosition::new(x, y));
}

pub fn resize_direction(
    pos: PhysicalPosition<f64>,
    size: PhysicalSize<u32>,
    scale: f64,
    maximized: bool,
) -> Option<ResizeDirection> {
    if maximized || size.width < 2 || size.height < 2 {
        return None;
    }
    let border = (6.0 * scale).max(4.0);
    let x = pos.x;
    let y = pos.y;
    let w = size.width as f64;
    let h = size.height as f64;
    let left = x <= border;
    let right = x >= w - border;
    let top = y <= border;
    let bottom = y >= h - border;
    match (left, right, top, bottom) {
        (true, false, true, false) => Some(ResizeDirection::NorthWest),
        (false, true, true, false) => Some(ResizeDirection::NorthEast),
        (true, false, false, true) => Some(ResizeDirection::SouthWest),
        (false, true, false, true) => Some(ResizeDirection::SouthEast),
        (true, false, false, false) => Some(ResizeDirection::West),
        (false, true, false, false) => Some(ResizeDirection::East),
        (false, false, true, false) => Some(ResizeDirection::North),
        (false, false, false, true) => Some(ResizeDirection::South),
        _ => None,
    }
}

pub fn cursor_for_resize(dir: ResizeDirection) -> CursorIcon {
    CursorIcon::from(dir)
}
