use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::window::{CursorIcon, ResizeDirection, Theme, Window, WindowAttributes};

pub fn window_attributes() -> WindowAttributes {
    let attrs = WindowAttributes::default()
        .with_title("rustcoin3d Studio")
        .with_decorations(false)
        .with_resizable(true)
        .with_min_inner_size(LogicalSize::new(960.0, 640.0))
        .with_inner_size(LogicalSize::new(1440.0, 900.0));
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
