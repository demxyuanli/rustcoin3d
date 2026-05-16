use rc3d_render::background::{BgMode, ImageFit};

#[derive(Clone, Debug)]
pub struct BackgroundSettings {
    pub mode: BgMode,
    pub image_fit: ImageFit,
    pub image_path: Option<String>,
    pub clear_color: [f32; 4],
}

impl Default for BackgroundSettings {
    fn default() -> Self {
        Self {
            mode: BgMode::Solid,
            image_fit: ImageFit::Stretch,
            image_path: None,
            clear_color: [0.15, 0.15, 0.15, 1.0],
        }
    }
}

impl From<BackgroundSettings> for rc3d_render::background::BgSettings {
    fn from(bg: BackgroundSettings) -> Self {
        rc3d_render::background::BgSettings {
            mode: bg.mode,
            image_fit: bg.image_fit,
            top_color: bg.clear_color,
            bot_color: bg.clear_color,
            image_path: bg.image_path,
            cube_faces: [None, None, None, None, None, None],
        }
    }
}
