//! Font rasterization for world-space labels (annotations, Text3).

use glyphon::SwashCache;

use glyphon::FontSystem;

use crate::font_loader::{self, LabelFont};

pub struct WorldLabelFont {
    pub font_system: FontSystem,
    pub swash_cache: SwashCache,
    label_font: LabelFont,
}

impl WorldLabelFont {
    pub fn new() -> Self {
        let (font_system, label_font) = font_loader::new_label_font_system();
        log::info!("World label font: {}", label_font.family_name);
        Self {
            font_system,
            swash_cache: SwashCache::new(),
            label_font,
        }
    }

    pub fn label_attrs(&self) -> glyphon::Attrs<'static> {
        self.label_font.attrs()
    }

    #[allow(dead_code)]
    pub fn family_name(&self) -> &str {
        &self.label_font.family_name
    }
}

impl Default for WorldLabelFont {
    fn default() -> Self {
        Self::new()
    }
}
