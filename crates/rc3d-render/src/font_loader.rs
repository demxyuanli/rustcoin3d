//! Load fonts for world-space labels and HUD text (glyphon / cosmic-text).

use std::path::{Path, PathBuf};

use glyphon::{Attrs, Family, FontSystem};
use log::{info, warn};

/// Optional override: path to a `.ttf` / `.otf` file.
/// Example: `set RC3D_FONT_PATH=C:\Windows\Fonts\arial.ttf`
pub const ENV_FONT_PATH: &str = "RC3D_FONT_PATH";

/// Optional directory; tries `label.ttf`, `Roboto-Regular.ttf`, `DejaVuSans.ttf`.
pub const ENV_FONT_DIR: &str = "RC3D_FONT_DIR";

/// Active label/HUD font attributes (use after [`configure_font_system`]).
#[derive(Clone)]
pub struct LabelFont {
    attrs: Attrs<'static>,
    #[allow(dead_code)]
    pub family_name: String,
}

impl LabelFont {
    pub fn attrs(&self) -> Attrs<'static> {
        self.attrs
    }
}

/// Create a [`FontSystem`] with platform / env fonts installed for labels.
pub fn new_label_font_system() -> (FontSystem, LabelFont) {
    let mut font_system = FontSystem::new();
    let family_name = configure_font_system(&mut font_system);
    let label = LabelFont {
        attrs: Attrs::new().family(Family::SansSerif),
        family_name,
    };
    (font_system, label)
}

/// Load custom fonts into an existing system; returns active sans-serif family name.
pub fn configure_font_system(font_system: &mut FontSystem) -> String {
    let mut active = font_system
        .db()
        .faces()
        .find(|f| !f.monospaced)
        .and_then(|f| f.families.first().map(|(name, _)| name.clone()))
        .unwrap_or_else(|| "Sans Serif".to_string());

    for path in font_search_paths() {
        if load_font_file(font_system, &path) {
            if let Some(name) = family_name_for_path(font_system, &path) {
                font_system.db_mut().set_sans_serif_family(&name);
                active = name;
                info!("Installed label font: {}", path.display());
                break;
            }
        }
    }

    active
}

fn font_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(p) = std::env::var(ENV_FONT_PATH) {
        if !p.is_empty() {
            paths.push(PathBuf::from(p));
        }
    }
    if let Ok(dir) = std::env::var(ENV_FONT_DIR) {
        if !dir.is_empty() {
            let d = PathBuf::from(dir);
            for name in ["label.ttf", "Roboto-Regular.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
            {
                paths.push(d.join(name));
            }
        }
    }
    // Project-bundled font directory (search relative to working dir)
    let assets_fonts = PathBuf::from("assets/fonts");
    if assets_fonts.is_dir() {
        for name in ["label.ttf", "Roboto-Regular.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
        {
            let p = assets_fonts.join(name);
            if p.is_file() {
                paths.push(p);
            }
        }
    }
    paths.extend(platform_font_candidates());
    paths
}

fn platform_font_candidates() -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        let windir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
        let fonts = PathBuf::from(windir).join("Fonts");
        vec![
            fonts.join("arial.ttf"),
            fonts.join("segoeui.ttf"),
            fonts.join("calibri.ttf"),
        ]
    }
    #[cfg(target_os = "macos")]
    {
        vec![
            PathBuf::from("/System/Library/Fonts/Supplemental/Arial.ttf"),
            PathBuf::from("/Library/Fonts/Arial.ttf"),
        ]
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        vec![
            PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            PathBuf::from("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        ]
    }
    #[cfg(not(any(windows, unix)))]
    {
        Vec::new()
    }
}

fn load_font_file(font_system: &mut FontSystem, path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    match font_system.db_mut().load_font_file(path) {
        Ok(()) => true,
        Err(e) => {
            warn!("Failed to load font {}: {e}", path.display());
            false
        }
    }
}

fn family_name_for_path(font_system: &FontSystem, path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    let stem_norm = stem.replace(['-', '_'], "").to_lowercase();
    for face in font_system.db().faces() {
        for (family, _lang) in &face.families {
            let family_norm = family.replace(['-', '_'], "").to_lowercase();
            if family_norm == stem_norm || family.eq_ignore_ascii_case(stem) {
                return Some(family.clone());
            }
        }
    }
    font_system
        .db()
        .faces()
        .last()
        .and_then(|f| f.families.first().map(|(name, _)| name.clone()))
}
