//! Load fonts for world-space labels and HUD text (glyphon / cosmic-text).

use std::path::{Path, PathBuf};

use glyphon::{Attrs, Family, FontSystem};
use log::{info, warn};
use rc3d_scene::node_data::FontStyle;

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
        self.attrs.clone()
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
    load_generic_families(font_system);

    active
}

/// Glyphon attrs for a Coin3D `SoFont` name + style.
pub fn attrs_from_font(name: &str, style: FontStyle) -> Attrs<'_> {
    let family = if !name.is_empty() {
        Family::Name(name)
    } else {
        match style {
            FontStyle::Sans => Family::SansSerif,
            FontStyle::Serif => Family::Serif,
            FontStyle::Typewriter => Family::Monospace,
        }
    };
    Attrs::new().family(family)
}

/// Load `name` into the font db if it is not already present.
pub fn ensure_named_font(font_system: &mut FontSystem, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    if family_loaded(font_system, name) {
        return true;
    }
    for path in named_font_paths(name) {
        if load_font_file(font_system, &path) {
            info!("Installed SoFont face: {}", path.display());
            return true;
        }
    }
    false
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

fn family_loaded(font_system: &FontSystem, name: &str) -> bool {
    font_system.db().faces().any(|f| {
        f.families
            .iter()
            .any(|(family, _)| family.eq_ignore_ascii_case(name))
    })
}

fn named_font_paths(name: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let file_stem = name.replace(' ', "");
    #[cfg(windows)]
    {
        let windir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
        let fonts = PathBuf::from(windir).join("Fonts");
        let mapped = match name.to_ascii_lowercase().as_str() {
            "times" | "times new roman" | "timesnewroman" => Some("times.ttf"),
            "courier" | "courier new" | "couriernew" => Some("cour.ttf"),
            "consolas" => Some("consola.ttf"),
            "arial" => Some("arial.ttf"),
            "georgia" => Some("georgia.ttf"),
            "verdana" => Some("verdana.ttf"),
            "segoe ui" | "segoeui" => Some("segoeui.ttf"),
            _ => None,
        };
        if let Some(file) = mapped {
            paths.push(fonts.join(file));
        }
        paths.push(fonts.join(format!("{file_stem}.ttf")));
        paths.push(fonts.join(format!("{file_stem}.otf")));
        paths.push(fonts.join(format!("{name}.ttf")));
    }
    #[cfg(not(windows))]
    {
        let _ = file_stem;
        paths.push(PathBuf::from("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"));
        paths.push(PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"));
        paths.push(PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"));
    }
    paths
}

fn load_generic_families(font_system: &mut FontSystem) {
    #[cfg(windows)]
    {
        let windir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
        let fonts = PathBuf::from(windir).join("Fonts");
        if load_font_file(font_system, &fonts.join("times.ttf")) {
            if let Some(name) = family_name_for_path(font_system, &fonts.join("times.ttf")) {
                font_system.db_mut().set_serif_family(&name);
            }
        }
        let mono = fonts.join("consola.ttf");
        let mono_alt = fonts.join("cour.ttf");
        let mono_path = if mono.is_file() { mono } else { mono_alt };
        if load_font_file(font_system, &mono_path) {
            if let Some(name) = family_name_for_path(font_system, &mono_path) {
                font_system.db_mut().set_monospace_family(&name);
            }
        }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let serif = PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf");
        if load_font_file(font_system, &serif) {
            if let Some(name) = family_name_for_path(font_system, &serif) {
                font_system.db_mut().set_serif_family(&name);
            }
        }
        let mono = PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf");
        if load_font_file(font_system, &mono) {
            if let Some(name) = family_name_for_path(font_system, &mono) {
                font_system.db_mut().set_monospace_family(&name);
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        let serif = PathBuf::from("/System/Library/Fonts/Supplemental/Times New Roman.ttf");
        if load_font_file(font_system, &serif) {
            if let Some(name) = family_name_for_path(font_system, &serif) {
                font_system.db_mut().set_serif_family(&name);
            }
        }
        let mono = PathBuf::from("/System/Library/Fonts/Supplemental/Courier New.ttf");
        if load_font_file(font_system, &mono) {
            if let Some(name) = family_name_for_path(font_system, &mono) {
                font_system.db_mut().set_monospace_family(&name);
            }
        }
    }
}
