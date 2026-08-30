use std::fs;
use std::path::PathBuf;

use rc3d_editor::{UiLocale, UiTheme};

pub struct UiPrefs {
    pub theme: UiTheme,
    pub locale: UiLocale,
}

impl Default for UiPrefs {
    fn default() -> Self {
        Self {
            theme: UiTheme::Dark,
            locale: UiLocale::En,
        }
    }
}

fn prefs_path() -> Option<PathBuf> {
    let appdata = std::env::var_os("APPDATA")?;
    Some(PathBuf::from(appdata).join("rustcoin3d").join("ui-prefs.json"))
}

pub fn load() -> UiPrefs {
    let Some(path) = prefs_path() else {
        return UiPrefs::default();
    };
    let Ok(text) = fs::read_to_string(path) else {
        return UiPrefs::default();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return UiPrefs::default();
    };
    UiPrefs {
        theme: UiTheme::from_id(v.get("theme").and_then(|x| x.as_str()).unwrap_or("")),
        locale: UiLocale::from_id(v.get("locale").and_then(|x| x.as_str()).unwrap_or("")),
    }
}

pub fn save(theme: UiTheme, locale: UiLocale) {
    let Some(path) = prefs_path() else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let text = format!(
        "{{\n  \"theme\": \"{}\",\n  \"locale\": \"{}\"\n}}\n",
        theme.as_id(),
        locale.as_id()
    );
    let _ = fs::write(path, text);
}
