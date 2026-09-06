use std::fs;
use std::path::PathBuf;

use rc3d_editor::{BottomTab, Keymap, SideTab, UiLocale, UiTheme, Workspace};
use rc3d_render::viewport::LayoutMode;

/// Read a JSON field as `f32` (`serde_json` has no native f32).
fn f32_at(v: &serde_json::Value, key: &str) -> Option<f32> {
    v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32)
}

/// Read an optional JSON field as `Option<f32>`.
fn opt_f32_at(v: &serde_json::Value, key: &str) -> Option<Option<f32>> {
    f32_at(v, key).map(Some)
}

pub struct UiPrefs {
    pub theme: UiTheme,
    pub locale: UiLocale,
    pub keymap: Keymap,
    pub side_tab: Option<SideTab>,
    pub bottom_tab: Option<BottomTab>,
    pub workspace: Workspace,
    pub tool_strip_pos: Option<[f32; 2]>,
    pub side_dock_width: Option<f32>,
    pub bottom_dock_height: Option<f32>,
    pub inspector_ratio: Option<f32>,
    pub viewport_layout_mode: LayoutMode,
    pub viewport_h_split: f32,
    pub viewport_v_split: f32,
    pub last_document: Option<PathBuf>,
    pub last_folder: Option<PathBuf>,
    pub recent: Vec<PathBuf>,
}

impl Default for UiPrefs {
    fn default() -> Self {
        Self {
            theme: UiTheme::Dark,
            locale: UiLocale::En,
            keymap: Keymap::standard(),
            side_tab: Some(SideTab::Hierarchy),
            bottom_tab: Some(BottomTab::Document),
            workspace: Workspace::Model,
            tool_strip_pos: None,
            side_dock_width: None,
            bottom_dock_height: None,
            inspector_ratio: None,
            viewport_layout_mode: LayoutMode::Single,
            viewport_h_split: 0.5,
            viewport_v_split: 0.5,
            last_document: None,
            last_folder: None,
            recent: Vec::new(),
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
    let v = &v;
    let mut prefs = UiPrefs {
        theme: UiTheme::from_id(v.get("theme").and_then(|x| x.as_str()).unwrap_or("")),
        locale: UiLocale::from_id(v.get("locale").and_then(|x| x.as_str()).unwrap_or("")),
        keymap: Keymap::standard(),
        ..UiPrefs::default()
    };
    if let Some(km) = v.get("keymap") {
        prefs.keymap.apply_prefs(km);
    }
    prefs.side_tab = parse_side(v.get("side_tab").and_then(|x| x.as_str()));
    let bottom_raw = v.get("bottom_tab").and_then(|x| x.as_str());
    prefs.bottom_tab = parse_bottom(bottom_raw);
    // Old prefs kept History/Assets on the bottom dock; migrate to the side dock.
    match bottom_raw {
        Some("history") => prefs.side_tab = Some(SideTab::History),
        Some("assets") => prefs.side_tab = Some(SideTab::Assets),
        _ => {}
    }
    prefs.workspace = match v.get("workspace").and_then(|x| x.as_str()) {
        Some("lookdev") => Workspace::LookDev,
        Some("compositor") => Workspace::Compositor,
        _ => Workspace::Model,
    };
    if let (Some(x), Some(y)) = (f32_at(v, "strip_x"), f32_at(v, "strip_y")) {
        prefs.tool_strip_pos = Some([x, y]);
    }
    prefs.side_dock_width = opt_f32_at(v, "side_dock_width").unwrap_or(None);
    prefs.inspector_ratio = opt_f32_at(v, "inspector_ratio")
        .unwrap_or(None)
        .map(|x| x.clamp(0.2, 0.8));
    prefs.bottom_dock_height = opt_f32_at(v, "bottom_dock_height").unwrap_or(None);
    prefs.viewport_layout_mode = match v.get("viewport_layout_mode").and_then(|x| x.as_str()) {
        Some("quad") => LayoutMode::Quad,
        Some("leftright") => LayoutMode::LeftRight,
        Some("topbottom") => LayoutMode::TopBottom,
        _ => LayoutMode::Single,
    };
    prefs.viewport_h_split = f32_at(v, "viewport_h_split")
        .unwrap_or(0.5)
        .clamp(0.1, 0.9);
    prefs.viewport_v_split = f32_at(v, "viewport_v_split")
        .unwrap_or(0.5)
        .clamp(0.1, 0.9);
    prefs.last_document = v
        .get("last_document")
        .and_then(|x| x.as_str())
        .map(PathBuf::from);
    prefs.last_folder = v
        .get("last_folder")
        .and_then(|x| x.as_str())
        .map(PathBuf::from);
    if let Some(arr) = v.get("recent").and_then(|x| x.as_array()) {
        prefs.recent = arr
            .iter()
            .filter_map(|x| x.as_str().map(PathBuf::from))
            .collect();
    }
    prefs
}

pub fn save(prefs: &UiPrefs) {
    let Some(path) = prefs_path() else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let keymap = serde_json::Value::Object(prefs.keymap.to_prefs());
    let recent: Vec<serde_json::Value> = prefs
        .recent
        .iter()
        .map(|p| serde_json::Value::String(p.to_string_lossy().into_owned()))
        .collect();
    let mut obj = serde_json::Map::new();
    obj.insert("theme".into(), prefs.theme.as_id().into());
    obj.insert("locale".into(), prefs.locale.as_id().into());
    obj.insert("keymap".into(), keymap);
    obj.insert(
        "side_tab".into(),
        serde_json::Value::String(side_id(prefs.side_tab).to_string()),
    );
    obj.insert(
        "bottom_tab".into(),
        serde_json::Value::String(bottom_id(prefs.bottom_tab).to_string()),
    );
    obj.insert(
        "workspace".into(),
        serde_json::Value::String(
            match prefs.workspace {
                Workspace::Model => "model",
                Workspace::LookDev => "lookdev",
                Workspace::Compositor => "compositor",
            }
            .into(),
        ),
    );
    if let Some([x, y]) = prefs.tool_strip_pos {
        obj.insert("strip_x".into(), serde_json::json!(x));
        obj.insert("strip_y".into(), serde_json::json!(y));
    }
    if let Some(w) = prefs.side_dock_width {
        obj.insert("side_dock_width".into(), serde_json::json!(w));
    }
    if let Some(r) = prefs.inspector_ratio {
        obj.insert("inspector_ratio".into(), serde_json::json!(r));
    }
    if let Some(h) = prefs.bottom_dock_height {
        obj.insert("bottom_dock_height".into(), serde_json::json!(h));
    }
    obj.insert(
        "viewport_layout_mode".into(),
        serde_json::Value::String(
            match prefs.viewport_layout_mode {
                LayoutMode::Single => "single",
                LayoutMode::Quad => "quad",
                LayoutMode::LeftRight => "leftright",
                LayoutMode::TopBottom => "topbottom",
            }
            .into(),
        ),
    );
    obj.insert("viewport_h_split".into(), serde_json::json!(prefs.viewport_h_split));
    obj.insert("viewport_v_split".into(), serde_json::json!(prefs.viewport_v_split));
    if let Some(doc) = &prefs.last_document {
        obj.insert(
            "last_document".into(),
            serde_json::Value::String(doc.to_string_lossy().into_owned()),
        );
    }
    if let Some(folder) = &prefs.last_folder {
        obj.insert(
            "last_folder".into(),
            serde_json::Value::String(folder.to_string_lossy().into_owned()),
        );
    }
    obj.insert("recent".into(), serde_json::Value::Array(recent));
    let _ = fs::write(path, serde_json::Value::Object(obj).to_string());
}

pub fn push_recent(recent: &mut Vec<PathBuf>, path: PathBuf) {
    recent.retain(|p| p != &path);
    recent.insert(0, path);
    recent.truncate(12);
}

pub fn autosave_sidecar(path: &std::path::Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".autosave.json");
    PathBuf::from(s)
}

pub fn write_autosave(graph: &rc3d_scene::SceneGraph, path: &std::path::Path) {
    let side = autosave_sidecar(path);
    let _ = rc3d_editor::document::save_native_scene(graph, &side);
}

fn side_id(tab: Option<SideTab>) -> &'static str {
    match tab {
        Some(SideTab::Hierarchy) => "hierarchy",
        Some(SideTab::Render) => "render",
        Some(SideTab::History) => "history",
        Some(SideTab::Assets) => "assets",
        None => "none",
    }
}

fn parse_side(s: Option<&str>) -> Option<SideTab> {
    match s {
        Some("hierarchy") => Some(SideTab::Hierarchy),
        // The standalone Inspector tab was merged into Hierarchy.
        Some("inspector") => Some(SideTab::Hierarchy),
        Some("render") => Some(SideTab::Render),
        Some("history") => Some(SideTab::History),
        Some("assets") => Some(SideTab::Assets),
        Some("none") => None,
        _ => Some(SideTab::Hierarchy),
    }
}

fn bottom_id(tab: Option<BottomTab>) -> &'static str {
    match tab {
        Some(BottomTab::Document) => "document",
        Some(BottomTab::Compositor) => "compositor",
        None => "none",
    }
}

fn parse_bottom(s: Option<&str>) -> Option<BottomTab> {
    match s {
        Some("document") => Some(BottomTab::Document),
        Some("compositor") => Some(BottomTab::Compositor),
        // Console tab removed; fall back to Document.
        Some("console") | Some("history") | Some("assets") => Some(BottomTab::Document),
        Some("none") => None,
        _ => Some(BottomTab::Document),
    }
}
