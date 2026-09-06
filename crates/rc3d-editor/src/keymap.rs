//! Shared keymap: one table for menus, Studio host input, and the Settings rebind UI.

use std::collections::HashMap;

use winit::keyboard::KeyCode;

use crate::commands::EditorCommand;
use crate::ui::i18n::{t, UiLocale};
use crate::ui::types::EditorDisplayMode;
use rc3d_engine_api::camera::ViewPreset;
use rc3d_gizmo::GizmoMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyAction {
    NewScene,
    OpenScene,
    SaveScene,
    SaveSceneAs,
    Undo,
    Redo,
    GizmoTranslate,
    GizmoRotate,
    GizmoScale,
    ToggleSection,
    FitSelection,
    FitAll,
    DisplayWire,
    DisplayShaded,
    DisplayShadedEdges,
    DisplayHidden,
    DisplayFlatEdges,
    CycleShading,
    CycleIbl,
    Cancel,
    ViewFront,
    ViewRight,
    ViewTop,
    ViewBottom,
    ViewLeft,
    ViewBack,
    ViewIso,
    FlyWalk,
    HideSelected,
    IsolateSelected,
    RevealHidden,
    ToggleLock,
}

impl KeyAction {
    pub const ALL: &'static [KeyAction] = &[
        Self::NewScene,
        Self::OpenScene,
        Self::SaveScene,
        Self::SaveSceneAs,
        Self::Undo,
        Self::Redo,
        Self::GizmoTranslate,
        Self::GizmoRotate,
        Self::GizmoScale,
        Self::ToggleSection,
        Self::FitSelection,
        Self::FitAll,
        Self::DisplayWire,
        Self::DisplayShaded,
        Self::DisplayShadedEdges,
        Self::DisplayHidden,
        Self::DisplayFlatEdges,
        Self::CycleShading,
        Self::CycleIbl,
        Self::Cancel,
        Self::ViewFront,
        Self::ViewRight,
        Self::ViewTop,
        Self::ViewBottom,
        Self::ViewLeft,
        Self::ViewBack,
        Self::ViewIso,
        Self::FlyWalk,
        Self::HideSelected,
        Self::IsolateSelected,
        Self::RevealHidden,
        Self::ToggleLock,
    ];

    pub fn as_id(self) -> &'static str {
        match self {
            Self::NewScene => "new",
            Self::OpenScene => "open",
            Self::SaveScene => "save",
            Self::SaveSceneAs => "save_as",
            Self::Undo => "undo",
            Self::Redo => "redo",
            Self::GizmoTranslate => "gizmo_t",
            Self::GizmoRotate => "gizmo_r",
            Self::GizmoScale => "gizmo_g",
            Self::ToggleSection => "section",
            Self::FitSelection => "fit_sel",
            Self::FitAll => "fit_all",
            Self::DisplayWire => "shade_w",
            Self::DisplayShaded => "shade_s",
            Self::DisplayShadedEdges => "shade_e",
            Self::DisplayHidden => "shade_h",
            Self::DisplayFlatEdges => "shade_l",
            Self::CycleShading => "shade_z",
            Self::CycleIbl => "ibl",
            Self::Cancel => "cancel",
            Self::ViewFront => "view_front",
            Self::ViewRight => "view_right",
            Self::ViewTop => "view_top",
            Self::ViewBottom => "view_bottom",
            Self::ViewLeft => "view_left",
            Self::ViewBack => "view_back",
            Self::ViewIso => "view_iso",
            Self::FlyWalk => "fly",
            Self::HideSelected => "hide",
            Self::IsolateSelected => "isolate",
            Self::RevealHidden => "reveal",
            Self::ToggleLock => "lock",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|a| a.as_id() == id)
    }

    pub fn i18n_key(self) -> &'static str {
        match self {
            Self::NewScene => "file.new",
            Self::OpenScene => "file.open",
            Self::SaveScene => "file.save",
            Self::SaveSceneAs => "file.save_as",
            Self::Undo => "edit.undo",
            Self::Redo => "edit.redo",
            Self::GizmoTranslate => "tools.move",
            Self::GizmoRotate => "tools.rotate",
            Self::GizmoScale => "tools.scale",
            Self::ToggleSection => "tools.section",
            Self::FitSelection => "edit.fit_sel",
            Self::FitAll => "edit.fit_all",
            Self::DisplayWire => "shade.wireframe",
            Self::DisplayShaded => "shade.shaded",
            Self::DisplayShadedEdges => "shade.shaded_edges",
            Self::DisplayHidden => "shade.hidden",
            Self::DisplayFlatEdges => "shade.flat_edges",
            Self::CycleShading => "key.cycle_shading",
            Self::CycleIbl => "key.cycle_ibl",
            Self::Cancel => "key.cancel",
            Self::ViewFront => "view.front",
            Self::ViewRight => "view.right",
            Self::ViewTop => "view.top",
            Self::ViewBottom => "view.bottom",
            Self::ViewLeft => "view.left",
            Self::ViewBack => "view.back",
            Self::ViewIso => "view.iso",
            Self::FlyWalk => "tools.walk",
            Self::HideSelected => "key.hide",
            Self::IsolateSelected => "key.isolate",
            Self::RevealHidden => "key.reveal",
            Self::ToggleLock => "key.lock",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KeyChord {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub code: KeyCode,
}

impl KeyChord {
    pub const fn new(code: KeyCode) -> Self {
        Self {
            ctrl: false,
            shift: false,
            alt: false,
            code,
        }
    }

    pub const fn ctrl(mut self) -> Self {
        self.ctrl = true;
        self
    }

    pub const fn shift(mut self) -> Self {
        self.shift = true;
        self
    }

    pub const fn alt(mut self) -> Self {
        self.alt = true;
        self
    }

    pub fn matches(self, code: KeyCode, ctrl: bool, shift: bool, alt: bool) -> bool {
        normalize_code(self.code) == normalize_code(code)
            && self.ctrl == ctrl
            && self.shift == shift
            && self.alt == alt
    }

    pub fn label(self) -> String {
        let mut parts = Vec::new();
        if self.ctrl {
            parts.push("Ctrl");
        }
        if self.shift {
            parts.push("Shift");
        }
        if self.alt {
            parts.push("Alt");
        }
        parts.push(code_label(self.code));
        parts.join("+")
    }

    pub fn to_pref(self) -> String {
        format!(
            "{}{}{}{}",
            if self.ctrl { "C" } else { "-" },
            if self.shift { "S" } else { "-" },
            if self.alt { "A" } else { "-" },
            code_id(self.code)
        )
    }

    pub fn from_pref(s: &str) -> Option<Self> {
        if s.len() < 4 {
            return None;
        }
        let bytes = s.as_bytes();
        let code = code_from_id(&s[3..])?;
        Some(Self {
            ctrl: bytes[0] == b'C',
            shift: bytes[1] == b'S',
            alt: bytes[2] == b'A',
            code,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Keymap {
    bindings: HashMap<KeyAction, KeyChord>,
}

impl Default for Keymap {
    fn default() -> Self {
        Self::standard()
    }
}

impl Keymap {
    pub fn standard() -> Self {
        let mut bindings = HashMap::new();
        for (action, chord) in DEFAULTS {
            bindings.insert(*action, *chord);
        }
        Self { bindings }
    }

    pub fn resolve(&self, code: KeyCode, ctrl: bool, shift: bool, alt: bool) -> Option<KeyAction> {
        self.bindings
            .iter()
            .find(|(_, c)| c.matches(code, ctrl, shift, alt))
            .map(|(a, _)| *a)
    }

    pub fn chord(&self, action: KeyAction) -> Option<KeyChord> {
        self.bindings.get(&action).copied()
    }

    pub fn set(&mut self, action: KeyAction, chord: KeyChord) {
        self.bindings.retain(|a, c| *a == action || *c != chord);
        self.bindings.insert(action, chord);
    }

    pub fn shortcut_label(&self, action: KeyAction) -> Option<String> {
        self.chord(action).map(|c| c.label())
    }

    pub fn menu_label(&self, loc: UiLocale, action: KeyAction) -> String {
        let name = t(loc, action.i18n_key());
        match self.shortcut_label(action) {
            Some(s) => format!("{name}  ({s})"),
            None => name.to_string(),
        }
    }

    pub fn to_prefs(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut m = serde_json::Map::new();
        for action in KeyAction::ALL {
            if let Some(c) = self.chord(*action) {
                m.insert(
                    action.as_id().to_string(),
                    serde_json::Value::String(c.to_pref()),
                );
            }
        }
        m
    }

    pub fn apply_prefs(&mut self, v: &serde_json::Value) {
        let Some(obj) = v.as_object() else {
            return;
        };
        for (k, val) in obj {
            let Some(action) = KeyAction::from_id(k) else {
                continue;
            };
            let Some(s) = val.as_str() else {
                continue;
            };
            let Some(chord) = KeyChord::from_pref(s) else {
                continue;
            };
            self.set(action, chord);
        }
    }
}

pub fn command_for(action: KeyAction) -> Option<EditorCommand> {
    Some(match action {
        KeyAction::NewScene => EditorCommand::NewScene,
        KeyAction::Undo => EditorCommand::Undo,
        KeyAction::Redo => EditorCommand::Redo,
        KeyAction::GizmoTranslate => EditorCommand::SetGizmoMode(GizmoMode::Translate),
        KeyAction::GizmoRotate => EditorCommand::SetGizmoMode(GizmoMode::Rotate),
        KeyAction::GizmoScale => EditorCommand::SetGizmoMode(GizmoMode::Scale),
        KeyAction::ToggleSection => EditorCommand::ToggleSectionEdit,
        KeyAction::FitSelection => EditorCommand::FitSelection,
        KeyAction::FitAll => EditorCommand::FitAll,
        KeyAction::DisplayWire => EditorCommand::SetDisplayMode(EditorDisplayMode::Wireframe),
        KeyAction::DisplayShaded => EditorCommand::SetDisplayMode(EditorDisplayMode::Shaded),
        KeyAction::DisplayShadedEdges => {
            EditorCommand::SetDisplayMode(EditorDisplayMode::ShadedWithEdges)
        }
        KeyAction::DisplayHidden => EditorCommand::SetDisplayMode(EditorDisplayMode::HiddenLine),
        KeyAction::DisplayFlatEdges => {
            EditorCommand::SetDisplayMode(EditorDisplayMode::FlatWithEdge)
        }
        KeyAction::CycleShading => EditorCommand::CycleDisplayMode,
        KeyAction::CycleIbl => EditorCommand::CycleIbl,
        KeyAction::Cancel => EditorCommand::CancelTool,
        KeyAction::ViewFront => EditorCommand::SetViewPreset(ViewPreset::Front),
        KeyAction::ViewRight => EditorCommand::SetViewPreset(ViewPreset::Right),
        KeyAction::ViewTop => EditorCommand::SetViewPreset(ViewPreset::Top),
        KeyAction::ViewBottom => EditorCommand::SetViewPreset(ViewPreset::Bottom),
        KeyAction::ViewLeft => EditorCommand::SetViewPreset(ViewPreset::Left),
        KeyAction::ViewBack => EditorCommand::SetViewPreset(ViewPreset::Back),
        KeyAction::ViewIso => EditorCommand::SetViewPreset(ViewPreset::Iso),
        KeyAction::FlyWalk => EditorCommand::SetWalkMode(true),
        KeyAction::HideSelected => EditorCommand::HideSelected,
        KeyAction::IsolateSelected => EditorCommand::IsolateSelected,
        KeyAction::RevealHidden => EditorCommand::RevealHidden,
        KeyAction::ToggleLock => EditorCommand::ToggleLockSelected,
        KeyAction::OpenScene | KeyAction::SaveScene | KeyAction::SaveSceneAs => return None,
    })
}

const DEFAULTS: &[(KeyAction, KeyChord)] = &[
    (KeyAction::NewScene, KeyChord::new(KeyCode::KeyN).ctrl()),
    (KeyAction::OpenScene, KeyChord::new(KeyCode::KeyO).ctrl()),
    (KeyAction::SaveScene, KeyChord::new(KeyCode::KeyS).ctrl()),
    (KeyAction::SaveSceneAs, KeyChord::new(KeyCode::KeyS).ctrl().shift()),
    (KeyAction::Undo, KeyChord::new(KeyCode::KeyZ).ctrl()),
    (KeyAction::Redo, KeyChord::new(KeyCode::KeyY).ctrl()),
    (KeyAction::GizmoTranslate, KeyChord::new(KeyCode::KeyT)),
    (KeyAction::GizmoRotate, KeyChord::new(KeyCode::KeyR)),
    (KeyAction::GizmoScale, KeyChord::new(KeyCode::KeyG)),
    (KeyAction::ToggleSection, KeyChord::new(KeyCode::KeyP)),
    (KeyAction::FitSelection, KeyChord::new(KeyCode::KeyF)),
    (KeyAction::FitAll, KeyChord::new(KeyCode::KeyA).shift()),
    (KeyAction::DisplayWire, KeyChord::new(KeyCode::KeyW)),
    (KeyAction::DisplayShaded, KeyChord::new(KeyCode::KeyS)),
    (KeyAction::DisplayShadedEdges, KeyChord::new(KeyCode::KeyE)),
    (KeyAction::DisplayHidden, KeyChord::new(KeyCode::KeyZ).shift()),
    (KeyAction::DisplayFlatEdges, KeyChord::new(KeyCode::KeyL)),
    (KeyAction::CycleShading, KeyChord::new(KeyCode::KeyZ)),
    (KeyAction::CycleIbl, KeyChord::new(KeyCode::KeyI)),
    (KeyAction::Cancel, KeyChord::new(KeyCode::Escape)),
    (KeyAction::ViewFront, KeyChord::new(KeyCode::Digit1)),
    (KeyAction::ViewRight, KeyChord::new(KeyCode::Digit3)),
    (KeyAction::ViewTop, KeyChord::new(KeyCode::Digit7)),
    (KeyAction::ViewIso, KeyChord::new(KeyCode::Digit9)),
    (KeyAction::ViewBack, KeyChord::new(KeyCode::Digit1).ctrl()),
    (KeyAction::ViewLeft, KeyChord::new(KeyCode::Digit3).ctrl()),
    (KeyAction::ViewBottom, KeyChord::new(KeyCode::Digit7).ctrl()),
    (KeyAction::FlyWalk, KeyChord::new(KeyCode::KeyF).shift()),
    (KeyAction::HideSelected, KeyChord::new(KeyCode::KeyH)),
    (KeyAction::IsolateSelected, KeyChord::new(KeyCode::KeyH).shift()),
    (KeyAction::RevealHidden, KeyChord::new(KeyCode::KeyH).alt()),
    (KeyAction::ToggleLock, KeyChord::new(KeyCode::KeyK)),
];

fn normalize_code(code: KeyCode) -> KeyCode {
    match code {
        KeyCode::Numpad1 => KeyCode::Digit1,
        KeyCode::Numpad3 => KeyCode::Digit3,
        KeyCode::Numpad7 => KeyCode::Digit7,
        KeyCode::Numpad9 => KeyCode::Digit9,
        other => other,
    }
}

fn code_label(code: KeyCode) -> &'static str {
    match code {
        KeyCode::Escape => "Esc",
        KeyCode::Digit1 => "1",
        KeyCode::Digit3 => "3",
        KeyCode::Digit7 => "7",
        KeyCode::Digit9 => "9",
        KeyCode::Numpad1 => "Num1",
        KeyCode::Numpad3 => "Num3",
        KeyCode::Numpad7 => "Num7",
        KeyCode::Numpad9 => "Num9",
        KeyCode::KeyA => "A",
        KeyCode::KeyE => "E",
        KeyCode::KeyF => "F",
        KeyCode::KeyG => "G",
        KeyCode::KeyH => "H",
        KeyCode::KeyI => "I",
        KeyCode::KeyK => "K",
        KeyCode::KeyL => "L",
        KeyCode::KeyN => "N",
        KeyCode::KeyO => "O",
        KeyCode::KeyP => "P",
        KeyCode::KeyR => "R",
        KeyCode::KeyS => "S",
        KeyCode::KeyT => "T",
        KeyCode::KeyW => "W",
        KeyCode::KeyY => "Y",
        KeyCode::KeyZ => "Z",
        _ => "Key",
    }
}

fn code_id(code: KeyCode) -> &'static str {
    match code {
        KeyCode::Escape => "Esc",
        KeyCode::Digit1 => "D1",
        KeyCode::Digit3 => "D3",
        KeyCode::Digit7 => "D7",
        KeyCode::Digit9 => "D9",
        KeyCode::Numpad1 => "N1",
        KeyCode::Numpad3 => "N3",
        KeyCode::Numpad7 => "N7",
        KeyCode::Numpad9 => "N9",
        KeyCode::KeyA => "A",
        KeyCode::KeyE => "E",
        KeyCode::KeyF => "F",
        KeyCode::KeyG => "G",
        KeyCode::KeyH => "H",
        KeyCode::KeyI => "I",
        KeyCode::KeyK => "K",
        KeyCode::KeyL => "L",
        KeyCode::KeyN => "N",
        KeyCode::KeyO => "O",
        KeyCode::KeyP => "P",
        KeyCode::KeyR => "R",
        KeyCode::KeyS => "S",
        KeyCode::KeyT => "T",
        KeyCode::KeyW => "W",
        KeyCode::KeyY => "Y",
        KeyCode::KeyZ => "Z",
        _ => "X",
    }
}

fn code_from_id(id: &str) -> Option<KeyCode> {
    Some(match id {
        "Esc" => KeyCode::Escape,
        "D1" => KeyCode::Digit1,
        "D3" => KeyCode::Digit3,
        "D7" => KeyCode::Digit7,
        "D9" => KeyCode::Digit9,
        "N1" => KeyCode::Numpad1,
        "N3" => KeyCode::Numpad3,
        "N7" => KeyCode::Numpad7,
        "N9" => KeyCode::Numpad9,
        "A" => KeyCode::KeyA,
        "E" => KeyCode::KeyE,
        "F" => KeyCode::KeyF,
        "G" => KeyCode::KeyG,
        "H" => KeyCode::KeyH,
        "I" => KeyCode::KeyI,
        "K" => KeyCode::KeyK,
        "L" => KeyCode::KeyL,
        "N" => KeyCode::KeyN,
        "O" => KeyCode::KeyO,
        "P" => KeyCode::KeyP,
        "R" => KeyCode::KeyR,
        "S" => KeyCode::KeyS,
        "T" => KeyCode::KeyT,
        "W" => KeyCode::KeyW,
        "Y" => KeyCode::KeyY,
        "Z" => KeyCode::KeyZ,
        _ => return None,
    })
}
