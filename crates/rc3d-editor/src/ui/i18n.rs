//! UI locale catalogs. English is the fallback; other locales live in `i18n/*.json`.

use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum UiLocale {
    #[default]
    En,
    ZhHans,
}

impl UiLocale {
    pub fn as_id(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::ZhHans => "zh-Hans",
        }
    }

    pub fn from_id(id: &str) -> Self {
        match id {
            "zh" | "zh-CN" | "zh-Hans" | "zh_hans" => Self::ZhHans,
            _ => Self::En,
        }
    }
}

fn parse_catalog(json: &str) -> HashMap<String, String> {
    serde_json::from_str(json).unwrap_or_default()
}

fn catalog(locale: UiLocale) -> &'static HashMap<String, String> {
    match locale {
        UiLocale::En => {
            static EN: OnceLock<HashMap<String, String>> = OnceLock::new();
            EN.get_or_init(|| parse_catalog(include_str!("../../i18n/en.json")))
        }
        UiLocale::ZhHans => {
            static ZH: OnceLock<HashMap<String, String>> = OnceLock::new();
            ZH.get_or_init(|| parse_catalog(include_str!("../../i18n/zh_hans.json")))
        }
    }
}

pub fn t(locale: UiLocale, key: &'static str) -> &'static str {
    if let Some(s) = catalog(locale).get(key) {
        return s.as_str();
    }
    if locale != UiLocale::En {
        if let Some(s) = catalog(UiLocale::En).get(key) {
            return s.as_str();
        }
    }
    key
}

pub fn tf(locale: UiLocale, key: &'static str, n: u32) -> String {
    t(locale, key).replace("{n}", &n.to_string())
}
