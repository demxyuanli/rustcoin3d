//! Part21 conformance metadata from HEADER.

use crate::step::header::HeaderInfo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImplementationLevel {
    pub version: u8,
    pub conformance: u8,
}

impl ImplementationLevel {
    pub fn prefers_external_complex(&self) -> bool {
        self.conformance == 2
    }
}

pub fn parse_implementation_level(s: &str) -> Option<ImplementationLevel> {
    let t = s.trim().trim_matches('\'');
    let (ver, conf) = t.split_once(';')?;
    let version: u8 = ver.trim().parse().ok()?;
    let conformance: u8 = conf.trim().parse().ok()?;
    if !(1..=3).contains(&version) || !(1..=2).contains(&conformance) {
        return None;
    }
    Some(ImplementationLevel {
        version,
        conformance,
    })
}

pub fn from_header(header: &HeaderInfo) -> Option<ImplementationLevel> {
    for s in &header.file_description {
        if let Some(level) = parse_implementation_level(s) {
            return Some(level);
        }
    }
    for (_, args) in &header.extra {
        if args.contains(';') {
            if let Some(level) = parse_implementation_level(args) {
                return Some(level);
            }
        }
    }
    None
}
