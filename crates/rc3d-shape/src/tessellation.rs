//! Tessellation cache keyed by solid + config + orientation.
//! (mesh processing removed — stub types retained for API compatibility)

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use crate::topo::{FaceKey, Orientation, SolidKey};

/// Face triangle range placeholder (mesh processing removed).
#[derive(Debug, Clone, Default)]
pub struct FaceTriRange {
    pub offset: usize,
    pub count: usize,
}

/// Mesh result placeholder (mesh processing removed).
#[derive(Debug, Clone, Default)]
pub struct MeshResult {
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

/// Shell mesh report placeholder (mesh processing removed).
#[derive(Debug, Clone, Default)]
pub struct ShellMeshReport;

#[derive(Debug, Clone)]
pub struct TessEntry {
    pub mesh: MeshResult,
    pub face_tri_ranges: HashMap<FaceKey, FaceTriRange>,
    pub face_split_viable: bool,
    pub report: ShellMeshReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TessKey {
    pub solid_key: SolidKey,
    pub config_hash: u64,
    pub orientation: Orientation,
}

impl Hash for TessKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.solid_key.hash(state);
        self.config_hash.hash(state);
        match self.orientation {
            Orientation::Forward => 0u8.hash(state),
            Orientation::Reversed => 1u8.hash(state),
            Orientation::Internal => 2u8.hash(state),
            Orientation::External => 3u8.hash(state),
        }
    }
}

#[derive(Debug, Default)]
pub struct TessellationCache {
    pub entries: HashMap<TessKey, TessEntry>,
}

impl TessellationCache {
    pub fn new() -> Self { Self { entries: HashMap::new() } }
    pub fn get(&self, key: &TessKey) -> Option<&TessEntry> { self.entries.get(key) }
    pub fn insert(&mut self, key: TessKey, entry: TessEntry) { self.entries.insert(key, entry); }
    pub fn invalidate_solid(&mut self, solid_key: SolidKey) {
        self.entries.retain(|k, _| k.solid_key != solid_key);
    }
}
