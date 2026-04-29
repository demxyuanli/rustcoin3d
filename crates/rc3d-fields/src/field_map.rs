use std::collections::{HashSet, VecDeque};

use rc3d_core::{FieldId, NodeId};
use slotmap::SlotMap;

use crate::field_value::FieldValue;

pub struct FieldMap {
    entries: SlotMap<FieldId, FieldEntry>,
}

pub struct FieldEntry {
    pub value: FieldValue,
    pub dirty: bool,
    pub owner: NodeId,
    pub field_index: u16,
    pub connections: Vec<FieldId>,
}

impl FieldMap {
    pub fn new() -> Self {
        Self {
            entries: SlotMap::with_key(),
        }
    }

    pub fn insert(&mut self, owner: NodeId, field_index: u16, value: FieldValue) -> FieldId {
        self.entries.insert(FieldEntry {
            value,
            dirty: false,
            owner,
            field_index,
            connections: Vec::new(),
        })
    }

    pub fn get(&self, id: FieldId) -> Option<&FieldValue> {
        self.entries.get(id).map(|e| &e.value)
    }

    pub fn get_entry(&self, id: FieldId) -> Option<&FieldEntry> {
        self.entries.get(id)
    }

    pub fn set(&mut self, id: FieldId, value: FieldValue) {
        if let Some(entry) = self.entries.get_mut(id) {
            entry.value = value;
            entry.dirty = true;
        }
        self.propagate_dirty_from(id);
    }

    pub fn is_dirty(&self, id: FieldId) -> bool {
        self.entries.get(id).is_some_and(|e| e.dirty)
    }

    pub fn clear_dirty(&mut self, id: FieldId) {
        if let Some(entry) = self.entries.get_mut(id) {
            entry.dirty = false;
        }
    }

    pub fn connect(&mut self, from: FieldId, to: FieldId) {
        if let Some(entry) = self.entries.get_mut(from) {
            entry.connections.push(to);
        }
    }

    /// Copy `id`'s value along outgoing `connect` edges (transitive), marking targets dirty.
    pub fn propagate(&mut self, id: FieldId) {
        let value = self.entries.get(id).map(|e| e.value.clone());
        let Some(value) = value else { return };
        let mut q = VecDeque::new();
        let mut seen = HashSet::new();
        q.push_back(id);
        seen.insert(id);
        while let Some(fid) = q.pop_front() {
            let conns = self.entries.get(fid).map(|e| e.connections.clone()).unwrap_or_default();
            for target in conns {
                if seen.insert(target) {
                    if let Some(entry) = self.entries.get_mut(target) {
                        entry.value = value.clone();
                        entry.dirty = true;
                    }
                    q.push_back(target);
                }
            }
        }
    }

    fn propagate_dirty_from(&mut self, start: FieldId) {
        let mut q = VecDeque::new();
        let mut visited = HashSet::new();
        q.push_back(start);
        visited.insert(start);
        while let Some(fid) = q.pop_front() {
            let conns = self.entries.get(fid).map(|e| e.connections.clone()).unwrap_or_default();
            for target in conns {
                if visited.insert(target) {
                    if let Some(entry) = self.entries.get_mut(target) {
                        entry.dirty = true;
                    }
                    q.push_back(target);
                }
            }
        }
    }

    pub fn any_dirty(&self) -> bool {
        self.entries.iter().any(|(_, e)| e.dirty)
    }

    pub fn mark_all_dirty(&mut self) {
        for (_, e) in self.entries.iter_mut() {
            e.dirty = true;
        }
    }
}

impl Default for FieldMap {
    fn default() -> Self {
        Self::new()
    }
}
