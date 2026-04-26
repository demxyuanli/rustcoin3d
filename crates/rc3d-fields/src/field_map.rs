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

    pub fn propagate(&mut self, id: FieldId) {
        let value = self.entries.get(id).map(|e| e.value.clone());
        let Some(value) = value else { return };
        let connections = self
            .entries
            .get(id)
            .map(|e| e.connections.clone())
            .unwrap_or_default();
        for target in connections {
            self.set(target, value.clone());
        }
    }
}

impl Default for FieldMap {
    fn default() -> Self {
        Self::new()
    }
}
