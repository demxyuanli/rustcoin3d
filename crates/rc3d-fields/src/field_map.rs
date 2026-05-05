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

    pub fn remove(&mut self, id: FieldId) -> bool {
        if self.entries.contains_key(id) {
            self.entries.remove(id);
            true
        } else {
            false
        }
    }

    pub fn connect(&mut self, from: FieldId, to: FieldId) {
        if let Some(entry) = self.entries.get_mut(from) {
            entry.connections.push(to);
        }
    }

    /// Copy `id`'s value along outgoing `connect` edges (transitive), marking targets dirty.
    /// Note: propagate() deep-clones values. For large arrays, consider Arc<FieldValue>.
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

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::{Vec2, Vec3};

    fn make_node() -> NodeId {
        NodeId::default()
    }

    // ── Insert / Get / Set ──

    #[test]
    fn test_insert_and_get() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let id = fm.insert(n, 0, FieldValue::Float(1.0));
        assert_eq!(fm.get(id), Some(&FieldValue::Float(1.0)));
    }

    #[test]
    fn test_set_updates_value_and_dirties() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let id = fm.insert(n, 0, FieldValue::Int32(42));
        assert!(!fm.is_dirty(id));

        fm.set(id, FieldValue::Int32(99));
        assert_eq!(fm.get(id), Some(&FieldValue::Int32(99)));
        assert!(fm.is_dirty(id));
    }

    #[test]
    fn test_clear_dirty() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let id = fm.insert(n, 0, FieldValue::Int32(1));
        fm.set(id, FieldValue::Int32(2));
        assert!(fm.is_dirty(id));

        fm.clear_dirty(id);
        assert!(!fm.is_dirty(id));
    }

    #[test]
    fn test_remove_existing() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let id = fm.insert(n, 0, FieldValue::Float(3.14));
        assert!(fm.remove(id));
        assert!(fm.get(id).is_none());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let id = fm.insert(n, 0, FieldValue::Bool(true));
        fm.remove(id);
        assert!(!fm.remove(id)); // second remove returns false
    }

    // ── String / Binary / Float64 ──

    #[test]
    fn test_insert_string_value() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let id = fm.insert(n, 1, FieldValue::String("part_123".into()));
        assert_eq!(fm.get(id), Some(&FieldValue::String("part_123".into())));
    }

    #[test]
    fn test_insert_binary_value() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let id = fm.insert(n, 2, FieldValue::Binary(data.clone()));
        assert_eq!(fm.get(id), Some(&FieldValue::Binary(data)));
    }

    #[test]
    fn test_insert_float64_value() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let id = fm.insert(n, 0, FieldValue::Float64(std::f64::consts::PI));
        match fm.get(id) {
            Some(FieldValue::Float64(v)) => assert!((v - std::f64::consts::PI).abs() < 1e-10),
            other => panic!("expected Float64, got {:?}", other),
        }
    }

    // ── Connect / Propagate ──

    #[test]
    fn test_propagate_copies_value_to_connected() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let a = fm.insert(n, 0, FieldValue::Float(10.0));
        let b = fm.insert(n, 1, FieldValue::Float(0.0));

        fm.connect(a, b);
        fm.propagate(a);

        assert_eq!(fm.get(b), Some(&FieldValue::Float(10.0)));
        assert!(fm.is_dirty(b));
    }

    #[test]
    fn test_propagate_transitive() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let a = fm.insert(n, 0, FieldValue::Vec3f(Vec3::X));
        let b = fm.insert(n, 1, FieldValue::Vec3f(Vec3::ZERO));
        let c = fm.insert(n, 2, FieldValue::Vec3f(Vec3::ZERO));

        fm.connect(a, b);
        fm.connect(b, c);
        fm.propagate(a);

        assert_eq!(fm.get(c), Some(&FieldValue::Vec3f(Vec3::X)));
    }

    #[test]
    fn test_any_dirty() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let id = fm.insert(n, 0, FieldValue::Bool(false));
        assert!(!fm.any_dirty());

        fm.set(id, FieldValue::Bool(true));
        assert!(fm.any_dirty());
    }

    #[test]
    fn test_mark_all_dirty() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let a = fm.insert(n, 0, FieldValue::Int32(1));
        let b = fm.insert(n, 1, FieldValue::Int32(2));
        assert!(!fm.is_dirty(a));
        assert!(!fm.is_dirty(b));

        fm.mark_all_dirty();
        assert!(fm.is_dirty(a));
        assert!(fm.is_dirty(b));
    }

    #[test]
    fn test_get_entry_returns_metadata() {
        let mut fm = FieldMap::new();
        let n = make_node();
        let id = fm.insert(n, 7, FieldValue::Vec2f(Vec2::new(1.0, 2.0)));
        let entry = fm.get_entry(id).unwrap();
        assert_eq!(entry.field_index, 7);
        assert!(!entry.dirty);
    }

    #[test]
    fn test_default_is_empty() {
        let fm = FieldMap::default();
        assert!(!fm.any_dirty());
    }
}
