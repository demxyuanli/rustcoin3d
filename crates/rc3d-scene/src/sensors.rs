//! Field and node sensors: Coin3D `SoFieldSensor` / `SoNodeSensor` style notifications.
//! Callbacks are registered at runtime; the scene graph does not execute them automatically
//! until `SceneGraph::notify_sensors` is called after edits.

use std::sync::Arc;

use rc3d_core::NodeId;

/// Identifies a field on a node (use `NodeData::field_descriptors` indices or stable names).
pub type FieldIndex = u16;

/// Invoked when a field value may have changed.
pub trait FieldChangeCallback: Send + Sync {
    fn on_field_change(&self, node: NodeId, field: FieldIndex);
}

/// Invoked when a node is about to be removed.
pub trait NodeDeleteCallback: Send + Sync {
    fn on_node_delete(&self, node: NodeId);
}

/// Global registry; applications wire concrete callbacks here.
pub struct SensorRegistry {
    field_listeners: Vec<Arc<dyn FieldChangeCallback + Send + Sync>>,
    node_listeners: Vec<Arc<dyn NodeDeleteCallback + Send + Sync>>,
}

impl SensorRegistry {
    pub fn new() -> Self {
        Self {
            field_listeners: Vec::new(),
            node_listeners: Vec::new(),
        }
    }

    pub fn add_field_listener(&mut self, cb: Arc<dyn FieldChangeCallback + Send + Sync>) {
        self.field_listeners.push(cb);
    }

    pub fn add_node_listener(&mut self, cb: Arc<dyn NodeDeleteCallback + Send + Sync>) {
        self.node_listeners.push(cb);
    }

    /// Broadcast a field change to all field listeners.
    pub fn fire_field(&self, node: NodeId, field: FieldIndex) {
        for c in &self.field_listeners {
            c.on_field_change(node, field);
        }
    }

    /// Broadcast a node delete to all node listeners.
    pub fn fire_node_delete(&self, node: NodeId) {
        for c in &self.node_listeners {
            c.on_node_delete(node);
        }
    }
}

impl Default for SensorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
