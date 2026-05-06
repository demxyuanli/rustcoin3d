use std::fmt::Debug;

use crate::node_data::FieldDescriptor;

/// Trait for user-defined node types stored in `NodeData::Custom`.
///
/// Implementors must be `Debug + Send + Sync` (required by `NodeData`).
/// For serde roundtrip, implement `serialize_custom` / `deserialize_custom`.
pub trait CustomNodeData: Debug + Send + Sync {
    /// Unique type name for display and serialization.
    fn type_name(&self) -> &'static str;

    /// Deep-clone this node data (required because `Box<dyn CustomNodeData>` is not `Clone`).
    fn clone_box(&self) -> Box<dyn CustomNodeData>;

    /// Fields exposed to the editor inspector.
    fn field_descriptors(&self) -> Vec<FieldDescriptor> {
        Vec::new()
    }

    /// Serialize to a string payload (default: type name only).
    /// Override to persist custom fields.
    fn serialize_custom(&self) -> String {
        self.type_name().to_string()
    }

    /// Deserialize from a string payload.
    /// Returns `None` if the payload is unrecognized or malformed.
    fn deserialize_custom(data: &str) -> Option<Box<dyn CustomNodeData>>
    where
        Self: Sized;
}
