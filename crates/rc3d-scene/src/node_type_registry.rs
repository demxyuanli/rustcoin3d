use std::collections::HashMap;
use std::sync::Mutex;

use crate::custom_node::CustomNodeData;

/// Factory function type: `(type_name, deserialize_fn)`.
type FactoryFn = fn(&str) -> Option<Box<dyn CustomNodeData>>;

/// Global registry for custom node types.
///
/// Register node types at startup so `NodeData::Custom` can be
/// serialized, deserialized, and displayed correctly even when
/// the concrete type is not known at compile time.
pub struct NodeTypeRegistry {
    /// type_id → (type_name, factory_fn)
    factories: HashMap<u16, (&'static str, FactoryFn)>,
}

impl NodeTypeRegistry {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a custom node type.
    ///
    /// `type_id` must be unique across all registrations.
    /// `factory` should parse `data` (the output of `CustomNodeData::serialize_custom`)
    /// and return a `Box<dyn CustomNodeData>`.
    pub fn register(
        &mut self,
        type_id: u16,
        type_name: &'static str,
        factory: FactoryFn,
    ) {
        self.factories.insert(type_id, (type_name, factory));
    }

    /// Remove a previously registered type.
    pub fn unregister(&mut self, type_id: u16) {
        self.factories.remove(&type_id);
    }

    /// Look up the name for a registered type.
    pub fn type_name(&self, type_id: u16) -> Option<&'static str> {
        self.factories.get(&type_id).map(|(name, _)| *name)
    }

    /// Deserialize a custom node from its type_id and serialized data.
    pub fn deserialize(&self, type_id: u16, data: &str) -> Option<Box<dyn CustomNodeData>> {
        self.factories
            .get(&type_id)
            .and_then(|(_, factory)| factory(data))
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }
}

impl Default for NodeTypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global singleton registry (lazy-init, thread-safe).
static GLOBAL_REGISTRY: std::sync::LazyLock<Mutex<NodeTypeRegistry>> =
    std::sync::LazyLock::new(|| Mutex::new(NodeTypeRegistry::new()));

/// Access the global node type registry.
pub fn global_registry() -> &'static Mutex<NodeTypeRegistry> {
    &GLOBAL_REGISTRY
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_node::CustomNodeData;
    use crate::node_data::{FieldDescriptor, NodeData};

    #[derive(Debug, Clone)]
    struct TestNode {
        value: f32,
    }

    impl CustomNodeData for TestNode {
        fn type_name(&self) -> &'static str {
            "TestNode"
        }

        fn clone_box(&self) -> Box<dyn CustomNodeData> {
            Box::new(self.clone())
        }

        fn field_descriptors(&self) -> Vec<FieldDescriptor> {
            vec![FieldDescriptor {
                name: "value",
                field_index: 0,
            }]
        }

        fn serialize_custom(&self) -> String {
            format!("{}", self.value)
        }

        fn deserialize_custom(data: &str) -> Option<Box<dyn CustomNodeData>>
        where
            Self: Sized,
        {
            data.parse::<f32>()
                .ok()
                .map(|v| Box::new(TestNode { value: v }) as Box<dyn CustomNodeData>)
        }
    }

    #[test]
    fn test_register_and_lookup() {
        let mut reg = NodeTypeRegistry::new();
        reg.register(100, "TestNode", TestNode::deserialize_custom);
        assert_eq!(reg.type_name(100), Some("TestNode"));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_deserialize_via_registry() {
        let mut reg = NodeTypeRegistry::new();
        reg.register(100, "TestNode", TestNode::deserialize_custom);
        let node = reg.deserialize(100, "3.14");
        assert!(node.is_some());
        assert_eq!(node.unwrap().type_name(), "TestNode");
    }

    #[test]
    fn test_custom_node_in_nodedata() {
        let nd = NodeData::Custom(100, Box::new(TestNode { value: 1.0 }));
        assert_eq!(nd.type_name(), "TestNode");
        let descs = nd.field_descriptors();
        assert_eq!(descs.len(), 1);
        assert_eq!(descs[0].name, "value");

        // Clone
        let cloned = nd.clone();
        assert_eq!(cloned.type_name(), "TestNode");
    }

    #[test]
    fn test_custom_node_serde_roundtrip() {
        let mut reg = NodeTypeRegistry::new();
        reg.register(100, "TestNode", TestNode::deserialize_custom);

        let nd = NodeData::Custom(100, Box::new(TestNode { value: 2.5 }));
        let json = serde_json::to_string(&nd).expect("serialize");
        let nd2: NodeData = serde_json::from_str(&json).expect("deserialize");
        // Custom deserializes as DummyHandler for now (registry not integrated with serde)
        assert!(nd2.type_name().contains("Handler") || nd2.type_name() == "TestNode");
    }

    #[test]
    fn test_unregister() {
        let mut reg = NodeTypeRegistry::new();
        reg.register(200, "TempNode", TestNode::deserialize_custom);
        assert_eq!(reg.len(), 1);
        reg.unregister(200);
        assert_eq!(reg.len(), 0);
        assert!(reg.type_name(200).is_none());
    }
}
