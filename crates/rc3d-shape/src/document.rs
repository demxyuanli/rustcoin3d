//! Shape document root (OCC TDocStd_Document + XDE).

use std::collections::HashMap;

use rc3d_core::math::Mat4;
use slotmap::SlotMap;

use crate::shape::{Shape, ShapeId, ShapeKind, ShapeNode};
use crate::store::BRepStore;
use crate::tessellation::TessellationCache;
use crate::topo::SolidKey;
use crate::xde::{LabelId, XdeLabelForest};

#[derive(Debug, Default)]
pub struct ProvenanceMap {
    pub shape_to_step: HashMap<ShapeId, u64>,
    pub label_to_step: HashMap<LabelId, u64>,
}

#[derive(Debug, Default)]
pub struct PmiDataSet {
    pub entries: Vec<PmiEntry>,
}

#[derive(Debug, Clone)]
pub struct PmiEntry {
    pub id: u32,
    pub label: String,
    pub step_entity_id: Option<u64>,
    pub origin: [f32; 3],
    pub normal: [f32; 3],
}

#[derive(Debug)]
pub struct ShapeDocument {
    pub store: BRepStore,
    pub shapes: SlotMap<ShapeId, ShapeNode>,
    pub roots: Vec<ShapeId>,
    pub labels: XdeLabelForest,
    pub pmi_pool: PmiDataSet,
    pub tessellation: TessellationCache,
    pub provenance: ProvenanceMap,
    world_transform_cache: HashMap<ShapeId, Mat4>,
}

impl Default for ShapeDocument {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeDocument {
    pub fn new() -> Self {
        Self {
            store: BRepStore::new(),
            shapes: SlotMap::with_key(),
            roots: Vec::new(),
            labels: XdeLabelForest::new(),
            pmi_pool: PmiDataSet::default(),
            tessellation: TessellationCache::new(),
            provenance: ProvenanceMap::default(),
            world_transform_cache: HashMap::new(),
        }
    }

    pub fn free_shapes(&self) -> &[ShapeId] {
        &self.roots
    }

    pub fn shape_for_label(&self, label_id: LabelId) -> Option<Shape> {
        self.labels
            .labels
            .get(label_id)
            .and_then(|l| l.shape)
            .map(|id| Shape { id })
    }

    pub fn labels_for_solid(&self, solid_key: SolidKey) -> Vec<LabelId> {
        let mut out = Vec::new();
        for (label_id, label) in self.labels.labels.iter() {
            let Some(shape_id) = label.shape else {
                continue;
            };
            let Some(node) = self.shapes.get(shape_id) else {
                continue;
            };
            if node.kind.solid_key() == Some(solid_key) {
                out.push(label_id);
            }
        }
        out
    }

    pub fn resolved_label_color(&self, label_id: LabelId) -> Option<[f32; 3]> {
        self.labels.resolved_color(label_id)
    }

    pub fn resolved_label_opacity(&self, label_id: LabelId) -> Option<f32> {
        self.labels.resolved_opacity(label_id)
    }

    pub fn insert_shape(&mut self, node: ShapeNode) -> ShapeId {
        self.world_transform_cache.clear();
        self.shapes.insert(node)
    }

    pub fn set_parent(&mut self, child: ShapeId, parent: ShapeId) {
        if let Some(node) = self.shapes.get_mut(child) {
            node.parent = Some(parent);
        }
        if let Some(parent_node) = self.shapes.get_mut(parent) {
            if !parent_node.children.contains(&child) {
                parent_node.children.push(child);
            }
        }
        self.world_transform_cache.clear();
    }

    pub fn world_transform(&mut self, shape_id: ShapeId) -> Mat4 {
        if let Some(&cached) = self.world_transform_cache.get(&shape_id) {
            return cached;
        }
        let world = self.compute_world_transform(shape_id);
        self.world_transform_cache.insert(shape_id, world);
        world
    }

    fn compute_world_transform(&self, shape_id: ShapeId) -> Mat4 {
        let Some(node) = self.shapes.get(shape_id) else {
            return Mat4::IDENTITY;
        };
        let local = node.location;
        match node.parent {
            None => local,
            Some(parent) => {
                let parent_world = self.compute_world_transform(parent);
                parent_world * local
            }
        }
    }

    pub fn add_solid_instance(
        &mut self,
        solid_key: SolidKey,
        location: Mat4,
        parent: Option<ShapeId>,
    ) -> ShapeId {
        let shape_id = self.insert_shape(ShapeNode {
            kind: ShapeKind::Solid(solid_key),
            orientation: crate::topo::Orientation::Forward,
            location,
            parent: None,
            children: Vec::new(),
        });
        if let Some(p) = parent {
            self.set_parent(shape_id, p);
        } else {
            self.roots.push(shape_id);
        }
        shape_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::Orientation;

    #[test]
    fn world_transform_parent_chain() {
        let mut doc = ShapeDocument::new();
        let parent = doc.insert_shape(ShapeNode {
            kind: ShapeKind::Compound,
            orientation: Orientation::Forward,
            location: Mat4::from_translation([10.0, 0.0, 0.0].into()),
            parent: None,
            children: Vec::new(),
        });
        doc.roots.push(parent);

        let sk = doc.store.solids.insert(crate::topo::BRepSolid {
            outer_shell: doc.store.shells.insert(crate::topo::BRepShell {
                faces: vec![],
                closed: true,
                step_id: None,
            }),
            void_shells: vec![],
        });

        let child = doc.add_solid_instance(sk, Mat4::from_translation([0.0, 5.0, 0.0].into()), Some(parent));
        let world = doc.world_transform(child);
        assert!((world.w_axis.x - 10.0).abs() < 1e-5);
        assert!((world.w_axis.y - 5.0).abs() < 1e-5);
    }

    #[test]
    fn world_transform_cache_hit() {
        let mut doc = ShapeDocument::new();
        let id = doc.insert_shape(ShapeNode {
            kind: ShapeKind::Compound,
            orientation: Orientation::Forward,
            location: Mat4::IDENTITY,
            parent: None,
            children: Vec::new(),
        });
        doc.roots.push(id);
        let _ = doc.world_transform(id);
        assert!(doc.world_transform_cache.contains_key(&id));
        let _again = doc.world_transform(id);
        assert_eq!(doc.world_transform_cache.len(), 1);
    }
}
