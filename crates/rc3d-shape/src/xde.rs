//! XDE-style label tree and attributes.

use rc3d_core::math::Real;
use slotmap::new_key_type;

use crate::shape::ShapeId;

new_key_type! { pub struct LabelId; }

#[derive(Debug, Clone, Default)]
pub struct AttributeBag {
    pub name: Option<String>,
    pub description: Option<String>,
    pub color: Option<[Real; 3]>,
    pub opacity: Option<Real>,
    pub layer: Option<String>,
    pub step_entity_id: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct XdeLabel {
    pub parent: Option<LabelId>,
    pub children: Vec<LabelId>,
    pub attrs: AttributeBag,
    pub shape: Option<ShapeId>,
}

#[derive(Debug, Default)]
pub struct XdeLabelForest {
    pub labels: slotmap::SlotMap<LabelId, XdeLabel>,
    pub root_labels: Vec<LabelId>,
}

impl XdeLabelForest {
    pub fn new() -> Self {
        Self {
            labels: slotmap::SlotMap::with_key(),
            root_labels: Vec::new(),
        }
    }

    pub fn add_label(&mut self, label: XdeLabel) -> LabelId {
        self.labels.insert(label)
    }

    pub fn walk<F>(&self, mut visitor: F)
    where
        F: FnMut(LabelId, &XdeLabel, usize),
    {
        for &root in &self.root_labels {
            self.walk_label(root, 0, &mut visitor);
        }
    }

    fn walk_label<F>(&self, id: LabelId, depth: usize, visitor: &mut F)
    where
        F: FnMut(LabelId, &XdeLabel, usize),
    {
        let Some(label) = self.labels.get(id) else {
            return;
        };
        visitor(id, label, depth);
        for &child in &label.children {
            self.walk_label(child, depth + 1, visitor);
        }
    }

    pub fn resolved_color(&self, label_id: LabelId) -> Option<[Real; 3]> {
        let mut current = Some(label_id);
        while let Some(id) = current {
            let label = self.labels.get(id)?;
            if let Some(color) = label.attrs.color {
                return Some(color);
            }
            current = label.parent;
        }
        None
    }

    pub fn resolved_opacity(&self, label_id: LabelId) -> Option<Real> {
        let mut current = Some(label_id);
        while let Some(id) = current {
            let label = self.labels.get(id)?;
            if let Some(opacity) = label.attrs.opacity {
                return Some(opacity);
            }
            current = label.parent;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_color_inherits_from_parent() {
        let mut forest = XdeLabelForest::new();
        let parent = forest.add_label(XdeLabel {
            parent: None,
            children: vec![],
            attrs: AttributeBag {
                color: Some([0.2, 0.4, 0.6]),
                opacity: Some(0.5),
                ..Default::default()
            },
            shape: None,
        });
        let child = forest.add_label(XdeLabel {
            parent: Some(parent),
            children: vec![],
            attrs: AttributeBag::default(),
            shape: None,
        });
        if let Some(p) = forest.labels.get_mut(parent) {
            p.children.push(child);
        }
        assert_eq!(forest.resolved_color(child), Some([0.2, 0.4, 0.6]));
        assert_eq!(forest.resolved_opacity(child), Some(0.5));
    }

    #[test]
    fn walk_visits_children() {
        let mut forest = XdeLabelForest::new();
        let root = forest.add_label(XdeLabel {
            parent: None,
            children: vec![],
            attrs: AttributeBag::default(),
            shape: None,
        });
        let child = forest.add_label(XdeLabel {
            parent: Some(root),
            children: vec![],
            attrs: AttributeBag::default(),
            shape: None,
        });
        forest.labels.get_mut(root).unwrap().children.push(child);
        forest.root_labels.push(root);
        let mut count = 0;
        forest.walk(|_, _, _| count += 1);
        assert_eq!(count, 2);
    }
}
