use rc3d_core::{math::Vec3, Aabb, NodeId};
use rc3d_scene::{NodeData, SceneGraph};

use crate::State;

pub struct GetBoundingBoxAction {
    pub state: State,
    pub bounding_box: Aabb,
}

impl GetBoundingBoxAction {
    pub fn new() -> Self {
        Self {
            state: State::new(),
            bounding_box: Aabb::empty(),
        }
    }

    fn traverse_node(&mut self, graph: &SceneGraph, node: NodeId) {
        let Some(entry) = graph.get(node) else {
            return;
        };
        match &entry.data {
            NodeData::Separator(_) => {
                self.state.push_all();
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
                self.state.pop_all();
            }
            NodeData::Group(_) | NodeData::Environment(_) | NodeData::ShapeHints(_) | NodeData::Annotation(_) | NodeData::ResetTransform(_) | NodeData::Texture2Transform(_) | NodeData::MaterialBinding(_) | NodeData::IndexedLineSet(_) | NodeData::File(_) | NodeData::Decal(_) | NodeData::ExplodedView(_) | NodeData::ReflectionPlane(_) | NodeData::Billboard(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::Switch(sw) => {
                match sw.which_child {
                    -2 => {} // none
                    -1 => {
                        for &child in &sw.children {
                            self.traverse_node(graph, child);
                        }
                    }
                    idx if idx >= 0 => {
                        let i = idx as usize;
                        if i < sw.children.len() {
                            self.traverse_node(graph, sw.children[i]);
                        }
                    }
                    _ => {}
                }
            }
            NodeData::MultipleCopy(mc) => {
                let base = self.state.model_matrix();
                for &copy_mat in &mc.copies {
                    self.state.set_model_matrix(base * copy_mat);
                    for &child in &mc.children {
                        self.traverse_node(graph, child);
                    }
                }
                self.state.set_model_matrix(base);
            }
            NodeData::Lod(lod) => {
                let level = lod.current_level.min(lod.levels.len().saturating_sub(1));
                if let Some(level_data) = lod.levels.get(level) {
                    for &child in &level_data.children {
                        self.traverse_node(graph, child);
                    }
                }
            }
            NodeData::HandlerNode(h) => {
                h.traverse(graph, node, &entry.children, &mut |id| self.traverse_node(graph, id));
            }
            NodeData::Transform(t) => {
                let current = self.state.model_matrix();
                self.state.set_model_matrix(current * t.to_matrix());
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::Cube(cube) => {
                let hw = cube.width / 2.0;
                let hh = cube.height / 2.0;
                let hd = cube.depth / 2.0;
                let local = Aabb {
                    min: Vec3::new(-hw, -hh, -hd),
                    max: Vec3::new(hw, hh, hd),
                };
                let world = local.transform(self.state.model_matrix());
                self.bounding_box = self.bounding_box.union(&world);
            }
            NodeData::Sphere(s) => {
                let local = Aabb {
                    min: Vec3::splat(-s.radius),
                    max: Vec3::splat(s.radius),
                };
                let world = local.transform(self.state.model_matrix());
                self.bounding_box = self.bounding_box.union(&world);
            }
            NodeData::Cone(c) => {
                let half_h = c.height / 2.0;
                let local = Aabb {
                    min: Vec3::new(-c.bottom_radius, -half_h, -c.bottom_radius),
                    max: Vec3::new(c.bottom_radius, half_h, c.bottom_radius),
                };
                let world = local.transform(self.state.model_matrix());
                self.bounding_box = self.bounding_box.union(&world);
            }
            NodeData::Cylinder(c) => {
                let half_h = c.height / 2.0;
                let local = Aabb {
                    min: Vec3::new(-c.radius, -half_h, -c.radius),
                    max: Vec3::new(c.radius, half_h, c.radius),
                };
                let world = local.transform(self.state.model_matrix());
                self.bounding_box = self.bounding_box.union(&world);
            }
            NodeData::Coordinate3(coord) => {
                let model = self.state.model_matrix();
                for p in &coord.point {
                    let wp = model.transform_point3(*p);
                    self.bounding_box = self.bounding_box.union(&Aabb::from_point(wp));
                }
            }
            NodeData::SectionPlane(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::Text2(_) | NodeData::Text3(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::Measurement(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::Markup(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::Triangle(_) => {
                let model = self.state.model_matrix();
                let coords = self.state.coordinate();
                for p in &coords.points {
                    let wp = model.transform_point3(*p);
                    self.bounding_box = self.bounding_box.union(&Aabb::from_point(wp));
                }
            }
            NodeData::IndexedFaceSet(ifs) => {
                let model = self.state.model_matrix();
                let coords = self.state.coordinate();
                for &ci in &ifs.coord_index {
                    if ci >= 0 && (ci as usize) < coords.points.len() {
                        let wp = model.transform_point3(coords.points[ci as usize]);
                        self.bounding_box = self.bounding_box.union(&Aabb::from_point(wp));
                    }
                }
            }
            NodeData::EventCallback(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::MorphTarget(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::SkinnedMesh(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            _ => {}
        }
    }
}

impl Default for GetBoundingBoxAction {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::Action for GetBoundingBoxAction {
    fn kind(&self) -> crate::ActionKind {
        crate::ActionKind::GetBoundingBox
    }

    fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        self.traverse_node(graph, root);
    }
}
