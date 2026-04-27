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

    pub fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        self.traverse_node(graph, root);
    }

    fn traverse_node(&mut self, graph: &SceneGraph, node: NodeId) {
        let Some(entry) = graph.get(node) else {
            return;
        };
        let data = entry.data.clone();

        match &data {
            NodeData::Separator(_) => {
                self.state.push_all();
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
                self.state.pop_all();
            }
            NodeData::Group(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
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
            _ => {}
        }
    }
}

impl Default for GetBoundingBoxAction {
    fn default() -> Self {
        Self::new()
    }
}
