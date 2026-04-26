use rc3d_core::{math::Vec3, NodeId};
use rc3d_scene::{NodeData, SceneGraph};

use crate::State;

#[derive(Clone, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::MAX),
            max: Vec3::splat(f32::MIN),
        }
    }

    pub fn from_point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn transform(&self, matrix: rc3d_core::math::Mat4) -> Aabb {
        let corners = [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ];
        let mut result = Aabb::empty();
        for c in &corners {
            let t = matrix.transform_point3(*c);
            result = result.union(&Aabb::from_point(t));
        }
        result
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
}

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
            // Cameras, lights, normals, material, IFS don't contribute to bbox directly
            _ => {}
        }
    }
}

impl Default for GetBoundingBoxAction {
    fn default() -> Self {
        Self::new()
    }
}
