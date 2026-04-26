use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

use crate::State;

#[derive(Clone, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Create a ray from a screen-space point using inverse view-projection.
    pub fn from_screen_point(
        screen_x: f32,
        screen_y: f32,
        width: f32,
        height: f32,
        view: Mat4,
        projection: Mat4,
    ) -> Self {
        let x = 2.0 * screen_x / width - 1.0;
        let y = -(2.0 * screen_y / height - 1.0);
        let inv_vp = (projection * view).inverse();
        let near = inv_vp * Vec3::new(x, y, 0.0).extend(1.0);
        let far = inv_vp * Vec3::new(x, y, 1.0).extend(1.0);
        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;
        Ray::new(near, far - near)
    }

    /// Intersect with a sphere. Returns distance if hit.
    pub fn intersect_sphere(&self, center: Vec3, radius: f32) -> Option<f32> {
        let oc = self.origin - center;
        let a = self.direction.dot(self.direction);
        let b = 2.0 * oc.dot(self.direction);
        let c = oc.dot(oc) - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let t = (-b - disc.sqrt()) / (2.0 * a);
        if t > 0.001 {
            Some(t)
        } else {
            None
        }
    }

    /// Intersect with a triangle. Returns (distance, barycentric coords) if hit.
    pub fn intersect_triangle(&self, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f32, Vec3)> {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let h = self.direction.cross(e2);
        let a = e1.dot(h);
        if a.abs() < 1e-8 {
            return None;
        }
        let f = 1.0 / a;
        let s = self.origin - v0;
        let u = f * s.dot(h);
        if u < 0.0 || u > 1.0 {
            return None;
        }
        let q = s.cross(e1);
        let v = f * self.direction.dot(q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = f * e2.dot(q);
        if t > 0.001 {
            Some((t, Vec3::new(1.0 - u - v, u, v)))
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct PickHit {
    pub node: NodeId,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

pub struct RayPickAction {
    pub state: State,
    pub ray: Ray,
    pub hits: Vec<PickHit>,
}

impl RayPickAction {
    pub fn new(ray: Ray) -> Self {
        Self {
            state: State::new(),
            ray,
            hits: Vec::new(),
        }
    }

    pub fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        self.traverse_node(graph, root);
        self.hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
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
            NodeData::PerspectiveCamera(cam) => {
                self.state.set_view_matrix(cam.view_matrix());
                self.state.set_projection_matrix(cam.projection_matrix());
            }
            NodeData::OrthographicCamera(cam) => {
                self.state.set_view_matrix(cam.view_matrix());
                self.state.set_projection_matrix(cam.projection_matrix());
            }
            NodeData::Coordinate3(_) | NodeData::Normal(_) | NodeData::Material(_) => {}
            NodeData::DirectionalLight(_)
            | NodeData::PointLight(_)
            | NodeData::SpotLight(_) => {}
            // Shape nodes: do intersection test
            NodeData::Triangle(_) => {
                let coord = self.state.coordinate();
                if coord.points.len() >= 3 {
                    let model = self.state.model_matrix();
                    let v0 = model.transform_point3(coord.points[0]);
                    let v1 = model.transform_point3(coord.points[1]);
                    let v2 = model.transform_point3(coord.points[2]);
                    if let Some((t, _)) = self.ray.intersect_triangle(v0, v1, v2) {
                        let point = self.ray.origin + self.ray.direction * t;
                        let normal = (v1 - v0).cross(v2 - v0).normalize();
                        self.hits.push(PickHit {
                            node,
                            point,
                            normal,
                            distance: t,
                        });
                    }
                }
            }
            NodeData::Cube(cube) => {
                self.pick_cube(node, cube.width, cube.height, cube.depth);
            }
            NodeData::Sphere(sphere) => {
                self.pick_sphere(node, sphere.radius);
            }
            NodeData::Cone(cone) => {
                // Use bounding sphere for rough pick
                let radius = cone.bottom_radius.max(cone.height * 0.5);
                self.pick_sphere_with_radius(node, radius);
            }
            NodeData::Cylinder(cyl) => {
                let radius = cyl.radius.max(cyl.height * 0.5);
                self.pick_sphere_with_radius(node, radius);
            }
            NodeData::IndexedFaceSet(ifs) => {
                self.pick_indexed_face_set(node, &ifs.coord_index);
            }
        }
    }

    fn pick_cube(&mut self, node: NodeId, w: f32, h: f32, d: f32) {
        let model = self.state.model_matrix();
        let hw = w / 2.0;
        let hh = h / 2.0;
        let hd = d / 2.0;
        let faces: [(Vec3, [Vec3; 4]); 6] = [
            (
                Vec3::new(0.0, 0.0, 1.0),
                [
                    Vec3::new(-hw, -hh, hd),
                    Vec3::new(hw, -hh, hd),
                    Vec3::new(hw, hh, hd),
                    Vec3::new(-hw, hh, hd),
                ],
            ),
            (
                Vec3::new(0.0, 0.0, -1.0),
                [
                    Vec3::new(hw, -hh, -hd),
                    Vec3::new(-hw, -hh, -hd),
                    Vec3::new(-hw, hh, -hd),
                    Vec3::new(hw, hh, -hd),
                ],
            ),
            (
                Vec3::new(1.0, 0.0, 0.0),
                [
                    Vec3::new(hw, -hh, hd),
                    Vec3::new(hw, -hh, -hd),
                    Vec3::new(hw, hh, -hd),
                    Vec3::new(hw, hh, hd),
                ],
            ),
            (
                Vec3::new(-1.0, 0.0, 0.0),
                [
                    Vec3::new(-hw, -hh, -hd),
                    Vec3::new(-hw, -hh, hd),
                    Vec3::new(-hw, hh, hd),
                    Vec3::new(-hw, hh, -hd),
                ],
            ),
            (
                Vec3::new(0.0, 1.0, 0.0),
                [
                    Vec3::new(-hw, hh, hd),
                    Vec3::new(hw, hh, hd),
                    Vec3::new(hw, hh, -hd),
                    Vec3::new(-hw, hh, -hd),
                ],
            ),
            (
                Vec3::new(0.0, -1.0, 0.0),
                [
                    Vec3::new(-hw, -hh, -hd),
                    Vec3::new(hw, -hh, -hd),
                    Vec3::new(hw, -hh, hd),
                    Vec3::new(-hw, -hh, hd),
                ],
            ),
        ];
        for (normal, corners) in &faces {
            let v = corners.map(|c| model.transform_point3(c));
            if let Some((t, _)) = self.ray.intersect_triangle(v[0], v[1], v[2]) {
                let point = self.ray.origin + self.ray.direction * t;
                let world_normal = model.transform_vector3(*normal).normalize();
                self.hits.push(PickHit {
                    node,
                    point,
                    normal: world_normal,
                    distance: t,
                });
                return;
            }
            if let Some((t, _)) = self.ray.intersect_triangle(v[0], v[2], v[3]) {
                let point = self.ray.origin + self.ray.direction * t;
                let world_normal = model.transform_vector3(*normal).normalize();
                self.hits.push(PickHit {
                    node,
                    point,
                    normal: world_normal,
                    distance: t,
                });
                return;
            }
        }
    }

    fn pick_sphere(&mut self, node: NodeId, radius: f32) {
        self.pick_sphere_with_radius(node, radius);
    }

    fn pick_sphere_with_radius(&mut self, node: NodeId, radius: f32) {
        let model = self.state.model_matrix();
        let center = model.transform_point3(Vec3::ZERO);
        // Approximate: use max scale axis for radius
        let sx = model.transform_vector3(Vec3::X).length();
        let sy = model.transform_vector3(Vec3::Y).length();
        let sz = model.transform_vector3(Vec3::Z).length();
        let world_radius = radius * sx.max(sy).max(sz);
        if let Some(t) = self.ray.intersect_sphere(center, world_radius) {
            let point = self.ray.origin + self.ray.direction * t;
            let normal = (point - center).normalize();
            self.hits.push(PickHit {
                node,
                point,
                normal,
                distance: t,
            });
        }
    }

    fn pick_indexed_face_set(&mut self, node: NodeId, coord_index: &[i32]) {
        let coord = self.state.coordinate();
        if coord.points.is_empty() {
            return;
        }
        let model = self.state.model_matrix();
        let mut face_points = Vec::new();
        for &idx in coord_index {
            if idx < 0 {
                if face_points.len() >= 3 {
                    let v0 = model.transform_point3(coord.points[face_points[0]]);
                    for j in 1..face_points.len() - 1 {
                        let v1 = model.transform_point3(coord.points[face_points[j]]);
                        let v2 = model.transform_point3(coord.points[face_points[j + 1]]);
                        if let Some((t, _)) = self.ray.intersect_triangle(v0, v1, v2) {
                            let point = self.ray.origin + self.ray.direction * t;
                            let normal = (v1 - v0).cross(v2 - v0).normalize();
                            self.hits.push(PickHit {
                                node,
                                point,
                                normal,
                                distance: t,
                            });
                            return;
                        }
                    }
                }
                face_points.clear();
            } else {
                face_points.push(idx as usize);
            }
        }
    }
}
