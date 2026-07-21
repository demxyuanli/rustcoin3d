use rc3d_core::math::Mat4;
use rc3d_core::{math::Vec3, Aabb, NodeId};
use crate::{
    scene_traverse, ChildPolicy, NodeData, NodeEntry, SceneGraph, SceneVisitor, State,
    TraversalMatrices,
};

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

    fn union_local_aabb(&mut self, local: Aabb) {
        let world = local.transform(self.state.model_matrix());
        self.bounding_box = self.bounding_box.union(&world);
    }
}

impl TraversalMatrices for GetBoundingBoxAction {
    fn model_matrix(&self) -> Mat4 {
        self.state.model_matrix()
    }

    fn set_model_matrix(&mut self, matrix: Mat4) {
        self.state.set_model_matrix(matrix);
    }

    fn view_matrix(&self) -> Mat4 {
        self.state.view_matrix()
    }
}

impl SceneVisitor for GetBoundingBoxAction {
    fn enter_separator(&mut self) {
        self.state.push_all();
    }

    fn leave_separator(&mut self) {
        self.state.pop_all();
    }

    fn visit_node(
        &mut self,
        _graph: &SceneGraph,
        _node: NodeId,
        entry: &NodeEntry,
    ) -> ChildPolicy {
        match &entry.data {
            NodeData::IndexedLineSet(ils) => {
                let coord = self.state.coordinate();
                let m = self.state.model_matrix();
                for i in (0..ils.coord_index.len()).step_by(2) {
                    if i + 1 < ils.coord_index.len() {
                        let a = ils.coord_index[i].max(0) as usize;
                        let b = ils.coord_index[i + 1].max(0) as usize;
                        if a < coord.points.len() && b < coord.points.len() {
                            let wa = m.transform_point3(coord.points[a]);
                            let wb = m.transform_point3(coord.points[b]);
                            self.bounding_box =
                                self.bounding_box.union(&rc3d_core::Aabb::from_point(wa));
                            self.bounding_box =
                                self.bounding_box.union(&rc3d_core::Aabb::from_point(wb));
                        }
                    }
                }
                ChildPolicy::Recurse
            }
            NodeData::Cube(cube) => {
                let hw = cube.width / 2.0;
                let hh = cube.height / 2.0;
                let hd = cube.depth / 2.0;
                self.union_local_aabb(Aabb {
                    min: Vec3::new(-hw, -hh, -hd),
                    max: Vec3::new(hw, hh, hd),
                });
                ChildPolicy::Skip
            }
            NodeData::Sphere(s) => {
                self.union_local_aabb(Aabb {
                    min: Vec3::splat(-s.radius),
                    max: Vec3::splat(s.radius),
                });
                ChildPolicy::Skip
            }
            NodeData::Cone(c) => {
                let half_h = c.height / 2.0;
                self.union_local_aabb(Aabb {
                    min: Vec3::new(-c.bottom_radius, -half_h, -c.bottom_radius),
                    max: Vec3::new(c.bottom_radius, half_h, c.bottom_radius),
                });
                ChildPolicy::Skip
            }
            NodeData::Cylinder(c) => {
                let half_h = c.height / 2.0;
                self.union_local_aabb(Aabb {
                    min: Vec3::new(-c.radius, -half_h, -c.radius),
                    max: Vec3::new(c.radius, half_h, c.radius),
                });
                ChildPolicy::Skip
            }
            NodeData::Torus(torus) => {
                let r = torus.major_radius + torus.minor_radius;
                self.union_local_aabb(Aabb {
                    min: Vec3::new(-r, -torus.minor_radius, -r),
                    max: Vec3::new(r, torus.minor_radius, r),
                });
                ChildPolicy::Skip
            }
            NodeData::Volume(vol) => {
                let hx = vol.dimensions[0] as f32 * 0.5;
                let hy = vol.dimensions[1] as f32 * 0.5;
                let hz = vol.dimensions[2] as f32 * 0.5;
                self.union_local_aabb(Aabb {
                    min: Vec3::new(-hx, -hy, -hz),
                    max: Vec3::new(hx, hy, hz),
                });
                ChildPolicy::Recurse
            }
            NodeData::Coordinate3(coord) => {
                let model = self.state.model_matrix();
                for p in &coord.point {
                    let wp = model.transform_point3(*p);
                    self.bounding_box = self.bounding_box.union(&Aabb::from_point(wp));
                }
                ChildPolicy::Skip
            }
            NodeData::Triangle(_) => {
                let model = self.state.model_matrix();
                let coords = self.state.coordinate();
                for p in &coords.points {
                    let wp = model.transform_point3(*p);
                    self.bounding_box = self.bounding_box.union(&Aabb::from_point(wp));
                }
                ChildPolicy::Skip
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
                ChildPolicy::Skip
            }
            // Property / camera / light leaves — no spatial contribution
            NodeData::Material(_)
            | NodeData::Normal(_)
            | NodeData::TextureCoordinate2(_)
            | NodeData::PerspectiveCamera(_)
            | NodeData::OrthographicCamera(_)
            | NodeData::DirectionalLight(_)
            | NodeData::PointLight(_)
            | NodeData::SpotLight(_)
            | NodeData::PointCloud(_) => ChildPolicy::Skip,
            // Group-like: recurse children
            _ => ChildPolicy::Recurse,
        }
    }
}

impl Default for GetBoundingBoxAction {
    fn default() -> Self {
        Self::new()
    }
}

impl GetBoundingBoxAction {
    /// Traverse all roots and return the world AABB if non-empty.
    pub fn compute_scene_aabb(graph: &SceneGraph) -> Option<Aabb> {
        let mut action = Self::new();
        for &root in graph.roots() {
            scene_traverse(&mut action, graph, root);
        }
        if action.bounding_box.min.x <= action.bounding_box.max.x {
            Some(action.bounding_box)
        } else {
            None
        }
    }

    pub fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        scene_traverse(self, graph, root);
    }
}
