use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_mesh::{Bvh, BvhTriangle};
use rc3d_scene::{NodeData, SceneGraph};

use crate::State;

/// Use BVH ray cast when fan triangle count reaches this threshold.
const IFS_BVH_TRIANGLE_THRESHOLD: usize = 4096;

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
        if a <= f32::EPSILON {
            return None;
        }
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
        if !(0.0..=1.0).contains(&u) {
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
    pub face_index: Option<u32>,
    pub edge_index: Option<u32>,
    /// Barycentric coordinates for triangle hits (when available).
    pub barycentric: Option<[f32; 3]>,
}

/// Structured detail for what was hit (Coin3D SoDetail pattern).
#[derive(Clone, Debug)]
pub enum DetailInfo {
    Face {
        face_index: u32,
        barycentric: [f32; 3],
        texcoord: Option<[f32; 2]>,
    },
    Edge {
        edge_index: u32,
    },
    Point {
        point_index: u32,
    },
    None,
}

/// Enhanced pick result with structured detail.
#[derive(Clone, Debug)]
pub struct PickDetail {
    pub node: NodeId,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub detail: DetailInfo,
}

impl PickDetail {
    pub fn from_hit(hit: &PickHit) -> Self {
        let bary0 = [0.33, 0.33, 0.34];
        let detail = match (hit.face_index, hit.edge_index) {
            (Some(fi), _) => DetailInfo::Face {
                face_index: fi,
                barycentric: hit.barycentric.unwrap_or(bary0),
                texcoord: None,
            },
            (_, Some(ei)) => DetailInfo::Edge { edge_index: ei },
            _ => DetailInfo::None,
        };
        Self { node: hit.node, point: hit.point, normal: hit.normal, distance: hit.distance, detail }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickMode {
    Node,
    Face,
    Edge,
}

pub struct RayPickAction {
    pub state: State,
    pub ray: Ray,
    pub hits: Vec<PickHit>,
    pub mode: PickMode,
    /// Screen-space pick tolerance in pixels. Hits beyond this radius from
    /// the cursor are filtered out. 0 = exact ray intersection only.
    pub tolerance_px: f32,
    /// Cursor position (used with tolerance_px for hit filtering).
    pub cursor_pos: Option<(f32, f32)>,
    /// View/projection matrices (used with tolerance_px for hit filtering).
    pub pick_vp: Option<(Mat4, Mat4)>,
    /// Stack tracking current pickability.
    pickable_stack: Vec<bool>,
}

impl RayPickAction {
    pub fn new(ray: Ray) -> Self {
        Self {
            state: State::new(),
            ray,
            hits: Vec::new(),
            mode: PickMode::Node,
            tolerance_px: 0.0,
            cursor_pos: None,
            pick_vp: None,
            pickable_stack: vec![true],
        }
    }

    pub fn with_mode(ray: Ray, mode: PickMode) -> Self {
        Self {
            state: State::new(),
            ray,
            hits: Vec::new(),
            mode,
            tolerance_px: 0.0,
            cursor_pos: None,
            pick_vp: None,
            pickable_stack: vec![true],
        }
    }

    fn is_pickable(&self) -> bool {
        self.pickable_stack.last().copied().unwrap_or(true)
    }

    pub fn details(&self) -> Vec<PickDetail> {
        self.hits.iter().map(PickDetail::from_hit).collect()
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
            NodeData::Group(_) | NodeData::Billboard(_) => {
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
            NodeData::EventCallback(_) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
            NodeData::PickStyle(ps) => {
                self.pickable_stack.push(ps.pickable);
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
                self.pickable_stack.pop();
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
            NodeData::Measurement(_) | NodeData::Markup(_) => {
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
            NodeData::Coordinate3(_) | NodeData::TextureCoordinate2(_) | NodeData::Normal(_) | NodeData::Material(_) => {}
            NodeData::DirectionalLight(_)
            | NodeData::PointLight(_)
            | NodeData::SpotLight(_) | NodeData::AreaLight(_) => {}
            // Shape nodes: do intersection test
            NodeData::Triangle(_) => {
                if !self.is_pickable() { return; }
                let coord = self.state.coordinate();
                if coord.points.len() >= 3 {
                    let model = self.state.model_matrix();
                    let v0 = model.transform_point3(coord.points[0]);
                    let v1 = model.transform_point3(coord.points[1]);
                    let v2 = model.transform_point3(coord.points[2]);
                    if let Some((t, bary)) = self.ray.intersect_triangle(v0, v1, v2) {
                        let point = self.ray.origin + self.ray.direction * t;
                        let c = (v1 - v0).cross(v2 - v0);
                        let normal = if c.length_squared() > 1e-20 { c.normalize() } else { Vec3::Y };
                        let face_index = if self.mode != PickMode::Node { Some(0) } else { None };
                        let edge_index = if self.mode == PickMode::Edge {
                            Some(closest_edge_from_bary(&bary))
                        } else {
                            None
                        };
                        self.hits.push(PickHit {
                            node,
                            point,
                            normal,
                            distance: t,
                            face_index,
                            edge_index,
                            barycentric: Some(bary.to_array()),
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
                self.pick_cone(node, cone.bottom_radius, cone.height);
            }
            NodeData::Cylinder(cyl) => {
                self.pick_cylinder(node, cyl.radius, cyl.height);
            }
            NodeData::IndexedFaceSet(ifs) => {
                self.pick_indexed_face_set(node, &ifs.coord_index);
            }
            NodeData::Custom(_, _) => {
                for &child in &entry.children {
                    self.traverse_node(graph, child);
                }
            }
        }
    }

    fn pick_cube(&mut self, node: NodeId, w: f32, h: f32, d: f32) {
        if !self.is_pickable() { return; }
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
        let mut tri_idx = 0u32;
        let mut best_t = f32::MAX;
        let mut best_hit: Option<(Vec3, Vec3, u32, Vec3)> = None;
        for (normal, corners) in &faces {
            let v = corners.map(|c| model.transform_point3(c));
            if let Some((t, bary)) = self.ray.intersect_triangle(v[0], v[1], v[2]) {
                if t > 0.001 && t < best_t {
                    best_t = t;
                    let point = self.ray.origin + self.ray.direction * t;
                    let world_normal = model.transform_vector3(*normal).normalize();
                    best_hit = Some((point, world_normal, tri_idx, bary));
                }
            }
            tri_idx += 1;
            if let Some((t, bary)) = self.ray.intersect_triangle(v[0], v[2], v[3]) {
                if t > 0.001 && t < best_t {
                    best_t = t;
                    let point = self.ray.origin + self.ray.direction * t;
                    let world_normal = model.transform_vector3(*normal).normalize();
                    best_hit = Some((point, world_normal, tri_idx, bary));
                }
            }
            tri_idx += 1;
        }
        if let Some((point, normal, tri_idx, bary)) = best_hit {
            self.push_hit(node, point, normal, best_t, tri_idx, &bary);
        }
    }

    fn pick_sphere(&mut self, node: NodeId, radius: f32) {
        self.pick_sphere_with_radius(node, radius);
    }

    fn pick_sphere_with_radius(&mut self, node: NodeId, radius: f32) {
        if !self.is_pickable() { return; }
        let model = self.state.model_matrix();
        let center = model.transform_point3(Vec3::ZERO);
        let sx = model.transform_vector3(Vec3::X).length();
        let sy = model.transform_vector3(Vec3::Y).length();
        let sz = model.transform_vector3(Vec3::Z).length();
        let world_radius = radius * sx.max(sy).max(sz);
        if let Some(t) = self.ray.intersect_sphere(center, world_radius) {
            let point = self.ray.origin + self.ray.direction * t;
            let normal = (point - center).normalize();
            self.push_hit(node, point, normal, t, 0, &Vec3::new(1.0/3.0, 1.0/3.0, 1.0/3.0));
        }
    }

    fn pick_indexed_face_set(&mut self, node: NodeId, coord_index: &[i32]) {
        if !self.is_pickable() { return; }
        let coord = self.state.coordinate();
        if coord.points.is_empty() {
            return;
        }
        let model = self.state.model_matrix();
        let tris = collect_ifs_world_triangles(coord, coord_index, model);
        if tris.len() >= IFS_BVH_TRIANGLE_THRESHOLD {
            let bvh = Bvh::from_triangles(tris);
            if let Some(h) = bvh.intersect_ray(self.ray.origin, self.ray.direction, 0.001) {
                let point = self.ray.origin + self.ray.direction * h.t;
                self.push_hit(node, point, h.normal, h.t, h.tri_id, &h.bary);
            }
            return;
        }
        let mut face_points = Vec::new();
        let mut tri_idx = 0u32;
        let mut best_t = f32::MAX;
        let mut best_hit: Option<(Vec3, Vec3, u32, Vec3)> = None;
        for &idx in coord_index {
            if idx < 0 {
                if face_points.len() >= 3 {
                    let v0 = model.transform_point3(coord.points[face_points[0]]);
                    for j in 1..face_points.len() - 1 {
                        let v1 = model.transform_point3(coord.points[face_points[j]]);
                        let v2 = model.transform_point3(coord.points[face_points[j + 1]]);
                        if let Some((t, bary)) = self.ray.intersect_triangle(v0, v1, v2) {
                            if t > 0.001 && t < best_t {
                                best_t = t;
                                let point = self.ray.origin + self.ray.direction * t;
                                let c = (v1 - v0).cross(v2 - v0);
                                let normal = if c.length_squared() > 1e-20 { c.normalize() } else { Vec3::Y };
                                best_hit = Some((point, normal, tri_idx, bary));
                            }
                        }
                        tri_idx += 1;
                    }
                }
                face_points.clear();
            } else {
                if (idx as usize) < coord.points.len() {
                    face_points.push(idx as usize);
                }
            }
        }
        if let Some((point, normal, tri_idx, bary)) = best_hit {
            self.push_hit(node, point, normal, best_t, tri_idx, &bary);
        }
    }

    fn push_hit(
        &mut self,
        node: NodeId,
        point: Vec3,
        normal: Vec3,
        distance: f32,
        tri_idx: u32,
        bary: &Vec3,
    ) {
        let face_index = if self.mode != PickMode::Node {
            Some(tri_idx)
        } else {
            None
        };
        let edge_index = if self.mode == PickMode::Edge {
            Some(closest_edge_from_bary(bary))
        } else {
            None
        };
        self.hits.push(PickHit {
            node,
            point,
            normal,
            distance,
            face_index,
            edge_index,
            barycentric: Some([bary.x, bary.y, bary.z]),
        });
    }

    fn pick_cone(&mut self, node: NodeId, radius: f32, height: f32) {
        if !self.is_pickable() { return; }
        let model = self.state.model_matrix();
        let half_h = height / 2.0;
        let segments = 24u32;
        let mut tri_idx = 0u32;
        let mut best_t = f32::MAX;
        let mut best_hit: Option<(Vec3, Vec3, u32, Vec3)> = None;

        for i in 0..segments {
            let t0 = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
            let t1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / segments as f32;
            let bl = model.transform_point3(Vec3::new(t0.cos() * radius, -half_h, t0.sin() * radius));
            let br = model.transform_point3(Vec3::new(t1.cos() * radius, -half_h, t1.sin() * radius));
            let tip = model.transform_point3(Vec3::new(0.0, half_h, 0.0));

            if let Some((t, bary)) = self.ray.intersect_triangle(bl, br, tip) {
                if t > 0.001 && t < best_t {
                    best_t = t;
                    let point = self.ray.origin + self.ray.direction * t;
                    let c = (br - bl).cross(tip - bl);
                    let normal = if c.length_squared() > 1e-20 { c.normalize() } else { Vec3::Y };
                    best_hit = Some((point, normal, tri_idx, bary));
                }
            }
            tri_idx += 1;
        }

        let center = model.transform_point3(Vec3::new(0.0, -half_h, 0.0));
        for i in 0..segments {
            let t0 = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
            let t1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / segments as f32;
            let p0 = model.transform_point3(Vec3::new(t1.cos() * radius, -half_h, t1.sin() * radius));
            let p1 = model.transform_point3(Vec3::new(t0.cos() * radius, -half_h, t0.sin() * radius));
            if let Some((t, bary)) = self.ray.intersect_triangle(center, p0, p1) {
                if t > 0.001 && t < best_t {
                    best_t = t;
                    let point = self.ray.origin + self.ray.direction * t;
                    best_hit = Some((point, Vec3::new(0.0, -1.0, 0.0), tri_idx, bary));
                }
            }
            tri_idx += 1;
        }
        if let Some((point, normal, tri_idx, bary)) = best_hit {
            self.push_hit(node, point, normal, best_t, tri_idx, &bary);
        }
    }

    fn pick_cylinder(&mut self, node: NodeId, radius: f32, height: f32) {
        if !self.is_pickable() { return; }
        let model = self.state.model_matrix();
        let half_h = height / 2.0;
        let segments = 24u32;
        let mut tri_idx = 0u32;
        let mut best_t = f32::MAX;
        let mut best_hit: Option<(Vec3, Vec3, u32, Vec3)> = None;

        for i in 0..segments {
            let t0 = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
            let t1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / segments as f32;
            let bl = model.transform_point3(Vec3::new(t0.cos() * radius, -half_h, t0.sin() * radius));
            let br = model.transform_point3(Vec3::new(t1.cos() * radius, -half_h, t1.sin() * radius));
            let tl = model.transform_point3(Vec3::new(t0.cos() * radius, half_h, t0.sin() * radius));
            let tr = model.transform_point3(Vec3::new(t1.cos() * radius, half_h, t1.sin() * radius));

            if let Some((t, bary)) = self.ray.intersect_triangle(bl, br, tl) {
                if t > 0.001 && t < best_t {
                    best_t = t;
                    let point = self.ray.origin + self.ray.direction * t;
                    let c = (br - bl).cross(tl - bl);
                    let n = if c.length_squared() > 1e-20 { c.normalize() } else { Vec3::Y };
                    best_hit = Some((point, n, tri_idx, bary));
                }
            }
            tri_idx += 1;
            if let Some((t, bary)) = self.ray.intersect_triangle(br, tr, tl) {
                if t > 0.001 && t < best_t {
                    best_t = t;
                    let point = self.ray.origin + self.ray.direction * t;
                    let c = (tr - br).cross(tl - br);
                    let n = if c.length_squared() > 1e-20 { c.normalize() } else { Vec3::Y };
                    best_hit = Some((point, n, tri_idx, bary));
                }
            }
            tri_idx += 1;
        }

        let top_center = model.transform_point3(Vec3::new(0.0, half_h, 0.0));
        for i in 0..segments {
            let t0 = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
            let t1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / segments as f32;
            let p0 = model.transform_point3(Vec3::new(t0.cos() * radius, half_h, t0.sin() * radius));
            let p1 = model.transform_point3(Vec3::new(t1.cos() * radius, half_h, t1.sin() * radius));
            if let Some((t, bary)) = self.ray.intersect_triangle(top_center, p0, p1) {
                if t > 0.001 && t < best_t {
                    best_t = t;
                    let point = self.ray.origin + self.ray.direction * t;
                    best_hit = Some((point, Vec3::new(0.0, 1.0, 0.0), tri_idx, bary));
                }
            }
            tri_idx += 1;
        }

        let bot_center = model.transform_point3(Vec3::new(0.0, -half_h, 0.0));
        for i in 0..segments {
            let t0 = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
            let t1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / segments as f32;
            let p0 = model.transform_point3(Vec3::new(t1.cos() * radius, -half_h, t1.sin() * radius));
            let p1 = model.transform_point3(Vec3::new(t0.cos() * radius, -half_h, t0.sin() * radius));
            if let Some((t, bary)) = self.ray.intersect_triangle(bot_center, p0, p1) {
                if t > 0.001 && t < best_t {
                    best_t = t;
                    let point = self.ray.origin + self.ray.direction * t;
                    best_hit = Some((point, Vec3::new(0.0, -1.0, 0.0), tri_idx, bary));
                }
            }
            tri_idx += 1;
        }
        if let Some((point, normal, tri_idx, bary)) = best_hit {
            self.push_hit(node, point, normal, best_t, tri_idx, &bary);
        }
    }
}

fn collect_ifs_world_triangles(
    coord: &crate::element::CoordinateElement,
    coord_index: &[i32],
    model: Mat4,
) -> Vec<BvhTriangle> {
    let mut out = Vec::new();
    let mut face_points = Vec::new();
    let mut tri_idx = 0u32;
    for &idx in coord_index {
        if idx < 0 {
            if face_points.len() >= 3 {
                let v0 = model.transform_point3(coord.points[face_points[0]]);
                for j in 1..face_points.len() - 1 {
                    let v1 = model.transform_point3(coord.points[face_points[j]]);
                    let v2 = model.transform_point3(coord.points[face_points[j + 1]]);
                    out.push(BvhTriangle {
                        vertices: [v0, v1, v2],
                        id: tri_idx,
                    });
                    tri_idx += 1;
                }
            }
            face_points.clear();
        } else {
            if (idx as usize) < coord.points.len() {
                face_points.push(idx as usize);
            }
        }
    }
    out
}

/// Given barycentric coords (w0, w1, w2), return which edge is closest (0, 1, or 2).
fn closest_edge_from_bary(bary: &Vec3) -> u32 {
    let weights = [bary.x, bary.y, bary.z];
    let min_idx = weights.iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    min_idx as u32
}

impl crate::Action for RayPickAction {
    fn kind(&self) -> crate::ActionKind {
        crate::ActionKind::RayPick
    }

    fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        self.traverse_node(graph, root);
        self.hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    }
}
