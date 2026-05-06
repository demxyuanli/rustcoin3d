//! SoIntersectionDetectionAction — pairwise geometry intersection detection.
//!
//! Traverses the scene graph, collects shape nodes with world-space transforms,
//! and tests pairs for intersection (AABB pre-filter + triangle-triangle test).
//! Returns a list of intersecting shape pairs with world-space hit points.

use rc3d_core::aabb::Aabb;
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

/// Result of an intersection test between two shapes.
#[derive(Clone, Debug)]
pub struct IntersectionResult {
    pub node_a: NodeId,
    pub node_b: NodeId,
    pub contact_points: Vec<Vec3>,
}

/// Collects shape-geometry data during scene traversal.
#[derive(Clone, Debug)]
struct ShapeData {
    node: NodeId,
    world_transform: Mat4,
    local_aabb: Aabb,
    vertices: Vec<Vec3>,
    indices: Vec<u32>,
}

/// Action that detects intersecting geometry pairs in the scene graph.
pub struct IntersectionDetectionAction {
    /// All shape nodes found during traversal.
    shapes: Vec<ShapeData>,
    /// All intersecting pairs.
    pub results: Vec<IntersectionResult>,
    /// Current model matrix during traversal.
    model_stack: Vec<Mat4>,
    /// Current coordinate set during traversal.
    current_coords: Vec<Vec3>,
}

impl IntersectionDetectionAction {
    pub fn new() -> Self {
        Self {
            shapes: Vec::new(),
            results: Vec::new(),
            model_stack: vec![Mat4::IDENTITY],
            current_coords: Vec::new(),
        }
    }

    fn model(&self) -> Mat4 {
        *self.model_stack.last().unwrap_or(&Mat4::IDENTITY)
    }

    /// Run the intersection test on the entire scene graph.
    pub fn detect(&mut self, graph: &SceneGraph) {
        self.shapes.clear();
        self.results.clear();

        for &root in graph.roots() {
            self.traverse(graph, root);
        }

        // Pairwise test with AABB pre-filter
        let n = self.shapes.len();
        for i in 0..n {
            let a = &self.shapes[i];
            let aabb_a = a.local_aabb.transform(a.world_transform);
            for j in (i + 1)..n {
                let b = &self.shapes[j];
                let aabb_b = b.local_aabb.transform(b.world_transform);

                // AABB pre-filter
                if !aabb_a.intersects(&aabb_b) {
                    continue;
                }

                // Triangle-triangle test
                let contacts = test_shape_pair(a, b);
                if !contacts.is_empty() {
                    self.results.push(IntersectionResult {
                        node_a: a.node,
                        node_b: b.node,
                        contact_points: contacts,
                    });
                }
            }
        }
    }

    fn traverse(&mut self, graph: &SceneGraph, node: NodeId) {
        let Some(entry) = graph.get(node) else { return };

        match &entry.data {
            NodeData::Separator(_) => {
                for &child in &entry.children {
                    self.traverse(graph, child);
                }
            }
            NodeData::Group(_) | NodeData::Environment(_) | NodeData::ShapeHints(_) | NodeData::Annotation(_) | NodeData::ResetTransform(_) | NodeData::Texture2Transform(_) | NodeData::MaterialBinding(_) | NodeData::IndexedLineSet(_) | NodeData::File(_) | NodeData::Decal(_) | NodeData::Billboard(_)
            | NodeData::EventCallback(_)
            | NodeData::SectionPlane(_)
            | NodeData::Switch(_)
            | NodeData::Lod(_)
            | NodeData::MultipleCopy(_)
            | NodeData::Text2(_)
            | NodeData::Text3(_)
            | NodeData::Markup(_)
            | NodeData::Measurement(_)
            | NodeData::MorphTarget(_)
            | NodeData::SkinnedMesh(_) => {
                for &child in &entry.children {
                    self.traverse(graph, child);
                }
            }
            NodeData::HandlerNode(h) => {
                h.traverse(graph, node, &entry.children, &mut |id| self.traverse(graph, id));
            }
            NodeData::Transform(t) => {
                let m = self.model() * t.to_matrix();
                self.model_stack.push(m);
                for &child in &entry.children {
                    self.traverse(graph, child);
                }
                self.model_stack.pop();
            }
            NodeData::Coordinate3(c) => {
                self.current_coords = c.point.clone();
                for &child in &entry.children {
                    self.traverse(graph, child);
                }
            }
            // Shape nodes: collect geometry
            NodeData::Cube(cube) => {
                let hw = cube.width * 0.5;
                let hh = cube.height * 0.5;
                let hd = cube.depth * 0.5;
                let verts = vec![
                    Vec3::new(-hw, -hh, -hd), Vec3::new(hw, -hh, -hd),
                    Vec3::new(hw, -hh, hd), Vec3::new(-hw, -hh, hd),
                    Vec3::new(-hw, hh, -hd), Vec3::new(hw, hh, -hd),
                    Vec3::new(hw, hh, hd), Vec3::new(-hw, hh, hd),
                ];
                let idx: Vec<u32> = vec![
                    0,1,2,0,2,3,4,6,5,4,7,6,
                    0,4,5,0,5,1,1,5,6,1,6,2,
                    2,6,7,2,7,3,3,7,4,3,4,0,
                ];
                self.shapes.push(ShapeData {
                    node,
                    world_transform: self.model(),
                    local_aabb: Aabb { min: Vec3::new(-hw, -hh, -hd), max: Vec3::new(hw, hh, hd) },
                    vertices: verts,
                    indices: idx,
                });
            }
            NodeData::Sphere(s) => {
                let r = s.radius;
                let aabb = Aabb { min: Vec3::splat(-r), max: Vec3::splat(r) };
                self.shapes.push(ShapeData {
                    node,
                    world_transform: self.model(),
                    local_aabb: aabb,
                    vertices: Vec::new(),
                    indices: Vec::new(),
                });
            }
            NodeData::Cylinder(cyl) => {
                let r = cyl.radius;
                let hh = cyl.height * 0.5;
                let aabb = Aabb { min: Vec3::new(-r, -hh, -r), max: Vec3::new(r, hh, r) };
                self.shapes.push(ShapeData {
                    node,
                    world_transform: self.model(),
                    local_aabb: aabb,
                    vertices: Vec::new(),
                    indices: Vec::new(),
                });
            }
            NodeData::Cone(c) => {
                let r = c.bottom_radius;
                let hh = c.height * 0.5;
                let aabb = Aabb { min: Vec3::new(-r, -hh, -r), max: Vec3::new(r, hh, r) };
                self.shapes.push(ShapeData {
                    node,
                    world_transform: self.model(),
                    local_aabb: aabb,
                    vertices: Vec::new(),
                    indices: Vec::new(),
                });
            }
            NodeData::Triangle(_) => {
                if self.current_coords.len() >= 3 {
                    let min = self.current_coords[0];
                    let max = self.current_coords[0];
                    let mut aabb = Aabb { min, max };
                    for p in &self.current_coords[1..] {
                        aabb.min = aabb.min.min(*p);
                        aabb.max = aabb.max.max(*p);
                    }
                    let idx: Vec<u32> = (0..self.current_coords.len() as u32).collect();
                    self.shapes.push(ShapeData {
                        node,
                        world_transform: self.model(),
                        local_aabb: aabb,
                        vertices: self.current_coords.clone(),
                        indices: idx,
                    });
                }
            }
            NodeData::IndexedFaceSet(ifs) => {
                if !self.current_coords.is_empty() {
                    let min = self.current_coords[0];
                    let max = self.current_coords[0];
                    let mut aabb = Aabb { min, max };
                    for p in &self.current_coords[1..] {
                        aabb.min = aabb.min.min(*p);
                        aabb.max = aabb.max.max(*p);
                    }
                    // Fan-triangulate polygon faces (split on -1 separators)
                    let mut indices = Vec::new();
                    let mut face_start = 0;
                    for (k, &ci) in ifs.coord_index.iter().enumerate() {
                        if ci == -1 {
                            let face: Vec<u32> = ifs.coord_index[face_start..k]
                                .iter().map(|&x| x as u32).collect();
                            if face.len() >= 3 {
                                for m in 1..face.len() - 1 {
                                    indices.push(face[0]);
                                    indices.push(face[m]);
                                    indices.push(face[m + 1]);
                                }
                            }
                            face_start = k + 1;
                        }
                    }
                    // Handle last face (no trailing -1)
                    let tail: Vec<u32> = ifs.coord_index[face_start..]
                        .iter().filter(|&&x| x >= 0).map(|&x| x as u32).collect();
                    if tail.len() >= 3 {
                        for m in 1..tail.len() - 1 {
                            indices.push(tail[0]);
                            indices.push(tail[m]);
                            indices.push(tail[m + 1]);
                        }
                    }
                    self.shapes.push(ShapeData {
                        node,
                        world_transform: self.model(),
                        local_aabb: aabb,
                        vertices: self.current_coords.clone(),
                        indices,
                    });
                }
            }
            _ => {}
        }
    }
}

impl Default for IntersectionDetectionAction {
    fn default() -> Self {
        Self::new()
    }
}

/// Test if two shapes intersect using AABB + triangle-triangle tests.
fn test_shape_pair(a: &ShapeData, b: &ShapeData) -> Vec<Vec3> {
    let mut contacts = Vec::new();

    // If either shape is a primitive (no triangles), use AABB intersection
    if a.vertices.is_empty() || b.vertices.is_empty() || a.indices.is_empty() || b.indices.is_empty() {
        let aabb_a = a.local_aabb.transform(a.world_transform);
        let aabb_b = b.local_aabb.transform(b.world_transform);
        if aabb_a.intersects(&aabb_b) {
            contacts.push(aabb_a.center());
        }
        return contacts;
    }

    // Transform vertices to world space for both shapes
    let wa: Vec<Vec3> = a.vertices.iter().map(|v| a.world_transform.transform_point3(*v)).collect();
    let wb: Vec<Vec3> = b.vertices.iter().map(|v| b.world_transform.transform_point3(*v)).collect();

    // Triangle-triangle intersection (Möller's algorithm)
    let tri_count_a = a.indices.len() / 3;
    let tri_count_b = b.indices.len() / 3;

    // Uniform sampling step to stay within max_checks budget
    let max_checks = 1000;
    let step = ((tri_count_a * tri_count_b) as f32 / max_checks as f32).sqrt().ceil().max(1.0) as usize;
    let offset = (step / 2).max(1) * 3; // start offset avoids always testing index 0

    let step_bytes = step.max(1) * 3;
    for ti in (offset..a.indices.len()).step_by(step_bytes) {
        if ti + 2 >= a.indices.len() { break; }
        let ai0 = a.indices[ti] as usize;
        let ai1 = a.indices[ti + 1] as usize;
        let ai2 = a.indices[ti + 2] as usize;
        if ai0 >= wa.len() || ai1 >= wa.len() || ai2 >= wa.len() { continue; }
        let ta = [wa[ai0], wa[ai1], wa[ai2]];

        for tj in (offset..b.indices.len()).step_by(step_bytes) {
            if tj + 2 >= b.indices.len() { break; }
            let bi0 = b.indices[tj] as usize;
            let bi1 = b.indices[tj + 1] as usize;
            let bi2 = b.indices[tj + 2] as usize;
            if bi0 >= wb.len() || bi1 >= wb.len() || bi2 >= wb.len() { continue; }
            let tb = [wb[bi0], wb[bi1], wb[bi2]];

            if triangles_intersect(ta, tb) {
                let mid = (ta[0] + ta[1] + ta[2] + tb[0] + tb[1] + tb[2]) / 6.0;
                contacts.push(mid);
                if contacts.len() >= 8 { return contacts; }
            }
        }
    }

    contacts
}

/// Möller triangle-triangle intersection test.
fn triangles_intersect(t1: [Vec3; 3], t2: [Vec3; 3]) -> bool {
    // Compute plane of t1
    let n1 = (t1[1] - t1[0]).cross(t1[2] - t1[0]).normalize();
    let d1 = -n1.dot(t1[0]);

    // Signed distances of t2 vertices to t1's plane
    let sd: [f32; 3] = [
        n1.dot(t2[0]) + d1,
        n1.dot(t2[1]) + d1,
        n1.dot(t2[2]) + d1,
    ];

    // All same side → no intersection
    if sd[0].signum() == sd[1].signum() && sd[1].signum() == sd[2].signum() && sd[0].abs() > 1e-6 {
        return false;
    }

    // Compute plane of t2
    let n2 = (t2[1] - t2[0]).cross(t2[2] - t2[0]).normalize();
    let d2 = -n2.dot(t2[0]);

    let sd2: [f32; 3] = [
        n2.dot(t1[0]) + d2,
        n2.dot(t1[1]) + d2,
        n2.dot(t1[2]) + d2,
    ];

    if sd2[0].signum() == sd2[1].signum() && sd2[1].signum() == sd2[2].signum() && sd2[0].abs() > 1e-6 {
        return false;
    }

    // Compute intersection line direction
    let dir = n1.cross(n2);
    let dir_len = dir.length();
    if dir_len < 1e-12 {
        return false; // parallel planes
    }
    let dir = dir / dir_len;

    // Compute the two intersection points where triangle 2's edges cross plane 1
    let interval_a = compute_interval(t2, n1, d1, dir);
    let interval_b = compute_interval(t1, n2, d2, dir);

    interval_a[0] <= interval_b[1] + 1e-6 && interval_b[0] <= interval_a[1] + 1e-6
}

/// Compute the intersection interval of a triangle with a plane, projected onto dir.
fn compute_interval(tri: [Vec3; 3], plane_n: Vec3, plane_d: f32, dir: Vec3) -> [f32; 2] {
    let ds: [f32; 3] = [
        plane_n.dot(tri[0]) + plane_d,
        plane_n.dot(tri[1]) + plane_d,
        plane_n.dot(tri[2]) + plane_d,
    ];

    let mut interval = [f32::NEG_INFINITY, f32::INFINITY];
    for i in 0..3 {
        let j = (i + 1) % 3;
        let d_i = ds[i];
        let d_j = ds[j];

        // Edge crosses the plane
        if d_i * d_j < 0.0 {
            let t = d_i / (d_i - d_j);
            let p = tri[i] + (tri[j] - tri[i]) * t;
            let proj = p.dot(dir);
            interval[0] = interval[0].max(proj);
            interval[1] = interval[1].min(proj);
        } else if d_i.abs() < 1e-8 {
            // Vertex lies on the plane
            let proj = tri[i].dot(dir);
            interval[0] = interval[0].max(proj);
            interval[1] = interval[1].min(proj);
        }
    }

    if interval[0] > interval[1] {
        interval.swap(0, 1);
    }
    interval
}
