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
            NodeData::Group(_)
            | NodeData::EventCallback(_)
            | NodeData::SectionPlane(_)
            | NodeData::Switch(_)
            | NodeData::Lod(_)
            | NodeData::MultipleCopy(_)
            | NodeData::Text2(_)
            | NodeData::Text3(_)
            | NodeData::Markup(_)
            | NodeData::Measurement(_) => {
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
                    self.shapes.push(ShapeData {
                        node,
                        world_transform: self.model(),
                        local_aabb: aabb,
                        vertices: self.current_coords.clone(),
                        indices: ifs.coord_index.iter().filter(|&&x| x >= 0).map(|&x| x as u32).collect(),
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

    // Limit checks for performance
    let max_checks = 1000;
    let step_a = (tri_count_a as f32 / (max_checks as f32 / tri_count_b as f32).min(tri_count_b as f32).max(1.0) as f32).max(1.0) as usize;
    let step_b = (tri_count_b as f32 / (max_checks as f32 / tri_count_a as f32).min(tri_count_a as f32).max(1.0) as f32).max(1.0) as usize;

    for ti in (0..a.indices.len()).step_by(step_a.max(1) * 3) {
        if ti + 2 >= a.indices.len() { break; }
        let ai0 = a.indices[ti] as usize;
        let ai1 = a.indices[ti + 1] as usize;
        let ai2 = a.indices[ti + 2] as usize;
        if ai0 >= wa.len() || ai1 >= wa.len() || ai2 >= wa.len() { continue; }
        let ta = [wa[ai0], wa[ai1], wa[ai2]];

        for tj in (0..b.indices.len()).step_by(step_b.max(1) * 3) {
            if tj + 2 >= b.indices.len() { break; }
            let bi0 = b.indices[tj] as usize;
            let bi1 = b.indices[tj + 1] as usize;
            let bi2 = b.indices[tj + 2] as usize;
            if bi0 >= wb.len() || bi1 >= wb.len() || bi2 >= wb.len() { continue; }
            let tb = [wb[bi0], wb[bi1], wb[bi2]];

            if triangles_intersect(ta, tb) {
                let mid = (ta[0] + ta[1] + ta[2] + tb[0] + tb[1] + tb[2]) / 6.0;
                contacts.push(mid);
                if contacts.len() >= 8 { return contacts; } // cap contact points
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
    let dir = n1.cross(n2).normalize();
    if dir.length() < 1e-6 {
        return false; // parallel planes
    }

    // Project onto the intersection line and check overlap
    let project = |p: Vec3| p.dot(dir);
    let a_vals = [project(t1[0]), project(t1[1]), project(t1[2])];
    let b_vals = [project(t2[0]), project(t2[1]), project(t2[2])];

    let a_min = a_vals[0].min(a_vals[1]).min(a_vals[2]);
    let a_max = a_vals[0].max(a_vals[1]).max(a_vals[2]);
    let b_min = b_vals[0].min(b_vals[1]).min(b_vals[2]);
    let b_max = b_vals[0].max(b_vals[1]).max(b_vals[2]);

    a_min <= b_max && b_min <= a_max
}
