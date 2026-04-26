use rc3d_actions::{LightData, LightType, State};
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

use crate::vertex::Vertex;

/// Collected draw data from scene graph traversal.
#[derive(Clone, Debug, Default)]
pub struct DrawCall {
    pub vertices: Vec<Vertex>,
    pub indices: Option<Vec<u32>>,
    pub mvp: Mat4,
    pub model_matrix: Mat4,
    pub camera_pos: Vec3,
    pub light_dir: Vec3,
    pub light_color: Vec3,
    pub color: Vec3,
    pub aabb: Option<rc3d_actions::Aabb>,
}

/// Traverses the scene graph, accumulates state, and collects draw calls.
pub struct RenderCollector {
    pub state: State,
    pub draw_calls: Vec<DrawCall>,
    pub camera_pos: Vec3,
    pub first_light_dir: Vec3,
    pub first_light_color: Vec3,
}

impl RenderCollector {
    pub fn new() -> Self {
        Self {
            state: State::new(),
            draw_calls: Vec::new(),
            camera_pos: Vec3::new(0.0, 0.0, 5.0),
            first_light_dir: Vec3::new(0.0, 0.0, -1.0),
            first_light_color: Vec3::ONE,
        }
    }

    pub fn traverse(&mut self, graph: &SceneGraph, root: NodeId) {
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
            NodeData::Coordinate3(coord) => {
                self.state.set_coordinate(coord.point.clone());
            }
            NodeData::Normal(norm) => {
                self.state.set_normal(norm.vector.clone());
            }
            NodeData::Material(mat) => {
                self.state.set_material(rc3d_actions::MaterialElement {
                    diffuse: mat.diffuse_color,
                    ambient: mat.ambient_color,
                    specular: mat.specular_color,
                    shininess: mat.shininess,
                });
            }
            NodeData::PerspectiveCamera(cam) => {
                self.state.set_view_matrix(cam.view_matrix());
                self.state.set_projection_matrix(cam.projection_matrix());
                self.camera_pos = cam.position;
            }
            NodeData::OrthographicCamera(cam) => {
                self.state.set_view_matrix(cam.view_matrix());
                self.state.set_projection_matrix(cam.projection_matrix());
                self.camera_pos = cam.position;
            }
            NodeData::DirectionalLight(light) => {
                if self.state.lights().is_empty() {
                    self.first_light_dir = light.direction;
                    self.first_light_color = light.color * light.intensity;
                }
                self.state.add_light(LightData {
                    light_type: LightType::Directional,
                    direction: light.direction,
                    location: Vec3::ZERO,
                    color: light.color,
                    intensity: light.intensity,
                });
            }
            NodeData::PointLight(light) => {
                self.state.add_light(LightData {
                    light_type: LightType::Point,
                    direction: Vec3::ZERO,
                    location: light.location,
                    color: light.color,
                    intensity: light.intensity,
                });
            }
            NodeData::SpotLight(light) => {
                self.state.add_light(LightData {
                    light_type: LightType::Spot,
                    direction: light.direction,
                    location: light.location,
                    color: light.color,
                    intensity: light.intensity,
                });
            }
            NodeData::Triangle(_) => {
                let coord = self.state.coordinate();
                if coord.points.len() < 3 {
                    return;
                }
                let normals = self.state.normal();
                let face_n = if normals.vectors.len() >= 3 {
                    normals.vectors[..3].to_vec()
                } else {
                    let n = (coord.points[1] - coord.points[0])
                        .cross(coord.points[2] - coord.points[0])
                        .normalize();
                    vec![n, n, n]
                };
                let vertices: Vec<Vertex> = coord.points[..3]
                    .iter()
                    .zip(face_n.iter())
                    .map(|(p, n)| Vertex {
                        position: p.to_array(),
                        normal: n.to_array(),
                    })
                    .collect();
                self.emit_draw_call(vertices, None);
            }
            NodeData::Cube(cube) => {
                let (vertices, indices) = tessellate_cube(cube.width, cube.height, cube.depth);
                self.emit_draw_call(vertices, Some(indices));
            }
            NodeData::Sphere(sphere) => {
                let (vertices, indices) = tessellate_sphere(sphere.radius, 24, 16);
                self.emit_draw_call(vertices, Some(indices));
            }
            NodeData::Cone(cone) => {
                let (vertices, indices) = tessellate_cone(cone.bottom_radius, cone.height, 24);
                self.emit_draw_call(vertices, Some(indices));
            }
            NodeData::Cylinder(cyl) => {
                let (vertices, indices) = tessellate_cylinder(cyl.radius, cyl.height, 24);
                self.emit_draw_call(vertices, Some(indices));
            }
            NodeData::IndexedFaceSet(ifs) => {
                let coord = self.state.coordinate();
                if coord.points.is_empty() {
                    return;
                }
                let mut vertices = Vec::new();
                let mut indices = Vec::new();
                let mut vi = 0u32;
                let mut face_points = Vec::new();
                for &idx in &ifs.coord_index {
                    if idx < 0 {
                        if face_points.len() >= 3 {
                            for j in 1..face_points.len() - 1 {
                                let p0: Vec3 = coord.points[face_points[0]];
                                let p1: Vec3 = coord.points[face_points[j]];
                                let p2: Vec3 = coord.points[face_points[j + 1]];
                                let n = (p1 - p0).cross(p2 - p0).normalize();
                                vertices.push(Vertex { position: p0.to_array(), normal: n.to_array() });
                                vertices.push(Vertex { position: p1.to_array(), normal: n.to_array() });
                                vertices.push(Vertex { position: p2.to_array(), normal: n.to_array() });
                                indices.extend_from_slice(&[vi, vi + 1, vi + 2]);
                                vi += 3;
                            }
                        }
                        face_points.clear();
                    } else {
                        face_points.push(idx as usize);
                    }
                }
                if !vertices.is_empty() {
                    self.emit_draw_call(vertices, Some(indices));
                }
            }
        }
    }

    fn emit_draw_call(&mut self, vertices: Vec<Vertex>, indices: Option<Vec<u32>>) {
        let model = self.state.model_matrix();
        let mvp = self.state.projection_matrix() * self.state.view_matrix() * model;
        let mat = self.state.material();

        // Compute world-space AABB from transformed vertices
        let aabb = if vertices.is_empty() {
            None
        } else {
            let first = model.transform_point3(Vec3::from_array(vertices[0].position));
            let mut aabb = rc3d_actions::Aabb::from_point(first);
            for v in &vertices[1..] {
                let p = model.transform_point3(Vec3::from_array(v.position));
                aabb = aabb.union(&rc3d_actions::Aabb::from_point(p));
            }
            Some(aabb)
        };

        self.draw_calls.push(DrawCall {
            vertices,
            indices,
            mvp,
            model_matrix: model,
            camera_pos: self.camera_pos,
            light_dir: self.first_light_dir,
            light_color: self.first_light_color,
            color: mat.diffuse,
            aabb,
        });
    }
}

impl Default for RenderCollector {
    fn default() -> Self {
        Self::new()
    }
}

// --- Shape tessellation ---

fn tessellate_cube(w: f32, h: f32, d: f32) -> (Vec<Vertex>, Vec<u32>) {
    let hw = w / 2.0;
    let hh = h / 2.0;
    let hd = d / 2.0;
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 0.0, 1.0], [[-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd]]),
        ([0.0, 0.0, -1.0], [[hw, -hh, -hd], [-hw, -hh, -hd], [-hw, hh, -hd], [hw, hh, -hd]]),
        ([1.0, 0.0, 0.0], [[hw, -hh, hd], [hw, -hh, -hd], [hw, hh, -hd], [hw, hh, hd]]),
        ([-1.0, 0.0, 0.0], [[-hw, -hh, -hd], [-hw, -hh, hd], [-hw, hh, hd], [-hw, hh, -hd]]),
        ([0.0, 1.0, 0.0], [[-hw, hh, hd], [hw, hh, hd], [hw, hh, -hd], [-hw, hh, -hd]]),
        ([0.0, -1.0, 0.0], [[-hw, -hh, -hd], [hw, -hh, -hd], [hw, -hh, hd], [-hw, -hh, hd]]),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, corners) in &faces {
        let base = vertices.len() as u32;
        for c in corners {
            vertices.push(Vertex {
                position: *c,
                normal: *normal,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}

fn tessellate_sphere(radius: f32, slices: u32, stacks: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=stacks {
        let theta = std::f32::consts::PI * i as f32 / stacks as f32;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        for j in 0..=slices {
            let phi = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
            let x = sin_t * phi.cos();
            let y = cos_t;
            let z = sin_t * phi.sin();
            vertices.push(Vertex {
                position: [x * radius, y * radius, z * radius],
                normal: [x, y, z],
            });
        }
    }

    for i in 0..stacks {
        for j in 0..slices {
            let a = i * (slices + 1) + j;
            let b = a + slices + 1;
            indices.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
        }
    }
    (vertices, indices)
}

fn tessellate_cone(radius: f32, height: f32, segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let half_h = height / 2.0;

    for i in 0..=segments {
        let theta = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let x = theta.cos();
        let z = theta.sin();
        let slope_len = (radius * radius + height * height).sqrt();
        let nx = height * x / slope_len;
        let nz = height * z / slope_len;
        let ny = radius / slope_len;
        vertices.push(Vertex {
            position: [x * radius, -half_h, z * radius],
            normal: [nx, ny, nz],
        });
        vertices.push(Vertex {
            position: [0.0, half_h, 0.0],
            normal: [nx, ny, nz],
        });
    }

    for i in 0..segments {
        let bl = i * 2;
        let tl = bl + 1;
        let br = bl + 2;
        let tr = tl + 2;
        indices.extend_from_slice(&[bl, br, tl, br, tr, tl]);
    }

    let cap_base = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, -half_h, 0.0],
        normal: [0.0, -1.0, 0.0],
    });
    for i in 0..=segments {
        let theta = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        vertices.push(Vertex {
            position: [theta.cos() * radius, -half_h, theta.sin() * radius],
            normal: [0.0, -1.0, 0.0],
        });
    }
    for i in 0..segments {
        indices.extend_from_slice(&[cap_base, cap_base + i + 2, cap_base + i + 1]);
    }

    (vertices, indices)
}

fn tessellate_cylinder(radius: f32, height: f32, segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let half_h = height / 2.0;

    for i in 0..=segments {
        let theta = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let x = theta.cos();
        let z = theta.sin();
        vertices.push(Vertex {
            position: [x * radius, -half_h, z * radius],
            normal: [x, 0.0, z],
        });
        vertices.push(Vertex {
            position: [x * radius, half_h, z * radius],
            normal: [x, 0.0, z],
        });
    }

    for i in 0..segments {
        let bl = i * 2;
        let tl = bl + 1;
        let br = bl + 2;
        let tr = tl + 2;
        indices.extend_from_slice(&[bl, br, tl, br, tr, tl]);
    }

    let top_base = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, half_h, 0.0],
        normal: [0.0, 1.0, 0.0],
    });
    for i in 0..=segments {
        let theta = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        vertices.push(Vertex {
            position: [theta.cos() * radius, half_h, theta.sin() * radius],
            normal: [0.0, 1.0, 0.0],
        });
    }
    for i in 0..segments {
        indices.extend_from_slice(&[top_base, top_base + i + 1, top_base + i + 2]);
    }

    let bot_base = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, -half_h, 0.0],
        normal: [0.0, -1.0, 0.0],
    });
    for i in 0..=segments {
        let theta = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        vertices.push(Vertex {
            position: [theta.cos() * radius, -half_h, theta.sin() * radius],
            normal: [0.0, -1.0, 0.0],
        });
    }
    for i in 0..segments {
        indices.extend_from_slice(&[bot_base, bot_base + i + 2, bot_base + i + 1]);
    }

    (vertices, indices)
}
