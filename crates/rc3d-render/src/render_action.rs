use rc3d_actions::{LightData, LightType, State};
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::{NodeId, DisplayMode};
use rc3d_scene::{NodeData, SceneGraph};

use crate::vertex::Vertex;

/// Collected draw data from scene graph traversal.
#[derive(Clone, Debug, Default)]
pub struct DrawCall {
    pub vertices: Vec<Vertex>,
    pub indices: Option<Vec<u32>>,
    pub edge_positions: Vec<[f32; 3]>,
    pub mvp: Mat4,
    pub model_matrix: Mat4,
    pub camera_pos: Vec3,
    pub light_dir: Vec3,
    pub light_color: Vec3,
    pub color: Vec3,
    pub aabb: Option<rc3d_core::Aabb>,
    pub display_mode: DisplayMode,
    pub selected: bool,
    pub overlay_color: Option<[f32; 4]>,
}

/// Traverses the scene graph, accumulates state, and collects draw calls.
pub struct RenderCollector {
    pub state: State,
    pub draw_calls: Vec<DrawCall>,
    pub camera_pos: Vec3,
    pub first_light_dir: Vec3,
    pub first_light_color: Vec3,
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
    pub global_display_mode: DisplayMode,
}

impl RenderCollector {
    pub fn new() -> Self {
        Self {
            state: State::new(),
            draw_calls: Vec::new(),
            camera_pos: Vec3::new(0.0, 0.0, 5.0),
            first_light_dir: Vec3::new(0.0, 0.0, -1.0),
            first_light_color: Vec3::ONE,
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            global_display_mode: DisplayMode::ShadedWithEdges,
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
        let is_selected = graph.is_selected(node);
        let node_display_mode = entry.display_mode;

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
                self.view_matrix = cam.view_matrix();
                self.projection_matrix = cam.projection_matrix();
                self.camera_pos = cam.position;
            }
            NodeData::OrthographicCamera(cam) => {
                self.state.set_view_matrix(cam.view_matrix());
                self.state.set_projection_matrix(cam.projection_matrix());
                self.view_matrix = cam.view_matrix();
                self.projection_matrix = cam.projection_matrix();
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
                let positions = vec![coord.points[0], coord.points[1], coord.points[2]];
                let mesh = rc3d_mesh::TriangleMesh::from_tris(&positions);
                let mut vertices = Vec::with_capacity(3);
                for (i, v) in mesh.phong_buffers().0.iter().enumerate() {
                    let n = if i < face_n.len() { face_n[i].to_array() } else { [v[3], v[4], v[5]] };
                    vertices.push(Vertex { position: [v[0], v[1], v[2]], normal: n });
                }
                let edge_positions = Vec::new();
                self.emit_draw_call_with_edges(vertices, Some((0..3u32).collect()), edge_positions, is_selected, node_display_mode);
            }
            NodeData::Cube(cube) => {
                let mesh = rc3d_mesh::tessellate_cube(cube.width, cube.height, cube.depth);
                self.emit_mesh_draw_call(&mesh, is_selected, node_display_mode);
            }
            NodeData::Sphere(sphere) => {
                let mesh = rc3d_mesh::tessellate_sphere(sphere.radius, 24, 16);
                self.emit_mesh_draw_call(&mesh, is_selected, node_display_mode);
            }
            NodeData::Cone(cone) => {
                let mesh = rc3d_mesh::tessellate_cone(cone.bottom_radius, cone.height, 24);
                self.emit_mesh_draw_call(&mesh, is_selected, node_display_mode);
            }
            NodeData::Cylinder(cyl) => {
                let mesh = rc3d_mesh::tessellate_cylinder(cyl.radius, cyl.height, 24);
                self.emit_mesh_draw_call(&mesh, is_selected, node_display_mode);
            }
            NodeData::IndexedFaceSet(ifs) => {
                let coord = self.state.coordinate();
                if coord.points.is_empty() {
                    return;
                }
                let mesh = rc3d_mesh::TriangleMesh::from_indexed_face_set(&coord.points, &ifs.coord_index);
                if mesh.positions.is_empty() {
                    return;
                }
                self.emit_mesh_draw_call(&mesh, is_selected, node_display_mode);
            }
        }
    }

    fn emit_mesh_draw_call(&mut self, mesh: &rc3d_mesh::TriangleMesh, selected: bool, node_display_mode: Option<DisplayMode>) {
        let (phong_verts, indices) = mesh.phong_buffers();
        let vertices: Vec<Vertex> = phong_verts
            .iter()
            .map(|v| Vertex { position: [v[0], v[1], v[2]], normal: [v[3], v[4], v[5]] })
            .collect();
        self.emit_draw_call_with_edges(vertices, Some(indices), Vec::new(), selected, node_display_mode);
    }

    fn extract_edge_positions(&self, mesh: &rc3d_mesh::TriangleMesh) -> Vec<[f32; 3]> {
        let edge_ids = match self.global_display_mode {
            DisplayMode::ShadedWithEdges | DisplayMode::HiddenLine => {
                let world_view_dir = self.view_direction();
                let model = self.state.model_matrix();
                let local_view_dir = model.inverse().transform_vector3(world_view_dir).normalize();
                let mut edges = mesh.silhouette_edges(local_view_dir);
                edges.extend(mesh.boundary_edges());
                edges
            }
            _ => return Vec::new(),
        };

        let line_indices = mesh.edge_line_indices(&edge_ids);
        line_indices.iter().map(|&idx| mesh.positions[idx as usize].to_array()).collect()
    }

    fn view_direction(&self) -> Vec3 {
        Vec3::new(
            self.view_matrix.z_axis.x,
            self.view_matrix.z_axis.y,
            self.view_matrix.z_axis.z,
        ).normalize()
    }

    fn emit_draw_call_with_edges(
        &mut self,
        vertices: Vec<Vertex>,
        indices: Option<Vec<u32>>,
        edge_positions: Vec<[f32; 3]>,
        selected: bool,
        node_display_mode: Option<DisplayMode>,
    ) {
        let model = self.state.model_matrix();
        let mvp = self.state.projection_matrix() * self.state.view_matrix() * model;
        let mat = self.state.material();

        let aabb = if vertices.is_empty() {
            None
        } else {
            let first = model.transform_point3(Vec3::from_array(vertices[0].position));
            let mut aabb = rc3d_core::Aabb::from_point(first);
            for v in &vertices[1..] {
                let p = model.transform_point3(Vec3::from_array(v.position));
                aabb = aabb.union(&rc3d_core::Aabb::from_point(p));
            }
            Some(aabb)
        };

        self.draw_calls.push(DrawCall {
            vertices,
            indices,
            edge_positions,
            mvp,
            model_matrix: model,
            camera_pos: self.camera_pos,
            light_dir: self.first_light_dir,
            light_color: self.first_light_color,
            color: mat.diffuse,
            aabb,
            display_mode: node_display_mode.unwrap_or(DisplayMode::ShadedWithEdges),
            selected,
            overlay_color: None,
        });
    }
}

impl Default for RenderCollector {
    fn default() -> Self {
        Self::new()
    }
}
