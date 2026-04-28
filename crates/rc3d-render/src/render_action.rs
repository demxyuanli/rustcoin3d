use rc3d_actions::{LightData, LightType, State};
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::{DisplayMode, NodeId};
use rc3d_scene::{NodeData, SceneGraph};
use slotmap::Key;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;

use crate::vertex::Vertex;
const MAX_LIGHTS: usize = 4;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ShapeKey {
    Cube { w: u32, h: u32, d: u32 },
    Sphere { r: u32, slices: u32, stacks: u32 },
    Cone { r: u32, h: u32, segments: u32 },
    Cylinder { r: u32, h: u32, segments: u32 },
    IndexedFaceSet {
        node: u64,
        coord_len: u32,
        coord_index_len: u32,
        points_sig: [u32; 6],
        index_sig: [i32; 2],
    },
}

type CachedShapeData = (Arc<Vec<Vertex>>, Arc<Vec<u32>>, Arc<Vec<[f32; 3]>>, rc3d_core::Aabb);
type PackedLights = (
    [[f32; 4]; MAX_LIGHTS],
    [[f32; 4]; MAX_LIGHTS],
    [[f32; 4]; MAX_LIGHTS],
    [[f32; 4]; MAX_LIGHTS],
    [[f32; 4]; MAX_LIGHTS],
    u32,
);

const MAX_EDGE_POSITIONS: usize = 2_000_000;

/// Collected draw data from scene graph traversal.
#[derive(Clone, Debug, Default)]
pub struct DrawCall {
    pub vertices: Arc<Vec<Vertex>>,
    pub indices: Option<Arc<Vec<u32>>>,
    pub edge_positions: Arc<Vec<[f32; 3]>>,
    pub mvp: Mat4,
    pub model_matrix: Mat4,
    pub camera_pos: Vec3,
    pub light_dirs: [[f32; 4]; MAX_LIGHTS],
    pub light_colors: [[f32; 4]; MAX_LIGHTS],
    pub light_types: [[f32; 4]; MAX_LIGHTS],
    pub light_positions: [[f32; 4]; MAX_LIGHTS],
    pub spot_params: [[f32; 4]; MAX_LIGHTS],
    pub light_count: u32,
    pub diffuse_color: Vec3,
    pub ambient_color: Vec3,
    pub specular_color: Vec3,
    pub shininess: f32,
    pub aabb: Option<rc3d_core::Aabb>,
    pub display_mode: DisplayMode,
    pub selected: bool,
    pub overlay_color: Option<[f32; 4]>,
    /// Cached mesh hash for GPU upload dedup. None until first computed.
    pub mesh_hash: Option<u64>,
}

/// Traverses the scene graph, accumulates state, and collects draw calls.
pub struct RenderCollector {
    pub state: State,
    pub draw_calls: Vec<DrawCall>,
    pub camera_pos: Vec3,
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
    pub global_display_mode: DisplayMode,
    mesh_cache: HashMap<ShapeKey, CachedShapeData>,
}

impl RenderCollector {
    pub fn new() -> Self {
        Self {
            state: State::new(),
            draw_calls: Vec::new(),
            camera_pos: Vec3::new(0.0, 0.0, 5.0),
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            global_display_mode: DisplayMode::ShadedWithEdges,
            mesh_cache: HashMap::new(),
        }
    }

    pub fn traverse(&mut self, graph: &SceneGraph, root: NodeId) {
        self.traverse_node(graph, root);
    }

    pub fn invalidate_mesh_cache(&mut self) {
        self.mesh_cache.clear();
    }

    fn traverse_node(&mut self, graph: &SceneGraph, node: NodeId) {
        let Some(entry) = graph.get(node) else {
            return;
        };
        let is_selected = graph.is_selected(node);
        let node_display_mode = entry.display_mode;

        match &entry.data {
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
                self.state.add_light(LightData {
                    light_type: LightType::Directional,
                    direction: light.direction,
                    location: Vec3::ZERO,
                    color: light.color,
                    intensity: light.intensity,
                    cut_off_angle: 0.0,
                    drop_off_rate: 0.0,
                });
            }
            NodeData::PointLight(light) => {
                self.state.add_light(LightData {
                    light_type: LightType::Point,
                    direction: Vec3::ZERO,
                    location: light.location,
                    color: light.color,
                    intensity: light.intensity,
                    cut_off_angle: 0.0,
                    drop_off_rate: 0.0,
                });
            }
            NodeData::SpotLight(light) => {
                self.state.add_light(LightData {
                    light_type: LightType::Spot,
                    direction: light.direction,
                    location: light.location,
                    color: light.color,
                    intensity: light.intensity,
                    cut_off_angle: light.cut_off_angle,
                    drop_off_rate: light.drop_off_rate,
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
                self.emit_cached_shape(
                    ShapeKey::Cube {
                        w: cube.width.to_bits(),
                        h: cube.height.to_bits(),
                        d: cube.depth.to_bits(),
                    },
                    || rc3d_mesh::tessellate_cube(cube.width, cube.height, cube.depth),
                    is_selected,
                    node_display_mode,
                );
            }
            NodeData::Sphere(sphere) => {
                const SLICES: u32 = 24;
                const STACKS: u32 = 16;
                self.emit_cached_shape(
                    ShapeKey::Sphere {
                        r: sphere.radius.to_bits(),
                        slices: SLICES,
                        stacks: STACKS,
                    },
                    || rc3d_mesh::tessellate_sphere(sphere.radius, SLICES, STACKS),
                    is_selected,
                    node_display_mode,
                );
            }
            NodeData::Cone(cone) => {
                const SEGMENTS: u32 = 24;
                self.emit_cached_shape(
                    ShapeKey::Cone {
                        r: cone.bottom_radius.to_bits(),
                        h: cone.height.to_bits(),
                        segments: SEGMENTS,
                    },
                    || rc3d_mesh::tessellate_cone(cone.bottom_radius, cone.height, SEGMENTS),
                    is_selected,
                    node_display_mode,
                );
            }
            NodeData::Cylinder(cyl) => {
                const SEGMENTS: u32 = 24;
                self.emit_cached_shape(
                    ShapeKey::Cylinder {
                        r: cyl.radius.to_bits(),
                        h: cyl.height.to_bits(),
                        segments: SEGMENTS,
                    },
                    || rc3d_mesh::tessellate_cylinder(cyl.radius, cyl.height, SEGMENTS),
                    is_selected,
                    node_display_mode,
                );
            }
            NodeData::IndexedFaceSet(ifs) => {
                let coord = self.state.coordinate();
                if coord.points.is_empty() {
                    return;
                }
                let points_sig = if coord.points.len() >= 2 {
                    let first = coord.points[0].to_array();
                    let last = coord.points[coord.points.len() - 1].to_array();
                    [
                        first[0].to_bits(),
                        first[1].to_bits(),
                        first[2].to_bits(),
                        last[0].to_bits(),
                        last[1].to_bits(),
                        last[2].to_bits(),
                    ]
                } else {
                    let p = coord.points[0].to_array();
                    [
                        p[0].to_bits(),
                        p[1].to_bits(),
                        p[2].to_bits(),
                        p[0].to_bits(),
                        p[1].to_bits(),
                        p[2].to_bits(),
                    ]
                };
                let index_sig = if ifs.coord_index.len() >= 2 {
                    [ifs.coord_index[0], ifs.coord_index[ifs.coord_index.len() - 1]]
                } else if ifs.coord_index.len() == 1 {
                    [ifs.coord_index[0], ifs.coord_index[0]]
                } else {
                    [0, 0]
                };
                let key = ShapeKey::IndexedFaceSet {
                    node: node.data().as_ffi(),
                    coord_len: coord.points.len() as u32,
                    coord_index_len: ifs.coord_index.len() as u32,
                    points_sig,
                    index_sig,
                };
                #[allow(clippy::map_entry)]
                if !self.mesh_cache.contains_key(&key) {
                    let mesh = rc3d_mesh::TriangleMesh::from_indexed_face_set(&coord.points, &ifs.coord_index);
                    if !mesh.positions.is_empty() {
                        let (phong_verts, indices) = mesh.phong_buffers();
                        let edge_positions = mesh.edge_line_positions();
                        let local_aabb = mesh.bounding_box();
                        let vertices: Vec<Vertex> = phong_verts
                            .iter()
                            .map(|v| Vertex {
                                position: [v[0], v[1], v[2]],
                                normal: [v[3], v[4], v[5]],
                            })
                            .collect();
                        self.mesh_cache.insert(key, (Arc::new(vertices), Arc::new(indices), Arc::new(edge_positions), local_aabb));
                    }
                }
                if let Some((vertices, indices, edge_positions, local_aabb)) = self.mesh_cache.get(&key) {
                    let edge_positions = if edge_positions.len() > MAX_EDGE_POSITIONS {
                        Arc::new(Vec::new())
                    } else {
                        Arc::clone(edge_positions)
                    };
                    self.emit_draw_call_with_cached_aabb(
                        Arc::clone(vertices),
                        Some(Arc::clone(indices)),
                        edge_positions,
                        local_aabb.clone(),
                        is_selected,
                        node_display_mode,
                    );
                }
            }
        }
    }

    fn emit_cached_shape<F>(
        &mut self,
        key: ShapeKey,
        build_mesh: F,
        selected: bool,
        node_display_mode: Option<DisplayMode>,
    ) where
        F: FnOnce() -> rc3d_mesh::TriangleMesh,
    {
        match self.mesh_cache.entry(key) {
            Entry::Occupied(_) => {}
            Entry::Vacant(vacant) => {
                let mesh = build_mesh();
                if mesh.positions.is_empty() {
                    return;
                }
                let (phong_verts, indices) = mesh.phong_buffers();
                let edge_positions = mesh.edge_line_positions();
                let local_aabb = mesh.bounding_box();
                let vertices: Vec<Vertex> = phong_verts
                    .iter()
                    .map(|v| Vertex {
                        position: [v[0], v[1], v[2]],
                        normal: [v[3], v[4], v[5]],
                    })
                    .collect();
                vacant.insert((Arc::new(vertices), Arc::new(indices), Arc::new(edge_positions), local_aabb));
            }
        }

        if let Some((vertices, indices, edge_positions, local_aabb)) = self.mesh_cache.get(&key) {
            let edge_positions = if edge_positions.len() > MAX_EDGE_POSITIONS {
                Arc::new(Vec::new())
            } else {
                Arc::clone(edge_positions)
            };
            self.emit_draw_call_with_cached_aabb(
                Arc::clone(vertices),
                Some(Arc::clone(indices)),
                edge_positions,
                local_aabb.clone(),
                selected,
                node_display_mode,
            );
        }
    }

    fn emit_draw_call_with_cached_aabb(
        &mut self,
        vertices: Arc<Vec<Vertex>>,
        indices: Option<Arc<Vec<u32>>>,
        edge_positions: Arc<Vec<[f32; 3]>>,
        local_aabb: rc3d_core::Aabb,
        selected: bool,
        node_display_mode: Option<DisplayMode>,
    ) {
        let model = self.state.model_matrix();
        let mvp = self.state.projection_matrix() * self.state.view_matrix() * model;
        let mat = self.state.material();
        let aabb = Some(local_aabb.transform(model));
        let (light_dirs, light_colors, light_types, light_positions, spot_params, light_count) =
            self.collect_lights();

        self.draw_calls.push(DrawCall {
            vertices,
            indices,
            edge_positions,
            mvp,
            model_matrix: model,
            camera_pos: self.camera_pos,
            light_dirs,
            light_colors,
            light_types,
            light_positions,
            spot_params,
            light_count,
            diffuse_color: mat.diffuse,
            ambient_color: mat.ambient,
            specular_color: mat.specular,
            shininess: mat.shininess,
            aabb,
            display_mode: node_display_mode.unwrap_or(DisplayMode::ShadedWithEdges),
            selected,
            overlay_color: None,
            mesh_hash: None,
        });
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
        let (light_dirs, light_colors, light_types, light_positions, spot_params, light_count) =
            self.collect_lights();

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
            vertices: Arc::new(vertices),
            indices: indices.map(Arc::new),
            edge_positions: Arc::new(edge_positions),
            mvp,
            model_matrix: model,
            camera_pos: self.camera_pos,
            light_dirs,
            light_colors,
            light_types,
            light_positions,
            spot_params,
            light_count,
            diffuse_color: mat.diffuse,
            ambient_color: mat.ambient,
            specular_color: mat.specular,
            shininess: mat.shininess,
            aabb,
            display_mode: node_display_mode.unwrap_or(DisplayMode::ShadedWithEdges),
            selected,
            overlay_color: None,
            mesh_hash: None,
        });
    }

    fn collect_lights(&self) -> PackedLights {
        let mut dirs = [[0.0f32; 4]; MAX_LIGHTS];
        let mut colors = [[0.0f32; 4]; MAX_LIGHTS];
        let mut types = [[0.0f32; 4]; MAX_LIGHTS];
        let mut positions = [[0.0f32; 4]; MAX_LIGHTS];
        let mut spot_params = [[0.0f32; 4]; MAX_LIGHTS];
        let mut count = 0u32;
        for light in self.state.lights() {
            if (count as usize) < MAX_LIGHTS {
                let idx = count as usize;
                dirs[idx] = [light.direction.x, light.direction.y, light.direction.z, 0.0];
                let c = light.color * light.intensity;
                colors[idx] = [c.x, c.y, c.z, 1.0];
                positions[idx] = [light.location.x, light.location.y, light.location.z, 1.0];
                types[idx][0] = match light.light_type {
                    LightType::Directional => 0.0,
                    LightType::Point => 1.0,
                    LightType::Spot => 2.0,
                };
                spot_params[idx] = [light.cut_off_angle.cos(), light.drop_off_rate, 0.0, 0.0];
                count += 1;
            }
        }
        if count == 0 {
            dirs[0] = [0.0, 0.0, -1.0, 0.0];
            colors[0] = [1.0, 1.0, 1.0, 1.0];
            types[0][0] = 0.0;
            count = 1;
        }
        (dirs, colors, types, positions, spot_params, count)
    }
}

impl Default for RenderCollector {
    fn default() -> Self {
        Self::new()
    }
}
