impl SceneVisitor for RenderCollector {
    fn enter_separator(&mut self) {
        self.state.push_all();
    }

    fn leave_separator(&mut self) {
        self.state.pop_all();
    }

    fn appearance(&self) -> Appearance {
        self.state.appearance()
    }

    fn set_appearance(&mut self, app: Appearance) {
        self.state.set_appearance(app);
    }

    fn separator_policy(&self) -> SeparatorPolicy {
        SeparatorPolicy::FlattenDirectTransforms
    }

    fn should_visit(&self, node: NodeId) -> bool {
        !self.hidden_nodes.contains(&node)
    }

    fn set_instance_transforms(&mut self, transforms: &[Mat4]) {
        if transforms.is_empty() {
            self.pending_instance_transforms = None;
        } else {
            self.pending_instance_transforms = Some(Arc::new(transforms.to_vec()));
        }
    }

    fn visit_node(
        &mut self,
        graph: &SceneGraph,
        node: NodeId,
        entry: &NodeEntry,
    ) -> ChildPolicy {
        let is_selected = graph.is_in_selection(node);
        let node_type_label = entry.data.type_name();

        match &entry.data {

            // Group/File and pass-through nodes: traverse children, then skip.
            // RayTracing: wgpu lacks native DXR/VKRT — deferred until wgpu adds ray tracing.
            NodeData::Group(_)
            | NodeData::File(_)
            | NodeData::StereoCamera(_)
            | NodeData::RayTracing(_)
            | NodeData::ShapeHints(_)
            | NodeData::MaterialBinding(_)
            | NodeData::Texture2Transform(_)
            | NodeData::Environment(_)
            | NodeData::CubeCamera(_)
            | NodeData::EventCallback(_)
            | NodeData::PickStyle(_)
            | NodeData::SectionPlane(_)
            | NodeData::Text2(_)
            | NodeData::Text3(_)
            | NodeData::Measurement(_)
            | NodeData::Markup(_)
            | NodeData::Custom(_, _) => self.traverse_entry_children(graph, entry),
            NodeData::Decal(decal) => {
                self.effect_commands.decals.push(crate::render_passes::pass_effects::DecalDrawCommand {
                    model_matrix: self.state.model_matrix(),
                    position: decal.position,
                    direction: decal.direction,
                    size: decal.size,
                    texture_path: decal.texture_path.clone(),
                    color: decal.color,
                    opacity: decal.opacity,
                    is_overlay: self.inside_annotation,
                });
                for &child in &entry.children {
                    scene_traverse(self, graph, child);
                }
                ChildPolicy::Skip
            }
            NodeData::ReflectionPlane(rp) => {
                if !rp.enabled {
                    return ChildPolicy::Recurse;
                }
                // Mirror view matrix across the reflection plane
                let view = self.state.view_matrix();
                let n = rp.normal.normalize();
                let o = rp.origin;
                let eye = view.inverse().w_axis.truncate();
                let d = -(eye - o).dot(n);
                let refl = Mat4::from_cols(
                    Vec4::new(1.0-2.0*n.x*n.x, -2.0*n.y*n.x, -2.0*n.z*n.x, 0.0),
                    Vec4::new(-2.0*n.x*n.y, 1.0-2.0*n.y*n.y, -2.0*n.z*n.y, 0.0),
                    Vec4::new(-2.0*n.x*n.z, -2.0*n.y*n.z, 1.0-2.0*n.z*n.z, 0.0),
                    Vec4::new(2.0*d*n.x, 2.0*d*n.y, 2.0*d*n.z, 1.0),
                );
                let mirrored = view * refl;
                let saved = self.state.view_matrix();
                self.state.set_view_matrix(mirrored);
                for &child in &entry.children {
                    scene_traverse(self, graph, child);
                }
                self.state.set_view_matrix(saved);
                ChildPolicy::Skip
            }
            NodeData::Volume(volume) => {
                self.effect_commands.volumes.push(crate::render_passes::pass_effects::VolumeDrawCommand {
                    model_matrix: self.state.model_matrix(),
                    dimensions: volume.dimensions,
                    texture_path: volume.texture_path.clone(),
                    density_scale: volume.density_scale,
                    color_map: volume.color_map,
                    is_overlay: self.inside_annotation,
                });
                for &child in &entry.children {
                    scene_traverse(self, graph, child);
                }
                ChildPolicy::Skip
            }
NodeData::PointCloud(point_cloud) => {
                let gpu_emitter = point_cloud
                    .emitter
                    .as_ref()
                    .filter(|e| !e.simulate_on_cpu)
                    .map(crate::render_passes::pass_effects::GpuEmitterParams::from_emitter);
                self.effect_commands.point_clouds.push(crate::render_passes::pass_effects::PointCloudDrawCommand {
                    model_matrix: self.state.model_matrix(),
                    file_path: point_cloud.file_path.clone(),
                    max_visible_points: point_cloud.max_visible_points,
                    point_size: point_cloud.point_size,
                    color: point_cloud.color,
                    is_overlay: self.inside_annotation,
                    points: if gpu_emitter.is_some() {
                        Arc::new(Vec::new())
                    } else {
                        crate::render_passes::pass_effects::pack_point_cloud_gpu(point_cloud)
                    },
                    sim_key: node.data().as_ffi(),
                    emitter: gpu_emitter,
                });
                for &child in &entry.children {
                    scene_traverse(self, graph, child);
                }
                ChildPolicy::Skip
            }
NodeData::Annotation(_) => {
                let was_inside_annotation = self.inside_annotation;
                self.inside_annotation = true;
                for &child in &entry.children { scene_traverse(self, graph, child); }
                self.inside_annotation = was_inside_annotation;
                ChildPolicy::Skip
            }
NodeData::AnnotationSet(ann) => {
                if ann.visible {
                    let set_matrix = self.state.model_matrix();
                    for el in &ann.elements {
                        let (element, el_model) =
                            rc3d_scene::annotation::prepare_annotation_for_render(
                                graph, set_matrix, el,
                            );
                        self.effect_commands.annotation_elements.push(
                            crate::render_passes::pass_effects::ProjectedAnnotation {
                                element,
                                model_matrix: el_model,
                                style: ann.style.clone(),
                                visibility: crate::render_passes::pass_effects::AnnotationVisibility::default(),
                            },
                        );
                    }
                }
                ChildPolicy::Skip
            }
            NodeData::BatchedMesh(batch) => {
                self.emit_batched_mesh(node, batch, is_selected, node_type_label);
                ChildPolicy::Skip
            }
            NodeData::IndexedLineSet(ils) => {
                let coord = self.state.coordinate();
                let mvp = self.state.projection_matrix() * self.state.view_matrix() * self.state.model_matrix();
                let mut edge_positions: Vec<[f32; 3]> = Vec::new();
                let mut aabb = rc3d_core::Aabb::empty();
                for i in (0..ils.coord_index.len()).step_by(2) {
                    if i + 1 < ils.coord_index.len() {
                        let a = ils.coord_index[i].max(0) as usize;
                        let b = ils.coord_index[i + 1].max(0) as usize;
                        if a < coord.points.len() && b < coord.points.len() {
                            let pa = coord.points[a]; let pb = coord.points[b];
                            edge_positions.push([pa.x, pa.y, pa.z]);
                            edge_positions.push([pb.x, pb.y, pb.z]);
                            aabb = aabb.union(&rc3d_core::Aabb::from_point(pa));
                            aabb = aabb.union(&rc3d_core::Aabb::from_point(pb));
                        }
                    }
                }
                if !edge_positions.is_empty() {
                    self.draw_calls.push(DrawCall {
                        vertices: Arc::new(Vec::new()),
                        edge_positions: Arc::new(edge_positions),
                        mvp,
                        model_matrix: self.state.model_matrix(),
                        camera_pos: self.camera_pos,
                        aabb: Some(aabb),
                        overlay_color: Some(ils.color),
                        node_type_label: Arc::from("IndexedLineSet"),
                        is_overlay: self.inside_annotation,
                        ..Default::default()
                    });
                }
                for &child in &entry.children { scene_traverse(self, graph, child); }
                ChildPolicy::Skip
            }
NodeData::MorphTarget(mt) => {
                self.state.set_morph_targets(mt.clone());
                for &child in &entry.children {
                    scene_traverse(self, graph, child);
                }
                ChildPolicy::Skip
            }
NodeData::SkinnedMesh(sm) => {
                self.state.set_skinned_mesh(sm.clone());
                for &child in &entry.children {
                    scene_traverse(self, graph, child);
                }
                self.state.clear_skinned_mesh();
                ChildPolicy::Skip
            }
NodeData::Coordinate3(coord) => {
                self.state.set_coordinate(coord.point.clone());
                ChildPolicy::Skip
            }
NodeData::TextureCoordinate2(tex) => {
                self.state.set_texture_coordinate2(tex.point.clone());
                ChildPolicy::Skip
            }
NodeData::Normal(norm) => {
                self.state.set_normal(norm.vector.clone());
                ChildPolicy::Skip
            }
NodeData::Material(mat) => {
                let el = material_element_for_node(mat, entry.name.as_deref(), self.material_library.as_ref());
                self.state.set_material(el);
                ChildPolicy::Skip
            }
NodeData::PerspectiveCamera(cam) => {
                self.state.set_view_matrix(cam.view_matrix());
                self.state.set_projection_matrix(cam.projection_matrix());
                self.view_matrix = cam.view_matrix();
                self.projection_matrix = cam.projection_matrix();
                self.camera_pos = cam.position;
                self.projection_orthographic = false;
                ChildPolicy::Skip
            }
NodeData::OrthographicCamera(cam) => {
                self.state.set_view_matrix(cam.view_matrix());
                self.state.set_projection_matrix(cam.projection_matrix());
                self.view_matrix = cam.view_matrix();
                self.projection_matrix = cam.projection_matrix();
                self.camera_pos = cam.position;
                self.projection_orthographic = true;
                ChildPolicy::Skip
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
                    ground_color: Vec3::ZERO,
                });
                ChildPolicy::Skip
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
                    ground_color: Vec3::ZERO,
                });
                ChildPolicy::Skip
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
                    ground_color: Vec3::ZERO,
                });
                ChildPolicy::Skip
            }
            NodeData::HemisphereLight(light) => {
                self.state.add_light(LightData {
                    light_type: LightType::Hemisphere,
                    direction: light.direction,
                    location: Vec3::ZERO,
                    color: light.sky_color,
                    intensity: light.intensity,
                    cut_off_angle: 0.0,
                    drop_off_rate: 0.0,
                    ground_color: light.ground_color,
                });
                ChildPolicy::Skip
            }
            NodeData::LightProbe(probe) => {
                self.light_probe_sh = probe.packed_sh_l2();
                self.light_probe_intensity = probe.intensity.max(0.0);
                ChildPolicy::Skip
            }
            NodeData::AreaLight(light) => {
                self.state.add_light(LightData {
                    light_type: LightType::Point,
                    direction: light.direction, location: light.position,
                    color: light.color, intensity: light.intensity,
                    cut_off_angle: 0.0, drop_off_rate: 0.0,
                    ground_color: Vec3::ZERO,
                });
                for &child in &entry.children { scene_traverse(self, graph, child); }
                ChildPolicy::Skip
            }
            NodeData::Sprite(sprite) => {
                let saved_model = self.state.model_matrix();
                let facing = billboard_facing(
                    &rc3d_scene::BillboardNode { axis_aligned: false },
                    self.state.view_matrix(),
                );
                let mut scale = sprite.size.max(0.001);
                if !sprite.size_attenuation {
                    let world = saved_model.transform_point3(Vec3::ZERO);
                    let dist = (world - self.camera_pos).length().max(0.01);
                    scale *= dist * 0.25;
                }
                let cx = (0.5 - sprite.center[0]) * scale;
                let cy = (0.5 - sprite.center[1]) * scale;
                let local = Mat4::from_translation(Vec3::new(cx, cy, 0.0))
                    * Mat4::from_scale(Vec3::splat(scale));
                self.state.set_model_matrix(saved_model * facing * local);
                let saved_mat = self.state.material().clone();
                let mut mat = saved_mat.clone();
                mat.base_color = Vec3::new(sprite.color[0], sprite.color[1], sprite.color[2]);
                mat.diffuse = mat.base_color;
                mat.emissive_color = mat.base_color;
                mat.opacity = sprite.opacity * sprite.color[3];
                mat.alpha_mode = rc3d_scene::AlphaMode::Blend;
                mat.double_sided = true;
                mat.metallic = 0.0;
                mat.roughness = 1.0;
                if !sprite.texture_path.is_empty() {
                    mat.albedo_texture = Some(sprite.texture_path.clone());
                }
                self.state.set_material(mat);
                self.emit_cached_shape(
                    ShapeKey::Quad {
                        w: 1.0f32.to_bits(),
                        h: 1.0f32.to_bits(),
                    },
                    || rc3d_mesh::tessellate_quad_xy(1.0, 1.0),
                    is_selected,
                    node_type_label,
                );
                self.state.set_material(saved_mat);
                self.state.set_model_matrix(saved_model);
                ChildPolicy::Skip
            }
            NodeData::Triangle(_) => {
                let coord = self.state.coordinate();
                if coord.points.len() < 3 {
                    return ChildPolicy::Skip;
                }
                let normals = self.state.normal();
                let face_n = if normals.vectors.len() >= 3 {
                    normals.vectors[..3].to_vec()
                } else {
                    let c = (coord.points[1] - coord.points[0])
                        .cross(coord.points[2] - coord.points[0]);
                    let n = rc3d_core::utils::math::safe_normalize(c, Vec3::Y);
                    vec![n, n, n]
                };
                let positions = vec![coord.points[0], coord.points[1], coord.points[2]];
                let mut mesh = rc3d_mesh::TriangleMesh::from_tris(&positions);
                mesh.compute_tangents();
                let edge_feature = mesh.edge_line_positions_feature(
                    feature_crease_angle(),
                );
                let edge_full = mesh.edge_line_positions();
                let mut vertices = Vec::with_capacity(3);
                for (i, v) in mesh.phong_buffers().0.iter().enumerate() {
                    let n = if i < face_n.len() { face_n[i].to_array() } else { [v[3], v[4], v[5]] };
                    vertices.push(Vertex {
                        position: [v[0], v[1], v[2]],
                        normal: n,
                        texcoord: [v[6], v[7]],
                        tangent: [v[8], v[9], v[10], v[11]],
                    });
                }
                self.emit_draw_call_with_edges(
                    vertices,
                    Some((0..3u32).collect()),
                    edge_feature,
                    edge_full,
                    is_selected,
                    node_type_label,
                );
                ChildPolicy::Skip
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
                    node_type_label,
                );
                self.apply_node_sub_entity(graph, node, |tri| tri / 2);
                ChildPolicy::Skip
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
                    node_type_label,
                );
                ChildPolicy::Skip
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
                    node_type_label,
                );
                ChildPolicy::Skip
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
                    node_type_label,
                );
                ChildPolicy::Skip
            }
NodeData::Torus(torus) => {
                const MAJOR_SEGMENTS: u32 = 32;
                const MINOR_SEGMENTS: u32 = 16;
                self.emit_cached_shape(
                    ShapeKey::Torus {
                        major_r: torus.major_radius.to_bits(),
                        minor_r: torus.minor_radius.to_bits(),
                        major_segments: MAJOR_SEGMENTS,
                        minor_segments: MINOR_SEGMENTS,
                    },
                    || rc3d_mesh::tessellate_torus(torus.major_radius, torus.minor_radius, MAJOR_SEGMENTS, MINOR_SEGMENTS),
                    is_selected,
                    node_type_label,
                );
                ChildPolicy::Skip
            }
NodeData::IndexedFaceSet(ifs) => {
                let coord = self.state.coordinate();
                if coord.points.is_empty() {
                    return ChildPolicy::Skip;
                }
                // Full-content hash: any data change invalidates the shape cache.
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                coord.points.len().hash(&mut hasher);
                for p in &coord.points {
                    p.x.to_bits().hash(&mut hasher);
                    p.y.to_bits().hash(&mut hasher);
                    p.z.to_bits().hash(&mut hasher);
                }
                ifs.coord_index.len().hash(&mut hasher);
                for &idx in &ifs.coord_index {
                    idx.hash(&mut hasher);
                }
                let tex_el = self.state.texture_coordinate2();
                let tex_len = tex_el.coords.len() as u32;
                tex_len.hash(&mut hasher);
                for tc in &tex_el.coords {
                    tc[0].to_bits().hash(&mut hasher);
                    tc[1].to_bits().hash(&mut hasher);
                }
                let norm_el = self.state.normal();
                let normal_len = if norm_el.vectors.len() == coord.points.len() && !norm_el.vectors.is_empty() {
                    norm_el.vectors.len() as u32
                } else {
                    0u32
                };
                normal_len.hash(&mut hasher);
                for nv in &norm_el.vectors {
                    nv.x.to_bits().hash(&mut hasher);
                    nv.y.to_bits().hash(&mut hasher);
                    nv.z.to_bits().hash(&mut hasher);
                }
                let content_hash = hasher.finish();

                let key = ShapeKey::IndexedFaceSet {
                    node: node.data().as_ffi(),
                    coord_len: coord.points.len() as u32,
                    coord_index_len: ifs.coord_index.len() as u32,
                    content_hash,
                    tex_len,
                    normal_len,
                };
                let cached: Option<CachedShapeData> = match self.mesh_cache.entry(key) {
                    Entry::Occupied(occupied) => Some(occupied.get().clone()),
                    Entry::Vacant(vacant) => {
                        let mut mesh = {
                            let use_tex =
                                !tex_el.coords.is_empty() && tex_el.coords.len() == coord.points.len();
                            if use_tex {
                                rc3d_mesh::TriangleMesh::from_indexed_face_set_tex(
                                    &coord.points,
                                    &tex_el.coords,
                                    &ifs.coord_index,
                                )
                            } else {
                                rc3d_mesh::TriangleMesh::from_indexed_face_set(
                                    &coord.points,
                                    &ifs.coord_index,
                                )
                            }
                        };
                        if norm_el.vectors.len() == mesh.positions.len()
                            && !norm_el.vectors.is_empty()
                        {
                            let computed_backup = mesh.normals.clone();
                            mesh.normals.clone_from(&norm_el.vectors);
                            for (i, n) in mesh.normals.iter_mut().enumerate() {
                                let len = n.length();
                                if len > 1e-20 {
                                    *n /= len;
                                } else {
                                    let fb =
                                        computed_backup.get(i).copied().unwrap_or(Vec3::Y);
                                    let l = fb.length();
                                    *n = if l > 1e-20 { fb / l } else { Vec3::Y };
                                }
                            }
                        }
                        mesh.compute_tangents();
                        if mesh.positions.is_empty() {
                            None
                        } else {
                            let (phong_verts, indices) = mesh.phong_buffers();
                            let edge_feature = mesh.edge_line_positions_feature(
                                feature_crease_angle(),
                            );
                            let edge_full = mesh.edge_line_positions();
                            let local_aabb = mesh.bounding_box();
                            let vertices: Vec<Vertex> = phong_verts
                                .iter()
                                .map(|v| Vertex {
                                    position: [v[0], v[1], v[2]],
                                    normal: [v[3], v[4], v[5]],
                                    texcoord: [v[6], v[7]],
                                    tangent: [v[8], v[9], v[10], v[11]],
                                })
                                .collect();
                            let tri_count = indices.len() / 3;
                            let meshlet_data = if tri_count > MESHLET_TRIANGLE_THRESHOLD
                                && self.state.skinned_mesh().is_none()
                            {
                                let md = rc3d_mesh::build_meshlets_from_mesh(
                                    &mesh.positions,
                                    &mesh.normals,
                                    &mesh.texcoords,
                                    &mesh.tangents,
                                    &mesh.tri_indices,
                                );
                                log::info!(
                                    "Meshlet: {} tris -> {} meshlets ({} verts)",
                                    tri_count,
                                    md.total_meshlets,
                                    md.vertices.len(),
                                );
                                Some(Arc::new(md))
                            } else {
                                None
                            };
                            Some(vacant.insert((
                                Arc::new(vertices),
                                Arc::new(indices),
                                Arc::new(edge_feature),
                                Arc::new(edge_full),
                                local_aabb,
                                meshlet_data,
                            )).clone())
                        }
                    }
                };
                if let Some((vertices, indices, edge_feature, edge_full, local_aabb, meshlet_data)) =
                    cached
                {
                    let edge_feature = clamp_edge_positions(edge_feature.clone());
                    let edge_full = clamp_edge_positions(edge_full.clone());
                    self.emit_indexed_face_set_draws(
                        vertices,
                        indices,
                        edge_feature,
                        edge_full,
                        local_aabb,
                        meshlet_data,
                        &ifs.material_groups,
                        &ifs.materials,
                        is_selected,
                        node_type_label,
                    );
                    if ifs.material_groups.is_empty() {
                        self.apply_node_sub_entity(graph, node, |tri| ifs.face_id(tri));
                    }
                }
                ChildPolicy::Skip
            }
            // Structural nodes are owned by scene_traverse; never reached here.
            NodeData::Separator(_)
            | NodeData::Billboard(_)
            | NodeData::InstancedMesh(_)
            | NodeData::TransformManip(_)
            | NodeData::Dragger(_)
            | NodeData::ResetTransform(_)
            | NodeData::ExplodedView(_)
            | NodeData::Switch(_)
            | NodeData::MultipleCopy(_)
            | NodeData::Lod(_)
            | NodeData::HandlerNode(_)
            | NodeData::Transform(_)
            | NodeData::Rotation(_)
            | NodeData::RotationXYZ(_)
            | NodeData::Font(_) => ChildPolicy::Recurse,
        }
    }
}
