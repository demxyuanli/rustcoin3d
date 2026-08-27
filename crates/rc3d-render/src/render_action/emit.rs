impl RenderCollector {
    fn emit_indexed_face_set_draws(
        &mut self,
        vertices: Arc<Vec<Vertex>>,
        indices: Arc<Vec<u32>>,
        edge_feature: Arc<Vec<[f32; 3]>>,
        edge_full: Arc<Vec<[f32; 3]>>,
        local_aabb: rc3d_core::Aabb,
        meshlet_data: Option<Arc<rc3d_mesh::MeshletData>>,
        groups: &[rc3d_scene::FaceMaterialGroup],
        palette: &[rc3d_scene::MaterialNode],
        selected: bool,
        node_type_label: &str,
    ) {
        if groups.is_empty() {
            self.emit_draw_call_with_cached_aabb(
                vertices,
                Some(indices),
                edge_feature,
                edge_full,
                local_aabb,
                meshlet_data,
                selected,
                node_type_label,
            );
            return;
        }

        let saved = self.state.material().clone();
        let empty_edges: Arc<Vec<[f32; 3]>> = Arc::new(Vec::new());
        // Share the full index Arc (same GPU mesh) and slice with index_first/count,
        // matching three.js BufferGeometry.groups. Slicing into new Vecs uploaded as
        // separate meshes and caused coplanar z-fighting / shadow acne flicker.
        let index_len = indices.len() as u32;
        for (i, group) in groups.iter().enumerate() {
            if group.count == 0 {
                continue;
            }
            if group.start >= index_len {
                continue;
            }
            let count = group.count.min(index_len - group.start);
            if count == 0 {
                continue;
            }
            if let Some(mat) = palette.get(group.material_index as usize) {
                self.state
                    .set_material(material_element_for_node(mat, None, self.material_library.as_ref()));
            } else {
                self.state.set_material(saved.clone());
            }
            let (feat, full, meshlets) = if i == 0 {
                (edge_feature.clone(), edge_full.clone(), None)
            } else {
                (empty_edges.clone(), empty_edges.clone(), None)
            };
            self.emit_draw_call_with_cached_aabb(
                vertices.clone(),
                Some(indices.clone()),
                feat,
                full,
                local_aabb.clone(),
                meshlets,
                selected,
                node_type_label,
            );
            if let Some(dc) = self.draw_calls.last_mut() {
                dc.index_first = group.start;
                dc.index_draw_count = count;
            }
        }
        self.state.set_material(saved);
    }

    fn emit_cached_shape<F>(
        &mut self,
        key: ShapeKey,
        build_mesh: F,
        selected: bool,
        node_type_label: &str,
    ) where
        F: FnOnce() -> rc3d_mesh::TriangleMesh,
    {
        match self.mesh_cache.entry(key) {
            Entry::Occupied(_) => {}
            Entry::Vacant(vacant) => {
                let mut mesh = build_mesh();
                if mesh.positions.is_empty() {
                    vacant.insert((
                        Arc::new(Vec::new()),
                        Arc::new(Vec::new()),
                        Arc::new(Vec::new()),
                        Arc::new(Vec::new()),
                        rc3d_core::Aabb::empty(),
                        None,
                    ));
                    return;
                }
                mesh.compute_tangents();
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
                vacant.insert((
                    Arc::new(vertices),
                    Arc::new(indices),
                    Arc::new(edge_feature),
                    Arc::new(edge_full),
                    local_aabb,
                    None,
                ));
            }
        }

        if let Some((vertices, indices, edge_feature, edge_full, local_aabb, meshlet_data)) = self.mesh_cache.get(&key) {
            let edge_feature = clamp_edge_positions(Arc::clone(edge_feature));
            let edge_full = clamp_edge_positions(Arc::clone(edge_full));
            self.emit_draw_call_with_cached_aabb(
                Arc::clone(vertices),
                Some(Arc::clone(indices)),
                edge_feature,
                edge_full,
                local_aabb.clone(),
                meshlet_data.clone(),
                selected,
                node_type_label,
            );
        }
    }

    fn local_view_dir(&self) -> Vec3 {
        let inv = self.state.model_matrix().inverse();
        let cam_local = inv.transform_point3(self.camera_pos);
        rc3d_core::utils::math::safe_normalize(cam_local, Vec3::Z)
    }

    fn overlay_edge_positions(
        &self,
        feature: &Arc<Vec<[f32; 3]>>,
        vertices: &Arc<Vec<Vertex>>,
        indices: Option<&Arc<Vec<u32>>>,
    ) -> Arc<Vec<[f32; 3]>> {
        let style = self.state.appearance().edges;
        if !matches!(
            style,
            EdgeStyle::Silhouette | EdgeStyle::Perimeter | EdgeStyle::Hard | EdgeStyle::Adjacent
        ) {
            return Arc::clone(feature);
        }
        let positions: Vec<Vec3> = vertices.iter().map(|v| Vec3::from_array(v.position)).collect();
        // Phong buffers split corners; weld by position so shared edges are visible.
        let soup = match indices {
            Some(idx) if idx.len() >= 3 => {
                let mut tris = Vec::with_capacity(idx.len());
                for tri in idx.chunks_exact(3) {
                    tris.push(positions[tri[0] as usize]);
                    tris.push(positions[tri[1] as usize]);
                    tris.push(positions[tri[2] as usize]);
                }
                tris
            }
            _ => positions,
        };
        let mesh = rc3d_mesh::TriangleMesh::from_tris(&soup);
        let crease = feature_crease_angle();
        let lines = match style {
            EdgeStyle::Silhouette => mesh.edge_line_positions_silhouette(self.local_view_dir()),
            EdgeStyle::Perimeter => mesh.edge_line_positions_perimeter(),
            EdgeStyle::Hard => mesh.edge_line_positions_hard(crease),
            EdgeStyle::Adjacent => mesh.edge_line_positions_adjacent(crease),
            _ => return Arc::clone(feature),
        };
        Arc::new(lines)
    }

    fn emit_draw_call_with_cached_aabb(
        &mut self,
        vertices: Arc<Vec<Vertex>>,
        indices: Option<Arc<Vec<u32>>>,
        edge_positions: Arc<Vec<[f32; 3]>>,
        wireframe_edge_positions: Arc<Vec<[f32; 3]>>,
        local_aabb: rc3d_core::Aabb,
        meshlet_data: Option<Arc<rc3d_mesh::MeshletData>>,
        selected: bool,
        node_type_label: &str,
    ) {
        let edge_positions =
            self.overlay_edge_positions(&edge_positions, &vertices, indices.as_ref());
        let model = self.state.model_matrix();
        let mvp = self.state.projection_matrix() * self.state.view_matrix() * model;
        let mat = self.state.material();
        let aabb = if local_aabb.min.x <= local_aabb.max.x {
            Some(local_aabb.transform(model))
        } else {
            None
        };
        let packed = collect_lights(self.state.lights());
        let light_key = {
            let (ref light_dirs, ref light_colors, ref light_types, ref light_positions, ref spot_params, light_count) = packed;
            hash_light_params(light_dirs, light_colors, light_types, light_positions, spot_params, light_count)
        };

        self.draw_calls.push(DrawCall {
            vertices,
            is_overlay: self.inside_annotation,
            indices,
            edge_positions,
            wireframe_edge_positions,
            mvp,
            model_matrix: model,
            camera_pos: self.camera_pos,
            light_set_id: self.light_sets.intern(light_key, packed),
            light_key,
            diffuse_color: mat.diffuse,
            ambient_color: mat.ambient,
            specular_color: mat.specular,
            shininess: mat.shininess,
            base_color: mat.base_color,
            metallic: mat.metallic,
            roughness: mat.roughness,
            anisotropic: mat.anisotropic,
            opacity: mat.opacity,
            albedo_path: mat
                .albedo_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            normal_path: mat
                .normal_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            emissive_color: mat.emissive_color,
            emissive_path: mat
                .emissive_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            metallic_roughness_path: mat
                .metallic_roughness_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            occlusion_path: mat
                .occlusion_texture
                .as_ref()
                .map(|s| Arc::from(s.as_str())),
            alpha_mode: mat.alpha_mode,
            alpha_cutoff: mat.alpha_cutoff,
            double_sided: mat.double_sided,
            clearcoat_factor: mat.clearcoat_factor,
            clearcoat_roughness: mat.clearcoat_roughness,
            specular_factor: mat.specular_factor,
            specular_color_factor: mat.specular_color_factor,
            transmission_factor: mat.transmission_factor,
            ior: mat.ior,
            sheen_color: mat.sheen_color,
            sheen_roughness: mat.sheen_roughness,
            iridescence_factor: mat.iridescence_factor,
            iridescence_ior: mat.iridescence_ior,
            iridescence_thickness_min: mat.iridescence_thickness_min,
            iridescence_thickness_max: mat.iridescence_thickness_max,
            toon_steps: mat.toon_steps,
            visualize_normals: mat.visualize_normals,
            visualize_depth: mat.visualize_depth,
            custom_wgsl: mat.custom_wgsl.as_ref().map(|s| Arc::from(s.as_str())),
            custom_uniforms: mat.custom_uniforms,
            aabb,
            display_mode: self.state.appearance().to_display_mode(),
            fill_style: self.state.appearance().fill,
            edge_style: self.state.appearance().edges,
            selected,
            overlay_color: None,
            mesh_hash: None,
            meshlet_data,
            projection_orthographic: self.projection_orthographic,
            depth_reversed_z: rc3d_core::depth_reversed_z_from_projection(
                self.state.projection_matrix(),
            ),
            node_type_label: Arc::from(node_type_label),
            instance_transforms: self.pending_instance_transforms.take(),
            morph_weights: self.state.morph_targets().map(|mt| mt.weights.clone()).unwrap_or_default(),
            morph_target_deltas: self.state.morph_targets().map(|mt| Arc::new(mt.clone())),
            skinning: self.skinning_payload_for_draw(),
            index_first: 0,
            index_draw_count: 0,
        });
        if !self.cache_ptr.is_null() {
            self.emit_to_cache(self.draw_calls.last().unwrap());
        }
    }

    fn emit_batched_mesh(
        &mut self,
        node: NodeId,
        batch: &rc3d_scene::node_data::BatchedMeshNode,
        selected: bool,
        node_type_label: &str,
    ) {
        if batch.positions.is_empty() || batch.indices.is_empty() || batch.instances.is_empty() {
            return;
        }
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        batch.positions.len().hash(&mut hasher);
        batch.indices.len().hash(&mut hasher);
        for p in &batch.positions {
            p[0].to_bits().hash(&mut hasher);
            p[1].to_bits().hash(&mut hasher);
            p[2].to_bits().hash(&mut hasher);
        }
        for &i in &batch.indices {
            i.hash(&mut hasher);
        }
        let key = ShapeKey::BatchedMesh {
            node: node.data().as_ffi(),
            vert_len: batch.positions.len() as u32,
            index_len: batch.indices.len() as u32,
            content_hash: hasher.finish(),
        };
        match self.mesh_cache.entry(key) {
            Entry::Occupied(_) => {}
            Entry::Vacant(vacant) => {
                let n = batch.positions.len();
                let mut vertices = Vec::with_capacity(n);
                let mut local_aabb = rc3d_core::Aabb::empty();
                for i in 0..n {
                    let p = batch.positions[i];
                    local_aabb = local_aabb.union(&rc3d_core::Aabb::from_point(Vec3::from_array(p)));
                    let nrm = batch.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
                    let uv = batch.texcoords.get(i).copied().unwrap_or([0.0, 0.0]);
                    let tan = batch.tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]);
                    vertices.push(Vertex {
                        position: p,
                        normal: nrm,
                        texcoord: uv,
                        tangent: tan,
                    });
                }
                vacant.insert((
                    Arc::new(vertices),
                    Arc::new(batch.indices.clone()),
                    Arc::new(Vec::new()),
                    Arc::new(Vec::new()),
                    local_aabb,
                    None,
                ));
            }
        }
        let Some((vertices, indices, edge_feature, edge_full, _mesh_aabb, _)) =
            self.mesh_cache.get(&key).cloned()
        else {
            return;
        };
        let parent = self.state.model_matrix();
        let view = self.state.view_matrix();
        let proj = self.state.projection_matrix();
        let mat = self.state.material();
        let packed = collect_lights(self.state.lights());
        let light_key = {
            let (ref light_dirs, ref light_colors, ref light_types, ref light_positions, ref spot_params, light_count) = packed;
            hash_light_params(light_dirs, light_colors, light_types, light_positions, spot_params, light_count)
        };
        let light_set_id = self.light_sets.intern(light_key, packed);
        let empty_edges = Arc::clone(&edge_feature);
        let empty_wire = Arc::clone(&edge_full);
        let depth_rev = rc3d_core::depth_reversed_z_from_projection(proj);
        let appearance = self.state.appearance();

        for inst in &batch.instances {
            if !inst.visible {
                continue;
            }
            let Some(geo) = batch.geometries.get(inst.geometry as usize) else {
                continue;
            };
            if geo.index_count < 3 {
                continue;
            }
            let local = Mat4::from_cols_array_2d(&inst.transform);
            let model = parent * local;
            let mvp = proj * view * model;
            let mut inst_aabb = rc3d_core::Aabb::empty();
            let start = geo.index_first as usize;
            let end = (start + geo.index_count as usize).min(batch.indices.len());
            for &idx in &batch.indices[start..end] {
                if let Some(p) = batch.positions.get(idx as usize) {
                    inst_aabb = inst_aabb.union(&rc3d_core::Aabb::from_point(Vec3::from_array(*p)));
                }
            }
            let aabb = if inst_aabb.min.x <= inst_aabb.max.x {
                Some(inst_aabb.transform(model))
            } else {
                None
            };
            let tint = Vec3::new(inst.color[0], inst.color[1], inst.color[2]);
            let opacity = mat.opacity * inst.color[3];
            self.draw_calls.push(DrawCall {
                vertices: Arc::clone(&vertices),
                indices: Some(Arc::clone(&indices)),
                edge_positions: Arc::clone(&empty_edges),
                wireframe_edge_positions: Arc::clone(&empty_wire),
                mvp,
                model_matrix: model,
                camera_pos: self.camera_pos,
                light_set_id,
                light_key,
                diffuse_color: mat.diffuse * tint,
                ambient_color: mat.ambient,
                specular_color: mat.specular,
                shininess: mat.shininess,
                base_color: mat.base_color * tint,
                metallic: mat.metallic,
                roughness: mat.roughness,
                anisotropic: mat.anisotropic,
                opacity,
                albedo_path: mat.albedo_texture.as_ref().map(|s| Arc::from(s.as_str())),
                normal_path: mat.normal_texture.as_ref().map(|s| Arc::from(s.as_str())),
                emissive_color: mat.emissive_color,
                emissive_path: mat.emissive_texture.as_ref().map(|s| Arc::from(s.as_str())),
                metallic_roughness_path: mat
                    .metallic_roughness_texture
                    .as_ref()
                    .map(|s| Arc::from(s.as_str())),
                occlusion_path: mat.occlusion_texture.as_ref().map(|s| Arc::from(s.as_str())),
                alpha_mode: mat.alpha_mode,
                alpha_cutoff: mat.alpha_cutoff,
                double_sided: mat.double_sided,
                clearcoat_factor: mat.clearcoat_factor,
                clearcoat_roughness: mat.clearcoat_roughness,
                specular_factor: mat.specular_factor,
                specular_color_factor: mat.specular_color_factor,
                transmission_factor: mat.transmission_factor,
                ior: mat.ior,
                sheen_color: mat.sheen_color,
                sheen_roughness: mat.sheen_roughness,
                iridescence_factor: mat.iridescence_factor,
                iridescence_ior: mat.iridescence_ior,
                iridescence_thickness_min: mat.iridescence_thickness_min,
                iridescence_thickness_max: mat.iridescence_thickness_max,
                toon_steps: mat.toon_steps,
                visualize_normals: mat.visualize_normals,
                visualize_depth: mat.visualize_depth,
                custom_wgsl: mat.custom_wgsl.as_ref().map(|s| Arc::from(s.as_str())),
                custom_uniforms: mat.custom_uniforms,
                aabb,
                display_mode: appearance.to_display_mode(),
                fill_style: appearance.fill,
                edge_style: appearance.edges,
                selected,
                overlay_color: None,
                mesh_hash: None,
                meshlet_data: None,
                projection_orthographic: self.projection_orthographic,
                depth_reversed_z: depth_rev,
                is_overlay: self.inside_annotation,
                node_type_label: Arc::from(node_type_label),
                instance_transforms: None,
                morph_weights: Vec::new(),
                morph_target_deltas: None,
                skinning: None,
                index_first: geo.index_first,
                index_draw_count: geo.index_count,
            });
            if !self.cache_ptr.is_null() {
                self.emit_to_cache(self.draw_calls.last().unwrap());
            }
        }
    }

    fn emit_draw_call_with_edges(
        &mut self,
        vertices: Vec<Vertex>,
        indices: Option<Vec<u32>>,
        edge_feature: Vec<[f32; 3]>,
        edge_wireframe: Vec<[f32; 3]>,
        selected: bool,
        node_type_label: &str,
    ) {
        let local_aabb = if vertices.is_empty() {
            rc3d_core::Aabb::empty()
        } else {
            let first = Vec3::from_array(vertices[0].position);
            let mut aabb = rc3d_core::Aabb::from_point(first);
            for v in &vertices[1..] {
                aabb = aabb.union(&rc3d_core::Aabb::from_point(Vec3::from_array(v.position)));
            }
            aabb
        };

        self.emit_draw_call_with_cached_aabb(
            Arc::new(vertices),
            indices.map(Arc::new),
            Arc::new(edge_feature),
            Arc::new(edge_wireframe),
            local_aabb,
            None,
            selected,
            node_type_label,
        );
    }

    fn skinning_payload_for_draw(&self) -> Option<Arc<SkinnedMeshDrawPayload>> {
        self.state.skinned_mesh().map(|sm| {
            Arc::new(SkinnedMeshDrawPayload {
                skeleton: sm.skeleton.clone(),
                skin_data: Arc::new(sm.skin_data.clone()),
                clip: sm.clip.clone(),
            })
        })
    }
}
