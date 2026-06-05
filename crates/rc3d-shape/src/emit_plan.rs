//! Scene emit plan: tessellated meshes + transform instances (no SceneGraph dep).

use std::collections::HashMap;

use rc3d_core::math::{Mat4, Vec3};

use crate::document::ShapeDocument;
use crate::error::ShapeError;
use crate::mesh::{mesh_solid_with_voids, BRepMeshConfig};
use crate::mesh_result::MeshResult;
use crate::mesh_split::FaceTriRange;
use crate::shape::ShapeId;
use crate::tessellation::{TessEntry, TessKey, TessellationCache};
use crate::topo::{FaceKey, Orientation, SolidKey};
use crate::xde::LabelId;

/// Canonical mesh table slot (one per unique TessKey).
pub type MeshSlotId = u32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialDesc {
    pub diffuse: [f32; 3],
    pub opacity: f32,
}

impl Default for MaterialDesc {
    fn default() -> Self {
        Self {
            diffuse: [0.9, 0.9, 0.9],
            opacity: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FaceMaterialGroup {
    pub face_keys: Vec<FaceKey>,
    pub color: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct CachedMesh {
    pub solid_key: SolidKey,
    pub tess_key: TessKey,
    pub mesh: MeshResult,
    pub face_tri_ranges: HashMap<FaceKey, FaceTriRange>,
    pub face_split_viable: bool,
}

#[derive(Debug, Clone)]
pub struct EmitInstance {
    pub mesh_slot: MeshSlotId,
    pub world_transform: Mat4,
    pub label_material: MaterialDesc,
    pub label_id: LabelId,
    pub shape_id: ShapeId,
}

#[derive(Debug, Clone)]
pub struct EmitNode {
    pub label_id: LabelId,
    pub name: Option<String>,
    pub children: Vec<EmitNode>,
    pub instances: Vec<EmitInstance>,
}

#[derive(Debug, Clone)]
pub struct PmiPlacement {
    pub pmi_id: u32,
    pub label_id: LabelId,
    pub world_pose: Option<Mat4>,
}

#[derive(Debug, Default)]
pub struct SceneEmitPlan {
    pub hierarchy: Vec<EmitNode>,
    pub instances: Vec<EmitInstance>,
    pub mesh_table: HashMap<MeshSlotId, CachedMesh>,
    pub face_materials: HashMap<MeshSlotId, Vec<FaceMaterialGroup>>,
    pub pmi_refs: Vec<PmiPlacement>,
}

#[derive(Debug, Clone)]
pub struct EmitPlanOptions {
    pub mesh_config: BRepMeshConfig,
    pub heal_skip_faces: Vec<FaceKey>,
    pub default_material: MaterialDesc,
    /// Per-solid explode translation applied to instance world transforms (not vertex bake).
    pub explode_offsets: HashMap<SolidKey, Vec3>,
}

impl Default for EmitPlanOptions {
    fn default() -> Self {
        Self {
            mesh_config: BRepMeshConfig::default(),
            heal_skip_faces: Vec::new(),
            default_material: MaterialDesc::default(),
            explode_offsets: HashMap::new(),
        }
    }
}

#[cfg(test)]
pub(crate) fn tess_config_hash(config: &BRepMeshConfig) -> u64 {
    config_hash(config)
}

fn config_hash(config: &BRepMeshConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    config.relative_deflection.to_bits().hash(&mut h);
    config.same_parameter_tol.to_bits().hash(&mut h);
    config.edge.deflection.to_bits().hash(&mut h);
    config.edge.angle_deflection.to_bits().hash(&mut h);
    config.face.deflection_interior.to_bits().hash(&mut h);
    config.face.min_size.to_bits().hash(&mut h);
    h.finish()
}

pub fn mesh_solid_local(
    store: &crate::store::BRepStore,
    sk: SolidKey,
    config: &BRepMeshConfig,
    skip_faces: &[FaceKey],
) -> Option<TessEntry> {
    mesh_solid_with_voids(store, sk, config, skip_faces).map(|out| TessEntry {
        mesh: out.mesh,
        face_tri_ranges: out.face_tri_ranges,
        face_split_viable: true,
    })
}

impl ShapeDocument {
    pub fn tessellate_solid(
        &mut self,
        solid_key: SolidKey,
        orientation: Orientation,
        config: &BRepMeshConfig,
        skip_faces: &[FaceKey],
    ) -> Result<TessKey, ShapeError> {
        let key = TessKey {
            solid_key,
            config_hash: config_hash(config),
            orientation,
        };
        if self.tessellation.get(&key).is_some() {
            eprintln!("[emit_plan] tessellate_solid cache HIT for {:?}", solid_key);
            return Ok(key);
        }
        let _t_msh = std::time::Instant::now();
        let mut entry = mesh_solid_local(&self.store, solid_key, config, skip_faces)
            .ok_or_else(|| ShapeError::TessellationFailed(format!("solid {:?}", solid_key)))?;
        eprintln!("[emit_plan] tessellate_solid {:?}: {:.1}s ({} verts, {} faces)",
            solid_key, _t_msh.elapsed().as_secs_f32(),
            entry.mesh.vertices.len(), entry.mesh.indices.len() / 4);
        if orientation == Orientation::Reversed {
            entry.mesh.reverse_winding();
        }
        self.tessellation.insert(key, entry);
        Ok(key)
    }

    pub fn build_emit_plan(
        &mut self,
        options: &EmitPlanOptions,
    ) -> Result<SceneEmitPlan, ShapeError> {
        let _t_plan = std::time::Instant::now();
        let mut plan = SceneEmitPlan::default();
        let mut tess_keys_seen: HashMap<TessKey, MeshSlotId> = HashMap::new();
        let mut slot_counter = 0usize;

        let label_ids: Vec<LabelId> = if self.labels.root_labels.is_empty() {
            self.labels.labels.keys().collect()
        } else {
            self.labels.root_labels.clone()
        };

        if label_ids.is_empty() && !self.roots.is_empty() {
            for &shape_id in &self.roots.clone() {
                let Some(node) = self.shapes.get(shape_id) else {
                    continue;
                };
                let Some(sk) = node.kind.solid_key() else {
                    continue;
                };
                let orientation = node.orientation;
                let tess_key = self.tessellate_solid(
                    sk,
                    orientation,
                    &options.mesh_config,
                    &options.heal_skip_faces,
                )?;
                let slot = ensure_mesh_slot(
                    &mut plan,
                    &mut self.tessellation,
                    &mut tess_keys_seen,
                    &mut slot_counter,
                    sk,
                    tess_key,
                );
                collect_face_materials(&mut plan, &self.store, slot, sk);
                let world = apply_explode_offset(
                    self.world_transform(shape_id),
                    sk,
                    &options.explode_offsets,
                );
                plan.instances.push(EmitInstance {
                    mesh_slot: slot,
                    world_transform: world,
                    label_material: options.default_material,
                    label_id: LabelId::default(),
                    shape_id,
                });
            }
            populate_pmi_refs(self, &mut plan);
            eprintln!("[emit_plan] roots path: {:.1}s  {} instances, {} slots",
                _t_plan.elapsed().as_secs_f32(), plan.instances.len(), plan.mesh_table.len());
            return Ok(plan);
        }

        for &label_id in &label_ids {
            if let Some(node) = build_emit_node_recursive(
                self,
                label_id,
                options,
                &mut plan,
                &mut tess_keys_seen,
                &mut slot_counter,
            )? {
                plan.hierarchy.push(node);
            }
        }

        populate_pmi_refs(self, &mut plan);
        eprintln!("[emit_plan] labels path: {:.1}s  {} instances, {} slots",
            _t_plan.elapsed().as_secs_f32(), plan.instances.len(), plan.mesh_table.len());
        Ok(plan)
    }
}

fn apply_explode_offset(
    mut world: Mat4,
    solid_key: SolidKey,
    offsets: &HashMap<SolidKey, Vec3>,
) -> Mat4 {
    if let Some(offset) = offsets.get(&solid_key) {
        world.w_axis.x += offset.x;
        world.w_axis.y += offset.y;
        world.w_axis.z += offset.z;
    }
    world
}

fn default_pmi_label(doc: &ShapeDocument) -> LabelId {
    doc.labels
        .root_labels
        .first()
        .copied()
        .unwrap_or(LabelId::default())
}

fn nearest_shaped_label(doc: &mut ShapeDocument, origin: [f32; 3]) -> LabelId {
    let target = Vec3::from(origin);
    let shaped: Vec<(LabelId, ShapeId)> = doc
        .labels
        .labels
        .iter()
        .filter_map(|(label_id, label)| label.shape.map(|shape_id| (label_id, shape_id)))
        .collect();
    let mut best: Option<(LabelId, f32)> = None;
    for (label_id, shape_id) in shaped {
        let world = doc.world_transform(shape_id);
        let anchor = Vec3::new(world.w_axis.x, world.w_axis.y, world.w_axis.z);
        let dist = (anchor - target).length_squared();
        if best.is_none_or(|(_, best_dist)| dist < best_dist) {
            best = Some((label_id, dist));
        }
    }
    best.map(|(id, _)| id)
        .unwrap_or_else(|| default_pmi_label(doc))
}

fn populate_pmi_refs(doc: &mut ShapeDocument, plan: &mut SceneEmitPlan) {
    if doc.pmi_pool.entries.is_empty() {
        return;
    }
    let bindings: Vec<(u32, Option<u64>, [f32; 3])> = doc
        .pmi_pool
        .entries
        .iter()
        .map(|entry| (entry.id, entry.step_entity_id, entry.origin))
        .collect();
    for (pmi_id, step_entity_id, origin) in bindings {
        let label_id = step_entity_id
            .and_then(|sid| semantic_label_for_step_entity(doc, sid))
            .unwrap_or_else(|| nearest_shaped_label(doc, origin));
        plan.pmi_refs.push(PmiPlacement {
            pmi_id,
            label_id,
            world_pose: None,
        });
    }
}

fn semantic_label_for_step_entity(doc: &ShapeDocument, step_entity_id: u64) -> Option<LabelId> {
    doc.labels
        .labels
        .iter()
        .find_map(|(label_id, label)| (label.attrs.step_entity_id == Some(step_entity_id)).then_some(label_id))
        .or_else(|| {
            doc.provenance
                .label_to_step
                .iter()
                .find_map(|(label_id, sid)| (*sid == step_entity_id).then_some(*label_id))
        })
}

fn ensure_mesh_slot(
    plan: &mut SceneEmitPlan,
    cache: &TessellationCache,
    tess_keys_seen: &mut HashMap<TessKey, MeshSlotId>,
    slot_counter: &mut usize,
    solid_key: SolidKey,
    tess_key: TessKey,
) -> MeshSlotId {
    if let Some(&slot) = tess_keys_seen.get(&tess_key) {
        return slot;
    }
    let entry = cache
        .get(&tess_key)
        .expect("tessellation cache must contain key");
    let slot = *slot_counter as MeshSlotId;
    *slot_counter += 1;
    tess_keys_seen.insert(tess_key, slot);
    plan.mesh_table.insert(
        slot,
        CachedMesh {
            solid_key,
            tess_key,
            mesh: entry.mesh.clone(),
            face_tri_ranges: entry.face_tri_ranges.clone(),
            face_split_viable: entry.face_split_viable,
        },
    );
    slot
}

fn build_emit_node_recursive(
    doc: &mut ShapeDocument,
    label_id: LabelId,
    options: &EmitPlanOptions,
    plan: &mut SceneEmitPlan,
    tess_keys_seen: &mut HashMap<TessKey, MeshSlotId>,
    slot_counter: &mut usize,
) -> Result<Option<EmitNode>, ShapeError> {
    let (shape_id, children, name) = {
        let Some(label) = doc.labels.labels.get(label_id) else {
            return Ok(None);
        };
        (
            label.shape,
            label.children.clone(),
            label.attrs.name.clone(),
        )
    };

    let mut node = EmitNode {
        label_id,
        name,
        children: Vec::new(),
        instances: Vec::new(),
    };
    let material = label_material_for(doc, label_id, options.default_material);

    if let Some(shape_id) = shape_id {
        if let Some(shape_node) = doc.shapes.get(shape_id) {
            if let Some(sk) = shape_node.kind.solid_key() {
                let orientation = shape_node.orientation;
                let tess_key = doc.tessellate_solid(
                    sk,
                    orientation,
                    &options.mesh_config,
                    &options.heal_skip_faces,
                )?;
                let slot = ensure_mesh_slot(
                    plan,
                    &doc.tessellation,
                    tess_keys_seen,
                    slot_counter,
                    sk,
                    tess_key,
                );
                collect_face_materials(plan, &doc.store, slot, sk);
                let world = apply_explode_offset(
                    doc.world_transform(shape_id),
                    sk,
                    &options.explode_offsets,
                );
                let instance = EmitInstance {
                    mesh_slot: slot,
                    world_transform: world,
                    label_material: material,
                    label_id,
                    shape_id,
                };
                node.instances.push(instance.clone());
                plan.instances.push(instance);
            }
        }
    }

    for child_id in children {
        if let Some(child_node) = build_emit_node_recursive(
            doc,
            child_id,
            options,
            plan,
            tess_keys_seen,
            slot_counter,
        )? {
            node.children.push(child_node);
        }
    }

    if node.instances.is_empty() && node.children.is_empty() {
        Ok(None)
    } else {
        Ok(Some(node))
    }
}

fn label_material_for(doc: &ShapeDocument, label_id: LabelId, default: MaterialDesc) -> MaterialDesc {
    MaterialDesc {
        diffuse: doc
            .resolved_label_color(label_id)
            .unwrap_or(default.diffuse),
        opacity: doc
            .resolved_label_opacity(label_id)
            .unwrap_or(default.opacity),
    }
}

fn collect_face_materials(
    plan: &mut SceneEmitPlan,
    store: &crate::store::BRepStore,
    slot: MeshSlotId,
    solid_key: SolidKey,
) {
    let Some(solid) = store.solids.get(solid_key) else {
        return;
    };
    let Some(shell) = store.shells.get(solid.outer_shell) else {
        return;
    };
    let mut color_to_faces: HashMap<[u32; 3], Vec<FaceKey>> = HashMap::new();
    for &(fk, _) in &shell.faces {
        let Some(face) = store.faces.get(fk) else {
            continue;
        };
        if let Some(color) = face.color {
            let key = [
                color[0].to_bits(),
                color[1].to_bits(),
                color[2].to_bits(),
            ];
            color_to_faces.entry(key).or_default().push(fk);
        }
    }
    if color_to_faces.is_empty() {
        return;
    }
    let groups: Vec<FaceMaterialGroup> = color_to_faces
        .into_iter()
        .map(|(bits, face_keys)| FaceMaterialGroup {
            face_keys,
            color: [
                f32::from_bits(bits[0]),
                f32::from_bits(bits[1]),
                f32::from_bits(bits[2]),
            ],
        })
        .collect();
    plan.face_materials.insert(slot, groups);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::ShapeDocument;
    use crate::topo::{BRepShell, BRepSolid, Orientation};
    use crate::xde::{AttributeBag, XdeLabel};

    #[test]
    fn unique_tess_keys_share_one_mesh_slot() {
        let mut plan = SceneEmitPlan::default();
        let mut cache = TessellationCache::new();
        let mut seen = HashMap::new();
        let mut counter = 0usize;
        let sk = SolidKey::default();
        let tess_key = TessKey {
            solid_key: sk,
            config_hash: 1,
            orientation: Orientation::Forward,
        };
        cache.insert(
            tess_key,
            TessEntry {
                mesh: MeshResult::default(),
                face_tri_ranges: HashMap::new(),
                face_split_viable: true,
            },
        );

        let s1 = ensure_mesh_slot(&mut plan, &cache, &mut seen, &mut counter, sk, tess_key);
        let s2 = ensure_mesh_slot(&mut plan, &cache, &mut seen, &mut counter, sk, tess_key);
        assert_eq!(s1, s2);
        assert_eq!(plan.mesh_table.len(), 1);
        assert_eq!(counter, 1);
    }

    #[test]
    fn tess_key_orientation_split() {
        let mut cache = TessellationCache::new();
        let sk = SolidKey::default();
        let cfg = 42u64;
        let k_fwd = TessKey {
            solid_key: sk,
            config_hash: cfg,
            orientation: Orientation::Forward,
        };
        let k_rev = TessKey {
            solid_key: sk,
            config_hash: cfg,
            orientation: Orientation::Reversed,
        };
        cache.insert(
            k_fwd,
            TessEntry {
                mesh: MeshResult::default(),
                face_tri_ranges: HashMap::new(),
                face_split_viable: true,
            },
        );
        cache.insert(
            k_rev,
            TessEntry {
                mesh: MeshResult::default(),
                face_tri_ranges: HashMap::new(),
                face_split_viable: true,
            },
        );
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn resolved_label_color_in_emit_material() {
        let mut doc = ShapeDocument::new();
        let parent = doc.labels.add_label(XdeLabel {
            parent: None,
            children: vec![],
            attrs: AttributeBag {
                color: Some([1.0, 0.2, 0.3]),
                opacity: Some(0.75),
                ..Default::default()
            },
            shape: None,
        });
        let child = doc.labels.add_label(XdeLabel {
            parent: Some(parent),
            children: vec![],
            attrs: AttributeBag::default(),
            shape: None,
        });
        doc.labels.labels.get_mut(parent).unwrap().children.push(child);

        let material = label_material_for(&doc, child, MaterialDesc::default());
        assert_eq!(material.diffuse, [1.0, 0.2, 0.3]);
        assert!((material.opacity - 0.75).abs() < 1e-6);
    }

    #[test]
    fn three_label_instances_share_one_mesh_slot() {
        let mut doc = ShapeDocument::new();
        let sk = {
            let shell = doc.store.shells.insert(BRepShell {
                faces: vec![],
                closed: true,
                step_id: None,
            });
            doc.store.solids.insert(BRepSolid {
                outer_shell: shell,
                void_shells: vec![],
            })
        };
        let options = EmitPlanOptions::default();
        let tess_key = TessKey {
            solid_key: sk,
            config_hash: tess_config_hash(&options.mesh_config),
            orientation: Orientation::Forward,
        };
        doc.tessellation.insert(
            tess_key,
            TessEntry {
                mesh: MeshResult::default(),
                face_tri_ranges: HashMap::new(),
                face_split_viable: true,
            },
        );

        let mut label_ids = Vec::new();
        for i in 0..3 {
            let shape_id = doc.add_solid_instance(
                sk,
                Mat4::from_translation([i as f32, 0.0, 0.0].into()),
                None,
            );
            let label_id = doc.labels.add_label(XdeLabel {
                parent: None,
                children: vec![],
                attrs: AttributeBag::default(),
                shape: Some(shape_id),
            });
            label_ids.push(label_id);
        }
        doc.labels.root_labels = label_ids;

        let plan = doc
            .build_emit_plan(&options)
            .expect("emit plan");
        assert_eq!(plan.instances.len(), 3, "P0-3: one emit row per label/shape");
        assert_eq!(plan.mesh_table.len(), 1, "one mesh slot per TessKey");
        assert!(
            plan.instances.iter().all(|i| i.mesh_slot == 0),
            "all instances share mesh slot 0"
        );
        assert_eq!(plan.hierarchy.len(), 3, "one hierarchy node per root label");
    }

    #[test]
    fn pmi_refs_bind_to_nearest_shaped_label() {
        let mut doc = ShapeDocument::new();
        let sk = {
            let shell = doc.store.shells.insert(BRepShell {
                faces: vec![],
                closed: true,
                step_id: None,
            });
            doc.store.solids.insert(BRepSolid {
                outer_shell: shell,
                void_shells: vec![],
            })
        };
        let options = EmitPlanOptions::default();
        let tess_key = TessKey {
            solid_key: sk,
            config_hash: tess_config_hash(&options.mesh_config),
            orientation: Orientation::Forward,
        };
        doc.tessellation.insert(
            tess_key,
            TessEntry {
                mesh: MeshResult::default(),
                face_tri_ranges: HashMap::new(),
                face_split_viable: true,
            },
        );

        let left_shape = doc.add_solid_instance(sk, Mat4::IDENTITY, None);
        let right_shape = doc.add_solid_instance(
            sk,
            Mat4::from_translation([100.0, 0.0, 0.0].into()),
            None,
        );
        let left_label = doc.labels.add_label(XdeLabel {
            parent: None,
            children: vec![],
            attrs: AttributeBag {
                name: Some("left".into()),
                ..Default::default()
            },
            shape: Some(left_shape),
        });
        let right_label = doc.labels.add_label(XdeLabel {
            parent: None,
            children: vec![],
            attrs: AttributeBag {
                name: Some("right".into()),
                ..Default::default()
            },
            shape: Some(right_shape),
        });
        doc.labels.root_labels = vec![left_label, right_label];
        doc.pmi_pool.entries = vec![
            crate::document::PmiEntry {
                id: 0,
                label: "near_left".into(),
                step_entity_id: None,
                origin: [2.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
            },
            crate::document::PmiEntry {
                id: 1,
                label: "near_right".into(),
                step_entity_id: None,
                origin: [98.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
            },
        ];

        let plan = doc.build_emit_plan(&options).expect("emit plan");
        assert_eq!(plan.pmi_refs.len(), 2);
        assert_eq!(plan.pmi_refs[0].label_id, left_label);
        assert_eq!(plan.pmi_refs[1].label_id, right_label);
    }
}
