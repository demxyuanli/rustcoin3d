//! STEP CAF transfer: EntityIndex -> ShapeDocument (OCC STEPCAF-style).

use rc3d_core::math::Real;
use std::collections::{HashMap, HashSet};

use rc3d_core::math::PMat4;
use rc3d_shape::{
    AttributeBag, LabelId, PmiDataSet, PmiEntry, ShapeDocument, ShapeError, ShapeKind, XdeLabel,
};

use crate::step::assembly::{self, extract_shell_styles, AssemblyContext, ShellStyleMap, StyleInfo};
use crate::step::brep::{build_brep_with_options, BRepBuildOptions, BRepBuildReport};
use crate::step::parser::EntityIndex;
use crate::step::tree::AssemblyTree;
use crate::step::StepError;
use rc3d_shape::topo::SolidKey;

/// Dedup key source for shared topology instancing (P0-1 / P0-2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DedupKey {
    Cdsr(u64),
    MappedItem(u64),
    Shell(u64),
}

impl DedupKey {
    fn step_entity_id(self) -> u64 {
        match self {
            Self::Cdsr(id) | Self::MappedItem(id) | Self::Shell(id) => id,
        }
    }
}

#[derive(Debug, Default)]
struct DedupIndex {
    shell_to_key: HashMap<u64, DedupKey>,
    key_to_solid: HashMap<DedupKey, SolidKey>,
}

impl DedupIndex {
    fn from_entities(entities: &EntityIndex) -> Self {
        let mut index = Self::default();
        for (&eid, record) in entities.iter() {
            match record.name.as_str() {
                "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION" => {
                    for shape_repr in cdsr_shape_representations(record, entities) {
                        for shell_id in shells_in_shape_representation(shape_repr, entities) {
                            index
                                .shell_to_key
                                .insert(shell_id, DedupKey::Cdsr(shape_repr));
                        }
                    }
                }
                "MAPPED_ITEM" => {
                    let rep_map = record.params.nth_param(1).and_then(|v| v.as_ref_id());
                    if let Some(map_id) = rep_map {
                        if let Some(mapped_repr) = mapped_representation_from_map(map_id, entities)
                        {
                            for shell_id in shells_in_shape_representation(mapped_repr, entities)
                            {
                                index
                                    .shell_to_key
                                    .insert(shell_id, DedupKey::MappedItem(map_id));
                            }
                        }
                    }
                }
                _ => {}
            }
            let _ = eid;
        }
        index
    }

    fn resolve_solid(
        &mut self,
        shell_id: u64,
        shell_to_solid: &HashMap<u64, SolidKey>,
    ) -> Result<Option<SolidKey>, ShapeError> {
        let Some(sk) = shell_to_solid.get(&shell_id).copied() else {
            return Ok(None);
        };
        let key = self
            .shell_to_key
            .get(&shell_id)
            .copied()
            .unwrap_or(DedupKey::Shell(shell_id));
        if let Some(&existing) = self.key_to_solid.get(&key) {
            if existing != sk {
                return Err(ShapeError::DedupConflict {
                    entity_id: key.step_entity_id(),
                });
            }
            return Ok(Some(existing));
        }
        self.key_to_solid.insert(key, sk);
        Ok(Some(sk))
    }
}

pub struct CafTransferOutput {
    pub document: ShapeDocument,
    pub build_report: BRepBuildReport,
    pub root_solids: Vec<SolidKey>,
}

pub struct StepCafTransfer;

impl StepCafTransfer {
    pub fn transfer(
        entities: &EntityIndex,
        build_options: &BRepBuildOptions,
    ) -> Result<CafTransferOutput, StepError> {
        let brep = build_brep_with_options(entities, build_options)?;
        let build_report = brep.build_report.clone();
        let root_solids = brep.root_solids.clone();
        let mut doc = ShapeDocument::new();
        doc.store = brep.registry;
        let shell_to_solid = map_all_shell_step_ids(&doc.store);
        let mut dedup = DedupIndex::from_entities(entities);
        let assembly_ctx = AssemblyContext::build(entities);
        let assembly_tree = assembly_ctx.assembly_tree(entities);
        let shell_instances = assembly_ctx.shell_instances(entities);
        let shell_styles = extract_shell_styles(entities);
        // Inherit styles from parent products to children where missing
        let shell_styles = assembly_tree.resolve_inherited_styles(&shell_styles);

        build_xde_labels(
            &mut doc,
            &assembly_tree,
            &shell_to_solid,
            &shell_styles,
            &shell_instances,
            &mut dedup,
        )
        .map_err(|e| match e {
            ShapeError::DedupConflict { entity_id } => StepError::ImportQuality(format!(
                "CAF dedup conflict at STEP entity #{entity_id}"
            )),
            other => StepError::ImportQuality(format!("CAF transfer error: {other}")),
        })?;

        if doc.shapes.is_empty() {
            install_root_solids(&mut doc, &root_solids);
        }

        doc.pmi_pool = extract_pmi_pool(entities);
        bind_provenance(&mut doc, entities);
        Ok(CafTransferOutput {
            document: doc,
            build_report,
            root_solids,
        })
    }

    /// Count distinct `SolidKey` values after CDSR / MAPPED_ITEM dedup for the given shell step ids.
    pub fn count_deduped_solids(
        entities: &EntityIndex,
        shell_to_solid: &HashMap<u64, SolidKey>,
        shell_ids: &[u64],
    ) -> Result<usize, ShapeError> {
        let mut dedup = DedupIndex::from_entities(entities);
        let mut resolved = HashSet::new();
        for &shell_id in shell_ids {
            if let Some(sk) = dedup.resolve_solid(shell_id, shell_to_solid)? {
                resolved.insert(sk);
            }
        }
        Ok(resolved.len())
    }
}

fn install_root_solids(doc: &mut ShapeDocument, root_solids: &[SolidKey]) {
    for &sk in root_solids {
        doc.add_solid_instance(sk, PMat4::IDENTITY, None);
    }
}

fn map_all_shell_step_ids(store: &rc3d_shape::BRepStore) -> HashMap<u64, SolidKey> {
    let mut map = HashMap::new();
    for (sk, solid) in store.solids.iter() {
        let Some(shell) = store.shells.get(solid.outer_shell) else {
            continue;
        };
        if let Some(step_id) = shell.step_id {
            map.entry(step_id).or_insert(sk);
        }
    }
    map
}

fn build_xde_labels(
    doc: &mut ShapeDocument,
    tree: &AssemblyTree,
    shell_to_solid: &HashMap<u64, SolidKey>,
    shell_styles: &ShellStyleMap,
    shell_instances: &assembly::ShellInstanceList,
    dedup: &mut DedupIndex,
) -> Result<(), ShapeError> {
    if tree.nodes.is_empty() {
        return Ok(());
    }

    let mut node_labels: Vec<Option<LabelId>> = vec![None; tree.nodes.len()];

    for (idx, node) in tree.nodes.iter().enumerate() {
        let label_id = doc.labels.add_label(XdeLabel {
            parent: None,
            children: vec![],
            attrs: AttributeBag {
                name: Some(node.name.clone()),
                description: Some(node.description.clone()),
                step_entity_id: Some(node.product_id),
                ..Default::default()
            },
            shape: None,
        });
        node_labels[idx] = Some(label_id);
    }

    for (idx, node) in tree.nodes.iter().enumerate() {
        let Some(label_id) = node_labels[idx] else {
            continue;
        };
        for &child_idx in &node.children {
            let Some(child_label) = node_labels[child_idx] else {
                continue;
            };
            let needs_mirror = doc
                .labels
                .labels
                .get(child_label)
                .and_then(|c| c.parent)
                .is_some_and(|existing| existing != label_id);
            let link_label = if needs_mirror {
                mirror_label_for_parent(doc, child_label, label_id)
            } else {
                if let Some(child) = doc.labels.labels.get_mut(child_label) {
                    child.parent = Some(label_id);
                }
                child_label
            };
            if let Some(parent_label) = doc.labels.labels.get_mut(label_id) {
                if !parent_label.children.contains(&link_label) {
                    parent_label.children.push(link_label);
                }
            }
        }
    }

    let mut is_tree_child = vec![false; tree.nodes.len()];
    for node in &tree.nodes {
        for &child_idx in &node.children {
            if child_idx < is_tree_child.len() {
                is_tree_child[child_idx] = true;
            }
        }
    }
    for (idx, &is_child) in is_tree_child.iter().enumerate() {
        if is_child {
            continue;
        }
        if let Some(label_id) = node_labels[idx] {
            doc.labels.root_labels.push(label_id);
        }
    }
    if doc.labels.root_labels.is_empty() {
        if let Some(label_id) = node_labels.get(tree.root_index).copied().flatten() {
            doc.labels.root_labels.push(label_id);
        }
    }

    let mut shell_to_node: HashMap<u64, usize> = HashMap::new();
    for (idx, node) in tree.nodes.iter().enumerate() {
        for &shell_step_id in &node.shells {
            shell_to_node.insert(shell_step_id, idx);
        }
    }

    let placements: Vec<(u64, PMat4)> = if !shell_instances.is_empty() {
        shell_instances
            .iter()
            .map(|(shell_id, xform)| (*shell_id, xform.matrix))
            .collect()
    } else {
        tree.flatten_shells()
    };

    let mut seen_placements: HashSet<(SolidKey, [u64; 16])> = HashSet::new();

    for (shell_step_id, world) in placements {
        let Some(sk) = dedup.resolve_solid(shell_step_id, shell_to_solid)? else {
            continue;
        };
        let placement_key = (sk, world.to_cols_array().map(Real::to_bits));
        if !seen_placements.insert(placement_key) {
            continue;
        }
        let shape_id = doc.add_solid_instance(sk, world, None);
        if let Some(style) = shell_styles.get(&shell_step_id) {
            apply_style_to_shell_faces(&mut doc.store, sk, style);
        }

        if let Some(&node_idx) = shell_to_node.get(&shell_step_id) {
            let Some(label_id) = node_labels[node_idx] else {
                continue;
            };
            detach_shape_from_roots(doc, shape_id);
            attach_shape_to_label(
                doc,
                label_id,
                shape_id,
                &tree.nodes[node_idx].name,
                shell_styles.get(&shell_step_id),
            );
        } else if let Some(label_id) = node_labels.get(tree.root_index).copied().flatten() {
            detach_shape_from_roots(doc, shape_id);
            attach_shape_to_label(
                doc,
                label_id,
                shape_id,
                "instance",
                shell_styles.get(&shell_step_id),
            );
        }
    }

    Ok(())
}

fn mirror_label_for_parent(
    doc: &mut ShapeDocument,
    src_label: LabelId,
    parent_label: LabelId,
) -> LabelId {
    let Some(src) = doc.labels.labels.get(src_label).cloned() else {
        return src_label;
    };
    let mirror = doc.labels.add_label(XdeLabel {
        parent: Some(parent_label),
        children: vec![],
        attrs: src.attrs,
        shape: None,
    });
    log::debug!(
        "[CAF] multi-parent label mirror: {:?} under {:?} (source {:?})",
        mirror,
        parent_label,
        src_label,
    );
    mirror
}

fn attach_shape_to_label(
    doc: &mut ShapeDocument,
    label_id: LabelId,
    shape_id: rc3d_shape::ShapeId,
    instance_name: &str,
    style: Option<&StyleInfo>,
) {
    if doc.labels.labels.get(label_id).is_some_and(|l| l.shape.is_none()) {
        if let Some(label) = doc.labels.labels.get_mut(label_id) {
            label.shape = Some(shape_id);
            if let Some(style) = style {
                label.attrs.color = Some(style_rgb(style));
                label.attrs.opacity = Some(style.opacity);
            }
        }
    } else {
        let instance_label = doc.labels.add_label(XdeLabel {
            parent: Some(label_id),
            children: vec![],
            attrs: AttributeBag {
                name: Some(instance_name.to_string()),
                color: style.map(style_rgb),
                opacity: style.map(|s| s.opacity),
                ..Default::default()
            },
            shape: Some(shape_id),
        });
        if let Some(parent) = doc.labels.labels.get_mut(label_id) {
            parent.children.push(instance_label);
        }
    }
}

fn detach_shape_from_roots(doc: &mut ShapeDocument, shape_id: rc3d_shape::ShapeId) {
    if let Some(i) = doc.roots.iter().position(|&id| id == shape_id) {
        doc.roots.remove(i);
    }
}

fn style_rgb(style: &StyleInfo) -> [Real; 3] {
    [style.diffuse.x, style.diffuse.y, style.diffuse.z]
}

fn apply_style_to_shell_faces(
    store: &mut rc3d_shape::BRepStore,
    sk: SolidKey,
    style: &StyleInfo,
) {
    let color = style_rgb(style);
    let Some(solid) = store.solids.get(sk) else {
        return;
    };
    let Some(shell) = store.shells.get(solid.outer_shell) else {
        return;
    };
    let face_keys: Vec<_> = shell.faces.iter().map(|(fk, _)| *fk).collect();
    for fk in face_keys {
        if let Some(face) = store.faces.get_mut(fk) {
            if face.color.is_none() {
                face.color = Some(color);
            }
        }
    }
}

fn shells_in_shape_representation(rep_id: u64, entities: &EntityIndex) -> Vec<u64> {
    assembly::find_shells_in_representation_public(rep_id, entities)
}

fn cdsr_shape_representations(record: &crate::step::parser::EntityRecord, entities: &EntityIndex) -> Vec<u64> {
    let mut out = Vec::new();
    let Some(params) = record.params.as_list() else {
        return out;
    };
    for param in params {
        let Some(id) = param.as_ref_id() else {
            continue;
        };
        if let Some(sr) = resolve_shape_representation_id(id, entities) {
            if !out.contains(&sr) {
                out.push(sr);
            }
        }
    }
    out
}

fn resolve_shape_representation_id(id: u64, entities: &EntityIndex) -> Option<u64> {
    let record = entities.get(&id)?;
    match record.name.as_str() {
        "ADVANCED_BREP_SHAPE_REPRESENTATION"
        | "SHAPE_REPRESENTATION"
        | "MANIFOLD_SURFACE_SHAPE_REPRESENTATION" => Some(id),
        name if name.contains("REPRESENTATION_RELATIONSHIP") => {
            let mut fallback = None;
            for idx in 2..=4 {
                let Some(rid) = record.params.nth_param(idx).and_then(|v| v.as_ref_id()) else {
                    continue;
                };
                let Some(sr) = resolve_shape_representation_id(rid, entities) else {
                    continue;
                };
                if entities
                    .get(&sr)
                    .is_some_and(|r| r.name == "ADVANCED_BREP_SHAPE_REPRESENTATION")
                {
                    return Some(sr);
                }
                fallback = fallback.or(Some(sr));
            }
            fallback
        }
        _ => None,
    }
}

fn mapped_representation_from_map(map_id: u64, entities: &EntityIndex) -> Option<u64> {
    let record = entities.get(&map_id)?;
    if record.name != "REPRESENTATION_MAP" {
        return None;
    }
    record.params.nth_param(1).and_then(|v| v.as_ref_id())
}

fn extract_pmi_pool(entities: &EntityIndex) -> PmiDataSet {
    let pmi = crate::step::pmi::pmi_extract::extract_pmi(entities);
    let mut pool = PmiDataSet::default();
    let mut id = 0u32;
    for dim in &pmi.dimensions {
        let mid = [
            (dim.start.x + dim.end.x) * 0.5,
            (dim.start.y + dim.end.y) * 0.5,
            (dim.start.z + dim.end.z) * 0.5,
        ];
        pool.entries.push(PmiEntry {
            id,
            label: dim.text.clone(),
            step_entity_id: Some(dim.entity_id),
            origin: mid,
            normal: [dim.offset_dir.x, dim.offset_dir.y, dim.offset_dir.z],
        });
        id += 1;
    }
    for datum in &pmi.datums {
        pool.entries.push(PmiEntry {
            id,
            label: datum.label.clone(),
            step_entity_id: Some(datum.entity_id),
            origin: [datum.origin.x, datum.origin.y, datum.origin.z],
            normal: [datum.normal.x, datum.normal.y, datum.normal.z],
        });
        id += 1;
    }
    pool
}

fn bind_provenance(doc: &mut ShapeDocument, entities: &EntityIndex) {
    for (label_id, label) in doc.labels.labels.iter() {
        if let Some(step_id) = label.attrs.step_entity_id {
            doc.provenance.label_to_step.insert(label_id, step_id);
        }
    }
    for (shape_id, node) in doc.shapes.iter() {
        if let ShapeKind::Solid(sk) = node.kind {
            if let Some(solid) = doc.store.solids.get(sk) {
                if let Some(shell) = doc.store.shells.get(solid.outer_shell) {
                    if let Some(step_id) = shell.step_id {
                        doc.provenance.shape_to_step.insert(shape_id, step_id);
                    }
                }
            }
        }
    }
    let _ = entities;
}

#[cfg(test)]
mod dedup_tests {
    use super::*;
    use crate::step::parser::parse_exchange;
    use rc3d_shape::topo::SolidKey;

    fn count_entity(exchange: &crate::step::parser::Exchange, name: &str) -> usize {
        exchange
            .entities
            .values()
            .filter(|r| r.name == name)
            .count()
    }

    #[test]
    fn mapped_item_shell_maps_to_representation_map_key() {
        let input = "\
ISO-10303-21;
HEADER; ENDSEC;
DATA;
#1 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT(()) REPRESENTATION_CONTEXT('', '') );
#10 = ADVANCED_BREP_SHAPE_REPRESENTATION('', (#11), #1);
#11 = MANIFOLD_SOLID_BREP('', #12);
#12 = CLOSED_SHELL('', ());
#20 = REPRESENTATION_MAP('', #10);
#30 = MAPPED_ITEM('', #20);
ENDSEC;
END-ISO-10303-21;
";
        let exchange = parse_exchange(input).expect("parse mapped item snippet");
        let index = DedupIndex::from_entities(&exchange.entities);
        let shells = assembly::find_shells_in_representation_public(10, &exchange.entities);
        assert!(!shells.is_empty(), "shape rep should expose shell step id");
        for shell_id in shells {
            assert_eq!(
                index.shell_to_key.get(&shell_id),
                Some(&DedupKey::MappedItem(20)),
                "shell #{shell_id} should dedup via MAPPED_ITEM map #20"
            );
        }
    }

    #[test]
    fn cdsr_shell_maps_to_shape_representation_key() {
        let input = "\
ISO-10303-21;
HEADER; ENDSEC;
DATA;
#1 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT(()) REPRESENTATION_CONTEXT('', '') );
#10 = ADVANCED_BREP_SHAPE_REPRESENTATION('', (#11), #1);
#11 = MANIFOLD_SOLID_BREP('', #12);
#12 = CLOSED_SHELL('', ());
#20 = ( REPRESENTATION_RELATIONSHIP('','',#30,#10) SHAPE_REPRESENTATION_RELATIONSHIP() );
#30 = SHAPE_REPRESENTATION('', (#31), #1);
#31 = AXIS2_PLACEMENT_3D('', #32, #33, #34);
#32 = CARTESIAN_POINT('', (0.,0.,0.));
#33 = DIRECTION('', (0.,0.,1.));
#34 = DIRECTION('', (1.,0.,0.));
#40 = PRODUCT_DEFINITION_SHAPE('', '', #41);
#41 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('', '', '', #42, #43, $);
#42 = PRODUCT_DEFINITION('', '', #43, #44);
#43 = PRODUCT('', '', '', (#45));
#44 = PRODUCT_DEFINITION_CONTEXT('', #46, '');
#45 = PRODUCT_CONTEXT('', #46, '');
#46 = APPLICATION_CONTEXT('');
#50 = CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#20, #40);
ENDSEC;
END-ISO-10303-21;
";
        let exchange = parse_exchange(input).expect("parse cdsr snippet");
        assert!(count_entity(&exchange, "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION") >= 1);
        let index = DedupIndex::from_entities(&exchange.entities);
        let shells = assembly::find_shells_in_representation_public(10, &exchange.entities);
        assert!(!shells.is_empty());
        for shell_id in shells {
            assert_eq!(
                index.shell_to_key.get(&shell_id),
                Some(&DedupKey::Cdsr(10)),
                "shell #{shell_id} should dedup via CDSR shape rep #10"
            );
        }
    }

    #[test]
    fn cs_step_shell_instances_dedup_same_solid_and_transform() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/cs.step");
        if !path.exists() {
            return;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let exchange = parse_exchange(&text).unwrap();
        let ctx = assembly::AssemblyContext::build(&exchange.entities);
        assert!(
            ctx.shell_instances(&exchange.entities).len() >= 2,
            "fixture should list Cube + Sphere shell instances"
        );
        let transfer = StepCafTransfer::transfer(
            &exchange.entities,
            &BRepBuildOptions::from_import(&crate::step::StepImportOptions::default()),
        )
        .unwrap();
        let mut solid_keys = HashSet::new();
        for (_shape_id, node) in transfer.document.shapes.iter() {
            if let ShapeKind::Solid(sk) = node.kind {
                solid_keys.insert(sk);
            }
        }
        assert_eq!(
            transfer.document.shapes.len(),
            solid_keys.len(),
            "shape nodes should match unique solids (no duplicate placements)"
        );
        assert_eq!(
            transfer.document.shapes.len(),
            2,
            "Cube + Sphere should yield exactly two shape nodes"
        );
    }

    #[test]
    fn assembly_example_dedup_index_maps_cdsr_shells() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/AssemblyExample-Assembly.step");
        if !path.exists() {
            return;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let exchange = parse_exchange(&text).unwrap();
        let cdsr_count = exchange
            .entities
            .values()
            .filter(|r| r.name == "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION")
            .count();
        let index = DedupIndex::from_entities(&exchange.entities);
        println!(
            "AssemblyExample dedup index: cdsr={cdsr_count} shells={} unique_keys={}",
            index.shell_to_key.len(),
            HashSet::<&DedupKey>::from_iter(index.shell_to_key.values()).len()
        );
        assert!(cdsr_count >= 1, "fixture should contain CDSR entities");
        assert!(
            index.shell_to_key.len() >= 2,
            "CDSR index should map multiple assembly shells"
        );
    }

    #[test]
    fn cdsr_resolve_reuses_canonical_solid_for_shared_key() {
        let input = "\
ISO-10303-21;
HEADER; ENDSEC;
DATA;
#1 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT(()) REPRESENTATION_CONTEXT('', '') );
#10 = ADVANCED_BREP_SHAPE_REPRESENTATION('', (#11, #14), #1);
#11 = MANIFOLD_SOLID_BREP('', #12);
#12 = CLOSED_SHELL('', ());
#14 = MANIFOLD_SOLID_BREP('', #15);
#15 = CLOSED_SHELL('', ());
#20 = ( REPRESENTATION_RELATIONSHIP('','',#30,#10) SHAPE_REPRESENTATION_RELATIONSHIP() );
#30 = SHAPE_REPRESENTATION('', (#31), #1);
#31 = AXIS2_PLACEMENT_3D('', #32, #33, #34);
#32 = CARTESIAN_POINT('', (0.,0.,0.));
#33 = DIRECTION('', (0.,0.,1.));
#34 = DIRECTION('', (1.,0.,0.));
#40 = PRODUCT_DEFINITION_SHAPE('', '', #41);
#41 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('', '', '', #42, #43, $);
#42 = PRODUCT_DEFINITION('', '', #43, #44);
#43 = PRODUCT('', '', '', (#45));
#44 = PRODUCT_DEFINITION_CONTEXT('', #46, '');
#45 = PRODUCT_CONTEXT('', #46, '');
#46 = APPLICATION_CONTEXT('');
#50 = CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#20, #40);
ENDSEC;
END-ISO-10303-21;
";
        let exchange = parse_exchange(input).expect("parse");
        let sk_a = SolidKey::default();
        let sk_b = SolidKey::default();
        let shell_to_solid = HashMap::from([(12u64, sk_a), (15u64, sk_b)]);
        let unique = StepCafTransfer::count_deduped_solids(
            &exchange.entities,
            &shell_to_solid,
            &[12, 15],
        )
        .expect("resolve");
        assert_eq!(
            unique, 1,
            "shared CDSR shape rep should resolve to one canonical SolidKey"
        );
    }

    #[test]
    fn cdsr_resolve_errors_on_distinct_solids_for_shared_key() {
        let input = "\
ISO-10303-21;
HEADER; ENDSEC;
DATA;
#1 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT(()) REPRESENTATION_CONTEXT('', '') );
#10 = ADVANCED_BREP_SHAPE_REPRESENTATION('', (#11, #14), #1);
#11 = MANIFOLD_SOLID_BREP('', #12);
#12 = CLOSED_SHELL('', ());
#14 = MANIFOLD_SOLID_BREP('', #15);
#15 = CLOSED_SHELL('', ());
#20 = ( REPRESENTATION_RELATIONSHIP('','',#30,#10) SHAPE_REPRESENTATION_RELATIONSHIP() );
#30 = SHAPE_REPRESENTATION('', (#31), #1);
#31 = AXIS2_PLACEMENT_3D('', #32, #33, #34);
#32 = CARTESIAN_POINT('', (0.,0.,0.));
#33 = DIRECTION('', (0.,0.,1.));
#34 = DIRECTION('', (1.,0.,0.));
#40 = PRODUCT_DEFINITION_SHAPE('', '', #41);
#41 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('', '', '', #42, #43, $);
#42 = PRODUCT_DEFINITION('', '', #43, #44);
#43 = PRODUCT('', '', '', (#45));
#44 = PRODUCT_DEFINITION_CONTEXT('', #46, '');
#45 = PRODUCT_CONTEXT('', #46, '');
#46 = APPLICATION_CONTEXT('');
#50 = CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#20, #40);
ENDSEC;
END-ISO-10303-21;
";
        let exchange = parse_exchange(input).expect("parse");
        let mut store = rc3d_shape::BRepStore::new();
        let shell_a = store.shells.insert(rc3d_shape::topo::BRepShell {
            faces: vec![],
            closed: true,
            step_id: Some(12),
        });
        let shell_b = store.shells.insert(rc3d_shape::topo::BRepShell {
            faces: vec![],
            closed: true,
            step_id: Some(15),
        });
        let sk_a = store.solids.insert(rc3d_shape::topo::BRepSolid {
            outer_shell: shell_a,
            void_shells: vec![],
        });
        let sk_b = store.solids.insert(rc3d_shape::topo::BRepSolid {
            outer_shell: shell_b,
            void_shells: vec![],
        });
        let shell_to_solid = HashMap::from([(12u64, sk_a), (15u64, sk_b)]);
        let err = StepCafTransfer::count_deduped_solids(
            &exchange.entities,
            &shell_to_solid,
            &[12, 15],
        )
        .unwrap_err();
        assert!(
            matches!(err, ShapeError::DedupConflict { entity_id: 10 }),
            "expected CDSR shape rep #10 conflict, got {err:?}"
        );
    }

    #[test]
    fn mapped_item_resolve_errors_on_distinct_solids_for_shared_key() {
        let input = "\
ISO-10303-21;
HEADER; ENDSEC;
DATA;
#1 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT(()) REPRESENTATION_CONTEXT('', '') );
#10 = ADVANCED_BREP_SHAPE_REPRESENTATION('', (#11, #14), #1);
#11 = MANIFOLD_SOLID_BREP('', #12);
#12 = CLOSED_SHELL('', ());
#14 = MANIFOLD_SOLID_BREP('', #15);
#15 = CLOSED_SHELL('', ());
#20 = REPRESENTATION_MAP('', #10);
#30 = MAPPED_ITEM('', #20);
#40 = MAPPED_ITEM('', #20);
ENDSEC;
END-ISO-10303-21;
";
        let exchange = parse_exchange(input).expect("parse");
        let mut store = rc3d_shape::BRepStore::new();
        let shell_a = store.shells.insert(rc3d_shape::topo::BRepShell {
            faces: vec![],
            closed: true,
            step_id: Some(12),
        });
        let shell_b = store.shells.insert(rc3d_shape::topo::BRepShell {
            faces: vec![],
            closed: true,
            step_id: Some(15),
        });
        let sk_a = store.solids.insert(rc3d_shape::topo::BRepSolid {
            outer_shell: shell_a,
            void_shells: vec![],
        });
        let sk_b = store.solids.insert(rc3d_shape::topo::BRepSolid {
            outer_shell: shell_b,
            void_shells: vec![],
        });
        let shell_to_solid = HashMap::from([(12u64, sk_a), (15u64, sk_b)]);
        let err = StepCafTransfer::count_deduped_solids(
            &exchange.entities,
            &shell_to_solid,
            &[12, 15],
        )
        .unwrap_err();
        assert!(
            matches!(err, ShapeError::DedupConflict { entity_id: 20 }),
            "expected REPRESENTATION_MAP #20 conflict, got {err:?}"
        );
    }
}
