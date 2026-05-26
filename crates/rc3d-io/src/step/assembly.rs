//! Assembly hierarchy reconstruction from STEP entities.

use std::collections::HashMap;
use rc3d_core::math::{Mat4, Vec3};
use super::entity_types::EntityType;
use super::parser::EntityIndex;
use super::value::StepValue;
use super::geom;
use super::topology;

/// RGBA color extracted from STYLED_ITEM.
#[derive(Debug, Clone, Default)]
pub struct StyleInfo {
    pub diffuse: Vec3,   // RGB
    pub opacity: f32,      // 0.0-1.0
}

/// Map from shell entity ID to its style (color/opacity).
pub type ShellStyleMap = HashMap<u64, StyleInfo>;

/// Extract style (color/opacity) from STYLED_ITEM entities.
/// Returns a map from geometry entity ID to StyleInfo.
pub fn extract_shell_styles(entities: &EntityIndex) -> ShellStyleMap {
    let mut styles = HashMap::new();

    for (_, record) in entities.iter() {
        if record.entity_type != EntityType::StyledItem {
            continue;
        }
        // STYLED_ITEM(name, styles, item)
        let styles_list = record.params.nth_param(1)
            .and_then(|v| v.as_list());
        let item_ref = record.params.nth_param(2)
            .and_then(|v| v.as_ref_id());

        let mut style_info = StyleInfo::default();

        // Parse presentation style assignments
        if let Some(list) = styles_list {
            for psa_val in list {
                if let Some(psa_id) = psa_val.as_ref_id() {
                    if let Some(psa_record) = entities.get(&psa_id) {
                        if psa_record.entity_type == EntityType::PresentationStyleAssignment {
                            // PRESENTATION_STYLE_ASSIGNMENT(name, styles)
                            let style_list = psa_record.params.nth_param(1)
                                .and_then(|v| v.as_list());
                            if let Some(slist) = style_list {
                                for style_val in slist {
                                    if let Some(style_id) = style_val.as_ref_id() {
                                        extract_color_from_style(style_id, entities, &mut style_info);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // If we found a color, map it to the item
        if style_info.diffuse.length() > 1e-6 {
            if let Some(item_id) = item_ref {
                // The item may be a shape representation or a face
                // Walk through to find the shell
                let shell_id = resolve_item_to_shell(item_id, entities);
                styles.insert(shell_id, style_info);
            }
        }
    }

    styles
}

/// Try to resolve a styled item to a shell entity ID.
fn resolve_item_to_shell(item_id: u64, entities: &EntityIndex) -> u64 {
    // Direct: item is already a shell
    if let Some(record) = entities.get(&item_id) {
        match record.entity_type {
            EntityType::ClosedShell | EntityType::OpenShell | EntityType::Shell => {
                return item_id;
            }
            EntityType::OrientedClosedShell | EntityType::OrientedOpenShell => {
                if let Some(inner) = record.params.nth_param(3)
                    .and_then(|v| v.as_ref_id())
                    .or_else(|| record.params.nth_param(1).and_then(|v| v.as_ref_id()))
                {
                    return inner;
                }
            }
            _ => {}
        }
    }
    // Indirect: item may be a shape representation containing shells
    for (&eid, record) in entities.iter() {
        if record.entity_type == EntityType::ShapeRepresentation
            || record.entity_type == EntityType::AdvancedBrepShapeRepresentation
        {
            let items = record.params.nth_param(1)
                .and_then(|v| v.as_list());
            if let Some(item_list) = items {
                for item in item_list {
                    if let Some(iid) = item.as_ref_id() {
                        if iid == item_id {
                            // This representation contains our item
                            // Return the first shell found in this representation
                            let shells: Vec<u64> = find_shells_in_representation(eid, entities);
                            if !shells.is_empty() {
                                return shells[0];
                            }
                        }
                    }
                }
            }
        }
    }
    item_id // fallback: return the item itself
}

/// Extract color from a presentation style entity.
fn extract_color_from_style(style_id: u64, entities: &EntityIndex, info: &mut StyleInfo) {
    if let Some(record) = entities.get(&style_id) {
        match record.entity_type {
            EntityType::SurfaceStyleUsage => {
                // SURFACE_STYLE_USAGE(usage, side_style)
                let side_style_id = record.params.nth_param(1)
                    .and_then(|v| v.as_ref_id());
                if let Some(sid) = side_style_id {
                    extract_color_from_style(sid, entities, info);
                }
            }
            EntityType::SurfaceSideStyle => {
                // SURFACE_SIDE_STYLE(name, style)
                let fill_area_style_id = record.params.nth_param(1)
                    .and_then(|v| v.as_ref_id());
                if let Some(fid) = fill_area_style_id {
                    extract_color_from_style(fid, entities, info);
                }
            }
            EntityType::SurfaceStyleFillArea => {
                // SURFACE_STYLE_FILL_AREA(fill_area)
                let fill_id = record.params.nth_param(0)
                    .and_then(|v| v.as_ref_id());
                if let Some(fid) = fill_id {
                    extract_color_from_style(fid, entities, info);
                }
            }
            EntityType::FillAreaStyle => {
                // FILL_AREA_STYLE(name, fill_colour)
                let colour_id = record.params.nth_param(1)
                    .and_then(|v| v.as_ref_id());
                if let Some(cid) = colour_id {
                    extract_color_from_colour(cid, entities, info);
                }
            }
            EntityType::ColourRgb | EntityType::Colour => {
                extract_color_from_colour(style_id, entities, info);
            }
            _ => {}
        }
    }
}

/// Extract RGB color from a COLOUR_RGB entity.
fn extract_color_from_colour(colour_id: u64, entities: &EntityIndex, info: &mut StyleInfo) {
    if let Some(record) = entities.get(&colour_id) {
        let r = record.params.nth_param(1)
            .and_then(|v| v.as_real()).unwrap_or(0.8) as f32;
        let g = record.params.nth_param(2)
            .and_then(|v| v.as_real()).unwrap_or(0.8) as f32;
        let b = record.params.nth_param(3)
            .and_then(|v| v.as_real()).unwrap_or(0.8) as f32;
        info.diffuse = Vec3::new(r, g, b);
    }
}

/// Transform matrix extracted from AXIS2_PLACEMENT_3D.
#[derive(Debug, Clone)]
pub struct AssemblyTransform {
    pub matrix: Mat4,
}

impl Default for AssemblyTransform {
    fn default() -> Self {
        Self { matrix: Mat4::IDENTITY }
    }
}

impl AssemblyTransform {
    pub fn from_placement(origin: Vec3, x_axis: Vec3, z_axis: Vec3) -> Self {
        let z = z_axis.normalize();
        // Gram-Schmidt: project x_axis onto z's perpendicular plane
        let x_raw = x_axis - z * x_axis.dot(z);
        let x = if x_raw.length() > 1e-10 {
            x_raw.normalize()
        } else {
            Vec3::Y.cross(z).normalize()
        };
        let y = z.cross(x).normalize();

        let mat = Mat4::from_cols(
            x.extend(0.0),
            y.extend(0.0),
            z.extend(0.0),
            origin.extend(1.0),
        );
        Self { matrix: mat }
    }

    pub fn transform_point(&self, pt: Vec3) -> Vec3 {
        let v = self.matrix * pt.extend(1.0);
        Vec3::new(v.x, v.y, v.z)
    }

    pub fn compose(&self, other: &AssemblyTransform) -> AssemblyTransform {
        AssemblyTransform { matrix: self.matrix * other.matrix }
    }
}

/// Map from shell entity ID to its accumulated assembly transform.
pub type ShellTransformMap = HashMap<u64, AssemblyTransform>;
pub type ShellInstanceList = Vec<(u64, AssemblyTransform)>;

/// All (shell_id, world transform) pairs — supports repeated assembly instances.
pub fn extract_shell_instances(entities: &EntityIndex) -> ShellInstanceList {
    let mut graph = AssemblyGraph::default();

    // Pass 1: collect data using entity IDs
    // Track SHAPE_REPRESENTATION_RELATIONSHIP for geometry resolution
    let mut shape_repr_links: HashMap<u64, u64> = HashMap::new(); // linked_rep → geometry_rep

    for (&entity_id, record) in entities.iter() {
        // SHAPE_REPRESENTATION_RELATIONSHIP: links axis SR to geometry ABREP
        if record.name == "SHAPE_REPRESENTATION_RELATIONSHIP" {
            let rep1 = geom::nth_ref(&record.params, 2);
            let rep2 = geom::nth_ref(&record.params, 3);
            if let (Some(r1), Some(r2)) = (rep1, rep2) {
                let target = entities.get(&r2).map(|r| r.entity_type).unwrap_or(EntityType::Unknown);
                if target == EntityType::AdvancedBrepShapeRepresentation {
                    shape_repr_links.insert(r1, r2);
                } else {
                    shape_repr_links.insert(r2, r1);
                }
            }
            continue;
        }
        match record.entity_type {
            EntityType::ItemDefinedTransformation => {
                let placement_id = geom::nth_ref(&record.params, 2);
                let pd_id = geom::nth_ref(&record.params, 3);
                if let (Some(pid), Some(pdid)) = (placement_id, pd_id) {
                    if let Some(xform) = resolve_placement_transform(pid, entities) {
                        graph.idt_transforms.insert(pdid, xform);
                    }
                }
            }
            EntityType::NextAssemblyUsageOccurrence => {
                let relating = geom::nth_ref(&record.params, 3)
                    .or_else(|| geom::nth_ref(&record.params, 1));
                let related = geom::nth_ref(&record.params, 4)
                    .or_else(|| geom::nth_ref(&record.params, 2));
                let ap203_xform = geom::nth_ref(&record.params, 4)
                    .and_then(|tid| resolve_placement_transform(tid, entities));
                if let (Some(parent), Some(child)) = (relating, related) {
                    let xform = graph.idt_transforms.get(&child).cloned()
                        .or(ap203_xform)
                        .unwrap_or_default();
                    graph.parent_child.entry(parent).or_default()
                        .push((child, xform));
                }
            }
            EntityType::ProductDefinitionShape => {
                // PDS: AP242 (name,$,#product_def); AP203 (name,#product_def,desc)
                let pd_id = geom::nth_ref(&record.params, 2)
                    .or_else(|| geom::nth_ref(&record.params, 1));
                if let Some(pid) = pd_id {
                    // Map product_def_id → PDS entity_id for SDR lookup
                    graph.prod_to_pds.insert(pid, entity_id);
                }
            }
            EntityType::ShapeDefinitionRepresentation => {
                // SDR: (#pds_entity, #shape_repr) — params[0] references PDS entity
                let pds_entity = geom::nth_ref(&record.params, 0);
                let shape_repr = geom::nth_ref(&record.params, 1);
                if let (Some(pds), Some(sr)) = (pds_entity, shape_repr) {
                    graph.pds_to_shape.insert(pds, sr);
                }
            }
            _ => {}
        }
    }

    // Build reverse index for O(1) parent lookup in accumulate()
    graph.build_reverse_index();

    // Pass 2: build product_def → shape_repr chain, resolve through SRR→ABREP links
    for (&prod_def, &pds_entity) in &graph.prod_to_pds {
        if let Some(&shape_repr) = graph.pds_to_shape.get(&pds_entity) {
            // Follow SRR links to find actual geometry representation
            let geometry_repr = shape_repr_links.get(&shape_repr).copied().unwrap_or(shape_repr);
            graph.shapes.entry(prod_def).or_default().push(geometry_repr);
        }
    }

    let mut instances = ShellInstanceList::new();
    for (&pd_id, shape_ids) in &graph.shapes {
        let xform = graph.accumulate(pd_id);
        for &sid in shape_ids {
            for shell_id in find_shells_in_representation(sid, entities) {
                instances.push((shell_id, xform.clone()));
            }
        }
    }

    instances
}

/// Last occurrence wins per shell id (legacy API).
pub fn extract_shell_transforms(entities: &EntityIndex) -> ShellTransformMap {
    extract_shell_instances(entities)
        .into_iter()
        .map(|(id, xform)| (id, xform))
        .collect()
}

#[derive(Default)]
struct AssemblyGraph {
    idt_transforms: HashMap<u64, AssemblyTransform>,
    parent_child: HashMap<u64, Vec<(u64, AssemblyTransform)>>,
    /// Reverse index: child_id → (parent_id, transform) for O(1) parent lookup
    child_to_parent: HashMap<u64, (u64, AssemblyTransform)>,
    prod_to_pds: HashMap<u64, u64>,
    pds_to_shape: HashMap<u64, u64>,
    shapes: HashMap<u64, Vec<u64>>,
}

impl AssemblyGraph {
    fn accumulate(&self, pd_id: u64) -> AssemblyTransform {
        let mut chain = Vec::new();
        let mut current = pd_id;
        // Use reverse index if available, otherwise fall back to linear scan
        if !self.child_to_parent.is_empty() {
            while let Some((parent, xform)) = self.child_to_parent.get(&current) {
                chain.push(xform.clone());
                current = *parent;
            }
        } else {
            // Fallback linear scan for when build_reverse_index hasn't been called
            loop {
                let mut found = false;
                for (&_parent, children) in &self.parent_child {
                    for (child, xform) in children {
                        if *child == current {
                            chain.push(xform.clone());
                            current = _parent;
                            found = true;
                            break;
                        }
                    }
                    if found { break; }
                }
                if !found { break; }
            }
        }
        let mut result = AssemblyTransform::default();
        for t in chain.iter().rev() {
            result = result.compose(t);
        }
        result
    }

    /// Build the child_to_parent reverse index from parent_child data.
    fn build_reverse_index(&mut self) {
        for (&parent, children) in &self.parent_child {
            for (child, xform) in children {
                self.child_to_parent.insert(*child, (parent, xform.clone()));
            }
        }
    }
}

fn resolve_placement_transform(placement_id: u64, entities: &EntityIndex) -> Option<AssemblyTransform> {
    let record = entities.get(&placement_id)?;
    match record.entity_type {
        EntityType::Axis2Placement3D => {
            let origin_id = geom::nth_ref(&record.params, 1)?;
            let axis_id = geom::nth_ref(&record.params, 2)?;
            let refdir_id = geom::nth_ref(&record.params, 3);
            let origin = topology::resolve_point(origin_id, entities)?;
            let axis = topology::resolve_direction(axis_id, entities)
                .unwrap_or(Vec3::Z);
            let ref_dir = refdir_id
                .and_then(|id| topology::resolve_direction(id, entities))
                .unwrap_or(Vec3::X);
            Some(AssemblyTransform::from_placement(origin, ref_dir, axis))
        }
        _ => None,
    }
}

fn find_shells_in_representation(rep_id: u64, entities: &EntityIndex) -> Vec<u64> {
    let record = match entities.get(&rep_id) {
        Some(r) => r,
        None => return vec![],
    };
    let items = nth_list(record.params.nth_param(1));
    let mut shells = Vec::new();
    for item in &items {
        if let Some(id) = item.as_ref_id() {
            if let Some(r) = entities.get(&id) {
                match r.entity_type {
                    EntityType::ShellBasedSurfaceModel | EntityType::ManifoldSolidBrep => {
                        shells.extend(extract_shells_from_brep(id, entities));
                    }
                    EntityType::ClosedShell | EntityType::OpenShell | EntityType::Shell => {
                        shells.push(id);
                    }
                    EntityType::OrientedClosedShell | EntityType::OrientedOpenShell => {
                        if let Some(inner) = r.params.nth_param(3)
                            .and_then(|v| v.as_ref_id())
                            .or_else(|| r.params.nth_param(1).and_then(|v| v.as_ref_id()))
                        {
                            shells.push(inner);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    shells
}

fn extract_shells_from_brep(brep_id: u64, entities: &EntityIndex) -> Vec<u64> {
    let record = match entities.get(&brep_id) {
        Some(r) => r,
        None => return vec![],
    };
    if record.entity_type == EntityType::ShellBasedSurfaceModel {
        return nth_list(record.params.nth_param(1))
            .iter().filter_map(|v| v.as_ref_id()).collect();
    }
    if record.entity_type == EntityType::ManifoldSolidBrep {
        return geom::nth_ref(&record.params, 1).into_iter().collect();
    }
    vec![]
}

/// Build the full assembly tree (preserves hierarchy, not just flattened transforms).
pub fn build_assembly_tree(entities: &EntityIndex) -> super::tree::AssemblyTree {
    use super::tree::{AssemblyNode, AssemblyTree};
    let mut nodes = Vec::new();
    let mut pd_to_node: HashMap<u64, usize> = HashMap::new();

    // Pass 1: collect PRODUCT entities as tree nodes
    for (&eid, record) in entities.iter() {
        if record.entity_type == EntityType::Product {
            let name = record.params.nth_param(1)
                .and_then(|v| match v {
                    StepValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| format!("#{}", eid));

            let description = record.params.nth_param(2)
                .and_then(|v| match v {
                    StepValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();

            let idx = nodes.len();
            pd_to_node.insert(eid, idx);
            nodes.push(AssemblyNode {
                name,
                description,
                transform: Mat4::IDENTITY,
                children: Vec::new(),
                shells: Vec::new(),
                product_id: eid,
            });
        }
    }

    // Pass 2: build parent-child links from NAUO
    for (_, record) in entities.iter() {
        if record.entity_type == EntityType::NextAssemblyUsageOccurrence {
            let relating = geom::nth_ref(&record.params, 3)
                .or_else(|| geom::nth_ref(&record.params, 1));
            let related = geom::nth_ref(&record.params, 4)
                .or_else(|| geom::nth_ref(&record.params, 2));

            if let (Some(parent), Some(child)) = (relating, related) {
                // Resolve through PRODUCT_DEFINITION -> PRODUCT_DEFINITION_FORMATION -> PRODUCT
                let parent_prod = resolve_pd_to_product(parent, entities);
                let child_prod = resolve_pd_to_product(child, entities);

                if let (Some(pi), Some(ci)) = (
                    parent_prod.and_then(|p| pd_to_node.get(&p)),
                    child_prod.and_then(|c| pd_to_node.get(&c)),
                ) {
                    if !nodes[*pi].children.contains(ci) && *pi != *ci {
                        nodes[*pi].children.push(*ci);
                    }
                }
            }
        }
    }

    // Pass 3: attach shell IDs via the transform map
    let shell_xforms = extract_shell_transforms(entities);
    // Walk each product node and find shells owned through
    // ProductDefinitionShape -> ShapeDefinitionRepresentation -> ShapeRepresentation
    let mut pd_to_pds: HashMap<u64, u64> = HashMap::new();
    let mut pds_to_sr: HashMap<u64, u64> = HashMap::new();

    for (&eid, record) in entities.iter() {
        if record.entity_type == EntityType::ProductDefinitionShape {
            let pd_id = geom::nth_ref(&record.params, 2)
                .or_else(|| geom::nth_ref(&record.params, 1));
            if let Some(pid) = pd_id {
                pd_to_pds.insert(pid, eid);
            }
        }
        if record.entity_type == EntityType::ShapeDefinitionRepresentation {
            let pds_entity = geom::nth_ref(&record.params, 0);
            let shape_repr = geom::nth_ref(&record.params, 1);
            if let (Some(pds), Some(sr)) = (pds_entity, shape_repr) {
                pds_to_sr.insert(pds, sr);
            }
        }
    }

    // Map product_definition -> shape_representation
    let mut sr_to_shells: HashMap<u64, Vec<u64>> = HashMap::new();
    for (_, record) in entities.iter() {
        if record.entity_type == EntityType::ProductDefinition {
            let formation_ref = record.params.nth_param(2)
                .and_then(|v| v.as_ref_id());
            if let Some(fid) = formation_ref {
                if let Some(&pds_id) = pd_to_pds.get(&fid) {
                    if let Some(&sr_id) = pds_to_sr.get(&pds_id) {
                        let shells = find_shells_in_representation(sr_id, entities);
                        if !shells.is_empty() {
                            sr_to_shells.insert(sr_id, shells);
                        }
                    }
                }
            }
        }
    }

    // Attach shells to product nodes
    for (_, record) in entities.iter() {
        if record.entity_type == EntityType::ProductDefinition {
            let formation_ref = record.params.nth_param(2)
                .and_then(|v| v.as_ref_id());
            if let Some(fid) = formation_ref {
                if let (Some(&pds_id), Some(prod_id)) = (
                    pd_to_pds.get(&fid),
                    resolve_pd_to_product_by_formation(fid, entities),
                ) {
                    if let (Some(&sr_id), Some(&node_idx)) = (
                        pds_to_sr.get(&pds_id),
                        pd_to_node.get(&prod_id),
                    ) {
                        if let Some(shells) = sr_to_shells.get(&sr_id) {
                            nodes[node_idx].shells.extend(shells);
                        }
                    }
                }
            }
        }
    }

    // Also attach shells from shell_xforms for any nodes that didn't get shells via the chain
    for (&shell_id, xform) in &shell_xforms {
        if let Some(root) = pd_to_node.values().next() {
            let node = &mut nodes[*root];
            if node.shells.is_empty() {
                node.shells.push(shell_id);
                node.transform = xform.matrix;
            }
        }
    }

    let root_index = nodes.iter().position(|n| !n.children.is_empty() || !n.shells.is_empty())
        .unwrap_or(0);

    AssemblyTree { nodes, root_index }
}

/// Resolve a PRODUCT_DEFINITION ID to its PRODUCT ID.
fn resolve_pd_to_product(pd_id: u64, entities: &EntityIndex) -> Option<u64> {
    for (_, record) in entities.iter() {
        if record.entity_type == EntityType::ProductDefinition {
            // PRODUCT_DEFINITION(id, description, formation_ref)
            let id = geom::nth_ref(&record.params, 0)?;
            if id == pd_id {
                let formation_id = geom::nth_ref(&record.params, 2)?;
                return resolve_pd_to_product_by_formation(formation_id, entities);
            }
        }
    }
    None
}

fn resolve_pd_to_product_by_formation(formation_id: u64, entities: &EntityIndex) -> Option<u64> {
    for (_, record) in entities.iter() {
        if record.entity_type == EntityType::ProductDefinitionFormation {
            let id = geom::nth_ref(&record.params, 0)
                .or_else(|| record.params.nth_param(0).and_then(|v| v.as_ref_id()))?;
            if id == formation_id {
                // PRODUCT_DEFINITION_FORMATION(id, description, product_ref)
                return geom::nth_ref(&record.params, 2)
                    .or_else(|| geom::nth_ref(&record.params, 3));
            }
        }
    }
    None
}

// ── Helpers ─────────────────────────────────────────────────

fn nth_list(val: Option<&StepValue>) -> Vec<StepValue> {
    match val {
        Some(StepValue::List(v)) => v.clone(),
        _ => vec![],
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_xform(tx: f32, ty: f32, tz: f32) -> AssemblyTransform {
        AssemblyTransform {
            matrix: Mat4::from_translation(Vec3::new(tx, ty, tz)),
        }
    }

    #[test]
    fn test_accumulate_transform_root_component() {
        // No parent → identity
        let graph = AssemblyGraph::default();
        let result = graph.accumulate(1);
        assert!((result.matrix - Mat4::IDENTITY).to_scale_rotation_translation().0.length() < 1e-6);
    }

    #[test]
    fn test_accumulate_transform_single_level() {
        // Parent(100) → Child(1) with transform T
        let mut graph = AssemblyGraph::default();
        graph.parent_child.insert(100, vec![(1, make_xform(5.0, 0.0, 0.0))]);

        let result = graph.accumulate(1);
        let (_, _, trans) = result.matrix.to_scale_rotation_translation();
        assert!((trans.x - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_accumulate_transform_two_levels() {
        // Root(200) → Part(100) with T1=(10,0,0)
        // Part(100) → SubPart(1) with T2=(5,0,0)
        // Accumulated for SubPart(1) = T1 * T2 = (15,0,0)
        let mut graph = AssemblyGraph::default();
        graph.parent_child.insert(200, vec![(100, make_xform(10.0, 0.0, 0.0))]);
        graph.parent_child.insert(100, vec![(1, make_xform(5.0, 0.0, 0.0))]);

        let result = graph.accumulate(1);
        let (_, _, trans) = result.matrix.to_scale_rotation_translation();
        assert!((trans.x - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_accumulate_transform_three_levels() {
        // Assembly(300) → SubAssy(200) with T1=(1,0,0)
        // SubAssy(200) → Part(100) with T2=(2,0,0)
        // Part(100) → Detail(1) with T3=(3,0,0)
        // Accumulated = T1 * T2 * T3 = (6,0,0)
        let mut graph = AssemblyGraph::default();
        graph.parent_child.insert(300, vec![(200, make_xform(1.0, 0.0, 0.0))]);
        graph.parent_child.insert(200, vec![(100, make_xform(2.0, 0.0, 0.0))]);
        graph.parent_child.insert(100, vec![(1, make_xform(3.0, 0.0, 0.0))]);

        let result = graph.accumulate(1);
        let (_, _, trans) = result.matrix.to_scale_rotation_translation();
        assert!((trans.x - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_accumulate_single_level_finds_parent() {
        // Equivalent to old find_parent test: parent(100) → child(1)
        let mut graph = AssemblyGraph::default();
        graph.parent_child.insert(100, vec![(1, make_xform(1.0, 0.0, 0.0))]);

        let result = graph.accumulate(1);
        let (_, _, trans) = result.matrix.to_scale_rotation_translation();
        assert!((trans.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_accumulate_no_parent_not_found() {
        // Equivalent to old find_parent not-found test
        let graph = AssemblyGraph::default();
        let result = graph.accumulate(1);
        assert!((result.matrix - Mat4::IDENTITY).to_scale_rotation_translation().0.length() < 1e-6);
    }
}
