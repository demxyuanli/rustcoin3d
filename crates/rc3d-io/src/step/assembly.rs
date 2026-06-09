//! Assembly hierarchy reconstruction from STEP entities.

use std::collections::{HashMap, HashSet};
use rc3d_core::math::{Mat4, Vec3};
use super::entity_types::EntityType;
use super::parser::EntityIndex;
use super::value::StepValue;
use super::entity_geom as geom;
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

/// Observability for assembly DAG shape (multi-parent NAUO links).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AssemblyDiagnostics {
    /// Product-definition nodes referenced as NAUO child from more than one parent.
    pub multi_parent_pd_count: usize,
    /// Parent-child NAUO edges dropped by the legacy single-parent reverse index.
    pub dropped_parent_link_count: usize,
    /// Shell placement rows emitted (after dedup by shell + transform).
    pub shell_instance_count: usize,
}

/// Cached NAUO / PDS / SRR assembly links (build once per import).
pub struct AssemblyContext {
    graph: AssemblyGraph,
}

impl AssemblyContext {
    pub fn build(entities: &EntityIndex) -> Self {
        Self {
            graph: AssemblyGraph::from_entities(entities),
        }
    }

    pub fn shell_instances(&self, entities: &EntityIndex) -> ShellInstanceList {
        let mut instances = ShellInstanceList::new();
        let mut seen: HashSet<(u64, [u32; 16])> = HashSet::new();
        let mut visiting = HashSet::new();
        for root_pd in self.graph.find_pd_roots() {
            self.graph.collect_shell_instances_dfs(
                root_pd,
                AssemblyTransform::default(),
                entities,
                &mut instances,
                &mut seen,
                &mut visiting,
            );
        }
        instances
    }

    pub fn diagnostics(&self) -> AssemblyDiagnostics {
        AssemblyDiagnostics {
            multi_parent_pd_count: self.graph.multi_parent_pd_count(),
            dropped_parent_link_count: self.graph.dropped_parent_link_count(),
            shell_instance_count: 0,
        }
    }

    pub fn assembly_tree(&self, entities: &EntityIndex) -> super::tree::AssemblyTree {
        build_assembly_tree_with_graph(&self.graph, entities)
    }
}

/// All (shell_id, world transform) pairs — supports repeated assembly instances.
pub fn extract_shell_instances(entities: &EntityIndex) -> ShellInstanceList {
    AssemblyContext::build(entities).shell_instances(entities)
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
    /// Axis/placement SR → geometry ABREP (from SHAPE_REPRESENTATION_RELATIONSHIP).
    shape_repr_links: HashMap<u64, u64>,
}

impl AssemblyGraph {
    fn from_entities(entities: &EntityIndex) -> Self {
        let mut graph = Self::default();

        for (&entity_id, record) in entities.iter() {
            if record.name == "SHAPE_REPRESENTATION_RELATIONSHIP" {
                let rep1 = geom::nth_ref(&record.params, 2);
                let rep2 = geom::nth_ref(&record.params, 3);
                if let (Some(r1), Some(r2)) = (rep1, rep2) {
                    let target = entities
                        .get(&r2)
                        .map(|r| r.entity_type)
                        .unwrap_or(EntityType::Unknown);
                    if target == EntityType::AdvancedBrepShapeRepresentation {
                        graph.shape_repr_links.insert(r1, r2);
                    } else {
                        graph.shape_repr_links.insert(r2, r1);
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
                        let xform = graph
                            .idt_transforms
                            .get(&child)
                            .cloned()
                            .or_else(|| find_idt_transform_for_pd(child, entities))
                            .or(ap203_xform)
                            .unwrap_or_default();
                        graph
                            .parent_child
                            .entry(parent)
                            .or_default()
                            .push((child, xform));
                    }
                }
                EntityType::ProductDefinitionShape => {
                    let pd_id = geom::nth_ref(&record.params, 2)
                        .or_else(|| geom::nth_ref(&record.params, 1));
                    if let Some(pid) = pd_id {
                        graph.prod_to_pds.insert(pid, entity_id);
                    }
                }
                EntityType::ShapeDefinitionRepresentation => {
                    let pds_entity = geom::nth_ref(&record.params, 0);
                    let shape_repr = geom::nth_ref(&record.params, 1);
                    if let (Some(pds), Some(sr)) = (pds_entity, shape_repr) {
                        graph.pds_to_shape.insert(pds, sr);
                    }
                }
                _ => {}
            }
        }

        graph.build_reverse_index();

        for (&prod_def, &pds_entity) in &graph.prod_to_pds {
            let Some(&shape_repr) = graph.pds_to_shape.get(&pds_entity) else {
                continue;
            };
            let is_direct_abrep = entities
                .get(&shape_repr)
                .is_some_and(|r| r.entity_type == EntityType::AdvancedBrepShapeRepresentation);
            let geometry_repr = if is_direct_abrep {
                shape_repr
            } else if graph.parent_child.contains_key(&prod_def) {
                // Placement/context reps on assembly containers; geometry comes from child PDs.
                continue;
            } else {
                graph
                    .shape_repr_links
                    .get(&shape_repr)
                    .copied()
                    .unwrap_or(shape_repr)
            };
            if find_shells_in_representation(geometry_repr, entities).is_empty() {
                continue;
            }
            graph
                .shapes
                .entry(prod_def)
                .or_default()
                .push(geometry_repr);
        }

        graph
    }

    fn multi_parent_pd_count(&self) -> usize {
        let mut parent_hits: HashMap<u64, usize> = HashMap::new();
        for children in self.parent_child.values() {
            for (child, _) in children {
                *parent_hits.entry(*child).or_default() += 1;
            }
        }
        parent_hits.values().filter(|&&n| n > 1).count()
    }

    fn dropped_parent_link_count(&self) -> usize {
        let mut edges = 0usize;
        let mut unique_children = HashSet::new();
        for children in self.parent_child.values() {
            for (child, _) in children {
                edges += 1;
                unique_children.insert(*child);
            }
        }
        edges.saturating_sub(unique_children.len())
    }

    fn find_pd_roots(&self) -> Vec<u64> {
        let mut is_child = HashSet::new();
        for children in self.parent_child.values() {
            for (child, _) in children {
                is_child.insert(*child);
            }
        }
        let mut roots = HashSet::new();
        for &parent in self.parent_child.keys() {
            if !is_child.contains(&parent) {
                roots.insert(parent);
            }
        }
        for &pd in self.shapes.keys() {
            if !is_child.contains(&pd) {
                roots.insert(pd);
            }
        }
        roots.into_iter().collect()
    }

    fn collect_shell_instances_dfs(
        &self,
        pd_id: u64,
        world: AssemblyTransform,
        entities: &EntityIndex,
        out: &mut ShellInstanceList,
        seen: &mut HashSet<(u64, [u32; 16])>,
        visiting: &mut HashSet<u64>,
    ) {
        if !visiting.insert(pd_id) {
            log::warn!("[assembly] cycle at product-definition #{pd_id}, skipping branch");
            return;
        }

        if let Some(shape_reprs) = self.shapes.get(&pd_id) {
            for &sid in shape_reprs {
                for shell_id in find_shells_in_representation(sid, entities) {
                    let key = (shell_id, world.matrix.to_cols_array().map(f32::to_bits));
                    if seen.insert(key) {
                        out.push((shell_id, world.clone()));
                    }
                }
            }
        }

        if let Some(children) = self.parent_child.get(&pd_id) {
            for (child_pd, local) in children {
                let child_world = world.compose(local);
                self.collect_shell_instances_dfs(
                    *child_pd,
                    child_world,
                    entities,
                    out,
                    seen,
                    visiting,
                );
            }
        }

        visiting.remove(&pd_id);
    }

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

fn find_idt_transform_for_pd(pd_id: u64, entities: &EntityIndex) -> Option<AssemblyTransform> {
    entities.values().find_map(|record| {
        if record.entity_type != EntityType::ItemDefinedTransformation {
            return None;
        }
        let target_pd = geom::nth_ref(&record.params, 3)?;
        if target_pd != pd_id {
            return None;
        }
        let placement_id = geom::nth_ref(&record.params, 2)?;
        resolve_placement_transform(placement_id, entities)
    })
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

/// Public wrapper for CAF transfer / dedup shell discovery.
pub fn find_shells_in_representation_public(rep_id: u64, entities: &EntityIndex) -> Vec<u64> {
    find_shells_in_representation(rep_id, entities)
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
    AssemblyContext::build(entities).assembly_tree(entities)
}

fn build_assembly_tree_with_graph(
    graph: &AssemblyGraph,
    entities: &EntityIndex,
) -> super::tree::AssemblyTree {
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

    // Pass 2: parent-child from NAUO; local transform = NAUO link (parent PD → child PD).
    for (_, record) in entities.iter() {
        if record.entity_type == EntityType::NextAssemblyUsageOccurrence {
            let relating = geom::nth_ref(&record.params, 3)
                .or_else(|| geom::nth_ref(&record.params, 1));
            let related = geom::nth_ref(&record.params, 4)
                .or_else(|| geom::nth_ref(&record.params, 2));

            if let (Some(parent_pd), Some(child_pd)) = (relating, related) {
                let parent_prod = resolve_pd_to_product(parent_pd, entities);
                let child_prod = resolve_pd_to_product(child_pd, entities);

                if let (Some(pi), Some(ci)) = (
                    parent_prod.and_then(|p| pd_to_node.get(&p)),
                    child_prod.and_then(|c| pd_to_node.get(&c)),
                ) {
                    if !nodes[*pi].children.contains(ci) && *pi != *ci {
                        nodes[*pi].children.push(*ci);
                    }
                    if let Some(local) = graph
                        .parent_child
                        .get(&parent_pd)
                        .and_then(|kids| kids.iter().find(|(c, _)| *c == child_pd))
                        .map(|(_, x)| x.matrix)
                    {
                        nodes[*ci].transform = local;
                    }
                }
            }
        }
    }

    // Pass 3: shells via PDS/SDR/SRR; root-only parts get accumulated placement when still identity.
    for (&pd_id, shape_reprs) in &graph.shapes {
        let record = match entities.get(&pd_id) {
            Some(r) if r.entity_type == EntityType::ProductDefinition => r,
            _ => continue,
        };
        let formation_id = match geom::nth_ref(&record.params, 2) {
            Some(id) => id,
            None => continue,
        };
        let prod_id = match resolve_pd_to_product_by_formation(formation_id, entities) {
            Some(id) => id,
            None => continue,
        };
        let node_idx = match pd_to_node.get(&prod_id) {
            Some(&idx) => idx,
            None => continue,
        };
        for &sr_id in shape_reprs {
            let shells = find_shells_in_representation(sr_id, entities);
            if !shells.is_empty() {
                nodes[node_idx].shells.extend(shells);
            }
        }
        if nodes[node_idx].transform == Mat4::IDENTITY {
            let world = graph.accumulate(pd_id).matrix;
            if world != Mat4::IDENTITY {
                nodes[node_idx].transform = world;
            }
        }
    }

    let root_index = find_assembly_root_index(&nodes);

    AssemblyTree { nodes, root_index }
}

/// Prefer a true assembly root (not referenced as NAUO child); fall back to first node with children/shells.
fn find_assembly_root_index(nodes: &[super::tree::AssemblyNode]) -> usize {
    let mut is_child = vec![false; nodes.len()];
    for node in nodes {
        for &child in &node.children {
            if child < is_child.len() {
                is_child[child] = true;
            }
        }
    }
    let forest_roots: Vec<usize> = is_child
        .iter()
        .enumerate()
        .filter_map(|(idx, &child)| (!child).then_some(idx))
        .collect();
    forest_roots
        .into_iter()
        .max_by_key(|&idx| {
            let n = &nodes[idx];
            (n.children.len(), n.shells.len(), n.name.len())
        })
        .or_else(|| {
            nodes
                .iter()
                .position(|n| !n.children.is_empty() || !n.shells.is_empty())
        })
        .unwrap_or(0)
}

/// Resolve a PRODUCT_DEFINITION ID to its PRODUCT ID.
fn resolve_pd_to_product(pd_id: u64, entities: &EntityIndex) -> Option<u64> {
    let record = entities.get(&pd_id)?;
    if record.entity_type != EntityType::ProductDefinition {
        return None;
    }
    let formation_id = geom::nth_ref(&record.params, 2)?;
    resolve_pd_to_product_by_formation(formation_id, entities)
}

fn resolve_pd_to_product_by_formation(formation_id: u64, entities: &EntityIndex) -> Option<u64> {
    let record = entities.get(&formation_id)?;
    if record.entity_type != EntityType::ProductDefinitionFormation {
        return None;
    }
    geom::nth_ref(&record.params, 2).or_else(|| geom::nth_ref(&record.params, 3))
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

    #[test]
    fn test_multi_parent_pd_detection_and_dropped_links() {
        let mut graph = AssemblyGraph::default();
        graph
            .parent_child
            .insert(100, vec![(1, make_xform(5.0, 0.0, 0.0))]);
        graph
            .parent_child
            .insert(200, vec![(1, make_xform(10.0, 0.0, 0.0))]);
        graph.build_reverse_index();
        assert_eq!(graph.multi_parent_pd_count(), 1);
        assert_eq!(graph.dropped_parent_link_count(), 1);
        let acc = graph.accumulate(1);
        let (_, _, trans) = acc.matrix.to_scale_rotation_translation();
        assert!(
            (trans.x - 10.0).abs() < 1e-6,
            "legacy single-parent chain keeps last NAUO link only, got {}",
            trans.x
        );
    }

    #[test]
    fn test_find_pd_roots_includes_multi_parent_sources() {
        let mut graph = AssemblyGraph::default();
        graph.parent_child.insert(100, vec![(1, make_xform(1.0, 0.0, 0.0))]);
        graph.parent_child.insert(200, vec![(1, make_xform(2.0, 0.0, 0.0))]);
        graph.shapes.insert(1, vec![999]);
        let roots: HashSet<_> = graph.find_pd_roots().into_iter().collect();
        assert!(roots.contains(&100));
        assert!(roots.contains(&200));
        assert!(!roots.contains(&1));
    }

    #[test]
    fn axis2_placement_golden_mat4() {
        // Column-basis placement: columns are X, Y, Z axes; translation in column 3 (w).
        // transform_point(p) = matrix * (p,1) — same as SceneGraph mesh path.
        let text = "ISO-10303-21;\nHEADER;ENDSEC;\nDATA;\n\
#10=CARTESIAN_POINT('',(1.,2.,3.));\n\
#11=DIRECTION('',(0.,0.,1.));\n\
#12=DIRECTION('',(1.,0.,0.));\n\
#1=AXIS2_PLACEMENT_3D('',#10,#11,#12);\n\
ENDSEC;\nEND-ISO-10303-21;\n";
        let ex = super::super::parser::parse_exchange(text).unwrap();
        let xform = resolve_placement_transform(1, &ex.entities).expect("AXIS2_PLACEMENT_3D");
        let world = xform.transform_point(Vec3::ZERO);
        assert!((world.x - 1.0).abs() < 1e-5, "origin.x");
        assert!((world.y - 2.0).abs() < 1e-5, "origin.y");
        assert!((world.z - 3.0).abs() < 1e-5, "origin.z");
        let unit_x = xform.transform_point(Vec3::X) - world;
        assert!((unit_x.x - 1.0).abs() < 1e-5 && unit_x.y.abs() < 1e-5);
    }

    #[test]
    fn item_defined_transformation_compose_order() {
        // accumulate(): chain root→leaf, then compose in rev order → M = T_parent * T_child.
        let mut graph = AssemblyGraph::default();
        graph.parent_child.insert(100, vec![(1, make_xform(10.0, 0.0, 0.0))]);
        graph.parent_child.insert(200, vec![(100, make_xform(1.0, 0.0, 0.0))]);
        graph.build_reverse_index();
        let result = graph.accumulate(1);
        let (_, _, trans) = result.matrix.to_scale_rotation_translation();
        assert!(
            (trans.x - 11.0).abs() < 1e-5,
            "expected parent*child translation (11,0,0), got {:?}",
            trans
        );
    }

    #[test]
    fn cs_step_shell_instances_unique_shells() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/cs.step");
        if !path.exists() {
            return;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let ex = super::super::parser::parse_exchange(&text).unwrap();
        let ctx = super::AssemblyContext::build(&ex.entities);
        let instances = ctx.shell_instances(&ex.entities);
        assert_eq!(
            instances.len(),
            2,
            "Cube + Sphere only; assembly root must not inherit child ABREP"
        );
        let shell_ids: HashSet<u64> = instances.iter().map(|(id, _)| *id).collect();
        assert_eq!(shell_ids.len(), 2);
    }

    #[test]
    fn assembly_tree_flatten_matches_shell_instances() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/cs.step");
        if !path.exists() {
            return;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let ex = super::super::parser::parse_exchange(&text).unwrap();
        let ctx = super::AssemblyContext::build(&ex.entities);
        let instances = ctx.shell_instances(&ex.entities);
        let tree = ctx.assembly_tree(&ex.entities);
        let flat = tree.flatten_shells();
        assert!(!instances.is_empty() && !flat.is_empty());
        for (shell_id, world) in flat {
            let inst = instances
                .iter()
                .find(|(id, _)| *id == shell_id)
                .map(|(_, x)| x.matrix)
                .expect("shell in flatten must exist in instances");
            let delta = (world.w_axis - inst.w_axis).truncate();
            assert!(
                delta.length() < 1e-3,
                "shell #{shell_id} world mismatch: tree {:?} vs instance {:?}",
                world.w_axis,
                inst.w_axis
            );
        }
    }

    #[test]
    fn cs_step_assembly_nodes_carry_shells() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/cs.step");
        if !path.exists() {
            return;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let ex = super::super::parser::parse_exchange(&text).unwrap();
        let tree = super::build_assembly_tree(&ex.entities);
        let with_shells: usize = tree.nodes.iter().filter(|n| !n.shells.is_empty()).count();
        assert!(
            with_shells >= 2,
            "Cube+Sphere should attach shells to product nodes, got {with_shells}: {:?}",
            tree.nodes
                .iter()
                .map(|n| (&n.name, n.shells.len()))
                .collect::<Vec<_>>()
        );
    }
}
