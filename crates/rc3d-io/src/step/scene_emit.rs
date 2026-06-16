//! SceneEmitPlan -> SceneGraph adapter (transform instancing, no vertex bake).

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::annotation::AnnotationPoint;
use rc3d_scene::node_data::{
    AnnotationElement, AnnotationSetNode, Coordinate3Node, IndexedFaceSetNode, MaterialNode,
    NormalNode, SeparatorNode, TransformNode,
};
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_shape::{
    face_material_draw_batches, EmitInstance, EmitNode, EmitPlanOptions, LabelId, MaterialDesc,
    PmiDataSet, PmiEntry, SceneEmitPlan,
};

use crate::step::StepError;

#[derive(Debug, Clone)]
#[derive(Default)]
pub struct SceneEmitOptions {
    pub default_material: MaterialDesc,
}


pub fn apply_plan(
    graph: &mut SceneGraph,
    root: NodeId,
    plan: &SceneEmitPlan,
    options: &SceneEmitOptions,
    pmi_pool: &PmiDataSet,
) -> Result<(), StepError> {
    if plan.instances.is_empty() {
        return Err(StepError::NoGeometry);
    }

    let scene_parent = graph.add_child(root, NodeData::Separator(SeparatorNode));
    let mut pmi_labels_attached = Vec::new();

    if plan.hierarchy.is_empty() {
        for instance in &plan.instances {
            emit_instance(graph, scene_parent, plan, instance);
        }
        attach_pmi_for_label(
            graph,
            scene_parent,
            LabelId::default(),
            plan,
            pmi_pool,
            &mut pmi_labels_attached,
        );
    } else {
        emit_hierarchy_roots(
            graph,
            scene_parent,
            plan,
            &plan.hierarchy,
            options,
            pmi_pool,
            &mut pmi_labels_attached,
        );
    }

    attach_orphan_pmi(
        graph,
        scene_parent,
        plan,
        pmi_pool,
        &pmi_labels_attached,
    );

    let _ = options;
    Ok(())
}

fn emit_hierarchy_roots(
    graph: &mut SceneGraph,
    parent: NodeId,
    plan: &SceneEmitPlan,
    nodes: &[EmitNode],
    options: &SceneEmitOptions,
    pmi_pool: &PmiDataSet,
    pmi_labels_attached: &mut Vec<LabelId>,
) {
    for node in nodes {
        if node.instances.is_empty() && !node.children.is_empty() {
            attach_pmi_for_label(
                graph,
                parent,
                node.label_id,
                plan,
                pmi_pool,
                pmi_labels_attached,
            );
            emit_hierarchy(
                graph,
                parent,
                plan,
                &node.children,
                options,
                pmi_pool,
                pmi_labels_attached,
            );
        } else {
            emit_hierarchy(
                graph,
                parent,
                plan,
                std::slice::from_ref(node),
                options,
                pmi_pool,
                pmi_labels_attached,
            );
        }
    }
}

fn emit_hierarchy(
    graph: &mut SceneGraph,
    parent: NodeId,
    plan: &SceneEmitPlan,
    nodes: &[EmitNode],
    options: &SceneEmitOptions,
    pmi_pool: &PmiDataSet,
    pmi_labels_attached: &mut Vec<LabelId>,
) {
    for node in nodes {
        let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));
        for instance in &node.instances {
            emit_instance(graph, sep, plan, instance);
        }
        attach_pmi_for_label(
            graph,
            sep,
            node.label_id,
            plan,
            pmi_pool,
            pmi_labels_attached,
        );
        emit_hierarchy(
            graph,
            sep,
            plan,
            &node.children,
            options,
            pmi_pool,
            pmi_labels_attached,
        );
    }
}

fn emit_instance(
    graph: &mut SceneGraph,
    parent: NodeId,
    plan: &SceneEmitPlan,
    instance: &EmitInstance,
) {
    let Some(cached) = plan.mesh_table.get(&instance.mesh_slot) else {
        return;
    };
    let comp = graph.add_child(parent, NodeData::Separator(SeparatorNode));
    graph.add_child(
        comp,
        NodeData::Transform(transform_node_from_mat4(instance.world_transform)),
    );

    let material = instance.label_material;
    let face_groups = plan.face_materials.get(&instance.mesh_slot);
    let batches = face_material_draw_batches(
        &cached.mesh,
        &cached.face_tri_ranges,
        cached.face_split_viable,
        face_groups.map(|g| g.as_slice()).unwrap_or(&[]),
        material.diffuse,
        material.opacity,
    );

    for batch in batches {
        let batch_sep = graph.add_child(comp, NodeData::Separator(SeparatorNode));
        graph.add_child(
            batch_sep,
            NodeData::Material(material_with_opacity(
                material_node_from_rgb(batch.color),
                batch.opacity,
            )),
        );
        add_mesh_nodes(graph, batch_sep, batch.mesh);
    }
}

fn attach_pmi_for_label(
    graph: &mut SceneGraph,
    parent: NodeId,
    label_id: LabelId,
    plan: &SceneEmitPlan,
    pmi_pool: &PmiDataSet,
    pmi_labels_attached: &mut Vec<LabelId>,
) {
    let elements = pmi_elements_for_label(label_id, plan, pmi_pool);
    if elements.is_empty() {
        return;
    }
    let set = AnnotationSetNode {
        elements,
        visible: true,
        style: pmi_style(),
    };
    graph.add_child(parent, NodeData::AnnotationSet(set));
    pmi_labels_attached.push(label_id);
}

fn attach_orphan_pmi(
    graph: &mut SceneGraph,
    scene_parent: NodeId,
    plan: &SceneEmitPlan,
    pmi_pool: &PmiDataSet,
    pmi_labels_attached: &[LabelId],
) {
    let orphans: Vec<_> = plan
        .pmi_refs
        .iter()
        .filter(|r| !pmi_labels_attached.contains(&r.label_id))
        .collect();
    if orphans.is_empty() {
        return;
    }
    let mut elements = Vec::new();
    for placement in orphans {
        if let Some(entry) = pmi_pool.entries.iter().find(|e| e.id == placement.pmi_id) {
            elements.push(pmi_entry_to_element(entry));
        }
    }
    if elements.is_empty() {
        return;
    }
    let set = AnnotationSetNode {
        elements,
        visible: true,
        style: pmi_style(),
    };
    graph.add_child(scene_parent, NodeData::AnnotationSet(set));
}

fn pmi_elements_for_label(
    label_id: LabelId,
    plan: &SceneEmitPlan,
    pmi_pool: &PmiDataSet,
) -> Vec<AnnotationElement> {
    let mut elements = Vec::new();
    for placement in plan.pmi_refs.iter().filter(|r| r.label_id == label_id) {
        if let Some(entry) = pmi_pool.entries.iter().find(|e| e.id == placement.pmi_id) {
            elements.push(pmi_entry_to_element(entry));
        }
    }
    elements
}

fn pmi_entry_to_element(entry: &PmiEntry) -> AnnotationElement {
    AnnotationElement::Leader {
        anchor: AnnotationPoint::local(entry.origin),
        label_offset: [40.0, -20.0],
        text: entry.label.clone(),
        color: [1.0, 1.0, 0.0, 1.0],
    }
}

fn pmi_style() -> rc3d_scene::annotation::AnnotationStyle {
    rc3d_scene::annotation::AnnotationStyle {
        decimals: 3,
        unit_suffix: " mm".to_string(),
        font_size: 14.0,
        ..Default::default()
    }
}

fn material_node_from_rgb(color: [f32; 3]) -> MaterialNode {
    MaterialNode {
        diffuse_color: Vec3::new(color[0], color[1], color[2]),
        base_color: Vec3::new(color[0], color[1], color[2]),
        roughness: 0.35,
        opacity: 1.0,
        ..Default::default()
    }
}

fn material_with_opacity(mut node: MaterialNode, opacity: f32) -> MaterialNode {
    node.opacity = opacity;
    node
}

fn transform_node_from_mat4(m: Mat4) -> TransformNode {
    TransformNode {
        translation: Vec3::new(m.w_axis.x, m.w_axis.y, m.w_axis.z),
        rotation: m,
        scale: Vec3::ONE,
        center: Vec3::ZERO,
    }
}

fn add_mesh_nodes(graph: &mut SceneGraph, comp: NodeId, mesh: rc3d_shape::MeshResult) {
    let vert_count = mesh.vertices.len();
    let normal_count = mesh.normals.len();
    let tri_count = mesh.indices.len() / 4;
    graph.add_child(
        comp,
        NodeData::Coordinate3(Coordinate3Node {
            point: mesh.vertices,
        }),
    );
    if !mesh.normals.is_empty() {
        graph.add_child(
            comp,
            NodeData::Normal(NormalNode::from_vectors(mesh.normals)),
        );
    }
    graph.add_child(
        comp,
        NodeData::IndexedFaceSet(IndexedFaceSetNode {
            coord_index: mesh.indices,
        }),
    );
    log::debug!(
        "[scene_emit] mesh: {} verts, {} normals, {} tris",
        vert_count,
        normal_count,
        tri_count
    );
}

pub fn emit_plan_options_from_step(options: &crate::step::StepImportOptions) -> EmitPlanOptions {
    // Use TessellationTier-driven policy (OCC alignment).
    // The tier maps through TessellationPolicy → BRepMeshConfig + HealPolicy + FallbackAllowlist.
    let tier = options.tessellation_tier.unwrap_or(rc3d_shape::mesh::TessellationTier::Standard);
    let policy = rc3d_shape::mesh::TessellationPolicy::for_tier(tier);
    let mut mesh_config = policy.to_mesh_config();
    // Override with explicit mesh_relative_deflection if user specified one
    if options.mesh_relative_deflection > 0.0 {
        mesh_config.relative_deflection = options.mesh_relative_deflection;
    }
    EmitPlanOptions {
        mesh_config,
        heal_skip_faces: Vec::new(),
        default_material: MaterialDesc::default(),
        explode_offsets: Default::default(),
    }
}
