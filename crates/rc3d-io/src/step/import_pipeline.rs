//! STEP import orchestration helpers (heal, continuity checks, scene mesh emit).

use std::collections::{HashMap, HashSet};

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_scene::node_data::{
    Coordinate3Node, IndexedFaceSetNode, MaterialNode, NormalNode, SeparatorNode, TransformNode,
};

use crate::step::assembly::{self, AssemblyTransform, ShellStyleMap, StyleInfo};
use crate::step::brep::heal::{auto_heal_shell, check_shell_continuity, HealLevel, HealReport};
use crate::step::brep::mesh::BRepMeshConfig;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::SolidKey;
use crate::step::brep;
use crate::step::import_options::{StepImportMode, StepImportReport};
use crate::step::mesh_result::MeshResult;
use crate::step::tree::AssemblyTree;
use crate::step::StepError;

/// Run auto-heal on all root solids and aggregate reports.
pub fn run_heal_pipeline(
    reg: &mut BRepRegistry,
    root_solids: &[SolidKey],
    heal_level: HealLevel,
    max_iterations: usize,
) -> HealReport {
    let mut total_heal = HealReport::default();
    for &sk in root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            total_heal.merge(auto_heal_shell(
                solid.outer_shell,
                reg,
                heal_level,
                max_iterations,
            ));
        }
    }
    log::info!("[STEP] healed: {:?}", total_heal);
    total_heal
}

/// Continuity check pass; updates `import_report.continuity_defects`.
pub fn run_continuity_checks(
    reg: &BRepRegistry,
    root_solids: &[SolidKey],
    g0_tol: f32,
    import_report: &mut StepImportReport,
) {
    for &sk in root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            let defects = check_shell_continuity(solid.outer_shell, reg, g0_tol, 5.0);
            import_report.continuity_defects += defects.len();
            if !defects.is_empty() {
                log::info!(
                    "[STEP] shell {:?}: {} continuity defect(s)",
                    solid.outer_shell,
                    defects.len()
                );
            }
        }
    }
}

/// Accumulated mesh property buffers from scene emit.
pub struct SceneMeshEmitResult {
    pub props_vertices: Vec<Vec3>,
    pub props_indices: Vec<i32>,
}

/// Resolve explode factor: explicit > 0, else Preview + overlapping solids → 0.35.
pub fn effective_assembly_explode(
    requested: f32,
    import_mode: StepImportMode,
    reg: &BRepRegistry,
    root_solids: &[SolidKey],
) -> f32 {
    if requested > 0.0 {
        return requested;
    }
    if !matches!(import_mode, StepImportMode::Preview) || root_solids.len() < 2 {
        return 0.0;
    }
    if root_solids_overlap(reg, root_solids) {
        0.35
    } else {
        0.0
    }
}

/// Emit meshed solids into the scene graph (assembly hierarchy or flat instances).
pub fn emit_scene_meshes(
    graph: &mut SceneGraph,
    root: NodeId,
    reg: &BRepRegistry,
    root_solids: &[SolidKey],
    mesh_config: &BRepMeshConfig,
    total_heal: &HealReport,
    assembly_tree: &AssemblyTree,
    assembly_geom_nodes: usize,
    shell_styles: &ShellStyleMap,
    shell_instances: &assembly::ShellInstanceList,
    assembly_explode: f32,
) -> Result<SceneMeshEmitResult, StepError> {
    fn style_rgb(style: &StyleInfo) -> [f32; 3] {
        [style.diffuse.x, style.diffuse.y, style.diffuse.z]
    }

    fn make_material(color: [f32; 3]) -> MaterialNode {
        MaterialNode {
            diffuse_color: Vec3::new(color[0], color[1], color[2]),
            base_color: Vec3::new(color[0], color[1], color[2]),
            roughness: 0.35,
            opacity: 1.0,
            ..Default::default()
        }
    }

    let default_color = [0.9, 0.9, 0.9];
    let default_xform = AssemblyTransform::default();
    let explode_offsets = compute_assembly_explode_offsets(
        reg,
        root_solids,
        assembly_tree,
        assembly_geom_nodes,
        assembly_explode,
    );
    if assembly_explode > 0.0 && !explode_offsets.is_empty() {
        log::info!(
            "[STEP] assembly preview explode: factor {:.2}, {} part(s)",
            assembly_explode,
            explode_offsets.len()
        );
    }
    let mut any_geom = false;
    let mut props_vertices: Vec<Vec3> = Vec::new();
    let mut props_indices: Vec<i32> = Vec::new();
    let use_assembly_hierarchy = assembly_geom_nodes > 1;
    let scene_parent = if use_assembly_hierarchy {
        graph.add_child(root, NodeData::Separator(SeparatorNode))
    } else {
        root
    };

    let mut emit_solid_to = |graph: &mut SceneGraph,
                             parent: NodeId,
                             sk: SolidKey,
                             xform: &AssemblyTransform,
                             any_geom: &mut bool,
                             explode_on_mesh: bool| {
        let Some(solid) = reg.solids.get(sk) else {
            return;
        };
        let Some(final_mesh) = mesh_solid_shell(sk, solid.outer_shell, reg, mesh_config, total_heal)
        else {
            return;
        };
        log::info!(
            "[STEP] mesh: {} verts, {} tris",
            final_mesh.vertices.len(),
            final_mesh.indices.len() / 4,
        );
        append_props_mesh(&final_mesh, &mut props_vertices, &mut props_indices);
        let shell_step_id = reg
            .shells
            .get(solid.outer_shell)
            .and_then(|s| s.step_id);
        let shell_color = shell_step_id
            .and_then(|sid| shell_styles.get(&sid))
            .map(style_rgb);
        let mut mesh = final_mesh;
        apply_mesh_transform(&mut mesh, xform);
        if explode_on_mesh {
            if let Some(offset) = explode_offsets.get(&sk) {
                apply_mesh_translation(&mut mesh, *offset);
            }
        }
        *any_geom = true;
        let comp = graph.add_child(parent, NodeData::Separator(SeparatorNode));
        graph.add_child(
            comp,
            NodeData::Material(make_material(shell_color.unwrap_or(default_color))),
        );
        add_mesh_nodes(graph, comp, mesh);
    };

    if use_assembly_hierarchy {
        let mut emitted = HashSet::new();
        assembly_tree.walk(&mut |node, world, depth| {
            if node.shells.is_empty() {
                return;
            }
            let part = graph.add_child(scene_parent, NodeData::Separator(SeparatorNode));
            let mut part_xform = *world;
            if node.shells.len() == 1 {
                if let Some(sk) =
                    solid_for_shell_step_id(reg, root_solids, node.shells[0])
                {
                    if let Some(offset) = explode_offsets.get(&sk) {
                        part_xform.w_axis.x += offset.x;
                        part_xform.w_axis.y += offset.y;
                        part_xform.w_axis.z += offset.z;
                    }
                }
            }
            graph.add_child(part, NodeData::Transform(transform_node_from_mat4(part_xform)));
            log::debug!(
                "[STEP] assembly part '{}' depth {} shells={}",
                node.name,
                depth,
                node.shells.len()
            );
            let mesh_local = AssemblyTransform::default();
            let explode_on_mesh = node.shells.len() != 1;
            for &shell_step_id in &node.shells {
                if let Some(sk) =
                    solid_for_shell_step_id(reg, root_solids, shell_step_id)
                {
                    if emitted.insert(sk) {
                        emit_solid_to(
                            graph,
                            part,
                            sk,
                            &mesh_local,
                            &mut any_geom,
                            explode_on_mesh,
                        );
                    }
                }
            }
        });
        for &sk in root_solids {
            if !emitted.contains(&sk) {
                emit_solid_to(
                    graph,
                    scene_parent,
                    sk,
                    &default_xform,
                    &mut any_geom,
                    true,
                );
            }
        }
    } else {
        for &sk in root_solids {
            if let Some(solid) = reg.solids.get(sk) {
                let final_mesh = match mesh_solid_shell(
                    sk,
                    solid.outer_shell,
                    reg,
                    mesh_config,
                    total_heal,
                ) {
                    Some(m) => m,
                    None => continue,
                };
                log::info!(
                    "[STEP] mesh: {} verts, {} tris",
                    final_mesh.vertices.len(),
                    final_mesh.indices.len() / 4,
                );
                append_props_mesh(&final_mesh, &mut props_vertices, &mut props_indices);
                let shell_step_id = reg
                    .shells
                    .get(solid.outer_shell)
                    .and_then(|s| s.step_id);
                let shell_color = shell_step_id
                    .and_then(|sid| shell_styles.get(&sid))
                    .map(style_rgb);
                let mut instances: Vec<&AssemblyTransform> = shell_step_id
                    .map(|sid| {
                        shell_instances
                            .iter()
                            .filter(|(id, _)| *id == sid)
                            .map(|(_, xform)| xform)
                            .collect()
                    })
                    .unwrap_or_default();
                if instances.is_empty() {
                    instances.push(&default_xform);
                }
                for xform in instances {
                    let mut mesh = final_mesh.clone();
                    apply_mesh_transform(&mut mesh, xform);
                    if let Some(offset) = explode_offsets.get(&sk) {
                        apply_mesh_translation(&mut mesh, *offset);
                    }
                    any_geom = true;
                    let comp = graph.add_child(root, NodeData::Separator(SeparatorNode));
                    graph.add_child(
                        comp,
                        NodeData::Material(make_material(shell_color.unwrap_or(default_color))),
                    );
                    add_mesh_nodes(graph, comp, mesh);
                }
            }
        }
    }

    if !any_geom {
        return Err(StepError::NoGeometry);
    }

    Ok(SceneMeshEmitResult {
        props_vertices,
        props_indices,
    })
}

fn apply_mesh_transform(mesh: &mut MeshResult, xform: &AssemblyTransform) {
    for v in &mut mesh.vertices {
        *v = xform.transform_point(*v);
    }
    for n in &mut mesh.normals {
        let t = xform.matrix.transform_vector3(*n);
        let len = t.length();
        if len > 1e-10 {
            *n = t * (1.0 / len);
        }
    }
}

fn apply_mesh_translation(mesh: &mut MeshResult, offset: Vec3) {
    for v in &mut mesh.vertices {
        *v += offset;
    }
}

fn shell_bbox(reg: &BRepRegistry, shell_key: crate::step::brep::topo::ShellKey) -> Option<(Vec3, Vec3)> {
    let shell = reg.shells.get(shell_key)?;
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut any = false;
    for &(face_key, _) in &shell.faces {
        let face = reg.faces.get(face_key)?;
        let wires = std::iter::once(face.outer_wire)
            .chain(face.inner_wires.iter().copied());
        for wire_key in wires {
            let wire = reg.wires.get(wire_key)?;
            for &(ek, _) in &wire.edges {
                let edge = reg.edges.get(ek)?;
                for vk in [edge.v_low, edge.v_high] {
                    if let Some(v) = reg.vertices.get(vk) {
                        min = min.min(v.position);
                        max = max.max(v.position);
                        any = true;
                    }
                }
            }
        }
    }
    if any {
        Some((min, max))
    } else {
        None
    }
}

fn aabb_overlap(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) -> bool {
    a_min.x <= b_max.x
        && a_max.x >= b_min.x
        && a_min.y <= b_max.y
        && a_max.y >= b_min.y
        && a_min.z <= b_max.z
        && a_max.z >= b_min.z
}

pub fn root_solids_overlap(reg: &BRepRegistry, root_solids: &[SolidKey]) -> bool {
    let mut bboxes = Vec::new();
    for &sk in root_solids {
        let solid = match reg.solids.get(sk) {
            Some(s) => s,
            None => continue,
        };
        if let Some(bb) = shell_bbox(reg, solid.outer_shell) {
            bboxes.push(bb);
        }
    }
    for i in 0..bboxes.len() {
        for j in i + 1..bboxes.len() {
            let (a_min, a_max) = bboxes[i];
            let (b_min, b_max) = bboxes[j];
            if aabb_overlap(a_min, a_max, b_min, b_max) {
                return true;
            }
        }
    }
    false
}

fn transform_point_mat4(m: &Mat4, p: Vec3) -> Vec3 {
    let v = *m * p.extend(1.0);
    Vec3::new(v.x, v.y, v.z)
}

fn compute_assembly_explode_offsets(
    reg: &BRepRegistry,
    root_solids: &[SolidKey],
    assembly_tree: &AssemblyTree,
    assembly_geom_nodes: usize,
    explode: f32,
) -> HashMap<SolidKey, Vec3> {
    if explode <= 0.0 || root_solids.len() < 2 {
        return HashMap::new();
    }

    let spread_dirs = [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::NEG_X,
        Vec3::NEG_Y,
        Vec3::NEG_Z,
    ];

    let mut centers: Vec<(SolidKey, Vec3)> = Vec::new();
    let mut union_min = Vec3::splat(f32::MAX);
    let mut union_max = Vec3::splat(f32::MIN);

    if assembly_geom_nodes > 1 {
        assembly_tree.walk(&mut |node, world, _| {
            if node.shells.is_empty() {
                return;
            }
            for &shell_step_id in &node.shells {
                let Some(sk) = solid_for_shell_step_id(reg, root_solids, shell_step_id) else {
                    continue;
                };
                let Some(solid) = reg.solids.get(sk) else {
                    continue;
                };
                let Some((min, max)) = shell_bbox(reg, solid.outer_shell) else {
                    continue;
                };
                let local_center = (min + max) * 0.5;
                let world_center = transform_point_mat4(world, local_center);
                centers.push((sk, world_center));
                union_min = union_min.min(min);
                union_max = union_max.max(max);
            }
        });
    }

    let mut seen = HashSet::new();
    for &(sk, _) in &centers {
        seen.insert(sk);
    }
    for &sk in root_solids {
        if seen.contains(&sk) {
            continue;
        }
        let Some(solid) = reg.solids.get(sk) else {
            continue;
        };
        let Some((min, max)) = shell_bbox(reg, solid.outer_shell) else {
            continue;
        };
        let center = (min + max) * 0.5;
        centers.push((sk, center));
        union_min = union_min.min(min);
        union_max = union_max.max(max);
    }

    if centers.len() < 2 {
        return HashMap::new();
    }

    let assembly_diag = (union_max - union_min).length().max(1e-6);
    let centroid = centers.iter().map(|(_, c)| *c).sum::<Vec3>() / centers.len() as f32;
    let mut offsets = HashMap::new();
    for (i, (sk, center)) in centers.iter().enumerate() {
        let mut dir = *center - centroid;
        if dir.length_squared() < 1e-12 {
            dir = spread_dirs[i % spread_dirs.len()];
        } else {
            dir = dir.normalize();
        }
        let mag = explode * assembly_diag * 0.5;
        offsets.insert(*sk, dir * mag);
    }
    offsets
}

fn solid_for_shell_step_id(
    reg: &BRepRegistry,
    root_solids: &[SolidKey],
    shell_step_id: u64,
) -> Option<SolidKey> {
    for &sk in root_solids {
        let solid = reg.solids.get(sk)?;
        let sid = reg.shells.get(solid.outer_shell)?.step_id?;
        if sid == shell_step_id {
            return Some(sk);
        }
    }
    None
}

fn mesh_solid_shell(
    sk: SolidKey,
    outer_shell: crate::step::brep::topo::ShellKey,
    reg: &BRepRegistry,
    mesh_config: &BRepMeshConfig,
    heal: &HealReport,
) -> Option<MeshResult> {
    let solid = reg.solids.get(sk)?;
    let base_mesh = brep::mesh::mesh_brep_shell(outer_shell, reg, mesh_config, &heal.skip_face_keys);
    if base_mesh.vertices.is_empty() || base_mesh.indices.is_empty() {
        return None;
    }
    let void_meshes: Vec<_> = solid
        .void_shells
        .iter()
        .map(|&vk| brep::mesh::mesh_brep_shell(vk, reg, mesh_config, &heal.skip_face_keys))
        .collect();
    let void_result = brep::mesh::void_subtract::subtract_void_meshes(&base_mesh, &void_meshes);
    if void_result.removed_tris > 0 {
        log::info!(
            "[STEP] void subtraction: removed {} tris, kept {}",
            void_result.removed_tris,
            void_result.mesh.indices.len() / 4,
        );
    }
    Some(void_result.mesh)
}

fn append_props_mesh(mesh: &MeshResult, props_vertices: &mut Vec<Vec3>, props_indices: &mut Vec<i32>) {
    let base_offset = props_vertices.len() as i32;
    props_vertices.extend_from_slice(&mesh.vertices);
    for chunk in mesh.indices.chunks(4) {
        if chunk.len() >= 3 {
            props_indices.extend_from_slice(&[
                chunk[0] + base_offset,
                chunk[1] + base_offset,
                chunk[2] + base_offset,
                -1,
            ]);
        }
    }
}

fn add_mesh_nodes(graph: &mut SceneGraph, comp: NodeId, mesh: MeshResult) {
    let vert_count = mesh.vertices.len();
    let normal_count = mesh.normals.len();
    let tri_count = mesh.indices.len() / 4;
    graph.add_child(
        comp,
        NodeData::Coordinate3(Coordinate3Node {
            point: mesh.vertices,  // moved, not cloned
        }),
    );
    if !mesh.normals.is_empty() {
        graph.add_child(
            comp,
            NodeData::Normal(NormalNode::from_vectors(mesh.normals)),  // moved
        );
    }
    graph.add_child(
        comp,
        NodeData::IndexedFaceSet(IndexedFaceSetNode {
            coord_index: mesh.indices,  // moved
        }),
    );
    log::debug!(
        "[STEP] scene mesh: {} verts, {} normals, {} tris ({} KB indices)",
        vert_count,
        normal_count,
        tri_count,
        tri_count * 12 / 1024,
    );
}

fn transform_node_from_mat4(m: Mat4) -> TransformNode {
    TransformNode {
        translation: Vec3::new(m.w_axis.x, m.w_axis.y, m.w_axis.z),
        rotation: m,
        scale: Vec3::ONE,
        center: Vec3::ZERO,
    }
}
