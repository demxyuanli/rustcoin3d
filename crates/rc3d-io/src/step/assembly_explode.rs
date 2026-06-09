//! Assembly preview explode offsets (transform-only, no vertex bake).

use std::collections::{HashMap, HashSet};

use rc3d_core::math::{Mat4, Vec3};
use rc3d_shape::{ShapeDocument, ShapeId};

use rc3d_shape::BRepStore;
use rc3d_shape::topo::{ShellKey, SolidKey};
use crate::step::import_options::StepImportMode;

/// Resolve explode factor: explicit > 0, else Preview + overlapping solids -> 0.35.
pub fn effective_assembly_explode(
    requested: f32,
    import_mode: StepImportMode,
    reg: &BRepStore,
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

/// Per-solid translation offsets for assembly preview explode.
pub fn compute_assembly_explode_offsets(
    doc: &mut ShapeDocument,
    root_solids: &[SolidKey],
    explode: f32,
) -> HashMap<SolidKey, Vec3> {
    if explode <= 0.0 || root_solids.len() < 2 {
        return HashMap::new();
    }

    let assembly_geom_nodes = doc
        .labels
        .labels
        .iter()
        .filter(|(_, label)| label.shape.is_some())
        .count();

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
        let shaped: Vec<(ShapeId, SolidKey)> = doc
            .labels
            .labels
            .iter()
            .filter_map(|(_, label)| {
                let shape_id = label.shape?;
                let node = doc.shapes.get(shape_id)?;
                let sk = node.kind.solid_key()?;
                root_solids.contains(&sk).then_some((shape_id, sk))
            })
            .collect();
        for (shape_id, sk) in shaped {
            let Some(solid) = doc.store.solids.get(sk) else {
                continue;
            };
            let Some((min, max)) = shell_bbox(&doc.store, solid.outer_shell) else {
                continue;
            };
            let local_center = (min + max) * 0.5;
            let world = doc.world_transform(shape_id);
            let world_center = transform_point_mat4(&world, local_center);
            centers.push((sk, world_center));
            union_min = union_min.min(min);
            union_max = union_max.max(max);
        }
    }

    let mut seen = HashSet::new();
    for &(sk, _) in &centers {
        seen.insert(sk);
    }
    for &sk in root_solids {
        if seen.contains(&sk) {
            continue;
        }
        let Some(solid) = doc.store.solids.get(sk) else {
            continue;
        };
        let Some((min, max)) = shell_bbox(&doc.store, solid.outer_shell) else {
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

pub fn root_solids_overlap(reg: &BRepStore, root_solids: &[SolidKey]) -> bool {
    let mut bboxes = Vec::new();
    for &sk in root_solids {
        let Some(solid) = reg.solids.get(sk) else {
            continue;
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

fn shell_bbox(reg: &BRepStore, shell_key: ShellKey) -> Option<(Vec3, Vec3)> {
    let shell = reg.shells.get(shell_key)?;
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut any = false;
    for &(face_key, _) in &shell.faces {
        let face = reg.faces.get(face_key)?;
        let wires = std::iter::once(face.outer_wire).chain(face.inner_wires.iter().copied());
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

fn transform_point_mat4(m: &Mat4, p: Vec3) -> Vec3 {
    let v = *m * p.extend(1.0);
    Vec3::new(v.x, v.y, v.z)
}
