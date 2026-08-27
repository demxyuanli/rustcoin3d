use std::collections::HashMap;

use rc3d_core::math::{Quat, Vec3};
use rc3d_scene::animation::{AnimationClip, JointKeyframe, JointTrack};

use super::types::*;

/// Resolve animation clips from parsed FBX data.
/// Maps bone_id -> joint_index using the bone_ids list from skeleton resolution.
pub fn resolve_animations(
    objects: &HashMap<i64, FbxObject>,
    connections: &[FbxConnection],
    bone_id_to_index: &HashMap<i64, usize>,
) -> Vec<AnimationClip> {
    // Build connection maps
    // AnimationStack -> AnimationLayer
    let mut stack_to_layers: HashMap<i64, Vec<i64>> = HashMap::new();
    // AnimationLayer -> AnimationCurveNode
    let mut layer_to_curve_nodes: HashMap<i64, Vec<i64>> = HashMap::new();
    // AnimationCurveNode -> AnimationCurve (multiple: X/Y/Z)
    let mut curve_node_to_curves: HashMap<i64, Vec<i64>> = HashMap::new();
    // AnimationCurveNode -> Model (bone)
    let mut curve_node_to_model: HashMap<i64, i64> = HashMap::new();

    for conn in connections {
        match (
            objects.get(&conn.child),
            objects.get(&conn.parent),
        ) {
            (Some(FbxObject::AnimationLayer(_)), Some(FbxObject::AnimationStack(_))) => {
                stack_to_layers.entry(conn.parent).or_default().push(conn.child);
            }
            (Some(FbxObject::AnimationCurveNode(_)), Some(FbxObject::AnimationLayer(_))) => {
                layer_to_curve_nodes
                    .entry(conn.parent)
                    .or_default()
                    .push(conn.child);
            }
            (Some(FbxObject::AnimationCurve(_)), Some(FbxObject::AnimationCurveNode(_))) => {
                curve_node_to_curves
                    .entry(conn.parent)
                    .or_default()
                    .push(conn.child);
            }
            (Some(FbxObject::AnimationCurveNode(_)), Some(FbxObject::Model(_))) => {
                curve_node_to_model.insert(conn.child, conn.parent);
            }
            _ => {}
        }
    }

    let mut clips = Vec::new();

    // Build clips from animation stacks
    for (&stack_id, stack_obj) in objects.iter() {
        let stack = match stack_obj {
            FbxObject::AnimationStack(s) => s,
            _ => continue,
        };

        let layers = stack_to_layers.get(&stack_id).cloned().unwrap_or_default();
        let mut all_tracks: Vec<JointTrack> = Vec::new();
        let mut max_time = 0.0f32;

        for layer_id in layers {
            let curve_nodes = layer_to_curve_nodes.get(&layer_id).cloned().unwrap_or_default();

            for cn_id in curve_nodes {
                let cn_name = match objects.get(&cn_id) {
                    Some(FbxObject::AnimationCurveNode(n)) => n.name.clone(),
                    _ => continue,
                };

                let target_model_id =
                    if let Some(FbxObject::AnimationCurveNode(n)) = objects.get(&cn_id) {
                        n.target_model_id
                            .or_else(|| curve_node_to_model.get(&cn_id).copied())
                    } else {
                        None
                    };
                let joint_index = target_model_id
                    .and_then(|mid| bone_id_to_index.get(&mid))
                    .copied()
                    .unwrap_or(0);

                let curve_ids = curve_node_to_curves.get(&cn_id).cloned().unwrap_or_default();

                // Determine target property from curve node name
                let is_rotation = cn_name.contains("R") || cn_name.contains("Rotation");
                let _is_translation = cn_name.contains("T") || cn_name.contains("Translation");
                let _is_scale = cn_name.contains("S") || cn_name.contains("Scale");

                // Collect curves (up to 3 for X/Y/Z)
                let mut x_curve: Option<&FbxAnimationCurve> = None;
                let mut y_curve: Option<&FbxAnimationCurve> = None;
                let mut z_curve: Option<&FbxAnimationCurve> = None;

                for (i, &curve_id) in curve_ids.iter().enumerate() {
                    if let Some(FbxObject::AnimationCurve(c)) = objects.get(&curve_id) {
                        match i {
                            0 => x_curve = Some(c),
                            1 => y_curve = Some(c),
                            2 => z_curve = Some(c),
                            _ => {}
                        }
                    }
                }

                // Build keyframes by merging the 3 curves
                let keyframes = build_keyframes(x_curve, y_curve, z_curve, is_rotation);

                if !keyframes.is_empty() {
                    if let Some(last) = keyframes.last() {
                        max_time = max_time.max(last.time);
                    }
                    all_tracks.push(JointTrack {
                        joint_index,
                        keyframes,
                    });
                }
            }
        }

        if !all_tracks.is_empty() {
            clips.push(AnimationClip {
                name: stack.name.clone(),
                duration: max_time,
                tracks: all_tracks,
                object_tracks: Vec::new(),
            });
        }
    }

    clips
}

fn build_keyframes(
    x_curve: Option<&FbxAnimationCurve>,
    y_curve: Option<&FbxAnimationCurve>,
    z_curve: Option<&FbxAnimationCurve>,
    is_rotation: bool,
) -> Vec<JointKeyframe> {
    let x = x_curve.map(|c| &c.times).map(|t| t.as_slice()).unwrap_or(&[]);
    let y = y_curve.map(|c| &c.times).map(|t| t.as_slice()).unwrap_or(&[]);
    let z = z_curve.map(|c| &c.times).map(|t| t.as_slice()).unwrap_or(&[]);

    let x_vals = x_curve.map(|c| &c.values).map(|v| v.as_slice()).unwrap_or(&[]);
    let y_vals = y_curve.map(|c| &c.values).map(|v| v.as_slice()).unwrap_or(&[]);
    let z_vals = z_curve.map(|c| &c.values).map(|v| v.as_slice()).unwrap_or(&[]);

    // Use X curve times as the reference (most common FBX convention)
    let ref_times = if !x.is_empty() { x } else if !y.is_empty() { y } else { z };
    let ref_x = if !x_vals.is_empty() { x_vals } else { &[] };
    let ref_y = if !y_vals.is_empty() { y_vals } else { &[] };
    let ref_z = if !z_vals.is_empty() { z_vals } else { &[] };

    if ref_times.is_empty() {
        return Vec::new();
    }

    let mut keyframes = Vec::with_capacity(ref_times.len());

    for (i, &t_raw) in ref_times.iter().enumerate() {
        // FBX stores time in FBX time units; convert to seconds
        // FBX time unit = 1/46186158000 seconds
        let t = (t_raw / 46186158000.0) as f32;

        let x_val = get_or_last(ref_x, i) as f32;
        let y_val = get_or_last(ref_y, i) as f32;
        let z_val = get_or_last(ref_z, i) as f32;

        if is_rotation {
            // FBX rotation is in degrees
            let rx = x_val.to_radians();
            let ry = y_val.to_radians();
            let rz = z_val.to_radians();
            // Convert Euler to quaternion (XYZ order)
            let q = Quat::from_rotation_x(rx)
                * Quat::from_rotation_y(ry)
                * Quat::from_rotation_z(rz);
            keyframes.push(JointKeyframe {
                time: t,
                translation: Vec3::ZERO,
                rotation: q,
                scale: Vec3::ONE,
            });
        } else {
            keyframes.push(JointKeyframe {
                time: t,
                translation: Vec3::new(x_val, y_val, z_val),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            });
        }
    }

    keyframes
}

fn get_or_last(slice: &[f64], idx: usize) -> f64 {
    if slice.is_empty() {
        0.0
    } else {
        slice[idx.min(slice.len() - 1)]
    }
}
