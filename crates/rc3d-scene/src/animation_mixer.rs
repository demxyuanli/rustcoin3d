//! three.js `AnimationMixer` analog: play clips and write object tracks into the graph.
//!
//! Skeleton poses stay on [`crate::animation::AnimationPlayer::sample_pose`]. This
//! mixer applies [`crate::object_track::ObjectTrack`] values to `TransformNode`
//! and `MorphTargetNode` each tick so CAD assembly parts can move without joints.

use std::collections::HashMap;

use rc3d_core::math::Mat4;

use crate::animation::{wrap_animation_time, AnimationClip, AnimationPlayer};
use crate::node_data::NodeData;
use crate::node_entry::dirty_flags;
use crate::object_track::{ObjectChannel, ObjectKeyframeValue, PropertyBinding};
use crate::SceneGraph;

/// Drives one [`AnimationPlayer`] and applies its object tracks to a scene.
#[derive(Clone, Debug, Default)]
pub struct AnimationMixer {
    pub player: AnimationPlayer,
}

impl AnimationMixer {
    pub fn new() -> Self {
        Self {
            player: AnimationPlayer::new(),
        }
    }

    pub fn from_clip(clip: AnimationClip) -> Self {
        let mut mixer = Self::new();
        mixer.play(clip, false);
        mixer
    }

    pub fn play(&mut self, clip: AnimationClip, blend: bool) {
        self.player.play(clip, blend);
    }

    pub fn tick(&mut self, dt: f32) {
        self.player.tick(dt);
    }

    /// Write sampled object tracks onto matching scene nodes.
    pub fn apply(&self, graph: &mut SceneGraph) {
        self.player.apply_object_tracks(graph);
    }
}

impl AnimationPlayer {
    /// Sample object tracks from the current (and blended) clip and write them.
    pub fn apply_object_tracks(&self, graph: &mut SceneGraph) {
        let Some(clip) = &self.clip else {
            return;
        };
        let t = wrap_animation_time(self.current_time, clip.duration, self.loop_mode);
        let mut values = sample_clip_object_tracks(clip, t);

        if let (Some(next), true) = (&self.next_clip, self.blend_factor > 0.0) {
            let nt = wrap_animation_time(self.current_time, next.duration, self.loop_mode);
            let next_values = sample_clip_object_tracks(next, nt);
            let a = self.blend_factor;
            for (binding, nv) in next_values {
                match values.get(&binding).copied() {
                    Some(cv) => {
                        values.insert(binding, cv.lerp(nv, a));
                    }
                    None => {
                        values.insert(binding, nv);
                    }
                }
            }
        }

        for (binding, value) in values {
            apply_binding(graph, binding, value);
        }
    }
}

fn sample_clip_object_tracks(
    clip: &AnimationClip,
    time: f32,
) -> HashMap<PropertyBinding, ObjectKeyframeValue> {
    let mut out = HashMap::with_capacity(clip.object_tracks.len());
    for track in &clip.object_tracks {
        if let Some(value) = track.sample(time) {
            out.insert(track.binding, value);
        }
    }
    out
}

fn apply_binding(graph: &mut SceneGraph, binding: PropertyBinding, value: ObjectKeyframeValue) {
    let Some(entry) = graph.get_mut(binding.node) else {
        return;
    };
    match (binding.channel, value, &mut entry.data) {
        (ObjectChannel::Translation, ObjectKeyframeValue::Vec3(v), NodeData::Transform(t)) => {
            t.translation = v;
            entry.dirty_flags |= dirty_flags::TRANSFORM;
        }
        (ObjectChannel::Rotation, ObjectKeyframeValue::Quat(q), NodeData::Transform(t)) => {
            t.rotation = Mat4::from_quat(q);
            entry.dirty_flags |= dirty_flags::TRANSFORM;
        }
        (ObjectChannel::Scale, ObjectKeyframeValue::Vec3(v), NodeData::Transform(t)) => {
            t.scale = v;
            entry.dirty_flags |= dirty_flags::TRANSFORM;
        }
        (
            ObjectChannel::MorphWeight(index),
            ObjectKeyframeValue::Scalar(w),
            NodeData::MorphTarget(mt),
        ) => {
            let i = index as usize;
            if mt.weights.len() <= i {
                mt.weights.resize(i + 1, 0.0);
            }
            mt.weights[i] = w;
            entry.dirty_flags |= dirty_flags::GEOMETRY;
        }
        _ => {}
    }
}
