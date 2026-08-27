//! Object (non-joint) animation tracks: TRS on any scene node, plus morph weights.
//!
//! Analogous to three.js `KeyframeTrack` + `PropertyBinding` targeting a node
//! rather than a skeleton joint. Bindings use live [`NodeId`] values, so clips
//! are built against a concrete `SceneGraph` (CAD assembly setup), not as
//! portable assets across SlotMap sessions.

use rc3d_core::math::{Quat, Vec3};
use rc3d_core::NodeId;
use serde::{Deserialize, Serialize};

/// Which property on [`PropertyBinding::node`] a track writes.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObjectChannel {
    Translation,
    Rotation,
    Scale,
    MorphWeight(u32),
}

/// three.js-style property binding: scene node + channel.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PropertyBinding {
    pub node: NodeId,
    pub channel: ObjectChannel,
}

/// Sampled or keyed value for one object channel.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum ObjectKeyframeValue {
    Vec3(Vec3),
    Quat(Quat),
    Scalar(f32),
}

impl ObjectKeyframeValue {
    pub fn lerp(self, other: Self, t: f32) -> Self {
        match (self, other) {
            (Self::Vec3(a), Self::Vec3(b)) => Self::Vec3(a.lerp(b, t)),
            (Self::Quat(a), Self::Quat(b)) => Self::Quat(a.slerp(b, t)),
            (Self::Scalar(a), Self::Scalar(b)) => Self::Scalar(a + (b - a) * t),
            (a, _) => a,
        }
    }
}

/// One keyframe on an [`ObjectTrack`].
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct ObjectKeyframe {
    pub time: f32,
    pub value: ObjectKeyframeValue,
}

/// Keyframe curve for one [`PropertyBinding`].
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ObjectTrack {
    pub binding: PropertyBinding,
    pub keyframes: Vec<ObjectKeyframe>,
}

impl ObjectTrack {
    pub fn translation(node: NodeId, times: Vec<f32>, values: Vec<Vec3>) -> Self {
        Self::from_pairs(
            PropertyBinding {
                node,
                channel: ObjectChannel::Translation,
            },
            times,
            values.into_iter().map(ObjectKeyframeValue::Vec3),
        )
    }

    pub fn rotation(node: NodeId, times: Vec<f32>, values: Vec<Quat>) -> Self {
        Self::from_pairs(
            PropertyBinding {
                node,
                channel: ObjectChannel::Rotation,
            },
            times,
            values.into_iter().map(ObjectKeyframeValue::Quat),
        )
    }

    pub fn scale(node: NodeId, times: Vec<f32>, values: Vec<Vec3>) -> Self {
        Self::from_pairs(
            PropertyBinding {
                node,
                channel: ObjectChannel::Scale,
            },
            times,
            values.into_iter().map(ObjectKeyframeValue::Vec3),
        )
    }

    pub fn morph_weight(node: NodeId, index: u32, times: Vec<f32>, values: Vec<f32>) -> Self {
        Self::from_pairs(
            PropertyBinding {
                node,
                channel: ObjectChannel::MorphWeight(index),
            },
            times,
            values.into_iter().map(ObjectKeyframeValue::Scalar),
        )
    }

    fn from_pairs(
        binding: PropertyBinding,
        times: Vec<f32>,
        values: impl IntoIterator<Item = ObjectKeyframeValue>,
    ) -> Self {
        let mut keyframes: Vec<ObjectKeyframe> = times
            .into_iter()
            .zip(values)
            .map(|(time, value)| ObjectKeyframe { time, value })
            .collect();
        keyframes.sort_by(|a, b| a.time.total_cmp(&b.time));
        Self { binding, keyframes }
    }

    /// Linear / slerp sample. Time is already wrapped by the clip; this clamps
    /// to the first/last keyframe (hold).
    pub fn sample(&self, time: f32) -> Option<ObjectKeyframeValue> {
        if self.keyframes.is_empty() {
            return None;
        }
        if self.keyframes.len() == 1 || time <= self.keyframes[0].time {
            return Some(self.keyframes[0].value);
        }
        let last = self.keyframes.last()?;
        if time >= last.time {
            return Some(last.value);
        }

        let mut next_idx = 1;
        for (i, kf) in self.keyframes.iter().enumerate() {
            if kf.time > time {
                next_idx = i;
                break;
            }
            next_idx = i;
        }
        if next_idx == 0 {
            return Some(self.keyframes[0].value);
        }

        let prev = &self.keyframes[next_idx - 1];
        let next = &self.keyframes[next_idx];
        let range = next.time - prev.time;
        let alpha = if range > 1e-6 {
            (time - prev.time) / range
        } else {
            0.0
        };
        Some(prev.value.lerp(next.value, alpha))
    }
}
