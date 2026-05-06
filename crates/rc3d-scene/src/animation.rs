//! Skeletal animation data: skeleton, animation clips, skinning weights.
//!
//! Joint hierarchy is stored as a flat array where each joint references its
//! parent by index. The root joint has parent == joint_count (sentinel).
//!
//! Animation clips store per-joint keyframe tracks with linear interpolation.
//! Skinning data attaches vertex bone indices + weights for GPU skinning.

use serde::{Deserialize, Serialize};

use rc3d_core::math::{Mat4, Quat, Vec3, Vec4};

/// One joint (bone) in a skeleton.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Joint {
    pub name: String,
    /// Index of parent joint, or `joint_count` for root.
    pub parent: usize,
    /// Local bind-pose transform (relative to parent).
    pub bind_transform: Mat4,
    /// Inverse of the global bind-pose transform.
    pub inverse_bind_matrix: Mat4,
}

/// Skeleton: flat array of joints + cached global transforms.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
    /// Pre-computed global bind-pose transforms.
    pub global_bind_poses: Vec<Mat4>,
}

impl Skeleton {
    pub fn new(mut joints: Vec<Joint>) -> Self {
        let n = joints.len();
        let mut global_bind_poses = vec![Mat4::IDENTITY; n];

        // Compute global bind poses via iteration over topological order.
        // Since joints[i].parent < i (flat hierarchy), forward scan works.
        for i in 0..n {
            let parent = if joints[i].parent < n {
                global_bind_poses[joints[i].parent]
            } else {
                Mat4::IDENTITY
            };
            global_bind_poses[i] = parent * joints[i].bind_transform;
            joints[i].inverse_bind_matrix = global_bind_poses[i].inverse();
        }

        Skeleton {
            joints,
            global_bind_poses,
        }
    }

    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// Resolve joint transforms from a pose (array of per-joint local transforms).
    /// Returns array of global joint matrices for skinning.
    pub fn resolve_global_poses(&self, local_poses: &[Mat4]) -> Vec<Mat4> {
        let n = self.joint_count().min(local_poses.len());
        let mut globals = vec![Mat4::IDENTITY; n];
        for i in 0..n {
            let parent = if self.joints[i].parent < n {
                globals[self.joints[i].parent]
            } else {
                Mat4::IDENTITY
            };
            globals[i] = parent * local_poses[i];
        }
        globals
    }

    /// Compute skinning matrices: global_poses[i] * inverse_bind_matrix[i].
    pub fn skinning_matrices(&self, local_poses: &[Mat4]) -> Vec<Mat4> {
        let globals = self.resolve_global_poses(local_poses);
        let n = self.joint_count().min(local_poses.len());
        let mut mats = vec![Mat4::IDENTITY; n];
        for i in 0..n {
            mats[i] = globals[i] * self.joints[i].inverse_bind_matrix;
        }
        mats
    }
}

/// Per-joint keyframe with timestamp and local transform.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct JointKeyframe {
    pub time: f32,
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl JointKeyframe {
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// Keyframe track for one joint in one animation clip.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct JointTrack {
    pub joint_index: usize,
    pub keyframes: Vec<JointKeyframe>,
}

impl JointTrack {
    /// Sample the track at a given time (linear interpolation).
    pub fn sample(&self, time: f32) -> Mat4 {
        if self.keyframes.is_empty() {
            return Mat4::IDENTITY;
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].to_matrix();
        }

        // Wrap time to clip duration
        let duration = self.keyframes.last().unwrap().time;
        let t = if duration > 0.0 { time % duration } else { 0.0 };

        // Find surrounding keyframes
        let mut next_idx = 0;
        for (i, kf) in self.keyframes.iter().enumerate() {
            if kf.time > t {
                next_idx = i;
                break;
            }
            next_idx = i;
        }

        if next_idx == 0 {
            return self.keyframes[0].to_matrix();
        }

        let prev = &self.keyframes[next_idx - 1];
        let next = &self.keyframes[next_idx];

        let range = next.time - prev.time;
        let alpha = if range > 1e-6 {
            (t - prev.time) / range
        } else {
            0.0
        };

        let translation = prev.translation.lerp(next.translation, alpha);
        let rotation = prev.rotation.slerp(next.rotation, alpha);
        let scale = prev.scale.lerp(next.scale, alpha);

        Mat4::from_scale_rotation_translation(scale, rotation, translation)
    }
}

/// Animation clip: named sequence of joint tracks.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub tracks: Vec<JointTrack>,
}

impl AnimationClip {
    /// Sample all joint tracks at the given time.
    /// Untracked joints use [`Joint::bind_transform`] from `skeleton` (rest/bind local pose).
    pub fn sample_all(&self, time: f32, skeleton: &Skeleton) -> Vec<Mat4> {
        let joint_count = skeleton.joint_count();
        let mut poses: Vec<Mat4> = skeleton
            .joints
            .iter()
            .map(|j| j.bind_transform)
            .collect();
        for track in &self.tracks {
            if track.joint_index < joint_count {
                poses[track.joint_index] = track.sample(time);
            }
        }
        poses
    }
}

/// Per-vertex skinning data: up to 4 bone indices + weights.
#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexSkinData {
    pub bone_indices: [u32; 4],
    pub bone_weights: [f32; 4],
}

impl VertexSkinData {
    pub fn empty() -> Self {
        Self {
            bone_indices: [0; 4],
            bone_weights: [0.0; 4],
        }
    }
}

/// Animation player: drives one animation clip playback with blending.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AnimationPlayer {
    pub clip: Option<AnimationClip>,
    pub current_time: f32,
    pub speed: f32,
    pub playing: bool,
    /// Blending: optional next clip + blend factor [0,1].
    pub next_clip: Option<AnimationClip>,
    pub blend_factor: f32,
}

impl AnimationPlayer {
    pub fn new() -> Self {
        Self {
            clip: None,
            current_time: 0.0,
            speed: 1.0,
            playing: true,
            next_clip: None,
            blend_factor: 0.0,
        }
    }

    /// Advance time by delta_seconds.
    pub fn tick(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        self.current_time += dt * self.speed;

        // Update blend factor
        if self.next_clip.is_some() {
            self.blend_factor = (self.blend_factor + dt * 3.0).min(1.0);
            if self.blend_factor >= 1.0 {
                self.clip = self.next_clip.take();
                self.blend_factor = 0.0;
                self.current_time = 0.0;
            }
        }
    }

    /// Play a new clip, optionally blending from current.
    pub fn play(&mut self, clip: AnimationClip, blend: bool) {
        if blend && self.clip.is_some() {
            self.next_clip = Some(clip);
            self.blend_factor = 0.0;
        } else {
            self.clip = Some(clip);
            self.current_time = 0.0;
            self.playing = true;
        }
    }

    /// Sample the current pose. Untracked joints use bind pose from `skeleton`.
    /// If blending, mixes current and next clip.
    pub fn sample_pose(&self, skeleton: &Skeleton) -> Vec<Mat4> {
        let mut pose = match &self.clip {
            Some(clip) => clip.sample_all(self.current_time, skeleton),
            None => skeleton
                .joints
                .iter()
                .map(|j| j.bind_transform)
                .collect(),
        };

        if let (Some(next), true) = (&self.next_clip, self.blend_factor > 0.0) {
            let next_pose = next.sample_all(self.current_time, skeleton);
            let a = self.blend_factor;
            for (p, np) in pose.iter_mut().zip(next_pose.iter()) {
                // Blend: lerp translation, slerp rotation, lerp scale
                let (t1, r1, s1) = decompose_matrix(*p);
                let (t2, r2, s2) = decompose_matrix(*np);
                let t = t1.lerp(t2, a);
                let r = r1.slerp(r2, a);
                let s = s1.lerp(s2, a);
                *p = Mat4::from_scale_rotation_translation(s, r, t);
            }
        }

        pose
    }
}

impl Default for AnimationPlayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Decompose a matrix into translation, rotation, scale.
/// Assumes the matrix is TRS-composed (no shear/projection).
fn decompose_matrix(m: Mat4) -> (Vec3, Quat, Vec3) {
    let translation = m.w_axis.truncate();
    let scale = Vec3::new(
        m.x_axis.truncate().length(),
        m.y_axis.truncate().length(),
        m.z_axis.truncate().length(),
    );
    let r00 = m.x_axis.x / scale.x;
    let r10 = m.x_axis.y / scale.x;
    let r20 = m.x_axis.z / scale.x;
    let r01 = m.y_axis.x / scale.y;
    let r11 = m.y_axis.y / scale.y;
    let r21 = m.y_axis.z / scale.y;
    let r02 = m.z_axis.x / scale.z;
    let r12 = m.z_axis.y / scale.z;
    let r22 = m.z_axis.z / scale.z;
    let rot_mat = Mat4::from_cols(
        Vec4::new(r00, r10, r20, 0.0),
        Vec4::new(r01, r11, r21, 0.0),
        Vec4::new(r02, r12, r22, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );
    let rotation = Quat::from_mat4(&rot_mat);
    (translation, rotation, scale)
}

/// One node in an animation blend tree.
/// Leaf nodes reference a clip; internal nodes blend children by weight.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum BlendNode {
    /// Plays a single animation clip.
    Clip {
        clip: AnimationClip,
        speed: f32,
        start_time: f32,
    },
    /// Linearly blends two child nodes by weight [0,1].
    Blend {
        left: Box<BlendNode>,
        right: Box<BlendNode>,
        weight: f32,
    },
    /// Additively combines two child nodes (second adds on top of first).
    Additive {
        base: Box<BlendNode>,
        additive: Box<BlendNode>,
        weight: f32,
    },
}

impl BlendNode {
    /// Sample the blend tree at a given time, using the provided skeleton for clip sampling.
    pub fn sample(&self, time: f32, skeleton: &Skeleton) -> Option<Vec<Mat4>> {
        match self {
            BlendNode::Clip { clip, speed, start_time } => {
                let t = start_time + (time * speed) % clip.duration.max(0.001);
                Some(clip.sample_all(t, skeleton))
            }
            BlendNode::Blend { left, right, weight } => {
                let l = left.sample(time, skeleton)?;
                let r = right.sample(time, skeleton)?;
                let w = weight.clamp(0.0, 1.0);
                // Per-joint lerp
                let blended: Vec<Mat4> = l.iter().zip(r.iter())
                    .map(|(lm, rm)| {
                        let (lt, lr, ls) = decompose_matrix(*lm);
                        let (rt, rr, rs) = decompose_matrix(*rm);
                        let t = lt.lerp(rt, w);
                        let rot = lr.slerp(rr, w);
                        let s = ls.lerp(rs, w);
                        Mat4::from_scale_rotation_translation(s, rot, t)
                    })
                    .collect();
                Some(blended)
            }
            BlendNode::Additive { base, additive, weight } => {
                let b = base.sample(time, skeleton)?;
                let a = additive.sample(time, skeleton)?;
                let w = weight.clamp(0.0, 1.0);
                let result: Vec<Mat4> = b.iter().zip(a.iter())
                    .map(|(bm, am)| {
                        let (bt, _br, bs) = decompose_matrix(*bm);
                        let (at, ar, as_) = decompose_matrix(*am);
                        let t = bt + at * w;
                        let s = bs + as_ * w;
                        *bm * Mat4::from_scale_rotation_translation(s, ar, t)
                    })
                    .collect();
                Some(result)
            }
        }
    }
}
