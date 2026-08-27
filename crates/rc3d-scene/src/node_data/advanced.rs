//! File reference and advanced rendering nodes.
use serde::{Deserialize, Serialize};

/// External file reference (Coin3D SoFile / SoWWWInline).
/// When encountered during traversal, the referenced file is imported
/// and its scene graph is merged in-place.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[derive(Default)]
pub struct FileNode {
    pub path: String,
}


/// GPU ray tracing render mode (compute-based path tracing, HOOPS Luminate equivalent).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RayTracingNode {
    pub max_samples: u32,
    pub max_bounces: u32,
    pub enabled: bool,
}
impl Default for RayTracingNode {
    fn default() -> Self { Self { max_samples: 64, max_bounces: 4, enabled: false } }
}

/// Volumetric cellular data rendered via ray-marching (HOOPS Cellular Volumes).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VolumeNode {
    pub dimensions: [u32; 3],
    pub texture_path: String,
    pub density_scale: f32,
    pub color_map: [[f32; 4]; 4],
}
impl Default for VolumeNode {
    fn default() -> Self {
        Self {
            dimensions: [64, 64, 64],
            texture_path: String::new(),
            density_scale: 1.0,
            color_map: [[0.0; 4]; 4],
        }
    }
}

/// Out-of-core point cloud reference (HOOPS OOC PointCloud equivalent).
///
/// Also hosts an optional CPU particle emitter (Three.js Points analogue):
/// when `emitter` is set, `particles` is simulated each frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PointCloudNode {
    pub file_path: String,
    pub max_visible_points: u32,
    pub point_size: f32,
    pub color: [f32; 4],
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub particles: Vec<crate::particle::Particle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emitter: Option<crate::particle::ParticleEmitter>,
}
impl Default for PointCloudNode {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            max_visible_points: 100000,
            point_size: 1.0,
            color: [1.0; 4],
            particles: Vec::new(),
            emitter: None,
        }
    }
}

impl PointCloudNode {
    /// Point cloud driven by a particle emitter (Three.js Points + emitter).
    pub fn with_emitter(emitter: crate::particle::ParticleEmitter) -> Self {
        let max_visible_points = emitter.max_particles;
        Self {
            max_visible_points,
            emitter: Some(emitter),
            ..Default::default()
        }
    }
}
