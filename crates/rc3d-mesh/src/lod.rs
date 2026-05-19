use meshopt::{simplify, SimplifyOptions};
use rc3d_core::math::Vec3;

use crate::topology::TriangleMesh;

/// One LOD level: reduced geometry with target triangle count.
#[derive(Clone, Debug)]
pub struct MeshLodLevel {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub texcoords: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub triangle_count: u32,
}

/// A chain of LOD levels from full detail to lowest.
#[derive(Clone, Debug)]
pub struct MeshLodChain {
    pub levels: Vec<MeshLodLevel>,
    /// Full-resolution bounding sphere radius for screen-size estimation.
    pub bounding_radius: f32,
}

impl MeshLodChain {
    /// Select the appropriate LOD index based on distance and screen parameters.
    ///
    /// `distance` is camera-to-object distance.
    /// `screen_height` is viewport height in pixels.
    /// `fov_y` is vertical field of view in radians.
    pub fn select_lod(&self, distance: f32, screen_height: f32, fov_y: f32) -> usize {
        if self.levels.is_empty() {
            return 0;
        }
        let radius = self.bounding_radius.max(0.001);
        // Screen-space size of bounding sphere
        let screen_radius = (radius / distance) * (screen_height / (2.0 * (fov_y / 2.0).tan()));
        
        ((1.0 / screen_radius.max(0.001)) as usize).min(self.levels.len() - 1)
    }

    /// Get a reference to a specific LOD level.
    pub fn level(&self, index: usize) -> &MeshLodLevel {
        &self.levels[index.min(self.levels.len().saturating_sub(1))]
    }

    /// Number of LOD levels.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

/// Generate LOD chains for mesh data.
pub struct LodGenerator;

/// Options for LOD generation.
#[derive(Clone, Copy, Debug)]
pub struct LodOptions {
    /// Target triangle count for LOD 0 (full detail). Default: original count.
    pub lod0_triangles: Option<u32>,
    /// Reduction factor per level. Default: 0.5 (each level has half the triangles).
    pub reduction_factor: f32,
    /// Maximum number of LOD levels. Default: 4.
    pub max_levels: usize,
    /// Minimum triangle count. Default: 24 (8 tris).
    pub min_triangles: u32,
    /// Target error threshold (higher = more aggressive simplification). Default: 0.01.
    pub target_error: f32,
}

impl Default for LodOptions {
    fn default() -> Self {
        Self {
            lod0_triangles: None,
            reduction_factor: 0.5,
            max_levels: 4,
            min_triangles: 24,
            target_error: 0.01,
        }
    }
}

impl LodGenerator {
    /// Generate a LOD chain from a TriangleMesh.
    pub fn generate(mesh: &TriangleMesh, options: &LodOptions) -> MeshLodChain {
        if mesh.positions.is_empty() || mesh.tri_indices.is_empty() {
            return MeshLodChain {
                levels: vec![],
                bounding_radius: 0.0,
            };
        }

        let positions: Vec<[f32; 3]> = mesh.positions.iter().map(|p| p.to_array()).collect();
        let normals: Vec<[f32; 3]> = mesh.normals.iter().map(|n| n.to_array()).collect();
        let texcoords: Vec<[f32; 2]> = mesh.texcoords.clone();
        let indices: Vec<u32> = mesh.tri_indices.clone();
        let original_tris = indices.len() as u32 / 3;

        let bounding_radius = compute_bounding_radius(&mesh.positions);

        let mut levels = Vec::with_capacity(options.max_levels);

        // LOD 0: full detail
        levels.push(MeshLodLevel {
            vertices: positions.clone(),
            normals: normals.clone(),
            texcoords: texcoords.clone(),
            indices: indices.clone(),
            triangle_count: original_tris,
        });

        let mut current_verts = positions;
        let mut current_normals = normals;
        let mut current_uvs = texcoords;
        let mut current_indices = indices;

        for level_idx in 1..options.max_levels {
            let target_tris = ((original_tris as f32)
                * options.reduction_factor.powi(level_idx as i32)) as u32;
            let target = target_tris.max(options.min_triangles);

            if target >= current_indices.len() as u32 / 3 {
                break;
            }

            // Simplify using meshopt
            let simplified = simplify_mesh(
                &current_indices,
                &current_verts,
                &current_normals,
                &current_uvs,
                target,
                options.target_error,
            );

            levels.push(MeshLodLevel {
                triangle_count: simplified.indices.len() as u32 / 3,
                vertices: simplified.vertices.clone(),
                normals: simplified.normals.clone(),
                texcoords: simplified.texcoords.clone(),
                indices: simplified.indices.clone(),
            });

            current_verts = simplified.vertices;
            current_normals = simplified.normals;
            current_uvs = simplified.texcoords;
            current_indices = simplified.indices;
        }

        MeshLodChain {
            levels,
            bounding_radius,
        }
    }
}

struct SimplifiedMesh {
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    texcoords: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

fn simplify_mesh(
    indices: &[u32],
    vertices: &[[f32; 3]],
    _normals: &[[f32; 3]],
    _texcoords: &[[f32; 2]],
    target_triangles: u32,
    target_error: f32,
) -> SimplifiedMesh {
    let vert_count = vertices.len();
    let target_indices = (target_triangles * 3) as usize;

    let positions_bytes: &[u8] =
        bytemuck::cast_slice(vertices);

    let vertex_adapter = meshopt::VertexDataAdapter::new(
        positions_bytes,
        std::mem::size_of::<[f32; 3]>(),
        0,
    )
    .expect("Failed to create VertexDataAdapter for LOD");

    // Use meshopt simplify
    let simplified_indices = simplify(
        indices,
        &vertex_adapter,
        target_indices,
        target_error,
        SimplifyOptions::empty(),
        None,
    );

    // Rebuild vertex arrays from the simplified index set
    // meshopt::simplify remaps vertices; we need to collect used vertices
    let mut used_vertices = vec![false; vert_count];
    for &idx in &simplified_indices {
        used_vertices[idx as usize] = true;
    }

    let mut remap = vec![u32::MAX; vert_count];
    let mut new_verts = Vec::new();
    let mut new_normals = Vec::new();
    let mut new_uvs = Vec::new();

    for (old_idx, &used) in used_vertices.iter().enumerate() {
        if used {
            remap[old_idx] = new_verts.len() as u32;
            new_verts.push(vertices[old_idx]);
            new_normals.push(
                _normals.get(old_idx).copied().unwrap_or([0.0, 1.0, 0.0]),
            );
            new_uvs.push(
                _texcoords.get(old_idx).copied().unwrap_or([0.0, 0.0]),
            );
        }
    }

    let new_indices: Vec<u32> = simplified_indices
        .iter()
        .map(|&idx| remap[idx as usize])
        .collect();

    SimplifiedMesh {
        vertices: new_verts,
        normals: new_normals,
        texcoords: new_uvs,
        indices: new_indices,
    }
}

fn compute_bounding_radius(positions: &[Vec3]) -> f32 {
    if positions.is_empty() {
        return 1.0;
    }
    let center: Vec3 =
        positions.iter().sum::<Vec3>() / positions.len() as f32;
    let max_dist = positions
        .iter()
        .map(|p| p.distance(center))
        .fold(0.0f32, f32::max);
    max_dist.max(0.01)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lod_generation() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.5, 0.0, 1.0),
        ];
        let indices: Vec<u32> = vec![0, 1, 2, 0, 3, 1, 0, 2, 4, 1, 3, 4];
        let mesh = TriangleMesh::from_indexed(&positions, &indices);
        let options = LodOptions {
            lod0_triangles: None,
            reduction_factor: 0.5,
            max_levels: 3,
            min_triangles: 1,
            target_error: 0.01,
        };
        let chain = LodGenerator::generate(&mesh, &options);
        assert!(chain.level_count() >= 1);
    }

    #[test]
    fn test_lod_selection() {
        let chain = MeshLodChain {
            levels: vec![
                MeshLodLevel {
                    vertices: vec![],
                    normals: vec![],
                    texcoords: vec![],
                    indices: vec![],
                    triangle_count: 1000,
                },
                MeshLodLevel {
                    vertices: vec![],
                    normals: vec![],
                    texcoords: vec![],
                    indices: vec![],
                    triangle_count: 100,
                },
            ],
            bounding_radius: 1.0,
        };
        let near = chain.select_lod(1.0, 1080.0, std::f32::consts::FRAC_PI_4);
        let far = chain.select_lod(100.0, 1080.0, std::f32::consts::FRAC_PI_4);
        assert!(near <= far); // Closer = lower LOD index (higher detail)
    }
}
