use rc3d_core::math::Vec3;
use meshopt::clusterize::{build_meshlets, compute_meshlet_bounds, Meshlet};
use meshopt::VertexDataAdapter;

const MESHLET_MAX_VERTICES: usize = 64;
const MESHLET_MAX_TRIANGLES: usize = 124;
const MESHLET_CONE_WEIGHT: f32 = 0.5;

fn typed_to_bytes<T>(data: &[T]) -> &[u8] {
    let byte_count = data.len() * std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_count) }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuMeshlet {
    pub index_offset: u32,
    pub index_count: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
}

unsafe impl bytemuck::Pod for GpuMeshlet {}
unsafe impl bytemuck::Zeroable for GpuMeshlet {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuMeshletBounds {
    pub center: [f32; 3],
    pub radius: f32,
    pub cone_axis_cutoff: [f32; 4],
}

unsafe impl bytemuck::Pod for GpuMeshletBounds {}
unsafe impl bytemuck::Zeroable for GpuMeshletBounds {}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshletVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub texcoord: [f32; 2],
}

#[derive(Clone, Debug)]
pub struct MeshletData {
    pub meshlets: Vec<GpuMeshlet>,
    pub meshlet_bounds: Vec<GpuMeshletBounds>,
    pub vertices: Vec<MeshletVertex>,
    pub indices: Vec<u32>,
    pub total_meshlets: u32,
    pub total_triangles: u32,
}

pub fn build_meshlets_from_mesh(
    positions: &[Vec3],
    normals: &[Vec3],
    texcoords: &[[f32; 2]],
    indices: &[u32],
) -> MeshletData {
    let vert_count = positions.len();
    if vert_count == 0 || indices.is_empty() {
        return MeshletData {
            meshlets: Vec::new(),
            meshlet_bounds: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            total_meshlets: 0,
            total_triangles: 0,
        };
    }

    let positions_flat: Vec<[f32; 3]> = positions.iter().map(|p| p.to_array()).collect();
    let normals_flat: Vec<[f32; 3]> = if normals.len() == vert_count {
        normals.iter().map(|n| n.to_array()).collect()
    } else {
        vec![[0.0, 1.0, 0.0]; vert_count]
    };
    let default_uv = [0.0f32, 0.0];
    let texcoords_flat: Vec<[f32; 2]> = if texcoords.len() == vert_count {
        texcoords.to_vec()
    } else {
        vec![default_uv; vert_count]
    };

    let vertex_adapter = VertexDataAdapter::new(
        typed_to_bytes(&positions_flat),
        std::mem::size_of::<[f32; 3]>(),
        0,
    ).expect("Failed to create VertexDataAdapter");

    let result = build_meshlets(
        indices,
        &vertex_adapter,
        MESHLET_MAX_VERTICES,
        MESHLET_MAX_TRIANGLES,
        MESHLET_CONE_WEIGHT,
    );

    let total_meshlets = result.meshlets.len();
    let mut gpu_meshlets = Vec::with_capacity(total_meshlets);
    let mut gpu_bounds = Vec::with_capacity(total_meshlets);
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    for raw in &result.meshlets {
        let vertex_offset = raw.vertex_offset;
        let triangle_offset = raw.triangle_offset;
        let vertex_count = raw.vertex_count;
        let triangle_count = raw.triangle_count;

        let local_verts = &result.vertices[vertex_offset as usize..][..vertex_count as usize];
        let local_tris = &result.triangles[triangle_offset as usize..][..(triangle_count as usize * 3)];

        let base_vertex = all_vertices.len() as u32;
        let index_offset = all_indices.len() as u32;

        for &global_idx in local_verts {
            let gi = global_idx as usize;
            all_vertices.push(MeshletVertex {
                position: positions_flat[gi],
                normal: normals_flat[gi],
                texcoord: texcoords_flat[gi],
            });
        }

        for &local_idx in local_tris {
            all_indices.push(base_vertex + local_idx as u32);
        }

        gpu_meshlets.push(GpuMeshlet {
            index_offset,
            index_count: triangle_count * 3,
            vertex_offset: base_vertex,
            vertex_count,
        });

        let m = Meshlet {
            vertices: local_verts,
            triangles: local_tris,
        };

        let bounds = compute_meshlet_bounds(m, &vertex_adapter);

        gpu_bounds.push(GpuMeshletBounds {
            center: bounds.center,
            radius: bounds.radius,
            cone_axis_cutoff: [
                bounds.cone_axis[0],
                bounds.cone_axis[1],
                bounds.cone_axis[2],
                bounds.cone_cutoff,
            ],
        });
    }

    let total_triangles = indices.len() as u32 / 3;

    MeshletData {
        meshlets: gpu_meshlets,
        meshlet_bounds: gpu_bounds,
        vertices: all_vertices,
        indices: all_indices,
        total_meshlets: total_meshlets as u32,
        total_triangles,
    }
}
