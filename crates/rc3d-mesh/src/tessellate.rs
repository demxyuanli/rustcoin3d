use rc3d_core::math::Vec3;

use crate::topology::TriangleMesh;

pub fn tessellate_cube(w: f32, h: f32, d: f32) -> TriangleMesh {
    let hw = w / 2.0;
    let hh = h / 2.0;
    let hd = d / 2.0;
    let faces: [([[f32; 3]; 4], [[f32; 2]; 4]); 6] = [
        (
            [
                [-hw, -hh, hd],
                [hw, -hh, hd],
                [hw, hh, hd],
                [-hw, hh, hd],
            ],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ),
        (
            [
                [hw, -hh, -hd],
                [-hw, -hh, -hd],
                [-hw, hh, -hd],
                [hw, hh, -hd],
            ],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ),
        (
            [
                [hw, -hh, hd],
                [hw, -hh, -hd],
                [hw, hh, -hd],
                [hw, hh, hd],
            ],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ),
        (
            [
                [-hw, -hh, -hd],
                [-hw, -hh, hd],
                [-hw, hh, hd],
                [-hw, hh, -hd],
            ],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ),
        (
            [
                [-hw, hh, hd],
                [hw, hh, hd],
                [hw, hh, -hd],
                [-hw, hh, -hd],
            ],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ),
        (
            [
                [-hw, -hh, -hd],
                [hw, -hh, -hd],
                [hw, -hh, hd],
                [-hw, -hh, hd],
            ],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ),
    ];
    let mut positions = Vec::with_capacity(36);
    let mut texcoords = Vec::with_capacity(36);
    for (corners, uvs) in &faces {
        for k in [0usize, 1, 2, 0, 2, 3] {
            positions.push(Vec3::from(corners[k]));
            texcoords.push(uvs[k]);
        }
    }
    TriangleMesh::from_triangle_list_with_texcoords(&positions, &texcoords)
}

fn sphere_uv(theta: f32, phi: f32) -> [f32; 2] {
    let u = phi / (2.0 * std::f32::consts::PI);
    let v = 1.0 - theta / std::f32::consts::PI;
    [u, v]
}

/// UV sphere: shared latitude rings (smooth shading); seam kept via distinct vertices per slice.
pub fn tessellate_sphere(radius: f32, slices: u32, stacks: u32) -> TriangleMesh {
    if slices < 3 || stacks < 2 {
        return TriangleMesh::from_tris(&[]);
    }

    let ring_count = stacks - 1;
    let mut positions = Vec::with_capacity(2 + (ring_count as usize) * (slices as usize));
    let mut texcoords = Vec::with_capacity(positions.capacity());
    let mut indices = Vec::new();

    positions.push(Vec3::new(0.0, radius, 0.0));
    texcoords.push([0.5, 0.0]);

    for r in 0..ring_count {
        let theta = std::f32::consts::PI * (r + 1) as f32 / stacks as f32;
        for j in 0..slices {
            let phi = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
            positions.push(sphere_point(radius, theta, phi));
            texcoords.push(sphere_uv(theta, phi));
        }
    }

    positions.push(Vec3::new(0.0, -radius, 0.0));
    texcoords.push([0.5, 1.0]);

    let south_idx = 1 + ring_count * slices;

    for j in 0..slices {
        let jp = (j + 1) % slices;
        indices.extend_from_slice(&[0, 1 + jp, 1 + j]);
    }

    if stacks > 2 {
        for r in 0..(stacks - 2) {
            for j in 0..slices {
                let jp = (j + 1) % slices;
                let a0 = 1 + r * slices + j;
                let a1 = 1 + r * slices + jp;
                let b0 = 1 + (r + 1) * slices + j;
                let b1 = 1 + (r + 1) * slices + jp;
                indices.extend_from_slice(&[a0, a1, b0, a1, b1, b0]);
            }
        }
    }

    let last_ring = stacks - 2;
    for j in 0..slices {
        let jp = (j + 1) % slices;
        let a0 = 1 + last_ring * slices + j;
        let a1 = 1 + last_ring * slices + jp;
        indices.extend_from_slice(&[south_idx, a0, a1]);
    }

    TriangleMesh::from_indexed_with_texcoords(&positions, &indices, &texcoords)
}

fn sphere_point(r: f32, theta: f32, phi: f32) -> Vec3 {
    let sin_t = theta.sin();
    Vec3::new(sin_t * phi.cos() * r, theta.cos() * r, sin_t * phi.sin() * r)
}

pub fn tessellate_cone(radius: f32, height: f32, segments: u32) -> TriangleMesh {
    if segments < 3 {
        return TriangleMesh::from_tris(&[]);
    }
    let half_h = height / 2.0;
    let mut positions = Vec::new();
    let mut texcoords = Vec::new();
    let mut indices = Vec::new();

    for j in 0..segments {
        let th = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
        positions.push(Vec3::new(th.cos() * radius, -half_h, th.sin() * radius));
        texcoords.push([j as f32 / segments as f32, 0.0]);
    }
    let tip_i = segments;
    positions.push(Vec3::new(0.0, half_h, 0.0));
    texcoords.push([0.5, 1.0]);

    for j in 0..segments {
        let jp = (j + 1) % segments;
        indices.extend_from_slice(&[tip_i, jp, j]);
    }

    let base_center = positions.len() as u32;
    positions.push(Vec3::new(0.0, -half_h, 0.0));
    texcoords.push([0.5, 0.5]);
    let rim_start = positions.len() as u32;
    for j in 0..segments {
        let th = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
        let c = th.cos();
        let s = th.sin();
        positions.push(Vec3::new(c * radius, -half_h, s * radius));
        texcoords.push([c * 0.5 + 0.5, s * 0.5 + 0.5]);
    }
    for j in 0..segments {
        let jp = (j + 1) % segments;
        indices.extend_from_slice(&[base_center, rim_start + j, rim_start + jp]);
    }

    TriangleMesh::from_indexed_with_texcoords(&positions, &indices, &texcoords)
}

pub fn tessellate_cylinder(radius: f32, height: f32, segments: u32) -> TriangleMesh {
    if segments < 3 {
        return TriangleMesh::from_tris(&[]);
    }
    let half_h = height / 2.0;
    let mut positions = Vec::new();
    let mut texcoords = Vec::new();
    let mut indices = Vec::new();

    for j in 0..segments {
        let th = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
        positions.push(Vec3::new(th.cos() * radius, -half_h, th.sin() * radius));
        texcoords.push([j as f32 / segments as f32, 0.0]);
    }
    for j in 0..segments {
        let th = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
        positions.push(Vec3::new(th.cos() * radius, half_h, th.sin() * radius));
        texcoords.push([j as f32 / segments as f32, 1.0]);
    }

    for j in 0..segments {
        let jp = (j + 1) % segments;
        let b0 = j;
        let b1 = jp;
        let t0 = segments + j;
        let t1 = segments + jp;
        indices.extend_from_slice(&[b0, t0, b1, b1, t0, t1]);
    }

    let top_c = positions.len() as u32;
    positions.push(Vec3::new(0.0, half_h, 0.0));
    texcoords.push([0.5, 0.5]);
    let top_rim = positions.len() as u32;
    for j in 0..segments {
        let th = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
        let c = th.cos();
        let s = th.sin();
        positions.push(Vec3::new(c * radius, half_h, s * radius));
        texcoords.push([c * 0.5 + 0.5, s * 0.5 + 0.5]);
    }
    for j in 0..segments {
        let jp = (j + 1) % segments;
        indices.extend_from_slice(&[top_c, top_rim + jp, top_rim + j]);
    }

    let bot_c = positions.len() as u32;
    positions.push(Vec3::new(0.0, -half_h, 0.0));
    texcoords.push([0.5, 0.5]);
    let bot_rim = positions.len() as u32;
    for j in 0..segments {
        let th = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
        let c = th.cos();
        let s = th.sin();
        positions.push(Vec3::new(c * radius, -half_h, s * radius));
        texcoords.push([c * 0.5 + 0.5, s * 0.5 + 0.5]);
    }
    for j in 0..segments {
        let jp = (j + 1) % segments;
        indices.extend_from_slice(&[bot_c, bot_rim + j, bot_rim + jp]);
    }

    TriangleMesh::from_indexed_with_texcoords(&positions, &indices, &texcoords)
}

/// Torus: major_radius = distance from center to tube center, minor_radius = tube radius.
pub fn tessellate_torus(major_radius: f32, minor_radius: f32, major_segments: u32, minor_segments: u32) -> TriangleMesh {
    if major_segments < 3 || minor_segments < 3 {
        return TriangleMesh::from_tris(&[]);
    }
    let mut positions = Vec::new();
    let mut texcoords = Vec::new();
    let mut indices = Vec::new();

    for i in 0..major_segments {
        let theta = 2.0 * std::f32::consts::PI * i as f32 / major_segments as f32;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        for j in 0..minor_segments {
            let phi = 2.0 * std::f32::consts::PI * j as f32 / minor_segments as f32;
            let cos_p = phi.cos();
            let sin_p = phi.sin();
            let x = (major_radius + minor_radius * cos_p) * cos_t;
            let y = minor_radius * sin_p;
            let z = (major_radius + minor_radius * cos_p) * sin_t;
            positions.push(Vec3::new(x, y, z));
            texcoords.push([i as f32 / major_segments as f32, j as f32 / minor_segments as f32]);
        }
    }

    for i in 0..major_segments {
        let i_next = (i + 1) % major_segments;
        for j in 0..minor_segments {
            let j_next = (j + 1) % minor_segments;
            let a = i * minor_segments + j;
            let b = i_next * minor_segments + j;
            let c = i_next * minor_segments + j_next;
            let d = i * minor_segments + j_next;
            indices.extend_from_slice(&[a, b, d, b, c, d]);
        }
    }

    TriangleMesh::from_indexed_with_texcoords(&positions, &indices, &texcoords)
}
