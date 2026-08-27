use rc3d_core::math::Vec3;

use crate::topology::TriangleMesh;

pub fn tessellate_cube(w: f32, h: f32, d: f32) -> TriangleMesh {
    let hw = w / 2.0;
    let hh = h / 2.0;
    let hd = d / 2.0;
    type CubeFace = ([[f32; 3]; 4], [[f32; 2]; 4]);
    let faces: [CubeFace; 6] = [
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

/// Tessellated plane in the XZ plane (facing +Y). Width along X, depth along Z.
pub fn tessellate_plane(w: f32, d: f32, w_segments: u32, d_segments: u32) -> TriangleMesh {
    let hw = w / 2.0;
    let hd = d / 2.0;
    let nx = w_segments + 1;
    let nz = d_segments + 1;
    let mut positions = Vec::with_capacity((nx * nz) as usize);
    let mut indices = Vec::new();
    for iz in 0..nz {
        for ix in 0..nx {
            let x = -hw + (ix as f32 / w_segments as f32) * w;
            let z = -hd + (iz as f32 / d_segments as f32) * d;
            positions.push(Vec3::new(x, 0.0, z));
        }
    }
    for iz in 0..d_segments {
        for ix in 0..w_segments {
            let a = iz * nx + ix;
            let b = a + 1;
            let c = a + nx;
            let d = c + 1;
            indices.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    TriangleMesh::from_indexed(&positions, &indices)
}

/// Camera-facing sprite quad in the XY plane (facing +Z).
pub fn tessellate_quad_xy(width: f32, height: f32) -> TriangleMesh {
    let hw = width * 0.5;
    let hh = height * 0.5;
    let positions = [
        Vec3::new(-hw, -hh, 0.0),
        Vec3::new(hw, -hh, 0.0),
        Vec3::new(hw, hh, 0.0),
        Vec3::new(-hw, hh, 0.0),
    ];
    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let indices = [0u32, 1, 2, 0, 2, 3];
    TriangleMesh::from_indexed_with_texcoords(&positions, &indices, &uvs)
}

/// Tessellated disc/circle in the XZ plane (facing +Y).
pub fn tessellate_circle(radius: f32, segments: u32) -> TriangleMesh {
    let n = segments.max(3);
    let mut positions = vec![Vec3::ZERO];
    let mut indices = Vec::new();
    for i in 0..n {
        let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
        positions.push(Vec3::new(radius * angle.cos(), 0.0, radius * angle.sin()));
    }
    for i in 0..n {
        let next = (i + 1) % n;
        indices.extend_from_slice(&[0, i + 1, next + 1]);
    }
    TriangleMesh::from_indexed(&positions, &indices)
}

/// Tessellated ring in the XZ plane (facing +Y).
pub fn tessellate_ring(inner_radius: f32, outer_radius: f32, segments: u32) -> TriangleMesh {
    let n = segments.max(3);
    let mut positions = Vec::with_capacity((2 * n) as usize);
    let mut indices = Vec::new();
    for i in 0..n {
        let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
        let c = angle.cos();
        let s = angle.sin();
        positions.push(Vec3::new(inner_radius * c, 0.0, inner_radius * s));
        positions.push(Vec3::new(outer_radius * c, 0.0, outer_radius * s));
    }
    for i in 0..n {
        let next = (i + 1) % n;
        let i0 = i * 2;
        let i1 = next * 2;
        indices.extend_from_slice(&[i0, i1, i1 + 1, i0, i1 + 1, i0 + 1]);
    }
    TriangleMesh::from_indexed(&positions, &indices)
}

/// Extrude a 2D polygon along the +Z axis. The profile is a flat list of XY points forming a closed loop.
pub fn tessellate_extrude(profile: &[[f32; 2]], height: f32) -> TriangleMesh {
    let n = profile.len();
    if n < 3 { return TriangleMesh::from_indexed(&[], &[]); }
    let hh = height * 0.5;
    let mut positions = Vec::with_capacity(n * 2);
    let mut indices = Vec::new();
    for p in profile { positions.push(Vec3::new(p[0], p[1], -hh)); }
    for p in profile { positions.push(Vec3::new(p[0], p[1], hh)); }
    for i in 0..n {
        let j = (i + 1) % n;
        let b0 = i as u32; let b1 = j as u32;
        let t0 = (i + n) as u32; let t1 = (j + n) as u32;
        indices.extend_from_slice(&[b0, b1, t1, b0, t1, t0]);
    }
    for i in 1..(n - 1) {
        indices.extend_from_slice(&[0, i as u32 + 1, i as u32]);
        indices.extend_from_slice(&[n as u32, n as u32 + i as u32, n as u32 + i as u32 + 1]);
    }
    TriangleMesh::from_indexed(&positions, &indices)
}

/// Revolve a 2D profile around the Y axis (lathe/revolution). The profile is XY points: x=radius, y=height.
pub fn tessellate_lathe(profile: &[[f32; 2]], segments: u32) -> TriangleMesh {
    let n = profile.len() as u32;
    let segs = segments.max(3);
    if n < 2 { return TriangleMesh::from_indexed(&[], &[]); }
    let mut positions = Vec::with_capacity((n * segs) as usize);
    let mut indices = Vec::new();
    for i in 0..segs {
        let angle = (i as f32 / segs as f32) * std::f32::consts::TAU;
        let c = angle.cos(); let s = angle.sin();
        for p in profile {
            positions.push(Vec3::new(p[0] * c, p[1], p[0] * s));
        }
    }
    for i in 0..segs {
        let next = (i + 1) % segs;
        for j in 0..n - 1 {
            let a = i * n + j;
            let b = next * n + j;
            let c = next * n + j + 1;
            let d = i * n + j + 1;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    TriangleMesh::from_indexed(&positions, &indices)
}

/// Icosahedron centered at origin with given radius.
pub fn tessellate_icosahedron(radius: f32) -> TriangleMesh {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let n = (1.0 + t * t).sqrt();
    let verts = [
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ];
    let positions: Vec<Vec3> = verts.iter()
        .map(|v| Vec3::new(v[0] / n * radius, v[1] / n * radius, v[2] / n * radius))
        .collect();
    let tris: [usize; 60] = [
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
    ];
    let indices: Vec<u32> = tris.iter().map(|&i| i as u32).collect();
    TriangleMesh::from_indexed(&positions, &indices)
}
