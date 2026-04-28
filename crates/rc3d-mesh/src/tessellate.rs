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

/// UV sphere: pole bands use triangle fans; latitude quads in between.
pub fn tessellate_sphere(radius: f32, slices: u32, stacks: u32) -> TriangleMesh {
    let mut positions = Vec::new();
    let mut texcoords = Vec::new();
    if slices < 3 || stacks < 2 {
        return TriangleMesh::from_tris(&[]);
    }

    let north = Vec3::new(0.0, radius, 0.0);
    let south = Vec3::new(0.0, -radius, 0.0);
    let uv_north = [0.5, 0.0];
    let uv_south = [0.5, 1.0];

    let theta1 = std::f32::consts::PI / stacks as f32;
    for j in 0..slices {
        let phi0 = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
        let phi1 = 2.0 * std::f32::consts::PI * (j + 1) as f32 / slices as f32;
        let p0 = sphere_point(radius, theta1, phi0);
        let p1 = sphere_point(radius, theta1, phi1);
        positions.extend_from_slice(&[north, p1, p0]);
        texcoords.extend_from_slice(&[
            uv_north,
            sphere_uv(theta1, phi1),
            sphere_uv(theta1, phi0),
        ]);
    }

    for k in 1..(stacks - 1) {
        let theta0 = std::f32::consts::PI * k as f32 / stacks as f32;
        let theta1b = std::f32::consts::PI * (k + 1) as f32 / stacks as f32;
        for j in 0..slices {
            let phi0 = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
            let phi1 = 2.0 * std::f32::consts::PI * (j + 1) as f32 / slices as f32;
            let a0 = sphere_point(radius, theta0, phi0);
            let a1 = sphere_point(radius, theta0, phi1);
            let b0 = sphere_point(radius, theta1b, phi0);
            let b1 = sphere_point(radius, theta1b, phi1);
            positions.extend_from_slice(&[a0, a1, b0, a1, b1, b0]);
            let t00 = sphere_uv(theta0, phi0);
            let t10 = sphere_uv(theta0, phi1);
            let t01 = sphere_uv(theta1b, phi0);
            let t11 = sphere_uv(theta1b, phi1);
            texcoords.extend_from_slice(&[t00, t10, t01, t10, t11, t01]);
        }
    }

    let theta_last = std::f32::consts::PI * (stacks - 1) as f32 / stacks as f32;
    for j in 0..slices {
        let phi0 = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
        let phi1 = 2.0 * std::f32::consts::PI * (j + 1) as f32 / slices as f32;
        let p0 = sphere_point(radius, theta_last, phi0);
        let p1 = sphere_point(radius, theta_last, phi1);
        positions.extend_from_slice(&[south, p0, p1]);
        texcoords.extend_from_slice(&[
            uv_south,
            sphere_uv(theta_last, phi0),
            sphere_uv(theta_last, phi1),
        ]);
    }

    TriangleMesh::from_triangle_list_with_texcoords(&positions, &texcoords)
}

fn sphere_point(r: f32, theta: f32, phi: f32) -> Vec3 {
    let sin_t = theta.sin();
    Vec3::new(sin_t * phi.cos() * r, theta.cos() * r, sin_t * phi.sin() * r)
}

pub fn tessellate_cone(radius: f32, height: f32, segments: u32) -> TriangleMesh {
    let mut positions = Vec::new();
    let mut texcoords = Vec::new();
    let half_h = height / 2.0;
    for i in 0..segments {
        let theta0 = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let theta1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / segments as f32;
        let c0 = theta0.cos();
        let s0 = theta0.sin();
        let c1 = theta1.cos();
        let s1 = theta1.sin();
        let bl = Vec3::new(c0 * radius, -half_h, s0 * radius);
        let br = Vec3::new(c1 * radius, -half_h, s1 * radius);
        let tip = Vec3::new(0.0, half_h, 0.0);
        let u0 = i as f32 / segments as f32;
        let u1 = (i + 1) as f32 / segments as f32;
        positions.extend_from_slice(&[bl, tip, br]);
        texcoords.extend_from_slice(&[
            [u0, 0.0],
            [(u0 + u1) * 0.5, 1.0],
            [u1, 0.0],
        ]);
        let center = Vec3::new(0.0, -half_h, 0.0);
        positions.extend_from_slice(&[center, bl, br]);
        texcoords.extend_from_slice(&[
            [0.5, 0.5],
            [c0 * 0.5 + 0.5, s0 * 0.5 + 0.5],
            [c1 * 0.5 + 0.5, s1 * 0.5 + 0.5],
        ]);
    }
    TriangleMesh::from_triangle_list_with_texcoords(&positions, &texcoords)
}

pub fn tessellate_cylinder(radius: f32, height: f32, segments: u32) -> TriangleMesh {
    let mut positions = Vec::new();
    let mut texcoords = Vec::new();
    let half_h = height / 2.0;
    for i in 0..segments {
        let theta0 = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let theta1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / segments as f32;
        let c0 = theta0.cos();
        let s0 = theta0.sin();
        let c1 = theta1.cos();
        let s1 = theta1.sin();
        let bl = Vec3::new(c0 * radius, -half_h, s0 * radius);
        let br = Vec3::new(c1 * radius, -half_h, s1 * radius);
        let tl = Vec3::new(c0 * radius, half_h, s0 * radius);
        let tr = Vec3::new(c1 * radius, half_h, s1 * radius);
        let u0 = i as f32 / segments as f32;
        let u1 = (i + 1) as f32 / segments as f32;
        positions.extend_from_slice(&[bl, tl, br, br, tl, tr]);
        texcoords.extend_from_slice(&[
            [u0, 0.0],
            [u0, 1.0],
            [u1, 0.0],
            [u1, 0.0],
            [u0, 1.0],
            [u1, 1.0],
        ]);
        let top_center = Vec3::new(0.0, half_h, 0.0);
        positions.extend_from_slice(&[top_center, tr, tl]);
        texcoords.extend_from_slice(&[
            [0.5, 0.5],
            [c1 * 0.5 + 0.5, s1 * 0.5 + 0.5],
            [c0 * 0.5 + 0.5, s0 * 0.5 + 0.5],
        ]);
        let bot_center = Vec3::new(0.0, -half_h, 0.0);
        positions.extend_from_slice(&[bot_center, bl, br]);
        texcoords.extend_from_slice(&[
            [0.5, 0.5],
            [c0 * 0.5 + 0.5, s0 * 0.5 + 0.5],
            [c1 * 0.5 + 0.5, s1 * 0.5 + 0.5],
        ]);
    }
    TriangleMesh::from_triangle_list_with_texcoords(&positions, &texcoords)
}
