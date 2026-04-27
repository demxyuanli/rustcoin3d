use rc3d_core::math::Vec3;

use crate::topology::TriangleMesh;

pub fn tessellate_cube(w: f32, h: f32, d: f32) -> TriangleMesh {
    let hw = w / 2.0;
    let hh = h / 2.0;
    let hd = d / 2.0;
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 0.0, 1.0], [[-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd]]),
        ([0.0, 0.0, -1.0], [[hw, -hh, -hd], [-hw, -hh, -hd], [-hw, hh, -hd], [hw, hh, -hd]]),
        ([1.0, 0.0, 0.0], [[hw, -hh, hd], [hw, -hh, -hd], [hw, hh, -hd], [hw, hh, hd]]),
        ([-1.0, 0.0, 0.0], [[-hw, -hh, -hd], [-hw, -hh, hd], [-hw, hh, hd], [-hw, hh, -hd]]),
        ([0.0, 1.0, 0.0], [[-hw, hh, hd], [hw, hh, hd], [hw, hh, -hd], [-hw, hh, -hd]]),
        ([0.0, -1.0, 0.0], [[-hw, -hh, -hd], [hw, -hh, -hd], [hw, -hh, hd], [-hw, -hh, hd]]),
    ];
    let mut positions = Vec::with_capacity(36);
    for (_normal, corners) in &faces {
        positions.push(Vec3::from(corners[0]));
        positions.push(Vec3::from(corners[1]));
        positions.push(Vec3::from(corners[2]));
        positions.push(Vec3::from(corners[0]));
        positions.push(Vec3::from(corners[2]));
        positions.push(Vec3::from(corners[3]));
    }
    TriangleMesh::from_tris(&positions)
}

pub fn tessellate_sphere(radius: f32, slices: u32, stacks: u32) -> TriangleMesh {
    let mut positions = Vec::new();
    for i in 0..stacks {
        let theta0 = std::f32::consts::PI * i as f32 / stacks as f32;
        let theta1 = std::f32::consts::PI * (i + 1) as f32 / stacks as f32;
        for j in 0..slices {
            let phi0 = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
            let phi1 = 2.0 * std::f32::consts::PI * (j + 1) as f32 / slices as f32;
            let p00 = sphere_point(radius, theta0, phi0);
            let p10 = sphere_point(radius, theta1, phi0);
            let p01 = sphere_point(radius, theta0, phi1);
            let p11 = sphere_point(radius, theta1, phi1);
            positions.extend_from_slice(&[p00, p01, p10, p01, p11, p10]);
        }
    }
    TriangleMesh::from_tris(&positions)
}

fn sphere_point(r: f32, theta: f32, phi: f32) -> Vec3 {
    let sin_t = theta.sin();
    Vec3::new(sin_t * phi.cos() * r, theta.cos() * r, sin_t * phi.sin() * r)
}

pub fn tessellate_cone(radius: f32, height: f32, segments: u32) -> TriangleMesh {
    let mut positions = Vec::new();
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
        positions.extend_from_slice(&[bl, tip, br]);
        let center = Vec3::new(0.0, -half_h, 0.0);
        positions.extend_from_slice(&[center, bl, br]);
    }
    TriangleMesh::from_tris(&positions)
}

pub fn tessellate_cylinder(radius: f32, height: f32, segments: u32) -> TriangleMesh {
    let mut positions = Vec::new();
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
        positions.extend_from_slice(&[bl, tl, br, br, tl, tr]);
        let top_center = Vec3::new(0.0, half_h, 0.0);
        positions.extend_from_slice(&[top_center, tr, tl]);
        let bot_center = Vec3::new(0.0, -half_h, 0.0);
        positions.extend_from_slice(&[bot_center, bl, br]);
    }
    TriangleMesh::from_tris(&positions)
}
