//! Gizmo handle geometry generators.
//!
//! Pre-computes arrow, ring, and cube handle meshes for translation,
//! rotation, and scale gizmos.

use glam::Vec3;
use rc3d_render::LineVertex;

/// Axis-aligned arrow geometry: shaft (cylinder approximated as 4 lines)
/// + cone tip for the given axis direction.
pub fn translate_arrow(axis: Vec3, origin: Vec3, length: f32) -> Vec<LineVertex> {
    let shaft_len = length * 0.7;
    let cone_start = origin + axis * shaft_len;
    let cone_tip = origin + axis * length;
    let cone_r = length * 0.12;

    // Generate perpendicular vectors for shaft width
    let perp = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = axis.cross(perp).normalize() * cone_r * 0.4;
    let v = axis.cross(u).normalize() * cone_r * 0.4;

    let mut lines = Vec::new();

    let o = origin;
    let a = o + u;
    let b = o + v;
    let c = o - u;
    let d = o - v;
    let cs = cone_start;

    // Shaft (4 lines from origin-end to cone-start-end)
    for &(start, end) in &[(a, cs + u), (b, cs + v), (c, cs - u), (d, cs - v)] {
        lines.push(LineVertex { position: start.to_array() });
        lines.push(LineVertex { position: end.to_array() });
    }

    // Cone base ring (4 lines)
    let cr = cone_r;
    for i in 0..4 {
        let angle0 = i as f32 * std::f32::consts::PI * 0.5;
        let angle1 = (i + 1) as f32 * std::f32::consts::PI * 0.5;
        let p0 = cs + (u * angle0.cos() + v * angle0.sin()) * (cr / cone_r * 0.4).max(0.0);
        let p1 = cs + (u * angle1.cos() + v * angle1.sin()) * (cr / cone_r * 0.4).max(0.0);
        lines.push(LineVertex { position: p0.to_array() });
        lines.push(LineVertex { position: p1.to_array() });
    }
    // Cone tip edges
    for i in 0..4 {
        let angle = i as f32 * std::f32::consts::PI * 0.5;
        let p = cs + (u * angle.cos() + v * angle.sin()) * cr;
        lines.push(LineVertex { position: p.to_array() });
        lines.push(LineVertex { position: cone_tip.to_array() });
    }

    lines
}

/// Rotation ring: circle in the plane perpendicular to the given axis.
pub fn rotate_ring(axis: Vec3, origin: Vec3, radius: f32, segments: u32) -> Vec<LineVertex> {
    let perp = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = axis.cross(perp).normalize() * radius;
    let v = axis.cross(u).normalize() * radius;

    let mut lines = Vec::with_capacity(segments as usize * 2);
    for i in 0..segments {
        let a0 = i as f32 / segments as f32 * std::f32::consts::TAU;
        let a1 = (i + 1) as f32 / segments as f32 * std::f32::consts::TAU;
        let p0 = origin + u * a0.cos() + v * a0.sin();
        let p1 = origin + u * a1.cos() + v * a1.sin();
        lines.push(LineVertex { position: p0.to_array() });
        lines.push(LineVertex { position: p1.to_array() });
    }
    lines
}

/// Scale handle: line with a small cube at the tip.
pub fn scale_handle(axis: Vec3, origin: Vec3, length: f32) -> Vec<LineVertex> {
    let tip = origin + axis * length;
    let cube_half = length * 0.06;

    let perp = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = axis.cross(perp).normalize() * cube_half;
    let v = axis.cross(u).normalize() * cube_half;

    let mut lines = Vec::new();

    // Axis shaft
    lines.push(LineVertex { position: origin.to_array() });
    lines.push(LineVertex { position: tip.to_array() });

    // Cube at tip (12 edges)
    let corners = [
        tip + u + v + axis * cube_half,
        tip + u - v + axis * cube_half,
        tip - u - v + axis * cube_half,
        tip - u + v + axis * cube_half,
        tip + u + v - axis * cube_half,
        tip + u - v - axis * cube_half,
        tip - u - v - axis * cube_half,
        tip - u + v - axis * cube_half,
    ];
    let edges = [
        (0,1),(1,2),(2,3),(3,0), // front face
        (4,5),(5,6),(6,7),(7,4), // back face
        (0,4),(1,5),(2,6),(3,7), // connecting edges
    ];
    for (a, b) in edges {
        lines.push(LineVertex { position: corners[a].to_array() });
        lines.push(LineVertex { position: corners[b].to_array() });
    }

    lines
}
