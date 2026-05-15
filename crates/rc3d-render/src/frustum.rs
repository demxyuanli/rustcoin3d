use rc3d_core::Aabb;
use rc3d_core::math::{Mat4, Vec3, Vec4};

#[derive(Clone, Debug)]
pub struct Frustum {
    planes: [(Vec3, f32); 6],
}

impl Frustum {
    /// Build frustum from a world→clip view-projection matrix.
    ///
    /// `depth_reversed_z` must match the projection matrix: `false` for standard
    /// wgpu (ndc_z ∈ [0,1], near→0), `true` for reverse-Z (ndc_z ∈ [1,0], near→1).
    pub fn from_view_projection(vp: Mat4, depth_reversed_z: bool) -> Self {
        let cols = vp.to_cols_array();
        // rows of the VP matrix
        let r0 = Vec4::new(cols[0], cols[4], cols[8], cols[12]);
        let r1 = Vec4::new(cols[1], cols[5], cols[9], cols[13]);
        let r2 = Vec4::new(cols[2], cols[6], cols[10], cols[14]);
        let r3 = Vec4::new(cols[3], cols[7], cols[11], cols[15]);
        let make_plane = |p: Vec4| {
            let n = p.truncate();
            let len = n.length();
            if len > 0.0 {
                (n / len, p.w / len)
            } else {
                (n, p.w)
            }
        };
        let (near_plane, far_plane) = if depth_reversed_z {
            // Reverse-Z: near at z_clip=w (r2 - r3), far at z_clip=0 (r2)
            (make_plane(r2 - r3), make_plane(r2))
        } else {
            // Forward-Z: near at z_clip=0 (r2), far at z_clip=w (r3 - r2)
            (make_plane(r2), make_plane(r3 - r2))
        };
        Self {
            planes: [
                make_plane(r3 + r0), // left
                make_plane(r3 - r0), // right
                make_plane(r3 + r1), // bottom
                make_plane(r3 - r1), // top
                near_plane,
                far_plane,
            ],
        }
    }

    /// Return frustum planes as `[[f32; 4]; 6]` for GPU upload.
    pub fn plane_array(&self) -> [[f32; 4]; 6] {
        let mut out = [[0.0f32; 4]; 6];
        for (i, (n, d)) in self.planes.iter().enumerate() {
            out[i] = [n.x, n.y, n.z, *d];
        }
        out
    }

    /// Returns true if the AABB is at least partially inside the frustum.
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        for (n, d) in &self.planes {
            let px = if n.x >= 0.0 { aabb.max.x } else { aabb.min.x };
            let py = if n.y >= 0.0 { aabb.max.y } else { aabb.min.y };
            let pz = if n.z >= 0.0 { aabb.max.z } else { aabb.min.z };
            let p = Vec3::new(px, py, pz);
            if n.dot(p) + *d < 0.0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    fn aabb_at(min: [f32; 3], max: [f32; 3]) -> Aabb {
        let min = Vec3::from_array(min);
        let max = Vec3::from_array(max);
        Aabb::from_point(min).union(&Aabb::from_point(max))
    }

    /// Forward-Z perspective: near=1, far=100, fov=90°, aspect=1.
    fn forward_z_vp() -> Mat4 {
        let proj = Mat4::perspective_rh(90.0f32.to_radians(), 1.0, 1.0, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        proj * view
    }

    #[test]
    fn forward_z_aabb_in_front_passes() {
        let f = Frustum::from_view_projection(forward_z_vp(), false);
        // Camera at z=5 looking at z=0, near=1. AABB at z=2 (3 units in front of camera).
        assert!(f.intersects_aabb(&aabb_at([-1.0, -1.0, 2.0], [1.0, 1.0, 3.0])));
    }

    #[test]
    fn forward_z_aabb_behind_near_rejected() {
        let f = Frustum::from_view_projection(forward_z_vp(), false);
        // Camera at z=5, near=1 → near plane at world z=4. AABB at z=4.6 discarded.
        assert!(!f.intersects_aabb(&aabb_at([-0.1, -0.1, 4.6], [0.1, 0.1, 4.7])));
    }

    #[test]
    fn forward_and_reverse_use_different_near() {
        let vp = forward_z_vp();
        let fwd = Frustum::from_view_projection(vp, false);
        let rev = Frustum::from_view_projection(vp, true);
        assert_ne!(fwd.planes[4], rev.planes[4]);
    }
}
