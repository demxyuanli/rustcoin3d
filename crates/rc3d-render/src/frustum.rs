use rc3d_core::Aabb;
use rc3d_core::math::{Mat4, Vec3, Vec4};

#[derive(Clone, Debug)]
pub struct Frustum {
    planes: [(Vec3, f32); 6],
}

impl Frustum {
    pub fn from_view_projection(vp: Mat4) -> Self {
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
        Self {
            planes: [
                make_plane(r3 + r0), // left
                make_plane(r3 - r0), // right
                make_plane(r3 + r1), // bottom
                make_plane(r3 - r1), // top
                make_plane(r3 + r2), // near
                make_plane(r3 - r2), // far
            ],
        }
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
