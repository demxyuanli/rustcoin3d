use rc3d_actions::Aabb;
use rc3d_core::math::{Mat4, Vec4};

#[derive(Clone, Debug)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_view_projection(vp: Mat4) -> Self {
        let cols = vp.to_cols_array();
        // rows of the VP matrix
        let r0 = Vec4::new(cols[0], cols[4], cols[8], cols[12]);
        let r1 = Vec4::new(cols[1], cols[5], cols[9], cols[13]);
        let r2 = Vec4::new(cols[2], cols[6], cols[10], cols[14]);
        let r3 = Vec4::new(cols[3], cols[7], cols[11], cols[15]);
        Self {
            planes: [
                (r3 + r0).normalize(), // left
                (r3 - r0).normalize(), // right
                (r3 + r1).normalize(), // bottom
                (r3 - r1).normalize(), // top
                (r3 + r2).normalize(), // near
                (r3 - r2).normalize(), // far
            ],
        }
    }

    /// Returns true if the AABB is at least partially inside the frustum.
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        let corners = [
            Vec4::new(aabb.min.x, aabb.min.y, aabb.min.z, 1.0),
            Vec4::new(aabb.max.x, aabb.min.y, aabb.min.z, 1.0),
            Vec4::new(aabb.min.x, aabb.max.y, aabb.min.z, 1.0),
            Vec4::new(aabb.max.x, aabb.max.y, aabb.min.z, 1.0),
            Vec4::new(aabb.min.x, aabb.min.y, aabb.max.z, 1.0),
            Vec4::new(aabb.max.x, aabb.min.y, aabb.max.z, 1.0),
            Vec4::new(aabb.min.x, aabb.max.y, aabb.max.z, 1.0),
            Vec4::new(aabb.max.x, aabb.max.y, aabb.max.z, 1.0),
        ];
        for plane in &self.planes {
            let all_outside = corners.iter().all(|c| plane.dot(*c) < 0.0);
            if all_outside {
                return false;
            }
        }
        true
    }
}
