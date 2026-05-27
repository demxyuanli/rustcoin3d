//! Global mesh properties (OCC BRepGProp equivalent).
//! Volume, surface area, center of mass via divergence theorem.

use rc3d_core::math::Vec3;

#[derive(Debug, Default, Clone)]
pub struct MeshProperties {
    pub volume: f64,
    pub surface_area: f64,
    pub center_of_mass: [f64; 3],
}

/// Compute mesh properties from triangle mesh.
/// Uses signed tetrahedra method (divergence theorem).
/// Assumes closed, watertight mesh with consistent orientation.
pub fn compute_mesh_properties(vertices: &[Vec3], indices: &[i32]) -> MeshProperties {
    let mut props = MeshProperties::default();
    if vertices.is_empty() || indices.is_empty() {
        return props;
    }

    let mut vol = 0.0f64;
    let mut area = 0.0f64;
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    let mut cz = 0.0f64;

    for chunk in indices.chunks(4) {
        if chunk.len() < 3 {
            continue;
        }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }

        let a = vertices[i0];
        let b = vertices[i1];
        let c = vertices[i2];

        // Signed volume of tetrahedron (origin, a, b, c)
        let tet_vol = triple_product(a, b, c) / 6.0;
        vol += tet_vol;

        // Surface area
        let ab = b - a;
        let ac = c - a;
        area += ab.cross(ac).length() as f64 * 0.5;

        // Center of mass contribution (tetrahedron centroid)
        let tri_cx = (a.x + b.x + c.x) as f64 / 4.0;
        let tri_cy = (a.y + b.y + c.y) as f64 / 4.0;
        let tri_cz = (a.z + b.z + c.z) as f64 / 4.0;
        cx += tet_vol * tri_cx;
        cy += tet_vol * tri_cy;
        cz += tet_vol * tri_cz;
    }

    props.volume = vol.abs();
    props.surface_area = area;

    if vol.abs() > 1e-15 {
        props.center_of_mass = [cx / vol, cy / vol, cz / vol];
    }

    props
}

fn triple_product(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    // a · (b × c)
    let cross = b.cross(c);
    (a.x as f64 * cross.x as f64)
        + (a.y as f64 * cross.y as f64)
        + (a.z as f64 * cross.z as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::Vec3;

    #[test]
    fn test_unit_sphere_properties() {
        // Approximate sphere with octahedron subdivision
        let r = 1.0f32;
        let mut verts = Vec::new();
        let mut idx = Vec::new();
        // Simple octahedron approximating a sphere
        let top = Vec3::new(0.0, 0.0, r);
        let bot = Vec3::new(0.0, 0.0, -r);
        let front = Vec3::new(r, 0.0, 0.0);
        let back = Vec3::new(-r, 0.0, 0.0);
        let left = Vec3::new(0.0, r, 0.0);
        let right = Vec3::new(0.0, -r, 0.0);
        verts.extend([top, front, left, back, right, bot]);
        // 8 triangles covering the octahedron
        idx.extend([
            0,1,2,-1, 0,2,3,-1, 0,3,4,-1, 0,4,1,-1, // top hemisphere
            5,2,1,-1, 5,3,2,-1, 5,4,3,-1, 5,1,4,-1, // bottom hemisphere
        ]);
        let props = compute_mesh_properties(&verts, &idx);
        // Octahedron volume = 4/3 < sphere's 4π/3
        assert!(props.volume > 0.5 && props.volume < 5.0, "octahedron volume should be in ballpark");
        assert!(props.surface_area > 3.0 && props.surface_area < 20.0, "octahedron area should be reasonable");
    }

    #[test]
    fn test_unit_cube_properties() {
        // Unit cube: 12 triangles, 8 vertices
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ];
        // 6 faces * 2 tris each, with consistent outward orientation
        let idx: Vec<i32> = vec![
            0, 2, 1, -1, 0, 3, 2, -1, // bottom
            4, 5, 6, -1, 4, 6, 7, -1, // top
            0, 1, 5, -1, 0, 5, 4, -1, // front
            2, 3, 7, -1, 2, 7, 6, -1, // back
            0, 4, 7, -1, 0, 7, 3, -1, // left
            1, 2, 6, -1, 1, 6, 5, -1, // right
        ];
        let props = compute_mesh_properties(&verts, &idx);
        assert!(
            (props.volume - 1.0).abs() < 0.01,
            "unit cube volume should be 1.0, got {}",
            props.volume
        );
        assert!(
            (props.surface_area - 6.0).abs() < 0.1,
            "unit cube area should be ~6.0, got {}",
            props.surface_area
        );
    }
}
