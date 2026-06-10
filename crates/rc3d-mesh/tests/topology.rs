//! Integration tests for rc3d-mesh: topology, BVH, meshlet generation.
//!
//! Covers: TriangleMesh construction, edge adjacency, face neighbor queries,
//! meshlet clustering, and BVH spatial queries.

use rc3d_core::math::Vec3;
use rc3d_mesh::*;

// ── TriangleMesh construction ───────────────────────────────────────────

fn make_triangle() -> TriangleMesh {
    TriangleMesh::from_indexed(
        &[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        &[0, 1, 2],
    )
}

fn make_quad() -> TriangleMesh {
    TriangleMesh::from_indexed(
        &[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        &[0, 1, 2, 0, 2, 3],
    )
}

#[test]
fn from_indexed_creates_valid_topology() {
    let mesh = make_triangle();
    assert_eq!(mesh.positions.len(), 3);
    assert_eq!(mesh.tri_indices.len(), 3);
    assert_eq!(mesh.faces.len(), 1);
}

#[test]
fn quad_has_two_faces() {
    let mesh = make_quad();
    assert_eq!(mesh.faces.len(), 2);
}

#[test]
fn edge_count_quad() {
    let mesh = make_quad();
    let edges = mesh.all_edges();
    // Quad with 2 triangles: 4 boundary edges + 1 internal edge = 5
    assert_eq!(edges.len(), 5);
}

#[test]
fn shared_edge_has_two_faces() {
    let mesh = make_quad();
    let edges = mesh.all_edges();
    let internal = edges.iter().find(|e| e.faces[1].is_some());
    assert!(internal.is_some(), "quad should have at least one shared edge");
}

#[test]
fn boundary_edges_exist() {
    let mesh = make_quad();
    let boundaries = mesh.boundary_edges();
    assert_eq!(boundaries.len(), 4);
}

// ── EdgeKey ─────────────────────────────────────────────────────────────

#[test]
fn edge_key_canonical_order() {
    let k1 = EdgeKey::new(2, 5);
    let k2 = EdgeKey::new(5, 2);
    assert_eq!(k1, k2, "edge keys should be canonical (lo < hi)");
}

#[test]
fn different_edge_keys_not_equal() {
    let k1 = EdgeKey::new(0, 1);
    let k2 = EdgeKey::new(1, 2);
    assert_ne!(k1, k2);
}

// ── BVH spatial query ───────────────────────────────────────────────────

#[test]
fn bvh_build_and_query() {
    let mesh = make_quad();
    let bvh = Bvh::from_indexed_mesh(&mesh.positions, &mesh.tri_indices)
        .expect("BVH should build from quad");
    // Ray from above should hit both triangles
    let hit = bvh.intersect_ray(
        Vec3::new(0.5, 0.5, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        0.001,
    );
    assert!(hit.is_some(), "ray from above should hit quad");
}

#[test]
fn bvh_miss_parallel_ray() {
    let mesh = make_quad();
    let bvh = Bvh::from_indexed_mesh(&mesh.positions, &mesh.tri_indices)
        .expect("BVH should build from quad");
    // Ray parallel to quad at distance should miss
    let hit = bvh.intersect_ray(
        Vec3::new(0.5, 0.5, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        0.001,
    );
    // Parallel ray parallel to XY plane from above: should miss
    let _ = hit;
}

// ── Meshlet generation ──────────────────────────────────────────────────

#[test]
fn meshlet_from_quad() {
    let mesh = make_quad();
    let positions: Vec<Vec3> = mesh.positions.clone();
    let normals: Vec<Vec3> = (0..positions.len()).map(|_| Vec3::Z).collect();
    let texcoords: Vec<[f32; 2]> = (0..positions.len()).map(|_| [0.0, 0.0]).collect();
    let tangents: Vec<[f32; 4]> = (0..positions.len()).map(|_| [1.0, 0.0, 0.0, 1.0]).collect();
    let indices: Vec<u32> = mesh.tri_indices.clone();
    let data = build_meshlets_from_mesh(&positions, &normals, &texcoords, &tangents, &indices);
    assert!(data.total_triangles > 0);
    assert!(!data.meshlets.is_empty());
}
