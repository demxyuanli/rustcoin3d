use std::borrow::Cow;
use std::sync::Arc;

use rc3d_core::math::{Real, PVec3};
use rc3d_core::utils::spatial::SpatialIndexF64;

/// Spatial index for boundary mesh vertex deduplication.
pub type BoundaryPosIndex = SpatialIndexF64<usize>;

/// Default positional dedup tolerance for boundary pool construction.
pub use crate::tolerance::DEFAULT_MODEL_TOLERANCE as BOUNDARY_DEDUP_TOLERANCE;

/// Read-only boundary vertex pool shared across parallel mesh chunks.
#[derive(Debug, Clone)]
pub struct SharedBoundaryPool {
    pub vertices: Arc<[PVec3]>,
    pub normals: Arc<[PVec3]>,
    pub index: Arc<BoundaryPosIndex>,
    pub count: usize,
    cell_size: Real,
}

impl SharedBoundaryPool {
    pub fn new(vertices: Vec<PVec3>, normals: Vec<PVec3>, index: BoundaryPosIndex) -> Self {
        let count = vertices.len();
        let cell_size = index.cell_size();
        Self {
            vertices: Arc::from(vertices.into_boxed_slice()),
            normals: Arc::from(normals.into_boxed_slice()),
            index: Arc::new(index),
            count,
            cell_size,
        }
    }

    pub fn spawn_interior_index(&self) -> BoundaryPosIndex {
        BoundaryPosIndex::with_cell_size(self.cell_size)
    }
}

/// Contiguous vertex slice for read-only mesh algorithms (boundary + optional interior).
#[allow(dead_code)]
pub fn mesh_vertices_view<'a>(
    interior_vertices: &'a [PVec3],
    shared: Option<&'a SharedBoundaryPool>,
) -> Cow<'a, [PVec3]> {
    match shared {
        None => Cow::Borrowed(interior_vertices),
        Some(pool) if interior_vertices.is_empty() => Cow::Borrowed(&pool.vertices),
        Some(pool) => {
            let mut merged = Vec::with_capacity(pool.count + interior_vertices.len());
            merged.extend_from_slice(&pool.vertices);
            merged.extend_from_slice(interior_vertices);
            Cow::Owned(merged)
        }
    }
}

/// Find an existing boundary vertex index within tolerance, if any.
pub fn find_boundary_index(
    pt: PVec3,
    tolerance: Real,
    global_vertices: &[PVec3],
    index: &BoundaryPosIndex,
) -> Option<usize> {
    index.find_near([pt.x, pt.y, pt.z], tolerance, |i| {
        let v = global_vertices[i];
        [v.x, v.y, v.z]
    })
}

fn register_boundary_point_inner<F>(
    pt: PVec3,
    tolerance: Real,
    interior_vertices: &mut Vec<PVec3>,
    interior_normals: &mut Vec<PVec3>,
    interior_index: &mut BoundaryPosIndex,
    shared_boundary: Option<&SharedBoundaryPool>,
    normal_on_insert: F,
) -> usize
where
    F: FnOnce() -> PVec3,
{
    if let Some(pool) = shared_boundary {
        if let Some(idx) = find_boundary_index(pt, tolerance, &pool.vertices, &pool.index) {
            return idx;
        }
        // Chunk buffers already carry the shared boundary prefix at [0, pool.count).
        if let Some(idx) = find_boundary_index(pt, tolerance, interior_vertices, interior_index) {
            return idx;
        }
        let idx = interior_vertices.len();
        interior_vertices.push(pt);
        interior_normals.push(normal_on_insert());
        interior_index.insert(idx, [pt.x, pt.y, pt.z]);
        return idx;
    }

    if let Some(idx) = find_boundary_index(pt, tolerance, interior_vertices, interior_index) {
        return idx;
    }
    let idx = interior_vertices.len();
    interior_vertices.push(pt);
    interior_normals.push(normal_on_insert());
    interior_index.insert(idx, [pt.x, pt.y, pt.z]);
    idx
}

/// Register a boundary point in the shared pool (tolerance-aware dedup).
pub fn register_boundary_point(
    pt: PVec3,
    tolerance: Real,
    global_vertices: &mut Vec<PVec3>,
    global_normals: &mut Vec<PVec3>,
    index: &mut BoundaryPosIndex,
) -> usize {
    register_boundary_point_inner(
        pt,
        tolerance,
        global_vertices,
        global_normals,
        index,
        None,
        || PVec3::ZERO,
    )
}

/// Like [`register_boundary_point`] using the index cell size as tolerance.
pub fn register_boundary_point_indexed(
    pt: PVec3,
    global_vertices: &mut Vec<PVec3>,
    global_normals: &mut Vec<PVec3>,
    index: &mut BoundaryPosIndex,
) -> usize {
    let tolerance = index.cell_size();
    register_boundary_point(pt, tolerance, global_vertices, global_normals, index)
}

/// Register a boundary point, computing the normal only when inserting a new vertex.
#[allow(dead_code)]
pub fn register_boundary_point_with_normal<F>(
    pt: PVec3,
    tolerance: Real,
    global_vertices: &mut Vec<PVec3>,
    global_normals: &mut Vec<PVec3>,
    index: &mut BoundaryPosIndex,
    normal_on_insert: F,
) -> usize
where
    F: FnOnce() -> PVec3,
{
    register_boundary_point_with_normal_shared(
        pt,
        tolerance,
        global_vertices,
        global_normals,
        index,
        None,
        normal_on_insert,
    )
}

/// Layered register when tessellating over a shared read-only boundary pool.
pub fn register_boundary_point_with_normal_shared<F>(
    pt: PVec3,
    tolerance: Real,
    interior_vertices: &mut Vec<PVec3>,
    interior_normals: &mut Vec<PVec3>,
    interior_index: &mut BoundaryPosIndex,
    shared_boundary: Option<&SharedBoundaryPool>,
    normal_on_insert: F,
) -> usize
where
    F: FnOnce() -> PVec3,
{
    register_boundary_point_inner(
        pt,
        tolerance,
        interior_vertices,
        interior_normals,
        interior_index,
        shared_boundary,
        normal_on_insert,
    )
}

/// Like [`register_boundary_point_with_normal`] using the index cell size as tolerance.
#[allow(dead_code)]
pub fn register_boundary_point_with_normal_indexed<F>(
    pt: PVec3,
    global_vertices: &mut Vec<PVec3>,
    global_normals: &mut Vec<PVec3>,
    index: &mut BoundaryPosIndex,
    normal_on_insert: F,
) -> usize
where
    F: FnOnce() -> PVec3,
{
    register_boundary_point_with_normal_indexed_shared(
        pt,
        global_vertices,
        global_normals,
        index,
        None,
        normal_on_insert,
    )
}

/// Indexed layered register for parallel chunk meshing.
pub fn register_boundary_point_with_normal_indexed_shared<F>(
    pt: PVec3,
    interior_vertices: &mut Vec<PVec3>,
    interior_normals: &mut Vec<PVec3>,
    interior_index: &mut BoundaryPosIndex,
    shared_boundary: Option<&SharedBoundaryPool>,
    normal_on_insert: F,
) -> usize
where
    F: FnOnce() -> PVec3,
{
    let tolerance = interior_index.cell_size();
    register_boundary_point_with_normal_shared(
        pt,
        tolerance,
        interior_vertices,
        interior_normals,
        interior_index,
        shared_boundary,
        normal_on_insert,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_boundary_point_merges_within_tolerance() {
        let mut verts = Vec::new();
        let mut norms = Vec::new();
        let mut index = BoundaryPosIndex::with_cell_size(BOUNDARY_DEDUP_TOLERANCE);
        let a = register_boundary_point(
            PVec3::new(1.0, 0.0, 0.0),
            BOUNDARY_DEDUP_TOLERANCE,
            &mut verts,
            &mut norms,
            &mut index,
        );
        let b = register_boundary_point(
            PVec3::new(1.0 + 5e-5, 0.0, 0.0),
            BOUNDARY_DEDUP_TOLERANCE,
            &mut verts,
            &mut norms,
            &mut index,
        );
        assert_eq!(a, b);
        assert_eq!(verts.len(), 1);
    }

    #[test]
    fn register_boundary_point_keeps_distinct_beyond_tolerance() {
        let mut verts = Vec::new();
        let mut norms = Vec::new();
        let mut index = BoundaryPosIndex::with_cell_size(1e-5);
        let a = register_boundary_point(
            PVec3::ZERO,
            1e-5,
            &mut verts,
            &mut norms,
            &mut index,
        );
        let b = register_boundary_point(
            PVec3::new(1e-4, 0.0, 0.0),
            1e-5,
            &mut verts,
            &mut norms,
            &mut index,
        );
        assert_ne!(a, b);
        assert_eq!(verts.len(), 2);
    }
}
