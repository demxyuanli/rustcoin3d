//! Spatial index for circumcircles (OCC BRepMesh_CircleTool equivalent).
//!
//! Uses a uniform grid cell filter for O(1) lookup of circumcircles
//! that potentially contain a query point.

use super::geom::Point2d;
use super::half_edge::TriIdx;

/// A circumcircle stored in the spatial index.
#[derive(Clone, Copy, Debug)]
struct CircleEntry {
    center_x: f64,
    center_y: f64,
    r_sq: f64,
    tri: TriIdx,
    alive: bool,
}

/// Cell-based spatial index for circumcircle queries.
/// Maps grid cells to lists of circumcircle indices whose bounding boxes
/// overlap that cell.
pub struct CircleIndex {
    circles: Vec<CircleEntry>,
    free_slots: Vec<usize>,
    /// Grid cell size.
    cell_size: f64,
    /// Grid origin.
    origin_x: f64,
    origin_y: f64,
    /// Number of cells in each dimension.
    cells_u: usize,
    cells_v: usize,
    /// Grid cells: each cell contains indices into circles[].
    cells: Vec<Vec<usize>>,
}

impl CircleIndex {
    /// Create an empty index. Call `set_bounds` before first query.
    pub fn new() -> Self {
        Self {
            circles: Vec::new(),
            free_slots: Vec::new(),
            cell_size: 1.0,
            origin_x: 0.0,
            origin_y: 0.0,
            cells_u: 1,
            cells_v: 1,
            cells: vec![Vec::new()],
        }
    }

    /// Configure grid dimensions from bounding box and point count.
    pub fn set_bounds(&mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64, n_points: usize) {
        let dx = (max_x - min_x).max(1e-10);
        let dy = (max_y - min_y).max(1e-10);

        // Target ~5-10 points per cell
        let target_cells = (n_points as f64 / 8.0).sqrt().max(2.0) as usize;
        let cell_size = (dx / target_cells as f64).max(dy / target_cells as f64).max(1e-10);

        self.cell_size = cell_size;
        self.origin_x = min_x;
        self.origin_y = min_y;
        self.cells_u = ((dx / cell_size).ceil() as usize).max(1);
        self.cells_v = ((dy / cell_size).ceil() as usize).max(1);
        self.cells.clear();
        self.cells.resize(self.cells_u * self.cells_v, Vec::new());
    }

    /// Add a circumcircle to the index. Returns an index for later removal.
    pub fn insert(&mut self, tri: TriIdx, center: Point2d, r_sq: f64) -> usize {
        let radius = r_sq.sqrt();
        let entry = CircleEntry {
            center_x: center.x,
            center_y: center.y,
            r_sq,
            tri,
            alive: true,
        };

        let slot = if let Some(slot) = self.free_slots.pop() {
            self.circles[slot] = entry;
            slot
        } else {
            let slot = self.circles.len();
            self.circles.push(entry);
            slot
        };

        // Insert into all overlapping cells (capped to avoid O(R^2) blow-up on huge circles).
        const MAX_CELLS_PER_AXIS: usize = 32;
        let min_cx = self.cell_x(center.x - radius);
        let mut max_cx = self.cell_x(center.x + radius);
        let min_cy = self.cell_y(center.y - radius);
        let mut max_cy = self.cell_y(center.y + radius);
        if max_cx.saturating_sub(min_cx) > MAX_CELLS_PER_AXIS {
            max_cx = min_cx + MAX_CELLS_PER_AXIS;
        }
        if max_cy.saturating_sub(min_cy) > MAX_CELLS_PER_AXIS {
            max_cy = min_cy + MAX_CELLS_PER_AXIS;
        }

        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(cell_idx) = self.cell_index(cx, cy) {
                    self.cells[cell_idx].push(slot);
                }
            }
        }

        slot
    }

    /// Remove a circumcircle by its slot index.
    pub fn remove(&mut self, slot: usize) {
        if slot < self.circles.len() && self.circles[slot].alive {
            self.circles[slot].alive = false;
            self.free_slots.push(slot);
        }
    }

    /// Find all triangles whose circumcircle contains the given point.
    /// Returns triangle indices.
    pub fn query_containing(&self, p: Point2d) -> Vec<TriIdx> {
        let cx = self.cell_x(p.x);
        let cy = self.cell_y(p.y);

        let mut result = Vec::new();
        if let Some(cell_idx) = self.cell_index(cx, cy) {
            for &slot in &self.cells[cell_idx] {
                let c = &self.circles[slot];
                if !c.alive {
                    continue;
                }
                let dx = p.x - c.center_x;
                let dy = p.y - c.center_y;
                if dx * dx + dy * dy <= c.r_sq {
                    result.push(c.tri);
                }
            }
        }
        result
    }

    #[inline]
    fn cell_x(&self, x: f64) -> usize {
        let cx = ((x - self.origin_x) / self.cell_size) as isize;
        cx.max(0) as usize
    }

    #[inline]
    fn cell_y(&self, y: f64) -> usize {
        let cy = ((y - self.origin_y) / self.cell_size) as isize;
        cy.max(0) as usize
    }

    #[inline]
    fn cell_index(&self, cx: usize, cy: usize) -> Option<usize> {
        if cx < self.cells_u && cy < self.cells_v {
            Some(cy * self.cells_u + cx)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_index_basic() {
        let mut idx = CircleIndex::new();
        idx.set_bounds(0.0, 0.0, 10.0, 10.0, 100);

        let tri0: TriIdx = 0;
        let center = Point2d::new(5.0, 5.0);
        let r_sq = 2.0;
        let slot = idx.insert(tri0, center, r_sq);

        let inside = Point2d::new(5.0, 5.5);
        let outside = Point2d::new(9.0, 9.0);

        let hits = idx.query_containing(inside);
        assert!(hits.contains(&tri0));

        let hits = idx.query_containing(outside);
        assert!(!hits.contains(&tri0));

        idx.remove(slot);
        let hits = idx.query_containing(inside);
        assert!(!hits.contains(&tri0));
    }
}
