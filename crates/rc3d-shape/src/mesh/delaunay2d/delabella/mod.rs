//! DelaBella — Newton Apple Wrapper 2D Delaunay triangulation (pure Rust port).
//!
//! Port of the MIT-licensed [DelaBella](https://github.com/msokalski/delabella)
//! C++ library by Marcin Sokalski, based on the Newton Apple Wrapper algorithm
//! by David Sinclair (arXiv:1602.04707).
//!
//! Key advantages over Bowyer-Watson:
//! - Radial sweep-hull ordering avoids worst-case O(n²) cavity sizes
//! - Adaptive-exact predicates prevent degenerate-triangle failures
//! - Pre-allocated face storage with free-list recycling

mod predicates;
mod table;
mod hull;
mod insert;
mod constrain;

pub use table::DelaBella;
pub use constrain::constrain_edge;
pub use insert::triangulate;
pub(crate) use predicates::{adaptive_orient2d, adaptive_incircle};

/// Incremental-mode wrapper: insert points one at a time, then optionally constrain edges.
#[allow(dead_code)]
pub struct DelaBellaBuilder {
    inner: DelaBella,
}

#[allow(dead_code)]
impl DelaBellaBuilder {
    /// Create a new builder with expected point count for pre-allocation.
    pub fn new(expected_points: usize) -> Self {
        let mut inner = DelaBella::new();
        inner.reserve_faces(expected_points);
        Self { inner }
    }

    /// Add a point (u, v) with associated user data index.
    /// Returns the internal vertex index.
    pub fn add_point(&mut self, u: f64, v: f64, data: u32) -> u32 {
        self.inner.add_vert(u, v, data)
    }

    /// Triangulate all added points.
    /// Returns the number of Delaunay triangles, or -1 if degenerate.
    pub fn triangulate(self) -> Result<DelaBella, String> {
        let n = self.inner.verts.len();
        if n < 3 {
            return Err(format!("need at least 3 points, got {}", n));
        }

        let points: Vec<(f64, f64)> = self.inner.verts.iter().map(|v| (v.x, v.y)).collect();
        let orig: Vec<u32> = self.inner.verts.iter().map(|v| v.orig_idx).collect();

        // Reset — triangulate will rebuild from scratch
        let mut fresh = DelaBella::new();
        let count = insert::triangulate(&mut fresh, &points, &orig);
        if count <= 0 {
            return Err("all points are collinear or degenerate".to_string());
        }
        Ok(fresh)
    }

    /// Consume the builder and return the inner DelaBella for manual control.
    pub fn into_inner(self) -> DelaBella {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_square() {
        let mut b = DelaBellaBuilder::new(4);
        b.add_point(0.0, 0.0, 0);
        b.add_point(1.0, 0.0, 1);
        b.add_point(1.0, 1.0, 2);
        b.add_point(0.0, 1.0, 3);
        let della = b.triangulate().unwrap();
        let faces = della.delaunay_faces();
        assert_eq!(faces.len(), 2);
    }

    #[test]
    fn test_builder_triangle() {
        let mut b = DelaBellaBuilder::new(3);
        b.add_point(0.0, 0.0, 0);
        b.add_point(1.0, 0.0, 1);
        b.add_point(0.0, 1.0, 2);
        let della = b.triangulate().unwrap();
        assert_eq!(della.delaunay_faces().len(), 1);
    }
}
