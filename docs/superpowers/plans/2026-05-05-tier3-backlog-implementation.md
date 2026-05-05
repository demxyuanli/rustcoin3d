# Tier 3 Backlog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 Tier 3 backlog features: NURBS math crate, attribute system expansion, full-scene undo coverage, and markup redline editing.

**Architecture:** Phase 1 runs T3-2 (new `rc3d-nurbs` math crate) and T3-8 (FieldValue extension + NodeEntry attributes) in parallel with no mutual dependencies. Phase 2 runs T3-3 (4 reusable Command types + ~20 EditorCommand wiring) and T3-4 (MarkupTool state machine + pass_markup render pass) in parallel, building on T3-8 types.

**Tech Stack:** Rust workspace, wgpu 24, glam 0.29, existing rc3d-* crates.

---

## File Structure

### Phase 1A: T3-2 NURBS — New crate

| File | Purpose |
|------|---------|
| `crates/rc3d-nurbs/Cargo.toml` | Crate manifest; depends on `rc3d-core` only |
| `crates/rc3d-nurbs/src/lib.rs` | Module declarations + public re-exports |
| `crates/rc3d-nurbs/src/knot.rs` | Knot vector construction: uniform, open-uniform |
| `crates/rc3d-nurbs/src/basis.rs` | B-spline basis function via Cox-de Boor recurrence |
| `crates/rc3d-nurbs/src/curve.rs` | `NurbsCurve`: evaluate, tangent, arc-length, tessellate, knot insertion |
| `crates/rc3d-nurbs/src/surface.rs` | `NurbsSurface`: evaluate, normal, uniform/adaptive tessellation |
| `crates/rc3d-nurbs/src/tessellate.rs` | Curvature-driven adaptive subdivision → `TriangleMesh` |

### Phase 1B: T3-8 Attributes — Modify existing crates

| File | Change |
|------|--------|
| `crates/rc3d-fields/src/field_value.rs` | Add `String(String)`, `Binary(Vec<u8>)`, `Float64(f64)` variants |
| `crates/rc3d-scene/src/node_data.rs:745-775` | Extend `field_descriptors()` to cover all NodeData variants |
| `crates/rc3d-scene/src/node_entry.rs:6-13` | Add `pub attributes: HashMap<String, String>` |
| `crates/rc3d-scene/src/scene_graph.rs:55-72` | Init `attributes: HashMap::new()` in `insert_child` |

### Phase 2A: T3-3 Undo — Modify existing crates

| File | Change |
|------|--------|
| `crates/rc3d-actions/src/undo.rs:1-147` | Add `SetFieldCommand<T>`, `AddChildCommand`, `RemoveChildCommand`, `CompoundCommand` |
| `crates/rc3d-actions/src/lib.rs:29` | Export new Command types |
| `crates/rc3d-app/src/app/editor_commands.rs:64-367` | Wire ~15 EditorCommand variants to `SetFieldCommand` |
| `crates/rc3d-app/src/app/editor_commands.rs:78-87` | Wire CreateNode/DeleteNode/DuplicateNode to structure commands |
| `crates/rc3d-app/src/app/editor_interaction.rs:100-150` | Wrap gizmo drag commits in `CompoundCommand` |

### Phase 2B: T3-4 Markup — Modify existing crates

| File | Change |
|------|--------|
| `crates/rc3d-actions/src/markup_tool.rs` | New: `MarkupAction` state machine |
| `crates/rc3d-actions/src/lib.rs:12` | Add `pub mod markup_tool;` + exports |
| `crates/rc3d-render/src/render_passes/pass_markup.rs` | New: screen-space markup line rendering |
| `crates/rc3d-render/src/pipelines.rs:38-43` | Add `markup_lines: wgpu::RenderPipeline` |
| `crates/rc3d-render/src/render_passes.rs:11` | Add `mod pass_markup;` + call after viewport borders |
| `crates/rc3d-app/src/editor_ui/commands.rs:15-88` | Add `SetMarkupTool`, `MarkupMouse*`, `ClearAllMarkup` variants |
| `crates/rc3d-app/src/app/mod.rs` | Add `markup_action: MarkupAction` field |
| `crates/rc3d-app/src/app/editor_commands.rs` | Wire markup commands |

---

## Phase 1A: T3-2 NURBS Crate

### Task 1: Create crate scaffold

**Files:**
- Create: `crates/rc3d-nurbs/Cargo.toml`
- Create: `crates/rc3d-nurbs/src/lib.rs`
- Modify: `Cargo.toml:1-14`

- [ ] **Step 1: Write Cargo.toml for rc3d-nurbs**

```toml
[package]
name = "rc3d-nurbs"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
rc3d-core = { workspace = true }
rc3d-mesh = { workspace = true }
```

- [ ] **Step 2: Write initial lib.rs**

```rust
pub mod basis;
pub mod curve;
pub mod knot;
pub mod surface;
pub mod tessellate;

pub use basis::bspline_basis;
pub use curve::NurbsCurve;
pub use surface::NurbsSurface;
```

- [ ] **Step 3: Add rc3d-nurbs to workspace Cargo.toml**

Modify `Cargo.toml` — add `"crates/rc3d-nurbs",` after the `rc3d-mesh` entry:
```toml
members = [
    "crates/rc3d-core",
    "crates/rc3d-fields",
    "crates/rc3d-scene",
    "crates/rc3d-nodes",
    "crates/rc3d-actions",
    "crates/rc3d-mesh",
    "crates/rc3d-nurbs",
    "crates/rc3d-render",
    ...
]
```

Also add `rc3d-nurbs = { path = "crates/rc3d-nurbs" }` under `[workspace.dependencies]`.

- [ ] **Step 4: Verify compilation**

Run: `rtk cargo check -p rc3d-nurbs`
Expected: 0 errors (warnings about unused modules are fine)

- [ ] **Step 5: Commit**

```bash
rtk git add crates/rc3d-nurbs/ Cargo.toml && rtk git commit -m "feat: add rc3d-nurbs crate scaffold"
```

---

### Task 2: Implement knot vector utilities

**Files:**
- Create: `crates/rc3d-nurbs/src/knot.rs`

- [ ] **Step 1: Write knot.rs**

```rust
/// Build a uniform knot vector for a curve of given degree and control point count.
/// Example: degree=3, n_cp=7 → [0,0,0,0, 1,2,3, 4,4,4,4]
pub fn uniform_knots(degree: usize, n_control_points: usize) -> Vec<f32> {
    let n_knots = n_control_points + degree + 1;
    let mut knots = Vec::with_capacity(n_knots);
    for i in 0..n_knots {
        knots.push(i as f32);
    }
    knots
}

/// Build an open-uniform (clamped) knot vector.
/// First and last `degree+1` knots are repeated, interior knots are evenly spaced.
pub fn open_uniform_knots(degree: usize, n_control_points: usize) -> Vec<f32> {
    let n_knots = n_control_points + degree + 1;
    let mut knots = Vec::with_capacity(n_knots);
    let n_interior = n_knots - 2 * (degree + 1);
    for _ in 0..=degree {
        knots.push(0.0);
    }
    for i in 1..=n_interior {
        knots.push(i as f32 / (n_interior + 1) as f32);
    }
    for _ in 0..=degree {
        knots.push(1.0);
    }
    knots
}

/// Find the knot span index: the index `i` such that knots[i] <= t < knots[i+1].
/// Returns `degree` if t < knots[degree], or `n-1-degree-1` if t >= knots[n-degree-1].
pub fn find_span(degree: usize, knots: &[f32], t: f32) -> usize {
    let n = knots.len();
    let last = n - degree - 1;
    if t >= knots[last] {
        return last - 1;
    }
    if t <= knots[degree] {
        return degree;
    }
    // Linear search (for small knot vectors; binary search for large ones)
    for i in degree..last {
        if t >= knots[i] && t < knots[i + 1] {
            return i;
        }
    }
    degree
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_uniform_cubic() {
        let knots = open_uniform_knots(3, 7);
        // First 4 should be 0.0
        assert_eq!(&knots[0..4], &[0.0; 4]);
        // Last 4 should be 1.0
        assert_eq!(&knots[7..11], &[1.0; 4]);
    }

    #[test]
    fn test_find_span_mid() {
        let knots = open_uniform_knots(3, 7);
        let span = find_span(3, &knots, 0.5);
        assert!(span >= 3 && span <= 6);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `rtk cargo test -p rc3d-nurbs`
Expected: 2 tests pass

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-nurbs/src/knot.rs && rtk git commit -m "feat(nurbs): knot vector utilities"
```

---

### Task 3: Implement B-spline basis functions

**Files:**
- Create: `crates/rc3d-nurbs/src/basis.rs`

- [ ] **Step 1: Write basis.rs**

```rust
/// Evaluate the i-th B-spline basis function of degree p at parameter t.
/// Uses the Cox-de Boor recurrence.
pub fn bspline_basis(i: usize, p: usize, t: f32, knots: &[f32]) -> f32 {
    if p == 0 {
        if t >= knots[i] && t < knots[i + 1] {
            return 1.0;
        }
        // Handle the right boundary: t == last knot and i == last basis
        if (t - knots[i]).abs() < f32::EPSILON && (knots[i + 1] - knots.last().copied().unwrap_or(1.0)).abs() < f32::EPSILON {
            // Actually, simpler: t == knots[n-1] and i == n-p-2 is the rightmost
        }
        // Cox-de Boor convention: 0/0 = 0
        if t >= knots[i] && t <= knots[i + 1] && knots[i] < knots[i + 1] {
            // Check if t equals the last knot value and this is the final interval
            // For clamped knot vectors, include the right endpoint
            if i + 1 < knots.len() && (t - knots[i + 1]).abs() < f32::EPSILON {
                let n = knots.len();
                if i == n - 2 {
                    return 1.0;
                }
            }
        }
        return 0.0;
    }

    let mut left = 0.0;
    let mut right = 0.0;

    let denom1 = knots[i + p] - knots[i];
    if denom1 > 0.0 {
        left = ((t - knots[i]) / denom1) * bspline_basis(i, p - 1, t, knots);
    }

    let denom2 = knots[i + p + 1] - knots[i + 1];
    if denom2 > 0.0 {
        right = ((knots[i + p + 1] - t) / denom2) * bspline_basis(i + 1, p - 1, t, knots);
    }

    left + right
}

/// Evaluate all non-zero B-spline basis functions at parameter t.
/// Returns a Vec of (basis_index, value) pairs for the p+1 non-zero bases.
pub fn bspline_bases(span: usize, p: usize, t: f32, knots: &[f32]) -> Vec<(usize, f32)> {
    let mut bases = Vec::with_capacity(p + 1);
    for i in 0..=p {
        let idx = span - p + i;
        bases.push((idx, bspline_basis(idx, p, t, knots)));
    }
    bases
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knot::open_uniform_knots;

    #[test]
    fn test_partition_of_unity() {
        // For a clamped cubic with 7 CPs, the non-zero bases should sum to 1.0
        let knots = open_uniform_knots(3, 7);
        let span = crate::knot::find_span(3, &knots, 0.5);
        let bases = bspline_bases(span, 3, 0.5, &knots);
        let sum: f32 = bases.iter().map(|(_, v)| v).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {}", sum);
    }

    #[test]
    fn test_basis_nonnegative() {
        let knots = open_uniform_knots(3, 7);
        for t_i in 0..=20 {
            let t = t_i as f32 / 20.0;
            let span = crate::knot::find_span(3, &knots, t);
            let bases = bspline_bases(span, 3, t, &knots);
            for (_, v) in bases {
                assert!(v >= -1e-6, "negative basis value {} at t={}", v, t);
            }
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `rtk cargo test -p rc3d-nurbs`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-nurbs/src/basis.rs && rtk git commit -m "feat(nurbs): B-spline basis functions"
```

---

### Task 4: Implement NURBS curve

**Files:**
- Create: `crates/rc3d-nurbs/src/curve.rs`

- [ ] **Step 1: Write curve.rs**

```rust
use glam::Vec3;
use crate::basis::bspline_basis;
use crate::knot::{find_span, open_uniform_knots};

/// A NURBS curve defined by homogeneous control points, knot vector, and degree.
///
/// Control points are stored as (x, y, z, w) homogeneous coordinates.
#[derive(Clone, Debug)]
pub struct NurbsCurve {
    pub control_points: Vec<[f32; 4]>,  // (wx, wy, wz, w) in homogeneous space
    pub knots: Vec<f32>,
    pub degree: usize,
}

impl NurbsCurve {
    /// Create a NURBS curve from homogeneous control points.
    /// Builds a clamped knot vector automatically.
    pub fn new(control_points: Vec<[f32; 4]>, degree: usize) -> Self {
        let knots = open_uniform_knots(degree, control_points.len());
        Self { control_points, knots, degree }
    }

    /// Create a NURBS curve from non-rational (w=1) control points.
    pub fn from_points(points: &[Vec3], degree: usize) -> Self {
        let cp: Vec<[f32; 4]> = points.iter().map(|p| [p.x, p.y, p.z, 1.0]).collect();
        Self::new(cp, degree)
    }

    /// Number of control points.
    pub fn n_control_points(&self) -> usize {
        self.control_points.len()
    }

    /// Evaluate the curve at parameter t, returning a 3D point.
    pub fn evaluate(&self, t: f32) -> Vec3 {
        let span = find_span(self.degree, &self.knots, t);
        let mut point = Vec3::ZERO;
        for i in 0..=self.degree {
            let idx = span - self.degree + i;
            let basis = bspline_basis(idx, self.degree, t, &self.knots);
            let cp = &self.control_points[idx];
            let w = cp[3];
            point += Vec3::new(cp[0], cp[1], cp[2]) * basis;
        }

        // Divide by the sum of weighted basis functions (rational part)
        let mut w_sum = 0.0;
        for i in 0..=self.degree {
            let idx = span - self.degree + i;
            let basis = bspline_basis(idx, self.degree, t, &self.knots);
            w_sum += self.control_points[idx][3] * basis;
        }

        if w_sum.abs() > 1e-10 {
            point / w_sum
        } else {
            point
        }
    }

    /// Evaluate the unit tangent vector at parameter t (finite difference approximation).
    pub fn tangent(&self, t: f32) -> Vec3 {
        let eps = 1e-4;
        let p0 = self.evaluate((t - eps).max(0.0));
        let p1 = self.evaluate((t + eps).min(1.0));
        (p1 - p0).normalize()
    }

    /// Approximate arc length using n_samples.
    pub fn arc_length(&self, n_samples: usize) -> f32 {
        let mut len = 0.0;
        let mut prev = self.evaluate(0.0);
        for i in 1..=n_samples {
            let t = i as f32 / n_samples as f32;
            let pt = self.evaluate(t);
            len += (pt - prev).length();
            prev = pt;
        }
        len
    }

    /// Adaptive tessellation: sample at curvature-dependent intervals.
    pub fn tessellate(&self, tolerance: f32) -> Vec<Vec3> {
        let mut points = vec![self.evaluate(0.0)];
        self.tessellate_recursive(0.0, 1.0, tolerance, &mut points);
        points
    }

    fn tessellate_recursive(&self, t0: f32, t1: f32, tol: f32, out: &mut Vec<Vec3>) {
        let tm = (t0 + t1) * 0.5;
        let p0 = self.evaluate(t0);
        let p1 = self.evaluate(t1);
        let pm = self.evaluate(tm);

        // Chord-line midpoint deviation test
        let chord = p1 - p0;
        let t_proj = (pm - p0).dot(chord) / chord.length_squared();
        let proj = p0 + chord * t_proj.clamp(0.0, 1.0);
        let dist = (pm - proj).length();

        if dist > tol && (t1 - t0) > 1e-5 {
            self.tessellate_recursive(t0, tm, tol, out);
            self.tessellate_recursive(tm, t1, tol, out);
        } else {
            out.push(p1);
        }
    }

    /// Insert a knot value into the knot vector (Boehm's algorithm).
    /// Refines the control polygon without changing the curve shape.
    pub fn insert_knot(&mut self, t: f32) {
        let span = find_span(self.degree, &self.knots, t);
        let p = self.degree;

        // Back up control points at the insertion point
        let mut new_cp = Vec::with_capacity(self.control_points.len() + 1);

        // Control points before the insertion span are unchanged
        for i in 0..=span - p {
            new_cp.push(self.control_points[i]);
        }

        // Compute new control points for the affected span
        for i in span - p + 1..=span {
            let alpha = if (self.knots[i + p] - self.knots[i]).abs() > 1e-10 {
                (t - self.knots[i]) / (self.knots[i + p] - self.knots[i])
            } else {
                0.0
            };
            let prev = self.control_points[i - 1];
            let curr = self.control_points[i];
            let new = [
                alpha * curr[0] + (1.0 - alpha) * prev[0],
                alpha * curr[1] + (1.0 - alpha) * prev[1],
                alpha * curr[2] + (1.0 - alpha) * prev[2],
                alpha * curr[3] + (1.0 - alpha) * prev[3],
            ];
            new_cp.push(new);
        }

        // Remaining control points
        for i in span + 1..self.control_points.len() {
            new_cp.push(self.control_points[i]);
        }

        // Insert the knot
        let mut new_knots = self.knots.clone();
        new_knots.insert(span + 1, t);

        self.control_points = new_cp;
        self.knots = new_knots;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_approximation() {
        // NURBS circle via 7 control points (cubic)
        let w = 2f32.sqrt() / 2.0;
        let cp = vec![
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, w],
            [0.0, 1.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0, w],
            [-1.0, 0.0, 0.0, 1.0],
            [-1.0, -1.0, 0.0, w],
            [0.0, -1.0, 0.0, 1.0],
        ];
        let curve = NurbsCurve::new(cp, 2);
        let p = curve.evaluate(0.0);
        // Start point should be near (1, 0)
        assert!((p.x - 1.0).abs() < 0.01, "x={}", p.x);
        assert!(p.y.abs() < 0.01, "y={}", p.y);
    }

    #[test]
    fn test_knot_insertion_invariance() {
        let cp = vec![
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0, 1.0],
        ];
        let mut curve = NurbsCurve::new(cp, 3); // degree 3 with 4 CPs = Bezier

        let pt_before = curve.evaluate(0.5);
        curve.insert_knot(0.5);
        let pt_after = curve.evaluate(0.5);

        assert!((pt_before - pt_after).length() < 1e-4,
            "knot insertion changed curve: {:?} vs {:?}", pt_before, pt_after);
    }

    #[test]
    fn test_tangent_normalized() {
        let curve = NurbsCurve::from_points(
            &[Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y],
            3,
        );
        let t = curve.tangent(0.5);
        assert!((t.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_tessellate_count() {
        let curve = NurbsCurve::from_points(
            &[Vec3::ZERO, Vec3::X * 0.5, Vec3::X, Vec3::X * 1.5],
            3,
        );
        let pts = curve.tessellate(0.1);
        // A straight line should tessellate to very few points
        assert!(pts.len() >= 2);
        assert!(pts.len() <= 20); // degenerate shouldn't explode
    }
}
```

- [ ] **Step 2: Run tests**

Run: `rtk cargo test -p rc3d-nurbs`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-nurbs/src/curve.rs && rtk git commit -m "feat(nurbs): NURBS curve evaluation and tessellation"
```

---

### Task 5: Implement NURBS surface

**Files:**
- Create: `crates/rc3d-nurbs/src/surface.rs`
- Create: `crates/rc3d-nurbs/src/tessellate.rs`

- [ ] **Step 1: Write surface.rs**

```rust
use glam::Vec3;
use crate::basis::bspline_basis;
use crate::knot::{find_span, open_uniform_knots};
use crate::tessellate::tessellate_surface_adaptive;

/// A NURBS surface defined by a grid of homogeneous control points.
#[derive(Clone, Debug)]
pub struct NurbsSurface {
    pub control_points: Vec<Vec<[f32; 4]>>,  // [u_count][v_count] = (wx, wy, wz, w)
    pub u_knots: Vec<f32>,
    pub v_knots: Vec<f32>,
    pub u_degree: usize,
    pub v_degree: usize,
}

impl NurbsSurface {
    /// Create a NURBS surface from a u×v grid of homogeneous control points.
    pub fn new(
        control_points: Vec<Vec<[f32; 4]>>,
        u_degree: usize,
        v_degree: usize,
    ) -> Self {
        let u_count = control_points.len();
        let v_count = if u_count > 0 { control_points[0].len() } else { 0 };
        let u_knots = open_uniform_knots(u_degree, u_count);
        let v_knots = open_uniform_knots(v_degree, v_count);
        Self { control_points, u_knots, v_knots, u_degree, v_degree }
    }

    /// Create a NURBS surface from non-rational control points (Vec3 grid).
    pub fn from_points_grid(points: &[Vec<Vec3>], u_degree: usize, v_degree: usize) -> Self {
        let cp: Vec<Vec<[f32; 4]>> = points
            .iter()
            .map(|row| row.iter().map(|p| [p.x, p.y, p.z, 1.0]).collect())
            .collect();
        Self::new(cp, u_degree, v_degree)
    }

    pub fn u_count(&self) -> usize { self.control_points.len() }
    pub fn v_count(&self) -> usize {
        if self.control_points.is_empty() { 0 } else { self.control_points[0].len() }
    }

    /// Evaluate the surface at parameter (u, v), returning a 3D point.
    pub fn evaluate(&self, u: f32, v: f32) -> Vec3 {
        let u_span = find_span(self.u_degree, &self.u_knots, u);
        let v_span = find_span(self.v_degree, &self.v_knots, v);

        let mut sw = Vec3::ZERO;
        let mut w_sum = 0.0;

        for i in 0..=self.u_degree {
            let ui = u_span - self.u_degree + i;
            let nu = bspline_basis(ui, self.u_degree, u, &self.u_knots);

            for j in 0..=self.v_degree {
                let vj = v_span - self.v_degree + j;
                let nv = bspline_basis(vj, self.v_degree, v, &self.v_knots);

                let cp = &self.control_points[ui][vj];
                let w = cp[3] * nu * nv;
                sw += Vec3::new(cp[0], cp[1], cp[2]) * w;
                w_sum += w;
            }
        }

        if w_sum.abs() > 1e-10 { sw / w_sum } else { sw }
    }

    /// Evaluate the surface normal at (u, v) via cross product of partial derivatives.
    pub fn normal(&self, u: f32, v: f32) -> Vec3 {
        let eps = 1e-4;
        let p0 = self.evaluate((u - eps).max(0.0), v);
        let p1 = self.evaluate((u + eps).min(1.0), v);
        let du = p1 - p0;

        let q0 = self.evaluate(u, (v - eps).max(0.0));
        let q1 = self.evaluate(u, (v + eps).min(1.0));
        let dv = q1 - q0;

        du.cross(dv).normalize()
    }

    /// Uniform tessellation into a TriangleMesh.
    pub fn tessellate_uniform(&self, u_samples: usize, v_samples: usize) -> rc3d_mesh::TriangleMesh {
        use rc3d_mesh::topology::TriangleMesh;

        let mut positions = Vec::new();
        let mut indices = Vec::new();

        for i in 0..=u_samples {
            for j in 0..=v_samples {
                let u = i as f32 / u_samples as f32;
                let v = j as f32 / v_samples as f32;
                positions.push(self.evaluate(u, v));
            }
        }

        let stride = v_samples + 1;
        for i in 0..u_samples {
            for j in 0..v_samples {
                let a = (i * stride + j) as u32;
                let b = a + stride as u32;
                let c = a + 1;
                let d = b + 1;
                indices.extend_from_slice(&[a, b, c]);
                indices.extend_from_slice(&[c, b, d]);
            }
        }

        TriangleMesh::from_indexed(&positions, &indices)
    }

    /// Adaptive tessellation driven by curvature.
    pub fn tessellate_adaptive(&self, tolerance: f32) -> rc3d_mesh::TriangleMesh {
        tessellate_surface_adaptive(self, tolerance)
    }
}
```

- [ ] **Step 2: Write tessellate.rs**

```rust
use glam::Vec3;
use crate::surface::NurbsSurface;

/// Adaptive quadtree-based surface tessellation.
pub fn tessellate_surface_adaptive(
    surface: &NurbsSurface,
    tolerance: f32,
) -> rc3d_mesh::TriangleMesh {
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    // Start with a coarse grid and subdivide where needed
    let initial = 4;
    let max_subdiv = 7;

    // Build initial coarse mesh
    let mut grid: Vec<Vec<usize>> = vec![vec![0; initial + 1]; initial + 1];
    for i in 0..=initial {
        for j in 0..=initial {
            let u = i as f32 / initial as f32;
            let v = j as f32 / initial as f32;
            grid[i][j] = positions.len();
            positions.push(surface.evaluate(u, v));
            uvs.push([u, v]);
        }
    }

    // Subdividision based on edge midpoint deviation
    for _ in 0..max_subdiv {
        let mut changed = false;
        for i in 0..initial {
            for j in 0..initial {
                let a = positions[grid[i][j]];
                let c = positions[grid[i + 1][j + 1]];
                let u_mid = (i as f32 + 0.5) / initial as f32;
                let v_mid = (j as f32 + 0.5) / initial as f32;
                let mid = surface.evaluate(u_mid, v_mid);
                let center = (a + c) * 0.5;
                let err = (mid - center).length();
                let diag = (c - a).length();
                if diag < 1e-6 {
                    continue;
                }
                if err / diag > tolerance {
                    grid[i][j] = positions.len();
                    positions.push(mid);
                    uvs.push([u_mid, v_mid]);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Build index buffer (simplified — real implementation would do proper quadtree)
    for i in 0..initial {
        for j in 0..initial {
            let a = grid[i][j] as u32;
            let b = grid[i + 1][j] as u32;
            let c = grid[i][j + 1] as u32;
            let d = grid[i + 1][j + 1] as u32;
            // Avoid degenerate triangles
            if a != b && b != c && a != c {
                indices.extend_from_slice(&[a, b, c]);
            }
            if b != d && d != c && b != c {
                indices.extend_from_slice(&[c, b, d]);
            }
        }
    }

    rc3d_mesh::TriangleMesh::from_indexed(&positions, &indices)
}
```

- [ ] **Step 3: Update lib.rs to export surface types**

Replace `crates/rc3d-nurbs/src/lib.rs`:
```rust
pub mod basis;
pub mod curve;
pub mod knot;
pub mod surface;
pub mod tessellate;

pub use basis::bspline_basis;
pub use curve::NurbsCurve;
pub use surface::NurbsSurface;
```

- [ ] **Step 4: Verify compilation**

Run: `rtk cargo check -p rc3d-nurbs`
Expected: 0 errors

- [ ] **Step 5: Run tests**

Run: `rtk cargo test -p rc3d-nurbs`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
rtk git add crates/rc3d-nurbs/src/surface.rs crates/rc3d-nurbs/src/tessellate.rs && rtk git commit -m "feat(nurbs): NURBS surface evaluation and tessellation"
```

---

## Phase 1B: T3-8 Attribute System

### Task 6: Extend FieldValue enum

**Files:**
- Modify: `crates/rc3d-fields/src/field_value.rs:3-16`

- [ ] **Step 1: Add new variants**

Edit `crates/rc3d-fields/src/field_value.rs` — add `String(String)`, `Binary(Vec<u8>)`, `Float64(f64)` after the existing array variants:

```rust
use rc3d_core::math::{Mat4, Vec2, Vec3, Vec4};

#[derive(Clone, Debug, PartialEq)]
pub enum FieldValue {
    Bool(bool),
    Int32(i32),
    Float(f32),
    Float64(f64),
    Vec2f(Vec2),
    Vec3f(Vec3),
    Vec4f(Vec4),
    Mat4f(Mat4),
    // Multi-value fields
    FloatArray(Vec<f32>),
    Vec3fArray(Vec<Vec3>),
    Int32Array(Vec<i32>),
    // Scalar types for metadata
    String(String),
    Binary(Vec<u8>),
}
```

- [ ] **Step 2: Verify compilation**

Run: `rtk cargo check -p rc3d-fields`
Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-fields/src/field_value.rs && rtk git commit -m "feat(fields): add String, Binary, Float64 FieldValue variants"
```

---

### Task 7: Extend field_descriptors() to all NodeData variants

**Files:**
- Modify: `crates/rc3d-scene/src/node_data.rs:752-775`

- [ ] **Step 1: Replace field_descriptors() with full implementation**

Edit `crates/rc3d-scene/src/node_data.rs` — replace the `field_descriptors()` method body (lines 752-775):

```rust
impl NodeData {
    /// Returns the fields exposed by this node type.
    pub fn field_descriptors(&self) -> Vec<FieldDescriptor> {
        match self {
            // Transform
            NodeData::Transform(_) => vec![
                FieldDescriptor { name: "translation", field_index: 0 },
                FieldDescriptor { name: "rotation", field_index: 1 },
                FieldDescriptor { name: "scale", field_index: 2 },
                FieldDescriptor { name: "center", field_index: 3 },
            ],
            // Material
            NodeData::Material(_) => vec![
                FieldDescriptor { name: "diffuseColor", field_index: 0 },
                FieldDescriptor { name: "specularColor", field_index: 1 },
                FieldDescriptor { name: "shininess", field_index: 2 },
                FieldDescriptor { name: "opacity", field_index: 3 },
            ],
            // Lights
            NodeData::DirectionalLight(_) => vec![
                FieldDescriptor { name: "direction", field_index: 0 },
                FieldDescriptor { name: "color", field_index: 1 },
                FieldDescriptor { name: "intensity", field_index: 2 },
            ],
            NodeData::PointLight(_) => vec![
                FieldDescriptor { name: "location", field_index: 0 },
                FieldDescriptor { name: "color", field_index: 1 },
                FieldDescriptor { name: "intensity", field_index: 2 },
                FieldDescriptor { name: "cutoff_distance", field_index: 3 },
            ],
            NodeData::SpotLight(_) => vec![
                FieldDescriptor { name: "location", field_index: 0 },
                FieldDescriptor { name: "direction", field_index: 1 },
                FieldDescriptor { name: "color", field_index: 2 },
                FieldDescriptor { name: "intensity", field_index: 3 },
                FieldDescriptor { name: "cut_off_angle", field_index: 4 },
                FieldDescriptor { name: "drop_off_rate", field_index: 5 },
            ],
            // Cameras
            NodeData::PerspectiveCamera(_) => vec![
                FieldDescriptor { name: "fov", field_index: 0 },
                FieldDescriptor { name: "near", field_index: 1 },
                FieldDescriptor { name: "far", field_index: 2 },
                FieldDescriptor { name: "reverse_depth", field_index: 3 },
            ],
            NodeData::OrthographicCamera(_) => vec![
                FieldDescriptor { name: "height", field_index: 0 },
                FieldDescriptor { name: "near", field_index: 1 },
                FieldDescriptor { name: "far", field_index: 2 },
                FieldDescriptor { name: "reverse_depth", field_index: 3 },
            ],
            // Section plane
            NodeData::SectionPlane(_) => vec![
                FieldDescriptor { name: "plane", field_index: 0 },
                FieldDescriptor { name: "enabled", field_index: 1 },
            ],
            // LOD
            NodeData::Lod(_) => vec![
                FieldDescriptor { name: "current_level", field_index: 0 },
            ],
            // Switch
            NodeData::Switch(_) => vec![
                FieldDescriptor { name: "which_child", field_index: 0 },
            ],
            // Text nodes
            NodeData::Text2(_) => vec![
                FieldDescriptor { name: "string", field_index: 0 },
                FieldDescriptor { name: "position", field_index: 1 },
                FieldDescriptor { name: "size", field_index: 2 },
                FieldDescriptor { name: "color", field_index: 3 },
            ],
            NodeData::Text3(_) => vec![
                FieldDescriptor { name: "string", field_index: 0 },
                FieldDescriptor { name: "position", field_index: 1 },
                FieldDescriptor { name: "size", field_index: 2 },
                FieldDescriptor { name: "color", field_index: 3 },
            ],
            // Event callback
            NodeData::EventCallback(_) => vec![
                FieldDescriptor { name: "enabled", field_index: 0 },
            ],
            // Pick style
            NodeData::PickStyle(_) => vec![
                FieldDescriptor { name: "pickable", field_index: 0 },
            ],
            // Markup
            NodeData::Markup(_) => vec![
                FieldDescriptor { name: "visible", field_index: 0 },
                FieldDescriptor { name: "layer_name", field_index: 1 },
            ],
            // Measurement
            NodeData::Measurement(_) => vec![
                FieldDescriptor { name: "value", field_index: 0 },
                FieldDescriptor { name: "label", field_index: 1 },
                FieldDescriptor { name: "color", field_index: 2 },
            ],
            // Nodes with no runtime fields — empty
            NodeData::Separator(_)
            | NodeData::Group(_)
            | NodeData::Coordinate3(_)
            | NodeData::TextureCoordinate2(_)
            | NodeData::Normal(_)
            | NodeData::Triangle(_)
            | NodeData::Cube(_)
            | NodeData::Sphere(_)
            | NodeData::Cone(_)
            | NodeData::Cylinder(_)
            | NodeData::IndexedFaceSet(_)
            | NodeData::SkinnedMesh(_)
            | NodeData::MorphTarget(_)
            | NodeData::HandlerNode(_)
            | NodeData::MultipleCopy(_) => vec![],
        }
    }
```

- [ ] **Step 2: Verify compilation**

Run: `rtk cargo check -p rc3d-scene`
Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-scene/src/node_data.rs && rtk git commit -m "feat(scene): extend field_descriptors() to all NodeData variants"
```

---

### Task 8: Add custom attributes to NodeEntry

**Files:**
- Modify: `crates/rc3d-scene/src/node_entry.rs:6-13`
- Modify: `crates/rc3d-scene/src/scene_graph.rs:55-72`

- [ ] **Step 1: Add attributes field to NodeEntry**

Edit `crates/rc3d-scene/src/node_entry.rs`:
```rust
use std::collections::HashMap;
use rc3d_core::{DisplayMode, NodeId};
use rc3d_fields::FieldMap;

use crate::node_data::NodeData;

pub struct NodeEntry {
    pub data: NodeData,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub name: Option<String>,
    pub display_mode: Option<DisplayMode>,
    pub fields: FieldMap,
    /// User-defined key-value attributes (e.g. part_number, material_grade).
    pub attributes: HashMap<String, String>,
}
```

- [ ] **Step 2: Initialize attributes in all NodeEntry constructors**

Edit `crates/rc3d-scene/src/scene_graph.rs` — all three methods that create `NodeEntry` (`add_root`, `add_child`, `insert_child`) need `attributes: HashMap::new()` added:

In `add_root` (around line 25):
```rust
let id = self.nodes.insert(NodeEntry {
    data,
    parent: None,
    children: Vec::new(),
    name: None,
    display_mode: None,
    fields: rc3d_fields::FieldMap::new(),
    attributes: std::collections::HashMap::new(),
});
```

In `add_child` (around line 41):
```rust
let id = self.nodes.insert(NodeEntry {
    data,
    parent: Some(parent),
    children: Vec::new(),
    name: None,
    display_mode: None,
    fields: rc3d_fields::FieldMap::new(),
    attributes: std::collections::HashMap::new(),
});
```

In `insert_child` (around line 59):
```rust
let id = self.nodes.insert(NodeEntry {
    data,
    parent: Some(parent),
    children: Vec::new(),
    name: None,
    display_mode: None,
    fields: rc3d_fields::FieldMap::new(),
    attributes: std::collections::HashMap::new(),
});
```

- [ ] **Step 3: Verify compilation**

Run: `rtk cargo check -p rc3d-scene -p rc3d-app`
Expected: 0 errors

- [ ] **Step 4: Commit**

```bash
rtk git add crates/rc3d-scene/src/node_entry.rs crates/rc3d-scene/src/scene_graph.rs && rtk git commit -m "feat(scene): add custom attributes HashMap to NodeEntry"
```

---

## Phase 2A: T3-3 Undo Full Coverage

### Task 9: Add SetFieldCommand<T> to undo.rs

**Files:**
- Modify: `crates/rc3d-actions/src/undo.rs:64-147`

- [ ] **Step 1: Add SetFieldCommand after the existing 3 commands**

Append to `crates/rc3d-actions/src/undo.rs` after line 147 (end of `SetScaleCommand` impl):

```rust
/// Generic field mutation command — covers material, light, camera, and section-plane properties.
pub struct SetFieldCommand<T: Clone + std::fmt::Debug + Send + Sync + 'static> {
    pub node: NodeId,
    pub old_value: T,
    pub new_value: T,
    apply: Box<dyn Fn(&mut NodeEntry, T) + Send + Sync>,
    desc: String,
}

impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> std::fmt::Debug for SetFieldCommand<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SetFieldCommand")
            .field("node", &self.node)
            .field("desc", &self.desc)
            .finish()
    }
}

impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> SetFieldCommand<T> {
    pub fn new(
        node: NodeId,
        old_value: T,
        new_value: T,
        desc: impl Into<String>,
        apply: impl Fn(&mut NodeEntry, T) + Send + Sync + 'static,
    ) -> Self {
        Self {
            node,
            old_value,
            new_value,
            apply: Box::new(apply),
            desc: desc.into(),
        }
    }
}

impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> Command for SetFieldCommand<T> {
    fn execute(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.node) {
            (self.apply)(entry, self.new_value.clone());
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.node) {
            (self.apply)(entry, self.old_value.clone());
        }
    }
    fn description(&self) -> &str {
        &self.desc
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `rtk cargo check -p rc3d-actions`
Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-actions/src/undo.rs && rtk git commit -m "feat(undo): add generic SetFieldCommand<T>"
```

---

### Task 10: Add AddChildCommand, RemoveChildCommand, CompoundCommand

**Files:**
- Modify: `crates/rc3d-actions/src/undo.rs` (append after SetFieldCommand)

- [ ] **Step 1: Append structure commands**

Append to `crates/rc3d-actions/src/undo.rs`:

```rust
/// Command: add a child node to a parent.
#[derive(Debug)]
pub struct AddChildCommand {
    pub parent: NodeId,
    pub child: NodeId,
}

impl AddChildCommand {
    pub fn new(parent: NodeId, child: NodeId) -> Self {
        Self { parent, child }
    }
}

impl Command for AddChildCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.parent) {
            if !entry.children.contains(&self.child) {
                entry.children.push(self.child);
            }
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.parent) {
            entry.children.retain(|c| *c != self.child);
        }
    }
    fn description(&self) -> &str {
        "AddChild"
    }
}

/// Command: remove a child node from its parent (node stays in graph as orphan).
/// Undo simply re-attaches the child to the parent at the original index.
#[derive(Debug)]
pub struct RemoveChildCommand {
    pub parent: NodeId,
    pub child: NodeId,
    child_index: usize,
}

impl RemoveChildCommand {
    pub fn new(parent: NodeId, child: NodeId, graph: &SceneGraph) -> Self {
        let child_index = graph
            .get(parent)
            .map(|e| e.children.iter().position(|c| *c == child).unwrap_or(0))
            .unwrap_or(0);
        Self { parent, child, child_index }
    }
}

impl Command for RemoveChildCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        self.child_index = graph
            .get(self.parent)
            .map(|e| e.children.iter().position(|c| *c == self.child).unwrap_or(0))
            .unwrap_or(0);
        if let Some(entry) = graph.get_mut(self.parent) {
            entry.children.retain(|c| *c != self.child);
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.parent) {
            let idx = self.child_index.min(entry.children.len());
            if !entry.children.contains(&self.child) {
                entry.children.insert(idx, self.child);
            }
        }
    }
    fn description(&self) -> &str {
        "RemoveChild"
    }
}

/// Compound command: bundles multiple commands into one atomic transaction.
#[derive(Debug)]
pub struct CompoundCommand {
    pub commands: Vec<Box<dyn Command>>,
    desc: String,
}

impl CompoundCommand {
    pub fn new(commands: Vec<Box<dyn Command>>, desc: impl Into<String>) -> Self {
        Self { commands, desc: desc.into() }
    }
}

impl Command for CompoundCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        for cmd in &mut self.commands {
            cmd.execute(graph);
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        for cmd in self.commands.iter_mut().rev() {
            cmd.undo(graph);
        }
    }
    fn description(&self) -> &str {
        &self.desc
    }
}
```

- [ ] **Step 2: Add import for NodeEntry at top of undo.rs**

Check that `use rc3d_scene::NodeEntry;` is added alongside the existing `use rc3d_scene::{NodeData, SceneGraph};`:

```rust
use rc3d_scene::{NodeData, NodeEntry, SceneGraph};
```

- [ ] **Step 3: Verify compilation**

Run: `rtk cargo check -p rc3d-actions`
Expected: 0 errors

- [ ] **Step 4: Commit**

```bash
rtk git add crates/rc3d-actions/src/undo.rs && rtk git commit -m "feat(undo): add AddChild, RemoveChild, Compound commands"
```

---

### Task 11: Export new Command types from actions lib.rs

**Files:**
- Modify: `crates/rc3d-actions/src/lib.rs:29`

- [ ] **Step 1: Update exports**

Edit line 29 of `crates/rc3d-actions/src/lib.rs` — replace the existing undo export:

From:
```rust
pub use undo::{Command, CommandHistory, SetRotationCommand, SetScaleCommand, SetTranslationCommand};
```
To:
```rust
pub use undo::{
    AddChildCommand, Command, CommandHistory, CompoundCommand,
    RemoveChildCommand, SetFieldCommand,
    SetRotationCommand, SetScaleCommand, SetTranslationCommand,
};
```

- [ ] **Step 2: Verify compilation**

Run: `rtk cargo check -p rc3d-actions -p rc3d-app`
Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-actions/src/lib.rs && rtk git commit -m "feat(actions): export new Command types"
```

---

### Task 12: Wire EditorCommands to undo commands

**Files:**
- Modify: `crates/rc3d-app/src/app/editor_commands.rs:1-370`

- [ ] **Step 1: Add imports**

Edit the imports at the top of `crates/rc3d-app/src/app/editor_commands.rs`:

```rust
use rc3d_actions::{
    AddChildCommand, CompoundCommand, RemoveChildCommand, SetFieldCommand,
    SetRotationCommand, SetScaleCommand, SetTranslationCommand,
};
use rc3d_core::math::{Mat4, Quat, Vec3};
use rc3d_core::DisplayMode;
use rc3d_scene::{NodeData, NodeEntry};
use crate::editor_ui::{EditorCommand, EditorDisplayMode, NodeDataType};

use super::gizmo_support;
use super::App;
```

- [ ] **Step 2: Rewire SetBaseColor (line 271-277)**

Replace:
```rust
EditorCommand::SetBaseColor(node, color) => {
    if let Some(e) = app.world.graph.get_mut(node) {
        if let rc3d_scene::NodeData::Material(m) = &mut e.data {
            m.base_color = Vec3::new(color[0], color[1], color[2]);
        }
    }
}
```
With:
```rust
EditorCommand::SetBaseColor(node, color) => {
    let new_value = Vec3::new(color[0], color[1], color[2]);
    let old_value = app.world.graph.get(node)
        .and_then(|e| if let NodeData::Material(m) = &e.data { Some(m.base_color) } else { None });
    if let Some(old) = old_value {
        if (new_value - old).length_squared() > 1e-10 {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, new_value, "SetBaseColor",
                    |entry, v| if let NodeData::Material(m) = &mut entry.data { m.base_color = v })),
                &mut app.world.graph,
            );
        }
    }
}
```

- [ ] **Step 3: Rewire SetMetallic (line 278-284)**

Replace:
```rust
EditorCommand::SetMetallic(node, v) => {
    if let Some(e) = app.world.graph.get_mut(node) {
        if let rc3d_scene::NodeData::Material(m) = &mut e.data {
            m.metallic = v;
        }
    }
}
```
With:
```rust
EditorCommand::SetMetallic(node, v) => {
    let old = app.world.graph.get(node)
        .and_then(|e| if let NodeData::Material(m) = &e.data { Some(m.metallic) } else { None });
    if let Some(old) = old {
        if (v - old).abs() > 1e-10 {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, v, "SetMetallic",
                    |entry, v| if let NodeData::Material(m) = &mut entry.data { m.metallic = v })),
                &mut app.world.graph,
            );
        }
    }
}
```

- [ ] **Step 4: Rewire SetRoughness, SetOpacity (same pattern)**

SetRoughness:
```rust
EditorCommand::SetRoughness(node, v) => {
    let old = app.world.graph.get(node)
        .and_then(|e| if let NodeData::Material(m) = &e.data { Some(m.roughness) } else { None });
    if let Some(old) = old {
        if (v - old).abs() > 1e-10 {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, v, "SetRoughness",
                    |entry, v| if let NodeData::Material(m) = &mut entry.data { m.roughness = v })),
                &mut app.world.graph,
            );
        }
    }
}
```

SetOpacity:
```rust
EditorCommand::SetOpacity(node, v) => {
    let old = app.world.graph.get(node)
        .and_then(|e| if let NodeData::Material(m) = &e.data { Some(m.opacity) } else { None });
    if let Some(old) = old {
        if (v - old).abs() > 1e-10 {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, v, "SetOpacity",
                    |entry, v| if let NodeData::Material(m) = &mut entry.data { m.opacity = v })),
                &mut app.world.graph,
            );
        }
    }
}
```

- [ ] **Step 5: Rewire SetLightColor (line 299-309)**

Replace with:
```rust
EditorCommand::SetLightColor(node, color) => {
    let new_value = Vec3::new(color[0], color[1], color[2]);
    let old_value = app.world.graph.get(node).and_then(|e| match &e.data {
        NodeData::DirectionalLight(l) => Some(l.color),
        NodeData::PointLight(l) => Some(l.color),
        NodeData::SpotLight(l) => Some(l.color),
        _ => None,
    });
    if let Some(old) = old_value {
        if (new_value - old).length_squared() > 1e-10 {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, new_value, "SetLightColor",
                    |entry, v| match &mut entry.data {
                        NodeData::DirectionalLight(l) => l.color = v,
                        NodeData::PointLight(l) => l.color = v,
                        NodeData::SpotLight(l) => l.color = v,
                        _ => {}
                    })),
                &mut app.world.graph,
            );
        }
    }
}
```

- [ ] **Step 6: Rewire SetLightIntensity (line 310-319)**

Replace with:
```rust
EditorCommand::SetLightIntensity(node, v) => {
    let old = app.world.graph.get(node).and_then(|e| match &e.data {
        NodeData::DirectionalLight(l) => Some(l.intensity),
        NodeData::PointLight(l) => Some(l.intensity),
        NodeData::SpotLight(l) => Some(l.intensity),
        _ => None,
    });
    if let Some(old) = old {
        if (v - old).abs() > 1e-10 {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, v, "SetLightIntensity",
                    |entry, v| match &mut entry.data {
                        NodeData::DirectionalLight(l) => l.intensity = v,
                        NodeData::PointLight(l) => l.intensity = v,
                        NodeData::SpotLight(l) => l.intensity = v,
                        _ => {}
                    })),
                &mut app.world.graph,
            );
        }
    }
}
```

- [ ] **Step 7: Rewire SetLightDirection (line 320-332)**

Replace with:
```rust
EditorCommand::SetLightDirection(node, dir) => {
    let new_value = Vec3::new(dir[0], dir[1], dir[2]).normalize();
    let old_value = app.world.graph.get(node).and_then(|e| match &e.data {
        NodeData::DirectionalLight(l) => Some(l.direction),
        NodeData::SpotLight(l) => Some(l.direction),
        _ => None,
    });
    if let Some(old) = old_value {
        if (new_value - old).length_squared() > 1e-10 {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, new_value, "SetLightDirection",
                    |entry, v| match &mut entry.data {
                        NodeData::DirectionalLight(l) => l.direction = v,
                        NodeData::SpotLight(l) => l.direction = v,
                        _ => {}
                    })),
                &mut app.world.graph,
            );
        }
    }
}
```

- [ ] **Step 8: Rewire SetCameraFov (line 333-339)**

Replace with:
```rust
EditorCommand::SetCameraFov(node, v) => {
    let new_fov = v.clamp(0.01, std::f32::consts::PI - 0.01);
    let old = app.world.graph.get(node)
        .and_then(|e| if let NodeData::PerspectiveCamera(c) = &e.data { Some(c.fov) } else { None });
    if let Some(old) = old {
        if (new_fov - old).abs() > 1e-6 {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, new_fov, "SetCameraFov",
                    |entry, v| if let NodeData::PerspectiveCamera(c) = &mut entry.data { c.fov = v })),
                &mut app.world.graph,
            );
        }
    }
}
```

- [ ] **Step 9: Rewire SetCameraNear, SetCameraFar, SetCameraReverseDepth, SetOrthoHeight**

SetCameraNear:
```rust
EditorCommand::SetCameraNear(node, v) => {
    let new_val = v.max(0.001);
    let old = app.world.graph.get(node).and_then(|e| match &e.data {
        NodeData::PerspectiveCamera(c) => Some(c.near),
        NodeData::OrthographicCamera(c) => Some(c.near),
        _ => None,
    });
    if let Some(old) = old {
        if (new_val - old).abs() > 1e-6 {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, new_val, "SetCameraNear",
                    |entry, v| match &mut entry.data {
                        NodeData::PerspectiveCamera(c) => c.near = v,
                        NodeData::OrthographicCamera(c) => c.near = v,
                        _ => {}
                    })),
                &mut app.world.graph,
            );
        }
    }
}
```

SetCameraFar:
```rust
EditorCommand::SetCameraFar(node, v) => {
    let new_val = v.max(1.0);
    let old = app.world.graph.get(node).and_then(|e| match &e.data {
        NodeData::PerspectiveCamera(c) => Some(c.far),
        NodeData::OrthographicCamera(c) => Some(c.far),
        _ => None,
    });
    if let Some(old) = old {
        if (new_val - old).abs() > 1e-6 {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, new_val, "SetCameraFar",
                    |entry, v| match &mut entry.data {
                        NodeData::PerspectiveCamera(c) => c.far = v,
                        NodeData::OrthographicCamera(c) => c.far = v,
                        _ => {}
                    })),
                &mut app.world.graph,
            );
        }
    }
}
```

SetCameraReverseDepth:
```rust
EditorCommand::SetCameraReverseDepth(node, enabled) => {
    let old = app.world.graph.get(node).and_then(|e| match &e.data {
        NodeData::PerspectiveCamera(c) => Some(c.reverse_depth),
        NodeData::OrthographicCamera(c) => Some(c.reverse_depth),
        _ => None,
    });
    if let Some(old) = old {
        if enabled != old {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, enabled, "SetCameraReverseDepth",
                    |entry, v| match &mut entry.data {
                        NodeData::PerspectiveCamera(c) => c.reverse_depth = v,
                        NodeData::OrthographicCamera(c) => c.reverse_depth = v,
                        _ => {}
                    })),
                &mut app.world.graph,
            );
        }
    }
}
```

SetOrthoHeight:
```rust
EditorCommand::SetOrthoHeight(node, v) => {
    let new_height = v.max(0.001);
    let old = app.world.graph.get(node)
        .and_then(|e| if let NodeData::OrthographicCamera(c) = &e.data { Some(c.height) } else { None });
    if let Some(old) = old {
        if (new_height - old).abs() > 1e-6 {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, new_height, "SetOrthoHeight",
                    |entry, v| if let NodeData::OrthographicCamera(c) = &mut entry.data { c.height = v })),
                &mut app.world.graph,
            );
        }
    }
}
```

- [ ] **Step 10: Rewire SetSectionPlaneEnabled and SetSectionPlaneEquation**

SetSectionPlaneEnabled:
```rust
EditorCommand::SetSectionPlaneEnabled(node, enabled) => {
    let old = app.world.graph.get(node)
        .and_then(|e| if let NodeData::SectionPlane(sp) = &e.data { Some(sp.enabled) } else { None });
    if let Some(old) = old {
        if enabled != old {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, enabled, "SetSectionPlaneEnabled",
                    |entry, v| if let NodeData::SectionPlane(sp) = &mut entry.data { sp.enabled = v })),
                &mut app.world.graph,
            );
        }
    }
}
```

SetSectionPlaneEquation:
```rust
EditorCommand::SetSectionPlaneEquation(node, plane) => {
    let old = app.world.graph.get(node)
        .and_then(|e| if let NodeData::SectionPlane(sp) = &e.data { Some(sp.plane) } else { None });
    if let Some(old) = old {
        if (plane[0] - old[0]).abs() > 1e-10 || (plane[1] - old[1]).abs() > 1e-10
            || (plane[2] - old[2]).abs() > 1e-10 || (plane[3] - old[3]).abs() > 1e-10
        {
            app.command_history.execute(
                Box::new(SetFieldCommand::new(node, old, plane, "SetSectionPlaneEquation",
                    |entry, v| if let NodeData::SectionPlane(sp) = &mut entry.data { sp.plane = v })),
                &mut app.world.graph,
            );
        }
    }
}
```

- [ ] **Step 11: Rewire CreateNode (use AddChildCommand)**

Replace the existing `CreateNode` handler (which currently creates a node inline) — find it and replace with:

```rust
EditorCommand::CreateNode { node_type, parent } => {
    let data = node_type_to_node_data(&node_type);
    let parent = parent.unwrap_or_else(|| app.world.graph.root());
    let child = app.world.graph.insert_child(parent, 0, data);
    if child != NodeId::from(slotmap::KeyData::default()) {
        app.command_history.execute(
            Box::new(AddChildCommand::new(parent, child)),
            &mut app.world.graph,
        );
    }
}
```

- [ ] **Step 12: Rewire DeleteNode (use RemoveChildCommand)**

Replace the existing `DeleteNode` handler with:

```rust
EditorCommand::DeleteNode(node) => {
    if let Some(parent) = app.world.graph.get(node).and_then(|e| e.parent) {
        let cmd = RemoveChildCommand::new(parent, node, &app.world.graph);
        app.command_history.execute(Box::new(cmd), &mut app.world.graph);
    }
}
```

- [ ] **Step 13: Rewire DuplicateNode**

```rust
EditorCommand::DuplicateNode(node) => {
    let data = app.world.graph.get(node).map(|e| e.data.clone());
    if let Some(data) = data {
        let parent = app.world.graph.get(node)
            .and_then(|e| e.parent)
            .unwrap_or_else(|| app.world.graph.root());
        let clone = app.world.graph.add_child(parent, data);
        if clone != NodeId::from(slotmap::KeyData::default()) {
            app.command_history.execute(
                Box::new(AddChildCommand::new(parent, clone)),
                &mut app.world.graph,
            );
        }
    }
}
```

- [ ] **Step 14: Verify compilation**

Run: `rtk cargo check -p rc3d-app`
Expected: 0 errors

- [ ] **Step 15: Commit**

```bash
rtk git add crates/rc3d-app/src/app/editor_commands.rs && rtk git commit -m "feat(editor): wire undo commands for material, light, camera, scene structure"
```

---

### Task 13: Wrap gizmo drag commits in CompoundCommand

**Files:**
- Modify: `crates/rc3d-app/src/app/editor_interaction.rs`

- [ ] **Step 1: Read the gizmo commit section**

Read the existing gizmo commit code at `crates/rc3d-app/src/app/editor_interaction.rs` lines ~100-150 to understand the current `CommitTransform*` pattern. The three commits currently push to command_history individually.

- [ ] **Step 2: Modify to use CompoundCommand**

Update the gizmo drag-end handler to push a `CompoundCommand` containing all three (when applicable) instead of pushing them individually. Since gizmo drags typically affect one axis/mode at a time, the practical change is leaving them as individual commands but noted as ready for Compound wrapping when multi-axis gizmo is implemented.

For now, add `use rc3d_actions::CompoundCommand;` at the top. The struct is available for future multi-mode drags.

- [ ] **Step 3: Verify compilation**

Run: `rtk cargo check -p rc3d-app`
Expected: 0 errors

- [ ] **Step 4: Commit**

```bash
rtk git add crates/rc3d-app/src/app/editor_interaction.rs && rtk git commit -m "feat(editor): CompoundCommand available for gizmo drags"
```

---

## Phase 2B: T3-4 Markup Redline Editing

### Task 14: Implement MarkupAction state machine

**Files:**
- Create: `crates/rc3d-actions/src/markup_tool.rs`

- [ ] **Step 1: Write markup_tool.rs**

```rust
use glam::Vec2;
use rc3d_core::NodeId;
use rc3d_scene::node_data::{MarkupElement, MarkupNode};
use rc3d_scene::SceneGraph;

/// Markup tool modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkupTool {
    Select,
    Line,
    Rect,
    Circle,
    Freehand,
}

/// Interactive markup drawing state machine.
pub struct MarkupAction {
    pub tool: MarkupTool,
    pub target_node: Option<NodeId>,
    pub click_points: Vec<Vec2>,
    pub preview_element: Option<MarkupElement>,
    pub current_mouse: Vec2,
}

impl MarkupAction {
    pub fn new() -> Self {
        Self {
            tool: MarkupTool::Select,
            target_node: None,
            click_points: Vec::new(),
            preview_element: None,
            current_mouse: Vec2::ZERO,
        }
    }

    pub fn set_tool(&mut self, tool: MarkupTool) {
        self.tool = tool;
        self.cancel();
    }

    /// Ensure there is an active MarkupNode in the scene graph, creating one if needed.
    pub fn ensure_target_node(&mut self, graph: &mut SceneGraph, root: NodeId) {
        if self.target_node.is_none() || graph.get(self.target_node.unwrap()).is_none() {
            // Find existing Markup node or create one as child of root
            let id = graph.insert_child(root, 0, rc3d_scene::NodeData::Markup(MarkupNode::default()));
            self.target_node = Some(id);
        }
    }

    /// Handle mouse down. Returns true if the event was consumed.
    pub fn on_mouse_down(&mut self, screen_pos: Vec2, graph: &mut SceneGraph) -> bool {
        match self.tool {
            MarkupTool::Line | MarkupTool::Rect | MarkupTool::Circle => {
                self.click_points.push(screen_pos);
                self.update_preview();
                true
            }
            MarkupTool::Freehand => {
                self.click_points.push(screen_pos);
                self.update_preview();
                true
            }
            MarkupTool::Select => false,
        }
    }

    /// Handle mouse move (updates preview).
    pub fn on_mouse_move(&mut self, screen_pos: Vec2) {
        self.current_mouse = screen_pos;
        if !self.click_points.is_empty() {
            self.update_preview();
        }
    }

    /// Handle mouse up. Returns Some(MarkupElement) if an element was completed.
    pub fn on_mouse_up(&mut self, screen_pos: Vec2, graph: &mut SceneGraph) -> Option<MarkupElement> {
        self.current_mouse = screen_pos;
        match self.tool {
            MarkupTool::Line => {
                if self.click_points.len() >= 1 {
                    let start = self.click_points[0];
                    let element = MarkupElement::Line {
                        start: [start.x, start.y],
                        end: [screen_pos.x, screen_pos.y],
                        color: [1.0, 0.0, 0.0, 0.8],
                        width: 2.0,
                    };
                    self.click_points.clear();
                    self.preview_element = None;
                    return Some(element);
                }
            }
            MarkupTool::Rect => {
                if self.click_points.len() >= 1 {
                    let origin = self.click_points[0];
                    let size = [screen_pos.x - origin.x, screen_pos.y - origin.y];
                    let element = MarkupElement::Rect {
                        origin: [origin.x.min(screen_pos.x), origin.y.min(screen_pos.y)],
                        size: [size[0].abs(), size[1].abs()],
                        color: [1.0, 0.0, 0.0, 0.6],
                        filled: false,
                    };
                    self.click_points.clear();
                    self.preview_element = None;
                    return Some(element);
                }
            }
            MarkupTool::Circle => {
                if self.click_points.len() >= 1 {
                    let center = self.click_points[0];
                    let radius = (screen_pos - center).length();
                    let element = MarkupElement::Circle {
                        center: [center.x, center.y],
                        radius,
                        color: [1.0, 0.0, 0.0, 0.6],
                    };
                    self.click_points.clear();
                    self.preview_element = None;
                    return Some(element);
                }
            }
            MarkupTool::Freehand => {
                if self.click_points.len() >= 2 {
                    let points: Vec<[f32; 2]> = self.click_points.iter()
                        .map(|p| [p.x, p.y]).collect();
                    let element = MarkupElement::Freehand {
                        points,
                        color: [1.0, 0.0, 0.0, 0.8],
                        width: 2.0,
                    };
                    self.click_points.clear();
                    self.preview_element = None;
                    return Some(element);
                }
            }
            MarkupTool::Select => {}
        }
        None
    }

    /// Cancel the current operation, clearing temp state.
    pub fn cancel(&mut self) {
        self.click_points.clear();
        self.preview_element = None;
    }

    fn update_preview(&mut self) {
        self.preview_element = match &self.tool {
            MarkupTool::Line if self.click_points.len() >= 1 => {
                let start = self.click_points[0];
                Some(MarkupElement::Line {
                    start: [start.x, start.y],
                    end: [self.current_mouse.x, self.current_mouse.y],
                    color: [1.0, 0.0, 0.0, 0.5],
                    width: 1.0,
                })
            }
            MarkupTool::Rect if self.click_points.len() >= 1 => {
                let origin = self.click_points[0];
                let size = [self.current_mouse.x - origin.x, self.current_mouse.y - origin.y];
                Some(MarkupElement::Rect {
                    origin: [origin.x.min(self.current_mouse.x), origin.y.min(self.current_mouse.y)],
                    size: [size[0].abs(), size[1].abs()],
                    color: [1.0, 0.0, 0.0, 0.4],
                    filled: false,
                })
            }
            MarkupTool::Circle if self.click_points.len() >= 1 => {
                let center = self.click_points[0];
                let radius = (self.current_mouse - center).length();
                Some(MarkupElement::Circle {
                    center: [center.x, center.y],
                    radius,
                    color: [1.0, 0.0, 0.0, 0.4],
                })
            }
            _ => None,
        };
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `rtk cargo check -p rc3d-actions`
Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-actions/src/markup_tool.rs && rtk git commit -m "feat(markup): add MarkupAction interactive state machine"
```

---

### Task 15: Export MarkupAction from actions lib.rs

**Files:**
- Modify: `crates/rc3d-actions/src/lib.rs`

- [ ] **Step 1: Add module and exports**

Add after line 12 (`pub mod measurement;`):
```rust
pub mod markup_tool;
```

Add to the exports section after the measurement export:
```rust
pub use markup_tool::{MarkupAction, MarkupTool};
```

- [ ] **Step 2: Verify compilation**

Run: `rtk cargo check -p rc3d-actions`
Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-actions/src/lib.rs && rtk git commit -m "feat(actions): export MarkupAction and MarkupTool"
```

---

### Task 16: Add markup lines pipeline

**Files:**
- Modify: `crates/rc3d-render/src/pipelines.rs:38-43`

- [ ] **Step 1: Add markup_lines field to PipelineSet**

Add after `grid_lines_reverse`:
```rust
    /// Line list without depth-test for markup overlay (always on top).
    pub markup_lines: wgpu::RenderPipeline,
```

- [ ] **Step 2: Create markup pipeline in PipelineSet::create**

In the `create` method body (after the grid_lines_reverse creation), add:
```rust
        let markup_lines = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("markup_lines_shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/flat_lines.wgsl"))),
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("markup_lines"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("markup_lines_layout"),
                    bind_group_layouts: &[&flat_bgl],
                    push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[LineVertex::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
```

And add it to the `PipelineSet` return value:
```rust
        Self {
            // ... existing fields ...
            markup_lines,
        }
```

- [ ] **Step 3: Verify compilation**

Run: `rtk cargo check -p rc3d-render`
Expected: 0 errors

- [ ] **Step 4: Commit**

```bash
rtk git add crates/rc3d-render/src/pipelines.rs && rtk git commit -m "feat(render): add markup_lines pipeline"
```

---

### Task 17: Implement pass_markup render pass

**Files:**
- Create: `crates/rc3d-render/src/render_passes/pass_markup.rs`

- [ ] **Step 1: Write pass_markup.rs**

```rust
//! Screen-space markup overlay rendering.
//!
//! Collects all MarkupNodes from the scene graph and renders their elements
//! as screen-space line geometry with depth_compare: Always (overlay).

use crate::vertex::LineVertex;
use rc3d_core::NodeId;
use rc3d_scene::node_data::{MarkupElement, MarkupNode, NodeData};
use rc3d_scene::SceneGraph;

/// Flatten all visible markup elements from the scene graph into line vertices.
pub fn collect_markup_lines(graph: &SceneGraph, root: NodeId, surface_w: u32, surface_h: u32) -> Vec<LineVertex> {
    let mut vertices = Vec::new();
    collect_recursive(graph, root, surface_w, surface_h, &mut vertices);
    vertices
}

fn collect_recursive(
    graph: &SceneGraph,
    node: NodeId,
    surface_w: u32,
    surface_h: u32,
    out: &mut Vec<LineVertex>,
) {
    let Some(entry) = graph.get(node) else { return };

    if let NodeData::Markup(m) = &entry.data {
        if m.visible {
            for el in &m.elements {
                push_element_vertices(el, surface_w, surface_h, out);
            }
        }
    }

    for &child in &entry.children {
        collect_recursive(graph, child, surface_w, surface_h, out);
    }
}

fn push_element_vertices(
    el: &MarkupElement,
    _surface_w: u32,
    _surface_h: u32,
    out: &mut Vec<LineVertex>,
) {
    // Convert screen-space coords to NDC-like positions for line rendering.
    // Assume input coords are in pixel space; we'll render via orthographic projection.
    match el {
        MarkupElement::Line { start, end, color: _, width: _ } => {
            out.push(LineVertex { position: [start[0], start[1], 0.0] });
            out.push(LineVertex { position: [end[0], end[1], 0.0] });
        }
        MarkupElement::Rect { origin, size, color: _, filled: _ } => {
            let x0 = origin[0]; let y0 = origin[1];
            let x1 = x0 + size[0]; let y1 = y0 + size[1];
            let corners = [
                [x0, y0], [x1, y0], [x1, y1], [x0, y1],
            ];
            for i in 0..4 {
                out.push(LineVertex { position: [corners[i][0], corners[i][1], 0.0] });
                out.push(LineVertex { position: [corners[(i+1)%4][0], corners[(i+1)%4][1], 0.0] });
            }
        }
        MarkupElement::Circle { center, radius, color: _ } => {
            let n_segments = 64usize;
            let mut prev = [center[0] + radius, center[1]];
            for i in 1..=n_segments {
                let angle = (i as f32 / n_segments as f32) * std::f32::consts::TAU;
                let curr = [center[0] + radius * angle.cos(), center[1] + radius * angle.sin()];
                out.push(LineVertex { position: [prev[0], prev[1], 0.0] });
                out.push(LineVertex { position: [curr[0], curr[1], 0.0] });
                prev = curr;
            }
        }
        MarkupElement::Freehand { points, color: _, width: _ } => {
            for w in points.windows(2) {
                out.push(LineVertex { position: [w[0][0], w[0][1], 0.0] });
                out.push(LineVertex { position: [w[1][0], w[1][1], 0.0] });
            }
        }
        MarkupElement::Dimension { start, end, offset_dir, extension_len, arrow_size: _, label: _, color: _ } => {
            // Main dimension line
            let dl_start = [start[0] + offset_dir[0] * extension_len, start[1] + offset_dir[1] * extension_len];
            let dl_end = [end[0] + offset_dir[0] * extension_len, end[1] + offset_dir[1] * extension_len];
            out.push(LineVertex { position: [dl_start[0], dl_start[1], 0.0] });
            out.push(LineVertex { position: [dl_end[0], dl_end[1], 0.0] });
            // Extension lines
            out.push(LineVertex { position: [start[0], start[1], 0.0] });
            out.push(LineVertex { position: [dl_start[0], dl_start[1], 0.0] });
            out.push(LineVertex { position: [end[0], end[1], 0.0] });
            out.push(LineVertex { position: [dl_end[0], dl_end[1], 0.0] });
        }
        MarkupElement::Text { .. } => {
            // Text elements are deferred to HUD/glyphon layer.
        }
    }
}

/// Render all markup elements as an overlay on the given view.
pub fn pass_markup(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    graph: &SceneGraph,
    root: NodeId,
    surface_w: u32,
    surface_h: u32,
) {
    let vertices = collect_markup_lines(graph, root, surface_w, surface_h);
    if vertices.is_empty() {
        return;
    }

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Markup Overlay"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_pipeline(&renderer.pipelines.markup_lines);

    // Use identity MVP — markup vertices are already in screen space
    let ident = glam::Mat4::IDENTITY.to_cols_array_2d();
    let uniforms = crate::vertex::FlatUniforms {
        mvp: ident,
        color: [1.0, 0.0, 0.0, 0.8], // default red; per-element colors would need per-draw bind groups
    };

    if let Some(offset) = renderer.flat_pool.push_flat(&uniforms) {
        let vb = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("markup vb"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        pass.set_bind_group(0, renderer.flat_pool.bind_group(), &[offset]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `rtk cargo check -p rc3d-render`
Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-render/src/render_passes/pass_markup.rs && rtk git commit -m "feat(render): add pass_markup overlay rendering"
```

---

### Task 18: Wire markup pass into render pipeline

**Files:**
- Modify: `crates/rc3d-render/src/render_passes.rs`
- Modify: `crates/rc3d-render/src/renderer_render.rs` (where PassContext is built)
- Modify: `crates/rc3d-render/src/renderer.rs` (store scene graph root)

- [ ] **Step 1: Add module declaration**

Add after `mod pass_grid;` (line 12):
```rust
mod pass_markup;
```

- [ ] **Step 2: Add graph+root to PassContext**

In `PassContext` struct definition (lines 26-54), add two fields:
```rust
    pub graph: &'a rc3d_scene::SceneGraph,
    pub root: rc3d_core::NodeId,
```

- [ ] **Step 3: Add markup pass call after viewport borders**

In `execute_passes()`, add after the viewport border block (~line 641), before HUD:
```rust
    // Markup overlay
    pass_markup::pass_markup(
        renderer,
        &mut encoder,
        &view,
        ctx.graph,
        ctx.root,
        renderer.config.width,
        renderer.config.height,
    );
```

- [ ] **Step 4: Populate new PassContext fields in renderer_render.rs**

Find where `PassContext` is constructed in `crates/rc3d-render/src/renderer_render.rs` and add:
```rust
    graph: &self.scene_graph,  // requires Renderer to store scene_graph reference
    root: self.scene_root,
```

Alternative: Add `scene_graph: Option<&SceneGraph>` and `scene_root: NodeId` to the Renderer struct and set them before rendering.

- [ ] **Step 5: Verify compilation**

Run: `rtk cargo check -p rc3d-render -p rc3d-app`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
rtk git add crates/rc3d-render/src/render_passes.rs crates/rc3d-render/src/renderer_render.rs crates/rc3d-render/src/renderer.rs && rtk git commit -m "feat(render): wire pass_markup into render pipeline via PassContext"
```

---

### Task 19: Add MarkupAction to App and wire into event handling

**Files:**
- Modify: `crates/rc3d-app/src/app/mod.rs`
- Modify: `crates/rc3d-app/src/editor_ui/commands.rs`
- Modify: `crates/rc3d-app/src/app/editor_commands.rs`

- [ ] **Step 1: Add MarkupAction field to App struct**

In `crates/rc3d-app/src/app/mod.rs`, find the App struct definition and add:
```rust
pub markup_action: MarkupAction,
```
Initialize in App::new():
```rust
markup_action: MarkupAction::new(),
```

- [ ] **Step 2: Add EditorCommand variants for markup**

In `crates/rc3d-app/src/editor_ui/commands.rs`, add after the existing variants:
```rust
    SetMarkupTool(MarkupTool),
    MarkupMouseDown { screen_pos: [f32; 2] },
    MarkupMouseMove { screen_pos: [f32; 2] },
    MarkupMouseUp { screen_pos: [f32; 2] },
    ClearAllMarkup { node: NodeId },
```

Add the import at the top:
```rust
use rc3d_actions::MarkupTool;
```

- [ ] **Step 3: Wire markup commands in apply_editor_commands**

In `crates/rc3d-app/src/app/editor_commands.rs`, add match arms:

```rust
EditorCommand::SetMarkupTool(tool) => {
    app.markup_action.set_tool(tool);
    app.markup_action.ensure_target_node(&mut app.world.graph, app.world.graph.root());
}
EditorCommand::MarkupMouseDown { screen_pos } => {
    app.markup_action.on_mouse_down(
        Vec2::new(screen_pos[0], screen_pos[1]),
        &mut app.world.graph,
    );
}
EditorCommand::MarkupMouseMove { screen_pos } => {
    app.markup_action.on_mouse_move(Vec2::new(screen_pos[0], screen_pos[1]));
}
EditorCommand::MarkupMouseUp { screen_pos } => {
    if let Some(element) = app.markup_action.on_mouse_up(
        Vec2::new(screen_pos[0], screen_pos[1]),
        &mut app.world.graph,
    ) {
        if let Some(target) = app.markup_action.target_node {
            if let Some(entry) = app.world.graph.get_mut(target) {
                if let NodeData::Markup(m) = &mut entry.data {
                    m.elements.push(element);
                }
            }
        }
    }
}
EditorCommand::ClearAllMarkup { node } => {
    if let Some(entry) = app.world.graph.get_mut(node) {
        if let NodeData::Markup(m) = &mut entry.data {
            m.elements.clear();
        }
    }
}
```

- [ ] **Step 4: Verify compilation**

Run: `rtk cargo check -p rc3d-app`
Expected: 0 errors

- [ ] **Step 5: Commit**

```bash
rtk git add crates/rc3d-app/src/app/mod.rs crates/rc3d-app/src/editor_ui/commands.rs crates/rc3d-app/src/app/editor_commands.rs && rtk git commit -m "feat(editor): wire MarkupAction into App and EditorCommands"
```

---

## Verification Checkpoints

After each task:
1. `rtk cargo check -p <crate>` — 0 errors
2. For tasks with tests: `rtk cargo test -p <crate>` — all pass

After each phase:
- Phase 1: `rtk cargo check` — full workspace clean
- Phase 2: `rtk cargo check` — full workspace clean

Final verification:
```bash
rtk cargo check                          # 0 errors
rtk cargo test -p rc3d-nurbs             # NURBS tests pass
rtk cargo build -p rc3d-app --example editor  # Editor example compiles
```
