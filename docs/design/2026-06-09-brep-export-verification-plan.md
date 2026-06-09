# BREP 导出与 OCC 对齐验证 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐 BRepStore 三个缺口，实现 OCC BREP ASCII 导出，建立 OCC ground truth 验证循环。

**Architecture:** 三个独立阶段——(1) BRepStore 数据模型补齐 (D1-D3)，(2) `.brep` ASCII 写入器， (3) 与 OCC ground truth 的 diff 验证工具链。

**Tech Stack:** Rust, rc3d-shape B-Rep kernel, OCC BREP ASCII 格式

---

## Phase 1: BRepStore 数据模型补齐

### Task 1: BRepEdge 加 t_min / t_max 字段 (D1)

**Files:**
- Modify: `crates/rc3d-shape/src/topo.rs:24-34`
- Modify: `crates/rc3d-shape/src/store.rs:119,160`
- Modify: 36 处 BRepEdge 构造（见下方清单）

**背景:** 38 个 BRepEdge 构造站点都需要添加 `t_min: 0.0, t_max: 1.0`。所有 `.v_low`、`.v_high`、`.curve` 读取站点不受影响（无 `..` 解构，所有字段访问均命名）。

- [ ] **Step 1: 修改 BRepEdge 结构体定义**

在 `crates/rc3d-shape/src/topo.rs`，于 `v_high` 之后添加字段：

```rust
pub struct BRepEdge {
    pub curve: CurveGeom,
    pub tolerance: f32,
    pub v_low: VertexKey,
    pub v_high: VertexKey,
    /// Parameter interval on the 3D curve: always t_min < t_max.
    /// Reverse traversal is expressed by Orientation at the wire level.
    pub t_min: f32,
    pub t_max: f32,
    pub pcurves: HashMap<FaceKey, (Curve2d, bool)>,  // D2 change — see Task 2
}
```

- [ ] **Step 2: 运行编译确认所有构造站点**

```bash
rtk cargo check -p rc3d-shape -p rc3d-io 2>&1 | head -200
```

预期：~38 个错误，指向缺少 `t_min` / `t_max` 的 BRepEdge 构造。

- [ ] **Step 3: 批量修 38 个构造站点**

所有站点加 `t_min: 0.0, t_max: 1.0`。清单：

| 文件 | 行号 | 数量 |
|------|------|------|
| `crates/rc3d-shape/src/store.rs` | 119, 160 | 2 |
| `crates/rc3d-shape/src/bool/split.rs` | 265 | 1 |
| `crates/rc3d-shape/src/bool/intersect.rs` | 776-779, 799-802 | 8 |
| `crates/rc3d-shape/src/bool/pave_filler.rs` | 311 | 1 |
| `crates/rc3d-shape/src/topo_iter.rs` | 318, 326, 333, 340 | 4 |
| `crates/rc3d-shape/src/mesh/same_param.rs` | 90 | 1 |
| `crates/rc3d-shape/src/heal/check.rs` | 937 | 1 |
| `crates/rc3d-shape/src/heal/edge_tolerance.rs` | 182, 231 | 2 |
| `crates/rc3d-shape/src/heal/pcurve_fix.rs` | 392, 396, 400, 404, 467, 519, 564 | 7 |
| `crates/rc3d-shape/src/heal/same_param_fix.rs` | 576, 733 | 2 |
| `crates/rc3d-shape/src/heal/same_param_reparam.rs` | 246 | 1 |
| `crates/rc3d-io/src/step/brep/fillet.rs` | 185, 192, 322, 385 | 4 |
| `crates/rc3d-io/src/step/brep/quality.rs` | 133, 142, 151 | 3 |
| `crates/rc3d-io/src/step/brep/same_parameter.rs` | 131 | 1 |

- [ ] **Step 4: 运行测试**

```bash
rtk cargo test --workspace
```

预期：全绿，计数不降。

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat(brep): add t_min/t_max fields to BRepEdge"
```

---

### Task 2: PCurve 加 same_sense 方向 (D2)

**Files:**
- Modify: `crates/rc3d-shape/src/topo.rs:33`
- Modify: `crates/rc3d-shape/src/store.rs` — `add_edge_with_pcurve`, `add_seam_edge`, `pcurve_mut`, `set_pcurve`, `apply_edit`
- Modify: `crates/rc3d-shape/src/mesh/edge_disc.rs:164,191,238-241,282-285`
- Modify: `crates/rc3d-shape/src/mesh/edge_pool.rs:341,440`
- Modify: `crates/rc3d-shape/src/mesh/face_cdt.rs:105-107,188-190`
- Modify: `crates/rc3d-shape/src/mesh/same_param.rs:20-47`
- Modify: `crates/rc3d-shape/src/mesh/solid_mesh.rs:96`
- Modify: `crates/rc3d-shape/src/mesh/t4_quality.rs:125`
- Modify: `crates/rc3d-shape/src/heal/check.rs:216,338,436,509`
- Modify: `crates/rc3d-shape/src/heal/curve_trim.rs:120`
- Modify: `crates/rc3d-shape/src/heal/edge_tolerance.rs:38,51`
- Modify: `crates/rc3d-shape/src/heal/face_fix.rs:114`
- Modify: `crates/rc3d-shape/src/heal/self_intersect.rs:54`
- Modify: `crates/rc3d-shape/src/heal/free_bounds.rs:108-114,135-141`
- Modify: `crates/rc3d-shape/src/heal/degenerated.rs:70,284,330,405`
- Modify: `crates/rc3d-shape/src/heal/pcurve_fix.rs:44,64`
- Modify: `crates/rc3d-shape/src/heal/same_param_reparam.rs:38,62,99`
- Modify: `crates/rc3d-shape/src/heal/same_param_fix.rs:48`
- Modify: `crates/rc3d-shape/src/heal/intersecting_wires.rs:166,204,212`
- Modify: `crates/rc3d-shape/src/heal/seam.rs:171,225,405`
- Modify: `crates/rc3d-shape/src/heal/lacking.rs:116`
- Modify: `crates/rc3d-shape/src/heal/wire_join.rs:167,298,302`
- Modify: `crates/rc3d-shape/src/heal/shell_fix.rs:223,232`
- Modify: `crates/rc3d-shape/src/heal/topo_diag.rs:172`
- Modify: `crates/rc3d-shape/src/heal/mod.rs:641` (pcurve_match_tol)
- Modify: `crates/rc3d-shape/src/bool/classify.rs:361`
- Modify: `crates/rc3d-shape/src/bool/builder_face.rs:124,230`
- Modify: `crates/rc3d-io/src/step/brep/same_parameter.rs:63,71`
- Modify: `crates/rc3d-io/src/step/brep/overlay.rs:216-220`
- Modify: `crates/rc3d-io/src/step/brep/build/shell.rs:72`
- Modify: ~35 处测试站点（HashMap 构造）

- [ ] **Step 1: 修改 pcurves 字段类型**

在 `crates/rc3d-shape/src/topo.rs:33`：

```rust
pub pcurves: HashMap<FaceKey, (Curve2d, bool)>,  // (curve, same_sense)
```

- [ ] **Step 2: 修改 store.rs 中的核心 API**

`add_edge_with_pcurve` 签名改为接受 `(Curve2d, bool)`：

```rust
pub fn add_edge_with_pcurve(
    &mut self,
    v_start: VertexKey,
    v_end: VertexKey,
    curve: CurveGeom,
    tolerance: f32,
    face: FaceKey,
    pcurve: (Curve2d, bool),  // was: pcurve: Curve2d
) -> EdgeKey
```

`add_seam_edge` 同理。

`pcurve_mut` 返回类型改为 `Option<&mut (Curve2d, bool)>`。

`set_pcurve` 接受并返回 `(Curve2d, bool)`。

- [ ] **Step 3: 运行编译，收集所有错误**

```bash
rtk cargo check -p rc3d-shape -p rc3d-io 2>&1
```

预期：~80+ 个错误，每个指向 `.pcurves` 访问点。

- [ ] **Step 4: 逐文件修复 .pcurves 读取站点**

对所有 `.get(&face_key)` 读取，解构元组：

```rust
// Before
let pcurve = edge.pcurves.get(&face_key)?;
// After
let (pcurve, _same_sense) = edge.pcurves.get(&face_key)?;
```

对所有迭代：

```rust
// Before
for (&face_key, pcurve) in &edge.pcurves { ... }
// After
for (&face_key, (pcurve, _same_sense)) in &edge.pcurves { ... }
```

- [ ] **Step 5: 修复 HashMap 构造站点（~35 个）**

```rust
// Before
pcurves: HashMap::from([(face_key, Curve2d::Line { ... })])
// After
pcurves: HashMap::from([(face_key, (Curve2d::Line { ... }, true))])
```

空 HashMap：`pcurves: HashMap::new()` 无需修改（类型推断自动适配）。

- [ ] **Step 6: 修复 STEP 导入调用点**

`crates/rc3d-io/src/step/brep/build/shell.rs:72`：

```rust
// Before
ctx.reg.add_edge_with_pcurve(v0, v1, curve, ctx.tol, face_key,
    rc3d_shape::geom::curve2d::Curve2d::from_pcurve_3d(&pcurve));
// After
ctx.reg.add_edge_with_pcurve(v0, v1, curve, ctx.tol, face_key,
    (rc3d_shape::geom::curve2d::Curve2d::from_pcurve_3d(&pcurve), true));
```

- [ ] **Step 7: 运行全量测试**

```bash
rtk cargo test --workspace
```

预期：全绿，405 tests，0 failures。

- [ ] **Step 8: 提交**

```bash
git add -A && git commit -m "feat(brep): add same_sense field to PCurve storage"
```

---

### Task 3: CurveGeom 加 BezierCurve 变体 (D3)

**Files:**
- Modify: `crates/rc3d-shape/src/geom/curve_eval.rs:327-800`
- Modify: `crates/rc3d-shape/src/geom/curve2d.rs:268-329`
- Modify: `crates/rc3d-io/tests/shape_topology.rs:111-143`

**背景:** 6 个穷举 match 站点会导致编译错误。其余 23 个使用 wildcard，不受影响。

- [ ] **Step 1: 在 CurveGeom 枚举加变体 + 构造器**

`crates/rc3d-shape/src/geom/curve_eval.rs:333`，在 BSpline 之后：

```rust
/// Bezier curve of arbitrary degree with optional rational weights.
/// Equivalent to BSpline with knot vector [0ⁿ⁺¹, 1ⁿ⁺¹] but evaluated
/// via De Casteljau for better performance and OCC type-6 compatibility.
BezierCurve {
    degree: usize,
    control_points: Vec<Vec3>,
    weights: Option<Vec<f32>>,
},
```

构造器：

```rust
/// Construct a Bezier curve with optional rational weights.
pub fn bezier(control_points: Vec<Vec3>, weights: Option<Vec<f32>>) -> Self {
    let degree = control_points.len().saturating_sub(1);
    CurveGeom::BezierCurve { degree, control_points, weights }
}
```

- [ ] **Step 2: 实现 De Casteljau 求值的 helper 函数**

```rust
/// De Casteljau evaluation of a Bezier curve at parameter t ∈ [0, 1].
fn de_casteljau_d0(points: &[Vec3], weights: Option<&[f32]>, t: f32) -> Vec3 {
    let n = points.len();
    if n == 0 { return Vec3::ZERO; }
    if n == 1 { return points[0]; }
    if let Some(w) = weights {
        // Rational De Casteljau
        let mut p: Vec<(f32, Vec3)> = points.iter().zip(w.iter())
            .map(|(&pt, &wt)| (wt, pt * wt))
            .collect();
        for r in 1..n {
            for i in 0..(n - r) {
                let w_new = p[i].0 * (1.0 - t) + p[i + 1].0 * t;
                let v_new = p[i].1 * (1.0 - t) + p[i + 1].1 * t;
                p[i] = (w_new, v_new);
            }
        }
        p[0].1 / p[0].0
    } else {
        // Polynomial De Casteljau
        let mut p = points.to_vec();
        for r in 1..n {
            for i in 0..(n - r) {
                p[i] = p[i] * (1.0 - t) + p[i + 1] * t;
            }
        }
        p[0]
    }
}

fn de_casteljau_d1(points: &[Vec3], weights: Option<&[f32]>, t: f32) -> Vec3 {
    let n = points.len();
    if n <= 1 { return Vec3::ZERO; }
    let degree = (n - 1) as f32;
    // Derivative via control-point differences
    let mut diff: Vec<Vec3> = points.windows(2).enumerate().map(|(i, w)| {
        if let Some(ws) = weights {
            (w[1] * ws[i + 1] - w[0] * ws[i]) * degree
        } else {
            (w[1] - w[0]) * degree
        }
    }).collect();
    if let Some(w) = weights {
        // Rational derivative evaluation
        let mut p_val: Vec<(f32, Vec3)> = points.iter().zip(w.iter())
            .map(|(&pt, &wt)| (wt, pt * wt)).collect();
        let n_p = points.len();
        for r in 1..n_p {
            for i in 0..(n_p - r) {
                let w_new = p_val[i].0 * (1.0 - t) + p_val[i + 1].0 * t;
                let v_new = p_val[i].1 * (1.0 - t) + p_val[i + 1].1 * t;
                p_val[i] = (w_new, v_new);
            }
        }
        let p = p_val[0].1 / p_val[0].0;
        let d: Vec<f32> = w.iter().copied().collect();
        // Compute w(t), w'(t)
        let wt = de_casteljau_d0_scalar(&d, t);
        let wdt = de_casteljau_d1_scalar(&d, t);
        de_casteljau_d0(&diff, None, t) / wt - p * (wdt / (wt * wt))
    } else {
        de_casteljau_d0(&diff, None, t)
    }
}
```

- [ ] **Step 3: 在 d0() 加 BezierCurve 分支**

`curve_eval.rs:400`，在 BSpline 分支之后：

```rust
CurveGeom::BezierCurve { control_points, weights, .. } => {
    de_casteljau_d0(control_points, weights.as_deref(), t)
}
```

- [ ] **Step 4: 在 d1() 加 BezierCurve 分支**

```rust
CurveGeom::BezierCurve { control_points, weights, .. } => {
    de_casteljau_d1(control_points, weights.as_deref(), t)
}
```

- [ ] **Step 5: 在 d2() 加 BezierCurve 分支** — 用有限差分解 d1

```rust
CurveGeom::BezierCurve { .. } => {
    let eps = 1e-4;
    (self.d1(t + eps) - self.d1(t - eps)) / (2.0 * eps)
}
```

- [ ] **Step 6: 在 d012() 加 BezierCurve 分支**

```rust
CurveGeom::BezierCurve { control_points, weights, .. } => {
    let p = de_casteljau_d0(control_points, weights.as_deref(), t);
    let d1 = de_casteljau_d1(control_points, weights.as_deref(), t);
    let eps = 1e-4;
    let d1_plus = de_casteljau_d1(control_points, weights.as_deref(), (t + eps).min(1.0));
    let d1_minus = de_casteljau_d1(control_points, weights.as_deref(), (t - eps).max(0.0));
    let d2 = (d1_plus - d1_minus) / (2.0 * eps);
    (p, d1, d2)
}
```

- [ ] **Step 7: 在 curve2d.rs 的 from_pcurve_3d() 加 BezierCurve 分支**

`crates/rc3d-shape/src/geom/curve2d.rs:268`，在 BSpline 分支之后：

```rust
CurveGeom::BezierCurve { .. } => {
    None // Bezier curves don't have a natural 2D projection; caller handles
}
```

- [ ] **Step 8: 在 shape_topology.rs 测试的 disc_val 加分支**

`crates/rc3d-io/tests/shape_topology.rs:111`：

```rust
CurveGeom::BezierCurve { .. } => "Bézier",
```

- [ ] **Step 9: 运行测试**

```bash
rtk cargo test --workspace
```

预期：全绿。

- [ ] **Step 10: 提交**

```bash
git add -A && git commit -m "feat(brep): add BezierCurve variant to CurveGeom with De Casteljau evaluation"
```

---

## Phase 2: OCC BREP ASCII 写入器

### Task 4: 写入器骨架 + 拓扑节

**Files:**
- Create: `crates/rc3d-shape/src/brep/mod.rs`
- Create: `crates/rc3d-shape/src/brep/write.rs`

- [ ] **Step 1: 创建 brep 模块 + 注册**

`crates/rc3d-shape/src/brep/mod.rs`：

```rust
//! OCC BREP ASCII format export.
mod write;
pub use write::write_brep;
```

在 `crates/rc3d-shape/src/lib.rs` 添加：

```rust
pub mod brep;
```

- [ ] **Step 2: 写 write.rs 骨架 + 拓扑节**

`crates/rc3d-shape/src/brep/write.rs`：

```rust
//! OCC BREP ASCII format writer.
//! Format reference: Open CASCADE Technology BRepTools::Write()
//! OCC BREP ASCII has sections in strict order.

use std::collections::HashMap;
use std::io::{self, Write};
use crate::geom::{CurveGeom, SurfaceGeom};
use crate::geom::curve2d::Curve2d;
use crate::store::BRepStore;
use crate::topo::*;

/// Write the full BRepStore as OCC BREP ASCII.
pub fn write_brep(store: &BRepStore, output: &mut impl Write) -> io::Result<()> {
    let mut w = BrepWriter::new(store);
    w.write_all(output)
}

struct BrepWriter<'a> {
    store: &'a BRepStore,
    /// CurveGeom -> sequential index (1-based, OCC convention)
    curve_indices: HashMap<usize, usize>,    // slotmap key index -> curve index
    surface_indices: HashMap<usize, usize>,  // slotmap key index -> surface index
    curve_count: usize,
    surface_count: usize,
    pcurve_count: usize,
}

impl<'a> BrepWriter<'a> {
    fn new(store: &'a BRepStore) -> Self {
        Self {
            store,
            curve_indices: HashMap::new(),
            surface_indices: HashMap::new(),
            curve_count: 0,
            surface_count: 0,
            pcurve_count: 0,
        }
    }

    fn write_all(&mut self, output: &mut impl Write) -> io::Result<()> {
        writeln!(output, "DBRep_DrawableShape")?;
        writeln!(output)?;

        // 1. Locations — all identity
        self.write_locations(output)?;
        // 2. Geometry: Curves 3D
        self.write_curves3d(output)?;
        // 3. Geometry: Surfaces
        self.write_surfaces(output)?;
        // 4. Curve2D (PCurves)
        self.write_pcurves(output)?;
        // 5. Topology: Vertices
        self.write_vertices(output)?;
        // 6. Topology: Edges
        self.write_edges(output)?;
        // 7. Topology: Wires
        self.write_wires(output)?;
        // 8. Topology: Faces
        self.write_faces(output)?;
        // 9. Topology: Shells
        self.write_shells(output)?;
        // 10. Topology: Solids
        self.write_solids(output)?;
        // 11. Topology: Compounds
        self.write_compounds(output)?;

        Ok(())
    }
}
```

- [ ] **Step 3: 实现 write_locations — 全 identity**

```rust
fn write_locations(&self, output: &mut impl Write) -> io::Result<()> {
    writeln!(output, "Locations 0")?;
    writeln!(output)
}
```

- [ ] **Step 4: 实现拓扑节**

顶点、边、wire、面、shell、solid、compound，按 OCC 格式写入。每节包含实体计数 + 逐实体数据。

```rust
fn write_vertices(&self, output: &mut impl Write) -> io::Result<()> {
    let count = self.store.vertices.len();
    writeln!(output, "TVertexes {}", count)?;
    for (vk, v) in &self.store.vertices {
        // OCC format: tolerance  location_index  1  X Y Z
        writeln!(output, "{}  0  1  {} {} {}",
            v.tolerance, v.position.x, v.position.y, v.position.z)?;
    }
    writeln!(output)?;
    Ok(())
}
```

- [ ] **Step 5: 运行编译**

```bash
rtk cargo check -p rc3d-shape
```

- [ ] **Step 6: 提交**

```bash
git add -A && git commit -m "feat(brep): add OCC BREP ASCII writer skeleton with topology sections"
```

---

### Task 5: 几何节写入（曲线 + 曲面）

**Files:**
- Modify: `crates/rc3d-shape/src/brep/write.rs`

- [ ] **Step 1: 实现 write_curves3d**

对照设计文档的类型映射表，写每种曲线类型。Trimmed 展开为基础曲线 + Edge t_min/t_max。

```rust
fn write_curves3d(&mut self, output: &mut impl Write) -> io::Result<()> {
    // First pass: collect unique curves
    let mut curves: Vec<(usize, &CurveGeom)> = Vec::new();
    for (ek, edge) in &self.store.edges {
        let cid = ek.data().as_ffi() as usize;
        if !self.curve_indices.contains_key(&cid) {
            self.curve_count += 1;
            self.curve_indices.insert(cid, self.curve_count);
            curves.push((cid, &edge.curve));
        }
    }
    writeln!(output, "Curve3ds {}", self.curve_count)?;

    for (cid, curve) in &curves {
        match self.expand_curve(curve) {
            // ... per-type formatting
        }
        // ...
    }
    writeln!(output)?;
    Ok(())
}
```

每条曲线的格式举例：

```
1 0 0 1 0 0 1 0      # Line: origin(0,0,0) direction(1,0,0)
1 0 0 1.0 0 0 1 0 0 0 0 1  # Circle: origin(0,0,1) radius(1.0) axis(0,0,1) x_dir(1,0,0) y_dir(0,1,0)
```

Polyline/Composite 边第一期写为类型 9，附 `-- TODO: expand to individual edges` 注释：

```rust
CurveGeom::Polyline { points } | CurveGeom::Composite { .. } => {
    writeln!(output, "9  -- TODO: expand Polyline/Composite to individual edges")?;
}
```

- [ ] **Step 2: 实现 write_surfaces**

```rust
fn write_surfaces(&mut self, output: &mut impl Write) -> io::Result<()> {
    // Collect unique surfaces from faces
    let mut surfaces: Vec<(usize, &SurfaceGeom)> = Vec::new();
    for (fk, face) in &self.store.faces {
        let sid = fk.data().as_ffi() as usize;
        if !self.surface_indices.contains_key(&sid) {
            self.surface_count += 1;
            self.surface_indices.insert(sid, self.surface_count);
            surfaces.push((sid, &face.surface));
        }
    }
    writeln!(output, "Surfaces {}", self.surface_count)?;

    for (sid, surface) in &surfaces {
        match surface {
            SurfaceGeom::Plane { origin, normal, u_dir } => {
                // Type 1: origin normal u_dir
                writeln!(output, "1 {} {} {}  {} {} {}  {} {} {}",
                    origin.x, origin.y, origin.z,
                    normal.x, normal.y, normal.z,
                    u_dir.x, u_dir.y, u_dir.z)?;
            }
            // ... Cylinder, Cone, Sphere, Torus, BSpline, Extrusion, Revolution, Offset
        }
    }
    writeln!(output)?;
    Ok(())
}
```

- [ ] **Step 3: 运行编译 + 提交**

```bash
rtk cargo check -p rc3d-shape
git add -A && git commit -m "feat(brep): add geometry sections (curves + surfaces) to BREP writer"
```

---

### Task 6: PCurve 节 + 边完整性

**Files:**
- Modify: `crates/rc3d-shape/src/brep/write.rs`

- [ ] **Step 1: 实现 write_pcurves**

```rust
fn write_pcurves(&mut self, output: &mut impl Write) -> io::Result<()> {
    // Count all pcurves across all edges
    let mut pcurve_entries: Vec<(usize, FaceKey, &Curve2d, bool)> = Vec::new();
    for (ek, edge) in &self.store.edges {
        let cid = ek.data().as_ffi() as usize;
        for (&fk, (pc, same_sense)) in &edge.pcurves {
            let fid = fk.data().as_ffi() as usize;
            pcurve_entries.push((cid, fk, pc, *same_sense));
        }
    }
    writeln!(output, "Curve2ds {}", pcurve_entries.len())?;

    for (cid, fk, pc, same_sense) in &pcurve_entries {
        let ci = self.curve_indices.get(cid).unwrap_or(&0);
        let fi = self.surface_indices.get(&(fk.data().as_ffi() as usize)).unwrap_or(&0);
        let orientation = if *same_sense { 1 } else { 0 };
        // Format: curve_index face_index orientation  then pcurve geometric data
        write!(output, "{} {} {}  ", ci, fi, orientation)?;
        match pc {
            Curve2d::Line { origin, direction } => {
                writeln!(output, "1 {} {}  {} {}", origin.0, origin.1, direction.0, direction.1)?;
            }
            // ... Circle, Ellipse, BSpline, Trimmed
        }
    }
    writeln!(output)?;
    Ok(())
}
```

- [ ] **Step 2: Wire 一致性检查**

在 `write_wires` 中对每条 Wire 的边做首尾相连检查：

```rust
fn check_wire_chain(&self, wire_key: WireKey) -> Vec<String> {
    let wire = match self.store.wires.get(wire_key) {
        Some(w) => w,
        None => return vec![],
    };
    let mut warnings = Vec::new();
    for i in 0..wire.edges.len() {
        let j = (i + 1) % wire.edges.len();
        let (ek_i, ori_i) = &wire.edges[i];
        let (ek_j, ori_j) = &wire.edges[j];
        let edge_i = self.store.edges.get(*ek_i);
        let edge_j = self.store.edges.get(*ek_j);
        if let (Some(ei), Some(ej)) = (edge_i, edge_j) {
            let end_i = match ori_i {
                Orientation::Forward => ei.v_high,
                Orientation::Reversed => ei.v_low,
            };
            let start_j = match ori_j {
                Orientation::Forward => ej.v_low,
                Orientation::Reversed => ej.v_high,
            };
            if end_i != start_j {
                warnings.push(format!(
                    "-- WARNING: broken wire chain at edge #{} -> #{}",
                    i, j
                ));
            }
        }
    }
    warnings
}
```

- [ ] **Step 3: 运行编译 + 提交**

```bash
rtk cargo check -p rc3d-shape
git add -A && git commit -m "feat(brep): add PCurve section + wire chain checks to BREP writer"
```

---

### Task 7: 端到端 .brep 写盘测试

**Files:**
- Create: `crates/rc3d-io/tests/brep_export.rs`

- [ ] **Step 1: 写测试——Cube.step → BRepStore → .brep**

```rust
//! BREP ASCII export test: import STEP, write .brep, verify output structure.

use rc3d_io::step::{parse_step, StepImportOptions};
use rc3d_shape::brep::write_brep;
use std::io::Write;

#[test]
fn brep_export_cube_has_expected_sections() {
    let step = include_str!("../../../test_data/Cube.step");
    let result = rc3d_io::step::import_step_with_options(
        step,
        &StepImportOptions::default(),
    ).expect("import Cube.step");

    let store = &result.document.store;
    let mut buf = Vec::new();
    write_brep(store, &mut buf).expect("write brep");

    let text = String::from_utf8(buf).expect("valid utf8");
    // Verify all required sections exist
    assert!(text.contains("DBRep_DrawableShape"), "missing header");
    assert!(text.contains("Locations 0"), "missing locations");
    assert!(text.contains("Curve3ds"), "missing curves");
    assert!(text.contains("Surfaces"), "missing surfaces");
    assert!(text.contains("Curve2ds"), "missing pcurves");
    assert!(text.contains("TVertexes"), "missing vertices");
    assert!(text.contains("TEdges"), "missing edges");
    assert!(text.contains("TWires"), "missing wires");
    assert!(text.contains("TFaces"), "missing faces");
    assert!(text.contains("TShells"), "missing shells");
    assert!(text.contains("TSolids"), "missing solids");
}

#[test]
fn brep_export_cube_writes_file() {
    let step = include_str!("../../../test_data/Cube.step");
    let result = rc3d_io::step::import_step_with_options(
        step,
        &StepImportOptions::default(),
    ).expect("import Cube.step");

    let store = &result.document.store;
    let path = std::env::temp_dir().join("cube_test.brep");
    let mut file = std::fs::File::create(&path).expect("create file");
    write_brep(store, &mut file).expect("write brep");

    let file_size = std::fs::metadata(&path).unwrap().len();
    assert!(file_size > 100, "brep file too small: {} bytes", file_size);
    println!("Cube.brep: {} bytes", file_size);
}
```

- [ ] **Step 2: 运行测试 + 修复问题**

```bash
rtk cargo test -p rc3d-io -- brep_export
```

预期：PASS。

- [ ] **Step 3: 提交**

```bash
git add -A && git commit -m "test(brep): add end-to-end .brep export test with Cube.step"
```

---

## Phase 3: 验证工具链

### Task 8: 结构差异对比工具

**Files:**
- Create: `crates/rc3d-io/tests/brep_diff.rs`

- [ ] **Step 1: 实现拓扑计数提取 + 比较**

```rust
use rc3d_io::step::{StepImportOptions, StepImportResult};
use std::collections::HashMap;

#[derive(Debug, Default)]
struct TopoCounts {
    vertices: usize,
    edges: usize,
    wires: usize,
    faces: usize,
    shells: usize,
    solids: usize,
    compounds: usize,
    pcurves_total: usize,
}

fn count_topo(result: &StepImportResult) -> TopoCounts {
    let store = &result.document.store;
    let pcurves_total: usize = store.edges.values()
        .map(|e| e.pcurves.len())
        .sum();
    TopoCounts {
        vertices: store.vertices.len(),
        edges: store.edges.len(),
        wires: store.wires.len(),
        faces: store.faces.len(),
        shells: store.shells.len(),
        solids: store.solids.len(),
        compounds: store.compounds.len(),
        pcurves_total,
    }
}

/// Diff two counts. Returns a score: 1.0 = identical, 0.0 = completely different.
fn diff_counts(a: &TopoCounts, b: &TopoCounts) -> f32 {
    let fields: [(usize, usize); 8] = [
        (a.vertices, b.vertices),
        (a.edges, b.edges),
        (a.wires, b.wires),
        (a.faces, b.faces),
        (a.shells, b.shells),
        (a.solids, b.solids),
        (a.compounds, b.compounds),
        (a.pcurves_total, b.pcurves_total),
    ];
    let mut matched = 0;
    for (av, bv) in &fields {
        if av == bv { matched += 1; }
    }
    matched as f32 / fields.len() as f32
}
```

- [ ] **Step 2: 运行编译 + 提交**

```bash
rtk cargo check -p rc3d-io
git add -A && git commit -m "feat(verify): add structure diff tool for BREP topology counts"
```

---

### Task 9: 批量验证 Loop 框架

**Files:**
- Create: `crates/rc3d-io/tests/brep_verify_loop.rs`

- [ ] **Step 1: 实现批量验证 harness**

```rust
//! BREP verification loop: import STEP files, compare with OCC ground truth.

use rc3d_io::step::{StepImportOptions, parse_step};
use rc3d_io::step::import_pipeline::run_heal_pipeline;
use rc3d_shape::brep::write_brep;
use std::path::{Path, PathBuf};

/// Files used for verification (from export_step_stl corpus).
const CORPUS: &[&str] = &[
    "Shape.step",
    "Shape-1.step",
    "Shape-2.step",
    "Cube.step",
    "cs.step",
    "OffsetPlaneHoleEdge.step",
    "asse.step",
    "HoledPlate.step",
];

/// Resolve test data path (same convention as export_step_stl).
fn test_data_path(name: &str) -> PathBuf {
    let primary = std::env::var("RC3D_STEP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../steps"));
    let candidate = primary.join(name);
    if candidate.exists() { return candidate; }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

#[test]
fn brep_verify_all_files_import() {
    for name in CORPUS {
        let path = test_data_path(name);
        if !path.exists() {
            eprintln!("SKIP: {} not found", name);
            continue;
        }
        let result = rc3d_io::step::import_step_file_with_options(
            &path, &StepImportOptions::default(),
        );
        match result {
            Ok(r) => {
                let topo = count_topo(&r);
                println!("{}: {:?}", name, topo);
            }
            Err(e) => {
                println!("FAIL {}: {}", name, e);
            }
        }
    }
}

/// Write .brep for all corpus files.
#[test]
fn brep_write_all_corpus() {
    let out_dir = PathBuf::from("tests/output/brep");
    std::fs::create_dir_all(&out_dir).ok();

    for name in CORPUS {
        let path = test_data_path(name);
        if !path.exists() { continue; }
        let result = rc3d_io::step::import_step_file_with_options(
            &path, &StepImportOptions::default(),
        );
        if let Ok(r) = result {
            let brep_path = out_dir.join(name).with_extension("brep");
            let mut f = std::fs::File::create(&brep_path).unwrap();
            write_brep(&r.document.store, &mut f).expect("write brep");
            let size = std::fs::metadata(&brep_path).unwrap().len();
            println!("{} -> {} ({} bytes)", name, brep_path.display(), size);
        }
    }
}
```

- [ ] **Step 2: 运行 Loop harness**

```bash
rtk cargo test -p rc3d-io -- brep_verify_all_files_import
rtk cargo test -p rc3d-io -- brep_write_all_corpus
```

预期：8 个文件全导入，全生成 .brep。

- [ ] **Step 3: 提交**

```bash
git add -A && git commit -m "feat(verify): add batch verification harness for 8-corpus BREP export"
```

---

### Task 10: 几何属性差异对比

**Files:**
- Modify: `crates/rc3d-io/tests/brep_diff.rs`

- [ ] **Step 1: 实现几何属性计算**

```rust
#[derive(Debug, Default)]
struct GeomProps {
    face_areas: Vec<f32>,
    edge_lengths: Vec<f32>,
    vertex_positions: Vec<[f32; 3]>,
}

fn compute_geom_props(result: &StepImportResult) -> GeomProps {
    let store = &result.document.store;
    let face_areas: Vec<f32> = store.faces.values()
        .map(|f| crate::brep::surface_area::face_surface_area(&f.surface))
        .collect();
    let edge_lengths: Vec<f32> = store.edges.values()
        .map(|e| crate::geom::curve_eval::estimate_curve_length(&e.curve))
        .collect();
    let vertex_positions: Vec<[f32; 3]> = store.vertices.values()
        .map(|v| [v.position.x, v.position.y, v.position.z])
        .collect();
    GeomProps { face_areas, edge_lengths, vertex_positions }
}
```

- [ ] **Step 2: 实现宽松 ε 比较**

```rust
fn diff_geom(a: &GeomProps, b: &GeomProps) -> GeomDiff {
    let area_eps = 0.001; // 0.1%
    let length_eps = 0.001;
    let pos_eps = 1e-4;

    let area_match = compare_lists(&a.face_areas, &b.face_areas, area_eps);
    let length_match = compare_lists(&a.edge_lengths, &b.edge_lengths, length_eps);
    let pos_match = compare_positions(&a.vertex_positions, &b.vertex_positions, pos_eps);

    GeomDiff {
        face_area_score: area_match,
        edge_length_score: length_match,
        vertex_position_score: pos_match,
    }
}
```

- [ ] **Step 3: 提交**

```bash
git add -A && git commit -m "feat(verify): add geometry properties diff with epsilon thresholds"
```

---

## 验证清单

完成所有 Phase 之后：

- [ ] `rtk cargo test --workspace` 全绿，≥405 tests
- [ ] 8 个 STEP 文件全导入，无 panic
- [ ] 8 个 .brep 文件成功写盘，字节数 > 100
- [ ] 每个 .brep 文件包含全部 11 个节
- [ ] 无断链 WARNING（或 WARNING 数量已记录追踪）
- [ ] 拓扑计数与 OCC ground truth 对比记录入 `brep_verify_loop` 输出
