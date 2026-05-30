# STEP B-Rep OCC 算法对齐 — 3 阶段改进设计

**日期**: 2026-05-30
**状态**: 待审批
**范围**: rc3d-io STEP 解析 → BRep 可视化管线中与 OCC 参考实现的算法差异修复

---

## 背景

`/simplify` 审查识别出 rc3d STEP 管线与 OCC (OpenCASCADE) 参考实现之间存在 7 类算法差异。本设计将这些差异组织为 3 个独立可提交阶段，按风险递增排列。

**当前基线** (阶段 1 已完成):
- 365 项测试通过，0 失败
- STL 导出: Shape (32K tris), Shape-1 (29K tris, 6 grid 回退), Shape-2 (4.4K tris), cs (32K tris)
- 3 文件 +21/-74 行简化改动待提交

## 阶段 1：代码清理 ✅ (已完成，待提交)

合并重复函数、删除死代码。已验证通过。

### 已完成的改动

| 改动 | 文件 | 说明 |
|------|------|------|
| 合并 `find_param_on_curve` / `find_closest_t_on_curve` | `curve_eval.rs`, `surface_eval.rs` | 统一为带参数的 `pub fn` |
| 删除 Torus project 中的 `let _ = (...)` 死代码 | `surface_eval.rs` | 改 match 为 `{ .. }` |
| 删除 `trim_circle_to_vertices` 未使用的 `_match_tol` 参数 | `curve_eval.rs` | 清理 API |
| 删除 `inverse_native_uv_extrusion` 重复实现 | `surface_eval.rs` | 委托给 `project()` |

---

## 阶段 2：效率优化 + 正确性修复

将热路径效率问题与已知正确性缺陷一并修复。不引入算法级变更。

### 2.1 bspline_d012 栈分配

**问题**: 每次调用分配 `Vec<Vec<f32>>` + 3× `Vec<f32>`，热路径上 5-7 次堆分配。
**OCC 对标**: `BSplCLib` 使用栈数组。

**方案**:
- 引入 `const MAX_DEGREE: usize = 16`
- `ndu` 使用扁平数组 `[f32; (MAX_DEGREE+1)*(MAX_DEGREE+2)/2]`（153 个 f32 ≈ 612 字节）
- `ndu1`, `ndu2`, `ndu1_pm1` 使用 `[f32; MAX_DEGREE+1]`
- 超过 MAX_DEGREE 时回退到 Vec（极少见）
- 索引函数 `ndu_idx(k, i) = k*(k+1)/2 + i`

**文件**: `crates/rc3d-io/src/step/brep/geom/curve_eval.rs`
**测试**: 现有 B-spline 单元测试 + 全量 STL 导出回归

### 2.2 build_ortho_axes 缓存

**问题**: 每次 `d0`/`d1`/`d2`/`project` 都重新计算 `build_ortho_axes(axis)`。
**OCC 对标**: OCC 在曲面/曲线构造时预计算辅助轴。

**方案**:
- 在 `CurveGeom::Circle/Ellipse` 变体中增加 `x_dir: Vec3, y_dir: Vec3` 字段
- 在 `SurfaceGeom::Cylinder/Cone/Torus` 变体中增加 `x_dir: Vec3, y_dir: Vec3` 字段
- 构造时调用 `build_ortho_axes` 并存储结果
- `d0`/`d1`/`d2`/`project` 直接使用缓存的轴

**文件**: `curve_eval.rs`, `surface_eval.rs`, 以及所有构造 CurveGeom/SurfaceGeom 的调用点
**风险**: 中等 — 需更新所有构造点（`brep/build/curve.rs`, `brep/build/surface.rs`, `geom.rs`）

### 2.3 BSpline/Torus project() 分层搜索

**问题**: 16×16 网格 = 289 次求值 + 32 次细化 = 321 次/调用。
**OCC 对标**: `Extrema_ExtPS` 使用多起始点 + Newton-Raphson（本阶段仅做分层搜索，Newton 留到阶段 3）。

**方案**:
- 阶段 A: 4×4 粗网格（25 次求值），取 top-3 候选
- 阶段 B: 每个候选 4×4 局部细化（48 次求值）
- 阶段 C: 最佳候选坐标下降细化（保持现有 8 轮）
- 总计: 25 + 48 + 32 = 105 次，节省 ~65%

**文件**: `surface_eval.rs` — 提取 `grid_project_2d` 辅助函数，BSpline 和 Torus 共享

### 2.4 Composite 曲线弧长缓存

**问题**: `approx_chordal_length` 在每次 `find_composite_segment` 调用时重新计算所有段。
**OCC 对标**: OCC 在 `Geom_CompositeCurve` 中缓存段长度。

**方案**:
- `CurveGeom::Composite` 增加 `cached_lengths: Option<Vec<f32>>` 字段
- 首次调用 `find_composite_segment` 时计算并缓存（使用 `OnceCell` 或惰性计算）
- `approx_chordal_length` 改为接收可选的预计算长度

**文件**: `curve_eval.rs`
**风险**: 低 — 仅影响 Composite 变体

### 2.5 NURBS evaluate + derivative 共享基函数

**问题**: `evaluate()` 和 `derivative()` 独立计算 `find_span` + `bspline_bases`，每点 6× 冗余。
**OCC 对标**: OCC `Geom_BSplineSurface::D0/D1` 共享基函数计算。

**方案**:
- 在 `nurbs.rs` 中新增 `evaluate_with_derivative(u, v) -> (Vec3, Vec3, Vec3)` 方法
- 内部一次计算 `find_span` + `bspline_bases` 在 u 和 v 方向
- `SurfaceGeom::d1(BSpline)` 调用新方法替代分别调用
- 将 `compute_bspline_derivatives` 中的 `HashMap` 替换为 `SmallVec<[(usize, f32); 8]>`

**文件**: `crates/rc3d-io/src/step/nurbs.rs`, `surface_eval.rs`

### 2.6 椭圆修剪 — 离心角 Newton 迭代

**问题**: `trim_circle_to_vertices` 对 Ellipse 使用 `circle_angle_geom(semi_major)`，高离心率时静默失败。
**OCC 对标**: OCC `ElCLib::Parameter` 对椭圆求解离心角。

**方案**:
- 新增 `ellipse_angle_geom(center, axis, semi_major, semi_minor, point) -> Option<f32>`
- 将 3D 点投影到椭圆平面，计算 `u = proj.dot(x_dir)/semi_major`, `v = proj.dot(y_dir)/semi_minor`
- 离心角 `θ = atan2(v, u)`
- Newton 迭代 1-2 轮修正: `f(θ) = |P - E(θ)|²`, `f'(θ) = -2(P-E(θ))·E'(θ)`
- `trim_circle_to_vertices` 的 Ellipse 分支调用新方法

**文件**: `curve_eval.rs`
**测试**: 新增 `test_ellipse_trim_eccentric` — 验证 a=10, b=1 的椭圆

### 2.7 解析二阶导数

**问题**: `SurfaceGeom::d2` 对所有曲面使用有限差分（4 次额外 d1 调用 + 浮点噪声）。
**OCC 对标**: OCC 所有解析曲面有闭式 d2。

**方案** — 为每种解析曲面实现闭式二阶导数:

| 曲面 | ∂²S/∂u² | ∂²S/∂u∂v | ∂²S/∂v² |
|------|---------|----------|---------|
| Plane | 0 | 0 | 0 |
| Cylinder | `-TAU²·r·(cos·x+sin·y)` | 0 | 0 |
| Cone | 类似 Cylinder，含 tan(α) 项 | 含 sin/cos 交叉项 | 0 |
| Sphere | `-TAU²·r·sin(φ)·(cos·x+sin·y)` | `TAU·PI·r·cos(φ)·(-sin·x+cos·y)` | `-PI²·r·(sin·φ·...-cos·φ·z)` |
| Torus | 含 (R+r·cos(φ)) 项 | 含 -r·sin(φ) 交叉项 | 含 -r·cos(φ) 项 |
| BSpline/Extrusion/Revolution/Offset | 保持现有数值差分 | | |

**文件**: `surface_eval.rs` — 修改 `d2()` 方法，match 每个解析变体

### 2.8 Offset 曲面 Weingarten 映射

**问题**: `Offset::d1` 用有限差分求 ∂n/∂u, ∂n/∂v 并静默吞没 NaN。
**OCC 对标**: OCC 使用 Weingarten 方程: `dn/du = (fM-eL)/(eg-f²)·dS/du + (fL-eM)/(eg-f²)·dS/dv`

**方案**:
- 在 `Offset::d1` 中调用基曲面的 `d1` + `d2`
- 计算第一基本形式 (e,f,g) 和第二基本形式 (L,M,N)
- 用 Weingarten 方程计算 ∂n/∂u, ∂n/∂v
- 退化时 (`|eg-f²| < 1e-10`) 回退到数值差分，并 `log::warn` 记录（不再静默吞没 NaN）

**文件**: `surface_eval.rs`
**依赖**: 2.7（需要解析 d2 以获得准确结果）

### 2.9 normalize_edge_curve_to_vertices 容差改进

**问题**: `match_tol = (tol.max(1e-4) * 100.0).max(len * 0.005).max(0.01)` — 100× 放大过于宽松。
**OCC 对标**: OCC `ShapeFix_Edge` 使用边自身公差字段。

**方案**:
- 将 `match_tol` 改为 `tol.max(1e-4) * 10.0`（10× 而非 100×）
- 增加 `min(len * 0.001, 0.1)` 上限，防止长边过度容差
- 当匹配失败时 `log::warn` 输出具体数值（当前静默返回原曲线）

**文件**: `curve_eval.rs`

---

## 阶段 3：Newton-Raphson 投影（核心算法对齐）

这是最关键的阶段，将 OCC 的 Extrema 框架核心算法引入 rc3d。

### 3.1 曲线最近点 Newton-Raphson

**问题**: 当前暴力采样 + 步长减半，高曲率曲线可能漏掉全局最近点。
**OCC 对标**: `Extrema_ExtPC` 在 `(C(t)-P)·C'(t)=0` 上 Newton 迭代。

**方案**:
- 提取 `pub fn project_point_on_curve(curve, target, config) -> Vec<(f32, f32)>` 到新的 `crates/rc3d-io/src/step/brep/geom/project.rs`
- 多起始点采样（保留现有 24/64 采样）
- 每个候选点 Newton 迭代: `t_{n+1} = t_n - f(t_n)/f'(t_n)`，其中:
  - `f(t) = (C(t) - P) · C'(t)`（站点方程）
  - `f'(t) = C'(t)·C'(t) + (C(t)-P)·C''(t)`
- 收敛阈值: `|f(t)| < 1e-10` 或 `|Δt| < 1e-12`
- 最大迭代 20 次
- 返回所有收敛的候选 (t, distance²)，调用者选最小

**文件**: 新文件 `brep/geom/project.rs`，修改 `curve_eval.rs` 和 `surface_eval.rs` 的调用点
**测试**: 新增 `test_project_on_curve_high_curvature` — S 形 B-spline

### 3.2 曲面投影 Newton-Raphson

**问题**: 当前 16×16 网格 + 坐标下降，收敛慢且可能卡在局部最优。
**OCC 对标**: `Extrema_ExtPS` 在 2×2 系统上 Newton 迭代。

**方案**:
- 在 `project.rs` 中新增 `pub fn project_point_on_surface(surface, target, config) -> Vec<(f32, f32, f32)>`
- 多起始点: 保留阶段 2.3 的分层搜索作为种子
- 2×2 Newton 步:
  ```
  J = [ ∂S/∂u·∂S/∂u + (S-P)·∂²S/∂u²,   ∂S/∂u·∂S/∂v + (S-P)·∂²S/∂u∂v ]
      [ ∂S/∂v·∂S/∂u + (S-P)·∂²S/∂u∂v,   ∂S/∂v·∂S/∂v + (S-P)·∂²S/∂v² ]
  b = [ -(S-P)·∂S/∂u, -(S-P)·∂S/∂v ]
  Δ(u,v) = J⁻¹ · b
  ```
- 步长限制: `|Δu| < 0.1·u_span`, `|Δv| < 0.1·v_span`（防止跳出参数域）
- 收敛: `|Δ(u,v)| < 1e-10` 或 `|f| < 1e-10`
- 依赖: 阶段 2.7（解析 d2）以获得准确的 Jacobian

**文件**: `brep/geom/project.rs`，修改 `surface_eval.rs`
**测试**: 新增 `test_project_on_bspline_surface` — 对比网格搜索精度

### 3.3 统一投影接口

**方案**: 将所有 `project()` / `inverse_native_uv()` / `find_param_on_curve()` 统一路由到 `project.rs` 的新实现:
- `SurfaceGeom::project(point)` → `project_point_on_surface(self, point)`
- `SurfaceGeom::inverse_native_uv(point, tol)` → `project_point_on_surface` + 距离校验
- `find_param_on_curve(curve, target, n, iters)` → `project_point_on_curve(curve, target)`

**文件**: `surface_eval.rs`, `curve_eval.rs`, `brep/geom/project.rs`

---

## 验证策略

每个阶段完成后:
1. `cargo test -p rc3d-io` — 全部测试通过
2. `cargo test --test export_step_stl --release` — STL 导出，对比三角面数和 grid 回退数
3. `cargo test --test shape_corpus --release` — 质量基线指标
4. Shape-1.step 的 grid 回退数应减少（阶段 2.3 和阶段 3.2 的预期效果）

## 风险矩阵

| 项 | 风险 | 缓解措施 |
|----|------|---------|
| 2.2 build_ortho_axes 缓存 | 中 — 构造点更新遗漏 | 编译器会报错（缺少字段） |
| 2.6 椭圆 Newton | 低 — 1-2 轮迭代足够 | 保留 circle_angle_geom 作为 fallback |
| 2.7 解析 d2 | 低 — 闭式公式确定 | 与数值 d2 交叉验证 |
| 2.8 Weingarten | 中 — 退化点处理 | 保留数值回退 + warn |
| 3.1 曲线 Newton | 中 — 收敛不保证 | 保留网格采样作为 fallback |
| 3.2 曲面 Newton | 高 — 2×2 系统可能奇异 | det(J) 检查 + 坐标下降回退 |
