# OCC 对齐总体规划

> 日期: 2026-06-03
> 状态: Approved
> 决策: 统一 NurbsSurface 变体 · 完整 PCurve 重参数化 · 拉伸面网格提前至 Phase A
> 范围: 几何核心 + 网格核心 + 布尔运算 + 修复管线

## 1. 现状评估

### 1.1 对齐度总览

| 模块 | 对齐度 | 关键差距 |
|------|--------|----------|
| 曲面分类 | 95% | SurfaceGeom 全覆盖 |
| 曲线分类 | 80% | 缺 Hyperbola/Parabola/Offset |
| 求值接口 | 90% | 缺 DN(N>2) 高阶导数 |
| B-Spline/NURBS | 70% | 缺曲面节点插入、升阶 |
| 拓扑层级 | 80% | CompSolid/Compound 空壳 |
| 拓扑查询 | 40% | 无 TopExp 通用遍历器 |
| 增量网格 | 95% | 完整的阶段管线 |
| CDT | 95% | Bowyer-Watson + 约束边 |
| 布尔管线 | 70% | 5 阶段完整但 Phase 1 PCurve 为空 |
| 修复管线 | 80% | 18 个修复器，缺 SameParameter 重参数化 |
| SameParameter | 60% | 有公差传播+离散捕捉+偏移修复，缺核心重参数化 |

### 1.2 SameParameter 现状（修正后的评估）

**非 0% — 已有两处部分实现：**

1. **BRep 公差传播** (`rc3d-io/step/brep/same_parameter.rs`, 152 行)
   - STEP 导入后对每条边采样 32 点，计算 3D curve vs PCurve→surface 的偏差
   - 将偏差反映到 `edge.tolerance` — 不修正曲线，只放宽公差
   - 已集成到 STEP 加载流程

2. **网格离散化捕捉** (`rc3d-shape/mesh/same_param.rs`, 134 行)
   - 网格化阶段将边离散点 snap 到 PCurve 对应的曲面位置
   - 已集成到 shell_impl.rs 的阶段 1b

3. **周期偏移修复** (`rc3d-shape/heal/pcurve_fix.rs`, 577 行)
   - 检测/修复 Cylinder/Torus/Sphere 等周期面上 PCurve 偏移了一个周期的问题
   - 已集成到 heal 管线 Standard 级别

4. **边曲线端点修正** (pcurve_fix.rs 中的 `fix_edge_curves_wire`)
   - 调整 3D 曲线端点使其与顶点位置一致
   - 支持 Line/Circle/Ellipse/BSpline/Polyline

**缺失的核心算法：**
- PCurve 迭代重参数化 — 调整 PCurve 参数使 3D curve(t) 和 surface(pcurve(t)) 的偏差收敛到指定公差内
- 中间采样点的一致性检查和修正（非仅端点）
- 自适应采样（非均匀 32 点）

## 2. 项目清单与依赖关系

```
A0: NurbsSurface 统一
 ├── (独立，无前置依赖)
 │
A1: SameParameter 完整重参数化
 ├── (独立，无前置依赖)
 │
A2: 曲面节点插入
 ├── 依赖: A0 (统一后的 NurbsSurface)
 ├── 依赖: A1 (分割后需要 SameParameter 保证一致性)
 │
A3: 求交包围盒加速
 ├── (独立，无前置依赖)
 │
A4: TopExp 通用遍历器
 ├── (独立，无前置依赖)
 │
A5: 拉伸面专用网格路径
 ├── (独立，无前置依赖，最小改动)
 │
B1: 面-边/边-边求交
 ├── 依赖: A3 包围盒加速 (性能基础)
 ├── 依赖: A1 SameParameter (求交精度)
 │
B2: Hyperbola/Parabola 曲线
 ├── (独立，无前置依赖)
 │
C1: 升阶/节点细化
 ├── 依赖: A0 NurbsSurface 统一
 ├── 依赖: A2 曲面节点插入 (升阶需先有节点操作基础)
 │
C2: 面级自相交检测
 ├── 依赖: B1 面-边求交 (复用求交框架)
```

## 3. 分期计划

### Phase A：基础强化（P0 + P1 独立项 + 拉伸面）

**目标：** 补齐管线正确性基石、统一 NURBS 基础设施、加速布尔运算。

| 序号 | 项目 | 现有代码 | 新增预估 | 工作量 |
|------|------|---------|---------|--------|
| A0 | NurbsSurface 统一 | 两个变体共 1,279 行 | 重构 ~800 行 + 新增 ~200 行 | 高 |
| A1 | SameParameter 完整重参数化 | 863 行部分实现 | ~500 行 | 高 |
| A2 | 曲面节点插入 | 曲线 Boehm 已有 (50 行) | ~200 行 | 中 |
| A3 | 求交包围盒加速 | 零 (bool/ 路径) | ~400 行 | 中 |
| A4 | TopExp 通用遍历器 | 34+ 内联遍历 | ~300 行 | 中 |
| A5 | 拉伸面专用网格路径 | 直纹面网格 689 行已实现 | ~50 行 | 低 |

**A0: NurbsSurface 统一（前置依赖）**
- 问题: rc3d-nurbs 使用 `Vec<Vec<[f32; 4]>>` (齐次坐标)，rc3d-shape 使用 `Vec<Vec<Vec3>> + Vec<Vec<f32>>` (分离控制点+权重)
- 方案: 统一为 rc3d-nurbs 的齐次坐标格式 `[f32; 4]`（即 `(x*w, y*w, z*w, w)`），rc3d-shape 的 NurbsSurface 改为包装 rc3d-nurbs::NurbsSurface
- 涉及文件:
  - `rc3d-shape/src/nurbs.rs` (696 行) — 重构为包装器或直接替换
  - `rc3d-nurbs/src/surface.rs` (583 行) — 扩展为唯一的 NurbsSurface
  - `rc3d-io/src/step/brep/geom/nurbs_build.rs` (231 行) — 构建时转换为齐次格式
  - 所有使用 `rc3d_shape::nurbs::NurbsSurface` 的调用点
- 验收: 统一后所有现有测试通过；STEP 模型导入结果不变；evaluate_with_derivative/hessian 行为一致
- 注意: 这是最高的架构风险项 — 建议先添加集成测试确保迁移不引入回归

**A1: SameParameter 完整重参数化**
- 位置: `rc3d-shape/src/heal/same_param_fix.rs` (新文件)
- 算法:
  1. 自适应采样 3D curve(t) vs surface(pcurve(t)) 的偏差
  2. 对超过公差的 PCurve 段，构造新的 PCurve 使 3D/PCurve 偏差收敛
  3. 对于简单 PCurve (Line)：端点修正 + 中间采样点 Newton 精化
  4. 对于复杂 PCurve (BSpline)：通过节点插入增加控制点自由度，然后最小二乘拟合
  5. 迭代直至偏差 < tolerance 或达到最大迭代次数
- 集成: heal 管线 Basic 级别（在 fix_shifted 和 fix_edge_curves 之前），STEP 导入后，布尔运算后
- 测试: 至少 5 个测试 (对齐/不对齐-Line/不对齐-BSpline/周期面/自适应采样验证)
- 替代现有: `rc3d-io/step/brep/same_parameter.rs` 中的均匀采样公差传播被此替代（保留为 fallback）

**A3: 求交包围盒加速**
- 位置: `rc3d-shape/src/bool/bvh.rs` (新文件) + 修改 `intersect.rs`
- 算法: 构建面的 AABB 列表，Sweep-and-Prune 或排序后重叠检测
- 接口: `fn aabb_overlap_filter(faces_a, faces_b, reg) -> Vec<(FaceKey, FaceKey)>`
- 集成: 替换 `compute_intersections_brep` 中的暴力双重循环
- 可复用: mesh/report.rs 中已有的 `shell_vertex_bbox` 计算

**A4: TopExp 通用遍历器**
- 位置: `rc3d-shape/src/topo_iter.rs` (新文件)
- 接口: 
  ```rust
  fn iter_edges_of_face(face: FaceKey, reg: &BRepStore) -> impl Iterator<Item = EdgeKey>
  fn iter_faces_of_shell(shell: ShellKey, reg: &BRepStore) -> impl Iterator<Item = (FaceKey, Orientation)>
  fn iter_edges_of_shell(shell: ShellKey, reg: &BRepStore) -> impl Iterator<Item = EdgeKey>  // 去重
  fn iter_vertices_of_edge(edge: EdgeKey, reg: &BRepStore) -> (VertexKey, VertexKey)
  fn iter_wires_of_face(face: FaceKey, reg: &BRepStore) -> impl Iterator<Item = WireKey>
  ```
- 集成: 逐步替换 34+ 处内联 wire 遍历和 14+ 处内联 shell 遍历
- 注意: 不需要一次性替换所有内联遍历 — 新代码使用 TopExp，旧代码保持不变

### Phase B：能力扩展（P2 项目）

**目标：** 扩展几何覆盖面和布尔运算能力。

| 序号 | 项目 | 现有代码 | 新增预估 | 工作量 |
|------|------|---------|---------|--------|
| B1 | 面-边/边-边求交 | SSI 8 对已实现 | ~500 行 | 高 |
| B2 | Hyperbola/Parabola 曲线 | STEP 采样已有，降级为 Polyline | ~400 行 | 中 |

**B1: 面-边/边-边求交**
- 位置: `rc3d-shape/src/bool/intersect_edge.rs` (新文件)
- 面-边: 曲线参数 t 扫描 + surface.project(curve.d0(t)) 最近点检测 + Newton 精化
- 边-边: 两条曲线的最近点对 (复用 project_point_on_curve 的双曲线版)
- 集成: 新的求交入口 `compute_edge_intersections_brep()`
- 依赖: A3 包围盒加速、A1 SameParameter

**B2: Hyperbola/Parabola 曲线**
- 位置:
  - `curve_eval.rs` — 新增 CurveGeom::Hyperbola/Parabola 变体 + d0/d1/d2 求值
  - `brep/build/curve.rs` — 将降级逻辑替换为解析构建
- Hyperbola 参数化: P(t) = center + cosh(t)·major·major_axis + sinh(t)·minor·minor_axis
- Parabola 参数化: P(t) = center + t·major_axis + t²/(4f)·minor_axis
- 依赖: 无，独立实现
- 注意: 新增枚举变体需要更新所有 match arm (约 7+ crate)

### Phase C：高级功能（P3 项目）

**目标：** 补齐 NURBS 操作完整性。

| 序号 | 项目 | 现有代码 | 新增预估 | 工作量 |
|------|------|---------|---------|--------|
| C1 | 升阶/节点细化 | 零 | ~500 行 | 高 |
| C2 | 面级自相交检测 | UV 线框级已有 346 行 | ~300 行 | 中 |

**C1: 升阶/节点细化**
- 位置: `rc3d-nurbs/src/degree.rs` (新文件) + 扩展 curve.rs/surface.rs
- 算法: 
  - 节点细化: 重复 Boehm 插入至目标多重性
  - 升阶: Oktober 算法 — 保留形状的同时提高 B-spline 阶数
- 接口: `NurbsCurve::elevate_degree(target)` / `NurbsSurface::elevate_degree_u/v(target)`
- 依赖: A2 曲面节点插入

**C2: 面级自相交检测**
- 位置: `rc3d-shape/src/heal/face_self_intersect.rs` (新文件)
- 算法: 面上参数网格采样 + 法向量一致性检测 + 区域分割
- 可复用: intersect.rs 的 SSI 框架 (face_a == face_b)
- 依赖: B1 面-边求交

## 4. 建议执行顺序

```
Week 1:    A5 (拉伸面网格) + A4 (TopExp)         ← 两个独立的小/中项目，并行
Week 1-2:  A0 (NurbsSurface 统一)                 ← 架构重构，前置依赖
Week 2-4:  A1 (SameParameter 重参数化) + A3 (包围盒加速)  ← 并行
Week 4-5:  A2 (曲面节点插入)                       ← 依赖 A0 + A1
Week 5-7:  B1 (面-边/边-边求交) + B2 (Hyperbola/Parabola) ← 并行
Week 7-9:  C1 (升阶) + C2 (面级自相交)             ← 收尾
```

**关键路径:** A0 → A2 → C1 (NurbsSurface 统一 → 曲面节点插入 → 升阶)
**最长路径:** A0 → A1 → B1 → C2 (NurbsSurface → SameParameter → 面-边求交 → 面级自相交)

## 5. 决策记录

### 5.1 NurbsSurface 统一 ✅ 决定：统一为齐次坐标格式

统一为 rc3d-nurbs 的 `Vec<Vec<[f32; 4]>>` 齐次坐标格式。rc3d-shape 的 `NurbsSurface` 重构为包装器或直接替换。

理由：齐次格式天然适合节点插入/升阶等操作（控制点变换无需分别处理坐标和权重），避免在每次节点操作时做分离/合并转换。

风险：涉及 696 行的 rc3d-shape NurbsSurface 重构 + 所有调用点迁移。缓解措施：先添加集成测试确保迁移前后行为一致。

### 5.2 SameParameter 策略 ✅ 决定：完整 PCurve 重参数化

直接实现完整的 PCurve 迭代重参数化算法，不自适应公差传播作为中间方案。

理由：公差传播只是"掩盖"不一致而非修正它；完整重参数化从根本上保证 3D curve 和 PCurve 的一致性，是 OCC 的标准做法，后续布尔运算的精度依赖于此。

### 5.3 拉伸面网格 ✅ 决定：提前至 Phase A

在 `prefers_native_uv_ruled()` 中加入 Extrusion（~2 行改动），与 TopExp 同期完成。

理由：工作量极低（<1 小时），但能立即提升拉伸体网格质量，且独立于其他项目。

### 5.4 TopExp 替换策略

新增 TopExp 模块，新代码使用，旧代码保持不变。逐步在修改旧代码时替换。不做一次性全局替换。

## 6. 验收标准

每个项目的验收标准：

| 项目 | 验收标准 |
|------|---------|
| A0 NurbsSurface 统一 | 齐次坐标格式统一；所有现有测试通过；STEP 模型导入结果不变 |
| A1 SameParameter | 完整 PCurve 重参数化；自适应采样 + Newton 迭代修正；至少 5 个测试；集成到 heal 管线 Basic 级别 |
| A2 曲面节点插入 | NurbsSurface::insert_knot_u/v；形状不变性测试；STEP 模型验证 |
| A3 包围盒加速 | O(F²) → O(F·logF) 面对筛选；性能测试 100+ 面模型 |
| A4 TopExp | 5+ 遍历函数；替代至少 10 处内联遍历；零功能回归 |
| A5 拉伸面网格 | 拉伸体模型走直纹条带路径；弦偏差不退化 |
| B1 面-边求交 | 至少 Plane-Edge 和 Cylinder-Edge；集成到布尔管线 |
| B2 Hyperbola/Parabola | CurveGeom 新变体 + d0/d1/d2；STEP 导入不再降级为 Polyline |
| C1 升阶 | 曲线+曲面升阶；不变性测试 |
| C2 面级自相交 | 检测复杂面自交；至少 2 个测试用例 |
