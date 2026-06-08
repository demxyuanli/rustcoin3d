# rustcoin3d B-Rep 几何核心 vs OpenCASCADE 对齐度说明书

> 版本: 1.0 | 日期: 2026-06-08 | 审查范围: `rc3d-shape` + `rc3d-io` + `rc3d-nurbs`

---

## 目录

1. [总体评估](#1-总体评估)
2. [拓扑层 (TopoDS)](#2-拓扑层-topods)
3. [几何层 (Geom/Geom2d)](#3-几何层-geomgeom2d)
4. [算法层 (BOPAlgo/BRepMesh)](#4-算法层-bopalgobrepmesh)
5. [修复与导入层 (ShapeHealing/STEP)](#5-修复与导入层-shapehealingstep)
6. [缺口汇总与修复路线图](#6-缺口汇总与修复路线图)
7. [rustcoin3d 架构优势](#7-rustcoin3d-架构优势)
8. [术语对照表](#8-术语对照表)

---

## 1. 总体评估

```
层级         ████████████████████████████████░ 95%  拓扑层 (TopoDS)
层级         █████████████████████████████░░░░ 88%  几何层 (Geom/Geom2d)
层级         ████████████████████████░░░░░░░░░ 70%  算法层 (BOPAlgo/BRepMesh)
层级         ████████████████████████████░░░░░ 85%  修复与导入 (ShapeHealing/STEP)
─────────────────────────────────────────────────────────
综合         ████████████████████████████░░░░░ ~85%  全局对齐度
```

| 层级 | OCC 对应模块 | 对齐度 | 关键特征 |
|------|-------------|--------|----------|
| **拓扑层** | TopoDS, BRep_Builder, TopTools | **95%** (A) | SlotMap 存储 + 自动倒排索引 + 位置分离 超越 OCC |
| **几何层** | Geom_Curve, Geom_Surface, Geom2d_Curve | **88%** (A-) | 双参数化系统优于 OCC，曲线 8/9 曲面 10/10 覆盖 |
| **算法层** | BOPAlgo (Boolean), BRepMesh (Meshing) | **70%** (B) | 组件扎实，但 PaveBlock 数据在阶段间被丢弃 |
| **修复导入** | ShapeHealing, STEPControl | **85%** (B+) | 迭代收敛优于 OCC，部分模块未接入管道 |

---

## 2. 拓扑层 (TopoDS)

### 2.1 架构设计：两层分离

rustcoin3d 将 OCC 中耦合在一起的元素进行了清晰的分离：

| 层 | OCC 等价物 | rustcoin3d |
|----|-----------|------------|
| **原始 TShape** (纯几何) | `Handle<TopoDS_TShape>` | `BRepVertex` / `BRepEdge` 等，存储在 `BRepStore` SlotMap 中 |
| **定位实例** | `TopoDS_Shape { TShape*, Location, Orientation }` | `Shape { id: ShapeId }` + `ShapeNode { kind: ShapeKind, location: Mat4, orientation, parent, children }` |

**优势**: BRepStore 中的所有几何体均在局部空间中。`ShapeNode` 层处理空间定位（世界变换矩阵），保持几何运算无状态。避免 OCC 中 `TopLoc_Location` 带来的指针追踪和引用计数开销。

### 2.2 逐项对比

#### 基本形状 (TShape)

| OCC | rustcoin3d | 状态 |
|-----|-----------|------|
| `TopoDS_TVertex` | `BRepVertex { position: Vec3, tolerance: f32 }` | **1:1** |
| `TopoDS_TEdge` | `BRepEdge { curve: CurveGeom, v_low, v_high, pcurves: HashMap<FaceKey, Curve2d>, tolerance }` | **1:1** — PCurve 为 O(1) HashMap 查找，优于 OCC 的线性列表 |
| `TopoDS_TWire` | `BRepWire { edges: Vec<(EdgeKey, Orientation)> }` | **1:1** |
| `TopoDS_TFace` | `BRepFace { surface, outer_wire, inner_wires, same_sense, seam_edges, degenerated_edges, color, tolerance }` | **更优** — 显式 seam/degen 边 + STEP 颜色 |
| `TopoDS_TShell` | `BRepShell { faces: Vec<(FaceKey, Orientation)>, closed: bool, step_id }` | **1:1** + STEP 溯源 |
| `TopoDS_TSolid` | `BRepSolid { outer_shell, void_shells }` | **1:1** — 显式 outer + voids 更清晰 |
| `TopAbs_Orientation` | `Orientation { Forward, Reversed, Internal, External }` | **1:1** |

#### 存储与索引

| OCC 机制 | rustcoin3d 等价物 | 评估 |
|----------|------------------|------|
| `TopTools_IndexedMapOfShape` | `SlotMap<VertexKey, BRepVertex>` 等 6 个独立映射 | **更优** — 代际 arena，缓存友好，类型安全键 |
| 手动 `TopExp::MapShapes` 顶点去重 | `vertex_hash_index: HashMap<[u32;3], VertexKey>` O(1) 空间哈希 | **更优** |
| 手动 `TopExp::MapShapesAndAncestors` | `edge_to_faces: HashMap<EdgeKey, Vec<FaceKey>>` 构建时自动维护 | **更优** |
| `TopExp::MapShapesAndAncestors` (vertex→edge) | `vertex_to_edges: HashMap<VertexKey, Vec<EdgeKey>>` 构建时自动维护 | **更优** |

#### 遍历与探索

| OCC | rustcoin3d | 状态 |
|-----|-----------|------|
| `TopoDS_Iterator` (多态，单层) | `topo_iter.rs` 类型特化函数 (`iter_wires_of_face`, `iter_edges_of_face`, 等) | **部分** — 无单个多态迭代器 |
| `TopExp_Explorer` (多态，递归) | **无** — 需手动组合浅层遍历器 | **缺失** |
| `BRep_Builder` (构建器) | `BRepStore` 方法 (`find_or_add_vertex`, `add_edge_with_pcurve`, `add_face`, `set_pcurve`, `apply_pcurve_edit`) | **1:1** |

### 2.3 拓扑层缺口

| 缺口 | 严重程度 | 说明 |
|------|----------|------|
| `TopExp_Explorer` — 深度递归遍历 | 中 | 无"查找 solid 中所有 edge"的通用函数。需手动组合现有函数 |
| PCurve 级 `TopLoc_Location` | 低 | `BRepEdge.pcurves` 不存储每个 PCurve 的位置；在单一体局部坐标系内可接受 |

---

## 3. 几何层 (Geom/Geom2d)

### 3.1 设计决策：枚举 vs 继承

| 方面 | OCC (虚继承) | rustcoin3d (Rust 枚举) |
|------|-------------|----------------------|
| 分发方式 | 虚表（运行时） | `match`（编译期） |
| 添加新曲线类型 | 不改动基类 | 需修改每个 match 臂 |
| 添加新运算 | 所有子类均需实现 | 添加一个 match 臂即可 |
| 内存布局 | 每对象堆分配 | 小变体内联，`Trimmed`/`Offset` 需 `Box` |
| 序列化 | 不可直接序列化 | `#[derive(Clone, Debug)]` 零成本 |

**结论**: 枚举方式是本领域的正确选择。STEP 曲线类型集合是固定的，添加新运算的频率远高于添加新曲线类型。

### 3.2 3D 曲线覆盖 (CurveGeom)

```
Geom_Line              → CurveGeom::Line            { origin, direction }                    ✓ 1:1
Geom_Circle            → CurveGeom::Circle          { center, axis, radius, x_dir, y_dir }   ✓ 1:1
Geom_Ellipse           → CurveGeom::Ellipse         { center, axis, semi_major, semi_minor } ✓ 1:1
Geom_Hyperbola         → CurveGeom::Hyperbola       { center, axis, semi_major, semi_minor } ✓ 1:1
Geom_Parabola          → CurveGeom::Parabola        { center, axis, focal_dist }             ✓ 1:1
Geom_BSplineCurve      → CurveGeom::BSpline         { degree, control_points, knots, weights } ✓ 1:1
Geom_TrimmedCurve      → CurveGeom::Trimmed         { basis: Box<CurveGeom>, t_min, t_max }  ✓ 1:1
Geom_OffsetCurve       → CurveGeom::Offset          { basis: Box<CurveGeom>, offset_dir, distance } ✓ 1:1
Geom_BezierCurve       → (无)                       由 BSpline 通用覆盖                       ✗
```

**8/9 覆盖。** Bezier 可通过 `BSpline { degree=N, CPs=N+1, knots: clamped }` 精确表示，正确但无 de Casteljau 专用加速。

附加类型（无直接 OCC 对应）：

| rustcoin3d | 说明 |
|-----------|------|
| `CurveGeom::Polyline { points: Vec<Vec3> }` | 折线 — 用于交线近似 |
| `CurveGeom::Composite { segments: Vec<(CurveGeom, bool)>, cached_lengths }` | 复合曲线 — 含可选的弦长权重 |

### 3.3 曲面覆盖 (SurfaceGeom)

```
Geom_Plane                    → SurfaceGeom::Plane       { origin, normal, u_dir }            ✓
Geom_CylindricalSurface       → SurfaceGeom::Cylinder    { origin, axis, radius }             ✓
Geom_ConicalSurface           → SurfaceGeom::Cone        { apex, axis, semi_angle }           ✓
Geom_SphericalSurface         → SurfaceGeom::Sphere      { center, radius }                   ✓
Geom_ToroidalSurface          → SurfaceGeom::Torus       { center, axis, major_r, minor_r }   ✓
Geom_BSplineSurface           → SurfaceGeom::BSpline     (包装 NurbsSurface)                   ✓
Geom_SurfaceOfLinearExtrusion → SurfaceGeom::Extrusion   { generatrix, direction }            ✓
Geom_SurfaceOfRevolution      → SurfaceGeom::Revolution  { generatrix, axis_origin, axis_dir }✓
Geom_OffsetSurface            → SurfaceGeom::Offset      { basis: Box<SurfaceGeom>, distance }✓
```

**10/10 全部覆盖。** Offset 的 d1 使用 Weingarten 矩阵（形状算子）计算 — 解析方法，非有限差分。

### 3.4 2D 曲线覆盖 (Curve2d)

```
Geom2d_Line            → Curve2d::Line           { origin: (f32,f32), direction }            ✓
Geom2d_Circle          → Curve2d::Circle         { center, radius }                          ✓
Geom2d_Ellipse         → Curve2d::Ellipse        { center, semi_major, semi_minor }          ✓
Geom2d_BSplineCurve    → Curve2d::BSpline        { degree, CPs, knots, weights }             ✓
Geom2d_TrimmedCurve    → Curve2d::Trimmed        { basis: Box<Curve2d>, t_min, t_max }       ✓
Geom2d_Polyline        → Curve2d::Polyline       { points: Vec<(f32,f32)> }                  ✓
Geom2d_CompositeCurve  → Curve2d::Composite      { segments: Vec<(Curve2d, bool)> }          ✓
Geom2d_OffsetCurve     → (缺失)                                                                  ✗
Geom2d_Hyperbola       → (近似为 Ellipse)                                                       △
Geom2d_Parabola        → (近似为 Polyline 采样)                                                  △
```

**7/10 覆盖。** 缺失的 2D 类型不影响 STEP 互操作（STEP PCurve 极少使用这些类型）。

### 3.5 双参数化系统 — 超越 OCC 的设计

```
┌──────────────────────────────────────────────────┐
│  Layer 1: d0(t), d1(t), d2(t)  — t ∈ [0,1]      │  算法内部使用
│  Layer 2: d0_native(u,v)       — STEP 原始参数     │  PCurve 使用
├──────────────────────────────────────────────────┤
│  桥接函数:                                         │
│    native_uv_to_d0(u,v) → (t_u, t_v)              │
│    d0_uv_to_native(t_u, t_v) → (u,v)              │
│    param_range() → SurfaceParamRange { u_min,      │
│      u_max, v_min, v_max }                         │
├──────────────────────────────────────────────────┤
│  导数正确应用链式法则:                               │
│    Cylinder: d1_du_native = d1_du / TAU           │
│    BSpline:  d0_d1_d2_native 对导数进行             │
│              knot 域宽度缩放                        │
└──────────────────────────────────────────────────┘
```

| 曲面 | U 原始范围 | V 原始范围 | 归一化 |
|------|-----------|-----------|--------|
| Plane | [0, 1] | [0, 1] | 恒等 |
| Cylinder | [0, TAU] | 无界 | u/TAU |
| Cone | [0, TAU] | 无界 | u/TAU |
| Sphere | [0, TAU] | [0, PI] | u/TAU, v/PI |
| Torus | [0, TAU] | [0, TAU] | u/TAU, v/TAU |
| BSpline | [knots[p], knots[n]] | 同 | 线性映射 |
| Extrusion | [0, 1] | [0, 1] | 恒等 |
| Revolution | [0, 1] | [0, TAU] | v/TAU |
| Offset | 委托基础曲面 | 委托基础曲面 | 委托 |

### 3.6 NURBS 分离

```
┌─ rc3d-shape ─────────────────────┐  ┌─ rc3d-nurbs ────────────────────┐
│ NurbsSurface                       │  │ NurbsRenderSurface                 │
│ ├ 分离 CP + 权重                     │  │ ├ 齐次 [x,y,z,w] CP (GPU上传)         │
│ ├ 度提升 / knot 插入                  │  │ ├ 屏幕空间自适应曲面细分                  │
│ ├ Hessian (解析二阶导数)               │  │ ├ 轮廓/终止线检测                     │
│ └ From<NurbsRenderSurface>          │  │ └ From<NurbsSurface>              │
└────────────────────────────────────┘  └──────────────────────────────────┘
```

两个 crate 各自实现了 Cox-de Boor。分离的原因：不同的参数约定、CP 表示、使用场景和依赖配置。双向 `From` 实现提供了兼容性。

### 3.7 几何层缺口

| 缺口 | 严重程度 | 修复路径 |
|------|----------|----------|
| 无 `Geom_BezierCurve` (3D) | 低 | 为 B-spline deg=N, CPs=N+1 添加特化路径 |
| `CurveGeom` 无 `is_periodic()` | 低 | 为 Circle/Ellipse 添加方法，返回 TAU |
| `Curve2d::BSpline d1` 为数值导数 | 中 | 在 de Boor 循环中同时计算解析导数 |
| `Curve2d::Composite` 无弦长权重 | 中 | 从 3D `CurveGeom::Composite` 移植 |
| Sphere 使用全局轴 | 低 | 球面在 STEP 中轴向对齐，通过放置变换 |
| 无 `CurveEvaluator` trait | 低 | 在当前范围内可接受 |

---

## 4. 算法层 (BOPAlgo/BRepMesh)

### 4.1 布尔运算管道 — 5 个阶段

| 阶段 | OCC 对应 | rustcoin3d | 状态 |
|------|---------|-----------|------|
| 1. 求交 | `BOPAlgo_PaveFiller` | `pave_filler.rs` + `face_intersector.rs` | ✓ 实现 — AABB 扫掠裁剪 + marching+Newton |
| 2. 分割 | `BOPAlgo_BuilderFace` | `split.rs` + `builder_face.rs` | ⚠ 偏离 — UV 中点偏移几何分割，非拓扑分割 |
| 3. 分类 | `BRepClass3d_SolidClassifier` | `classify.rs` (6射线多数表决) | △ 简化 |
| 4. 选择 | `BOPAlgo_BOP` | `select.rs` | ✓ 基本 |
| 5. 缝合 | `BRepBuilderAPI_Sewing` | `stitch.rs` | △ 部分 |

**两步 SSI 架构** (与 OCC 一致):
- **解析路径** (`intersect.rs`): plane-plane, plane-cylinder, plane-sphere, sphere-sphere, cylinder-cylinder, cylinder-sphere, cylinder-cone, plane-cone, plane-torus。对支持的配对使用闭式解。
- **Marching 回退** (`marching.rs`): 对不支持的类型使用 `find_seeds()` + `trace_curve_bidirectional()`。等效于 OCC 的 `IntPatch_TheIWalking`。
- **Newton 细化** (`ssi_newton.rs`): 4参数 Gauss-Newton，带显式 3×3 逆矩阵和伪逆。与 OCC 的 `math_Gauss` 相比较简单，但容忍度检查 (`det.abs() < 1e-20`) 是唯一天然数值稳定性保护。

### 4.2 关键集成缺口

| # | 缺口 | 严重程度 | 影响 |
|---|------|----------|------|
| 1 | **PaveBlock 数据未被消费** | **高** | `pave_filler.rs` 构建了 PaveBlock（边分割点、参数范围），但 `split.rs` 和 `builder_face.rs` 完全忽略它们。数据在阶段 1→2 边界被丢弃 |
| 2 | **几何分割而非拓扑** | **高** | `split_face_along_curves` 计算 UV 子区域，但不构建新的 wire 环。OCC 通过 `BOPAlgo_BuilderFace` 从分割边构建拓扑 wires |
| 3 | **BOPDS 双重转换** | 中 | `FaceFaceInterf` → `FaceIntersectionResult` → `BRepIntersectionCurve`。中间转换丢失 PaveBlock 数据 |
| 4 | **分类仅针对第一个 shell** | **高** | `classify_brep_regions(&split_a_uv, shells_b[0], reg)` — 多 shell 输入失效 |
| 5 | **无历史/映射** | 中 | 无法将结果面映射回输入组件 |
| 6 | **无 sections/wire 提取** | 中 | 不支持 `BOPAlgo_Section` |
| 7 | **共面仅支持凸多边形** | 中 | Sutherland-Hodgman 对凹共面面可能不正确 |
| 8 | **无统一容忍度策略** | 低 | 各模块独立乘以 `* 10.0`, `* 100.0`。OCC 有 `BOPTools_Tools3D` |

### 4.3 网格管道

| OCC 组件 | rustcoin3d | 状态 |
|----------|-----------|------|
| `BRepMesh_IncrementalMesh` | `mesh_brep_shell_with_report_impl` | ✓ 对齐 |
| `BRepMesh_FastDiscret` | `edge_disc.rs` (PCurve-on-surface, 自适应分割) | ✓ 对齐 |
| `BRepMesh_Delaun` | `delaunay2d/` (BowyerWatson, 圆形索引, 翻转约束) | ✓ 对齐 |
| `BRepMesh_DeflectionControl` | 弦误差重试级联 | **更积极** — 多遍重试，逐次收紧 deflection |
| `BRepMesh_NodeInsertion` | `face_cdt.rs` Steiner 分割 (边中点) | △ 部分 |
| `BRepMesh_SameParameter` | `same_param.rs` | ✓ 对齐 |
| `BRepMesh_EdgeTessellation` | 边多边形构建 | ✓ 对齐 |

**网格优势**:
- 7 策略分派 (封闭参数化, 直纹, CDT, 曲面填充, UV 网格, 平面填充, 旋转填充)
- 双 CDT 后端 (`NativeCdt` 适配器封装 BowyerWatson + DelaBella)
- 多级弦误差重试级联 (限制 CDT → 收紧 deflection → UV 网格 → 曲面填充)
- 通过 `rayon::par_chunks` 并行面网格划分
- 受保护边界顶点焊接 (水密性)
- 快速导出模式

---

## 5. 修复与导入层 (ShapeHealing/STEP)

### 5.1 修复框架 — 22 个模块

```
heal/
├── mod.rs              — heal_shell() 编排器, HealConfig, HealReport
├── pipeline.rs          — auto_heal_shell(), HealLevel, select_fixes()
├── check.rs             — CheckReport, shell/wire/edge/face 验证
│
├── 线级别修复 (ShapeFix_Wire):
│   ├── wire_ops.rs      — FixReorder, FixSmall
│   ├── wire_join.rs     — FixConnected, FixGap3d, FixGap2d
│   ├── same_param_fix.rs — FixSameParameter
│   ├── pcurve_fix.rs    — FixShifted, FixEdgeCurves
│   ├── lacking.rs       — FixLacking
│   ├── self_intersect.rs — FixSelfIntersection
│   └── curve_trim.rs    — 边修剪/分割
│
├── 面级别修复 (ShapeFix_Face):
│   ├── face_fix.rs      — FixNaturalBound, FixReversed2d
│   ├── seam.rs          — FixMissingSeam
│   ├── degenerated.rs   — FixDegenerated, FixPeriodicDegenerated
│   ├── intersecting_wires.rs — FixIntersectingWires
│   └── face_self_intersect.rs — 曲面自交检测
│
├── 壳体级别修复 (ShapeFix_Shell):
│   ├── shell_fix.rs     — FixOrientation, FixVertexPosition, FixSplitFace
│   ├── free_bounds.rs   — FixFreeBounds (NEW)
│   └── compose_shell.rs — FixComposeShell (NEW)
│
├── 辅助模块:
│   ├── edge_tolerance.rs — FixVertexTolerance
│   ├── geom2d.rs        — 2D 工具 (点在多边形内, 线段求交)
│   ├── topo_diag.rs     — 拓扑诊断
│   └── continuity.rs    — G0/G1 连续性检查
```

### 5.2 超越 OCC 的修复特性

| 特性 | rustcoin3d | OCC |
|------|-----------|-----|
| **迭代收敛循环** | `auto_heal_shell()` — 运行 → 检查 → 选择修复 → 重复直到收敛 | `ShapeProcess` 仅应用固定序列一次 |
| **HealLevel 分层** | Basic → Standard → Advanced，基于层级的选择性修复 | 无；用户手动注册操作符 |
| **表驱动的 select_fixes** | 基于 `CheckReport` 标志的声明式修复选择 | 每应用程序硬编码 |
| **PCurve 解析回退链** | 4 层：STEP PCurve → 合成投影 → 参数回退 → 强制 STEP | OCC 仅有一种方法，回退灵活性更低 |
| **自适应迭代次数** | 预览 5 次，快速导出 2 次 | 无 |

### 5.3 HealLevel 层级定义

```
Basic:    FixConnected + FixSmall + FixReorder + FixGap3d + FixOrientation
Standard: Basic + FixSameParameter + FixGap2d + FixShifted + FixPeriodicDegen
          + FixEdgeCurves + FixLacking + FixMissingSeam + FixNaturalBound
          + FixReversed2d + FixVertexTolerance + FixSmallArea
          + FixVertexPosition + FixFreeBounds
Advanced: Standard + FixSelfIntersection + FixDegenerated + FixIntersectingWires
```

迭代 0 ("批量通道"): 一次性启用 Standard 级别的所有修复。
迭代 1+ ("精确通道"): 使用新的 `check_shell()` 结果驱动选择性重新修复。

### 5.4 STEP 导入管道

```
解析 (Part21/XML)
  ↓
EntityIndex → 验证
  ↓
CafTransfer → BRepStore + root_solids
  ↓
SameParameter 预通道 (无条件)
  ↓
Assembly 树构建
  ↓
run_heal_pipeline() → auto_heal_shell() 逐壳体
  ↓
连续性检查 (G0/G1)
  ↓
Mesh 发射计划
```

**关键特性**:
- PCurve 解析: 4 层回退 (STEP → 合成投影 → 参数化 → 强制)
- 表面构建: 覆盖 10 种 STEP 表面实体类型
- 曲面细分几何支持 (AP242 `TRIANGULATED_FACE`)
- 装配体爆炸 (heal 后)
- 面跳过传播: healed→mesh (防止无效面导致下游崩溃)
- 严格 void 模式: 投票判断哪个外表面与每个 void 面共享最多边

### 5.5 修复框架缺口

| 缺口 | 严重程度 | 说明 |
|------|----------|------|
| `close_free_bounds` 报告但未合并 | **高** | 检测到几何重合的开放边对但"计数为已修复"未实际共享边。注释: "Since we can't easily add cross-face pcurves without the face keys" |
| `fix_compose_shell` 未接入管道 | 中 | 模块存在但 `select_fixes()` 未在任何层级启用。需独立调用 |
| `fix_face_self_intersection` 仅检测 | 中 | `has_face_self_intersections` 标志已填充但无修复 pass |
| 无 `ShapeFix_Solid` 等价物 | 中 | 无统一的 `fix_solid()` pass |
| 无容忍度级联 | 低 | 面容忍度不传播到边；边容忍度不传播到顶点 |
| 重复的 SameParameter 工作 | 低 | 同时运行无条件预通道 + heal-pipeline 版本 |

---

## 6. 缺口汇总与修复路线图

### 6.1 优先级分类

| 优先级 | 缺口 | 工作量 | 影响域 |
|--------|------|--------|--------|
| **P0** | PaveBlock 数据在 stage 1→2 被丢弃 | 大 | 布尔正确性 |
| **P0** | 多 shell 布尔分类 | 中 | 多组件布尔正确性 |
| **P1** | `close_free_bounds` 实际合并边 | 中 | 修复正确性 |
| **P1** | `compose_shell` 接入 `select_fixes()` | 小 | 修复完整性 |
| **P1** | `fix_face_self_intersection` 修复 pass | 中 | 修复完整性 |
| **P2** | `TopExp_Explorer` 深度递归遍历 | 小 | 减少样板代码 |
| **P2** | 非凸共面布尔 (Weiler-Atherton) | 中 | 一般共面面 |
| **P2** | 布尔历史/映射 | 大 | 参数化/特征跟踪 |
| **P2** | `BOPAlgo_Section` (仅线框求交) | 中 | 线框提取 |
| **P3** | `Geom_BezierCurve` 作为一等类型 | 小 | 专用曲线优化 |
| **P3** | `Curve2d::BSpline d1` 解析导数 | 小 | 2D 求交精度 |
| **P3** | `Curve2d::Composite` 弦长权重 | 小 | 2D 参数准确性 |
| **P3** | 统一容忍度策略 (`BOPTools_Tools3D`) | 中 | 代码可维护性 |

### 6.2 已完成的修复 (本审查会话)

| # | 修复项 | 文件 |
|---|--------|------|
| 1 | PCurve 构建 bug 修复 (按曲线分区 UV 点) | `face_intersector.rs` |
| 2 | 共面面布尔 (Sutherland-Hodgman 裁剪) | `coplanar.rs` (新建) |
| 3 | PCurve 从 `CurveGeom/Vec3` 迁移到 `Curve2d` | 50+ 文件 |
| 4 | ShapeFix 补齐 (FreeBounds + ComposeShell) | `free_bounds.rs`, `compose_shell.rs` (新建) |

---

## 7. rustcoin3d 架构优势

以下方面 rustcoin3d 的架构**超越**了 OpenCASCADE：

| # | 优势 | 超越 OCC 的原因 |
|---|------|----------------|
| 1 | **SlotMap 存储** | 代际 arena 避免悬垂引用，缓存友好的密集存储，通过 `slotmap::new_key_type!` 实现类型安全键 |
| 2 | **自动倒排索引** | `edge_to_faces` 和 `vertex_to_edges` 在构建时填充，无需 OCC 的 `TopExp::MapShapesAndAncestors` 后处理遍历 |
| 3 | **位置分离** | BRepStore 为纯局部几何；`ShapeNode.layer: Mat4` 处理空间定位。消除所有几何操作中的位置处理 |
| 4 | **双参数化系统** | 显式的原生/归一化分离，链式法则正确的导数缩放。比 OCC 的隐式 `BRepAdaptor` 更清晰 |
| 5 | **迭代修复收敛** | 自适应重新检查 + 基于 `CheckReport` 的选择性重新修复。OCC 的 `ShapeProcess` 仅应用一个固定序列 |
| 6 | **表驱动的修复选择** | 声明式、可扩展、由诊断驱动。OCC 采用每应用程序硬编码 |
| 7 | **PCurve HashMap 访问** | O(1) 通过 `HashMap<FaceKey, Curve2d>` 查找，替代 OCC 的线性列表。面级别的局部坐标隐式处理 |
| 8 | **显式 BRepFace 字段** | `seam_edges`, `degenerated_edges`, `color` 都是显式的，而 OCC 隐式存储或根本不存在 |

---

## 8. 术语对照表

| OpenCASCADE | rustcoin3d |
|-------------|-----------|
| `TopoDS_TVertex` | `BRepVertex` |
| `TopoDS_TEdge` | `BRepEdge` |
| `TopoDS_TWire` | `BRepWire` |
| `TopoDS_TFace` | `BRepFace` |
| `TopoDS_TShell` | `BRepShell` |
| `TopoDS_TSolid` | `BRepSolid` |
| `TopAbs_Orientation` | `Orientation` |
| `TopAbs_ShapeEnum` | `ShapeKind` |
| `TopoDS_Iterator` | `topo_iter.rs` (类型特化函数) |
| `TopExp_Explorer` | (缺失) |
| `TopLoc_Location` | `ShapeNode.location: Mat4` |
| `TopTools_IndexedMapOfShape` | `SlotMap<Key, Value>` |
| `BRep_Builder` | `BRepStore` 方法 |
| `BRep_Tool` (PCurve 访问) | `edge.pcurves.get(&face_key)` |
| `Geom_Curve` | `CurveGeom` 枚举 |
| `Geom_Surface` | `SurfaceGeom` 枚举 |
| `Geom2d_Curve` | `Curve2d` 枚举 |
| `Geom_BSplineCurve` | `CurveGeom::BSpline` / `NurbsSurface` |
| `Geom_BSplineSurface` | `SurfaceGeom::BSpline(NurbsSurface)` |
| `Geom_BezierCurve` | (缺失 — 由 B-spline 覆盖) |
| `BRepAdaptor_Curve` | `eval_pcurve_on_surface()` (自由函数) |
| `BOPAlgo_PaveFiller` | `pave_filler.rs` |
| `BOPAlgo_BuilderFace` | `builder_face.rs` |
| `BOPAlgo_BOP` | `boolean_brep()` in `mod.rs` |
| `BRepClass3d_SolidClassifier` | `classify.rs` |
| `BRepMesh_IncrementalMesh` | `mesh_brep_shell_with_report_impl` |
| `BRepMesh_FastDiscret` | `edge_disc.rs` |
| `BRepMesh_Delaun` | `delaunay2d/` |
| `ShapeFix_Shape` | `heal_shell()` + `auto_heal_shell()` |
| `ShapeFix_Wire` | `heal_wire_passes()` |
| `ShapeFix_Face` | `heal_face_passes()` |
| `ShapeFix_Edge` | `same_param_fix.rs` + `edge_tolerance.rs` |
| `ShapeFix_Shell` | `shell_fix.rs` |
| `ShapeFix_FreeBounds` | `free_bounds.rs` |
| `ShapeFix_ComposeShell` | `compose_shell.rs` |
| `BRepCheck_Analyzer` | `check.rs` |
| `ShapeProcess_Operator` | `auto_heal_shell()` + `select_fixes()` |
| `STEPControl_Reader` | `step_reader.rs` |
| `STEPControl_ActorRead` | `entity_types.rs` + `build/` 模块 |

---

> **文档维护**: 随着 P0-P3 缺口的修复，更新本说明书。目标为每个缺口添加修复状态 (OPEN / IN PROGRESS / COMPLETED) 和修复提交。
