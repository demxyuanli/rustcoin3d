# BREP 导出与 OCC 对齐验证 — 设计文档

## 目标

将 STEP 导入管线拆分为两个独立验证阶段：
1. **STEP → B-Rep**：导入 STEP 并生成 OCC 兼容的 `.brep` 文件，与 OCC 产出做逐量对比
2. **B-Rep → Mesh**：从已验证的 B-Rep 离散为可视化网格/STL

## 架构

```
STEP → [现有管线] → BRepStore ─┬─→ .brep 文件（新增）→ 与 OCC .brep diff
                               └─→ 现有 Mesh 管线（不变）
```

中间以 `.brep` 文件为契约边界。`.brep` 格式选用 OCC BREP ASCII，可用 OCC 工具直接打开验证。

## 决策汇总

### 第一阶段：补齐 BRepStore 缺口

| # | 决策 | 位置 |
|---|------|------|
| D1 | `BRepEdge` 加 `t_min: f32`, `t_max: f32`，始终 `t_min < t_max`，反向由 Orientation 表达 | `topo.rs` |
| D2 | `pcurves` 类型改为 `HashMap<FaceKey, (Curve2d, bool)>`，`bool` = `same_sense`，新插入默认 `true` | `topo.rs` + 所有读写点 |
| D3 | `CurveGeom` 加 `BezierCurve { degree, control_points, weights }` 变体，De Casteljau 求值 | `curve_eval.rs` + 全量 match 点 |

### 第二阶段：.brep 写入器

| # | 决策 |
|---|------|
| D4 | 位置：`rc3d-shape/src/brep/write.rs`（单文件，~600 行） |
| D5 | 格式：OCC BREP ASCII，所有 Location 写 identity |
| D6 | `Trimmed` 展开为 inner basis curve + Edge 的 `t_min/t_max` 做裁剪 |
| D7 | `Composite`/`Polyline` 拆为独立 `BRepEdge`（如存在） |
| D8 | Wire 导出加首尾相连一致性检查，断链写 `-- WARNING` 注释，不重排 |
| D9 | Polygon3D、Triangulations 节不写入（B-Rep 阶段不包含离散数据） |

### 第三阶段：验证工具链

| # | 决策 |
|---|------|
| D10 | 对比方案：结构计数（顶点/边/面/shell 数）+ 几何属性 diff（面积 ε=0.1%、弧长 ε=0.1%、坐标距离 ε=1e-4） |
| D11 | 测试语料：`export_step_stl` 的 8 个文件起步（Shape.step, Shape-1.step, Shape-2.step, Cube.step, cs.step, OffsetPlaneHoleEdge.step, asse.step, HoledPlate.step），按需从 OCCT 测试仓库扩展 |
| D12 | Loop 模式：每轮全跑拓扑计数 diff → 挑最差文件 → 全属性 diff → 修一个问题 → 重新全跑 |

## 数据流

```
STEP 文件
  │
  ├─→ OCC STEPControl_Reader → .brep (ground truth)
  │
  └─→ rustcoin3d parse_step → CafTransfer → BRepStore
                                                  │
                                                  ├─→ write_brep() → candidate.brep
                                                  │
                                                  └─→ build_emit_plan() → Mesh → STL/可视化
                                                         （现有管线，不变）

candidate.brep vs ground_truth.brep
  │
  ├─ 结构计数 diff：顶点数、边数、面数、shell 数
  └─ 几何属性 diff：逐面面积、逐边弧长、逐顶点坐标
```

## 关键映射

### CurveGeom → OCC BREP 类型号

| 编号 | OCC 类型 | CurveGeom 变体 |
|------|----------|----------------|
| 1 | Line | Line |
| 2 | Circle | Circle |
| 3 | Ellipse | Ellipse |
| 4 | Hyperbola | Hyperbola |
| 5 | Parabola | Parabola |
| 6 | BezierCurve | BezierCurve (新增 D3) |
| 7 | BSplineCurve | BSpline |
| 8 | OffsetCurve | Offset |
| — | Trimmed/Composite/Polyline | 展开为基础曲线 + Edge 裁剪 (D6/D7) |

### SurfaceGeom → OCC BREP 类型号

| 编号 | OCC 类型 | SurfaceGeom 变体 |
|------|----------|-------------------|
| 1 | Plane | Plane |
| 2 | Cylinder | Cylinder |
| 3 | Cone | Cone |
| 4 | Sphere | Sphere |
| 5 | Torus | Torus |
| 6 | LinearExtrusion | Extrusion |
| 7 | Revolution | Revolution |
| 8 | BSplineSurface | BSpline(NurbsSurface) |
| 9 | OffsetSurface | Offset |

### BRepStore → OCC BREP 拓扑映射

| OCC 节 | BRepStore 数据 | 状态 |
|--------|---------------|------|
| TVertex | BRepVertex { position, tolerance } | ✅ 带公差 |
| TEdge | BRepEdge { curve, t_min, t_max, tolerance } | D1 补 t_min/t_max |
| PCurve | BRepEdge.pcurves → (Curve2d, same_sense) | D2 补方向 |
| TFace | BRepFace { surface, outer_wire, inner_wires } | ✅ |
| TFace.location | identity | D5 全写 identity |
| TShell | BRepShell { faces, closed } | ✅ |
| TSolid | BRepSolid { outer_shell, void_shells } | ✅ |
| TCompound | BRepCompound { solids } | ✅ |

## 不在范围内的内容

- BRepStore 序列化/反序列化（读写分离）— 当前只需要写
- 几何实例化共享（Location != identity）— D5
- Polygon3D/Triangulations 节 — D9
- 图同构语义对齐 — 验证层后续扩展
- Mesh 管线改动 — 现有管线不变

## 验证指标

成功标准（Loop 每轮检查）：
1. 拓扑计数与 OCC ground truth 匹配（误差 < 5%）
2. 逐面面积误差 < 0.1%
3. 逐边弧长误差 < 0.1%
4. 逐顶点坐标距离 < 1e-4
5. 无断链 Warning
6. `rtk cargo test` 全绿，用例数不降
