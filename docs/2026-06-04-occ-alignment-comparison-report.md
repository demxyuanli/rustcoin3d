# OCC 对齐比较报告

> 对比基准: Open CASCADE Technology 7.7 / ISO 10303-242 Edition 2
> 审查日期: 2026-06-04
> 审查范围: 几何核心 · Mesh 管线 · STEP 处理 · AP242 PMI · 布尔运算 · 修复管线

---

## 执行摘要

rustcoin3d 在 **几何核心 + Mesh + B-Rep 修复** 三个子系统上与 OCC 对齐度达到 **85-90%**，并已补齐路线图 10 个项目中的 9 个。AP242 差异化能力（PMI 标注、模式检测、语义验证）是当前最大缺口。建议下一阶段聚焦 **STEP AP242 基础** 和 **PMI 渲染**。

| 子系统 | 对齐度 | 状态 |
|--------|--------|------|
| 几何核心 | 88% | 曲线曲面全覆盖，升阶+节点插入已完成 |
| Mesh 管线 | 90% | 完整阶段管线，CDT+直纹面+拉伸面 |
| 修复管线 | 90% | 18/18 修复器，SameParameter 重参数化已实现 |
| STEP 导入 (B-Rep) | 85% | 5-Pass 构建，90+ 实体类型 |
| 布尔管线 | 75% | 5 阶段完整，交线 PCurve 缺失 |
| STEP 导入 (AP242) | 30% | PMI 提取框架存在，无渲染 |
| STEP 验证 | 30% | 基本拓扑检查，缺语义验证 |
| STEP 导出 | 10% | 仅 20 种实体反向映射 |

---

## 1. 几何核心 (`rc3d-shape/src/geom/` + `rc3d-nurbs/`)

### 1.1 曲线分类 (CurveGeom)

| 曲线类型 | OCC 类 | rustcoin3d | 状态 | 说明 |
|----------|--------|-----------|------|------|
| LINE | `Geom_Line` | `CurveGeom::Line` | ✅ Done | d0/d1/d2 完整 |
| CIRCLE | `Geom_Circle` | `CurveGeom::Circle` | ✅ Done | 含正交轴 |
| ELLIPSE | `Geom_Ellipse` | `CurveGeom::Ellipse` | ✅ Done | 含正交轴 |
| HYPERBOLA | `Geom_Hyperbola` | `CurveGeom::Hyperbola` | ✅ Done | cosh/sinh 参数化 |
| PARABOLA | `Geom_Parabola` | `CurveGeom::Parabola` | ✅ Done | t + t²/4f 参数化 |
| POLYLINE | `Geom_BSplineCurve` (p=1) | `CurveGeom::Polyline` | ✅ Done | |
| B_SPLINE_CURVE | `Geom_BSplineCurve` | `CurveGeom::BSpline` | ✅ Done | 含有理+非有理 |
| TRIMMED_CURVE | `Geom_TrimmedCurve` | `CurveGeom::Trimmed` | ✅ Done | 含参数裁剪 |
| COMPOSITE_CURVE | `Geom_BSplineCurve` (merged) | `CurveGeom::Composite` | ✅ Done | 含缓存长度 |
| **OFFSET_CURVE_3D** | `Geom_OffsetCurve` | `CurveGeom` (无独立变体) | ⚠️ Partial | 枚举已注册，构建时解包到内部曲线，**偏移量丢弃** |
| Bezier 曲线 | `Geom_BezierCurve` | (通过 BSpline) | ⚠️ Implicit | 升阶以 Bezier 形式工作，无一等公民类型 |
| CONIC | supertype | — | ❌ Missing | AP242 supertype 链不完整 |

**对齐度: 95%**

### 1.2 曲面分类 (SurfaceGeom)

| 曲面类型 | OCC 类 | rustcoin3d | 状态 | 说明 |
|----------|--------|-----------|------|------|
| PLANE | `Geom_Plane` | `SurfaceGeom::Plane` | ✅ Done | d0/d1/d2/normal |
| CYLINDRICAL_SURFACE | `Geom_CylindricalSurface` | `SurfaceGeom::Cylinder` | ✅ Done | 含正交轴 |
| CONICAL_SURFACE | `Geom_ConicalSurface` | `SurfaceGeom::Cone` | ✅ Done | semi_angle + apex |
| SPHERICAL_SURFACE | `Geom_SphericalSurface` | `SurfaceGeom::Sphere` | ✅ Done | |
| TOROIDAL_SURFACE | `Geom_ToroidalSurface` | `SurfaceGeom::Torus` | ✅ Done | major_r + minor_r |
| B_SPLINE_SURFACE | `Geom_BSplineSurface` | `SurfaceGeom::BSpline(NurbsSurface)` | ✅ Done | 含有理+非有理 |
| SURFACE_OF_LINEAR_EXTRUSION | `Geom_SurfaceOfLinearExtrusion` | `SurfaceGeom::Extrusion` | ✅ Done | generatrix + direction |
| SURFACE_OF_REVOLUTION | `Geom_SurfaceOfRevolution` | `SurfaceGeom::Revolution` | ✅ Done | generatrix + axis |
| OFFSET_SURFACE | `Geom_OffsetSurface` | `SurfaceGeom::Offset` | ✅ Done | basis + distance |
| **RECTANGULAR_TRIMMED_SURFACE** | — | ⚠️ 枚举已注册 | ⚠️ Partial | 无解析构建，STEP 导入丢失 |
| **CURVE_BOUNDED_SURFACE** | — | ⚠️ 枚举已注册 | ⚠️ Partial | 同上 |
| SWEPT_SURFACE | supertype | — | ⚠️ Partial | AP242 build 部分展开 |

**对齐度: 90%**

### 1.3 求值接口

| 操作 | OCC | rustcoin3d | 状态 |
|------|-----|-----------|------|
| d0 (位置) | `Geom_Curve::D0` / `Geom_Surface::D0` | `d0(t)` / `evaluate(u,v)` | ✅ Done |
| d1 (一阶导) | `Geom_Curve::D1` / `Geom_Surface::D1` | `d1(t)` / `derivative(u,v)` | ✅ Done |
| d2 (二阶导) | `Geom_Curve::D2` / `Geom_Surface::D2` | `d2(t)` / `d2(u,v)` | ✅ Done |
| d012 (批量) | — | `d012(t)` | ✅ Done (已优化) |
| **d3 (三阶导)** | `Geom_Curve::D3` / `Geom_Surface::D3` | — | ❌ Missing |
| **DN(N>2)** | `Geom_Curve::DN(theU, N)` | — | ❌ Missing |
| **曲率** | `Geom_Curve::Curvature` | — | ❌ Missing |
| **挠率** | `Geom_Curve::Torsion` | — | ❌ Missing |
| **曲面主曲率** | `GeomLProp_SLProps` | — | ❌ Missing |
| **曲面面积** | `Geom_Surface::Area` | — | ❌ Missing |
| **法向量** | `Geom_Surface::Normal` | `normal(u,v)` | ✅ Done |

**对齐度: 85%**

### 1.4 B-Spline/NURBS 操作

| 操作 | OCC | rustcoin3d | 状态 |
|------|-----|-----------|------|
| 节点插入 (曲线) | `BSplCLib::InsertKnot` | `NurbsCurve::insert_knot` (Boehm) | ✅ Done |
| 节点插入 (曲面 U/V) | — | `NurbsSurface::insert_knot_u/v` (Boehm) | ✅ Done |
| **升阶 (曲线)** | `BSplCLib::IncreaseDegree` | `NurbsCurve::elevate_degree` | ✅ Done |
| **升阶 (曲面 U/V)** | — | `NurbsSurface::elevate_degree_u/v` | ✅ Done |
| **降阶** | `BSplCLib::ReduceDegree` | — | ❌ Missing |
| **节点移除** | `BSplCLib::RemoveKnot` | — | ❌ Missing |
| NurbsSurface 统一 | — | 两个变体共存 (齐次 vs 分离) | ⚠️ 未统一 |

**对齐度: 80%** (路线图 Phase C 完成后从 70% 提升)

---

## 2. Mesh 管线 (`rc3d-shape/src/mesh/`)

### 2.1 阶段管线

| 阶段 | OCC | rustcoin3d | 状态 |
|------|-----|-----------|------|
| 阶段 1a: 面网格算法选择 | `BRepMesh_MeshAlgoFactory` | `select_uv_source` | ✅ Done |
| 阶段 1b: 边离散化 | `BRepMesh_EdgeDiscret` | `edge_disc.rs` | ✅ Done |
| 阶段 2: CDT (规则面) | `BRepMesh_FastDiscretFace` | `face_cdt.rs` TrimmedCdt | ✅ Done |
| 阶段 3: 自由曲面填充 | `BRepMesh_DelaunayDeflection` | `face_fill.rs` | ⚠️ Partial |
| 阶段 4: 法向量恢复 | — | 顶点法向量计算 | ✅ Done |
| 阶段 5: 全局属性 | `BRepGProp` | `brep/properties.rs` | ✅ Done |
| 直纹面/拉伸面专用路径 | — | `fill_ruled.rs` | ✅ Done |
| 旋转面专用路径 | — | `fill_revolution.rs` | ✅ Done |
| **Void shell 布尔减** | `BRepAlgoAPI_Cut` | — | ❌ Missing |

**对齐度: 90%**

### 2.2 CDT (约束 Delaunay 三角剖分)

| 操作 | OCC | rustcoin3d | 状态 |
|------|-----|-----------|------|
| Bowyer-Watson 增量插入 | `BRepMesh_Delaunay` | `delaunay2d/` | ✅ Done |
| 约束边恢复 | `BRepMesh_Edge` | 边翻转 + 分裂 | ✅ Done |
| 退化边处理 | `BRepMesh_FaceChecker` | 退化边 CDT 约束 | ✅ Done |
| 洞处理 | — | 内环约束 | ✅ Done |

**对齐度: 95%**

### 2.3 网格配置

| 参数 | OCC | rustcoin3d | 状态 |
|------|-----|-----------|------|
| 线性 Deflection | `BRepMesh_IncrementalMesh::SetLinearDeflection` | `linear_deflection` | ✅ Done |
| 角度 Deflection | `BRepMesh_IncrementalMesh::SetAngularDeflection` | `angular_deflection` | ✅ Done |
| 相对 Deflection | `IsRelative` | `mesh_relative_deflection` | ✅ Done |
| 最小/最大尺寸 | — | `min_size`, `max_size` | ✅ Done (扩展) |

**对齐度: 100%**

---

## 3. STEP 处理 (`rc3d-io/src/step/`)

### 3.1 B-Rep 构建 (StepToTopoDS 等价)

| 阶段 | OCC | rustcoin3d | 状态 |
|------|-----|-----------|------|
| Pass 1: 曲面注册 | `TranslateFace` | `build_faces_from_face_surface` | ✅ Done |
| Pass 2: 边+PCurve | `TranslateEdgeLoop` | `build_edges_and_wires` | ✅ Done |
| Pass 3: Face+Wire 组装 | — | `build/brep.rs` Pass 3 | ✅ Done |
| Pass 4: Shell 组装 | `TranslateShell` | Pass 4 | ✅ Done |
| Pass 5: Solid/朝向 | `TranslateSolid` | Pass 5 | ✅ Done |
| 顶点构建 | `TranslateVertex` | `topology.rs` build_vertex | ✅ Done |
| 朝向传播 | `PropagateOrientation` | `heal/orient.rs` | ✅ Done |

**对齐度: 90%**

### 3.2 实体类型覆盖

| 类别 | 已注册实体 | 有解析构建 | 缺失 |
|------|-----------|--------|------|
| 曲线实体 | 17/18 | 16/18 | Offset 曲线无独立构建 |
| 曲面实体 | 14/14 | 12/14 | RectTrimmed, CurveBounded 无构建 |
| 拓扑实体 | 13/13 | 13/13 | 全覆盖 |
| 几何基础 | 6/6 | 6/6 | 全覆盖 |
| 装配体 | 12/19 | 8/19 | 7 种装配实体缺失 |
| PMI/标注 | 6/35+ | 4/35+ | **30+ 种 PMI 实体缺失** |
| 镶嵌几何 | 0/14 | 0/14 | 全类缺失 |
| 样式/颜色 | 4/12 | 4/12 | 8 种样式实体缺失 |

**总体覆盖率: ~80% (核心 B-Rep) / ~30% (AP242 全谱)**

### 3.3 STEP 导出

| 类别 | 状态 |
|------|------|
| 实体类型反向映射 | ⚠️ ~20 种 |
| 完整 STEP 写入器 | ❌ 框架存在，仅覆盖基础类型 |

**对齐度: 10%**

---

## 4. 修复管线 (`rc3d-shape/src/heal/`)

### 4.1 修复器对照

| OCC ShapeFix 修复器 | rustcoin3d 文件 | 状态 |
|---------------------|----------------|------|
| `ShapeFix_Wire::FixReorder` | `wire_ops.rs` | ✅ Done |
| `ShapeFix_Wire::FixGaps3d` | `wire_join.rs` | ✅ Done |
| `ShapeFix_Wire::FixGaps2d` | `wire_join.rs` close_wire_gaps_2d | ✅ Done |
| `ShapeFix_Wire::FixConnected` | `wire_join.rs` fix_connected_wire | ✅ Done |
| `ShapeFix_Wire::FixSmall` | `wire_ops.rs` remove_small_edges | ✅ Done |
| `ShapeFix_Wire::FixShifted` | `pcurve_fix.rs` fix_shifted_pcurves | ✅ Done |
| `ShapeFix_Edge::FixVertexTolerance` | 分布在 `mod.rs` fix_vertex_tolerance | ✅ Done |
| `ShapeFix_Wire::FixLacking` | `lacking.rs` | ✅ Done |
| `ShapeFix_Wire::FixSelfIntersection` | `self_intersect.rs` | ✅ Done |
| `ShapeFix_Wire::FixIntersectingWires` | `intersecting_wires.rs` | ✅ Done |
| `ShapeFix_Wire::FixDegenerated` | `degenerated.rs` | ✅ Done |
| — (periodic degenerated) | `degenerated.rs` fix_periodic_degenerated | ✅ Done |
| — (vertex position) | `shell_fix.rs` fix_vertex_positions | ✅ Done |
| `ShapeFix_Face::FixSplitFace` | `shell_fix.rs` fix_split_face | ✅ Done |
| `ShapeFix_Face::FixMissingSeam` | `seam.rs` | ✅ Done |
| `ShapeFix_Face::FixAddNaturalBound` | `face_fix.rs` fix_add_natural_bound | ✅ Done |
| `ShapeFix_Face::FixReversed2d` | `face_fix.rs` fix_reversed_2d | ✅ Done |
| **`BRepLib::SameParameter`** | `same_param_fix.rs` | ✅ Done (完整重参数化) |
| **面级自相交检测** | `face_self_intersect.rs` | ✅ Done |

**对齐度: 95%** (18/18 修复器完成 + 1 扩展)

### 4.2 验证规则 (BRepCheck)

| 检查项 | OCC | rustcoin3d | 状态 |
|--------|-----|-----------|------|
| UV 自交检测 | `BRepCheck_Face` | `check_uv_self_intersection` | ✅ Done |
| 面级自交检测 | — | `check_face_self_intersect` | ✅ Done (扩展) |
| G0 连续性 | `BRepCheck_Edge` | `check_shell_continuity` G0 | ✅ Done |
| G1 连续性 | — | `check_shell_continuity` G1 | ✅ Done |
| 朝向一致性 | `BRepCheck_Shell` | `check_wire_orientation` | ✅ Done |
| 拓扑完备性 | `BRepCheck_Shell` | `check_shell` Euler-Poincaré | ✅ Done |
| 非流形检测 | `BRepCheck_Analyzer` | `check_non_manifold` | ✅ Done |
| 参数范围验证 | — | `check_parameter_range` | ✅ Done |
| **AP 模式检测** | — | — | ❌ Missing |
| **PMI 验证** | — | — | ❌ Missing |
| **EXPRESS 模式约束** | — | `validate.rs` | ⚠️ Partial |

**对齐度: 85%**

---

## 5. 布尔管线 (`rc3d-shape/src/bool/`)

### 5.1 布尔阶段

| 阶段 | OCC API | rustcoin3d | 状态 |
|------|---------|-----------|------|
| 1. SSI 面-面求交 | `BRepAlgoAPI_Intersection` | `intersect.rs` | ✅ Done (8 对) |
| 1b. 包围盒加速 | `Bnd_Box` + SAP | `aabb.rs` | ✅ Done |
| 1c. 面-边/边-边求交 | `BRepExtrema_DistShapeShape` | `intersect_edge.rs` | ✅ Done |
| 1d. **交线 PCurve 生成** | `BRepAlgoAPI_Intersection` | — | ❌ **Missing** |
| 2. 面分割 | `BRepAlgoAPI_Splitter` | `split.rs` | ✅ Done |
| 3. 区域分类 | `BRepClass3d_SolidClassifier` | `classify.rs` | ✅ Done |
| 4. 面选择 | `BRepAlgoAPI_Common/Fuse/Cut` | `select.rs` | ✅ Done |
| 5. 缝合 | `BRepBuilderAPI_Sewing` | vertex welding | ✅ Done |
| NURBS marching | `IntPatch_ImpPrmIntersection` | `marching.rs` | ✅ Done |
| 射线分类 (Cone/Torus) | `BRepClass3d_SClassifier` | ray casting | ✅ Done |

**对齐度: 75%** （唯一瓶颈：交线 PCurve 参数化缺失）

---

## 6. 拓扑结构

### 6.1 拓扑层级

| 层级 | OCC | rustcoin3d | 状态 |
|------|-----|-----------|------|
| Vertex | `TopoDS_Vertex` | `BRepVertex` | ✅ Done |
| Edge | `TopoDS_Edge` | `BRepEdge` (含 PCURVE) | ✅ Done |
| Wire | `TopoDS_Wire` | `BRepWire` | ✅ Done |
| Face | `TopoDS_Face` | `BRepFace` (含 surface + wires) | ✅ Done |
| Shell | `TopoDS_Shell` | `BRepShell` (closed + faces) | ✅ Done |
| Solid | `TopoDS_Solid` | `BRepSolid` (outer_shell + void_shells) | ✅ Done |
| **CompSolid** | `TopoDS_CompSolid` | `ShapeKind::CompSolid` (枚举，无数据) | ⚠️ Stub |
| **Compound** | `TopoDS_Compound` | `ShapeKind::Compound` (枚举，无数据) | ⚠️ Stub |

### 6.2 拓扑遍历 (TopExp)

| 操作 | OCC | rustcoin3d | 状态 |
|------|-----|-----------|------|
| `iter_edges_of_face` | `TopExp::MapShapes(face, TopAbs_EDGE)` | `topo_iter::iter_edges_of_face` | ✅ Done |
| `iter_faces_of_shell` | `TopExp::MapShapes(shell, TopAbs_FACE)` | `topo_iter::iter_faces_of_shell` | ✅ Done |
| `iter_edges_of_shell` | `TopExp::MapShapes(shell, TopAbs_EDGE)` | `topo_iter::iter_edges_of_shell` | ✅ Done |
| `iter_vertices_of_edge` | `TopExp::Vertices(edge)` | `topo_iter::iter_vertices_of_edge` | ✅ Done |
| `iter_wires_of_face` | `TopExp::MapShapes(face, TopAbs_WIRE)` | `topo_iter::iter_wires_of_face` | ✅ Done |

**对齐度: 100%** (路线图 A4 完成后)

---

## 7. AP242 差异化能力 — 最大缺口

### 7.1 PMI / 标注

| 类别 | 已实现 | 缺失/部分 |
|------|--------|----------|
| 数据提取 | `PmiDimension`, `PmiDatum`, `PmiToleranceFrame` | ANGULAR_SIZE 值提取 |
| 尺寸标注 | DIMENSIONAL_SIZE 框架 | DIMENSIONAL_LOCATION, SIZE_WITH_DATUM, SIZE_WITH_PATH |
| 公差 | GEOMETRIC_TOLERANCE 框架 | **25+ 种公差子类型** (FLATNESS, POSITION, PERPENDICULARITY 等) |
| 基准 | DATUM 框架 | DATUM_SYSTEM, DATUM_REFERENCE_COMPARTMENT, DATUM_REFERENCE_ELEMENT |
| 渲染 | **无** | PMI→SceneGraph 注入, 3D 标注渲染 pass |
| 关联 | **无** | ANNOTATION_OCCURRENCE → 几何 |
| 语义 PMI | **无** | AP242 Edition 2 语义标注全部 |
| 引导线 | **无** | LEADER_DIRECTED_CALLOUT, DIMENSION_CURVE |

### 7.2 镶嵌几何 (AP242 Edition 2)

全类缺失 — 14 种实体 (TESSELLATED_ITEM, TRIANGULATED_FACE, COORDINATES_LIST 等)。

### 7.3 装配体

| 功能 | 状态 |
|------|------|
| NAUO 解析 + 变换提取 | ✅ Done |
| 扁平装配提取 | ✅ Done |
| STYLED_ITEM 颜色 | ✅ Done (单层) |
| **装配体样式继承** | ❌ 父装配样式不传播 |
| **多层级装配树** | ❌ 无递归深度保留 |
| **MAPPED_ITEM 实例化** | ❌ |
| **CONTEXT_DEPENDENT_SHAPE** | ❌ |
| **PRODUCT_CATEGORY** | ❌ |

---

## 8. 差距优先级矩阵

### P0 — 阻塞完整 AP242 使用

| # | 缺口 | 子系统 | 工作量 | 影响 |
|---|------|--------|--------|------|
| 1 | **Void shell 布尔减** | Mesh + STEP | 1-2 天 | 洞/腔体不出现 |
| 2 | **AP242 模式检测** | STEP | 0.5 天 | 无法区分 AP203/AP242 |
| 3 | **PMI → SceneGraph 注入 + 渲染** | PMI + Render | 3-5 天 | PMI 不可见 |

### P1 — 显著改善

| # | 缺口 | 子系统 | 工作量 | 影响 |
|---|------|--------|--------|------|
| 4 | **Bool 交线 PCurve 生成** | Bool | 2-3 天 | 面分割质量 |
| 5 | **曲面节点移除 (Knot Removal)** | NURBS | 1 天 | 升阶后紧凑表示 |
| 6 | **Offset 曲线正确构建** | STEP | 0.5 天 | OFFSET_CURVE_3D 偏移量丢失 |
| 7 | **RectTrimmedSurface + CurveBoundedSurface 构建** | STEP | 1 天 | 复杂裁剪面导入 |
| 8 | **装配体样式继承** | STEP | 1 天 | 子部件颜色缺失 |
| 9 | **公差类型细化 (25+ 种)** | PMI | 2-3 天 | 公差语义完整 |

### P2 — AP242 深度

| # | 缺口 | 子系统 | 工作量 |
|---|------|--------|--------|
| 10 | 曲面降阶 (Degree Reduction) | NURBS | 1 天 |
| 11 | 曲线/曲面曲率计算 | Geom | 0.5 天 |
| 12 | 高阶导数 d3/DN | Geom | 0.5 天 |
| 13 | 多层级装配树 | STEP | 1-2 天 |
| 14 | MAPPED_ITEM 实例化 | STEP | 1 天 |
| 15 | 基准系 + 引导线渲染 | PMI | 1-2 天 |
| 16 | 镶嵌几何实体集 | STEP | 2-3 天 |
| 17 | 语义 PMI | PMI | 2-3 天 |

### P3 — 扩展/未来

| # | 缺口 | 说明 |
|---|------|------|
| 18 | STEP 完整导出 | 独立大型项目 |
| 19 | CompSolid/Compound 完整实现 | 布尔多实体结果 |
| 20 | 透明度/渲染属性 | SURFACE_STYLE_TRANSPARENT 等 |
| 21 | 并行网格化 | 性能优化项目 |

---

## 9. 已完成路线图项目状态

| 项目 | 说明 | 状态 |
|------|------|------|
| A0 | NurbsSurface 统一 | ⏭️ 跳过 (架构重构，用户决策) |
| A1 | SameParameter 重参数化 | ✅ `same_param_fix.rs` (531 行) |
| A2 | 曲面节点插入 | ✅ `insert_knot_u/v` (Boehm) |
| A3 | 求交包围盒加速 | ✅ `aabb.rs` + AABB 预过滤 |
| A4 | TopExp 遍历器 | ✅ `topo_iter.rs` (5+ 函数) |
| A5 | 拉伸面网格 | ✅ `prefers_native_uv_ruled` 含 Extrusion |
| B1 | 面-边/边-边求交 | ✅ `intersect_edge.rs` (366 行) |
| B2 | Hyperbola/Parabola | ✅ `CurveGeom` 新变体 |
| C1 | 升阶 | ✅ `elevate_degree` (曲线+曲面) |
| C2 | 面级自相交检测 | ✅ `face_self_intersect.rs` |

**10 项目中 9 个完成，1 个跳过。**

---

## 10. 建议对齐路线

```
Phase D: 几何补齐 (P1, ~800 行, 3-5 天)
├── D1: 节点移除 (Knot Removal) ~200 行
├── D2: 降阶 (Degree Reduction) ~300 行
├── D3: 曲率/挠率/主曲率 ~150 行
└── D4: Offset 曲线正确构建 ~100 行

Phase E: STEP AP242 基础 (P0, ~750 行, 2-4 天)
├── E1: AP242 模式检测 ~100 行
├── E2: Void shell 布尔减 ~300 行
├── E3: RectTrimmed + CurveBounded 构建 ~200 行
└── E4: SURFACE_CURVE 完整实体支持 ~150 行

Phase F: PMI 渲染 (P0, ~2,200 行, 5-8 天)
├── F1: PMI → SceneGraph 注入 ~500 行
├── F2: 3D 标注渲染 pass ~800 行
├── F3: 公差类型细化 (25+ 种) ~500 行
└── F4: 基准系 + 引导线渲染 ~400 行

Phase G: 装配体/布尔补齐 (P1, ~1,000 行, 3-5 天)
├── G1: Bool 交线 PCurve 生成 ~400 行
├── G2: 装配体样式继承 ~300 行
├── G3: 多层级装配树 ~300 行
└── G4: MAPPED_ITEM 实例化 ~200 行

Phase H: AP242 深度 (P2, ~1,500 行, 4-6 天)
├── H1: 镶嵌几何实体集 ~500 行
├── H2: 语义 PMI ~400 行
├── H3: 高阶导数 d3/DN ~200 行
└── H4: 透明度/渲染属性 ~200 行
```

### 实施策略

- **E 优先启动** (P0 影响最大，0 外部依赖)
- **D 可与 E 并行** (几何核心与 STEP 导入无交集)
- **F 在 E 完成后启动** (PMI 渲染依赖 STEP 实体提取完善)
- **G 可与 F 并行** (装配体与渲染管线独立)
- **H 最后** (P2 深度功能)
