# AP242 覆盖分析报告

**日期**: 2026-05-27
**对比基准**: Open CASCADE Technology (OCCT) StepToTopoDS + XCAF
**分析范围**: `crates/rc3d-io/src/step/` — 实体类型、B-Rep、装配体、PMI、验证

---

## 一、执行摘要

AP242 (ISO 10303-242) = AP203 (显式几何) + AP214 (汽车核心数据) + PMI (产品制造信息) + 镶嵌几何 (Edition 2+) + 3D 打印/复材/线束 (Edition 3).

当前 `rc3d-io/step/` 覆盖了 AP203 的核心 B-Rep 几何管线（~90 种实体类型 + 16 个 heal 修复器 + OCC 对齐的 mesh 工厂），但 AP242 最关键的差异化能力——**PMI 标注渲染**和**AP242 模式感知文件头**——仍处于框架/缺失状态。

**总体覆盖估计**: 几何核心 ~75%, 装配体 ~40%, PMI ~15%, 验证规则 ~25%.

**最大缺口 (按影响排序)**:
1. PMI 标注渲染管线未完成（标注已提取但无渲染路径）
2. AP242 文件头/模式检测缺失（无法区分 AP203 vs AP242 文件）
3. AP242 supertype 子类型链延伸不完整
4. 装配体样式继承未处理
5. 镶嵌几何 (Tessellated Geometry) 实体集完全缺失

---

## 二、分析方法

以 OCCT 源码为参照基准，对以下模块逐项对比：

| OCCT 模块 | 对应本仓库 | 分析维度 |
|-----------|-----------|---------|
| `StepToTopoDS` | `brep/build.rs` + `topology.rs` + `geom.rs` + `nurbs.rs` | 实体类型映射、几何/拓扑构建 |
| `StepToGeom` | `brep/geom.rs` + `nurbs.rs` + `curve.rs` | 曲线/曲面统一表示 |
| `ShapeFix` | `brep/heal/*` (16 模块) | 修复管线 |
| `XCAF` | `assembly.rs` + `pmi/` | 装配体 + PMI |
| `BRepCheck` | `heal/check.rs` | 验证规则 |
| `BRepMesh` | `brep/mesh/*` + `surface_tess.rs` | 网格化 + CDT |

覆盖等级:
- **Done** — 有完整实现，与 OCCT 对齐
- **Partial** — 有框架或部分实现，存在已知缺口
- **Missing** — 完全未实现
- **NA** — 不适用于本仓库目标（如 BIM 实体）

---

## 三、AP242 实体类型覆盖矩阵

### 3.1 曲线实体

| 实体 | OCCT 类 | 覆盖率 | 状态 | 备注 |
|------|---------|--------|------|------|
| LINE | `Geom_Line` → NURBS | Done | | |
| CIRCLE | `Geom_Circle` → NURBS | Done | | |
| ELLIPSE | `Geom_Ellipse` → NURBS | Done | | |
| HYPERBOLA | `Geom_Hyperbola` → NURBS | Done | | Phase 1 添加 |
| PARABOLA | `Geom_Parabola` → NURBS | Done | | Phase 1 添加 |
| POLYLINE | `Geom_BSplineCurve` (degree 1) | Done | 枚举已注册 | |
| B_SPLINE_CURVE | `Geom_BSplineCurve` | Done | | |
| B_SPLINE_CURVE_WITH_KNOTS | `Geom_BSplineCurve` | Done | | |
| RATIONAL_B_SPLINE_CURVE | `Geom_BSplineCurve` | Done | | |
| TRIMMED_CURVE | `Geom_TrimmedCurve` | Done | | |
| COMPOSITE_CURVE | `Geom_BSplineCurve` (merged) | Done | 枚举已注册 | |
| SEAM_CURVE | — | Done | 拓扑层处理 | |
| INTERSECTION_CURVE | — | Done | 枚举已注册 | |
| OFFSET_CURVE_3D | `Geom_OffsetCurve` | Done | 枚举已注册 | |
| BOUNDED_CURVE | supertype | Done | 枚举已注册 (meta) | |
| CURVE | supertype | Done | 枚举已注册 (meta) | |
| PCURVE / DEFINITIONAL_REPRESENTATION | `Geom2d_Curve` | Partial | PCURVE 提取框架存在，需完善裁剪 | |
| SURFACE_CURVE | — | Missing | AP242 常用，关联 3D 曲线 + PCURVE | |
| CONIC | supertype | Missing | AP242 supertype 链 | |

### 3.2 曲面实体

| 实体 | OCCT 类 | 覆盖率 | 状态 | 备注 |
|------|---------|--------|------|------|
| PLANE | `Geom_Plane` → NURBS | Done | | |
| CYLINDRICAL_SURFACE | `Geom_CylindricalSurface` | Done | | |
| CONICAL_SURFACE | `Geom_ConicalSurface` | Done | | |
| SPHERICAL_SURFACE | `Geom_SphericalSurface` | Done | | |
| TOROIDAL_SURFACE | `Geom_ToroidalSurface` | Done | | |
| B_SPLINE_SURFACE | `Geom_BSplineSurface` | Done | | |
| B_SPLINE_SURFACE_WITH_KNOTS | `Geom_BSplineSurface` | Done | | |
| RATIONAL_B_SPLINE_SURFACE | `Geom_BSplineSurface` | Done | | |
| SURFACE_OF_LINEAR_EXTRUSION | `Geom_SurfaceOfLinearExtrusion` | Done | | |
| SURFACE_OF_REVOLUTION | `Geom_SurfaceOfRevolution` | Done | | |
| OFFSET_SURFACE | `Geom_OffsetSurface` | Done | 枚举已注册 | |
| RECTANGULAR_TRIMMED_SURFACE | — | Done | 枚举已注册 | |
| CURVE_BOUNDED_SURFACE | — | Done | 枚举已注册 | |
| BOUNDED_SURFACE | supertype | Done | 枚举已注册 (meta) | |
| ELEMENTARY_SURFACE | supertype | Partial | AP242 build 有展开，但不完整 | |
| SWEPT_SURFACE | supertype | Partial | AP242 build 有展开，但不完整 | |
| SURFACE | supertype | Done | 枚举已注册 (meta) | |

### 3.3 拓扑实体

| 实体 | OCCT 类 | 覆盖率 | 状态 | 备注 |
|------|---------|--------|------|------|
| CLOSED_SHELL | `TopoDS_Shell` | Done | | |
| OPEN_SHELL | `TopoDS_Shell` | Done | | |
| ORIENTED_CLOSED_SHELL | `TopoDS_Shell` | Done | | |
| ORIENTED_OPEN_SHELL | `TopoDS_Shell` | Done | | |
| ADVANCED_FACE | `TopoDS_Face` | Done | | |
| FACE_SURFACE / FACE | `TopoDS_Face` | Done | | |
| FACE_OUTER_BOUND | `TopoDS_Wire` | Done | | |
| FACE_BOUND | `TopoDS_Wire` | Done | | |
| EDGE_CURVE | `TopoDS_Edge` | Done | | |
| ORIENTED_EDGE | `TopoDS_Edge` | Done | | |
| EDGE_LOOP | `TopoDS_Wire` | Done | | |
| POLY_LOOP | `TopoDS_Wire` (poly) | Done | 枚举已注册 | |
| VERTEX_POINT | `TopoDS_Vertex` | Done | | |

### 3.4 几何基础实体

| 实体 | OCCT 类 | 覆盖率 | 状态 | 备注 |
|------|---------|--------|------|------|
| CARTESIAN_POINT | `gp_Pnt` | Done | | |
| DIRECTION | `gp_Dir` | Done | | |
| VECTOR | `gp_Vec` | Done | | |
| AXIS2_PLACEMENT_3D | `gp_Ax3` → transform | Done | | |
| AXIS2_PLACEMENT_2D | `gp_Ax2d` | Done | | |
| REPRESENTATION_ITEM | supertype | Done | 枚举已注册 (meta) | |
| GEOMETRIC_REPRESENTATION_ITEM | supertype | Done | 枚举已注册 (meta) | |

### 3.5 装配体实体

| 实体 | OCCT 类 | 覆盖率 | 状态 | 备注 |
|------|---------|--------|------|------|
| NEXT_ASSEMBLY_USAGE_OCCURRENCE | `XCAFDoc_AssemblyItemId` | Done | | |
| PRODUCT_DEFINITION_SHAPE | `XCAFDoc_ShapeTool` | Done | | |
| SHAPE_DEFINITION_REPRESENTATION | — | Done | AP242 context link | |
| SHAPE_REPRESENTATION | — | Done | | |
| ADVANCED_BREP_SHAPE_REPRESENTATION | — | Done | | |
| ITEM_DEFINED_TRANSFORMATION | `TopLoc_Location` | Done | | |
| MANIFOLD_SOLID_BREP | `TopoDS_Solid` | Done | | |
| BREP_WITH_VOIDS | `TopoDS_Compound` | Done | | |
| SHELL_BASED_SURFACE_MODEL | `TopoDS_Compound` | Done | | |
| PRODUCT | — | Done | | |
| PRODUCT_DEFINITION | — | Done | | |
| PRODUCT_DEFINITION_FORMATION | — | Done | | |
| PRODUCT_DEFINITION_CONTEXT | — | Missing | AP242 上下文语义 | |
| PRODUCT_RELATED_PRODUCT_CATEGORY | — | Missing | AP242 产品分类 | |
| PRODUCT_CATEGORY_RELATIONSHIP | — | Missing | AP242 产品层级 | |
| APPLICATION_CONTEXT | — | Missing | 模式识别需要 | |
| APPLICATION_PROTOCOL_DEFINITION | — | Missing | 模式识别需要 | |
| MAPPED_ITEM | — | Missing | 变换 + 实例化 | |
| CONTEXT_DEPENDENT_SHAPE_REPRESENTATION | — | Missing | AP242 装配上下文 | |
| REPRESENTATION_RELATIONSHIP | — | Missing | 装配关系 (含变换) | |
| SHAPE_REPRESENTATION_RELATIONSHIP | — | Done | SHAPE_REPRESENTATION 别名 | |

### 3.6 PMI / 标注实体

| 实体 | OCCT 类 | 覆盖率 | 状态 | 备注 |
|------|---------|--------|------|------|
| STYLED_ITEM | `XCAFDoc_ColorTool` | Done | 颜色提取 | |
| PRESENTATION_STYLE_ASSIGNMENT | — | Done | | |
| ANNOTATION_OCCURRENCE | `XCAFDoc_NotesTool` | Done | 枚举已注册 | |
| DIMENSIONAL_SIZE | — | Partial | 提取框架存在，无渲染 | |
| DIMENSIONAL_CHARACTERISTIC_REPRESENTATION | — | Partial | 嵌套提取未完成 | |
| ANGULAR_SIZE | — | Done | 枚举已注册 | |
| DATUM | — | Partial | 提取框架存在，无渲染 | |
| DATUM_FEATURE | — | Done | 枚举已注册 | |
| DATUM_TARGET | — | Done | 枚举已注册 | |
| GEOMETRIC_TOLERANCE | — | Partial | 提取框架存在，无渲染 | |

以下 PMI 实体在 OCCT 中有处理但仓库中完全缺失：

| 实体 | 状态 | AP242 角色 |
|------|------|-----------|
| DIMENSIONAL_LOCATION | Missing | 位置尺寸（中心距等） |
| DIMENSIONAL_SIZE_WITH_DATUM | Missing | 带基准的尺寸 |
| DIMENSIONAL_SIZE_WITH_PATH | Missing | 路径尺寸（弧长等） |
| GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE | Missing | 带基准引用的公差 |
| GEOMETRIC_TOLERANCE_WITH_MODIFIERS | Missing | 带修饰符的公差 |
| GEOMETRIC_TOLERANCE_WITH_DEFINED_UNIT | Missing | 带单位的公差 |
| DATUM_REFERENCE_COMPARTMENT | Missing | 基准引用框 |
| DATUM_SYSTEM | Missing | 基准系（多基准） |
| DATUM_REFERENCE_ELEMENT | Missing | 基准引用元素 |
| GENERAL_DATUM_REFERENCE | Missing | 通用基准引用 |
| MODIFIED_GEOMETRIC_TOLERANCE | Missing | 修饰公差 |
| UNEQUALLY_DISPOSED_GEOMETRIC_TOLERANCE | Missing | 非对称公差 |
| PROFILE_TOLERANCE | Missing | 轮廓公差 |
| FLATNESS_TOLERANCE | Missing | 平面度 |
| ROUNDNESS_TOLERANCE | Missing | 圆度 |
| CYLINDRICITY_TOLERANCE | Missing | 圆柱度 |
| PERPENDICULARITY_TOLERANCE | Missing | 垂直度 |
| PARALLELISM_TOLERANCE | Missing | 平行度 |
| ANGULARITY_TOLERANCE | Missing | 倾斜度 |
| SURFACE_PROFILE_TOLERANCE | Missing | 面轮廓度 |
| LINE_PROFILE_TOLERANCE | Missing | 线轮廓度 |
| POSITION_TOLERANCE | Missing | 位置度 |
| CONCENTRICITY_TOLERANCE | Missing | 同轴度 |
| SYMMETRY_TOLERANCE | Missing | 对称度 |
| CIRCULAR_RUNOUT_TOLERANCE | Missing | 圆跳动 |
| TOTAL_RUNOUT_TOLERANCE | Missing | 全跳动 |
| STRAIGHTNESS_TOLERANCE | Missing | 直线度 |
| PLUS_MINUS_TOLERANCE | Missing | 对称公差 |
| LIMITS_AND_FITS | Missing | 极限配合 |
| DIMENSION_CURVE_DIRECTED_CALLOUT | Missing | 标注引导线 |
| LEADER_DIRECTED_CALLOUT | Missing | 引线标注 |
| STYLED_ITEM / PMI 样式 | Missing | PMI 颜色/线型/字体 | |

### 3.7 镶嵌几何实体 (AP242 Edition 2+)

AP242 Edition 2 引入了并行于 B-Rep 的镶嵌几何表示。OCCT 通过 `RWStl` / `RWPly` 类处理。

| 实体 | 状态 | AP242 角色 |
|------|------|-----------|
| TESSELLATED_ITEM | Missing | 镶嵌几何的顶层容器 |
| TESSELLATED_SHAPE_REPRESENTATION | Missing | 镶嵌表示的 context |
| TESSELLATED_SHELL | Missing | 三角网格壳体 |
| TESSELLATED_SOLID | Missing | 三角网格实体 |
| TESSELLATED_FACE | Missing | 单个面网格 |
| TESSELLATED_EDGE | Missing | 边线网格 |
| TESSELLATED_VERTEX | Missing | 顶点 |
| COORDINATES_LIST | Missing | 顶点坐标数组 |
| COORDINATES_AXIS2_PLACEMENT_3D | Missing | 局部坐标系坐标 |
| TRIANGULATED_FACE | Missing | 三角形面片索引 |
| COMPLEX_TRIANGULATED_FACE | Missing | 复杂三角化（带法线/颜色） |
| TRIANGLE_STRIP | Missing | 三角带 |
| TRIANGLE_FAN | Missing | 三角扇 |
| POLYGONAL_FACE | Missing | 多边形面片 |
| TESSELLATED_GEOMETRIC_SET | Missing | 镶嵌几何集 |

### 3.8 AP242 高级曲面/实体 (Edition 3)

AP242 Edition 3 增加了复材、3D 打印和线束支持。这些实体在 OCCT 中也仅部分支持，对本仓库为 NA。

| 实体 | 状态 | 原因 |
|------|------|------|
| COMPOSITE_SHAPE_REPRESENTATION | NA | 复材，超出范围 |
| ADDITIVE_MANUFACTURING_SHAPE_REPRESENTATION | NA | 3D 打印，超出范围 |
| WIRING_HARNESS_SHAPE_REPRESENTATION | NA | 线束，超出范围 |

### 3.9 样式/颜色实体

| 实体 | OCCT 类 | 覆盖率 | 状态 |
|------|---------|--------|------|
| COLOUR_RGB | `Quantity_Color` | Done | |
| SURFACE_STYLE_USAGE | — | Done | |
| SURFACE_SIDE_STYLE | — | Done | |
| SURFACE_STYLE_FILL_AREA | — | Done | |
| FILL_AREA_STYLE | — | Done | |
| COLOUR | — | Done | |
| DRAUGHTING_PRE_DEFINED_COLOUR | — | Missing | 标准预定义颜色 |
| PRESENTATION_STYLE_BY_CONTEXT | — | Missing | AP242 上下文样式 |
| SURFACE_STYLE_TRANSPARENT | — | Missing | 透明度 |
| SURFACE_STYLE_PARAMETER_LINE | — | Missing | 参数线样式 |
| SURFACE_STYLE_CONTROL_GRID | — | Missing | 控制网格样式 |
| SURFACE_STYLE_SEGMENTATION_CURVE | — | Missing | 分曲线样式 |
| SURFACE_RENDERING_PROPERTIES | — | Missing | 渲染属性 |
| MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION | — | Missing | PMI 样式关联 |

---

## 四、B-Rep 处理管线覆盖

### 4.1 拓扑构建 (StepToTopoDS)

| 阶段 | OCCT | 本仓库 | 状态 | 备注 |
|------|------|--------|------|------|
| Vertex 构建 | `TranslateVertex` | `topology.rs` build_vertex | Done | |
| Edge + PCURVE | `TranslateEdgeLoop` | `brep/build.rs` Pass 2 | Done | |
| Wire 构建 | `TranslateVertexLoop` | `topology.rs` + `build_vertex_loop_wire` | Done | 含退化边 |
| Face 构建 | `TranslateFace` | `brep/build.rs` Pass 1+3 | Done | |
| Shell 构建 | `TranslateShell` | `brep/build.rs` Pass 3 | Done | |
| Solid 构建 | `TranslateSolid` | `brep/build.rs` Pass 4 | Done | |
| Void 处理 | `TranslateCompound` | `brep/build.rs` 有统计但未布尔减 | Partial | |
| 朝向传播 | `PropagateOrientation` | `heal/orient.rs` fix_shell_orientation | Done | |

### 4.2 修复管线 (ShapeFix)

| 修复器 | OCCT 类 | 本仓库 | 状态 |
|--------|---------|--------|------|
| Reorder edges | `ShapeFix_Wire::FixReorder` | `heal/reorder.rs` | Done |
| Close gaps (3D) | `ShapeFix_Wire::FixGaps3d` | `heal/gap.rs` | Done |
| Close gaps (2D) | `ShapeFix_Wire::FixGaps2d` | `heal/gap.rs` close_wire_gaps_2d | Done |
| Connected wire | `ShapeFix_Wire::FixConnected` | `heal/connected.rs` | Done |
| Small edges | `ShapeFix_Wire::FixSmall` | `heal/small.rs` | Done |
| Shifted PCURVEs | `ShapeFix_Wire::FixShifted` | `heal/shifted.rs` | Done |
| Edge curves | `ShapeFix_Edge::FixVertexTolerance` | `heal/edge_curve.rs` | Done |
| Lacking edges | `ShapeFix_Wire::FixLacking` | `heal/lacking.rs` | Done |
| Self-intersection | `ShapeFix_Wire::FixSelfIntersection` | `heal/self_intersect.rs` | Done |
| Intersecting wires | `ShapeFix_Wire::FixIntersectingWires` | `heal/intersecting_wires.rs` | Done |
| Degenerated edges | `ShapeFix_Wire::FixDegenerated` | `heal/degenerated.rs` | Done |
| Periodic degenerated | — | `heal/periodic.rs` | Done |
| Vertex position | `ShapeFix_Wire::FixVertexPosition` | `heal/vertex_position.rs` | Done |
| Split face | `ShapeFix_Face::FixSplitFace` | `heal/split_face.rs` | Done |
| Missing seam | `ShapeFix_Face::FixMissingSeam` | `heal/seam.rs` | Done |
| Natural bound | `ShapeFix_Face::FixAddNaturalBound` | `heal/natural_bound.rs` | Done |
| Reversed 2D | `ShapeFix_Face::FixReversed2d` | `heal/reversed2d.rs` | Done |
| Same parameter | `BRepLib::SameParameter` | `brep/same_parameter.rs` + mesh same_param | Partial |
| 自愈管线 | — | `heal/pipeline.rs` auto_heal_shell | Done |

**结论**: B-Rep 修复管线对齐度很高（16/18 Done, 1 Partial）。

### 4.3 网格化 (BRepMesh)

| 功能 | OCCT | 本仓库 | 状态 |
|------|------|--------|------|
| Mesh algorithm selection | `BRepMesh_MeshAlgoFactory` | `mesh/algo_factory.rs` | Partial |
| 规则曲面 CDT | `BRepMesh_FastDiscretFace` | `mesh/face_cdt.rs` TrimmedCdt | Done |
| 自由曲面填充 | `BRepMesh_DelaunayDeflection` | `mesh/face_fill.rs` | Partial |
| Edge discretization | `BRepMesh_EdgeDiscret` | `mesh/edge_disc.rs` (via mesh) | Partial |
| 相对 Deflection | `BRepMesh_IncrementalMesh::IsRelative` | `StepImportOptions.mesh_relative_deflection` | Done |
| 退化边 CDT 约束 | `BRepMesh_FaceChecker` | `mesh/face_cdt.rs` | Done |
| 全局属性计算 | `BRepGProp` | `brep/properties.rs` (via mesh result) | Done |

---

## 五、装配体覆盖

### 5.1 当前实现

| 功能 | 状态 | 说明 |
|------|------|------|
| NEXT_ASSEMBLY_USAGE_OCCURRENCE 解析 | Done | 支持 AP203 + AP242 PDS |
| 变换提取和传播 | Done | `ITEM_DEFINED_TRANSFORMATION` 累积 |
| shell 实例化 | Done | `extract_shell_instances` |
| STYLED_ITEM 颜色提取 | Done | 单层样式 |
| SHAPE_DEFINITION_REPRESENTATION | Done | AP242 context link |
| 装配树构建 | Done | `build_assembly_tree` 扁平结构 |

### 5.2 缺口

| 功能 | 状态 | 说明 |
|------|------|------|
| 多层级装配树 (递归) | Partial | 仅扁平提取，无树结构保留 |
| 样式/颜色继承 | Missing | 父装配的 STYLED_ITEM 不传播到子部件 |
| 透明度和图层 | Missing | SURFACE_STYLE_TRANSPARENT 未处理 |
| AP242 CONTEXT_DEPENDENT 变换 | Missing | 上下文相关的形状表示 |
| PRODUCT_DEFINITION_CONTEXT 识别 | Missing | 无法判断产品定义上下文 |
| APPLICATION_PROTOCOL_DEFINITION 检测 | Missing | 无法识别文件的 AP 模式 |
| MAPPED_ITEM 实例化 | Missing | 共享几何的映射实例化 |

---

## 六、PMI / 标注覆盖

### 6.1 当前实现

PMI 模块 (`step/pmi/`) 提供数据提取框架：
- `PmiDimension` — 线性尺寸 (start, end, offset_dir, text)
- `PmiDatum` — 基准标识 (origin, normal, label)
- `PmiToleranceFrame` — 公差框 (origin, leader_points, text)
- `extract_pmi()` — 遍历 EntityIndex 提取 PMI 数据

### 6.2 缺口

| 功能 | 状态 | 说明 |
|------|------|------|
| 尺寸标注提取 | Partial | 仅 DIMENSIONAL_SIZE，缺少 ANGULAR_SIZE 提取 |
| 公差框提取 | Partial | 仅 GEOMETRIC_TOLERANCE，无基准引用/修饰符 |
| 基准提取 | Partial | 缺少 DATUM_FEATURE 和 DATUM_TARGET 解析 |
| PMI → SceneGraph 注入 | Missing | 提取的数据未转换为场景节点 |
| PMI 渲染 (3D 标注) | Missing | 无标注渲染 pass |
| ANNOTATION_OCCURRENCE 关联 | Missing | 标注与几何的关联未解析 |
| DIMENSIONAL_CHARACTERISTIC_REPRESENTATION | Missing | 嵌套值提取 |
| 公差类型细化 | Missing | 25+ 种公差子类型全部缺失 |
| Plus/Minus 公差 | Missing | 无对称/非对称公差提取 |
| 基准系 | Missing | 多基准引用未实现 |
| 标注引导线 (Leader) | Missing | 引线标注几何 |
| 语义 PMI (Semantic PMI) | Missing | AP242 Edition 2 语义标注 |

---

## 七、验证规则覆盖

| 检查项 | OCCT BRepCheck | 本仓库 | 状态 |
|--------|---------------|--------|------|
| 悬挂引用检测 | `BRepCheck_Analyzer` | `validate.rs` | Done |
| EXPRESS 模式约束 | — | `validate.rs` schema_violations | Partial |
| 拓扑完备性 | `BRepCheck_Shell` | `validate.rs` TopologyInfo | Partial |
| G0 连续性 | `BRepCheck_Edge` | `heal/continuity.rs` check_shell_continuity | Done |
| G1 连续性 | — | `heal/continuity.rs` | Done |
| UV 自交检测 | `BRepCheck_Face` | `heal/check.rs` check_uv_self_intersection | Done |
| 朝向一致性 | `BRepCheck_Shell` | `heal/orient.rs` | Done |
| 最小面/边尺寸 | — | `heal/small.rs` | Done |
| 非流形检测 | `BRepCheck_Analyzer` | `heal/check.rs` check_shell | Partial |
| AP 模式检测 | — | Missing | 无 AP203/AP214/AP242 区分 |
| 缺失实体类型检测 | — | StepImportReport.unknown_entity_count | Done |
| PMI 验证 | — | Missing | 无标注完整性检查 |

---

## 八、缺口优先级排序

### P0 — 阻塞基本 AP242 使用

| 缺口 | 影响 | 预估工作量 |
|------|------|-----------|
| AP242 模式检测 (APPLICATION_PROTOCOL_DEFINITION) | 无法区分文件类型，统计和诊断不准确 | 0.5 天 |
| PMI → SceneGraph 注入 + 3D 标注渲染 | PMI 数据提取了但不可见，AP242 核心价值缺失 | 3-5 天 |
| Void shell 布尔减处理 | BREP_WITH_VOIDS 被统计但未布尔运算 | 1-2 天 |
| SURFACE_CURVE 实体支持 | AP242 常用实体，影响边线正确性 | 0.5 天 |

### P1 — 显著改善 AP242 质量

| 缺口 | 影响 | 预估工作量 |
|------|------|-----------|
| SameParameter 完整实现 (tolerance 回写) | 网格化水密性 | 1-2 天 |
| ANNOTATION_OCCURRENCE 关联解析 | PMI 无法定位到几何 | 1 天 |
| DIMENSIONAL_CHARACTERISTIC_REPRESENTATION | 尺寸值无法读取 | 0.5 天 |
| 装配体样式/颜色继承 | 子部件缺失颜色 | 1 天 |
| DRAUGHTING_PRE_DEFINED_COLOUR | 标准颜色名无法解析 | 0.5 天 |
| CONTEXT_DEPENDENT_SHAPE_REPRESENTATION | AP242 装配上下文变换 | 1 天 |

### P2 — AP242 深度支持

| 缺口 | 影响 | 预估工作量 |
|------|------|-----------|
| 公差子类型细化 (25+ 种) | 公差显示不完整 | 3-5 天 |
| 基准系 (DATUM_SYSTEM) | 多基准语义缺失 | 1-2 天 |
| 标注引导线渲染 | 标注视觉质量 | 1 天 |
| 镶嵌几何实体集 | AP242 Edition 2 并行几何流 | 2-3 天 |
| 语义 PMI (AP242 Edition 2) | 富标注语义 | 2-3 天 |
| 多层级装配树保留 | 装配结构不可交互 | 1-2 天 |
| PRODUCT_CATEGORY_RELATIONSHIP | 产品分类 | 0.5 天 |
| MAPPED_ITEM 实例化 | 共享几何内存优化 | 1 天 |
| 透明度 / SURFACE_STYLE_TRANSPARENT | 透明面渲染 | 0.5 天 |

### P3 — 扩展 / NA

| 缺口 | 状态 | 原因 |
|------|------|------|
| 复材 (COMPOSITE_*) | NA | 超出工业机械范围 |
| 3D 打印 (ADDITIVE_MANUFACTURING_*) | NA | 超出范围 |
| 线束 (WIRING_HARNESS_*) | NA | 超出范围 |
| 运动学 (KINEMATIC_*) | NA | 超出范围 |
| NC 加工 (MACHINING_*) | NA | 超出范围 |

---

## 九、与现有开发计划的衔接

当前 Phase 1-4 计划主题是**几何正确性与修复**（mesh correctness, quality, automation），未覆盖 AP242 差异化能力。建议新增：

| Phase | 主题 | 涵盖的 P0 缺口 |
|-------|------|---------------|
| Phase 5: AP242 基础 | 模式检测 + void 处理 + SURFACE_CURVE | P0-1, P0-3, P0-4 |
| Phase 6: PMI 渲染 | 标注 SceneGraph 注入 + 渲染 pass | P0-2, P1-2, P1-3 |
| Phase 7: PMI 完整化 | 公差子类型 + 基准系 + 引导线 | P2-1, P2-2, P2-3 |
| Phase 8: AP242 深度 | 装配体继承 + 镶嵌几何 + 语义 PMI | P1-5, P2-4, P2-5 |

---

## 十、建议的下一步

1. **P0-1 优先**: 在 parser 中添加 `APPLICATION_PROTOCOL_DEFINITION` 解析，作为所有 AP242 工作的起点——先能识别文件是 AP242 还是 AP203。
2. **收集 AP242 测试文件**: 需要带 PMI 标注的真实 AP242 文件、带 void 的复杂实体、以及 AP242 多层级装配体。
3. **Phase 5-8 排期**: 在 Phase 1-4 完成后启动。

---

## 附录 A: OCCT 参考源文件

| 文件 | 模块 | 说明 |
|------|------|------|
| `src/StepToTopoDS/StepToTopoDS_Translate*` | StepToTopoDS | 实体 → B-Rep 转换 |
| `src/StepToGeom/StepToGeom_Make*` | StepToGeom | 实体 → 几何表示 |
| `src/ShapeFix/ShapeFix_*` | ShapeFix | B-Rep 修复工具集 |
| `src/BRepCheck/BRepCheck_*` | BRepCheck | 拓扑验证 |
| `src/BRepMesh/BRepMesh_*` | BRepMesh | Delaunay 网格化 |
| `src/XCAFDoc/XCAFDoc_*` | XCAF | 装配体/颜色/图层/PMI |
| `src/RWStl/RWStl_*` | RWStl | 镶嵌几何 |

## 附录 B: ISO 10303-242 关键章节索引

| 章节 | 内容 | 与分析的对应 |
|------|------|-------------|
| Clause 4.2 | AP242 ARM entities | 实体类型矩阵 |
| Clause 5 | Shape and assembly (AP203/AP214) | B-Rep + 装配体 |
| Clause 6 | PMI | PMI / 标注 |
| Clause 7 | Kinematics | NA |
| Clause 8 | Composites | NA (P3) |
| Clause 9 | Tessellated geometry | 镶嵌几何 (P2) |
| Clause 10 | Semantic PMI | 语义 PMI (P2) |
| Clause 11 | Additive manufacturing | NA (P3) |
| Clause 12 | Electrical harness | NA (P3) |

## 附录 C: 建议测试文件清单

| 优先级 | 文件类型 | 测试目标 |
|--------|---------|---------|
| P0 | AP242 带 PMI 的单零件 | 尺寸标注提取 + 渲染 |
| P0 | AP242 带 void 的实体 | 布尔减 + 孔洞 |
| P0 | AP242 带 SURFACE_CURVE | 边线正确性 |
| P1 | AP242 多层级装配体 | 装配树 + 样式继承 |
| P1 | AP242 带 DATUM_SYSTEM | 基准系渲染 |
| P1 | AP242 带多种公差 | 公差类型细化 |
| P2 | AP242 Edition 2 镶嵌几何 | 三角网格实体 |
