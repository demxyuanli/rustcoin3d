# STEP 解析可视化全面审查 — OCC 对齐路线图

**Date**: 2026-05-30
**Status**: Approved
**Scope**: 4阶段实施：遗留清理→性能优化→布尔/Offset→高级功能

---

## 架构现状

### 当前管线

```
parser.rs (Part21解析)
  → adapter/ (复合实体扁平化)
  → brep/build/ (4遍B-Rep构建: solid→shell→face→wire→edge)
  → brep/heal/ (22个修复pass，对齐OCC ShapeFix)
  → brep/mesh/ (边离散→UV环→CDT→Steiner→细化→优化)
  → import_pipeline.rs (场景图发射)
```

### 遗留管线（待清理）

```
topology/ → pcurve.rs → tessellate.rs → surface_tess.rs → refine.rs
```

### 模块规模

| 模块 | 大小 | 说明 |
|---|---|---|
| `brep/mesh/face_fill.rs` | 88.8KB | 最大文件，多策略混合 |
| `brep/geom/curve_surface.rs` | 81.7KB | 曲面+曲线求值混合 |
| `brep/mesh/mod.rs` | 71.3KB | 核心网格逻辑 |
| `surface_tess.rs` | 60.9KB | 遗留管线最大文件 |
| `brep/mesh/face_uv.rs` | 35.8KB | UV环处理 |
| `brep/mesh/face_cdt.rs` | 33.5KB | CDT三角化 |

---

## Phase 1: 遗留清理与代码去重（1-2周）

**目标**: 删除约2000行死代码，消除7处重大重复，建立干净的单一管线。

### Task 1.1: 删除遗留管线文件

| 文件 | 行数 | 动作 | 风险 |
|---|---|---|---|
| `step/tessellate.rs` | 350 | 删除 | 低 — 已标记`#[deprecated]` |
| `step/surface_tess.rs` | 1561 | 删除 | 低 — 标记为legacy，无活跃调用者 |
| `step/fillet.rs` | 47 | 删除 | 低 — 返回Err的存根 |
| `step/bool/` 中旧StepShell入口 | ~30 | 删除`boolean()`旧入口 | 低 — 仅测试引用 |

**验证**: `cargo test -p rc3d-io` 全通过 + `cargo build --release` 无 dead_code 警告。

### Task 1.2: NURBS构建代码抽取

**重复来源**:

- `surface_tess.rs::build_nurbs_from_bspline_surface()` (L79-141)
- `brep/build/surface.rs::build_nurbs_surface()` (L122-180)

**目标**: 新建 `brep/geom/nurbs_build.rs`，包含三个函数：

```rust
pub fn build_nurbs_surface(record: &EntityRecord, entities: &EntityIndex) -> Option<NurbsSurface>
fn build_surface_knots(mults: &[i64], vals: &[StepValue], deg: usize, cp: usize) -> Vec<f32>
fn find_surface_weights(params: &StepValue, rows: usize, cols: usize) -> Option<Vec<Vec<f32>>>
```

`brep/build/surface.rs` 改为调用此模块。删除 `surface_tess.rs` 后旧版本自然消失。

### Task 1.3: 小型重复消除

| 重复 | 统一到 | 动作 |
|---|---|---|
| `face_uv.rs::signed_area_2d()` (f64) vs `heal/check.rs::signed_area_2d()` (f32) | `brep/geom/` 中统一f64版 | heal/check.rs改用公共版本 |
| `algo_factory.rs::prefers_native_uv_trim()` vs `face_dispatch.rs::prefers_native_uv_trim()` | `face_dispatch.rs` 保留一处 | algo_factory.rs导入 |
| `brep/build/pcurve.rs::resolve_placement_2d()` vs `pcurve.rs::resolve_placement_2d()` | `brep/build/pcurve.rs` | 确认旧pcurve.rs版本无活跃调用者后移除 |

### Task 1.4: `step/mod.rs` 模块声明清理

移除 `pub mod tessellate;`、`pub mod surface_tess;`、`pub mod fillet;` 声明。

### Task 1.5: 布尔模块清理

- 删除 `bool/mod.rs::boolean()` 旧入口（操作 `StepShell` 类型）
- 保留 `boolean_brep()` 框架作为Phase 3入口点
- 删除 `classify.rs`、`split.rs`、`select.rs` 中操作旧类型的代码
- 保留 `intersect.rs` 中的面-面求交逻辑（可迁移到B-Rep）

### 验证计划

1. `cargo test -p rc3d-io` — 所有测试通过
2. `cargo build --release` — 无 dead_code / unused 警告
3. 对 `Shape.step` / `Cube.step` / `cs.step` 运行导入验证网格输出不变
4. `cargo test -p rc3d-io --test step_files` — 所有测试文件正常加载

---

## Phase 2: Heal 稳定性与性能优化（2-3周）

**目标**: 修复Offset面网格bug，提升Heal通过率，多线程加速4-8x。

### Task 2.1: OffsetSurface完整求值

当前 `SurfaceGeom::Offset` 存储了基面+距离，但 `d0()`/`project()`/`normal()` 未正确实现。

```rust
// brep/geom/curve_surface.rs 中需要实现:
impl SurfaceGeom {
    fn offset_d0(&self, basis: &SurfaceGeom, distance: f32, u: f32, v: f32) -> Vec3 {
        let p = basis.d0(u, v);
        let n = basis.normal(u, v);
        p + n * distance
    }
}
```

**验证**: Offset平面/圆柱/圆锥面的网格质量 vs 基面偏移一致性。

### Task 2.2: 闭合性与体积验证

- `check.rs` 增加 Euler-Poincaré 公式验证: `V - E + F = 2(1 - genus)`
- `properties.rs` 补全体积计算（有向面积法 / 散度定理）
- 新增测试: 封闭壳体积 > 0，开放壳体积 = 0

### Task 2.3: Rayon并行面网格化

**当前**: `mesh_brep_shell_with_report_impl()` 单线程遍历 `shell.faces`。

**改进**:
```rust
use rayon::prelude::*;

let face_results: Vec<_> = face_infos
    .par_iter()
    .map(|info| mesh_single_face(info, &reg, &config, &edge_polygons))
    .collect();

// 合并: 每面独立的 (vertices, normals, indices) → 全局merge
merge_face_results(face_results, &mut global_vertices, &mut global_normals, &mut all_indices);
```

**风险**: `pos_to_idx` 全局去重需要原子操作或后处理合并。建议先per-face独立mesh，最后merge去重。

### Task 2.4: 边→面索引加速

**当前**: `registry.rs::find_shared_edges()` O(n²) 遍历所有面对。

**改进**: 在 `BRepRegistry` 构建时维护:
```rust
pub edge_to_faces: HashMap<EdgeKey, Vec<FaceKey>>,
```

查找共享边: O(1) 通过边key直接获取关联面列表。

### Task 2.5: Mesh零拷贝发射

**当前**: `import_pipeline.rs::add_mesh_nodes()` 对 vertices/normals/indices 各做一次 `.clone()`。

**改进**:
- 单实例场景: 使用 `std::mem::take()` move 数据到SceneGraph
- 多实例场景: `Arc<MeshResult>` 共享，TransformNode 仅包含变换矩阵

### Task 2.6: 实例化装配渲染

**当前**: 每个装配实例独立clone整个网格。

**改进**:
```rust
struct InstancedMesh {
    mesh: Arc<MeshResult>,     // 共享网格数据
    instances: Vec<Mat4>,      // 每个实例的变换矩阵
}
```

内存收益: N个实例从 N×mesh_size 降低到 mesh_size + N×64bytes。

### Task 2.7: 大文件拆分

| 文件 | 拆分方案 |
|---|---|
| `curve_surface.rs` (81.7KB) | 拆为 `surface_eval.rs`(曲面求值/法线/投影) + `curve_eval.rs`(曲线求值/离散) |
| `face_fill.rs` (88.8KB) | 拆为 `fill_plane.rs`(平面扇形/中心扇) + `fill_revolution.rs`(旋转面网格) + `fill_surface.rs`(通用CDT/Steiner) |
| `mesh/mod.rs` (71.3KB) | 提取 `ruled_mesh.rs`(直纹面) + `parametric_grid.rs`(参数网格) |

### 验证计划

1. 性能基准: `Shape.step` / `bender assembly v54.step` / `Porsche_911_GT2.stl` 导入时间 (before/after)
2. 正确性: 所有现有测试 + 新增Offset面、闭合性测试
3. 内存: 峰值内存对比 (before/after)
4. 回归: `cargo test -p rc3d-io --test shape_compare` (需要OCC参考STL)

---

## Phase 3: 布尔运算与 Offset 基础（3-4周）

**目标**: 实现可工作的B-Rep布尔运算和完整Offset面支持。

### Task 3.1: B-Rep布尔运算重建

```
Step 1: 面-面求交 (BRepIntersector)
  - 曲面对求交 → 交线离散化 → 交点参数化
  - 复用现有 intersect.rs 中的基础几何求交
  - OCC参考: IntTools_FaceFace

Step 2: 面分割 (BRepSplitter)
  - 沿交线分割面 → 新边/新wire生成
  - 利用现有 heal/split_face.rs 的面分割基础设施
  - OCC参考: BRepAlgoAPI_Splitter

Step 3: 面分类 (PointInSolid)
  - 射线法点-实体分类 → 面内外判定
  - 扩展现有 classify.rs 的射线投射逻辑
  - OCC参考: BRepClass3d_SolidClassifier

Step 4: 面选择 + 缝合
  - 按布尔操作类型(Union/Intersect/Diff)选择面
  - 缝合为新壳 → 注册到BRepRegistry
  - OCC参考: BRepAlgoAPI selection tables
```

**新增类型**:
```rust
pub struct BRepBoolOp {
    pub op: BoolOp,
    pub intersections: Vec<IntersectionCurve>,
    pub split_faces_a: Vec<FaceRegion>,
    pub split_faces_b: Vec<FaceRegion>,
}
```

### Task 3.2: Offset曲面完整支持

| 子任务 | 描述 | 复杂度 |
|---|---|---|
| `SurfaceGeom::Offset::d0()` | 基面 d0() + normal() × distance | 低 |
| `SurfaceGeom::Offset::project()` | 牛顿法迭代投影 | 中 |
| `SurfaceGeom::Offset::normal()` | 基面 normal() (Offset不改变法线方向) | 低 |
| 网格路径 | CDT域填充，deflection考虑Offset距离 | 中 |
| 自交检测 | Offset距离 > 最小曲率半径时报警 | 高 |

### Task 3.3: 圆角/倒角最小框架

**Phase 3仅建立框架，不追求完整实现:**

```rust
// brep/fillet.rs (新)
pub fn constant_radius_fillet(
    edge: EdgeKey,
    radius: f32,
    reg: &mut BRepRegistry,
) -> Result<FilletResult, String>;

pub struct FilletResult {
    pub new_faces: Vec<FaceKey>,
    pub modified_faces: Vec<FaceKey>,
    pub removed_edges: Vec<EdgeKey>,
}
```

**最小验证**: 简单Cube单边圆角 → 生成圆柱面 + 裁剪相邻面。

### 验证计划

1. 布尔运算: Cube∪Sphere、Cube∩Sphere、Cube-Sphere 三组基准
2. Offset: 平面/圆柱/圆锥 Offset面网格质量
3. 圆角: 简单Cube单边圆角
4. 集成: 导入含布尔操作的STEP文件

---

## Phase 4: 高级功能与持续优化（持续）

**目标**: 逐步补全工业级功能，建立质量保证基线。

### Task 4.1: BRepOffsetAPI 抽壳

基于Phase 3的Offset + 布尔差集:
1. 对实体做Offset（向内）
2. 移除指定面
3. 布尔差集: 原始实体 - Offset实体
4. 缝合内外表面

### Task 4.2: AP242 PMI完整渲染

扩展 `pmi/` 模块:
- 尺寸标注 → `AnnotationNode` + 文本渲染
- 基准参考 → 几何约束标注
- 形位公差 → 特征控制框
- 表面粗糙度 → 符号标注

### Task 4.3: 增量网格更新

仅对修改的面重新网格化:
```rust
pub fn remesh_modified_faces(
    shell: ShellKey,
    modified: &[FaceKey],
    reg: &BRepRegistry,
    config: &BRepMeshConfig,
    existing_mesh: &mut ShellMeshResult,
) -> RemeshReport;
```

### Task 4.4: LOD层级

```rust
pub struct LodMesh {
    pub levels: Vec<MeshResult>,  // [高, 中, 低]
    pub transitions: Vec<f32>,    // 切换距离
}
```

### 质量保证基线

| 指标 | Phase 1后目标 | Phase 4目标 |
|---|---|---|
| STEP解析成功率 | >85% | >95% |
| Heal通过率(无skip) | >70% | >90% |
| 网格质量(max chord) | <defl×3 | <defl×2 |
| 导入时间(10MB, 4核) | <10s | <3s |
| 内存峰值(10MB) | <800MB | <300MB |

---

## 已知Bug清单

| # | 描述 | 位置 | 影响 | 阶段 |
|---|---|---|---|---|
| B1 | OffsetSurface无d0/project实现 | `curve_surface.rs` | Offset面网格错误 | P2 |
| B2 | 复合实体leaf_index优先级回退 | `adapter/subsuper.rs` | NURBS构建panic | 已修复 |
| B3 | `void_subtract` 空间网格精度 | `void_subtract.rs` | 大void mesh分类错误 | P2 |
| B4 | `face_fill` loops.clone() 后修改不反映原数据 | `face_fill.rs:1841` | 潜在UV不一致 | P2 |
| B5 | `signed_area_2d` f32/f64精度差异 | `face_uv.rs` vs `check.rs` | 边界判定不一致 | P1 |

---

## 依赖关系

```
Phase 1 (清理去重)
  └──→ Phase 2 (稳定性+性能)
         └──→ Phase 3 (布尔+Offset)
                └──→ Phase 4 (高级功能)
```

Phase 1 是后续所有阶段的前提 — 清理死代码减少维护面积。
Phase 2 的性能优化使Phase 3的布尔运算在合理时间内完成。
Phase 3 的布尔运算是Phase 4抽壳/特征操作的基础。

---

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 删除遗留管线影响外部使用者 | 低 | 中 | 保留一个版本的deprecation警告 |
| Rayon并行引入数据竞争 | 中 | 高 | per-face独立mesh + 后处理merge |
| 布尔运算数值不稳定 | 高 | 高 | 容差控制 + 退化情况回退到mesh-level |
| 大文件拆分引入模块循环依赖 | 低 | 中 | 先分析依赖图，逐步拆分 |
| Offset自交检测遗漏 | 中 | 中 | 保守策略：可疑Offset面降级为密集采样 |
