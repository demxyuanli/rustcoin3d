# STEP 格式解析 — 深度审查与推进计划

**日期**: 2026-05-23  
**审查范围**: `rc3d-io/src/step/` (branch: `feat+step-importer`)  
**对比基准**: Open CASCADE Technology (OCC) STEP 处理器

---

## 一、当前实现总览

### 1.1 模块结构

| 模块 | 文件 | 行数 | 状态 |
|------|------|------|------|
| STEP 解析器 | `parser.rs` | ~518 | 基础框架完成 |
| 实体类型分类 | `entity_types.rs` | ~130 | 覆盖 ~40 种类型 |
| B-rep 拓扑遍历 | `topology.rs` | ~500 | 基础框架完成 |
| 曲线几何求值 | `geom.rs` | ~400 | 部分完成 |
| 曲面细分 (UV 采样) | `surface_tess.rs` | ~240 | **有严重 Bug** |
| 面细分调度 | `tessellate.rs` | ~200 | 部分完成 |
| 装配体层级重建 | `assembly.rs` | ~300 | 部分完成 |
| PMI 标注 | `pmi/` | -- | 框架存在，未验证 |
| 主入口 | `mod.rs` | ~120 | 完成 |

### 1.2 数据流水线

```
STEP 文件 (.stp)
    │
    ▼
parser.rs: parse_exchange()  →  EntityIndex (HashMap<u64, EntityRecord>)
    │
    ▼
topology.rs: collect_shells()  →  Vec<StepShell>  (Shell → Face → Loop → Edge)
    │
    ▼
tessellate.rs: tessellate_faces()  →  MeshResult { vertices, indices }
    │
    ▼
mod.rs: build_hierarchical_scene()  →  SceneGraph
```

---

## 二、与 OCC 的对比分析

### 2.1 OCC STEP 处理管线 (参考)

OCC 的 STEP 处理是一个**多阶段、基于 NURBS 的统一表示**的管线：

```
STEP 文件
    │
    ▼
STEPControl_Reader::ReadFile()
    ├── 解析 ISO 10303-21 交换结构
    ├── 模式识别 (AP203 / AP214 / AP242)
    └── 复杂实体 (AND/OR) 展开
    │
    ▼
StepToTopoDS::Transfer()  →  TopoDS_Shape (B-rep 边界表示)
    ├── 所有曲面 → Geom_Surface (NURBS 统一表示)
    ├── 所有曲线 → Geom_Curve (NURBS)
    ├── 拓扑完整性验证 (闭合性、朝向一致性)
    └── PCURVE (参数曲线) 用于面修剪
    │
    ▼
BRepMesh_IncrementalMesh (自适应细分)
    ├── 基于曲率的自适应采样
    ├── 基于 deflection 的误差控制
    ├── 正确的法线计算 (曲面导数)
    └── 边线三角化 (正确轮廓)
    │
    ▼
XCAF 文档 (装配体 + PMI)
    ├── 装配层级树
    ├── 变换传播
    ├── 颜色/样式/图层
    └── PMI 标注 (尺寸、公差、基准)
```

### 2.2 关键差异对比

#### 2.2.1 几何表示

| 维度 | OCC | 当前实现 | 差距 |
|------|-----|----------|------|
| 曲面表示 | 统一 NURBS (`Geom_Surface`) | 按类型分发的解析曲面 + UV 采样 | **大** |
| 参数域 | 每曲面有正确参数域 (u_min, u_max, v_min, v_max) | 固定 [0,1]×[0,1] | **严重** |
| 曲面求值 | `Geom_Surface::Value(u,v)` + 导数 | 各曲面独立采样函数，忽略 placement | **严重** |
| 曲线表示 | 统一 NURBS (`Geom_Curve`) | 按类型分发的采样 | 中 |
| 圆/椭圆 | 正确参数化 (角度参数) | 固定采样点数 | 小 |

#### 2.2.2 拓扑处理

| 维度 | OCC | 当前实现 | 差距 |
|------|-----|----------|------|
| 朝向 (Orientation) | `TopoDS_Orientation` (FORWARD/REVERSED/INTERNAL/EXTERNAL) | `same_sense: bool` | **大** |
| 接缝边 (SEAM EDGE) | 正确处理 (圆柱/圆锥/圆环的 u=0/2π 边) | 未处理 | **严重** |
| PCURVE (参数曲线) | 用于面修剪，存储在 `TopoDS_Edge` 中 | 有提取框架，但未用于细分 | **严重** |
| 拓扑验证 | `BRepCheck_Analyzer` | 无 | 大 |

#### 2.2.3 细分 (Tessellation)

| 维度 | OCC | 当前实现 | 差距 |
|------|-----|----------|------|
| 采样策略 | 基于曲率 + deflection 自适应 | 固定网格 (24×24) | **严重** |
| 法线计算 | 曲面一阶导数 (du, dv) | 无 (平坦着色) | **严重** |
| 修剪 (Trim) | PCURVE 定义边界，精确裁剪 | 忽略修剪，全网格 | **严重** |
| 边线处理 | 边线单独三角化，保证轮廓正确 | 仅面网格，边线可能不精确 | 大 |

#### 2.2.4 装配体

| 维度 | OCC | 当前实现 | 差距 |
|------|-----|----------|------|
| 装配结构 | `XCAF` 文档，完整层级 | 基础 `NEXT_ASSEMBLY_USAGE_OCCURRENCE` | 大 |
| 变换传播 | 递归变换累积 | 基础变换累积 | 中 |
| 样式传播 | 颜色/透明度/图层继承 | 无 | 大 |
| 模式支持 | AP203/AP214/AP242 | 主要 AP203 | 大 |

---

## 三、严重 Bug 详细分析

### Bug 1: `extract_surface_params()` 提取错误参数

**文件**: `surface_tess.rs` 第 84-89 行  
**严重程度**: 🔴 严重

```rust
// 当前代码 (错误)
fn extract_surface_params(params: &StepValue) -> Vec<f32> {
    params.nth_param(1)   // ← 取第 2 个参数，对于 PLANE 是 #position (实体引用)
        .and_then(|v| v.as_list())  // ← 实体引用不是列表，返回 None
        ...
}
```

**STEP 实体参数结构**:
```
PLANE : (name, position)                    ← 只有 2 个参数
CYLINDRICAL_SURFACE : (name, position, radius)  ← 第 3 个参数才是 radius
```

`nth_param(1)` 取到的是 `#position` (一个 `AXIS2_PLACEMENT_3D` 的实体引用)，不是实数列表。

**后果**: 所有曲面采样器收到的 `p: &[f32]` 都是空的，全部退化为默认值 (radius=1.0, ...)。

**修复方案**:
```rust
fn extract_surface_params(surface_id: u64, entities: &EntityIndex) -> (Vec<f32>, Option<Mat4>) {
    let surface = entities.get(&surface_id)?;
    let position_id = surface.params.nth_param(1)?.as_entity_ref()?;
    let placement = resolve_placement(position_id, entities)?;
    
    let numeric_params = match surface.entity_type {
        Plane => vec![],  // 平面没有额外的数值参数
        CylindricalSurface => vec![surface.params.nth_param(2)?.as_real()? as f32],
        ConicalSurface => vec![
            surface.params.nth_param(2)?.as_real()? as f32,  // radius
            surface.params.nth_param(3)?.as_real()? as f32,  // semi_angle
        ],
        // ...
    };
    
    (numeric_params, Some(placement))
}
```

### Bug 2: `sample_plane_uv()` 忽略 Placement

**文件**: `surface_tess.rs` 第 95-97 行  
**严重程度**: 🔴 严重

```rust
// 当前代码 (错误)
fn sample_plane_uv(u: f32, v: f32, _p: &[f32]) -> Vec3 {
    Vec3::new(u, v, 0.0)  // ← 直接返回 (u,v,0)，忽略了平面位置和朝向！
}
```

**STEP 的 `PLANE` 实体**:
```
PLANE (position)
#position : AXIS2_PLACEMENT_3D (origin, axis, ref_direction)
```

`AXIS2_PLACEMENT_3D` 定义了一个局部坐标系：原点 `origin`，法轴 `axis`，参考方向 `ref_direction`。

**正确实现**:
```rust
fn sample_plane_uv(u: f32, v: f32, placement: &Mat4) -> Vec3 {
    let local = Vec3::new(u, v, 0.0);
    let world = *placement * local.extend(1.0);
    Vec3::new(world.x, world.y, world.z)
}
```

**后果**: 所有平面几何的位置和朝向错误。

### Bug 3: 圆柱/圆锥采样器参数理解错误

**文件**: `surface_tess.rs` 第 99-117 行  
**严重程度**: 🔴 严重

```rust
// 当前代码 (错误)
fn sample_cylinder_uv(u: f32, v: f32, p: &[f32]) -> Vec3 {
    let radius = p.first().copied().unwrap_or(1.0);
    let height = p.get(1).copied().unwrap_or(1.0);  // ← CYLINDRICAL_SURFACE 没有 height 参数！
    // ...
}
```

`CYLINDRICAL_SURFACE` 只有 `(name, position, radius)`，是一个**无限曲面**，靠 Face 的边界 (通过 PCURVE) 修剪。

**后果**: 
1. `height` 参数无意义
2. 没有考虑参数域 (`u` 应该是角度，不是 [0,1])
3. 没有应用 placement 变换

### Bug 4: 参数域错误

**所有采样器**: 假设 `u, v ∈ [0, 1]`

**实际情况** (ISO 10303-42):
- `CYLINDRICAL_SURFACE`: `u ∈ [0, 2π)`, `v ∈ (-∞, +∞)`
- `CONICAL_SURFACE`: `u ∈ [0, 2π)`, `v ∈ (-∞, +∞)`
- `SPHERICAL_SURFACE`: `u ∈ [0, 2π)`, `v ∈ [0, π]`
- `TOROIDAL_SURFACE`: `u ∈ [0, 2π)`, `v ∈ [0, 2π)`

**后果**: 几何比例和朝向错误。

---

## 四、推进计划

### 阶段 1: 修复严重 Bug (预计 1-2 周)

#### 任务 1.1: 修复参数提取和 Placement 处理

**目标**: 正确提取曲面参数，将 Placement 应用于采样点。

**步骤**:
1. 重写 `extract_surface_params()` → `resolve_surface_params(surface_id, entities)`
   - 对每种曲面类型单独解析参数
   - 同时返回数值参数和 placement 变换矩阵
2. 修改所有 `sample_*_uv()` 函数签名，接受 `placement: &Mat4`
3. 修正参数域：
   - 圆柱/圆锥: `u ∈ [0, 2π)`
   - 球面: `u ∈ [0, 2π)`, `v ∈ [0, π]`
   - 圆环: `u ∈ [0, 2π)`, `v ∈ [0, 2π)`

**验收标准**:
- 使用简单 STEP 文件测试 (一个平面、一个圆柱)
- 几何位置和朝向正确

#### 任务 1.2: 修复采样器参数理解

**目标**: 正确理解 STEP 曲面实体的参数结构。

**步骤**:
1. 圆柱曲面: 移除 `height` 参数 (它是无限曲面)
2. 圆锥曲面: 使用 `radius` 和 `semi_angle`
3. 所有曲面: 采样时使用正确的参数域

**验收标准**:
- 圆柱/圆锥几何形状正确
- 参数域正确映射

#### 任务 1.3: 添加基础法线计算

**目标**: 为细分结果计算正确的法线。

**步骤**:
1. 使用曲面的一阶导数计算法线:
   - `normal = ∂S/∂u × ∂S/∂v` (归一化)
2. 在 `MeshResult` 中添加 `normals: Vec<Vec3>` 字段
3. 修改细分管线，输出法线

**验收标准**:
- 渲染时平滑着色正确
- 法线方向一致 (朝向外部)

---

### 阶段 2: 引入 NURBS 统一表示 (预计 2-3 周)

#### 任务 2.1: 实现 NURBS 曲面求值器

**目标**: 将所有 STEP 曲面转换为 NURBS 表示，统一求值接口。

**设计**:
```rust
struct NurbsSurface {
    degree_u: usize,
    degree_v: usize,
    control_points: Vec<Vec<[f32; 4]>>,  // 齐次坐标
    knots_u: Vec<f32>,
    knots_v: Vec<f32>,
}

impl NurbsSurface {
    fn evaluate(&self, u: f32, v: f32) -> Vec3 {
        // NURBS 曲面求值
    }
    
    fn derivative(&self, u: f32, v: f32) -> (Vec3, Vec3) {
        // 返回 ∂S/∂u 和 ∂S/∂v，用于法线计算
    }
}
```

**转换**:
- `PLANE` → 双线性 NURBS (degree 1×1)
- `CYLINDRICAL_SURFACE` → 裁剪的 NURBS (通过圆 NURBS 旋转/拉伸)
- `CIRCLE` (用于圆锥/球面/圆环) → 有理 NURBS (degree 2)

**验收标准**:
- 所有解析曲面正确转换为 NURBS
- 求值时误差 < 1e-6

#### 任务 2.2: 实现 PCURVE 修剪

**目标**: 使用 PCURVE (参数曲线) 正确修剪面。

**设计**:
```rust
struct FaceTrim {
    outer_loop: TrimLoop,
    inner_loops: Vec<TrimLoop>,  // 孔洞
}

struct TrimLoop {
    edges: Vec<TrimSegment>,
}

enum TrimSegment {
    Pcurve { curve_id: u64, t_min: f32, t_max: f32 },
    Curve3D { curve_id: u64 },
}
```

**步骤**:
1. 从 `FACE_OUTER_BOUND` / `FACE_BOUND` 提取 PCURVE
2. 将 PCURVE 采样为 UV 空间的多边形
3. 使用多边形裁剪 UV 网格

**验收标准**:
- 带孔洞的面正确修剪
- 修剪边界与原始曲线一致

---

### 阶段 3: 自适应细分 (预计 2-3 周)

#### 任务 3.1: 基于曲率的采样

**目标**: 根据曲面曲率自适应采样，平衡质量和性能。

**算法**:
1. 计算曲面上每个点的高斯曲率 `K`
2. 高曲率区域: 增加采样密度
3. 低曲率区域: 减少采样密度
4. 使用误差度量 (deflection) 控制最大偏差

**参考 OCC**: `BRepMesh_IncrementalMesh` 使用类似策略。

**验收标准**:
- 高曲率区域 (边角、细小特征) 采样密度高
- 平面区域采样密度低
- 最大偏差 < 用户指定 tolerance

#### 任务 3.2: 边线三角化

**目标**: 单独三角化边线，保证轮廓正确。

**步骤**:
1. 对每个 `Edge`，采样 3D 曲线
2. 将边线采样点投影到相邻面的网格
3. 确保边线在网格中有对应的边

**验收标准**:
- 渲染时轮廓正确
- 相邻面的边线对齐 (水密)

---

### 阶段 4: 装配体和 PMI (预计 2-3 周)

#### 任务 4.1: 完善装配体支持

**目标**: 支持 AP203/AP214/AP242 装配体结构。

**步骤**:
1. 解析 `SHAPE_DEFINITION_REPRESENTATION` (AP242)
2. 解析 `NEXT_ASSEMBLY_USAGE_OCCURRENCE` (AP203)
3. 递归累积变换
4. 处理 `STYLED_ITEM` 的颜色/样式

**验收标准**:
- 多层级装配体正确加载
- 变换正确传播
- 颜色/样式正确显示

#### 任务 4.2: PMI 标注渲染

**目标**: 显示尺寸、公差、基准标注。

**步骤**:
1. 解析 PMI 实体 (`DIMENSIONAL_SIZE`, `GEOMETRIC_TOLERANCE`, ...)
2. 将 PMI 附加到几何 (通过 `STYLED_ITEM` 或 `ANNOTATION_OCCURRENCE`)
3. 在 3D 视图中渲染 PMI (文字 + 箭头 + 引线)

**验收标准**:
- PMI 标注正确解析
- PMI 标注正确渲染
- PMI 标注随几何正确变换

---

### 阶段 5: 鲁棒性和性能 (持续进行)

#### 任务 5.1: 错误处理和恢复

**目标**: 解析器能从错误中恢复，继续解析剩余实体。

**步骤**:
1. 在 `parse_entity()` 中添加错误恢复
2. 跳过无法解析的实体，继续解析下一个
3. 记录警告 (不支持的实体类型、参数错误等)

#### 任务 5.2: 大模型支持

**目标**: 支持大型装配体 (100MB+ STEP 文件)。

**步骤**:
1. 流式解析 (不一次性加载整个文件到内存)
2. 渐进式加载 (先加载结构，再加载几何)
3. LOD (Level of Detail) 支持

#### 任务 5.3: 测试覆盖

**目标**: 为每个模块编写单元测试。

**步骤**:
1. `parser.rs`: 测试各种 STEP 实体解析
2. `topology.rs`: 测试 B-rep 遍历
3. `geom.rs`: 测试曲线/曲面求值
4. `tessellate.rs`: 测试细分结果 (水密性、法线方向)
5. 集成测试: 使用真实 STEP 文件

---

## 五、优先级建议

| 优先级 | 任务 | 原因 |
|--------|------|------|
| P0 | 任务 1.1-1.3 | 当前实现几何不正确，无法使用 |
| P1 | 任务 2.1-2.2 | NURBS 统一表示是正确细分的基础 |
| P2 | 任务 3.1-3.2 | 自适应细分提升质量 |
| P3 | 任务 4.1-4.2 | 装配体和 PMI 是工业应用必需 |
| P4 | 任务 5.1-5.3 | 鲁棒性和性能提升用户体验 |

---

## 六、参考资料

1. **ISO 10303-21**: STEP 交换文件格式
2. **ISO 10303-42**: STEP 集成资源：几何与拓扑表示
3. **Open CASCADE Documentation**: https://dev.opencascade.org/doc/overview/html/index.html
4. **STEP AP242**: 基于模型的 3D 工程 (包含 PMI)
5. **NURBS 求值**: The NURBS Book (Les Piegl, Wayne Tiller)

---

## 七、附录: 测试文件建议

### 7.1 基础几何测试

1. `simple_plane.stp`: 一个平面
2. `simple_cylinder.stp`: 一个圆柱
3. `simple_cube.stp`: 一个立方体 (6 个平面面)
4. `simple_sphere.stp`: 一个球面

### 7.2 复杂几何测试

1. `threaded_fastener.stp`: 带螺纹的紧固件 (测试高精度几何)
2. `curved_surface.stp`: 自由曲面 (测试 NURBS)
3. `hollow_cylinder.stp`: 带孔洞的圆柱 (测试修剪)

### 7.3 装配体测试

1. `simple_assembly.stp`: 2-3 个零件的装配体
2. `multi_level_assembly.stp`: 多层级装配体

### 7.4 PMI 测试

1. `dimensioned_part.stp`: 带尺寸标注的零件
2. `toleranced_part.stp`: 带公差标注的零件
