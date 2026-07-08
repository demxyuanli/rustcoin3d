# Gate 1 — Direction Convergence — 2026-07-08

## Decision
按模块直接删除——从叶子模块向核心层逐层切除 geometry engine 内部实现，保留拓扑数据结构和 BREP 格式转换接口。

### Claim
删除 rc3d-shape 中 ~50,000 行几何求值/修复/布尔/网格化代码、rc3d-io 中 ~23,000 行 STEP/IGES 解析代码、2 个独立几何 crate，仅保留 ~4,500 行拓扑+转换层。可视化引擎 18 个 crate 不动。

### Ground
- `rc3d-shape` 53,450 行中，`heal/`（31文件）、`bool/`（17文件）、`mesh/`（30+文件）占 ~1,150,000 行且全部互不依赖可视化层（`rc3d-render` 不依赖 `rc3d-shape`）
- 唯一跨 crate 依赖链：`rc3d-io → rc3d-shape`，仅用于 BRepStore + BREP writer
- `store.rs` 仅需要 `normalize_edge_curve_to_vertices` 和 `curve_param_range_from_vertices` 两个函数
- 删除路径清晰：leaf modules first → 逐层验证编译

### Warrant
模块间依赖关系已通过代码审查确认（`docs/geometry-engine-review.md`）。删除顺序从叶子模块（不被其他模块引用）开始，每步验证编译，保证不会出现不可编译的中间状态。

### Backing
- 所有删除的模块（heal、bool、mesh、STEP）与可视化引擎完全解耦
- BREP writer 仅依赖 CurveGeom/SurfaceGeom 枚举类型（类型标签）和 Curve2d 枚举——不依赖求值逻辑
- 之前 4 次 BREP 格式修复 commit 证明了拓扑转换层的完整性

### Rebuttal
- **逐步内联而非直接删除**：内联会产生大量中间代码，违反"最小保留"原则。直接删除更干净。
- **保留为 optional feature**：增加 Cargo feature 复杂性。这些模块无下游用户，feature flag 是 YAGNI。
- **先抽象接口再删实现**：当前无接口抽象需求。BRepStore → BREP 是唯一转换方向。

### Qualifier
- 此决策在"不再继续实现几何解析"的产品约束下永久有效
- 如需恢复 STEP 导入能力，可通过二进制 BREP 加载器（`brep_binary.rs`）桥接外部 OCCT
- 边界：仅限 geometry engine；可视化层不受影响

## Verdict: PASSED
