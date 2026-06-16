# rustcoin3d 几何核心整改总结报告

**审查范围**：几何核心（`rc3d-core` / `rc3d-shape` / `rc3d-nurbs`）、几何拓扑、STEP 解析（`rc3d-io`）、可视化离散、STL 导出  
**审查方式**：code-simplifier 架构审查 + 三域专项代码审查 + 关键路径抽查验证  
**报告日期**：2026-06-10

---

## 1. 执行摘要

rustcoin3d 已具备较完整的 CAD 几何管线：**Part21 解析 → BRep 构建 → Heal → 自适应细分 → SceneGraph / STL 导出**。但在 OCC 关键语义实现、默认导入策略、细分缓存与 fallback 策略等方面存在**系统性缺陷**，可能导致：

- 几何**静默错误**（Preview 模式 fallback、裁剪面未生效、孔洞被填实）
- 网格**质量不一致**（视口 vs STL 导出、缓存 stale、PCurve 方向错误）
- **生产崩溃**（高阶 B-spline panic、CDT 退化输入 panic）

**结论**：功能广度足够，但**正确性保障不足**；策略抽象层（Tier / HealPolicy / FaceStrategyChain）与生产路径**严重脱节**。

**建议优先修复（ROI 最高）**：

1. PCurve `same_sense` 全链路统一
2. 裁剪面（RECTANGULAR_TRIMMED / CURVE_BOUNDED）domain 约束
3. 细分缓存完整性 + `surface_fill_3d` 禁止填孔

---

## 2. 管线架构

```
STEP 文件 (Part21/XML)
    ↓  rc3d-io/step/parser.rs
EntityIndex + 校验 (Preview 模式 warn-only)
    ↓  topology/collect.rs
StepFace/StepEdge 中间拓扑
    ↓  caf_transfer + brep/build/*
BRepStore (rc3d-shape)
    ↓  SameParameter + Heal (3 处独立实现)
    ↓  emit_plan → shell_pipeline/face_chunk.rs
MeshResult (Coin3D 索引格式)
    ↓                    ↓
SceneGraph/GPU         STL 导出 (mesh_export)
```

### 架构矛盾

| 已设计 | 生产现状 |
|--------|----------|
| `TessellationTier` / `HealPolicy` | 未贯通，`emit_plan_options_from_step` 手写逻辑 |
| `FaceStrategyChain` + `FallbackAllowlist` | `face_chunk.rs` 1100 行单体，allowlist 未接线 |
| SameParameter 统一阶段 | IO / heal / mesh 三处独立实现 |
| `shape_corpus` 测试 | 绕过 CAF/void/emit_plan，与用户路径不等价 |

---

## 3. 问题清单（按优先级）

### P0 — 必须优先整改

| ID | 问题 | 关键位置 | 影响 |
|----|------|----------|------|
| P0-1 | **PCurve `same_sense` 被忽略** | `mesh/edge_disc.rs`, `face_cdt.rs`, `io/brep/same_parameter.rs:71` | UV 环错误、CDT 失败、边界裂缝、法线翻转 |
| P0-2 | **默认 Preview 静默降级几何** | `import_options.rs`（default = Preview） | 未知曲面→平面、未知曲线→直线，import 仍成功 |
| P0-3 | **裁剪面未生效** | `brep/build/surface.rs:84–107` | Offset/trimmed 面在完整域细分，孔/offset 错误 |
| P0-4 | **`surface_fill_3d` 丢弃内环** | `fill_surface.rs`（`inners: vec![]`），`face_chunk.rs` 仍调用 | 带孔面被填实，STL 体积错误 |
| P0-5 | **细分缓存键不完整 + 不失效** | `emit_plan.rs:113–122`，`tessellation.rs` | 切换精度/编辑后仍用旧网格 |
| P0-6 | **SameParameter 导出路径跳过** | `mesh_export.rs:223`，`step/mod.rs:257` | STL 边界质量劣于视口 |
| P0-7 | **B-spline 阶数处理不一致** | `bspline.rs` panic vs `curve_eval.rs` 返回零 | debug 崩溃 / release 静默塌陷 |
| P0-8 | **CDT 退化输入 panic** | `delaunay2d/insertion.rs:69` | 单面失败导致整 job 中止 |

### P1 — 高影响

#### STEP / BRep

| ID | 问题 | 影响 |
|----|------|------|
| P1-1 | AP242 镶嵌几何未接入（`tessellated.rs` 无 caller） | 纯镶嵌 STEP 无几何 |
| P1-2 | ORIENTED_SHELL / SBSM 未收集 | 部分 AP203/AP214 空几何 |
| P1-3 | 无 LENGTH_UNIT 缩放 | 英寸/微米文件尺度错误 |
| P1-4 | FACE_BOUND 方向未应用 | 内环 winding 错误 |
| P1-5 | EDGE_CURVE same_sense 未传播 | 曲线参数反向 |
| P1-6 | 失败 shell/edge 静默跳过 | 多体文件丢零件 |
| P1-7 | 大文件全内存模型 | 工业 STEP OOM |
| P1-8 | BREP_WITH_VOIDS 处理不完整 | 孔洞/体积错误 |
| P1-9 | shell `closed` 启发式（`faces.len() >= 4`） | 开 sheet 误判为 solid |

#### 拓扑 / Heal

| ID | 问题 | 影响 |
|----|------|------|
| P1-10 | `check_wire_closed` 忽略边方向 | 假阳性 open wire |
| P1-11 | 边 dedup 中点启发式合并 | 共顶点不同曲线被合并 |
| P1-12 | `edge_to_faces` 可含重复 | shell 邻接错误 |
| P1-13 | SameParameter 三套实现 | tolerance/snap/reparam 互相打架 |

#### 细分 / 可视化

| ID | 问题 | 影响 |
|----|------|------|
| P1-14 | BSpline `fix_tri_winding` 跳过曲面法线 | 自由曲面局部翻面 |
| P1-15 | 边离散 `max_points=128` 硬上限 | 大模型 T 型缝 |
| P1-16 | void solid per-face 分割不可靠 | per-face STL 范围错位 |
| P1-17 | IndexedFaceSet 扇形剖分 + 弱 GPU 缓存键 | 凹面自交、stale cache |
| P1-18 | Offset 曲面奇点无处理 | apex/pole NaN |

#### STL 导出

| ID | 问题 | 影响 |
|----|------|------|
| P1-19 | OOB 索引静默跳过 | 丢三角形无报错 |
| P1-20 | 退化三角形法线 fallback `Vec3::Z` | CAM 法线错误 |
| P1-21 | 非均匀缩放法线变换错误 | 装配实例法线错 |
| P1-22 | per-face STL 双 import | 性能浪费、结果可能不一致 |

### P2 — 架构与可维护性

- **策略层未贯通**：`TessellationTier` / `HealPolicy` / `FaceStrategyChain` 定义但未 enforce
- **NURBS 三栈重复**：`rc3d-nurbs`、`rc3d-shape/nurbs.rs`(1615行)、`geom/*_eval.rs`(~4400行)
- **双 mesh 表示无 adapter**：`MeshResult`(i32/-1) vs `TriangleMesh`(u32)
- **EdgeKey 同名异构**：shape vs mesh crate
- **12 个文件超 800 行**：最高 `face_chunk.rs`(1102)、`nurbs.rs`(1615)
- **测试空白**：`rc3d-nurbs`、`rc3d-mesh` 零集成测试
- **生产 panic 风险**：`pipeline/mod.rs` `StageOutcome::unwrap`
- **调试残留**：`emit_plan.rs`、`step/mod.rs` 大量 `eprintln!`

---

## 4. 根因分析

| 根因类别 | 表现 | 涉及模块 |
|----------|------|----------|
| **OCC 语义不完整** | same_sense、裁剪域、bound orientation、SameParameter | shape/mesh, io/brep |
| **静默降级策略** | Preview default + fallback + 失败 continue | io/import_options, brep/build |
| **策略与生产脱节** | tier/cache/fallback/void 抽象未 enforce | shape/emit_plan, mesh/config |
| **实现分裂** | SameParameter×3、NURBS×3、测试路径≠用户路径 | 跨 crate |
| **错误处理不足** | panic/零坐标/静默 skip 替代 Result | shape/mesh, core/bspline |

---

## 5. 整改路线图

### 阶段 1 — 正确性止血（1–2 周）

| 序号 | 任务 | 关联 ID |
|------|------|---------|
| 1 | 统一 PCurve `effective_pcurve_t(t, same_sense)` | P0-1, P1-13 |
| 2 | 裁剪面 UV domain 约束 | P0-3 |
| 3 | 接入 FallbackAllowlist，带孔面禁 surface_fill_3d | P0-4 |
| 4 | 扩展 config_hash + BRep 变更 invalidate | P0-5 |
| 5 | SameParameter 与 skip_visualization 解耦 | P0-6 |
| 6 | B-spline degree 统一 Result 策略 | P0-7 |
| 7 | CDT 传播 Result，per-face fail | P0-8 |

### 阶段 2 — STEP 完整性（2–3 周）

| 序号 | 任务 | 关联 ID |
|------|------|---------|
| 8 | 导出路径 Strict / ExportQuality profile | P0-2 |
| 9 | 接入 tessellated shell | P1-1 |
| 10 | 扩展 ORIENTED_SHELL / SBSM 收集 | P1-2 |
| 11 | LENGTH_UNIT 解析与缩放 | P1-3 |
| 12 | FACE_BOUND / EDGE_CURVE 方向传播 | P1-4, P1-5 |
| 13 | Strict 模式 fail-fast（shell/edge） | P1-6 |
| 14 | BREP_WITH_VOIDS 闭环 | P1-8 |

### 阶段 3 — 架构收敛（3–4 周）

| 序号 | 任务 |
|------|------|
| 15 | SameParameterPipeline 单入口 |
| 16 | TierContext 贯通 tier/heal/mesh/fallback |
| 17 | FaceStrategyChain 接入或删除 |
| 18 | MeshAdapter + STL 质量门控 |
| 19 | 拆分 face_chunk / pcurve / eval 模块 |
| 20 | full-import 测试 corpus 对齐 |

---

## 6. 验证方案

| 测试用例 | 验证点 |
|----------|--------|
| `steps/OffsetPlaneHoleEdge.step` | 裁剪域、PCurve same_sense、孔保留 |
| `steps/comp/occ-HoledPlate.brep` | inner wire CDT、禁止填孔 |
| degree > 16 B-spline STEP | 无 panic、非零求值 |
| 英寸单位 STEP | 尺度/tolerance 一致 |
| Preview vs Strict 同文件 | fallback 计数可见、几何差异可观测 |
| tier 切换 | 缓存 miss、网格变化 |
| export STL vs 视口 | SameParameter 后边界一致 |
| 共线 UV 边界面 | CDT fail 不 panic |
| BREP_WITH_VOIDS | 体积/孔洞 golden mesh |

```bash
rtk cargo test -p rc3d-io export_step_stl
rtk cargo test -p rc3d-io --test shape_corpus
rtk cargo test -p rc3d-io --test brep_verify_loop
rtk cargo check -p rc3d-shape -p rc3d-io -p rc3d-nurbs -p rc3d-mesh
```

---

## 7. 风险矩阵

| 风险 | 概率 | 严重度 | 典型场景 |
|------|------|--------|----------|
| 静默几何错误 | 高 | 高 | Preview 导入 + STL 导出 |
| 孔洞消失 | 中 | 高 | CDT 失败 → surface_fill_3d |
| 网格 stale | 中 | 中 | 编辑后重复 tessellate |
| 生产 panic | 低 | 高 | 高阶 B-spline / 退化 CDT |
| 尺度错误 | 中 | 高 | 非 SI 单位 STEP |
| 内存 OOM | 中 | 中 | 大型工业装配 |

---

## 8. 总结

| 维度 | 评价 |
|------|------|
| **功能覆盖** | 较好 — Part21 流式、BRep heal、CDT 细分、装配、STL 均有实现 |
| **OCC 语义 fidelity** | 不足 — same_sense、裁剪、SameParameter、void 等关键语义缺失或分裂 |
| **错误可见性** | 差 — Preview 默认 + 静默 fallback/skip |
| **架构一致性** | 差 — 策略层与生产层脱节，测试路径不等价 |
| **可维护性** | 中 — 多个 1000+ 行单体，NURBS/mesh 三栈重复 |

**整改核心目标**：从"能跑通"升级到"可信赖"——优先消除静默错误，再收敛架构。


