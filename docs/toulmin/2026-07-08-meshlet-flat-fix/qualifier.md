# Unified Qualifier — 2026-07-08

**Target**: rustcoin3d meshlet渲染子系统的深度约定架构

## Design Scope Statement

> Under **forward-Z or reverse-Z depth convention** (explicitly selected per pipeline), with **wgpu backends that correctly implement CompareFunction per WebGPU spec**, and **Depth32Float format throughout**, the meshlet rendering pipeline correctly handles meshlet/non-meshlet geometry across all DisplayMode variants (including Flat/FlatWithEdge).
>
> Based on: **internal code review** (verify equivalent), **external spec validation** (audit), and **failure backtracking** (premortem).
>
> The dual-convention architecture **FAILS** when a new rendering feature is added with only one depth convention tested.
>
> The architecture **DEGRADES** when GPU drivers diverge in floating-point denormal handling under `LessEqual`/`GreaterEqual` comparisons.

## Confidence: **MEDIUM**

## Hard Boundaries (design invalid if violated)

| # | Condition | Effect | Source | Severity |
|---|-----------|--------|--------|----------|
| H1 | 向前渲染pass中 depth compare function 与投影矩阵的深度约定不一致 | 深度排序完全错误，近/远物体关系颠倒 | audit F2 | fatal |
| H2 | HZB pyramid 的 mip 链在 forward-Z 路径中产生全零值 | 所有物体被遮挡剔除，画面全黑 | premortem P2 | fatal |
| H3 | 新添加的 pipeline 在 forward-Z 路径未被测试 | 行为和视觉输出在 forward-Z 模式下未定义 | premortem P1 | fatal |

## Soft Boundaries (design degrades if violated)

| # | Condition | Degradation | Source | Severity |
|---|-----------|-------------|--------|----------|
| S1 | GPU驱动将 denormal 深度值 flush 为 0 | meshlet Flat 模式下几何体间歇性不可见 | premortem P3 + audit F1 | severe |
| S2 | prepass 和 main pass 渲染的几何体子集不完全一致 | LoadOp::Load 暴露 prepass 残留像素 | audit F3 | severe |
| S3 | 深度比较函数通过裸枚举值传递（无编译期验证） | 复制粘贴错误编译通过但运行时静默错误 | premortem P1 | manageable |

## Monitor Triggers

| # | Signal | Why | Source |
|---|--------|-----|--------|
| M1 | 新增 pipeline 时 forward-Z 变体的测试覆盖率 < 1 | 双约定架构依赖对称测试 | premortem P1 |
| M2 | HZB forward-Z chain 的 mip3+ 层级值域异常 | 数值边界不对称 | premortem P2 |
| M3 | 新的 wgpu 后端被添加但 CI 未覆盖 | GPU 厂商间浮点行为差异 | premortem P3 |

## Open Risks (accepted, not mitigated)

| # | Risk | Acceptance rationale | Source |
|---|------|---------------------|--------|
| R1 | 双深度约定使 pipeline 数量翻倍，维护成本翻倍 | forward-Z 废弃路线图存在但未排期 | premortem P1 |
| R2 | forward-Z HZB 链在极端深度分布下有数值不稳定风险 | forward-Z 主要用于阴影贴图，深度范围受限 | premortem P2 |

## Evidence Sources

| Tool | Status | Key contribution |
|------|--------|-----------------|
| Audit | ⚠️ NARROW | F1: LessEqual方案正确；F2: reverse-Z是行业标准；F3: LoadOp::Load 需prepass覆盖完整 |
| Pre-mortem | N/A | P1: pipeline翻倍测试盲区；P2: HZB双链数值不对称；P3: GPU浮点分歧 |

## Revision History

| Date | Change | Trigger |
|------|--------|---------|
| 2026-07-08 | Initial synthesis | Audit + Premortem applied |
