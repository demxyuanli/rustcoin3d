# Large File Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `render_action.rs` (2034行) 和 `render_passes.rs` (1323行) 重构为可维护的模块化结构。

**Architecture:**
- `render_action.rs` 职责混杂：形状缓存、DrawCall聚合、光照打包、遍历缓存
- 建议按职责拆分为: shape_cache.rs (形状缓存) + action.rs (保留 DrawCall/RenderCollector)
- `render_passes.rs` 结构较好，主要问题是大函数 `execute_passes` (~800行) 和 `submit_meshlet_cull`

**Tech Stack:** Rust, wgpu, slotmap

---

## Phase 1: render_action.rs 拆分 (2034行 → ~400行/文件)

### Task 1: 提取 Shape Cache 模块

**Files:**
- Create: `crates/rc3d-render/src/shape_cache.rs`
- Modify: `crates/rc3d-render/src/render_action.rs`

**Step 1: 创建 shape_cache.rs 并迁移代码**

提取内容:
- `ShapeKey` enum (line 46)
- `CachedShapeData` type alias (line 65)
- `PackedLights` type alias (line 73)
- `MAX_EDGE_POSITIONS`, `MESHLET_TRIANGLE_THRESHOLD` 常量 (line 82-83)
- `feature_crease_angle()`, `set_feature_crease_angle()` (line 91-98)
- `clamp_edge_positions()` helper (line 101)
- shape 生成函数: `cache_cube`, `cache_sphere`, `cache_cone`, `cache_cylinder`, `cache_indexed_face_set`

**Step 2: 更新 render_action.rs 引用**

```rust
mod shape_cache;
pub use shape_cache::*;
```

**Step 3: 运行测试验证**

```bash
cargo test -p rc3d-render
cargo build -p rc3d-render
```

### Task 2: 提取 Light Packing 模块

**Files:**
- Create: `crates/rc3d-render/src/light_packing.rs`
- Modify: `crates/rc3d-render/src/render_action.rs`

**Step 1: 迁移 light packing 代码**

提取内容:
- `hash_light_params()` function (line 110)

**Step 2: 更新 render_action.rs**

**Step 3: 运行测试验证**

### Task 3: 提取 Traversal/Cache 模块

**Files:**
- Create: `crates/rc3d-render/src/traversal.rs`
- Modify: `crates/rc3d-render/src/render_action.rs`

**Step 1: 迁移 traversal 代码**

提取内容:
- `TraversalChunk` struct (line 1691)
- `merge_chunk_into_cache()` (line 1700)
- `invalidate_cache_for_subtree()` (line 1719)
- `traverse_into_cache()` (line 1727)
- `count_all_nodes()`, `count_subtree()` helpers (line 1770-1774)
- `convert_collector_to_cache_textures()` (line 1787)
- `convert_collector_to_cache()` (line 1803)
- `populate_cache_from_draw_calls()` (line 1906)
- `tex_id_to_arcstr()` helper (line 1966)
- `cache_to_draw_calls()` (line 1978)

**Step 2: 更新 render_action.rs**

**Step 3: 运行测试验证**

---

## Phase 2: render_passes.rs 拆分 (1323行)

### Task 4: 提取 Meshlet Cull 模块

**Files:**
- Modify: `crates/rc3d-render/src/render_passes.rs`
- Create: `crates/rc3d-render/src/render_passes/meshlet_cull.rs`

**Step 1: 迁移 submit_meshlet_cull**

将 `submit_meshlet_cull` 函数 (line 89-146) 移动到新模块

**Step 2: 更新 render_passes.rs**

```rust
mod meshlet_cull;
use meshlet_cull::submit_meshlet_cull;
```

**Step 3: 运行测试验证**

### Task 5: 提取 Viewport 渲染逻辑

**Files:**
- Modify: `crates/rc3d-render/src/render_passes.rs`
- Create: `crates/rc3d-render/src/render_passes/viewport_render.rs`

**Step 1: 分析 execute_passes 函数结构**

`execute_passes` (~800行) 包含:
- Swapchain/OffscreenSurface 处理
- 多个渲染 Pass 调用
- HUD 渲染

建议按 pass 类型拆分:
- `solid.rs` - 不透明物体渲染
- `selection.rs` - 选择高亮渲染
- `edge.rs` - 边缘渲染
- `post.rs` - 后处理
- `viewport_render.rs` - 主渲染流程编排

**Step 2: 渐进式提取**

先提取辅助函数和子 pass，保持 `execute_passes` 结构不变

**Step 3: 运行测试验证**

---

## Phase 3: 最终验证

### Task 6: 完整测试验证

**Step 1: 运行完整测试套件**

```bash
cargo test --workspace
cargo clippy -p rc3d-render
cargo build -p rc3d-render --examples
```

**Step 2: 检查代码行数**

```bash
wc -l crates/rc3d-render/src/*.rs
```

目标:
- render_action.rs: 2034 → ~300行
- render_passes.rs: 1323 → ~400行
- 新模块: shape_cache.rs (~400行), light_packing.rs (~50行), traversal.rs (~500行), meshlet_cull.rs (~100行)

---

## Errors Encountered

| Error | Attempt | Resolution |
| --- | --- | --- |
| ShapeKey 字段名不匹配 | render_action 使用 `r`, `h` 而非 `radius`, `height` | 修正 shape_cache.rs 中字段名 |
| `set_feature_crease_angle` 私有 | 使用 `use crate::shape_cache::*` 无法导出 | 使用 `pub use` 显式重导出 |
| `submit_meshlet_cull` 可见性冲突 | `pub fn` vs `PassContext` `pub(crate)` | 改为 `pub(crate) fn` |

## Status

- [x] Phase 1: render_action.rs 拆分
  - [x] Task 1: Shape Cache 模块 (90 lines) - ✅ COMPLETED
  - [x] Task 2: Light Packing 模块 (39 lines) - ✅ COMPLETED
  - [x] Task 3: Traversal 模块 (358 lines) - ✅ COMPLETED
- [x] Phase 2: render_passes.rs 拆分
  - [x] Task 4: Meshlet Cull 模块 (68 lines) - ✅ COMPLETED
  - [x] Task 5: Viewport 渲染逻辑 - ✅ COMPLETED
    - [x] Screen-space edge detection (87 lines) - ✅ COMPLETED
    - [x] HUD overlay 提取 (68 lines) - ✅ COMPLETED
    - [x] HDR 后处理管线 (231 lines) - ✅ COMPLETED
    - [x] FrameStats 构建 + 日志 - ✅ COMPLETED
    - [x] Task I: Surface acquire 共享逻辑 (pass_shared::acquire_surface) - ✅ COMPLETED
- [x] Phase 3: 最终验证
  - [x] Task 6: 完整测试套件验证 - ✅ COMPLETED

## 重构成果

| 文件 | 原始行数 | 当前行数 | 变化 |
|------|----------|----------|------|
| render_action.rs | 2034 | 1574 | -460 (-23%) |
| render_passes.rs | 1323 | ~990 | ~-333 (-25%) |
| shape_cache.rs | - | 90 | +90 (新增) |
| light_packing.rs | - | 84 | +84 (新增, 含 collect_lights) |
| meshlet_cull.rs | - | 68 | +68 (新增) |
| traversal.rs | - | 358 | +358 (新增) |
| ss_edge.rs | - | 87 | +87 (新增) |
| pass_hud.rs | - | 68 | +68 (新增) |
| pass_post.rs | 216 | 447 | +231 (encode_post_processing) |
| pass_shared.rs | - | 52 | +52 (新增, acquire_surface) |

**总计减少**: render_action.rs + render_passes.rs 减少约 793 行 (-30%)