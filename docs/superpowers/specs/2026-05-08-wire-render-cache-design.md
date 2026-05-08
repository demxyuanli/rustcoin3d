# 渲染循环接线 FlatDrawCache — 设计文档

> 将已构建的 FlatDrawCache / dirty flags / 增量遍历接入 render_draw_calls_core，消除每帧全图遍历 + BVH 重建

## 架构

```
每帧数据流（新）：
  ① texture_streamer.poll_completed()
  ② traverse_into_cache(graph, &mut self.draw_cache, &self.texture_table, hidden_nodes)
     ├─ 无 dirty → 复用上一帧 cache
     ├─ 少量 dirty → 仅重遍历脏子树（并行：rayon）
     └─ >50% dirty → 全量重建
  ③ cache_to_draw_calls(&draw_cache, &texture_table) → Vec<DrawCall>
  ④ render_draw_calls_core(&draw_calls, scene, ...)  ← 现有路径不变
  ⑤ clear_all_dirty_flags(graph)
```

核心原则：**接入缓存层，不改渲染路径。** Adapter 做新旧转换。

## 改动文件

| 文件 | 改动 | 行数 |
|------|------|------|
| `renderer_render.rs` | `render_draw_calls_core` 入口：用 draw_cache 取代外部 collector 输入 | +30/-10 |
| `render_action.rs` | 新增 `cache_to_draw_calls()` adapter | +60 |
| `event_handler.rs` | 移除 `collector.traverse()` 调用，清理 collector 引用 | +5/-15 |

## Adapter 函数

```rust
pub fn cache_to_draw_calls(
    cache: &FlatDrawCache,
    texture_table: &TexturePathTable,
) -> Vec<DrawCall>
```

从 `GpuDrawData` + `CachedDrawMetadata` 重建旧 `DrawCall`。字段映射：
- model_matrix ← GpuDrawData.model_matrix
- material 字段 ← CachedDrawMetadata.material_params
- texture paths ← texture_table.get(tex_id)
- aabb ← 暂不设置（BVH 用旧路径）

## 遍历接入点

`render_draw_calls_core` 的 `draw_calls: &[DrawCall]` 参数改为内部生成：

```rust
fn render_draw_calls_core<'p>(
    &'p mut self,
    scene: &SceneGraph,  // 不再需要外部传入 draw_calls
    ...
) -> FrameStats {
    self.texture_streamer.poll_completed(&self.device, &self.queue);
    
    // 收集场景文本
    collect_text_to_hud(self, scene);
    
    // 增量遍历 → FlatDrawCache
    traverse_into_cache(scene, &mut self.draw_cache, &self.texture_table, &self.frame.hidden_nodes);
    
    // Adapter 转换
    let draw_calls = cache_to_draw_calls(&self.draw_cache, &self.texture_table);
    
    if draw_calls.is_empty() { return FrameStats::default(); }
    
    // ... 现有渲染逻辑不变 ...
    
    // 帧末清理
    clear_all_dirty_flags(scene); // 需要 &mut SceneGraph
}
```

## 验证

1. `cargo check --workspace --all-targets` 零 warning
2. `cargo test --workspace` 全通过
3. cube example release 模式 ≥60fps
4. profile_viewer 静态场景 CPU traversal <0.5ms
