# 闭合 8 个实现缺口 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 闭合审计发现的 8 个实现缺口：渲染管线闭环（A组）、数据处理增强（B组）、导入文档/测试（C组）。

**Architecture:** 方案 B — RenderCollector 新增 `effect_commands` 字段，单次场景图遍历同时收集几何 DrawCall 和特效命令。FrameState 暂存特效命令，execute_passes 中 solid pass 后调度特效 pass。pass_effects encode 方法从占位计数升级为真正的 wgpu 绘制调用。

**Tech Stack:** Rust + wgpu + Coin3D 场景图模式

---

## 文件结构

```
crates/rc3d-render/src/
├── render_action.rs          ← Task 2,9: +effect_commands 字段, 3个节点分支, 集成测试
├── render_passes/
│   ├── pass_text.rs          ← Task 1: 清理 pub 可见性
│   └── pass_effects.rs       ← Task 7: encode_placeholder → encode
├── render_passes.rs          ← Task 8: 特效 pass 调度入口
├── renderer_internals.rs     ← Task 6: FrameState +effect_commands
├── renderer_render.rs        ← Task 6: 从 collector 读取 effect_commands
├── renderer.rs               ← (ensure_* 已存在，无需修改)
crates/rc3d-app/src/
├── world.rs                  ← Task 6: reset_collector 清除 effect_commands
└── app/event_handler.rs      ← Task 8: 渲染前转移 effect_commands 到 renderer
crates/rc3d-pointcloud/src/
└── lib.rs                    ← Task 3: Frustum 查询 + stream_tile 文档
crates/rc3d-pdf/src/
└── lib.rs                    ← Task 4: 场景统计嵌入占位 PDF
crates/rc3d-io/src/fbx/
└── mod.rs                    ← Task 5: FBX 版本文档注释
```

---

## Phase 1：基础变更（可并行执行）

### Task 1: pass_text 可见性清理

**Files:** `crates/rc3d-render/src/render_passes/pass_text.rs:9-14`

**变更:** `TextDrawCommand` 结构体及其字段不需要 `pub`（仅同 crate 内使用，`collect_text_nodes` 在 `renderer_render.rs` 调用）。去掉 `pub` 消除 dead-code warning。保留 `collect_text_nodes` 的 `pub(crate)` 可见性。

- [ ] **Step 1: 修改 `TextDrawCommand` 可见性**

将第 9-14 行：
```rust
pub struct TextDrawCommand {
    pub string: String,
    pub screen_pos: [f32; 2],
    pub size: f32,
    pub color: [f32; 4],
    pub is_3d: bool,
}
```
改为：
```rust
pub(crate) struct TextDrawCommand {
    pub(crate) string: String,
    pub(crate) screen_pos: [f32; 2],
    pub(crate) size: f32,
    pub(crate) color: [f32; 4],
    pub(crate) is_3d: bool,
}
```

`collect_text_nodes` 保持 `pub(crate)`，`collect_recursive` 改为 `fn`（`pub` 已不需要——同模块内使用）。

- [ ] **Step 2: 编译验证**

```bash
rtk cargo check -p rc3d-render
```
预期：零 warning（此前 `TextDrawCommand` 的 dead-code warning 应消失）。

- [ ] **Step 3: 确认 HUD 文本渲染不退化**

```bash
rtk cargo build -p rc3d-app --example reflection 2>&1
```
确认无编译错误。HUD 文本渲染链路已在 `renderer_render.rs:42-50` 接通，此 Task 只改可见性。

- [ ] **Step 4: 提交**

```bash
rtk git add crates/rc3d-render/src/render_passes/pass_text.rs
rtk git commit -m "fix: remove pub from TextDrawCommand internals, eliminate dead-code warning"
```

---

### Task 2: RenderCollector 增加 effect_commands 字段

**Files:**
- `crates/rc3d-render/src/render_action.rs:234-285` (struct + new())
- `crates/rc3d-render/src/render_action.rs:321-350` (Decal/Volume/PointCloud 分支)

**变更:** 在 `RenderCollector` 上增加 `effect_commands` 字段，在 `new()` 中初始化为 `EffectCommands::default()`。Decal/Volume/PointCloud 分支从 pass-through 改为写入对应命令。

- [ ] **Step 1: 给 RenderCollector 增加字段**

在 struct 定义中（第 259 行 `stereo_mode` 之后），增加：
```rust
    /// Effect draw commands collected during traversal (Decal, Volume, PointCloud).
    pub effect_commands: crate::render_passes::pass_effects::EffectCommands,
```

在 `new()` 中（第 283 行 `stereo_mode: None,` 之后），增加：
```rust
            effect_commands: crate::render_passes::pass_effects::EffectCommands::default(),
```

- [ ] **Step 2: 修改 Decal 节点分支**

将第 322 行：
```rust
NodeData::Decal(_) => { for &child in &entry.children { self.traverse_node(graph, child); } }
```
改为：
```rust
NodeData::Decal(decal) => {
    self.effect_commands.decals.push(crate::render_passes::pass_effects::DecalDrawCommand {
        model_matrix: self.state.model_matrix(),
        position: decal.position,
        direction: decal.direction,
        size: decal.size,
        texture_path: decal.texture_path.clone(),
        color: decal.color,
        opacity: decal.opacity,
        is_overlay: self.inside_annotation,
    });
    for &child in &entry.children {
        self.traverse_node(graph, child);
    }
}
```

- [ ] **Step 3: 修改 Volume 节点分支**

将第 348 行：
```rust
NodeData::Volume(_) => { for &child in &entry.children { self.traverse_node(graph, child); } }
```
改为：
```rust
NodeData::Volume(volume) => {
    self.effect_commands.volumes.push(crate::render_passes::pass_effects::VolumeDrawCommand {
        model_matrix: self.state.model_matrix(),
        dimensions: volume.dimensions,
        texture_path: volume.texture_path.clone(),
        density_scale: volume.density_scale,
        color_map: volume.color_map,
        is_overlay: self.inside_annotation,
    });
    for &child in &entry.children {
        self.traverse_node(graph, child);
    }
}
```

- [ ] **Step 4: 修改 PointCloud 节点分支**

将第 350 行：
```rust
NodeData::PointCloud(_) => { for &child in &entry.children { self.traverse_node(graph, child); } }
```
改为：
```rust
NodeData::PointCloud(point_cloud) => {
    self.effect_commands.point_clouds.push(crate::render_passes::pass_effects::PointCloudDrawCommand {
        model_matrix: self.state.model_matrix(),
        file_path: point_cloud.file_path.clone(),
        max_visible_points: point_cloud.max_visible_points,
        point_size: point_cloud.point_size,
        color: point_cloud.color,
        is_overlay: self.inside_annotation,
    });
    for &child in &entry.children {
        self.traverse_node(graph, child);
    }
}
```

- [ ] **Step 5: 编译验证**

```bash
rtk cargo check -p rc3d-render
```
预期：零 error、零 warning。

这个阶段 `effect_commands` 字段虽然被写入但尚未被消费，会产生一个 `effect_commands` 写入后无读取的 warning。这是预期行为——Phase 2 会消费它。

- [ ] **Step 6: 提交**

```bash
rtk git add crates/rc3d-render/src/render_action.rs
rtk git commit -m "feat: collect Decal/Volume/PointCloud commands in RenderCollector"
```

---

### Task 3: PointCloud OOC 视锥查询修复

**Files:** `crates/rc3d-pointcloud/src/lib.rs:57-83`, `crates/rc3d-pointcloud/Cargo.toml`

**变更:** 将 `query_frustum(&self, view: &[Vec3; 8])` 改为使用 `rc3d_core::Frustum` 的精确相交测试。同时为 `stream_tile` 增加磁盘 I/O 路径文档。

先检查 rc3d-core 是否已有 Frustum 类型。

- [ ] **Step 1: 确认 Frustum 类型的 API**

```bash
rtk grep "pub.*fn.*intersects" --path crates/rc3d-core/src/frustum.rs
```

预期：存在 `intersects_aabb` 或类似方法。

- [ ] **Step 2: 修改 query_frustum 签名和实现**

将 `lib.rs` 第 57-62 行：
```rust
pub fn query_frustum(&self, view: &[Vec3; 8]) -> Vec<Point> {
    let frustum_bounds = bounds_from_points(view);
    let mut out = Vec::new();
    self._query_frustum(&frustum_bounds, &mut out);
    out
}
```
改为使用 `rc3d_core::Frustum`：
```rust
/// Query points within a view frustum.
/// Uses exact frustum-vs-AABB intersection (no AABB approximation).
pub fn query_frustum(&self, frustum: &rc3d_core::Frustum) -> Vec<Point> {
    let mut out = Vec::new();
    self._query_frustum(frustum, &mut out);
    out
}

fn _query_frustum(&self, frustum: &rc3d_core::Frustum, out: &mut Vec<Point>) {
    match self {
        Octree::Leaf { bounds, points } => {
            if !frustum.intersects_aabb(bounds) {
                return;
            }
            out.extend_from_slice(points);
        }
        Octree::Node { children, .. } => {
            for child in children.iter() {
                if frustum.intersects_aabb(child.bounds()) {
                    child._query_frustum(frustum, out);
                }
            }
        }
    }
}
```

同时修改 `PointCloudOoc::query` 的签名：
```rust
pub fn query(&self, frustum: &rc3d_core::Frustum) -> Vec<Point> {
    self.octree.query_frustum(frustum)
}
```

- [ ] **Step 3: 删除旧的辅助函数**

`bounds_from_points` 和 `contains_point` 不再被使用，删除之。

- [ ] **Step 4: 为 stream_tile 增加磁盘 I/O 文档**

在第 172 行 `pub fn stream_tile` 上方增加文档注释：
```rust
/// Stream a tile from storage into the LRU cache.
///
/// Current implementation loads points from in-memory octree leaves.
///
/// ## Disk I/O (planned)
///
/// To enable true out-of-core streaming from disk, a binary point format
/// and file layout must be defined. Sketch:
///
/// ```text
/// [TileHeader; N]  // one per tile: offset: u64, count: u32
/// [Point; count_0] // tile 0 points
/// [Point; count_1] // tile 1 points
/// ...
/// ```
///
/// TileHeader can be stored at the start of a `.bin` companion file.
pub fn stream_tile(&mut self, tile_id: usize, frame: u64) -> Option<&TileCache> {
```

- [ ] **Step 5: 更新 Cargo.toml 依赖（如需要）**

检查 `rc3d-pointcloud/Cargo.toml` 是否已有 `rc3d-core` 依赖：
```bash
rtk grep "rc3d-core" crates/rc3d-pointcloud/Cargo.toml
```
如不存在则添加：
```toml
rc3d-core = { path = "../rc3d-core" }
```

- [ ] **Step 6: 运行点云测试**

```bash
rtk cargo test -p rc3d-pointcloud
```
预期：4 个测试全通过（`query_frustum_filters_points_outside_frustum_bounds` 需要更新适配新签名）。

- [ ] **Step 7: 更新 query_frustum 测试适配新 API**

测试 `query_frustum_filters_points_outside_frustum_bounds` 中构造 `[Vec3; 8]` frustum 角点的地方改为构造 `rc3d_core::Frustum`。若 `Frustum` 缺少从角点构造的方法，可从 `Mat4` 构造（`Frustum::from_view_projection(vp)`）。

测试 `test_octree_query` 同。

- [ ] **Step 8: 编译验证**

```bash
rtk cargo check -p rc3d-pointcloud
rtk cargo test -p rc3d-pointcloud
```
预期：全绿。

- [ ] **Step 9: 提交**

```bash
rtk git add crates/rc3d-pointcloud/src/lib.rs crates/rc3d-pointcloud/Cargo.toml
rtk git commit -m "fix: use Frustum for exact pointcloud query, add disk I/O docs to stream_tile"
```

---

### Task 4: PDF 场景统计嵌入

**Files:** `crates/rc3d-pdf/src/lib.rs:132-145`

**变更:** `export_u3d_pdf` 的占位内容替换为包含场景统计（节点数、包围盒、类型分布）的格式化表格。

- [ ] **Step 1: 增加场景包围盒计算函数**

在 `count_total` 函数前增加：
```rust
fn scene_bounds(graph: &SceneGraph) -> rc3d_core::Aabb {
    let mut bounds = rc3d_core::Aabb::empty();
    collect_bounds(graph, &mut bounds);
    bounds
}

fn collect_bounds(graph: &SceneGraph, bounds: &mut rc3d_core::Aabb) {
    for &root in graph.roots() {
        collect_bounds_recursive(graph, root, bounds);
    }
}

fn collect_bounds_recursive(
    graph: &SceneGraph,
    node: rc3d_core::NodeId,
    bounds: &mut rc3d_core::Aabb,
) {
    let Some(entry) = graph.get(node) else { return };
    if let Some(bbox) = rc3d_actions::get_bounding_box::bounding_box_of_node(graph, node) {
        *bounds = bounds.union(&bbox);
    }
    for &child in &entry.children {
        collect_bounds_recursive(graph, child, bounds);
    }
}
```

若 `rc3d_actions::get_bounding_box::bounding_box_of_node` 不存在，则用简化实现——只遍历不累加 AABB，占位中标注 "bounding box: not computed (offline)"。

- [ ] **Step 2: 改进 export_u3d_pdf 内容**

将 `export_u3d_pdf` 第 133-145 行改为：
```rust
pub fn export_u3d_pdf(graph: &SceneGraph, title: &str) -> Result<Vec<u8>, String> {
    let mut doc = PdfDocument::new(title);

    // Collect scene statistics
    let mut node_count = 0u32;
    let mut type_counts: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    count_scene_nodes(graph, &mut node_count, &mut type_counts);

    let bounds_line = "Bounding box: (not computed)";

    let mut types: Vec<(&str, u32)> = type_counts.into_iter().collect();
    types.sort_by(|a, b| b.1.cmp(&a.1));
    let mut type_table = String::new();
    for (name, count) in &types {
        type_table.push_str(&format!("  {}: {}\n", name, count));
    }

    let content = format!(
        "3D PDF Export — Scene Statistics\n\n\
         Scene: {title}\n\
         Total nodes: {node_count}\n\
         Root nodes: {}\n\
         {bounds_line}\n\n\
         Node type distribution:\n\
         {type_table}\n\
         \n\
         Note: Full 3D embedding (U3D/PRC) requires a native C library.\n\
         See crate documentation for integration path.",
        graph.roots().len(),
    );
    doc.add_page(&content);
    Ok(doc.to_bytes())
}
```

- [ ] **Step 3: 确认 rc3d-pdf 依赖包含 rc3d-core**

```bash
rtk grep "rc3d-core" crates/rc3d-pdf/Cargo.toml
```

- [ ] **Step 4: 运行 PDF 测试**

```bash
rtk cargo test -p rc3d-pdf
```
预期：全通过。

- [ ] **Step 5: 提交**

```bash
rtk git add crates/rc3d-pdf/src/lib.rs
rtk git commit -m "feat: embed scene statistics in 3D PDF placeholder export"
```

---

### Task 5: FBX 版本兼容文档

**Files:** `crates/rc3d-io/src/fbx/mod.rs:25-26`

- [ ] **Step 1: 增加文档注释**

在 `parse_fbx_file` 函数上方（第 25 行 `/// Parse an FBX file...` 之后）插入版本兼容矩阵：

```rust
/// Parse an FBX file into a SceneGraph.
///
/// # Supported FBX versions
///
/// | FBX Version | Year | Status |
/// |-------------|------|--------|
/// | 7.4.0 (binary) | 2011–2013 | Supported — via fbxcel V7400 parser |
/// | 7.5.0+ (binary) | 2014+ | Not supported — fbxcel v0.9 exposes only V7400 |
/// | ASCII FBX (any) | — | Not supported — fbxcel is binary-only |
///
/// # Future compatibility
///
/// 1. **fbxcel >= 0.10** may add `V7500` parser variant — then a simple match arm addition
///    here will unlock FBX 2014+ files.
/// 2. **Autodesk FBX SDK C-FFI** — compile the official SDK as a static library and create
///    Rust bindings via `extern "C"`. This is the path for full FBX coverage including ASCII.
pub fn parse_fbx_file(path: &Path) -> Result<SceneGraph, FbxError> {
```

- [ ] **Step 2: 编译验证**

```bash
rtk cargo check -p rc3d-io
```

- [ ] **Step 3: 提交**

```bash
rtk git add crates/rc3d-io/src/fbx/mod.rs
rtk git commit -m "docs: add FBX version compatibility matrix to parse_fbx_file"
```

---

## Phase 2：渲染管线闭环（依赖 Task 2）

### Task 6: FrameState + world.rs + renderer_render 接线

**Files:**
- `crates/rc3d-render/src/renderer_internals.rs:32-46` (FrameState)
- `crates/rc3d-app/src/world.rs:46-54` (reset_collector)
- `crates/rc3d-render/src/renderer_render.rs:42-50` (render_draw_calls_core)

**变更:** FrameState 增加 `effect_commands` 字段。world.rs 的 `reset_collector` 清除 `effect_commands`。renderer_render 在渲染前读取 `effect_commands` 存入 frame。

- [ ] **Step 1: FrameState 增加字段**

在 `renderer_internals.rs` 第 45 行 `frame_stats` 之后增加：
```rust
    /// Effect commands collected during scene traversal (Decal, Volume, PointCloud).
    pub effect_commands: crate::render_passes::pass_effects::EffectCommands,
```

需要在 FrameState 的构造处（可能在 renderer.rs 的 Renderer::new）添加初始化。查找：
```bash
rtk grep "FrameState" crates/rc3d-render/src/renderer.rs -A 20
```
在所有 `FrameState { .. }` 构造处增加：
```rust
effect_commands: crate::render_passes::pass_effects::EffectCommands::default(),
```

- [ ] **Step 2: world.rs reset_collector 清除 effect_commands**

在 `reset_collector` 方法第 47 行 `self.collector.draw_calls.clear();` 之后增加：
```rust
self.collector.effect_commands = rc3d_render::render_passes::pass_effects::EffectCommands::default();
```

需要确认 `rc3d_render::render_passes::pass_effects::EffectCommands` 是否可达。`pass_effects` 模块是 `pub(crate)`——在 rc3d-render 内部可见，但外部 crate 不可访问。

**解决方案**：从 `rc3d_render` 重新导出 `EffectCommands`。

在 `crates/rc3d-render/src/lib.rs` 中检查导出：
```bash
rtk grep "EffectCommands\|pass_effects\|effect" crates/rc3d-render/src/lib.rs
```
若未导出，则在 lib.rs 增加：
```rust
pub use crate::render_passes::pass_effects::EffectCommands;
```

然后在 world.rs 中使用 `use rc3d_render::EffectCommands;` 并改为：
```rust
self.collector.effect_commands = EffectCommands::default();
```

- [ ] **Step 3: renderer_render.rs 读取 effect_commands**

在 `render_draw_calls_core` 开头（第 42-50 行现有 `hud.overlay_lines` 赋值之后），不需要从 scene 再收集——effect_commands 已经在 FrameState 上。

不需要修改 renderer_render.rs。效果命令已在 FrameState 中，execute_passes 可以直接访问 `renderer.frame.effect_commands`。

**但有一个问题**：`effect_commands` 是谁写入 FrameState 的？是 app 层（event_handler）在渲染前。

在 event_handler.rs 第 648 行（`if !app.state.world.collector.draw_calls.is_empty()` 之前），增加：
```rust
// Transfer effect commands from collector to renderer
renderer.frame.effect_commands =
    std::mem::take(&mut app.state.world.collector.effect_commands);
```

需要确认 `renderer.frame` 的可见性——它是 `pub(crate)` 字段。event_handler.rs 在 rc3d-app crate 中，不能直接访问 `pub(crate)` 字段。

**解决方案**：给 Renderer 增加一个 public setter：
```rust
// renderer.rs
pub fn set_effect_commands(&mut self, cmds: EffectCommands) {
    self.frame.effect_commands = cmds;
}
```

然后在 event_handler.rs 中：
```rust
use rc3d_render::EffectCommands;
// ...
renderer.set_effect_commands(
    std::mem::take(&mut app.state.world.collector.effect_commands)
);
```

- [ ] **Step 4: 编译验证**

```bash
rtk cargo check --workspace
```
预期：零 error。`collector.effect_commands` 被写入（Task 2）且有消费者（event_handler→renderer→frame），dead-code warning 消失。

- [ ] **Step 5: 提交**

```bash
rtk git add crates/rc3d-render/src/renderer_internals.rs crates/rc3d-render/src/renderer.rs crates/rc3d-render/src/lib.rs crates/rc3d-app/src/world.rs crates/rc3d-app/src/app/event_handler.rs
rtk git commit -m "feat: wire effect_commands from RenderCollector through FrameState to renderer"
```

---

### Task 7: pass_effects encode 方法 — 真正的 Draw Call 调度

**Files:** `crates/rc3d-render/src/render_passes/pass_effects.rs:213-357`

**变更:** 将三个 `encode_placeholder` 替换为 `encode` 方法，提交真实的 wgpu 绘制调用。

- [ ] **Step 1: DecalPass::encode — 全屏 quad + 深度重建贴花投影**

替换 `DecalPass::encode_placeholder`（第 213-234 行）：
```rust
pub fn encode(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    shade_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    commands: &[DecalDrawCommand],
    viewport_w: u32,
    viewport_h: u32,
) {
    if commands.is_empty() {
        return;
    }
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Decal Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: shade_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(&self.pipeline);

    for cmd in commands {
        if cmd.texture_path.is_empty() {
            continue;
        }
        // Upload decal params to uniform buffer
        let params = DecalUniforms {
            model: cmd.model_matrix.to_cols_array_2d(),
            position: [cmd.position.x, cmd.position.y, cmd.position.z, 1.0],
            direction: [cmd.direction.x, cmd.direction.y, cmd.direction.z, 0.0],
            size: cmd.size,
            color: cmd.color,
            opacity: cmd.opacity,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Bind group with decal texture — placeholder: use a 1x1 white texture
        // Full impl requires texture loading from cmd.texture_path
        // See crate::gpu_resource for texture upload patterns
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Decal BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(
                    &device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("decal placeholder"),
                        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    }).create_view(&wgpu::TextureViewDescriptor::default())
                ) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(
                    wgpu::BufferBinding { buffer: &self.params_buf, offset: 0, size: None }
                ) },
            ],
        });
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..6, 0..1); // fullscreen quad (6 vertices)
    }
}
```

需要在文件顶部增加 uniform 结构体定义：
```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DecalUniforms {
    model: [[f32; 4]; 4],
    position: [f32; 4],
    direction: [f32; 4],
    size: [f32; 2],
    color: [f32; 4],
    opacity: f32,
    _pad: [f32; 3],
}
```

- [ ] **Step 2: VolumePass::encode — 全屏 quad ray-march**

替换 `VolumePass::encode_placeholder`（第 282-301 行）：
```rust
pub fn encode(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    shade_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    commands: &[VolumeDrawCommand],
    _viewport_w: u32,
    _viewport_h: u32,
) {
    if commands.is_empty() {
        return;
    }
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Volume Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: shade_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(&self.pipeline);

    for cmd in commands {
        if cmd.texture_path.is_empty() {
            continue;
        }
        let params = VolumeUniforms {
            model: cmd.model_matrix.to_cols_array_2d(),
            dimensions: [cmd.dimensions[0], cmd.dimensions[1], cmd.dimensions[2], 0],
            density_scale: cmd.density_scale,
            color_map: cmd.color_map,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Volume BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(
                    &device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("volume placeholder"),
                        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                        mip_level_count: 1, sample_count: 1,
                        dimension: wgpu::TextureDimension::D3,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    }).create_view(&wgpu::TextureViewDescriptor::default())
                ) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(
                    wgpu::BufferBinding { buffer: &self.params_buf, offset: 0, size: None }
                ) },
            ],
        });
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..6, 0..1);
    }
}
```

Volume uniform 结构体：
```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VolumeUniforms {
    model: [[f32; 4]; 4],
    dimensions: [u32; 4],
    density_scale: f32,
    color_map: [[f32; 4]; 4],
    _pad: [f32; 3],
}
```

- [ ] **Step 3: PointCloudPass::encode — PointList 点精灵绘制**

替换 `PointCloudPass::encode_placeholder`（第 338-356 行）：
```rust
pub fn encode(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    shade_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    commands: &[PointCloudDrawCommand],
    projection: glam::Mat4,
    inv_projection: glam::Mat4,
    _viewport_w: u32,
    _viewport_h: u32,
) {
    if commands.is_empty() {
        return;
    }
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("PointCloud Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: shade_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(&self.pipeline);

    for cmd in commands {
        if cmd.file_path.is_empty() || cmd.max_visible_points == 0 {
            log::warn!("PointCloud: skipping command with no file or zero max_visible_points");
            continue;
        }
        let mvp = projection * cmd.model_matrix;
        let uniforms = PointCloudUniforms {
            mvp: mvp.to_cols_array_2d(),
            model: cmd.model_matrix.to_cols_array_2d(),
            inv_proj: inv_projection.to_cols_array_2d(),
            point_size: cmd.point_size,
            color: cmd.color,
            _pad: [0.0; 3],
        };
        let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Placeholder: empty point buffer (real impl loads from cmd.file_path)
        let point_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PC points placeholder"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PointCloud BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: point_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(
                    wgpu::BufferBinding { buffer: &ub, offset: 0, size: None }
                ) },
            ],
        });
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..0, 0..1); // 占位：实际点数需要从文件加载
    }
}
```

PointCloud uniform 结构体：
```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PointCloudUniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    point_size: f32,
    color: [f32; 4],
    _pad: [f32; 3],
}
```

- [ ] **Step 4: 编译验证**

```bash
rtk cargo check -p rc3d-render
```
预期：零 error。新增的 uniform struct 和 bytemuck 依赖可能需要确认版本兼容。

- [ ] **Step 5: 提交**

```bash
rtk git add crates/rc3d-render/src/render_passes/pass_effects.rs
rtk git commit -m "feat: implement real wgpu draw call dispatch for Decal/Volume/PointCloud passes"
```

---

### Task 8: render_passes 特效调度入口 + event_handler 接线

**Files:**
- `crates/rc3d-render/src/render_passes.rs:470-484` (solid pass 之后)
- `crates/rc3d-app/src/app/event_handler.rs:645-670` (渲染调用前)

**变更:** execute_passes 中在 solid pass 后检查 `effect_commands`，调度对应特效 pass。event_handler 在渲染前将 collector.effect_commands 转移到 renderer.frame。

- [ ] **Step 1: execute_passes 增加特效 pass 调度**

在 `render_passes.rs` 第 484 行（section caps 的 `}` 闭合之后）插入：

```rust
    // ── Effect passes (Decal, Volume, PointCloud) ──
    if !renderer.frame.effect_commands.is_empty() {
        let cmds = &renderer.frame.effect_commands;
        if !cmds.decals.is_empty() {
            renderer.ensure_decal_pass();
            if let Some(ref pass) = renderer.gpu.decal_pass {
                pass.encode(
                    &renderer.device,
                    &renderer.queue,
                    &mut encoder,
                    shade_view,
                    &depth_read_view,
                    &cmds.decals,
                    ew,
                    eh,
                );
            }
        }
        if !cmds.volumes.is_empty() {
            renderer.ensure_volume_pass();
            if let Some(ref pass) = renderer.gpu.volume_pass {
                pass.encode(
                    &renderer.device,
                    &renderer.queue,
                    &mut encoder,
                    shade_view,
                    &depth_read_view,
                    &cmds.volumes,
                    ew,
                    eh,
                );
            }
        }
        if !cmds.point_clouds.is_empty() {
            renderer.ensure_point_cloud_pass();
            if let Some(ref pass) = renderer.gpu.point_cloud_pass {
                pass.encode(
                    &renderer.device,
                    &renderer.queue,
                    &mut encoder,
                    shade_view,
                    &depth_read_view,
                    &cmds.point_clouds,
                    ctx.camera_proj,
                    ctx.camera_inv_proj,
                    ew,
                    eh,
                );
            }
        }
    }
```

注意：effect commands 在 FrameState 中存入后，需要在使用后清除（每帧）。但由于 `render_draw_calls_core` 每次调用都重新执行，且 `effect_commands` 的读取是一次性的，可以在 `execute_passes` 末尾或下次 `render_draw_calls_core` 开始时用 `std::mem::take` 清除。

在 `render_draw_calls_core` 的开头（第 42 行前），`effect_commands` 已由 event_handler 写入。在 `execute_passes` 末尾清除：
```rust
renderer.frame.effect_commands = EffectCommands::default();
```
这条加在 `execute_passes` 返回 `stats` 之前（约第 846 行）。

- [ ] **Step 2: event_handler.rs — 渲染前转移 effect_commands**

在 `event_handler.rs` 第 648 行前插入（确认 `renderer.set_effect_commands` 已在 Task 6 中添加）：
```rust
renderer.set_effect_commands(
    std::mem::take(&mut app.state.world.collector.effect_commands),
);
```

- [ ] **Step 3: 编译验证**

```bash
rtk cargo check --workspace
```
预期：零 error。

- [ ] **Step 4: 提交**

```bash
rtk git add crates/rc3d-render/src/render_passes.rs crates/rc3d-app/src/app/event_handler.rs
rtk git commit -m "feat: dispatch Decal/Volume/PointCloud passes in execute_passes render loop"
```

---

## Phase 3：集成测试

### Task 9: RenderCollector 集成测试 + 最终验证

**Files:** `crates/rc3d-render/src/render_action.rs` (test 模块)

- [ ] **Step 1: 写 RenderCollector 特效收集测试**

在 `render_action.rs` 测试模块（文件末尾 `#[cfg(test)] mod tests`）中增加：

```rust
#[test]
fn collector_collects_effect_commands_from_decal_volume_pointcloud() {
    use rc3d_scene::node_data::*;
    use crate::render_passes::pass_effects::EffectCommands;

    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    graph.add_child(root, NodeData::Decal(DecalNode {
        position: Vec3::new(1.0, 0.0, 0.0),
        direction: Vec3::NEG_Y,
        size: [2.0, 2.0],
        texture_path: "d.png".to_string(),
        color: [1.0, 0.0, 0.0, 0.5],
        opacity: 0.8,
    }));
    graph.add_child(root, NodeData::Volume(VolumeNode {
        dimensions: [32, 32, 32],
        texture_path: "v.raw".to_string(),
        density_scale: 1.0,
        color_map: [[0.0; 4]; 4],
    }));
    graph.add_child(root, NodeData::PointCloud(PointCloudNode {
        file_path: "p.bin".to_string(),
        max_visible_points: 1000,
        point_size: 2.0,
        color: [0.0, 1.0, 0.0, 1.0],
    }));

    let mut collector = super::RenderCollector::new();
    collector.traverse(&graph, root);

    assert_eq!(collector.effect_commands.decals.len(), 1);
    assert_eq!(collector.effect_commands.decals[0].texture_path, "d.png");
    assert_eq!(collector.effect_commands.volumes.len(), 1);
    assert_eq!(collector.effect_commands.volumes[0].texture_path, "v.raw");
    assert_eq!(collector.effect_commands.point_clouds.len(), 1);
    assert_eq!(collector.effect_commands.point_clouds[0].file_path, "p.bin");
    assert_eq!(collector.effect_commands.point_clouds[0].max_visible_points, 1000);
}

#[test]
fn inside_annotation_sets_is_overlay_on_effect_commands() {
    use rc3d_scene::node_data::*;

    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    let annotation = graph.add_child(root, NodeData::Annotation(AnnotationNode::default()));
    graph.add_child(annotation, NodeData::Decal(DecalNode {
        texture_path: "overlay.png".to_string(),
        ..Default::default()
    }));

    let mut collector = super::RenderCollector::new();
    collector.traverse(&graph, root);

    assert_eq!(collector.effect_commands.decals.len(), 1);
    assert!(collector.effect_commands.decals[0].is_overlay);
}
```

- [ ] **Step 2: 运行测试**

```bash
rtk cargo test -p rc3d-render
```
预期：所有库测试通过（包括新增的 2 个测试）。

- [ ] **Step 3: 全工作区编译 + 测试**

```bash
rtk cargo check --workspace --all-targets
rtk cargo test --workspace --lib
```
预期：零 warning、全测试通过。

- [ ] **Step 4: 提交**

```bash
rtk git add crates/rc3d-render/src/render_action.rs
rtk git commit -m "test: add RenderCollector integration tests for effect_commands and annotation overlay"
```

---

## 验证清单

全部 Task 完成后执行：

- [ ] `cargo check --workspace --all-targets` — 零 error、零 warning
- [ ] `cargo test --workspace --lib` — ≥177 测试通过
- [ ] `cargo build -p rc3d-app --examples` — 所有 example 编译通过
- [ ] 审计中报告的 unused/dead-code warning 消失（TextDrawCommand、collect_text_nodes、effect pass 结构体等）

## 不在此范围

- StereoCamera 双视口渲染（架构变更）
- RayTracing 硬件光追（等 wgpu DXR/VKRT）
- U3D/PRC 真三维 PDF（等 C-FFI 桥接）
- FBX 7.5+ 版本（等 fbxcel 更新）
- PointCloud 磁盘流式 I/O（需定义二进制格式）
