# 闭合 8 个实现缺口 — 设计文档

> 审计结论：仓库能编译通过（175 测试），但存在类型/节点/管线已声明、主实现未闭环的缺口。本文档覆盖全部 8 个缺口的修复设计。

## 架构概览

```
SceneGraph
    ↓ RenderCollector::traverse_node (遍历一次)
    ↓ 产生: DrawCall[] + EffectCommands + TextCommands
    ↓ render_draw_calls_core
    ↓ execute_passes
        ├─ shadow pass
        ├─ solid pass
        ├─ section cap pass
        ├─ effect passes (Decal/Volume/PointCloud)  ← 新增调度
        ├─ post-processing
        ├─ HUD overlay (glyphon 渲染文本)  ← 已有
        └─ markup overlay
    ↓ Frame
```

核心原则：**单次场景图遍历产生所有渲染命令**，符合 Coin3D Action 模式。

---

## A 组：渲染管线闭环

### 缺口 #1, #3：Decal/Volume/PointCloud 节点 + Annotation 标记

**现状**：`RenderCollector::traverse_node` 中 Decal/Volume/PointCloud 分支只遍历子节点，不产生渲染命令。Annotation 分支已正确设置 `inside_annotation` 标记，此处无需修改。

**设计**（方案 B — RenderCollector 独立字段收集）：

在 `RenderCollector` 上增加 `effect_commands: EffectCommands` 字段，三个特效节点分支直接推入对应命令：

```rust
// render_action.rs — RenderCollector 新增字段
pub effect_commands: crate::render_passes::pass_effects::EffectCommands,
```

Decal 分支改为：
```rust
NodeData::Decal(decal) => {
    self.effect_commands.decals.push(DecalDrawCommand {
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

Volume 和 PointCloud 同理。

不污染 `DrawCall` 结构——特效命令的字段（纹理路径、颜色映射、点大小等）与几何 DrawCall 的字段（vertices、indices、meshlet_data）无交集。

### 缺口 #2：Text2/Text3 渲染

**现状**：`collect_text_nodes()` 已在 `renderer_render.rs:42-50` 被调用，结果推入 `hud.overlay_lines`，HUD pass 用 glyphon 渲染。渲染链路**已完整**。

**修复**：仅需清理 `pass_text.rs` 中的 `pub` 可见性——`TextDrawCommand` 及其字段不需要 `pub`（同 crate 内使用）。这消除 cargo dead-code warning。

### 缺口 #4：特效 Pass 调度

**现状**：`ensure_decal_pass()`、`ensure_volume_pass()`、`ensure_point_cloud_pass()` 存在，但无人调用。`encode_placeholder()` 只过滤计数，不提交绘制。

**设计**：

**① 数据流**：RenderCollector → FrameState → execute_passes

```rust
// renderer.rs — FrameState 新增字段
pub effect_commands: EffectCommands,
```

```rust
// renderer_render.rs — 在 render_draw_calls_core 中遍历后取出
self.frame.effect_commands =
    std::mem::take(&mut collector.effect_commands);
```

**② render_passes.rs 调度入口**：在 solid pass（第 470-484 行）之后、HUD 之前插入：

```rust
// solid pass 之后
if !renderer.frame.effect_commands.is_empty() {
    let cmds = &renderer.frame.effect_commands;
    if !cmds.decals.is_empty() {
        renderer.ensure_decal_pass();
        if let Some(ref pass) = renderer.gpu.decal_pass {
            pass.encode(&renderer.device, &renderer.queue, &mut encoder,
                        shade_view, &depth_read_view, &cmds.decals, ew, eh);
        }
    }
    if !cmds.volumes.is_empty() {
        renderer.ensure_volume_pass();
        if let Some(ref pass) = renderer.gpu.volume_pass {
            pass.encode(&renderer.device, &renderer.queue, &mut encoder,
                        shade_view, &depth_read_view, &cmds.volumes, ew, eh);
        }
    }
    if !cmds.point_clouds.is_empty() {
        renderer.ensure_point_cloud_pass();
        if let Some(ref pass) = renderer.gpu.point_cloud_pass {
            pass.encode(&renderer.device, &renderer.queue, &mut encoder,
                        shade_view, &depth_read_view, &cmds.point_clouds,
                        ctx.camera_proj, ctx.camera_inv_proj, ew, eh);
        }
    }
}
```

**③ pass_effects.rs encode 方法**：将 `encode_placeholder` 替换为真正的 `encode`。

- **DecalPass::encode**：对每条命令，写入 uniform buffer（model、position、size、color、opacity、投影矩阵），绑定深度纹理+贴花纹理+采样器，画全屏 quad（6 顶点），片元着色器从深度重建世界坐标并投影贴花。
- **VolumePass::encode**：对每条命令，写入 uniform buffer（dimensions、density_scale、color_map），绑定 3D 纹理+深度纹理+采样器，画全屏 quad，片元着色器 ray-march 体数据。
- **PointCloudPass::encode**：对每条命令，加载/创建点缓冲区作为 storage buffer（如已有路径），写入 uniform buffer（mvp、point_size、color），以 PointList 拓扑绘制。

所有三个 encode 方法签名统一为：
```rust
pub fn encode(&self, device: &wgpu::Device, queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder, shade_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView, commands: &[XxxDrawCommand], w: u32, h: u32);
```

**首次实现简化**：全屏 quad 顶点缓冲区使用 `renderer.gpu.pipelines` 中已有的 `fs_quad` 设施（如存在），或创建简单的 6 顶点三角形。PointCloud 的存储缓冲区若文件路径非空则尝试加载 `.bin` 点文件，否则在 encode 中跳过该命令并 log warn。

### StereoCamera 和 RayTracing（不修复）

- **StereoCamera**：需要双视口渲染+合成，架构变更量大。保留 TODO 注释。
- **RayTracing**：wgpu 无硬件光追。保留 TODO 注释，标注在 wgpu 支持 DXR/VKRT 后实现。

---

## B 组：数据处理

### 缺口 #5：PointCloud OOC 流式加载

**现状分析**：
- `query_frustum` 将 8 个视锥角点转为 AABB 后做包围盒相交（第 57-62 行），第 67 行确实检查了 intersection，但 AABB 近似导致**过度保守**——包围盒可比实际视锥大数倍
- `stream_tile` 实现完整（第 172-185 行）：LRU 缓存命中检查 → octree leaf 加载 → 写入缓存。审计称"永远返回 None"有误，它是从内存 octree 加载而非磁盘

**修复**：

1. 替换 `query_frustum(&self, view: &[Vec3; 8])` 为 `query_frustum(&self, frustum: &Frustum)`，使用 `rc3d_core::Frustum` 的精确视锥体-AABB 相交测试
2. `stream_tile` 添加文档注释，标注磁盘 I/O 路径：

```rust
/// Stream a tile from disk into cache.
///
/// Current implementation loads from in-memory octree leaves.
/// To enable disk streaming:
/// 1. Define binary point format: `[u32 tile_id][u32 point_count][Point; point_count]`
/// 2. Read tile at offset: `tile_id * TILE_FILE_SIZE_BYTES`
/// 3. Deserialize into Vec<Point> and insert into LRU cache
pub fn stream_tile(&mut self, tile_id: usize, frame: u64) -> Option<&TileCache> {
```

### 缺口 #6：3D PDF 导出

**设计**：改进占位 PDF，嵌入场景统计信息作为格式化文本表格：

```rust
pub fn export_u3d_pdf(graph: &SceneGraph, title: &str) -> Result<Vec<u8>, String> {
    let mut doc = PdfDocument::new(title);
    let node_count = count_total(graph);
    let bounds = scene_bounds(graph);
    // 收集节点类型分布
    let mut type_counts: std::collections::HashMap<&str, u32> = ...;
    
    let content = format!(
        "3D PDF Export — Scene Statistics\n\n\
         Scene: {}\n\
         Total nodes: {}\n\
         Bounding box: [{:.2},{:.2},{:.2}] — [{:.2},{:.2},{:.2}]\n\
         Root nodes: {}\n\n\
         Node type distribution:\n\
         {}",
        title, node_count,
        bounds.min.x, bounds.min.y, bounds.min.z,
        bounds.max.x, bounds.max.y, bounds.max.z,
        graph.roots().len(),
        sorted_types.iter().map(|(n,c)| format!("  {}: {}", n, c))
            .collect::<Vec<_>>().join("\n"),
    );
    // ...
}
```

真 U3D/PRC 三维嵌入仍需要 C-FFI 桥接（已在文档注释中说明）。

---

## C 组：导入/测试

### 缺口 #7：FBX 版本兼容

`fbxcel` v0.9 只暴露 `AnyParser::V7400`。这是上游库限制。

**修复**：在 `parse_fbx_file` 增加 Rust 文档注释：

```rust
/// Parse an FBX file into a SceneGraph.
///
/// # Supported FBX versions
///
/// | FBX Version | Year | Status |
/// |-------------|------|--------|
/// | 7.4.0 | 2011–2013 | Supported (fbxcel V7400) |
/// | 7.5.0+ | 2014+ | Not supported — fbxcel v0.9 has no V7500 parser |
/// | ASCII FBX | — | Not supported — fbxcel is binary-only |
///
/// # Future
///
/// - fbxcel >= 0.10 may add V7500 support
/// - Alternative: Autodesk FBX SDK via C-FFI bridge
pub fn parse_fbx_file(path: &Path) -> Result<SceneGraph, FbxError> {
```

### 缺口 #8：渲染集成测试

**设计**：增加一个集成测试验证 RenderCollector 对特效节点的处理：

```rust
// render_action.rs test 模块新增
#[test]
fn render_collector_collects_effect_commands() {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Group(GroupNode::default()));
    graph.add_child(root, NodeData::Decal(DecalNode {
        texture_path: "test.png".into(),
        ..Default::default()
    }));
    graph.add_child(root, NodeData::Volume(VolumeNode {
        texture_path: "vol.raw".into(),
        ..Default::default()
    }));
    
    let mut collector = RenderCollector::new(/* params */);
    collector.traverse(&graph); // 或等效遍历
    
    assert_eq!(collector.effect_commands.decals.len(), 1);
    assert_eq!(collector.effect_commands.volumes.len(), 1);
}

#[test]
fn inside_annotation_propagates_to_effect_commands() {
    // 构造 Annotation > Decal 嵌套
    // 验证 decal.is_overlay == true
}
```

---

## 文件变更清单

| 文件 | 改动类型 | 行数估计 | 缺口 |
|------|----------|----------|------|
| `crates/rc3d-render/src/render_action.rs` | 修改 + 新增测试 | +30 / -5 | #1, #3, #8 |
| `crates/rc3d-render/src/renderer.rs` | 修改 (FrameState) | +2 | #1 |
| `crates/rc3d-render/src/renderer_render.rs` | 修改 | +3 | #1, #4 |
| `crates/rc3d-render/src/render_passes.rs` | 修改 | +30 | #4 |
| `crates/rc3d-render/src/render_passes/pass_effects.rs` | 修改 (encode) | +90 / -30 | #4 |
| `crates/rc3d-render/src/render_passes/pass_text.rs` | 修改 (可见性) | -5 | #2 |
| `crates/rc3d-pointcloud/src/lib.rs` | 修改 | +15 / -5 | #5 |
| `crates/rc3d-pdf/src/lib.rs` | 修改 | +20 | #6 |
| `crates/rc3d-io/src/fbx/mod.rs` | 修改 (文档) | +10 | #7 |

总计：~200 行新增，~50 行删除/修改。

## 验证标准

1. `cargo check --workspace --all-targets` 零警告
2. `cargo test --workspace --lib` 全通过（≥175 测试 + 新增测试）
3. Decal/Volume/PointCloud 示例场景运行时不崩溃
4. HUD 正确显示 Text2 文本（已有示例验证）
5. 空缺口的 dead-code warning 消失

---

## 不在此次范围的已知限制

- StereoCamera 双视口渲染（需架构变更）
- RayTracing 硬件光追（等 wgpu 支持）
- U3D/PRC 真三维 PDF（需 C-FFI）
- FBX 7.5+ 版本（等上游 fbxcel 更新）
- PointCloud 磁盘流式 I/O（需定义二进制格式）
