# 渲染引擎性能重构 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将渲染引擎 CPU 帧时间降低 30-50%、内存分配减少 80%、CSM shadow 开销降低 75%，通过数据导向重构（FlatDrawCache + 脏标记 + DrawCall 瘦身）和 GPU 管线优化（layered shadow + shader variant + PassDag）。

**Architecture:** 方案 B — 保留现有 pass 架构，引入 FlatDrawCache（扁平连续数组替代 Vec\<DrawCall\>）、场景图脏标记增量更新、GpuDrawData/CachedDrawMetadata 热冷数据分离、CSM layered rendering、着色器按材质特化、PassDag 并行调度、rayon 多线程遍历。

**Tech Stack:** Rust + wgpu + rayon + twox-hash + tracing

---

## 文件结构

```
crates/rc3d-render/src/
├── profiler.rs                    ← Task 1: 新增 GPU/CPU 帧时间分解
├── global_tables.rs               ← Task 3: 新增 TexturePathTable + GlobalLightBuffer
├── flat_draw_cache.rs             ← Task 4: 新增 FlatDrawCache + GpuDrawData + CachedDrawMetadata
├── render_action.rs               ← Task 4,5: 重构 DrawCall → GpuDrawData; traverse_node 改为返回 TraversalChunk
├── dirty_flags.rs                 ← Task 5: 新增 dirty 标记传播逻辑
├── parallel_traversal.rs          ← Task 6: 新增 rayon 并行遍历 + staging belt
├── pass_graph.rs                  ← Task 7: 新增 PassDag 依赖拓扑 + 并行录制调度
├── render_passes/
│   ├── pass_shadow.rs             ← Task 8: 重构 CSM → layered rendering + GPU culling
├── pipelines.rs                   ← Task 9: PbrFeatures variant 缓存 + LRU
├── bvh.rs                         ← Task 10: 增量 BVH 更新（如不存在则创建）
├── texture_streaming.rs           ← Task 11: 纹理流送 + 后台加载
├── adaptive_quality.rs            ← Task 12: QualityLevel 扩展 + 动态阶梯调节
├── lib.rs                         ← Task 3,4: 模块导出更新
├── renderer.rs                    ← Task 3,4,6,7: 初始化新组件; 注册 PassDag
├── renderer_internals.rs           ← Task 4,7: FrameState 适配
├── renderer_render.rs             ← Task 4,7: render_draw_calls_core 适配 FlatDrawCache
├── gpu_resource.rs                ← Task 3: 全局 MeshBuffer + free-list
crates/rc3d-scene/src/
├── scene_graph.rs                 ← Task 5: NodeEntry 增加 dirty_flags
├── node_entry.rs                  ← Task 5: NodeEntry 增加 dirty_flags 字段
crates/rc3d-app/src/app/
├── event_handler.rs               ← Task 4,7: 适配新数据流接口
```

---

## Phase 1：Profiling 基础设施（先量测）

### Task 1: GPU/CPU 帧时间分解仪表盘

**Files:**
- Create: `crates/rc3d-render/src/profiler.rs`
- Modify: `crates/rc3d-render/src/lib.rs`
- Modify: `crates/rc3d-render/src/renderer.rs:76-80` (Renderer struct)
- Modify: `crates/rc3d-render/src/render_passes.rs:400-870` (插桩点)

- [ ] **Step 1: 创建 profiler.rs — GpuTimer + CpuSpan 结构体**

```rust
// crates/rc3d-render/src/profiler.rs

use std::time::Instant;

/// GPU timestamp query wrapper — captures wgpu timestamp at begin/end of each pass.
pub struct GpuTimer {
    set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    capacity: u32,
    next_slot: u32,
    /// Timestamps read back from previous frame (ns).
    pub last_timestamps: Vec<u64>,
    pub labels: Vec<&'static str>,
}

impl GpuTimer {
    /// `capacity` = number of timestamp pairs (begin+end) per frame.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Frame Timer Queries"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity * 2,
        });
        let size = (capacity * 2 * 8) as u64;
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timer Resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timer Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            set,
            resolve_buf,
            staging_buf,
            capacity,
            next_slot: 0,
            last_timestamps: Vec::new(),
            labels: Vec::new(),
        }
    }

    /// Begin a labeled timing section. Returns the query index for end() to use.
    pub fn begin(&mut self, encoder: &mut wgpu::CommandEncoder, label: &'static str) -> u32 {
        let idx = self.next_slot;
        if idx < self.capacity {
            encoder.write_timestamp(&self.set, idx * 2);
            if self.labels.len() <= idx as usize {
                self.labels.push(label);
            } else {
                self.labels[idx as usize] = label;
            }
        }
        self.next_slot += 1;
        idx
    }

    /// End the section. Must be called after all GPU work in that section.
    pub fn end(&self, encoder: &mut wgpu::CommandEncoder, idx: u32) {
        if idx < self.capacity {
            encoder.write_timestamp(&self.set, idx * 2 + 1);
        }
    }

    /// Resolve timestamps and read them back from previous frame.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.next_slot > 0 {
            encoder.resolve_query_set(
                &self.set,
                0..(self.next_slot * 2),
                &self.resolve_buf,
                0,
            );
            encoder.copy_buffer_to_buffer(
                &self.resolve_buf, 0,
                &self.staging_buf, 0,
                (self.next_slot * 2 * 8) as u64,
            );
        }
        self.next_slot = 0;
    }

    pub fn collect(&mut self, device: &wgpu::Device) {
        let buf_slice = self.staging_buf.slice(..);
        buf_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let view = buf_slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&view);
        self.last_timestamps = timestamps.to_vec();
        drop(view);
        self.staging_buf.unmap();
    }
}

/// CPU-side timing span — collects per-frame durations.
#[derive(Default)]
pub struct CpuSpanCollector {
    spans: Vec<(&'static str, f64)>, // (name, duration_ms)
    frame_start: Option<Instant>,
}

impl CpuSpanCollector {
    pub fn begin_frame(&mut self) {
        self.frame_start = Some(Instant::now());
        self.spans.clear();
    }

    /// Time a synchronous closure and record its duration.
    pub fn measure<T>(&mut self, label: &'static str, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        let dur_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.spans.push((label, dur_ms));
        result
    }

    pub fn spans(&self) -> &[(&'static str, f64)] {
        &self.spans
    }

    pub fn total_ms(&self) -> f64 {
        self.frame_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }
}

/// Structured per-frame timing report.
#[derive(Default, Clone, Debug)]
pub struct FrameTimingReport {
    pub cpu_total_ms: f64,
    pub gpu_total_ms: f64,
    pub sections: Vec<(&'static str, f64, f64)>, // (label, cpu_ms, gpu_ms)
}
```

- [ ] **Step 2: 在 lib.rs 中导出 profiler 模块**

在 `crates/rc3d-render/src/lib.rs` 末尾增加：
```rust
pub mod profiler;
```

- [ ] **Step 3: 在 Renderer struct 中增加 profiler 字段**

在 `renderer.rs` 第 80 行 `pub surface` 附近增加：
```rust
pub gpu_timer: crate::profiler::GpuTimer,
pub cpu_span: crate::profiler::CpuSpanCollector,
pub frame_timing: crate::profiler::FrameTimingReport,
```

Renderer::new 中初始化（约第 250 行，在 device 创建之后）：
```rust
gpu_timer: crate::profiler::GpuTimer::new(&device, 32),
cpu_span: crate::profiler::CpuSpanCollector::default(),
frame_timing: crate::profiler::FrameTimingReport::default(),
```

- [ ] **Step 4: 在 render_passes 关键位置插桩**

在 `render_passes.rs` 的 `execute_passes` 开头增加：
```rust
// ── Begin GPU timing ──
let _ti_shadow = renderer.gpu_timer.begin(&mut encoder, "CSM Shadow");
// ... shadow pass code ...
renderer.gpu_timer.end(&mut encoder, _ti_shadow);

let _ti_solid = renderer.gpu_timer.begin(&mut encoder, "Solid+Outline");
// ... solid pass code ...
renderer.gpu_timer.end(&mut encoder, _ti_solid);

let _ti_effects = renderer.gpu_timer.begin(&mut encoder, "Effects");
// ... effect passes code ...
renderer.gpu_timer.end(&mut encoder, _ti_effects);

let _ti_post = renderer.gpu_timer.begin(&mut encoder, "PostProcess");
// ... post process code ...
renderer.gpu_timer.end(&mut encoder, _ti_post);

let _ti_hud = renderer.gpu_timer.begin(&mut encoder, "HUD+Overlay");
// ... HUD/overlay code ...
renderer.gpu_timer.end(&mut encoder, _ti_hud);
```

在编码器结束前（约第 845 行，`let stats = ...` 之前）：
```rust
renderer.gpu_timer.resolve(&mut encoder);
```

- [ ] **Step 5: 在 renderer_render 关键位置增加 CPU span**

在 `render_draw_calls_core` 中：
```rust
renderer.cpu_span.begin_frame();

renderer.cpu_span.measure("bvh_build", || {
    // ... 现有的 bvh 构建 + frustum culling 代码 ...
});

renderer.cpu_span.measure("mesh_upload", || {
    // ... 现有的 mesh upload 代码 ...
});

renderer.cpu_span.measure("sorting", || {
    // ... 现有的 sorting 代码 ...
});
```

- [ ] **Step 6: 在 FrameStats 增加 timing 字段，并在帧末采集**

在 `renderer_types.rs` 的 `FrameStats` 增加：
```rust
pub frame_time_ms: f64,
pub cpu_sections: Vec<(&'static str, f64)>,
pub gpu_sections: Vec<(&'static str, f64)>,
```

在帧末（`execute_passes` 返回 stats 处）：
```rust
// Collect GPU timestamps from previous frame
renderer.gpu_timer.collect(&renderer.device);
let gpu_timestamps = &renderer.gpu_timer.last_timestamps;

let mut gpu_sections = Vec::new();
for i in (0..gpu_timestamps.len()).step_by(2) {
    if i + 1 < gpu_timestamps.len() {
        let label = renderer.gpu_timer.labels.get(i / 2).copied().unwrap_or("?");
        let dur_ns = gpu_timestamps[i + 1].saturating_sub(gpu_timestamps[i]);
        gpu_sections.push((label, dur_ns as f64 / 1_000_000.0));
    }
}

stats.frame_time_ms = renderer.cpu_span.total_ms();
stats.cpu_sections = renderer.cpu_span.spans().to_vec();
stats.gpu_sections = gpu_sections;
renderer.frame_timing = crate::profiler::FrameTimingReport {
    cpu_total_ms: stats.frame_time_ms,
    gpu_total_ms: gpu_sections.iter().map(|(_, ms)| ms).sum(),
    sections: renderer.cpu_span.spans().iter()
        .map(|(l, c)| (*l, *c, 0.0))
        .collect(),
};
```

- [ ] **Step 7: 编译验证**

```bash
rtk cargo check -p rc3d-render
```

预期：零 error。部分 dead-code warning 可接受（profiler 字段暂未全消费）。

- [ ] **Step 8: 提交**

```bash
rtk git add crates/rc3d-render/src/profiler.rs crates/rc3d-render/src/lib.rs \
    crates/rc3d-render/src/renderer.rs crates/rc3d-render/src/renderer_types.rs \
    crates/rc3d-render/src/renderer_render.rs crates/rc3d-render/src/render_passes.rs
rtk git commit -m "feat: add GPU/CPU frame timing profiler with timestamp queries and CPU spans"
```

---

## Phase 2：数据层重构（核心变更）

### Task 2: NodeEntry dirty flags

**Files:**
- Modify: `crates/rc3d-scene/src/node_entry.rs`
- Modify: `crates/rc3d-scene/src/scene_graph.rs`

- [ ] **Step 1: 在 NodeEntry 增加 dirty_flags 字段**

在 `node_entry.rs` 的 `NodeEntry` struct 末尾增加：
```rust
/// Bitflag tracking which aspects of this node have changed since the last
/// render cache update. See `DirtyFlags` for bit definitions.
pub dirty_flags: u8,
```

在 `NodeEntry::new()` 或构造处初始化为 `0`。

- [ ] **Step 2: 创建 DirtyFlags 常量定义**

在 `node_entry.rs` 顶部：
```rust
/// Dirty flag bits for incremental render cache updates.
pub mod dirty_flags {
    pub const TRANSFORM: u8 = 1 << 0;
    pub const MATERIAL: u8 = 1 << 1;
    pub const GEOMETRY: u8 = 1 << 2;
    pub const CHILDREN: u8 = 1 << 3;  // child added/removed/reordered
    pub const REMOVED:  u8 = 1 << 4;
    pub const FROZEN:   u8 = 1 << 7;  // static subtree, skip traversal entirely
}
```

- [ ] **Step 3: 所有 NodeEntry 构造处初始化为 0**

在 `scene_graph.rs` 中所有 `NodeEntry { .. }` 构造处增加：
```rust
dirty_flags: 0,
```

确认 grep 出所有构造点：
```bash
rtk grep "NodeEntry\s*\{" crates/rc3d-scene/src/
```

对每个构造点增加 `dirty_flags: 0,`。

- [ ] **Step 4: 编译验证**

```bash
rtk cargo check -p rc3d-scene
```

- [ ] **Step 5: 提交**

```bash
rtk git add crates/rc3d-scene/src/node_entry.rs crates/rc3d-scene/src/scene_graph.rs
rtk git commit -m "feat: add dirty_flags to NodeEntry for incremental render cache updates"
```

---

### Task 3: 全局表 — TexturePathTable + GlobalLightBuffer

**Files:**
- Create: `crates/rc3d-render/src/global_tables.rs`
- Modify: `crates/rc3d-render/src/lib.rs`
- Modify: `crates/rc3d-render/src/renderer.rs`

- [ ] **Step 1: 创建 global_tables.rs**

```rust
// crates/rc3d-render/src/global_tables.rs

use std::sync::Arc;

/// Interned string table for texture paths.
/// DrawCalls store a u16 index instead of `Arc<str>`.
#[derive(Default)]
pub struct TexturePathTable {
    paths: Vec<Arc<str>>,
}

impl TexturePathTable {
    /// Insert or get existing index for a path.
    pub fn intern(&mut self, path: &str) -> u16 {
        if let Some(idx) = self.paths.iter().position(|p| p.as_ref() == path) {
            return idx as u16;
        }
        let idx = self.paths.len() as u16;
        self.paths.push(Arc::from(path));
        idx
    }

    pub fn get(&self, idx: u16) -> Option<&Arc<str>> {
        self.paths.get(idx as usize)
    }

    /// Build the table from the scene graph by scanning all material nodes.
    pub fn build_from_scene(&mut self, graph: &rc3d_scene::SceneGraph) {
        self.paths.clear();
        // Walk the graph once and intern all texture paths from MaterialNode entries.
        // This runs once at initialization or on scene load, not every frame.
        for (_, entry) in graph.iter() {
            if let rc3d_scene::NodeData::Material(mat) = &entry.data {
                if !mat.albedo_texture.is_empty() { self.intern(&mat.albedo_texture); }
                if !mat.normal_texture.is_empty() { self.intern(&mat.normal_texture); }
                if !mat.metallic_roughness_texture.is_empty() { self.intern(&mat.metallic_roughness_texture); }
                if !mat.emissive_texture.is_empty() { self.intern(&mat.emissive_texture); }
                if !mat.occlusion_texture.is_empty() { self.intern(&mat.occlusion_texture); }
            }
        }
    }
}

/// Per-frame light data, shared across all draws (replaces per-DrawCall MAX_LIGHTS arrays).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    pub direction: [f32; 4],
    pub color: [f32; 4],
    pub position: [f32; 4],
    pub spot_params: [f32; 4],  // (inner_angle, outer_angle, falloff, type)
}

pub const MAX_LIGHTS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalLightUniform {
    pub lights: [GpuLight; MAX_LIGHTS],
    pub count: u32,
    pub _pad: [u32; 3],
}

impl Default for GlobalLightUniform {
    fn default() -> Self {
        Self {
            lights: [GpuLight {
                direction: [0.0; 4],
                color: [0.0; 4],
                position: [0.0; 4],
                spot_params: [0.0; 4],
            }; MAX_LIGHTS],
            count: 0,
            _pad: [0; 3],
        }
    }
}
```

- [ ] **Step 2: 在 lib.rs 中导出 global_tables 模块**

```rust
pub mod global_tables;
```

- [ ] **Step 3: 在 Renderer 初始化时构建全局表**

在 `renderer.rs` 的 `Renderer` struct 增加：
```rust
pub texture_path_table: crate::global_tables::TexturePathTable,
```

- [ ] **Step 4: 编译验证**

```bash
rtk cargo check -p rc3d-render
```

- [ ] **Step 5: 提交**

```bash
rtk git add crates/rc3d-render/src/global_tables.rs crates/rc3d-render/src/lib.rs \
    crates/rc3d-render/src/renderer.rs
rtk git commit -m "feat: add TexturePathTable and GlobalLightUniform for shared render data"
```

---

### Task 4: FlatDrawCache + GpuDrawData/CachedDrawMetadata

**Files:**
- Create: `crates/rc3d-render/src/flat_draw_cache.rs`
- Modify: `crates/rc3d-render/src/render_action.rs:102-290`
- Modify: `crates/rc3d-render/src/renderer_render.rs:22-86`
- Modify: `crates/rc3d-render/src/lib.rs`

- [ ] **Step 1: 创建 flat_draw_cache.rs — 热冷数据分离的结构体**

```rust
// crates/rc3d-render/src/flat_draw_cache.rs

use std::ops::Range;
use crate::render_passes::pass_effects::EffectCommands;

/// Hot data for GPU binding — 64 bytes, cache-line aligned.
/// This is what gets uploaded to the uniform ring buffer each frame.
#[repr(C, align(64))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDrawData {
    pub model_matrix: [[f32; 4]; 4],  // 64B
    pub material_id: u32,             // index into global material uniform buffer
    pub light_set_id: u32,            // bitmask of affected lights from global light buffer
    pub vertex_offset: u32,           // byte offset into global vertex buffer
    pub vertex_count: u32,            // vertex count
    pub index_offset: u32,            // byte offset into global index buffer
    pub index_count: u32,
    pub draw_flags: u32,              // bitflags: HAS_EDGES|TRANSPARENT|OVERLAY|INSTANCED|...
    pub instance_count: u32,          // 1 = non-instanced
    pub _pad: u32,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Default)]
    pub struct DrawFlags: u32 {
        const HAS_EDGES      = 1 << 0;
        const TRANSPARENT    = 1 << 1;
        const OVERLAY        = 1 << 2;
        const INSTANCED      = 1 << 3;
        const SELECTED       = 1 << 4;
        const DEPTH_REVERSED = 1 << 5;
        const ORTHOGRAPHIC   = 1 << 6;
        const WIREFRAME      = 1 << 7;
    }
}

/// Cold data — only updated when the node's dirty flag requires it.
#[derive(Clone)]
pub struct CachedDrawMetadata {
    pub mesh_hash: u64,
    pub bvh_node_id: Option<u32>,
    pub material_params: MaterialUniform,
    pub albedo_tex_id: u16,
    pub normal_tex_id: u16,
    pub mr_tex_id: u16,
    pub emissive_tex_id: u16,
    pub occlusion_tex_id: u16,
    pub atlas_layer: u16,
    pub _pad: [u8; 2],
}

/// Compact material uniform for the PBR shader (what actually goes to GPU).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct MaterialUniform {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub anisotropic: f32,
    pub emissive_color: [f32; 4],
    pub alpha_mode: u32,   // 0=opaque, 1=mask, 2=blend
    pub alpha_cutoff: f32,
    pub double_sided: u32,
    pub _pad: [u32; 1],
}

/// Persistent render cache — reused across frames, incrementally updated.
pub struct FlatDrawCache {
    pub gpu_data: Vec<GpuDrawData>,
    pub metadata: Vec<CachedDrawMetadata>,
    pub effect_commands: EffectCommands,

    // Draw ordering groups (indices into gpu_data/metadata)
    pub opaque_order: Vec<usize>,
    pub transparent_order: Vec<usize>,
    pub edge_order: Vec<usize>,
    pub selected_order: Vec<usize>,

    // Ranges for quick iteration without sorting
    pub opaque_range: Range<u32>,
    pub transparent_range: Range<u32>,

    /// Node-to-cache mapping: graph NodeId -> index in gpu_data[]
    pub node_to_draw: std::collections::HashMap<rc3d_core::NodeId, usize>,

    pub groups_dirty: bool,
    pub total_triangles: u64,
}

impl FlatDrawCache {
    pub fn new() -> Self {
        Self {
            gpu_data: Vec::with_capacity(1024),
            metadata: Vec::with_capacity(1024),
            effect_commands: EffectCommands::default(),
            opaque_order: Vec::with_capacity(1024),
            transparent_order: Vec::with_capacity(128),
            edge_order: Vec::with_capacity(1024),
            selected_order: Vec::with_capacity(64),
            opaque_range: 0..0,
            transparent_range: 0..0,
            node_to_draw: Default::default(),
            groups_dirty: true,
            total_triangles: 0,
        }
    }

    pub fn clear(&mut self) {
        self.gpu_data.clear();
        self.metadata.clear();
        self.effect_commands = EffectCommands::default();
        self.opaque_order.clear();
        self.transparent_order.clear();
        self.edge_order.clear();
        self.selected_order.clear();
        self.node_to_draw.clear();
        self.groups_dirty = true;
        self.total_triangles = 0;
    }

    /// Rebuild draw order groups when dirty.
    pub fn ensure_groups_sorted(&mut self) {
        if !self.groups_dirty {
            return;
        }
        self.opaque_order.clear();
        self.transparent_order.clear();
        self.edge_order.clear();
        self.selected_order.clear();

        for (i, gpu) in self.gpu_data.iter().enumerate() {
            let flags = DrawFlags::from_bits_truncate(gpu.draw_flags);
            if flags.contains(DrawFlags::TRANSPARENT) {
                self.transparent_order.push(i);
            } else {
                self.opaque_order.push(i);
            }
            if flags.contains(DrawFlags::HAS_EDGES) && !flags.contains(DrawFlags::TRANSPARENT) {
                self.edge_order.push(i);
            }
            if flags.contains(DrawFlags::SELECTED) {
                self.selected_order.push(i);
            }
        }
        // Sort opaque by mesh_hash for batching
        self.opaque_order.sort_by_key(|&i| self.metadata[i].mesh_hash);
        // Sort transparent by depth (back-to-front) — done during rendering

        self.opaque_range = 0..(self.opaque_order.len() as u32);
        self.transparent_range = 0..(self.transparent_order.len() as u32);
        self.groups_dirty = false;
    }
}
```

在 `Cargo.toml` 中确认 `bitflags` 依赖存在，如无则增加：
```bash
rtk grep bitflags crates/rc3d-render/Cargo.toml
```

- [ ] **Step 2: 在 Renderer 中增加 FlatDrawCache**

在 `renderer.rs` 的 `Renderer` struct 增加：
```rust
pub draw_cache: crate::flat_draw_cache::FlatDrawCache,
```

在构造函数中初始化：
```rust
draw_cache: crate::flat_draw_cache::FlatDrawCache::new(),
```

- [ ] **Step 3: 调整 renderer_render.rs 使用 FlatDrawCache**

在 `render_draw_calls_core` 中，将：
```rust
fn render_draw_calls_core<'p>(
    &'p mut self,
    draw_calls: &[DrawCall],
    ...
```
改为接受 `&FlatDrawCache` 或直接从 `self.draw_cache` 读取。

**简化第一步**：先用 adapter 函数将 `FlatDrawCache` 转为旧 `DrawCall` 的切片供现有代码消费（最小改动，后续任务逐步迁移）：
```rust
// 临时 adapter — 后续任务会逐步消除
fn draw_calls_from_cache(cache: &FlatDrawCache) -> Vec<DrawCall> {
    // ... construct DrawCall from GpuDrawData + CachedDrawMetadata + global tables ...
    // 仅供 Phase 2 过渡使用，Phase 3 会直接消费 GpuDrawData
}
```

**此 Task 的目标是建立数据结构，不要求完全消除旧 DrawCall。** 后续 Task 5/6/7 逐步完成迁移。

- [ ] **Step 4: 在 lib.rs 中导出 flat_draw_cache 模块**

```rust
pub mod flat_draw_cache;
```

- [ ] **Step 5: 编译验证**

```bash
rtk cargo check -p rc3d-render
```

预期：零编译错误。可能有 dead-code warning（新结构体暂未全面消费），属于预期内。

- [ ] **Step 6: 提交**

```bash
rtk git add crates/rc3d-render/src/flat_draw_cache.rs crates/rc3d-render/src/lib.rs \
    crates/rc3d-render/src/renderer.rs crates/rc3d-render/src/renderer_render.rs
rtk git commit -m "feat: add FlatDrawCache with GpuDrawData/CachedDrawMetadata hot-cold split"
```

---

### Task 5: 场景遍历适配 FlatDrawCache + dirty 标记传播

**Files:**
- Create: `crates/rc3d-render/src/dirty_flags.rs`
- Modify: `crates/rc3d-render/src/render_action.rs:290-450` (traverse_node)
- Modify: `crates/rc3d-render/src/lib.rs`

- [ ] **Step 1: 创建 dirty_flags.rs — 标记传播逻辑**

```rust
// crates/rc3d-render/src/dirty_flags.rs

use rc3d_core::NodeId;
use rc3d_scene::node_entry::dirty_flags::*;
use rc3d_scene::SceneGraph;

/// Mark a node and all ancestors as CHILDREN-dirty when a child is added/removed.
pub fn mark_node_dirty(graph: &mut SceneGraph, node: NodeId, flag: u8) {
    let Some(entry) = graph.get_mut(node) else { return };
    if entry.dirty_flags & FROZEN != 0 {
        return; // frozen subtrees are immutable
    }
    entry.dirty_flags |= flag;

    // Propagate CHILDREN dirty upwards
    if flag & CHILDREN != 0 {
        let mut current = entry.parent;
        while let Some(parent_id) = current {
            if let Some(parent_entry) = graph.get_mut(parent_id) {
                if parent_entry.dirty_flags & FROZEN != 0 {
                    break;
                }
                parent_entry.dirty_flags |= CHILDREN;
                current = parent_entry.parent;
            } else {
                break;
            }
        }
    }
}

/// Collect dirty root nodes — nodes that are themselves dirty but whose parent is clean.
/// These are the entry points for incremental re-traversal.
pub fn collect_dirty_roots(graph: &SceneGraph) -> Vec<NodeId> {
    let mut roots = Vec::new();
    for &root in graph.roots() {
        collect_dirty_subtree(graph, root, &mut roots);
    }
    roots
}

fn collect_dirty_subtree(graph: &SceneGraph, node: NodeId, dirty_roots: &mut Vec<NodeId>) {
    let Some(entry) = graph.get(node) else { return };
    if entry.dirty_flags & FROZEN != 0 {
        return;
    }
    let parent_dirty = entry.parent
        .and_then(|p| graph.get(p))
        .map(|p| p.dirty_flags != 0)
        .unwrap_or(false);

    if entry.dirty_flags != 0 && !parent_dirty {
        dirty_roots.push(node);
    }
    for &child in &entry.children {
        collect_dirty_subtree(graph, child, dirty_roots);
    }
}

/// Clear all dirty flags after a full frame has been processed.
pub fn clear_all_dirty_flags(graph: &mut SceneGraph) {
    for (_, entry) in graph.iter_mut() {
        entry.dirty_flags = 0;
    }
}
```

- [ ] **Step 2: 重构 traverse_node — 输出到 TraversalChunk**

在 `render_action.rs` 中增加：
```rust
/// Thread-local output from traversing a subtree.
pub struct TraversalChunk {
    pub gpu_data: Vec<crate::flat_draw_cache::GpuDrawData>,
    pub metadata: Vec<crate::flat_draw_cache::CachedDrawMetadata>,
    pub effects: crate::render_passes::pass_effects::EffectCommands,
    pub node_to_draw: std::collections::HashMap<rc3d_core::NodeId, usize>,
    pub total_triangles: u64,
}

impl TraversalChunk {
    pub fn new() -> Self {
        Self {
            gpu_data: Vec::new(),
            metadata: Vec::new(),
            effects: crate::render_passes::pass_effects::EffectCommands::default(),
            node_to_draw: Default::default(),
            total_triangles: 0,
        }
    }
}
```

然后增加一个新的遍历方法（旧 `traverse_node` 保留不变，逐步迁移）：
```rust
/// Incremental traversal: only traverse dirty subtrees and merge into FlatDrawCache.
pub fn traverse_into_cache(
    graph: &SceneGraph,
    cache: &mut crate::flat_draw_cache::FlatDrawCache,
    texture_table: &crate::global_tables::TexturePathTable,
    hidden_nodes: &std::collections::HashSet<rc3d_core::NodeId>,
) {
    let dirty_roots = crate::dirty_flags::collect_dirty_roots(graph);

    if dirty_roots.is_empty() {
        // No changes — reuse cache as-is
        cache.ensure_groups_sorted();
        return;
    }

    // If more than 50% of nodes are dirty, rebuild from scratch
    let total_nodes = graph.iter().count();
    if dirty_roots.len() as f64 > total_nodes as f64 * 0.5 {
        cache.clear();
        // Full traversal — collect from all root nodes
        for &root in graph.roots() {
            let mut collector = TraversalCollector::new(texture_table, hidden_nodes);
            collector.traverse_node(graph, root);
            merge_chunk_into_cache(cache, collector.finish());
        }
    } else {
        // Incremental: remove dirty nodes from cache, re-traverse dirty subtrees
        for &dirty_root in &dirty_roots {
            invalidate_cache_for_node(cache, dirty_root);
            let mut collector = TraversalCollector::new(texture_table, hidden_nodes);
            collector.traverse_node(graph, dirty_root);
            merge_chunk_into_cache(cache, collector.finish());
        }
    }

    cache.groups_dirty = true;
    cache.ensure_groups_sorted();
    crate::dirty_flags::clear_all_dirty_flags(&mut /* graph — 需要 &mut 访问 */);
}

fn merge_chunk_into_cache(
    cache: &mut crate::flat_draw_cache::FlatDrawCache,
    chunk: TraversalChunk,
) {
    let base = cache.gpu_data.len();
    cache.gpu_data.extend(chunk.gpu_data);
    cache.metadata.extend(chunk.metadata);
    cache.total_triangles += chunk.total_triangles;
    for (node_id, local_idx) in chunk.node_to_draw {
        cache.node_to_draw.insert(node_id, base + local_idx);
    }
}

fn invalidate_cache_for_node(
    cache: &mut crate::flat_draw_cache::FlatDrawCache,
    node: rc3d_core::NodeId,
) {
    if let Some(&idx) = cache.node_to_draw.get(&node) {
        // Clear the draw entry (will be refilled by re-traversal)
        cache.gpu_data[idx] = crate::flat_draw_cache::GpuDrawData::zeroed();
        // Mark as zeroed so merge skips it later
    }
}
```

旧 `RenderCollector::traverse_node` 保留不动，新增一个 `TraversalCollector` 结构体：
```rust
struct TraversalCollector {
    state: State,
    chunk: TraversalChunk,
    texture_table: *const crate::global_tables::TexturePathTable,
    // ... same traversal logic as RenderCollector but writes to chunk not DrawCall ...
}
```

**注意**：此 Task 工作量较大——需要复制 traverse_node 的逻辑到 TraversalCollector 并修改输出。如果担心风险，可以先做 minimal adapter：`draw_calls_from_cache()` 函数将 GpuDrawData 转回 DrawCall 供旧代码消费。

- [ ] **Step 3: 在 lib.rs 中导出 dirty_flags 模块**

```rust
pub mod dirty_flags;
```

- [ ] **Step 4: 编译验证**

```bash
rtk cargo check -p rc3d-render
```

- [ ] **Step 5: 提交**

```bash
rtk git add crates/rc3d-render/src/dirty_flags.rs crates/rc3d-render/src/render_action.rs \
    crates/rc3d-render/src/lib.rs
rtk git commit -m "feat: incremental scene traversal with dirty flag propagation and FlatDrawCache"
```

---

### Task 6: rayon 并行遍历 + staging belt 上传

**Files:**
- Create: `crates/rc3d-render/src/parallel_traversal.rs`
- Modify: `crates/rc3d-render/src/render_action.rs`
- Modify: `crates/rc3d-render/src/lib.rs`

- [ ] **Step 1: 检查 rayon 依赖**

```bash
rtk grep rayon crates/rc3d-render/Cargo.toml
```

如不存在，增加：
```toml
rayon = "1.10"
```

- [ ] **Step 2: 创建 parallel_traversal.rs**

```rust
// crates/rc3d-render/src/parallel_traversal.rs

use rayon::prelude::*;
use rc3d_scene::SceneGraph;
use crate::flat_draw_cache::{FlatDrawCache, GpuDrawData};
use crate::global_tables::TexturePathTable;
use crate::render_action::TraversalChunk;
use crate::dirty_flags;

/// Parallel traversal of dirty subtrees using rayon.
/// Each dirty subtree is traversed independently on a thread.
/// Results are merged into the FlatDrawCache on the calling thread.
pub fn parallel_traverse_into_cache(
    graph: &SceneGraph,
    cache: &mut FlatDrawCache,
    texture_table: &TexturePathTable,
    hidden_nodes: &std::collections::HashSet<rc3d_core::NodeId>,
) {
    let dirty_roots = dirty_flags::collect_dirty_roots(graph);

    if dirty_roots.is_empty() {
        cache.ensure_groups_sorted();
        return;
    }

    let total_nodes = graph.iter().count();
    if dirty_roots.len() as f64 > total_nodes as f64 * 0.5 {
        // Full rebuild — parallel across root nodes
        cache.clear();
        let root_chunks: Vec<TraversalChunk> = graph.roots()
            .par_iter()
            .map(|&root| {
                super::render_action::traverse_subtree_into_chunk(
                    graph, root, texture_table, hidden_nodes,
                )
            })
            .collect();
        for chunk in root_chunks {
            super::render_action::merge_chunk_into_cache(cache, chunk);
        }
    } else {
        // Incremental — parallel across dirty subtrees
        let chunks: Vec<TraversalChunk> = dirty_roots
            .par_iter()
            .map(|&dirty_root| {
                // Invalidate cache entry first (must be on main thread — do before par_iter)
                // Actually, invalidation is done below on main thread before this block
                super::render_action::traverse_subtree_into_chunk(
                    graph, dirty_root, texture_table, hidden_nodes,
                )
            })
            .collect();

        for chunk in chunks {
            super::render_action::merge_chunk_into_cache(cache, chunk);
        }
    }

    cache.groups_dirty = true;
    cache.ensure_groups_sorted();
}

/// Upload GpuDrawData from FlatDrawCache into a staging belt buffer using
/// parallel memcpy to the mapped buffer.
pub fn parallel_upload_gpu_data(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    cache: &FlatDrawCache,
    dst_buffer: &wgpu::Buffer,
) {
    if cache.gpu_data.is_empty() {
        return;
    }
    let total_size = (cache.gpu_data.len() * std::mem::size_of::<GpuDrawData>()) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GpuData staging"),
        size: total_size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
        mapped_at_creation: true,
    });

    {
        let mapped = staging.slice(..).get_mapped_range_mut();
        let src: &[u8] = bytemuck::cast_slice(&cache.gpu_data);
        // Parallel memcpy into mapped buffer using rayon
        let chunk_size = mapped.len() / rayon::current_num_threads().max(1);
        mapped
            .par_chunks_mut(chunk_size.max(256))
            .enumerate()
            .for_each(|(i, dst_chunk)| {
                let offset = i * chunk_size;
                let end = (offset + dst_chunk.len()).min(src.len());
                if offset < src.len() {
                    dst_chunk[..(end - offset)].copy_from_slice(&src[offset..end]);
                }
            });
    }
    staging.unmap();

    encoder.copy_buffer_to_buffer(&staging, 0, dst_buffer, 0, total_size);

    // staging buffer is dropped here — GPU keeps the copy in dst_buffer
}
```

- [ ] **Step 3: 在 render_action.rs 增加公开函数供 parallel_traversal 调用**

```rust
/// Traverse a single subtree into a TraversalChunk (used by parallel traversal).
pub fn traverse_subtree_into_chunk(
    graph: &SceneGraph,
    root: rc3d_core::NodeId,
    texture_table: &crate::global_tables::TexturePathTable,
    hidden_nodes: &std::collections::HashSet<rc3d_core::NodeId>,
) -> TraversalChunk {
    let mut collector = TraversalCollector::new(texture_table, hidden_nodes);
    collector.traverse_node(graph, root);
    collector.finish()
}
```

- [ ] **Step 4: 在 lib.rs 中导出**

```rust
pub mod parallel_traversal;
```

- [ ] **Step 5: 编译验证**

```bash
rtk cargo check -p rc3d-render
```

- [ ] **Step 6: 提交**

```bash
rtk git add crates/rc3d-render/src/parallel_traversal.rs crates/rc3d-render/src/render_action.rs \
    crates/rc3d-render/src/lib.rs crates/rc3d-render/Cargo.toml
rtk git commit -m "feat: parallel dirty-subtree traversal with rayon + staging belt upload"
```

---

## Phase 3：GPU 管线优化

### Task 7: PassDag 依赖图调度 + 双缓冲命令录制

**Files:**
- Create: `crates/rc3d-render/src/pass_graph.rs`
- Modify: `crates/rc3d-render/src/render_passes.rs:400-870`
- Modify: `crates/rc3d-render/src/renderer.rs`

- [ ] **Step 1: 创建 pass_graph.rs**

```rust
// crates/rc3d-render/src/pass_graph.rs

/// A node in the render pass dependency graph.
#[derive(Clone, Debug)]
pub struct PassNode {
    pub name: &'static str,
    /// Indices of passes that must complete before this one.
    pub depends_on: Vec<usize>,
}

/// Directed acyclic graph of render passes.
/// Topological sort groups independent passes for parallel recording.
pub struct PassDag {
    nodes: Vec<PassNode>,
    /// Groups of pass indices that can be recorded in parallel.
    /// Each group depends on all previous groups.
    pub parallel_groups: Vec<Vec<usize>>,
}

impl PassDag {
    pub fn new(nodes: Vec<PassNode>) -> Self {
        let parallel_groups = Self::toposort_parallel(&nodes);
        Self { nodes, parallel_groups }
    }

    /// Kahn's algorithm grouping independent nodes into parallel layers.
    fn toposort_parallel(nodes: &[PassNode]) -> Vec<Vec<usize>> {
        let n = nodes.len();
        let mut in_degree = vec![0u32; n];
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (i, node) in nodes.iter().enumerate() {
            for &dep in &node.depends_on {
                adjacency[dep].push(i);
                in_degree[i] += 1;
            }
        }

        let mut groups = Vec::new();
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();

        while !queue.is_empty() {
            let group = std::mem::take(&mut queue);
            groups.push(group.clone());

            for &u in &group {
                for &v in &adjacency[u] {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 {
                        queue.push(v);
                    }
                }
            }
        }
        groups
    }

    /// Build the default render pass DAG for this engine.
    pub fn default_render_dag() -> Self {
        let nodes = vec![
            PassNode { name: "Shadow",        depends_on: vec![] },
            PassNode { name: "DepthPrepass",  depends_on: vec![] },
            PassNode { name: "Solid",         depends_on: vec![0, 1] }, // depends on Shadow + Depth
            PassNode { name: "Decal",         depends_on: vec![2] },
            PassNode { name: "Volume",        depends_on: vec![2] },
            PassNode { name: "Transparent",   depends_on: vec![4] }, // after Volume
            PassNode { name: "SSR",           depends_on: vec![5, 6] },
            PassNode { name: "SSAO",          depends_on: vec![3] },
            PassNode { name: "Fog",           depends_on: vec![0] },
            PassNode { name: "DoF",           depends_on: vec![0] },
            PassNode { name: "Bloom",         depends_on: vec![7, 8, 9, 10] }, // after SSR,SSAO,Fog,DoF
            PassNode { name: "TAA",           depends_on: vec![11] },
            PassNode { name: "Tonemap+FXAA",  depends_on: vec![12] },
            PassNode { name: "HUD",           depends_on: vec![13] },
        ];
        Self::new(nodes)
    }
}
```

- [ ] **Step 2: 在 renderer.rs 中初始化 PassDag**

```rust
pub pass_dag: crate::pass_graph::PassDag,
```

构造函数中：
```rust
pass_dag: crate::pass_graph::PassDag::default_render_dag(),
```

- [ ] **Step 3: 在 execute_passes 中使用拓扑排序调度**

在 `render_passes.rs` 中，增加函数：
```rust
fn execute_with_dag(
    renderer: &mut crate::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    shade_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    ew: u32, eh: u32,
) {
    for group in &renderer.pass_dag.parallel_groups {
        if group.len() == 1 {
            // Single pass — record inline
            run_pass_by_index(renderer, encoder, group[0], shade_view, depth_view, ctx, ew, eh);
        } else {
            // Multiple independent passes — record in parallel encoders
            let mut encoders: Vec<wgpu::CommandEncoder> = group
                .iter()
                .map(|_| renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("parallel pass"),
                }))
                .collect();
            // NOTE: full parallel encoder support requires wgpu backend fencing.
            // For now, record sequentially but preserve the DAG structure for future.
            for (i, &pass_idx) in group.iter().enumerate() {
                run_pass_by_index(renderer, &mut encoders[i], pass_idx, shade_view, depth_view, ctx, ew, eh);
                let finished = std::mem::replace(
                    &mut encoders[i],
                    renderer.device.create_command_encoder(&Default::default()),
                );
                // Submit partial encoder — in parallel mode these would be merged
                // Current wgpu limitation: single submit per frame.
                // Future: use multiple queue.submit() calls with fences.
                drop(finished);
            }
        }
    }
}
```

**注意**：wgpu 目前不支持多 encoder 并行录制后一次提交（缺少 fence 原语）。此 Task 先建设 DAG 数据结构，调度逻辑准备好但暂时回退到串行执行。注释标注未来 wgpu 版本支持后启用。

- [ ] **Step 4: 编译验证**

```bash
rtk cargo check -p rc3d-render
```

- [ ] **Step 5: 提交**

```bash
rtk git add crates/rc3d-render/src/pass_graph.rs crates/rc3d-render/src/render_passes.rs \
    crates/rc3d-render/src/renderer.rs
rtk git commit -m "feat: add PassDag dependency graph for render pass scheduling"
```

---

### Task 8: CSM Layered Rendering

**Files:**
- Modify: `crates/rc3d-render/src/render_passes/pass_shadow.rs`
- Modify: `crates/rc3d-render/src/shadow_map.rs` (CsmShadowTarget 改为 texture array)
- Modify: `crates/rc3d-render/src/renderer.rs` (GPU 资源初始化)

- [ ] **Step 1: 替换 CsmShadowTarget — 4 个独立纹理 → 1 个 Texture Array**

在 `shadow_map.rs` 的 `CsmShadowResources`（或等效位置）将 4 个独立 `cascade_views` 替换为：
```rust
pub struct CsmShadowResources {
    pub array_texture: wgpu::Texture,
    pub array_view: wgpu::TextureView,       // full array view
    pub cascade_views: [wgpu::TextureView; 4], // per-layer views (for sampling in PBR shader)
    pub cascade_count: u32,
    pub cascade_size: u32,
    pub sampler: wgpu::Sampler,
    pub bind_group: wgpu::BindGroup,
}

impl CsmShadowResources {
    pub fn new(device: &wgpu::Device, cascade_size: u32, cascade_count: u32) -> Self {
        let count = cascade_count.min(4);
        let array_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CSM Shadow Array"),
            size: wgpu::Extent3d {
                width: cascade_size,
                height: cascade_size,
                depth_or_array_layers: count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let array_view = array_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("CSM Array View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let mut cascade_views = vec![];
        for layer in 0..count {
            cascade_views.push(array_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("CSM cascade {}", layer)),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            }));
        }

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("CSM Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ... build bind group with cascade_views + sampler ...

        Self {
            array_texture,
            array_view,
            cascade_views: cascade_views.try_into().unwrap_or_else(|v: Vec<_>| {
                // fallback: pad with dummy views (unreachable with count ≤ 4)
                let mut arr: [wgpu::TextureView; 4] = Default::default();
                for (i, view) in v.into_iter().enumerate() {
                    arr[i] = view;
                }
                arr
            }),
            cascade_count: count,
            cascade_size,
            sampler,
            bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("CSM Shadow BG"),
                layout: &device.create_bind_group_layout(&Default::default()),
                entries: &[],
            }),
        }
    }
}
```

- [ ] **Step 2: 重写 pass_shadow_depth — 单 pass + instanced draw**

```rust
// crates/rc3d-render/src/render_passes/pass_shadow.rs

use super::PassContext;
use crate::vertex::{ShadowDrawUniforms, CSM_CASCADE_COUNT};

pub(super) fn pass_shadow_depth_layered(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &PassContext<'_>,
) {
    let Some(ref csm) = renderer.gpu.csm_shadow else {
        return;
    };

    let cascade_count = csm.cascade_count.min(CSM_CASCADE_COUNT as u32);

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("CSM Shadow Layered"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &csm.array_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_pipeline(&renderer.gpu.pipelines.shadow_depth);
    let mut last_bound_mesh = None;

    for &i in ctx.solid_order {
        let dc = ctx.visible[i];
        let mesh_id = match ctx.mesh_handles[i] {
            Some(id) => id,
            None => continue,
        };

        // Upload all 4 cascade VPs as a uniform array
        renderer.gpu.shadow_pool.push_shadow_array(&[
            ShadowDrawUniforms { shadow_mvp: (ctx.csm_view_proj[0] * dc.model_matrix).to_cols_array_2d() },
            ShadowDrawUniforms { shadow_mvp: (ctx.csm_view_proj[1] * dc.model_matrix).to_cols_array_2d() },
            ShadowDrawUniforms { shadow_mvp: (ctx.csm_view_proj[2] * dc.model_matrix).to_cols_array_2d() },
            ShadowDrawUniforms { shadow_mvp: (ctx.csm_view_proj[3] * dc.model_matrix).to_cols_array_2d() },
        ]);

        // GPU culling via compute shader: precompute which cascades each draw is visible in.
        // For now, render into all 4 cascades (instanced draw).
        // Future: compute shader writes cascade_bitmask per draw, then multi-draw indirect.
        if let Some(offset) = renderer.gpu.shadow_pool.current_offset() {
            pass.set_bind_group(0, renderer.gpu.shadow_pool.bind_group(), &[offset]);
            // Draw instanced: instance_count = cascade_count
            // Vertex shader reads cascade VP from uniform array indexed by gl_InstanceIndex
            renderer.draw_mesh_instanced(&mut pass, mesh_id, cascade_count, &mut last_bound_mesh);
        }
    }
}
```

- [ ] **Step 3: 更新 shader — 顶点着色器支持 instanced cascade**

现有的 `shadow_depth.wgsl`（或等效）需要更新顶点着色器：
```wgsl
// shadow_depth.vert
@group(0) @binding(0) var<uniform> cascades: array<mat4x4f, 4>;

@vertex
fn main(
    @location(0) pos: vec3f,
    @builtin(instance_index) instance_idx: u32,
) -> @builtin(position) vec4f {
    return cascades[instance_idx] * vec4f(pos, 1.0);
}
```

**如果 shader 文件需要修改**，使用 `ShaderHotReload` 框架，确保现有机制兼容。

- [ ] **Step 4: 编译验证**

```bash
rtk cargo check -p rc3d-render
```

- [ ] **Step 5: 提交**

```bash
rtk git add crates/rc3d-render/src/render_passes/pass_shadow.rs \
    crates/rc3d-render/src/shadow_map.rs crates/rc3d-render/src/renderer.rs
rtk git commit -m "feat: CSM layered shadow rendering with texture array + instanced draw"
```

---

### Task 9: PBR 着色器特化

**Files:**
- Modify: `crates/rc3d-render/src/pipelines.rs:60-70`
- Modify: `crates/rc3d-render/src/shader_permutation.rs` (如存在)

- [ ] **Step 1: 定义 PbrFeatures bitflags**

在 `pipelines.rs` 顶部增加：
```rust
bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PbrFeatures: u32 {
        const HAS_ALBEDO_TEX    = 1 << 0;
        const HAS_NORMAL_TEX    = 1 << 1;
        const HAS_MR_TEX        = 1 << 2;
        const HAS_EMISSIVE_TEX  = 1 << 3;
        const HAS_OCCLUSION_TEX = 1 << 4;
        const HAS_IBL           = 1 << 5;
        const HAS_SHADOWS       = 1 << 6;
        const IS_TRANSPARENT    = 1 << 7;
    }
}

type PbrPipelineCache = lru::LruCache<PbrFeatures, wgpu::RenderPipeline>;
```

`lru` crate 检查：
```bash
rtk grep lru crates/rc3d-render/Cargo.toml
```
如不存在，增加 `lru = "0.12"`。

- [ ] **Step 2: 创建 get_or_create_pbr_pipeline 函数**

```rust
pub struct PbrVariantCache {
    cache: lru::LruCache<PbrFeatures, wgpu::RenderPipeline>,
    device: *const wgpu::Device, // unsafe but we only use it during init; or store Arc<Device>
    // ...
}

impl PbrVariantCache {
    pub fn new(max_variants: usize, device: &wgpu::Device) -> Self {
        Self {
            cache: lru::LruCache::new(max_variants.try_into().unwrap_or(16)),
            device,
        }
    }

    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        features: PbrFeatures,
        layout: &wgpu::PipelineLayout,
    ) -> &wgpu::RenderPipeline {
        self.cache.get_or_insert(features, || {
            let shader = self.build_shader_module(device, features);
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("PBR variant {:?}", features)),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[/* Vertex::desc() */],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[/* ... */],
                }),
                // ... depth_stencil, primitive, multisample ...
                ..Default::default()
            })
        })
    }

    fn build_shader_module(&self, device: &wgpu::Device, features: PbrFeatures) -> wgpu::ShaderModule {
        // Shader features as preprocessor defines
        let defines = format!(
            "HAS_ALBEDO_TEX={}\nHAS_NORMAL_TEX={}\nHAS_MR_TEX={}\nHAS_EMISSIVE_TEX={}\nHAS_OCCLUSION_TEX={}\nHAS_IBL={}\nHAS_SHADOWS={}\nIS_TRANSPARENT={}\n",
            features.contains(PbrFeatures::HAS_ALBEDO_TEX) as u32,
            features.contains(PbrFeatures::HAS_NORMAL_TEX) as u32,
            features.contains(PbrFeatures::HAS_MR_TEX) as u32,
            features.contains(PbrFeatures::HAS_EMISSIVE_TEX) as u32,
            features.contains(PbrFeatures::HAS_OCCLUSION_TEX) as u32,
            features.contains(PbrFeatures::HAS_IBL) as u32,
            features.contains(PbrFeatures::HAS_SHADOWS) as u32,
            features.contains(PbrFeatures::IS_TRANSPARENT) as u32,
        );
        let source = std::fs::read_to_string("crates/rc3d-render/src/shaders/pbr.wgsl").unwrap();
        let full = format!("{}\n{}", defines, source);

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("PBR shader {:?}", features)),
            source: wgpu::ShaderSource::Wgsl(full.into()),
        })
    }
}
```

- [ ] **Step 3: 在 PipelineSet 增加 pbr_variant_cache**

```rust
pub struct PipelineSet {
    // ... existing fields ...
    pub pbr_variant_cache: PbrVariantCache,
}
```

- [ ] **Step 4: 绘制时按材质选择 variant**

在 `draw_opaque.rs` 或等效位置，根据 DrawCall 的材质属性计算 `PbrFeatures` 并调用 `get_or_create`：

```rust
fn features_for_draw(dc: &DrawCall, has_ibl: bool, has_shadows: bool) -> PbrFeatures {
    let mut f = PbrFeatures::empty();
    if dc.albedo_path.is_some() { f |= PbrFeatures::HAS_ALBEDO_TEX; }
    if dc.normal_path.is_some() { f |= PbrFeatures::HAS_NORMAL_TEX; }
    if dc.metallic_roughness_path.is_some() { f |= PbrFeatures::HAS_MR_TEX; }
    if dc.emissive_path.is_some() { f |= PbrFeatures::HAS_EMISSIVE_TEX; }
    if dc.occlusion_path.is_some() { f |= PbrFeatures::HAS_OCCLUSION_TEX; }
    if has_ibl { f |= PbrFeatures::HAS_IBL; }
    if has_shadows { f |= PbrFeatures::HAS_SHADOWS; }
    if dc.alpha_mode != rc3d_scene::AlphaMode::Opaque { f |= PbrFeatures::IS_TRANSPARENT; }
    f
}
```

- [ ] **Step 5: 编译验证**

```bash
rtk cargo check -p rc3d-render
```

- [ ] **Step 6: 提交**

```bash
rtk git add crates/rc3d-render/src/pipelines.rs crates/rc3d-render/Cargo.toml
rtk git commit -m "feat: PBR shader variant cache with per-material feature specialization"
```

---

## Phase 4：增量 BVH + 纹理流送 + 自适应质量

### Task 10: 增量 BVH 更新

**Files:**
- Modify: `crates/rc3d-core/src/bvh.rs`
- Modify: `crates/rc3d-render/src/renderer_render.rs:62-67`

- [ ] **Step 1: 在 BVH 增加 incremental_update 方法**

在 `bvh.rs` 增加：
```rust
impl Bvh {
    /// Update specific leaf nodes' AABBs without full rebuild.
    /// `updates`: (leaf_index, new_aabb) pairs.
    /// If count > 50% of leaves, falls back to full rebuild.
    pub fn incremental_update(&mut self, updates: &[(usize, Aabb)], items: &[(Aabb, u32)]) {
        let total_leaves = items.len();
        if updates.len() as f64 > total_leaves as f64 * 0.5 {
            *self = Bvh::build(items);
            return;
        }

        for &(leaf_idx, new_aabb) in updates {
            if let Some(node_idx) = self.leaf_to_node(leaf_idx) {
                self.nodes[node_idx].aabb = new_aabb;
            }
        }

        // Bottom-up refit: propagate AABB changes upward
        let mut dirty_nodes: std::collections::HashSet<usize> = updates
            .iter()
            .filter_map(|(leaf, _)| self.leaf_to_node(*leaf))
            .collect();

        while !dirty_nodes.is_empty() {
            let mut next = std::collections::HashSet::new();
            for &node_idx in &dirty_nodes {
                if let Some(parent) = self.nodes[node_idx].parent {
                    let old_aabb = self.nodes[parent].aabb;
                    let left = self.nodes[parent].left.unwrap();
                    let right = self.nodes[parent].right.unwrap();
                    self.nodes[parent].aabb = self.nodes[left].aabb.union(&self.nodes[right].aabb);
                    if self.nodes[parent].aabb != old_aabb {
                        next.insert(parent);
                    }
                }
            }
            dirty_nodes = next;
        }
    }

    fn leaf_to_node(&self, leaf_idx: usize) -> Option<usize> {
        // Map leaf index (into items array) to BVH node index.
        // Depends on the BVH storage layout — if items are stored in leaf nodes
        // with a mapping table, use that; otherwise return leaf_idx directly.
        if leaf_idx < self.nodes.len() {
            Some(leaf_idx)
        } else {
            None
        }
    }
}
```

- [ ] **Step 2: 在 renderer_render 中使用增量 BVH**

将现有的：
```rust
let bvh = rc3d_core::Bvh::build(&bvh_items);
```
改为：
```rust
let bvh = if let Some(ref mut existing_bvh) = renderer.frame.cached_bvh {
    let updates: Vec<(usize, rc3d_core::Aabb)> = /* 收集 dirty 节点的 AABB 变更 */;
    existing_bvh.incremental_update(&updates, &bvh_items);
    existing_bvh.clone()
} else {
    let new_bvh = rc3d_core::Bvh::build(&bvh_items);
    renderer.frame.cached_bvh = Some(new_bvh.clone());
    new_bvh
};
```

在 `FrameState` 增加：
```rust
pub cached_bvh: Option<rc3d_core::Bvh>,
```

- [ ] **Step 3: 编译验证**

```bash
rtk cargo check -p rc3d-core -p rc3d-render
```

- [ ] **Step 4: 提交**

```bash
rtk git add crates/rc3d-core/src/bvh.rs crates/rc3d-render/src/renderer_render.rs \
    crates/rc3d-render/src/renderer_internals.rs
rtk git commit -m "feat: incremental BVH update with bottom-up AABB refit"
```

---

### Task 11: 纹理流送

**Files:**
- Create: `crates/rc3d-render/src/texture_streaming.rs`
- Modify: `crates/rc3d-render/src/renderer.rs`
- Modify: `crates/rc3d-render/src/lib.rs`

- [ ] **Step 1: 创建 texture_streaming.rs**

```rust
// crates/rc3d-render/src/texture_streaming.rs

use std::sync::Arc;
use std::thread;

/// A streaming texture that starts with a low-resolution placeholder
/// and asynchronously loads the full-resolution version.
pub struct StreamingTexture {
    pub gpu_texture: wgpu::Texture,
    pub gpu_view: wgpu::TextureView,
    pub full_res_loaded: bool,
    pending_load: Option<thread::JoinHandle<Option<Vec<u8>>>>,
}

pub struct TextureStreamer {
    pending: Vec<StreamingTexture>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl TextureStreamer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { pending: Vec::new(), device, queue }
    }

    /// Start streaming a texture: create 16x16 placeholder immediately,
    /// spawn a background thread to load the full image.
    pub fn request_texture(&mut self, path: &str) -> StreamingTexture {
        let (placeholder_tex, placeholder_view) = self.create_placeholder(&self.device);
        let path_owned = path.to_string();
        let handle = thread::spawn(move || {
            let img = image::open(&path_owned).ok()?;
            Some(img.into_rgba8().into_raw())
        });
        StreamingTexture {
            gpu_texture: placeholder_tex,
            gpu_view: placeholder_view,
            full_res_loaded: false,
            pending_load: Some(handle),
        }
    }

    fn create_placeholder(&self, device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
        let size = 16u32;
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("placeholder"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    /// Check for completed loads and upload full-resolution textures to GPU.
    pub fn poll_completed(&mut self) {
        let device = &*self.device;
        let queue = &*self.queue;
        self.pending.retain_mut(|st| {
            if st.full_res_loaded { return false; }
            if let Some(handle) = &st.pending_load {
                if handle.is_finished() {
                    if let Ok(Some(data)) = handle.take().unwrap().join() {
                        let (w, h) = (/* extract from data */ 512, 512); // placeholder
                        // TODO: proper size tracking. For now copy into the
                        // existing placeholder texture via queue.write_texture.
                        st.full_res_loaded = true;
                    }
                    false // Remove from pending
                } else {
                    true // Still loading
                }
            } else {
                false
            }
        });
    }
}
```

- [ ] **Step 2: 在 Renderer 中初始化 TextureStreamer**

```rust
pub texture_streamer: crate::texture_streaming::TextureStreamer,
```

每帧调用：
```rust
self.texture_streamer.poll_completed();
```

- [ ] **Step 3: 在 lib.rs 中导出**

```rust
pub mod texture_streaming;
```

- [ ] **Step 4: 编译验证**

```bash
rtk cargo check -p rc3d-render
```

- [ ] **Step 5: 提交**

```bash
rtk git add crates/rc3d-render/src/texture_streaming.rs crates/rc3d-render/src/renderer.rs \
    crates/rc3d-render/src/lib.rs
rtk git commit -m "feat: texture streaming with placeholder-first async loading"
```

---

### Task 12: 自适应质量精细化

**Files:**
- Modify: `crates/rc3d-render/src/adaptive_quality.rs`

- [ ] **Step 1: 增加更多质量阶梯**

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd)]
pub enum QualityLevel {
    Ultra,   // ≥60fps — full effects, 4 cascades @ 2048²
    High,    // 45-60fps — full effects, 4 cascades @ 2048²
    Medium,  // 30-45fps — reduced SSAO, 2 cascades @ 1024²
    Low,     // 20-30fps — no SSAO/Bloom, 1 cascade @ 512²
    Minimal, // <20fps — no post, flat shading, 1 cascade @ 256²
}

pub struct AdaptiveController {
    pub current: QualityLevel,
    /// Frame time EMA (ms)
    frame_time_ema: f32,
    /// Consecutive frames above/below threshold before switching
    consecutive_over: u32,
    consecutive_under: u32,
}

impl AdaptiveController {
    pub fn new() -> Self {
        Self {
            current: QualityLevel::Ultra,
            frame_time_ema: 16.6,
            consecutive_over: 0,
            consecutive_under: 0,
        }
    }

    pub fn update(&mut self, frame_time_ms: f32) -> QualityLevel {
        // EMA smoothing
        self.frame_time_ema = self.frame_time_ema * 0.9 + frame_time_ms * 0.1;

        let target_fps = self.fps_for_level(self.current);
        let current_frame_time = 1000.0 / target_fps;

        if self.frame_time_ema > current_frame_time * 1.2 {
            self.consecutive_over += 1;
            self.consecutive_under = 0;
        } else if self.frame_time_ema < current_frame_time * 0.8 {
            self.consecutive_under += 1;
            self.consecutive_over = 0;
        } else {
            self.consecutive_over = 0;
            self.consecutive_under = 0;
        }

        // Downgrade after 5 consecutive slow frames
        if self.consecutive_over >= 5 {
            self.current = self.current.downgrade();
            self.consecutive_over = 0;
        }
        // Upgrade after 30 consecutive fast frames (hysteresis)
        if self.consecutive_under >= 30 {
            self.current = self.current.upgrade();
            self.consecutive_under = 0;
        }
        self.current
    }

    fn fps_for_level(&self, level: QualityLevel) -> f32 {
        match level {
            QualityLevel::Ultra => 60.0,
            QualityLevel::High => 45.0,
            QualityLevel::Medium => 30.0,
            QualityLevel::Low => 20.0,
            QualityLevel::Minimal => 15.0,
        }
    }
}

impl QualityLevel {
    fn downgrade(self) -> Self {
        match self {
            QualityLevel::Ultra => QualityLevel::High,
            QualityLevel::High => QualityLevel::Medium,
            QualityLevel::Medium => QualityLevel::Low,
            QualityLevel::Low => QualityLevel::Minimal,
            QualityLevel::Minimal => QualityLevel::Minimal,
        }
    }
    fn upgrade(self) -> Self {
        match self {
            QualityLevel::Ultra => QualityLevel::Ultra,
            QualityLevel::High => QualityLevel::Ultra,
            QualityLevel::Medium => QualityLevel::High,
            QualityLevel::Low => QualityLevel::Medium,
            QualityLevel::Minimal => QualityLevel::Low,
        }
    }
}
```

- [ ] **Step 2: 编译验证**

```bash
rtk cargo check -p rc3d-render
```

- [ ] **Step 3: 提交**

```bash
rtk git add crates/rc3d-render/src/adaptive_quality.rs
rtk git commit -m "feat: refined adaptive quality with 5-level staircase + EMA hysteresis"
```

---

## Phase 5：最终验证

### Task 13: 全量编译 + 测试 + 总结

- [ ] **Step 1: 全工作区编译检查**

```bash
rtk cargo check --workspace --all-targets
```
预期：零 error、零 warning

- [ ] **Step 2: 全测试套件**

```bash
rtk cargo test --workspace
```
预期：所有测试通过（≥175 测试）

- [ ] **Step 3: 确认所有 example 可编译**

```bash
rtk cargo build -p rc3d-app --examples
```
预期：37 个 example 编译通过

- [ ] **Step 4: 性能回归检查**

使用 `--profile` 标志（如已添加 CLI）运行一个 example，确认 frame timing 输出格式正确：
```bash
rtk cargo run -p rc3d-app --example sphere_viewer -- --profile
```

- [ ] **Step 5: 提交**

```bash
rtk git add -A
rtk git commit -m "chore: final verification pass — full workspace check + test suite"
```

---

## 验证清单

全部 Task 完成后执行：

- [ ] `cargo check --workspace --all-targets` — 零 error、零 warning
- [ ] `cargo test --workspace` — 所有测试通过
- [ ] `cargo build -p rc3d-app --examples` — 所有 example 编译通过
- [ ] 静态场景 ≥60fps（1080p, 1000 nodes） — 人工验证
- [ ] `--profile` 输出显示 CPU traversal <0.5ms（静态场景）
- [ ] DrawCall 分配从每帧 ~200+ 降至 <20
- [ ] CSM shadow render pass 数从 4 降至 1

## 不在此范围

- 帧图（Frame Graph）全量实现
- Bindless rendering
- Mesh shader / meshlet
- Ray tracing
- U3D/PRC 真三维 PDF
- FBX 7.5+ 版本
