# 渲染引擎性能重构 — 设计文档

> 数据驱动、CPU+GPU 均衡、方案 B：数据导向重构 + 热路径外科手术

## 架构概览

```
场景图                       FlatDrawCache                       GPU
┌──────────────┐             ┌──────────────────┐             ┌──────────┐
│ dirty 标记    │ ─变更→     │ GpuDrawData[]    │ ──ring buf→│ uniform  │
│ 增量遍历      │            │ CachedDrawMeta[] │            │ buffers  │
│ traversal     │            │ EffectCommands   │            │ passes   │
└──────────────┘             │ IncrementalBvh   │            └──────────┘
                             │ draw_groups      │
                             └──────────────────┘
                                      ↑
                              PassDag 排序调度
                              (CSM→Solid→Effect→Post→HUD)
```

核心原则：**静态数据计算一次、缓存复用；动态数据增量更新；扁平连续内存；按依赖图并行调度。**

---

## 一、数据流架构：FlatDrawCache + 脏标记

### 问题
- 每帧递归遍历全图 O(N)，即使 99% 节点未变
- DrawCall 每帧重新分配 Vec + Arc clone
- BVH 每帧从零重建 O(N log N)

### 设计

**FlatDrawCache** — 替代 RenderCollector 的临时 Vec<DrawCall>：
```
opaque_draws: Vec<GpuDrawData>       // 连续内存，GPU uniform 直接映射
transparent_draws: Vec<GpuDrawData>
metadata: Vec<CachedDrawMetadata>    // 冷数据，仅脏时更新
effect_commands: EffectCommands
bvh: IncrementalBvh                 // 增量更新
groups_dirty: bool                  // 排序仅脏时重排
```

**脏标记传播** — NodeEntry 增加 `dirty_flags: DirtyFlags`（TRANSFORM | MATERIAL | GEOMETRY | CHILDREN | REMOVED）。父节点变更时自动置脏子节点。静态子树标记 `FROZEN`，遍历时跳过。

**缓存失效策略**：
- 无脏节点 → 直接复用 FlatDrawCache，零分配
- 少量脏节点 → 仅重遍历脏子树，更新缓存对应区间
- 超过 50% 脏节点 → 全量重建

---

## 二、DrawCall 瘦身 + 内存布局

### 问题
- DrawCall ~624B + 堆分配碎片（4×Arc<Vec> + 4×Arc<str> + 5×[MAX_LIGHTS]光照数组）
- 固定管线遗留字段（diffuse/ambient/specular/shininess）无实际使用

### 设计

**拆分为热/冷数据：**

GpuDrawData（64B, cache-line 对齐）：
```
model_matrix: [[f32;4];4]    64B
material_id: u32             → 索引到全局材质缓冲区
light_set_id: u32            → 索引到全局光照集合（替代 320B 光照数组）
vertex_range: (u32,u32)      → 全局顶点缓冲区的 offset/count
index_range: (u32,u32)       → 全局索引缓冲区的 offset/count
flags: u32                   → HAS_EDGES|TRANSPARENT|OVERLAY|...
```

CachedDrawMetadata（~80B，仅脏时写）：
```
mesh_hash: u64               → GPU mesh 去重 key
bvh_node_id: Option<u32>
material_params: MaterialUniform
albedo_tex_id: u16           → 索引到全局纹理路径表
normal_tex_id: u16
mr_tex_id: u16
emissive_tex_id: u16
```

**全局纹理路径表** — 启动时构建，DrawCall 存 u16 索引替代 Arc<str>：

**全局 Mesh Buffer** — 统一顶点/索引大缓冲区 + free-list sub-allocation，替代每个网格独立的 Arc<Vec<T>>：

**移除固定管线遗留字段** — diffuse_color, ambient_color, specular_color, shininess, emissive_intensity（合并到 MaterialUniform）

### 收益
1000 draw calls 场景：1MB+ → 144KB 连续内存，零堆分配碎片

---

## 三、CSM 阴影：Layered Rendering

### 问题
4 个级联 × 全部实体几何体 = 4 次完整场景提交，每次一个独立的 depth-only render pass

### 设计
- 2D Texture Array（2048×2048×4 layers）替代 4 个独立纹理
- 单次 draw instanced（instance_count=4），顶点着色器按 `gl_InstanceIndex` 选择 cascade VP
- GPU compute shader 预剔除不可见 draw → 每个 cascade 只绘制该级联可见的几何体

---

## 四、着色器特化

### 问题
PBR shader 始终以最大特性集编译（全部纹理绑定），纯色材质也执行无意义的 textureSample

### 设计
- `PbrFeatures` bitflags：HAS_ALBEDO_TEX | HAS_NORMAL_TEX | HAS_MR_TEX | HAS_EMISSIVE_TEX | HAS_OCCLUSION_TEX | HAS_IBL | HAS_SHADOWS | IS_TRANSPARENT
- 按材质属性运行时选择 shader variant
- LRU 缓存（最多 16 个 variant）→ 首次编译后命中缓存

---

## 五、Pass 调度优化

### 问题
所有 pass 严格串行；post-process 7 个全屏 pass 各自独立

### 设计
- PassDag 依赖图拓扑排序 → 识别可并行 pass 组
- SSAO 和 Fog 无依赖 → 两个 encoder 并行录制 → `queue.submit([enc1, enc2])`
- Post-process 合并：SSR+Fog+DoF → Bloom+SSAO（并行 compute）→ TAA+Tonemap+FXAA
- 双缓冲命令录制：上一帧 GPU 执行时 CPU 录制下一帧

---

## 六、渲染遍历并行化

### 问题
`traverse_node` 单线程递归 + `&mut self` 排他借用，无法并行

### 设计
**两阶段流水线：**
1. 脏子树并行遍历（rayon，最多 4 线程）→ thread-local TraversalChunk
2. 主线程合并 → FlatDrawCache → staging belt 一次上传

**关键约束解决：**
- Switch 节点：遍历前计算 which_child，只把活跃分支交给线程
- Separator 状态：TraversalState 实现 Clone（数据量小）
- 跨子树引用：全部用索引（material_id, light_set_id），跨线程无冲突

---

## 七、增量 BVH

- 场景变更仅重建受影响的叶节点，内节点自底向上重算 AABB
- 静态子树 BVH 冻结
- 50%+ 节点脏时退化为全量重建

---

## 八、Profiling 基础设施

- GPU：TimestampQuery 在关键 pass（shadow/solid/effect/post/HUD）插桩
- CPU：tracing span 覆盖遍历/缓存更新/BVH/上传/提交
- 帧时间分解 → FrameStats 结构化输出 + `--profile` CLI 标志

---

## 九、纹理流送 + 自适应质量

**纹理流送：**
- 首次加载低分辨率占位 + 后台异步上传全分辨率
- mip 链 CPU 降采样 + queue.write_texture 批量写入

**自适应质量：**
- 基于帧时间滑动窗口动态调整：CSM 级联数/分辨率、SSAO 半径、TAA 采样数
- QualityLevel：Ultra(>60fps) / High(30-60) / Medium(16-30) / Low(<16)

---

## 文件变更清单

| 文件 | 改动类型 | 描述 |
|------|----------|------|
| `crates/rc3d-render/src/flat_draw_cache.rs` | **新增** | FlatDrawCache 结构体 + 增量更新逻辑 |
| `crates/rc3d-render/src/render_action.rs` | 重构 | DrawCall → GpuDrawData + CachedDrawMetadata；traverse_node → 纯函数 |
| `crates/rc3d-scene/src/scene_graph.rs` | 修改 | NodeEntry 增加 dirty_flags |
| `crates/rc3d-render/src/parallel_traversal.rs` | **新增** | rayon 并行遍历 + staging belt 上传 |
| `crates/rc3d-render/src/pass_graph.rs` | **新增** | PassDag 依赖图 + 拓扑排序调度 |
| `crates/rc3d-render/src/render_passes.rs` | 重构 | pass 调度改用 PassDag；post-process 合并 |
| `crates/rc3d-render/src/pass_shadow.rs` | 重构 | CSM layered rendering + GPU culling |
| `crates/rc3d-render/src/pipelines.rs` | 修改 | PbrFeatures variant 缓存 + LRU |
| `crates/rc3d-render/src/gpu_resource.rs` | 重构 | 全局 MeshBuffer + free-list allocator；纹理路径表 |
| `crates/rc3d-render/src/global_tables.rs` | **新增** | TexturePathTable, GlobalLightBuffer |
| `crates/rc3d-render/src/bvh.rs` | 重构 | 增量 BVH 更新 |
| `crates/rc3d-render/src/profiler.rs` | **新增** | GpuTimer + tracing span + FrameStats |
| `crates/rc3d-render/src/texture_streaming.rs` | **新增** | 纹理流送：占位 + 后台加载 + 替换 |
| `crates/rc3d-render/src/adaptive_quality.rs` | 修改 | QualityLevel 枚举 + 基于帧时间的动态调节 |
| `crates/rc3d-render/src/lib.rs` | 修改 | 模块导出更新 |
| `crates/rc3d-app/src/app/event_handler.rs` | 修改 | 适配新数据流接口 |

总计：5 个新文件，~8 个文件重构，3 个文件小改。

## 不在此次范围

- 帧图（Frame Graph）全量实现（wgpu 生态无成熟库）
- Bindless rendering（wgpu 支持有限）
- Mesh shader / meshlet（wgpu 未稳定）
- Ray tracing（wgpu 不支持 DXR/VKRT）

## 验证标准

1. `cargo check --workspace --all-targets` 零 warning
2. `cargo test --workspace` 全通过
3. 静态场景 ≥60fps（1080p, 1000 nodes）
4. 每帧 CPU 分配次数从 ~200+ 降至 <20
5. `--profile` 输出显示遍历时间 <0.5ms（静态场景）
6. CSM shadow pass 数从 4 降至 1
