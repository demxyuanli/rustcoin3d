## Context

rustcoin3d 实现了 Coin3D 风格的低层场景图 API：
- `SceneGraph` + `NodeData`（47+ 变体）提供完整的场景图构建能力
- `rc3d-mesh` 已有内部拓扑结构 `TriangleMesh`（边/面邻接、边界/锐边/轮廓边计算）
- `rc3d-render` 有完整的 25-pass 渲染管线（阴影、HZB、PBR、后处理等）
- `rc3d-engine` 提供 `Engine trait` 用于场景行为（动画、物理、传感器）

但以上能力通过低层 API 暴露，开发者需要：
- 理解 Separator/StateStack/dirty flag 才能正确构建场景
- 手写递归遍历做节点查询（示例中 7 处重复）
- 渲染配置散落在 App builder、硬编码默认值、GPU buffer 直写

目标：新增两个 facade crate，将现有能力通过声明式、可组合的高层接口暴露，原引擎作为不可修改的静态 SDK。

## Goals / Non-Goals

**Goals:**
- 场景开发者只需描述"有什么"（Shape/Material/Light/Camera），不碰 NodeData/SceneGraph
- 渲染开发者只需声明"要什么效果"（Shadow/SSAO/SSR），不碰 Renderer/PassContext
- Shape → TriangleMesh 的编译作为独立步骤，支持验证和查询
- 原引擎 16 个 crate 的代码不动

**Non-Goals:**
- 不改变现有示例或 CLI editor
- 不重新实现渲染管线（复用 rc3d-render 所有 pass）
- 不处理 GUI/编辑器集成（那是后续工作）
- 不处理动画系统重构（Engine trait 保持现有接口）

## Decisions

### D1: 新增 crate 而非修改现有 crate

**选**：新建 `rc3d-scene-api` + `rc3d-effects`，通过 `rc3d-app` 桥接。

**弃**：在 rc3d-scene 或 rc3d-app 内新增模块。理由：原 crate 应作为不可修改的 SDK 分发；新 crate 明确区分"对外接口"和"内部实现"。Cargo 的 workspace dependency 天然支持这种分层。

### D2: Shape trait + Builder 模式

**选**：`trait Shape` + 每个实现带 `at()`/`scale()`/`rotate()`/`material()` builder 方法。

```rust
pub trait Shape {
    fn compile(&self) -> TriangleMesh;
    fn aabb(&self) -> Aabb;
}

impl Shape for Cube {
    fn compile(&self) -> TriangleMesh { tessellate_cube(self.width, self.height, self.depth) }
}

// Builder 方法在 impl 块或 extension trait
pub struct Cube {
    pub width: f32, pub height: f32, pub depth: f32,
    transform: Option<Transform>,
    material: Option<Material>,
}
```

**弃 A**：Bevy 式 ECS 组件。需要 ECS 基础设施，与 Coin3D 场景图哲学冲突。

**弃 B**：纯函数式 `scene.add_cube(pos, size, mat)`。参数爆炸（Cube 有 10+ 可配置项），builder 模式更清晰。

### D3: Scene::add 自动包裹 Separator

**选**：每个 `scene.add(thing)` 内部自动创建 `Separator → [Transform] → [Material] → ShapeNode` 子树。`Group` 显式创建父 Separator。

**弃**：暴露 Separator 给用户。这是 Coin3D 最大的 DX 痛点——忘记 Separator 导致变换泄漏。rustcoin3d 示例中已出现此 bug（CLAUDE.md 记录的 Separator transform propagation 修复）。

### D4: Geometry 作为独立编译产物

**选**：`Geometry` 结构体持有编译后的 `TriangleMesh` + 验证状态。

```rust
pub struct Geometry {
    mesh: TriangleMesh,
    validation: Option<ValidationResult>,
}
```

场景编译时所有 Shape 编译为 `Vec<Geometry>`，独立于渲染使用。`emit_cached_shape` 的缓存放到这里，`render_action.rs` 无需修改。

**弃**：Shape 直接转换为 GPU buffer。当前这样做（`emit_cached_shape` 内部调用 tessellate + triangle_buffers），但丢失了拓扑信息——边线渲染需要拓扑但无法跨帧复用。

### D5: EffectGraph 作为编译目标

**选**：`RenderConfig` 是用户接口，`EffectGraph` 是编译后的内部 DAG。编译时解析效果依赖：
- `SSR` → 自动插入 `HZB Build`
- `CSM 4 cascade` → 自动插入 `4 × DepthPrePass`
- `SSAO` → 自动分配 AO texture

编译结果注入 `rc3d-render::PassContext` 的布尔标志和参数。

**弃**：让用户手动排序 pass 或管理依赖。25 个 pass 的内部顺序不应该暴露。

### D6: 导入几何通过 from_raw 不耦合 IO

**选**：`Mesh::from_raw(positions, indices, normals?, texcoords?)` 接受外部解析结果。文件加载（STL/OBJ/glTF）由 rc3d-io 在外部完成。

**弃**：`Mesh::import(path)` 内置 IO。增加不必要的 crate 依赖，且阻塞式文件 IO 不适合所有场景。

## Risks / Trade-offs

- **Shape trait 对象安全**：如果 Shape 需要 `dyn Shape`（例如场景查询返回 `Vec<&dyn Shape>`），需要确保 trait 是 object-safe。当前 `fn compile(&self) -> TriangleMesh` 是安全的。→ 如需扩展，提供 `fn as_any(&self) -> &dyn Any`。

- **RenderConfig 向后兼容**：现有 App::new() 的 builder 方法（`with_hdr_post_processing` 等）需要保持可用。→ RenderConfig 作为新入口，旧方法标记 deprecated 但不删除。

- **TriangleMesh clone 成本**：Geometry 持有 TriangleMesh，大数据集下 clone 可能重。→ Shape::compile() 返回 TriangleMesh，Geometry 消费所有权；如需要共享，后续加 Arc<TriangleMesh>。

- **EffectGraph 和现有 PassContext 的映射**：PassContext 有 50+ 布尔/枚举字段控制 pass。EffectGraph 编译时逐个映射。→ 需要 integration test 确保所有组合正确。
