## Why

rustcoin3d 当前暴露的是 Coin3D 风格的底层场景图 API（NodeData 枚举 47+ 变体，手动 Separator/Transform/Material 组装），开发一个简单场景需要 40+ 行样板代码，且需要深入理解 StateStack、dirty flag、traversal 等内部机制。渲染管线配置分散在 App builder、硬编码默认值、GPU buffer 直写等 4 处，不可组合。引擎应作为静态 SDK 分发，场景开发者通过公开接口完成所有操作，不接触内部 NodeData/SceneGraph。

## What Changes

- **新增 `rc3d-scene-api` crate**：提供声明式场景构建 DSL。Shape trait（Cube、Sphere、Mesh 等）、Material builder、Light/Camera 类型、Group 分组、Query 查询、Geometry 编译层。内部下沉为 SceneGraph + NodeData，用户不可见。
- **新增 `rc3d-effects` crate**：提供可组合的渲染效果配置。RenderConfig 声明式 API，EffectGraph 编译为 pass 执行列表。Shadow::CSM 可配置级联参数，PostEffect 枚举（SSAO/SSR/Bloom/TAA/Tonemap）按需启用。
- **扩展 `rc3d-app` 桥接层**：新增 `App::new(scene).with_effects(render_config)` 入口，将 Scene 编译为 SceneGraph、EffectGraph 编译为 PassContext。
- **不修改** rc3d-scene、rc3d-render、rc3d-mesh、rc3d-core、rc3d-engine 等现有 crate 的任何内部代码。

## Capabilities

### New Capabilities

- `scene-building`: 声明式场景构建 — Scene 容器、Shape trait、Material builder、Light、Camera、Group、Query
- `geometry-compile`: 几何编译 — Shape 参数化描述到内部 TriangleMesh 的编译、验证、AABB 查询
- `render-config`: 渲染效果组合 — RenderConfig 声明式配置、EffectGraph、Shadow 可配、PostEffect 按需组合

### Modified Capabilities

（无，本次不修改现有 capability 的 spec 级需求）

## Impact

- 新增 crate：`rc3d-scene-api`、`rc3d-effects`（2 个 workspace member）
- 小幅修改 crate：`rc3d-app`（新增 `scene_api` 和 `effects` 桥接模块，不修改现有 public API）
- 现有示例：不受影响，仍可用旧 API
- 新示例：待添加，展示新 API 的完整开发流程
