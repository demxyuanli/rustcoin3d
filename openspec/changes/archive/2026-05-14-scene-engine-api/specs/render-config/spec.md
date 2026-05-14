## ADDED Requirements

### Requirement: RenderConfig builder
系统 SHALL 提供 `RenderConfig` 结构作为渲染效果的声明式配置入口。

#### Scenario: Default render config
- **WHEN** 用户调用 `RenderConfig::default()`
- **THEN** 生成 Shaded 显示模式、1 cascade 1×1 阴影、无后处理的配置（与当前 Engine 默认行为一致）

#### Scenario: High quality preset
- **WHEN** 用户调用 `RenderConfig::high_quality()`
- **THEN** 生成 Shaded、4 cascade 2048×2048 CSM 阴影、SSAO+SSR+Bloom+TAA+Tonemap 全部启用的配置

#### Scenario: Build render config
- **WHEN** 用户调用 `render_config.build()`
- **THEN** 返回编译后的 `EffectGraph`，可以直接注入 Renderer

---

### Requirement: Shadow configuration
系统 SHALL 支持通过 `RenderConfig` 配置级联阴影映射（CSM）。

#### Scenario: CSM with 4 cascades
- **WHEN** 用户调用 `RenderConfig::new().enable(Shadow::CSM { cascade_count: 4, resolution: 2048, soft: true })`
- **THEN** 编译后的 EffectGraph 包含 4 个级联的 shadow depth pass，每个使用 2048×2048 纹理

#### Scenario: Disable shadows
- **WHEN** 用户在 RenderConfig 中不调用 `.enable(Shadow::...)`
- **THEN** 渲染管线不执行任何阴影 pass

---

### Requirement: Post effect composition
系统 SHALL 支持按需组合后处理效果，效果间依赖自动解析。

#### Scenario: SSAO + SSR requires HZB
- **WHEN** 用户同时启用 SSAO 和 SSR
- **THEN** EffectGraph 自动插入 HZB Build pass，且确保在 SSAO/SSR 之前执行

#### Scenario: Bloom without HDR
- **WHEN** 用户启用 Bloom 但未启用 HDR
- **THEN** 系统发出警告（Bloom 在 LDR 下效果有限），但仍允许执行

#### Scenario: Selective post effects
- **WHEN** 用户仅启用 `PostEffect::Tonemap` 和 `PostEffect::TAA`
- **THEN** 仅 Tonemap 和 TAA resolve 两个后处理 pass 被执行，其他 pass（SSAO/SSR/DOF/MotionBlur/Bloom/VolumetricFog）被跳过

---

### Requirement: Display mode
系统 SHALL 支持通过 `RenderConfig` 设置全局显示模式。

#### Scenario: Wireframe mode
- **WHEN** 用户调用 `RenderConfig::new().display_mode(DisplayMode::Wireframe)`
- **THEN** 所有几何体以线框模式渲染

#### Scenario: Shaded with edges
- **WHEN** 用户调用 `RenderConfig::new().display_mode(DisplayMode::ShadedWithEdges)`
- **THEN** 几何体以 PBR 着色 + 特征边（边界+锐边）叠加渲染

---

### Requirement: EffectGraph compilation
系统 SHALL 将 `RenderConfig` 编译为 `EffectGraph`——一个有序的、带依赖解析的渲染 pass DAG。

#### Scenario: Compile to pass context
- **WHEN** 调用 `effect_graph.apply_to(pass_context)`
- **THEN** PassContext 的所有相关布尔标志和参数被正确设置（`run_shadow_pass`、`enable_ssao`、`cascade_count` 等）

#### Scenario: Effect ordering is correct
- **WHEN** EffectGraph 包含 SSAO + SSR + Bloom + Tonemap + TAA
- **THEN** Pass 执行顺序为 SSAO → SSR → Bloom → Tonemap → TAA（按 post-processing 链的标准顺序）

---

### Requirement: App integration
系统 SHALL 在 `rc3d-app` 中提供 `App::new(scene).with_effects(render_config)` 桥接入口。

#### Scenario: Full application with effects
- **WHEN** 用户调用 `App::new(scene).with_effects(render_config).with_camera_controller(ctrl).run()`
- **THEN** Scene 编译为 SceneGraph，RenderConfig 编译为 EffectGraph，两者注入 App 并启动渲染循环

#### Scenario: Backward compatibility
- **WHEN** 用户使用旧 API `App::new(graph).with_hdr_post_processing(true).run()`
- **THEN** 行为不变，旧 API 不被移除
