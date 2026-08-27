# rustcoin3d vs Coin3D / HOOPS：工业可视化引擎差距分析与阶段路线图

## 一、当前状态评估

### 强项（已超越工业引擎的领域）

| 能力 | 状态 | 对标 |
|------|------|------|
| **GPU-driven meshlet 渲染** + HZB 遮挡剔除 | 已实现，先进水平 | Coin3D 无此能力；HOOPS Visualize 2023+ 有类似机制 |
| **CSM 方向光阴影 + Omni 点光源阴影** | 已实现，完整 | Coin3D SoShadowGroup 为 CPU 端实现 |
| **HDR 后处理管线** (TAA/SSR/DOF/MotionBlur/Bloom/SSAO/ColorGrading) | 已实现 | 游戏引擎级别；工业引擎通常不需要此复杂度 |
| **Clustered Forward 光照** (256+ lights GPU culling) | 已实现 | 工业引擎通常有更少动态光源 |
| **PBR 材质** (metallic-roughness + IBL) | 已实现 | HOOPS 2022+ 引入 PBR；Coin3D 无 |
| **骨骼动画 + GPU Skinning compute shader** | 已实现 | 工业引擎非核心需求，但有则加分 |
| **Rapier3D 物理集成** | 已实现 | 工业引擎偶有刚体模拟 |
| **meshopt LOD 生成** + 渐进式流式加载 | 已实现 | `LodNode.range_scale` 按段缩放距离阈值 |
| **Shader 变体系统** (WGSL 预处理器 + 缓存) | 已实现 | 构建时优化，减少运行时分支 |
| **Open Inventor (.iv) 导入/导出** | 已实现 | 与 Coin3D 原生格式兼容 |

### 核心差距总览

与 Coin3D / HOOPS 对比，**渲染管线已超越**；**交互与工作流**已从「几乎空白」进入**部分落地**（见下表 **Implemented / Partial / Missing**），产品与 Coin3D 级编辑器仍差在场景图传感器、全量编辑工具与 CAD 格式链。

### 能力对照：Implemented / Partial / Missing

| 区域 | Implemented | Partial | Missing |
|------|-------------|---------|---------|
| Tier 0 编辑器 | 多视口、四视图包、Gizmo 库 + Engine 绘制/拖拽、`Engine::handle_window_event` 统一示例输入 | 场景图 EventCallback 消费仍浅（命中后无节点级拦截） | — |
| Tier 1 工业 | 透明排序、stencil 选择/包围盒、ScenePath/Search、剖面节点+GPU 裁剪、框选（frustum）、帽面+剖面线 | 剖面拖动手柄、OIT、离屏与主路径合一 | 套索、X-ray 全场景 |
| Tier 2 场景图 | Switch/MultiCopy/LOD 节点、Text2/3、EventCallback、结构化拾取、Engine 连接图 + SoEngine 族、跨节点 Field 图 | Field 描述符、面索引拾取、传感器 API | — |
| Tier 3 高级 | Undo 命令栈骨架、Measurement/Markup 数据节点 | — | 布尔/STEP/网格修复/立体等（**见 §八 Backlog**） |

---

## 二、差距详细清单（按优先级排列）

| # | 差距 | 状态 (2026) | 说明 |
|---|------|-------------|------|
| T0-1 | **Manipulator/Gizmo** | Done | `TransformManip` / `Dragger` 场景图节点 + `Engine.gizmo` 每帧同步；无选区时显示首个启用操纵器；子 dragger 过滤手柄 |
| T0-2 | **多视口** | Done | `ViewportLayout` Quad/分屏 + `Engine::apply_standard_quad_views` 前/侧/顶/透视相机包 |
| T0-3 | **场景图事件路由** | Done | `Engine::handle_window_event`：`HandleEventAction` + 相机 + 单击拾取；键盘/滚轮非 pick 全树；指针 view·proj 取光标下视口（viewport-local）；`run_example` / editor / annotation_edit 共用 |
| T0-4 | **相机与视口** | Done | `ViewportCameraSet` + 默认四视图包（`vp.Top/Front/Right/Persp`）；拟合、正交/透视、`ViewPreset` |

| # | 差距 | 状态 (2026) | 说明 |
|---|------|-------------|------|
| T1-1 | **透明排序** | Done | WBOIT 默认开启（LDR/HDR）；`Engine::set_wboit(false)` 回退画家算法 |
| T1-2 | **交互式剖面** | Partial | `SectionPlane` + 应用内拖动法向偏移（manip 模式） |
| T1-3 | **注释/标注/尺寸** | Partial | `Measurement`/`Markup` 节点；无完整 GD&T 工具 |
| T1-4 | **选择高亮** | Partial | Stencil/包围盒；框选已加，套索/过滤待办 |
| T1-5 | **LOD 场景图** | Done | `LodNode` 距离切层 + `range_scale` 按段缩放（`set_lod_range_scale`） |
| T1-6 | **离屏渲染** | Partial | `OffscreenTarget` + 与主渲染器同尺寸的截图桥接 API |
| T1-7 | **路径系统** | Partial | `ScenePath`/`SearchAction`/`GetMatrixAction` |

（Tier 2/3 细分项同上方总表与 §八。）

### Tier 2 / Tier 3 — 标题保留

*（原 Coin3D/HOOPS 对标列已并入「能力对照」表；详细条目未删除需求，仅压缩重复。）*

---

## 三、阶段路线图（进度摘要）

- **Phase 0**：Gizmo/多视口/事件/视口相机 — `handle_window_event` 已统一示例输入；持续打磨「节点级」操纵器。
- **Phase 1**：透明/剖面/高亮/LOD/Path/框选/拟合 — WBOIT 已落地；帽面、X-ray 为后续。
- **Phase 2**：Field 传感器、Engine 扩展、SoDetail 级拾取 — 已启动（传感器模块 + 三角索引）。
- **Phase 3**：见 **§八 Backlog**，不占用默认迭代。

---

## 四、TODO 执行清单（与仓库同步，2026）

### Phase 0

- [x] **`crates/rc3d-gizmo/`** — Transform Gizmo 库
- [x] **多视口** — `Viewport`、Quad、活动边框、split fractions、可拖拽分割条；`apply_standard_quad_views` 前/侧/顶/透视预设
- [x] **HandleEventAction** + `EventCallback` + 非鼠标遍历；编辑器指针与滚轮经光标命中视口的 view·proj 派发
- [x] **示例输入统一** — `Engine::handle_window_event`（`feed_input` + `dispatch_routed_event`）；`run_example` / `run_example_with_hooks` / editor / annotation_edit；`stl_diagnostic_full_test` 无交互循环
- [x] **视口级相机** — `ViewportCameraSet`、per-viewport controller、拟合选中、默认四视图包

### Phase 1

- [x] **透明** — WBOIT 默认（LDR/HDR）；`set_wboit(false)` 回退画家算法
- [x] **剖面** — 节点 + GPU clip + 应用内 manip（偏移）
- [x] **剖面线** — `SectionPlaneNode.hatch_*` 程序化 ANSI/ISO 交叉线（`section_caps`）
- [x] **高亮** — stencil + bbox；框选
- [x] **Isolate/Ghost** — 选中保持着色，未选中半透明（`set_ghost_unselected`；空选择为 no-op）
- [x] **边分类** — `EdgeStyle` Hard / Perimeter / Adjacent / Silhouette / Full；`Crease` = hard+perimeter
- [x] **HiddenLine Fast HLR** — 暗填充 + 可见折边 + 反转深度虚线被挡边（非解析 HLR）
- [x] **命名 Visual Style** — `VisualStyleLibrary` 可注册、可套到 Separator 子树（`triangle` / View 菜单）
- [x] **PMI 语义** — `AnnotationSetNode.pmi` 绑定命名节点/面/边 + 公差；`bind_pmi` / JSON sidecar（`gdt_demo`）。无 STEP AP242 解析器
- [x] **LOD** — 节点 + 自动距离切换 + `LodNode.range_scale` 按段缩放
- [x] **矢量 Hidden Line SVG** — `Engine::export_hidden_line_svg`（Fast HLR 边的矢量硬拷贝，非 HOOPS HIO 解析 HLR）
- [x] **点光阴影阵列** — `shadow_omni` cube-array，最多 4 盏；透明物体写入 CSM/Omni 深度
- [x] **Path/Search/GetMatrix** — 已实现
- [x] **离屏** — 截图桥（`OffscreenTarget` 复用与主帧相同像素格式路径由调用方提供尺寸）

### Phase 2

- [x] **Field 描述符**（既有）+ **FieldSensor/NodeSensor** 模块（`rc3d-scene::sensors`）
- [x] **Engine 连接图** — `EngineRegistry.connections` + `set_input`/`output` 端口；Kahn toposort；`rotating_cube` 演示 sine→calculator→compose→Transform
- [x] **SoEngine 族** — Gate / Decompose / Concatenate / SelectOne / BoolOp / Compose-Decompose Vec2/4 / matrix / rotation / TimeCounter / TransformVec3f（28 engines）
- [x] **跨节点 Field 图** — `SceneGraph::connect_fields`（`FieldRef`，`field_sources`/`field_targets`）；`propagate_fields` 在引擎之后
- [x] **Switch / MultipleCopy** 节点
- [x] **Text2/3** + glyphon
- [x] **拾取** — 三角/子对象 ID（IndexedFaceSet）
- [x] **子图元着色** — `set_face_tint` / `set_edge_tint`（立方体面 `tri/2`，IFS `face_ids`；`picking`）

### Phase 3

见 **§八**（明确不实现直至立项）。

---

## 五、架构建议（更新）

### 当前架构 vs Coin3D（2026）

```
Coin3D pattern:                    rustcoin3d current:

SoNode                            NodeData (enum)
  ├─ fields                         FieldDescriptor 部分 + rc3d-fields
  ├─ SoAction::apply()            Action (trait) + ActionKind
  ├─ SoCallback                   EventCallback / HandlerNode
  └─ SoSensor                     rc3d_scene::sensors (FieldSensor, NodeSensor)

SoMaterial + transparency         MaterialNode + opacity 字段

SoDragger / SoManip               TransformManipNode + DraggerNode + rc3d-gizmo overlay
SoText2 / SoText3                 Text2 / Text3 节点
SoLOD                             Lod 节点 + 运行时距离更新
SoSwitch / SoMultipleCopy         Switch, MultipleCopy
SoPath                            ScenePath, SearchAction, GetMatrixAction
SoEngine                          rc3d-engine：28 种（Gate/Decompose/Concatenate/…）+ 连接图 + SceneGraph FieldGraph
```

1. 操纵器以 **场景图节点**（`TransformManip` + 子 `Dragger`）驱动 `Engine.gizmo` overlay；无节点时仍可按选区绑定。

2. **Viewport** 已从单一 surface 演进到 `ViewportLayout` + 多通道渲染。

3. **winit → 场景图**：`Engine::handle_window_event` → `Event` + `HandleEventAction` + 全树键盘回调；示例经 `run_example`，editor 在 gizmo 前后拆 `feed_input` / `dispatch_routed_event`。

4. **Field 传感器** 在 `rc3d_scene` 中集中登记，与 `NodeData::field_descriptors` 互补。

5. **rc3d-nodes** 可逐步从 `node_data` 抽文件；当前仍以 `rc3d-scene` 为事实来源。

---

## 六、开发工作量估算

（表同前；Phase 0-2 引擎与 Field 图已落地，余量主要在 Phase 3 Backlog。）

---

## 七、当前渲染端优势的应用策略

（同前。）

---

## 八、Tier 3 / 远期 Backlog（未实现，需单独立项）

以下项**不进入默认实现清单**，仅作追踪：

| ID | 内容 |
|----|------|
| T3-1 | Mesh 布尔、稳健 BSP/第三方库 |
| T3-2 | NURBS 曲面、自适应细分 |
| T3-3 | 撤销/重做全场景事务与成组（命令栈已存在，需全覆盖） |
| T3-4 | Markup 全功能红线/云线编辑 |
| T3-5 | STEP / IGES / IFC 导入（ruststep、OCCT 等） |
| T3-6 | VR/AR 立体眼渲染（桌面 SBS/TB/anaglyph 已实现；无 VR compositor） |
| T3-7 | 网格修复（流形/补洞/自交） |
| T3-8 | 业务元数据与几何强绑定（属性系统） |

**CAD 与网格修复**依赖重型依赖与产品决策；确认需求后再开 crate 与 CI。

---

## 九、待验证 / 暂缓：FBX 骨骼蒙皮与动画（保留问题）

以下项**已实现代码路径**（`SkinnedMeshNode` → `DrawCall::skinning` → GPU skinning compute → 绘制），但在本仓库内**未完成端到端肉眼验收**，当前迭代**回到 Tier 0–1 基础问题**，不深追动画资产与渲染组合的细项，除非单独立项。

| 问题 | 说明 |
|------|------|
| **缺少标准动画 FBX 资产** | `test_data` 无 FBX；`cache/generated_models/test_cube.fbx` 为静态立方体，无法验证蒙皮变形与时间轴。 |
| **`import_viewer` / `render_features` 冒烟** | 自动化后台运行会因编译耗时、窗口事件循环或手动结束进程得到**非零退出码**，不能视为崩溃结论；真实验收需本机前台跑：``cargo run -p rc3d-app --example import_viewer -- <路径>.fbx`` 等。 |
| **阴影 + 蒙皮** | CSM 与蒙皮网格同一帧路径上已接 compute 预处理，但需在**带纹理/多部位动画**的模型上目视确认无撕裂、无错影。 |
| **部分关节有关键帧** | `AnimationClip::sample_all` 已对**无曲线关节**回退到 `Joint::bind_transform`；若姿势仍异常，需再查 FBX bind/rest 与 `Skeleton::new` 对逆绑定的处理，属数据/解析层而非单一采样技巧。 |

**结论**：上述条目作为追踪保留；**默认优先级回到 §三 Phase 0–1 / §四 checklist 中的基础交互与工业可视化项**，蒙皮动画深化另排。

---

## 文档修订记录

- 2026-04：与实现同步；增加 I/P/M 总表、§八 Backlog、 checklist 更新。
- 2026-05：增加 **§九** — FBX 蒙皮/动画待验证与暂缓说明；明确优先回到基础能力。
- 2026-08：CAD 装配物体轨道 — `AnimationClip.object_tracks` + `AnimationMixer` 写入任意 `TransformNode` / morph 权重，不依赖骨骼。 FBX 蒙皮验收仍见 §九。
