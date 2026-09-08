# rustcoin3d — 工业级 3D 可视化引擎

[English](README.md) · **简体中文** · [日本語](README.ja.md)

基于 Rust + wgpu 的 Coin3D/HOOPS 对齐的 3D 可视化引擎。专为大规模工业可视化而生：CAD 导入、实时 PBR 渲染、节点式合成与交互式场景编辑——全部集成在一款桌面 Studio 应用中。

![Studio](assets/studio_shot.png)

## 核心亮点

- **Rust + wgpu 30** 渲染器：集群延迟 PBR、CSM 阴影、HZB 遮挡、TAA/SSR/SSAO
- **场景图**，含 62 种节点类型（Coin3D/Inventor 风格 `Separator`/`Switch`/`LOD`、相机、灯光、标注、操纵器）
- **Blender 风格合成器** 节点图（`egui-snarl`），用于实时图像合成
- **面向工业 CAD**：NURBS、剖切/填充、GD&T/PMI、隐藏线、点云（OOC）
- **桌面 Studio**，带 Model / LookDev / Compositor 工作区，并支持完整 i18n（英文/简体中文）
- 51 个示例，覆盖渲染、光照、导入、动画、编辑器与诊断

## 文档

| 文档 | 说明 |
|------|------|
| [架构](docs/architecture.md) | crate 依赖图、核心设计原则、关键数据结构、NodeData 参考 |
| [渲染管线](docs/rendering-pipeline.md) | 完整帧管线、剔除、光照、PBR 着色、后处理、绘制调用批处理 |
| [场景图](docs/scene-graph.md) | SceneGraph API、全部节点类型、遍历模型、脏标记、动画、序列化 |
| [引擎系统](docs/engine-system.md) | 仿真引擎、时间管理、物理、传感器、字段连接 |
| [着色器](docs/shaders.md) | 完整 WGSL 着色器目录，含数据结构与性能说明 |
| [差距分析](docs/industrial-viz-gap-analysis.md) | Coin3D/HOOPS 对比、路线图、TODO 清单 |
| [优化指南](docs/optimization-guide.md) | GPU 剔除、网格池、静态帧快速路径、LightSetTable、共享工具 |
| [变更日志](CHANGELOG.md) | 版本说明与显著变更 |

## 快速开始

```bash
# 桌面编辑器
cargo run -p rc3d-studio

# 构建全部示例
cargo build -p rc3d-examples --examples

# 导入并查看 3D 文件
cargo run -p rc3d-examples --example import_viewer -- model.stl

# 自适应质量压力测试
cargo run -p rc3d-examples --example adaptive_stress_test

# CLI 编辑器（终端）
cargo run -p rc3d-cli-editor
```

## 架构

```
crates/
├── rc3d-core/       — 数学、AABB、BVH、ID 类型、共享工具（图、哈希、环形、排序）
├── rc3d-fields/     — 字段/连接系统（Coin3D 风格）
├── rc3d-scene/      — 场景图（SlotMap<NodeId, NodeEntry>）、节点类型、动画
├── rc3d-nodes/      — 再导出（便捷 crate）
├── rc3d-mesh/       — 三角网格、meshlet 生成、LOD、曲面细分
├── rc3d-nurbs/      — NURBS 曲线与曲面
├── rc3d-actions/    — 遍历动作（光线拾取、包围盒、撤销、事件、相交）
├── rc3d-engine/     — 仿真引擎、时间管理、物理、调度器
├── rc3d-io/         — 文件导入（STL、OBJ、glTF、FBX、Inventor）与导出
├── rc3d-render/     — wgpu 渲染器（PBR、阴影、剔除、后处理、着色器）
├── rc3d-gizmo/      — 3D 操纵器（平移、旋转、缩放）
├── rc3d-script/     — Rhai 脚本引擎
├── rc3d-pointcloud/ — 大规模点云八叉树（OOC）
├── rc3d-pdf/        — 3D PDF 导出（U3D）
├── rc3d-engine-api/ — Engine 门面（窗口、相机、渲染、合成器）
├── rc3d-editor/     — 编辑器库（键位、命令、应用、Fluent UI）
├── rc3d-examples/   — 演示应用（51 个示例）
├── rc3d-studio/     — 桌面编辑器宿主（工作区、i18n、案例库）
└── rc3d-cli-editor/ — 终端编辑器
```

## 渲染

集群延迟 PBR 渲染器，带完整 HDR 后处理链。

| 特性 | 说明 |
|------|------|
| **PBR** | 金属-粗糙度（GGX/Smith），带 IBL（HDR 环境贴图、BRDF LUT） |
| **阴影** | CSM（4 级联，8% 混合区），全向点光源阴影 |
| **光照** | 基于集群的前向（16×8×24 网格），LightSetTable 去重（1280B→4B/绘制） |
| **后处理** | TAA（YCoCg）、SSR（HIZ 加速）、SSAO、运动模糊、DoF、泛光、调色、体积雾、自动曝光 |
| **GPU 剔除** | 双路径：CPU BVH + GPU 计算（视锥 + HZB 遮挡），meshlet 集群树 |
| **选择** | 屏幕空间轮廓、边线叠加、包围框、X 光模式 |
| **显示** | 着色、线框、隐藏线、扁平、带边着色 |
| **自适应** | 5 级质量控制器，EMA+滞回、交互感知降级 |
| **CAD 分级** | Visualization / IndustrialDisplay / ProductRendering，带 GPU 钳制、轨道降级 + 冷却恢复 |
| **合成器** | 基于节点的合成图（Mix 混合模式、Math、变换、CAD 预设），以 GPU 乒乓 pass 执行 |

### PBR 与材质

![PBR 着色器变体](assets/pbr_shader_variant_viewer.png)

金属-粗糙度着色 + IBL，外加运行时 `PbrVariantCache`，按场景特性掩码编译并缓存至多 16 个专用着色器变体（清漆、光泽、虹彩、透射、各向异性）。

### 渲染特性

![渲染特性](assets/render_features.png)

### 光照与反射

![面光源](assets/area_light.png)
![反射](assets/reflection.png)
![阴影（CSM）](assets/shadow_demo.png)

集群前向光照，支持面光源、平面反射与级联阴影贴图。

### 后处理

![后处理](assets/post_effects.png)
![体积雾](assets/volumetric_demo.png)

完整 HDR 后处理链：SSAO → SSR → DoF → 泛光 → TAA → 色调映射，外加光线步进体积雾。

### 导入与拾取

![导入查看器](assets/import_viewer.png)
![拾取](assets/picking.png)

导入 STL / OBJ / glTF / FBX / Inventor 场景，并对面进行光线拾取以用于选择、测量与标注。

### 选择与显示模式

![选择轮廓](assets/selection_outline.png)
![选择集合](assets/selection_set.png)
![索引线集 / 隐藏线](assets/indexedlineset.png)

屏幕空间选择轮廓、隐藏线/线框显示模式，以及用于工程视图的索引线集。

## 场景图（最小示例）

```rust
use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Cube", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        graph.add_child(root, NodeData::PerspectiveCamera(
            PerspectiveCameraNode::look_at(
                Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y,
                std::f32::consts::FRAC_PI_4, 800.0 / 600.0,
            ),
        ));
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::ONE, intensity: 1.0, light_group: None,
        }));
        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.8, 0.2, 0.2),
            roughness: 0.4, metallic: 0.0, ..Default::default()
        }));
        graph.add_child(root, NodeData::Cube(CubeNode::default()));
    });
}
```

![场景图](assets/scene_graph.png)
![爆炸视图](assets/exploded_view.png)

## 示例（51 个演示）

| 分类 | 示例 |
|------|------|
| 入门 | `triangle`, `cube`, `rotating_cube`, `hello_scene` |
| 场景 | `scene_graph`, `annotation`, `billboard`, `environment_node`, `exploded_view`, `scripted_scene` |
| 渲染 | `pbr_scene`, `pbr_materials`, `pbr_variant_viewer`, `render_features`, `render_effects`, `material_variants`, `instancing`, `wboit_demo` |
| 光照 | `area_light`, `light_linking`, `shadow_demo`, `reflection` |
| 相机 | `stereo_camera`, `walk_camera` |
| 导入 | `import_viewer` |
| 动画 | `animation_demo`, `animation_control_panel`, `blend_animation` |
| 编辑器 | `selection_set`, `picking`, `markup_dimensions`, `annotation_edit` |
| 引擎 | `engines_demo`, `scripted_scene` |
| 特效 | `post_effects`, `volumetric_demo`, `decal_viewer`, `text3d` |
| 专用 | `nurbs_viewer`, `profile_viewer`, `section_caps`, `gdt_demo`, `stl_diagnostic` |
| 诊断 | `adaptive_stress_test`, `large_scene_stress`, `bench` |

### CAD 与工程

![NURBS](assets/nurbs_viewer.png)
![剖切端盖](assets/section_caps.png)
![剖面查看器](assets/profile_viewer.png)
![GD&T / PMI](assets/gdt_pmi.png)

NURBS 曲面、剖切端盖/填充，以及通过 `SceneGraph::bind_pmi` 绑定到命名零件的 GD&T/PMI 标注。

![标注尺寸](assets/markup_dimensions.png)

尺寸 / 角度 / 半径 / 引线标注被投影到 3D，并使用平面切线文本 pass（`Text2`/`Text3`）渲染。

### 点云与实例化

![点云](assets/point_cloud.png)
![实例化](assets/instancing.png)

外存八叉树点云渲染与重复几何体（BatchedMesh / InstancedMesh）的 GPU 实例化。

## 节点类型（62 个变体）

| 分类 | 变体 |
|------|------|
| **分组** | Separator, Group, Billboard, Transform, Rotation, RotationXYZ, Coordinate3, TextureCoordinate2, Normal, ShapeHints, MaterialBinding, ResetTransform, Texture2Transform, File |
| **形状** | Triangle, Cube, Sphere, Cone, Cylinder, IndexedFaceSet, IndexedLineSet, SkinnedMesh, MorphTarget, Sprite, BatchedMesh, InstancedMesh |
| **相机** | PerspectiveCamera, OrthographicCamera, StereoCamera, CubeCamera |
| **灯光** | DirectionalLight, PointLight, SpotLight, AreaLight, HemisphereLight, LightProbe |
| **遍历** | Lod, Switch, MultipleCopy, SectionPlane, PickStyle, EventCallback |
| **标注** | Text2, Text3, Measurement, Markup, Annotation, Font |
| **操纵器** | TransformManip, Dragger, Rotation |
| **专用** | ExplodedView, ReflectionPlane, Decal, RayTracing, Volume, PointCloud, Environment, Material |
| **可扩展** | HandlerNode(Arc\<dyn NodeHandler\>), Custom(u16, Box\<dyn CustomNodeData\>) |

## 性能特征

| 场景 | 对象 | 绘制调用 | 管线 | 帧耗时 |
|------|------|----------|------|--------|
| 压力测试 | 10K | ~10K | 实例化批处理 | ~5ms CPU |
| 压力测试（静态） | 10K | ~10K | 静态快速路径 | ~1ms CPU |
| 导入查看器 | 1-100K | 视情况 | 流式网格 | ~8-16ms |
| 目标（GPU） | 1M+ | 间接 | GPU 驱动 | 待定 |

关键优化：
- **灯光去重**：每次绘制调用 1280B → 4B（LightSetTable）
- **静态快速路径**：场景静止时零剔除工作
- **帧分配复用**：8 个 Vec 跨帧复用（@1M 对象约省 60MB）
- **BVH 增量**：仅脏 AABB 触发 BVH 更新
- **直接缓存发射**：FlatDrawCache 在遍历期间填充（无转换 pass）
- **池扩展**：phong 64K、flat 32K、mesh cache 4K

## 开发

```bash
cargo check --workspace          # 快速编译检查
cargo test                       # 328 项测试
cargo build -p rc3d-examples --examples
cargo run -p rc3d-studio         # 桌面编辑器
cargo clippy --workspace         # lint 检查
```

## Studio 桌面编辑器

![Studio 界面](assets/studio-ui.png)

`rc3d-studio` 是旗舰桌面应用：

- **工作区**：Model / LookDev / Compositor 快捷布局
- **停靠栏**：侧边停靠（Hierarchy+Inspector 分栏、Render、History、Assets），底部停靠（Document、Compositor），可移动工具条
- **合成器编辑器**：Blender 风格节点图（`egui-snarl`），带两级 Add 菜单与折叠状态持久化
- **案例库**：24 个参数/流程演示案例，含分步指导
- **i18n**：英文 / 简体中文（450 键目录）
- **键位**：默认快捷键，可在 `%APPDATA%\rustcoin3d\ui-prefs.json` 中按用户覆盖
- **CAD 矩阵检查**：`rc3d-studio --cad-matrix` 运行分级/合成器验证矩阵

## 依赖

| crate | 用途 |
|-------|------|
| wgpu 30 | GPU 抽象（Vulkan/Metal/DX12） |
| winit 0.30 | 窗口创建与事件循环 |
| egui 0.36 / eframe 0.36 | 即时模式 UI（编辑器 + Studio） |
| glam 0.29 | 线性代数（Vec3, Mat4, Quat） |
| slotmap | 场景图的稳定 ID 竞技场存储 |
| glyphon 0.12 | GPU 文本渲染（HUD） |
| rayon | 并行遍历 |
| rhai | 嵌入式脚本 |
| meshopt | 网格优化（meshlet、LOD） |
| serde/serde_json | 序列化 |
| image | 纹理加载 |
| tracy-client | GPU/CPU 分析 |

## 许可

BSD-3-Clause
