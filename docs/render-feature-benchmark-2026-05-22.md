# rustcoin3d 渲染功能对标审查报告

> 日期：2026-05-22（修正版）
> 对标：Coin3D / HOOPS / Three.js
> 方法：全代码库深度扫描 + 逐维度对比
> 修正：初版对多项已实现功能误判为"缺失"，本版基于代码实证修正

## 修正摘要

初版审查报告存在多处误判，经深入代码验证后修正如下：

| 功能 | 初版评估 | 修正后 |
|------|----------|--------|
| 3D 文本渲染 | ❌ 缺失 | ✅ **已完整实现** — Text2Node/Text3Node + glyphon + world labels + annotation |
| 点光阴影 | ❌ 缺失 | ✅ **已实现** — shadow_omni.rs（限1个点光源，512分辨率） |
| Transform Gizmo | ⚠️ 未集成 | ✅ **已完整集成** — rc3d-editor gizmo.rs + interaction.rs + Undo |
| 测量工具 | ❌ 缺失 | ✅ **已完整实现** — Distance/Angle/Radius + AnnotationSet 渲染 |
| 剖面帽面 | ❌ 缺失 | ✅ **已完整实现** — cap_enabled/cap_color + SectionCapUniforms + section_cap.wgsl |
| 标注/引线 | ❌ 缺失 | ✅ **已实现** — Dimension/AngleDimension/Radial/Diameter/Leader/Callout/Datum |

## 总体评估

rustcoin3d 的 **GPU 渲染管线**和**工业可视化功能层**均已达到较高水准。在渲染管线方面已超越对标引擎，在工业可视化功能方面覆盖了大部分核心需求，仅少数高级特性待补。

| 维度 | 状态 |
|------|------|
| **渲染管线** | ✅ 已超越 — meshlet + HZB、CSM、HDR 后处理链、clustered forward |
| **材质系统** | ✅ PBR 完整 — metallic-roughness、5 张纹理槽、IBL |
| **光照** | ✅ 4 类灯光 + cluster culling + light linking + Omni 阴影 |
| **显示模式** | ✅ 6 种（ShadedWithEdges/Shaded/Wireframe/HiddenLine/Flat/FlatWithEdge） |
| **GPU 驱动** | ✅ meshlet cull + GPU object cull + indirect draw |
| **后处理** | ✅ TAA/SSR/DOF/MotionBlur/Bloom/SSAO/ColorGrading/XRay/VolumetricFog |
| **透明渲染** | ✅ Alpha blend pass（本次新增接线） |
| **文本/标注** | ✅ Text2/Text3 + 完整标注系统 + Glyph cache |
| **交互/测量** | ✅ Pick/高亮/Gizmo/测量 + Undo/Redo |
| **剖面/帽面** | ✅ SectionPlane + cap fill |

---

## 一、几何体 / 场景图节点

| 特性 | Coin3D | HOOPS | Three.js | rustcoin3d | 差距 |
|------|--------|-------|----------|------------|------|
| 基础体 (Cube/Sphere/Cone/Cylinder) | ✅ | ✅ | ✅ | ✅ | 无 |
| Torus | ✅ SoTorus | ✅ | ✅ TorusGeometry | ✅ **TorusNode**（本次新增） | 无 |
| NURBS 曲线/曲面 | ✅ 核心 | ✅ | ❌ 需插件 | ⚠️ `rc3d-nurbs` 有数学库但未接入渲染 | **Partial** |
| IndexedFaceSet | ✅ | ✅ | ✅ BufferGeometry | ✅ | 无 |
| IndexedLineSet | ✅ | ✅ | ✅ LineSegments | ✅ | 无 |
| 点云 | ✅ SoPointSet | ✅ | ✅ Points | ✅ PointCloudNode | 无（缺 EDL 增强） |
| Text2/Text3 | ✅ | ✅ | ✅ TextGeometry | ✅ glyphon + world labels | 无 |
| 剖面/帽面 | ✅ SoClipPlane + cap | ✅ | ❌ 需手动 | ✅ SectionPlane + cap fill | 无 |
| 剖面线 (hatching) | ✅ | ✅ | ❌ | ❌ | **缺失** |

**已修复**：TorusNode 已添加（场景图节点 + tessellate_torus + ray pick）

**仍缺失**：
1. **NURBS 渲染集成** — 数学库存在但未接入场景图渲染
2. **剖面线 (Hatching)** — 工程制图传统表示法

---

## 二、材质 / 渲染特性

| 特性 | Coin3D | HOOPS | Three.js | rustcoin3d | 差距 |
|------|--------|-------|----------|------------|------|
| Phong | ✅ | ✅ | ✅ MeshPhongMaterial | ✅ | 无 |
| PBR (metallic-roughness) | ❌ | ✅ 2022+ | ✅ MeshStandardMaterial | ✅ | 无 |
| 透明排序 | ✅ depth-sort | ✅ depth-sort | ✅ depth-sort | ✅ **alpha blend pass**（本次接线） | 无 |
| OIT (WBOIT/per-pixel) | ❌ | ✅ | ✅ 有限 | ❌ | **缺失** |
| 材质覆盖 (override) | ✅ SoMaterialBinding | ✅ | ✅ material.wireframe | ⚠️ 仅 Flat 模式 | **Partial** |
| 双面渲染 | ✅ | ✅ | ✅ side: DoubleSide | ✅ | 无 |
| 线框覆盖材质 | ✅ | ✅ | ✅ wireframe: true | ✅ Wireframe 模式 | 无 |
| 屏幕空间边缘 | ❌ | ✅ | ❌ 需后处理 | ✅ SS edge pass | **领先** |

**已修复**：透明 Pass 已接线（solid_alpha pipeline + back-to-front 排序）

**仍缺失**：
1. **OIT (Order-Independent Transparency)** — 多层透明重叠时画家算法可能失败
2. **材质覆盖系统** — 全局覆盖材质（如全部设为线框、全部变灰），当前仅有 Flat 模式

---

## 三、光照 / 阴影

| 特性 | Coin3D | HOOPS | Three.js | rustcoin3d | 差距 |
|------|--------|-------|----------|------------|------|
| 方向光/点光/聚光 | ✅ | ✅ | ✅ | ✅ | 无 |
| Area light | ❌ | ✅ | ✅ RectAreaLight | ✅ AreaLightNode | 无 |
| IBL (环境贴图) | ❌ | ✅ | ✅ | ✅ | 无 |
| Light linking (逐对象) | ❌ | ✅ | ❌ | ✅ | **领先** |
| CSM 级联阴影 | ❌ | ✅ | ✅ | ✅ (4 cascade) | 无 |
| Omni 点光阴影 | ❌ | ✅ | ✅ PointLight.shadow | ✅ shadow_omni.rs | 无（限1个点光源） |
| 透明投影阴影 | ✅ | ✅ | ✅ | ❌ | **缺失** |
| 阴影接触硬化 (PCSS) | ❌ | ❌ | ❌ | ❌ | 均缺 |

**修正**：Area Light 和 Omni 阴影均已实现（初版误判为缺失）

**仍缺失**：
1. **多光源 Omni 阴影** — 当前仅支持 1 个点光源阴影
2. **透明物体投影阴影** — 透过玻璃/半透明件的阴影

---

## 四、选择 / 交互 / Gizmo

| 特性 | Coin3D | HOOPS | Three.js | rustcoin3d | 差距 |
|------|--------|-------|----------|------------|------|
| 点选 (pick) | ✅ SoRayPickAction | ✅ | ✅ Raycaster | ✅ | 无 |
| 框选 (marquee) | ✅ | ✅ | ❌ 需手动 | ⚠️ SelectionBox 存在 | **Partial** |
| 高亮 (highlight) | ✅ SoHighlight | ✅ | ✅ OutlinePass | ✅ | 无 |
| X-Ray 选择高亮 | ❌ | ✅ | ❌ | ✅ | **领先** |
| Transform Gizmo | ✅ SoTransformManip | ✅ | ✅ TransformControls | ✅ rc3d-gizmo + editor 集成 | 无 |
| 测量工具 | ❌ | ✅ | ❌ | ✅ Distance/Angle/Radius + Annotation | 无 |
| 爆炸视图 | ❌ | ✅ | ❌ 需手动 | ✅ ExplodedViewNode | 无 |
| 撤销/重做 | ✅ SoUndoManager | ✅ | ❌ | ✅ editor undo stack | 无 |

**修正**：Gizmo、测量、爆炸视图、Undo/Redo 均已实现（初版误判为缺失或未集成）

**仍缺失**：
1. **框选 (Marquee)** — SelectionBox 存在但未完整集成

---

## 五、文本 / 标注 / 剖面

| 特性 | Coin3D | HOOPS | Three.js | rustcoin3d | 差距 |
|------|--------|-------|----------|------------|------|
| 2D 屏幕文本 | ✅ SoText2 | ✅ | ✅ CSS2D | ✅ Text2Node + glyphon | 无 |
| 3D 世界文本 | ✅ SoText3 | ✅ | ✅ TextGeometry | ✅ Text3Node + world labels | 无 |
| 标注/引线 | ✅ SoAnnotation | ✅ | ❌ | ✅ 完整 7 种标注类型 | 无 |
| Leader line | ✅ | ✅ | ❌ | ✅ | 无 |
| 剖面平面 | ✅ SoClipPlane | ✅ | ✅ ClippingPlane | ✅ SectionPlane | 无 |
| **剖面帽面填充** | ✅ SoClipPlane+cap | ✅ | ❌ | ✅ cap_enabled + section_cap.wgsl | 无 |
| 剖面线 (hatching) | ✅ | ✅ | ❌ | ❌ | **缺失** |
| PMI (产品制造信息) | ❌ | ✅ | ❌ | ❌ | **缺失** |

**修正**：3D 文本、标注、剖面帽面均已实现（初版误判为缺失）

**仍缺失**：
1. **剖面线 (Hatching)** — 工程制图传统表示法
2. **PMI** — 产品制造信息（高级工业需求）

---

## 六、相机 / 视口 / 输出

| 特性 | Coin3D | HOOPS | Three.js | rustcoin3d | 差距 |
|------|--------|-------|----------|------------|------|
| 透视相机 | ✅ | ✅ | ✅ | ✅ | 无 |
| 正交相机 | ✅ | ✅ | ✅ | ✅ | 无 |
| 立体渲染 (Stereo) | ✅ SoStereoViewer | ✅ | ✅ StereoEffect | ⚠️ StereoCameraNode 存在 | **Partial** |
| 多视口 | ✅ SoAnnotation | ✅ | ✅ Scissor | ❌ | **缺失** |
| Fit All / Fit Selection | ✅ viewAll() | ✅ | ✅ fitToBox | ✅ | 无 |
| 截图输出 | ✅ | ✅ | ✅ toDataURL | ⚠️ 有 offscreen 但未暴露 API | **Partial** |
| 动画/关键帧 | ✅ SoElapsedTime | ✅ | ✅ AnimationMixer | ⚠️ 有 skeleton 但缺关键帧系统 | **Partial** |

**仍缺失**：
1. **多视口** — 前/侧/顶/等轴四视图是 CAD 标配
2. **立体渲染** — StereoCameraNode 存在但渲染路径未完整
3. **关键帧动画系统** — 当前只有骨骼动画，缺关键帧插值

---

## 七、修正后优先级矩阵

| 优先级 | 特性 | 理由 | 状态 |
|--------|------|------|------|
| ~~P0~~ | ~~剖面帽面填充~~ | ~~工业场景看内部结构~~ | ✅ 已实现 |
| ~~P0~~ | ~~3D 文本渲染~~ | ~~标注/PMI 基础~~ | ✅ 已实现 |
| **P1** | OIT (WBOIT) | 多层半透明装配体画家算法可能失败 | 待实现 |
| **P1** | 多光源 Omni 阴影 | 多点光源室内场景 | 待实现 |
| **P2** | 多视口 | CAD 四视图 | 待实现 |
| **P2** | 材质覆盖系统 | 批量检查用 | 待实现 |
| **P2** | NURBS 渲染集成 | 数学库已有，接入即可 | 待实现 |
| **P2** | 关键帧动画 | 非核心但有价值 | 待实现 |
| **P3** | 剖面线 (Hatching) | 工程制图传统 | 待实现 |
| **P3** | 立体渲染 | VR/AR 工程展示 | 待实现 |
| **P3** | PMI | 高级工业需求 | 待实现 |

---

## 八、本次实施成果

| 变更 | 文件 | 描述 |
|------|------|------|
| ✅ SectionPlaneNode field_descriptors | `rc3d-scene/src/node_data.rs` | 暴露 cap_color/cap_enabled 字段 |
| ✅ 透明 Pass 接线 | `rc3d-render/src/render_passes/pass_transparent.rs` | 新模块：solid_alpha pipeline + back-to-front 排序 |
| ✅ 透明 Pass 注册 | `rc3d-render/src/render_passes.rs` | 在 solid+outline 后、effects 前调用 |
| ✅ TorusNode 基元体 | `rc3d-scene/src/node_data.rs` | 新增 TorusNode 结构体 + NodeData 枚举变体 |
| ✅ Torus tessellation | `rc3d-mesh/src/tessellate.rs` | tessellate_torus() 函数 |
| ✅ Torus 渲染集成 | `rc3d-render/src/render_action.rs` | NodeData::Torus → emit_cached_shape |
| ✅ Torus ray pick | `rc3d-actions/src/ray_pick.rs` | pick_torus() 方法 |
| ✅ Torus UI | `rc3d-editor/src/ui/draw.rs` | 类型标签映射 |
| ✅ ShapeKey::Torus | `rc3d-render/src/shape_cache.rs` | 缓存 key 变体 |
| ✅ 测试更新 | 多文件 | field_descriptors / type_name / collector 测试 |

---

## 九、结论

经深入代码验证，rustcoin3d 的工业可视化功能覆盖度远超初版评估。**剖面帽面、3D 文本、标注系统、Gizmo、测量工具、点光阴影、爆炸视图、Undo/Redo** 均已完整实现。

真正剩余的 P1 缺口仅为 **WBOIT**（多层半透明排序）和**多光源 Omni 阴影**。P0 功能全部就绪。

本次实施新增了 **TorusNode 基元体**和**透明 Pass 接线**，进一步补齐了功能缺口。
