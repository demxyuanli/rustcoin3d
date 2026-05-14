## ADDED Requirements

### Requirement: Scene container
系统 SHALL 提供 `Scene` 结构作为场景构建的顶层容器。Scene 内部持有 `SceneGraph`，对外隐藏所有 NodeData/Separator/StateStack 细节。

#### Scenario: Create empty scene
- **WHEN** 用户调用 `Scene::new()`
- **THEN** 返回一个空场景，内部 SceneGraph 已初始化包含默认的根 Separator

#### Scenario: Add shape to scene
- **WHEN** 用户调用 `scene.add(Cube::default())`
- **THEN** 系统在内部 SceneGraph 中自动创建 `Separator → CubeNode` 子树，并返回一个 `NodeHandle`

#### Scenario: Build scene into graph
- **WHEN** 用户调用 `scene.build()`
- **THEN** 返回内部 `SceneGraph`，所有 Shape/Material/Light/Camera 已编译为对应 NodeData 变体

---

### Requirement: Shape trait and built-in shapes
系统 SHALL 提供 `Shape` trait 定义几何体的统一接口，并提供 Cube、Sphere、Cone、Cylinder、IndexedFaceSet、IndexedLineSet 等内置实现。

#### Scenario: Cube with all builder options
- **WHEN** 用户创建 `Cube::default().width(2.0).height(1.0).depth(3.0).at(x,y,z).scale(sx,sy,sz).rotate_y(45.0).material(mat)`
- **THEN** 系统编译为包含 Transform + Material + CubeNode 的完整子树

#### Scenario: Shape without material inherits current material
- **WHEN** 用户添加一个不带 material 的 Shape 到 Scene
- **THEN** Shape 使用场景当前材质状态（Coin3D 继承语义）

#### Scenario: Custom shape via IndexedFaceSet
- **WHEN** 用户创建 `Mesh::from_raw(positions, indices).with_normals(normals).with_texcoords(uvs)`
- **THEN** 系统生成 IndexedFaceSetNode 并正确设置 Coordinate3 属性

---

### Requirement: Material builder
系统 SHALL 提供 `Material::pbr()` builder 用于创建 PBR 材质，参数对应现有 `MaterialNode` 字段。

#### Scenario: PBR material with metallic-roughness
- **WHEN** 用户调用 `Material::pbr().base_color(0.8, 0.2, 0.2).metallic(0.0).roughness(0.5).build()`
- **THEN** 生成一个 MaterialNode，其 base_color=[0.8,0.2,0.2], metallic=0.0, roughness=0.5

#### Scenario: Simple diffuse material
- **WHEN** 用户调用 `Material::diffuse(0.2, 0.5, 0.8)`
- **THEN** 生成一个默认 PBR 参数 + 指定 diffuse_color 的 MaterialNode

---

### Requirement: Light types
系统 SHALL 提供 `DirectionalLight`、`PointLight`、`SpotLight` 类型，对应现有 `DirectionalLightNode`、`PointLightNode`、`SpotLightNode`。

#### Scenario: Add directional light
- **WHEN** 用户调用 `scene.add(DirectionalLight::sun(direction, intensity))`
- **THEN** 场景添加一个 DirectionalLightNode，内部自动设置 direction + color + intensity

#### Scenario: Add point light
- **WHEN** 用户调用 `scene.add(PointLight::new(position, color, intensity))`
- **THEN** 场景添加一个 PointLightNode

---

### Requirement: Camera types
系统 SHALL 提供 `PerspectiveCamera` 和 `OrthographicCamera` 类型。

#### Scenario: Perspective camera with look_at
- **WHEN** 用户调用 `PerspectiveCamera::look_at(eye, target, up, fov, aspect)`
- **THEN** 生成一个正确计算的 PerspectiveCameraNode

#### Scenario: Set camera on scene
- **WHEN** 用户调用 `scene.set_camera(camera)`
- **THEN** 场景的摄像机节点被设置（后续 App 使用此摄像机初始化视口）

---

### Requirement: Group node
系统 SHALL 提供 `Group` 类型用于显式分组，内部映射为 Separator 提供状态隔离。

#### Scenario: Group with multiple children
- **WHEN** 用户创建 `Group::new("assembly").add(shape_a).add(shape_b)` 并添加到 scene
- **THEN** 内部生成 Separator 包裹所有子节点，子节点间状态不泄漏

#### Scenario: Nested groups
- **WHEN** 用户将 Group A 添加为 Group B 的子节点
- **THEN** 内部生成嵌套的 Separator，外层 Separator 的状态在进入内层前被保存

---

### Requirement: Scene query
系统 SHALL 提供 `Query` 类型进行类型安全的场景图遍历，消除手写递归。

#### Scenario: Check if any node of type exists
- **WHEN** 用户调用 `scene.query().any::<PerspectiveCamera>()`
- **THEN** 返回 `true` 当场景中存在至少一个 PerspectiveCamera 时

#### Scenario: Collect all materials
- **WHEN** 用户调用 `scene.query().collect_all::<Material>()`
- **THEN** 返回所有 Material 节点的引用列表

#### Scenario: Get bounding box of subtree
- **WHEN** 用户调用 `scene.query().bounds_of(handle)`
- **THEN** 返回该子树所有几何体的合并 AABB

---

### Requirement: Engine compatibility
系统 SHALL 保持与现有 `Engine trait` 的兼容性。Scene 提供 `build()` 获取内部 SceneGraph，用于 EngineRegistry。

#### Scenario: Attach engines to scene
- **WHEN** 用户调用 `scene.build()` 后创建 `EngineRegistry` 并注册 Engine
- **THEN** 引擎可以正常操作 SceneGraph 中的节点
