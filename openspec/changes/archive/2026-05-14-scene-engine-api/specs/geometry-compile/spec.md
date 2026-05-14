## ADDED Requirements

### Requirement: Geometry compilation
系统 SHALL 提供 `Geometry::compile(shape: &dyn Shape)` 方法，将 Shape 的参数化描述编译为内部的 `TriangleMesh`。

#### Scenario: Compile cube to TriangleMesh
- **WHEN** 调用 `Geometry::compile(&Cube::default())`
- **THEN** 返回的 Geometry 内部 TriangleMesh 包含 24 个顶点（6 面 × 4 个唯一顶点）、36 个索引（6 面 × 2 三角形 × 3 顶点）

#### Scenario: Compile sphere to TriangleMesh
- **WHEN** 调用 `Geometry::compile(&Sphere::default())`
- **THEN** 返回的 Geometry 内部 TriangleMesh 包含正确的经纬球拓扑（默认 24 slices × 16 stacks）

---

### Requirement: Geometry validation
系统 SHALL 提供 `Geometry::validate()` 方法，检测非流形边、退化三角形、零面积面等常见几何问题。

#### Scenario: Valid geometry passes validation
- **WHEN** 对正常 Cube 的 Geometry 调用 `validate()`
- **THEN** 返回 `ValidationResult::ok()`，包含顶点数、面数、边界边数等统计信息

#### Scenario: Degenerate triangle detected
- **WHEN** 对包含三点共线三角形的 Geometry 调用 `validate()`
- **THEN** 返回 `ValidationResult::warning()` 包含退化面的索引列表

---

### Requirement: AABB query
系统 SHALL 提供 `Geometry::aabb()` 方法返回编译后几何体的局部空间包围盒。

#### Scenario: Cube bounding box
- **WHEN** 对 width=2.0, height=1.0, depth=3.0 的 Cube 调用 `aabb()`
- **THEN** 返回 AABB(min=[-1,-0.5,-1.5], max=[1,0.5,1.5])

---

### Requirement: Topology query
系统 SHALL 提供对内部 `TriangleMesh` 拓扑信息的查询接口。

#### Scenario: Query boundary edges
- **WHEN** 对开放网格（如无底面的圆柱）调用 `geometry.boundary_edges()`
- **THEN** 返回所有仅属于一个面的边

#### Scenario: Query face count
- **WHEN** 调用 `geometry.face_count()`
- **THEN** 返回 TriangleMesh 中的总面数

---

### Requirement: GPU buffer extraction
系统 SHALL 提供 `Geometry::triangle_buffers()` 和 `Geometry::phong_buffers()` 方法，将编译后的网格提取为 GPU 就绪的顶点/索引缓冲区。

#### Scenario: Triangle buffers for flat shading
- **WHEN** 调用 `geometry.triangle_buffers()`
- **THEN** 返回 `(Vec<[f32;3]>, Vec<u32>)` — 位置数组和三角形索引

#### Scenario: Phong buffers for lit shading
- **WHEN** 调用 `geometry.phong_buffers()`
- **THEN** 返回 `(Vec<[f32;12]>, Vec<u32>)` — 交错 position+normal+uv+tangent 的顶点数组和三角形索引
