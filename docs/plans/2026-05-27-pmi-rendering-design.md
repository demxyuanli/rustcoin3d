# PMI 标注渲染 — 设计文档

**日期**: 2026-05-27
**对比基准**: OCCT XCAFDoc_NotesTool + XCAFDoc_DimTolTool
**范围**: `crates/rc3d-io/src/step/pmi/` — 实体类型、几何提取、渲染桥接

---

## 一、架构

```
STEP DATA entities
  │
  ▼
extract_pmi() ─── 几何解析修复 + 公差子类型新增
  │
  │  PmiData { dimensions, datums, tolerances }
  ▼
attach_pmi_to_scene() ─── PmiDimension → AnnotationElement::Dimension
  │                       PmiDatum → AnnotationElement::Datum
  │                       PmiToleranceFrame → AnnotationElement::Callout
  ▼
AnnotationSetNode { elements, style }
  │
  ▼
graph.add_child(root, AnnotationSet(node))
  │
  ▼
pass_markup (已有) + plane_text (已有)
```

**不改动**: `rc3d-scene/annotation/`、render passes、NodeData 枚举、AnnotationElement 枚举

**改动文件**:
- `crates/rc3d-io/src/step/entity_types.rs` — +8 公差实体变体
- `crates/rc3d-io/src/step/pmi/pmi_extract.rs` — 几何解析修复 + 公差子类型
- `crates/rc3d-io/src/step/pmi/pmi_render.rs` — PmiData → AnnotationSetNode 桥接
- `crates/rc3d-io/src/step/mod.rs` — 接入 import 管线
- `crates/rc3d-io/Cargo.toml` — `pmi` feature 默认开启

---

## 二、实体类型新增

在 `EntityType` 枚举中新增 8 种 AP242 公差/标注实体：

| 变体 | STEP 实体 | 语义 |
|------|----------|------|
| `FlatnessTolerance` | FLATNESS_TOLERANCE | 平面度 |
| `PositionTolerance` | POSITION_TOLERANCE | 位置度 |
| `ProfileTolerance` | PROFILE_TOLERANCE / LINE_PROFILE_TOLERANCE / SURFACE_PROFILE_TOLERANCE | 轮廓度 |
| `ParallelismTolerance` | PARALLELISM_TOLERANCE | 平行度 |
| `PerpendicularityTolerance` | PERPENDICULARITY_TOLERANCE | 垂直度 |
| `RunoffTolerance` | CIRCULAR_RUNOUT_TOLERANCE / TOTAL_RUNOUT_TOLERANCE | 跳动 |
| `StraightnessTolerance` | STRAIGHTNESS_TOLERANCE | 直线度 |
| `DatumReferenceElement` | DATUM_REFERENCE_ELEMENT | 基准引用 |

所有公差复用 `PmiToleranceFrame` 数据结构 — `text` 字段存储完整格式化文本（含基准引用）。

---

## 三、PmiData 数据模型

现有数据结构不变。几何提取从占位符修复为真实 STEP 实体链解析。

### 3.1 提取链

```
ANNOTATION_OCCURRENCE (标注→几何关联)
  → STYLED_ITEM
    → PRESENTATION_STYLE_ASSIGNMENT (样式)
  → DIMENSIONAL_SIZE / GEOMETRIC_TOLERANCE (标注类型)
    → CARTESIAN_POINT / AXIS2_PLACEMENT_3D (锚点位置)
    → DIMENSIONAL_CHARACTERISTIC_REPRESENTATION (数值)
```

### 3.2 PmiDimension

| 字段 | 来源 |
|------|------|
| `start` | ANNOTATION_OCCURRENCE → reference_points[0] → CARTESIAN_POINT |
| `end` | ANNOTATION_OCCURRENCE → reference_points[1] → CARTESIAN_POINT |
| `offset_dir` | PRESENTATION_STYLE_ASSIGNMENT → 平面法向推断，fallback = (end - start) 正交 |
| `text` | 格式化: `"{value:0.N} {unit_suffix}"`, 从 DIMENSIONAL_CHARACTERISTIC_REPRESENTATION 读值 |

### 3.3 PmiDatum

| 字段 | 来源 |
|------|------|
| `origin` | DATUM_FEATURE → AXIS2_PLACEMENT_3D.origin → CARTESIAN_POINT |
| `normal` | AXIS2_PLACEMENT_3D.axis → DIRECTION |
| `label` | DATUM.name → STRING |

### 3.4 PmiToleranceFrame

| 字段 | 来源 |
|------|------|
| `origin` | GEOMETRIC_TOLERANCE → leader anchor → CARTESIAN_POINT |
| `leader_points` | ANNOTATION_OCCURRENCE → leader curve points (如有) |
| `text` | 格式化: `"{symbol} {value} {datum_refs}"` |

公差 text 格式：
- Flatness: `"▱ {value}"`
- Position: `"⊕ {value} {datum_labels}"`
- Profile: `"⌒ {value}"`
- Parallelism: `"∥ {value} {datum_labels}"`
- Perpendicularity: `"⊥ {value} {datum_labels}"`
- Runout: `"↗ {value}"`
- Straightness: `"— {value}"`

符号使用 GDT 标准字符。

---

## 四、桥接层

### 4.1 映射

```
PmiDimension        → AnnotationElement::Dimension {
                          start: d.start,
                          end: d.end,
                          offset_dir: d.offset_dir,
                          label: d.text,
                          label_mode: Fixed,
                      }

PmiDatum            → AnnotationElement::Datum {
                          origin: d.origin,
                          normal: d.normal,
                          label: d.label,
                      }

PmiToleranceFrame   → AnnotationElement::Callout {
                          origin: t.origin,
                          leader: t.leader_points,
                          label: t.text,
                          label_offset: [24.0, 12.0, 0.0],
                      }
```

### 4.2 入口签名

```rust
pub fn attach_pmi_to_scene(
    graph: &mut SceneGraph,
    parent: NodeId,
    pmi: &PmiData,
) -> NodeId
```

返回创建的 `AnnotationSet` 节点的 `NodeId`。

### 4.3 样式

`AnnotationStyle` 默认值：
- `decimals: 3`
- `unit_suffix: " mm"`
- `font_size: 14.0`
- `extension_len: 0.3`
- `arrow_size: 0.15`

从 STEP `PRESENTATION_STYLE_ASSIGNMENT` 读取样式（如有），覆盖默认值。样式解析为 optional best-effort — 无样式时不报错。

---

## 五、接入 Import 管线

在 `mod.rs` 的 `exchange_to_scene_graph()` 中，B-Rep mesh 生成之后、函数返回之前：

```rust
#[cfg(feature = "pmi")]
{
    let pmi_data = pmi::pmi_extract::extract_pmi(&exchange.entities);
    if !pmi_data.dimensions.is_empty() || !pmi_data.datums.is_empty() || !pmi_data.tolerances.is_empty() {
        log::info!(
            "[STEP] PMI: {} dims, {} datums, {} tolerances",
            pmi_data.dimensions.len(),
            pmi_data.datums.len(),
            pmi_data.tolerances.len(),
        );
        pmi::pmi_render::attach_pmi_to_scene(&mut graph, root, &pmi_data);
    }
}
```

### 5.1 Feature flag

修改 `crates/rc3d-io/Cargo.toml`：

```toml
[features]
default = ["pmi"]
pmi = []
```

移除 `mod.rs` 中的 `#[cfg(feature = "pmi")]` 条件编译块内的 conditional import — 改为无条件引入，因为 feature 默认开启。

---

## 六、公差子类型提取

每个新增的 `EntityType` 变体在 `extract_pmi()` 的 match 中增加分支：

```rust
EntityType::FlatnessTolerance
| EntityType::PositionTolerance
| EntityType::ProfileTolerance
| EntityType::ParallelismTolerance
| EntityType::PerpendicularityTolerance
| EntityType::RunoffTolerance
| EntityType::StraightnessTolerance => {
    if let Some(tol) = extract_tolerance(&record.params, entities) {
        pmi.tolerances.push(tol);
    }
}
EntityType::DatumReferenceElement => {
    // Accumulated per tolerance frame; resolved during tolerance extraction
    // via the entity_id → datum label lookup table
}
```

`extract_tolerance()` 统一处理所有公差类型，根据 `entity_type` 选取正确的 GDT 符号前缀。

---

## 七、验收标准

1. **实体识别**: 含 AP242 PMI 的 STEP 文件导入时，`extract_pmi()` 输出非零 `dimensions` / `datums` / `tolerances`
2. **几何正确**: 尺寸标注线定位在被标注几何附近，基准标识定位在基准面上
3. **渲染可见**: `AnnotationSet` 节点被正确添加到 SceneGraph，渲染时可见标注
4. **文本格式**: 公差框显示标准 GDT 符号 + 数值 + 基准引用
5. **无回归**: 无 PMI 的文件导入行为不变 (281 + 7 集成测试全通过)
6. **Feature 默认**: `rc3d-io` 默认编译包含 PMI 支持

---

## 八、不改动的范围

- `rc3d-scene/annotation/` 8 个模块 — 零改动
- `pass_markup` / `plane_text` 渲染 — 零改动
- `AnnotationElement` 枚举 — 不改动，复用现有变体
- `NodeData` 枚举 — 不改动
- `AnnotationSetNode` 数据结构 — 不改动
- 装配体样式继承 — 不改动 (P2 backlog)
- 语义 PMI — 不改动 (P2 backlog)
- 镶嵌几何 — 不改动 (NA)
