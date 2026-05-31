# Subsuper Adapter + BSpline 参数布局修复

**日期**: 2026-05-31
**状态**: 待审批
**范围**: Subsuper adapter 重构 + NURBS 构建自适应参数扫描

---

## 背景

Shape-2.step 的 BSpline 面被 subsuper adapter 命名为 "SURFACE"（最不派生的父类型），导致 `build_surface` 无法识别，回退到 Plane。根因是 adapter 的 `select_primary_record_structured` 使用 `leaf_index`（最后一条记录）选择主类型，STEP 中最后一条总是父类型。

同时，CompatMerge 合并后的参数布局取决于 STEP 文件中可选参数的省略情况，`build_nurbs_surface` 硬编码的参数偏移不可靠。

## 涉及文件

| 文件 | 改动 |
|------|------|
| `adapter/subsuper.rs` | `select_primary_record_structured` 重写为优先级+结构性检查 |
| `geom/nurbs_build.rs` | `build_nurbs_surface` 用参数扫描替代硬编码偏移 |
| `adapter/mod.rs` | 测试更新 |

## 修复 1：优先级选择器

### 三规则选择

```
1. 在 records 中按 PRIORITY_TYPES 顺序查找
2. 跳过首参不是 Integer 的记录（非结构化类型，如 WITH_KNOTS 只有 knot 数据）
3. 选中第一个满足条件的记录作为主类型
   未匹配 → 回退到 leaf_index（向后兼容）
```

### PRIORITY_TYPES 共享

`subsuper.rs` 引用 `primary_keyword.rs` 导出的 `PRIORITY_TYPES`，不在两处维护独立优先级表。

### 实现

```rust
fn select_primary_record_structured(
    pairs: &[(String, &StepValue)],
    leaf_index: usize,
) -> Result<(usize, String), String> {
    // Rule 1-2: priority-based with structural check
    for (prio_type, _) in PRIORITY_TYPES.iter().enumerate() {
        for (i, (name, params)) in pairs.iter().enumerate() {
            if name != *prio_type { continue; }
            // Structural check: first param must be Integer (degree/dimension)
            let has_struct = params.as_list().and_then(|l| l.first())
                .map_or(false, |v| matches!(v, StepValue::Integer(_)));
            if has_struct {
                return Ok((i, name.clone()));
            }
        }
    }
    // Rule 3: fallback to leaf_index
    let idx = leaf_index.min(pairs.len() - 1);
    Ok((idx, pairs[idx].0.clone()))
}
```

### 场景覆盖

| 场景 | PRIORITY_TYPES 命中 | 结构性检查 | 结果 |
|------|---------------------|-----------|------|
| `(BOUNDED_SURFACE() B_SPLINE_SURFACE(6,10,CPs))` | B_SPLINE_SURFACE (prio=4) | ✅ Integer | "B_SPLINE_SURFACE" |
| `(B_SPLINE_SURFACE(6,10,CPs) B_SPLINE_SURFACE_WITH_KNOTS(um,vm,...))` | S_WITH_KNOTS (prio=3)→Integer?❌→skip. S_SURFACE (prio=4)→✅ | ✅ Integer | "B_SPLINE_SURFACE" |
| `(ELEMENTARY_SURFACE() PLANE(...))` | PLANE (prio=24) | ❌ no Integer → skip? Wait: PLANE params[0] is #ref (placement) → falls through → leaf_index | falls to leaf |
| `(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.MILLI.))` | 无命中 | — | leaf_index→"SI_UNIT" |

**ELEMENTARY_SURFACE + PLANE 特殊处理**：PLANE 的首参是 entity ref（placement），不是 Integer。结构性检查会跳过它。但 PLANE 在 `build_surface` 中已有独立的 `"ELEMENTARY_SURFACE"` 处理路径（通过 `nth_ref(1)` unwrap basis surface），所以这个场景已被覆盖。

对于有 entity ref 作为首参的具体类型（PLANE, CYLINDER, SPHERE, TORUS, CONE, EXTRUSION, REVOLUTION, OFFSET），结构性检查应扩展为：首参是 Integer **或** entity ref（`StepValue::Ref`）。

修正后的结构性检查：
```rust
let has_struct = params.as_list().and_then(|l| l.first())
    .map_or(false, |v| matches!(v, StepValue::Integer(_) | StepValue::Ref(_)));
```

## 修复 2：智能参数扫描

### 问题

`build_nurbs_surface` 硬编码 knot 数据位置：
```rust
let mult_base = off + 7;  // ← 仅在 standalone WITH_KNOTS 格式下正确
```

CompatMerge 合并后的参数布局因 STEP 可选参数省略情况而异，`off+7` 不可靠。

### 方案

在合并参数列表中**扫描**查找 knot multiplicities 和 weights，不依赖硬编码位置：

```rust
/// Scan merged params for knot vector data.
/// Returns (u_mults, v_mults, u_knots, v_knots) or defaults.
fn scan_knot_data(params: &StepValue, cp_u: usize, cp_v: usize) 
    -> (Vec<i64>, Vec<i64>, Vec<f64>, Vec<f64>)
{
    let list = params.as_list()?;
    // Scan for consecutive List<Integer> after CP data (List<List<Ref>>)
    let mut int_lists: Vec<&[StepValue]> = Vec::new();
    let mut real_lists: Vec<&[StepValue]> = Vec::new();
    let mut past_cps = false;
    for val in list {
        match val {
            StepValue::List(inner) if !inner.is_empty() => {
                if inner.iter().all(|v| matches!(v, StepValue::Integer(_))) {
                    int_lists.push(inner.as_slice());
                } else if inner.iter().all(|v| matches!(v, StepValue::Real(_))) {
                    real_lists.push(inner.as_slice());
                } else if inner.iter().all(|v| matches!(v, StepValue::Ref(_))) {
                    past_cps = true; // This was the CP list
                }
            }
            _ => {}
        }
    }
    // First 2 int lists after CPs are u_mults, v_mults
    // First 2 real lists after CPs are u_knots, v_knots
    // ... extract and return
}
```

### 向后兼容

扫描逻辑仅在 `nth_list_ints(params, off+7)` 返回空时触发（即硬编码位置失败时才扫描）：

```rust
let u_mults = geom::nth_list_ints(params, mult_base);
let v_mults = geom::nth_list_ints(params, mult_base + 1);
if u_mults.is_empty() || v_mults.is_empty() {
    // Hardcoded position failed — scan for knot data
    let (um, vm, uk, vk) = scan_knot_data(params, cp_u, cp_v);
    // ... use scanned data
}
```

## 验证策略

1. `cargo test -p rc3d-io --lib` — 全部通过
2. `cargo test -p rc3d-io --test export_step_stl --release` — Shape-2 面数正确，BSpline 曲面正常
3. Shape-2 期望：不再有 "fallback to Plane" 警告，BSpline 面产生曲面三角形
