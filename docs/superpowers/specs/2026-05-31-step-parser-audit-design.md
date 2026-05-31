# STEP Parser Correctness Audit

**日期**: 2026-05-31
**状态**: 待审批
**范围**: Part21 词法/语法/subsurface/类型映射/参数合并五个层级

---

## 审计架构

按 ISO 10303-21 标准的解析器层级，自底向上：

```
L0: 词法分析 (lexer.rs, token.rs)     — Token 定义、空白跳过、注释处理、编码
L1: 语法分析 (params.rs, instance.rs)  — 参数列表、简单entity、复杂entity检测
L2: Subsuper处理 (instance.rs, subsuper.rs) — Internal/External判定、记录选择、参数合并
L3: 类型映射 (entity_types.rs, build/surface.rs, build/curve.rs) — EntityType枚举、表面/曲线构建
L4: 参数合并 (subsuper.rs)            — 多记录参数合并策略、去重、排序
```

每层独立：**diff 当前实现 vs OCC/规范 → 分析差距 → 修复 → 添加测试 → 提交**

## L0: 词法分析

### 当前状态
- Token 类型: Ref, Keyword, String, Enum, Integer, Real, Omitted, LParen, RParen, Comma, Semi, Eq
- 空白跳过: `is_ascii_whitespace()` 跳过空格、tab、换行
- 注释: `/* ... */` 块注释支持
- 编码: `decode_step_bytes` 处理 UTF-8/ISO-8859-1

### 差距 vs OCC/规范

| 项 | 当前 | OCC/规范 | 严重度 |
|----|------|---------|--------|
| Binary token (`"01AB"`) | 未实现 | ISO 10303-21 §7.4.5.5 | 低 — 极少出现在几何 STEP |
| 空白字符集 | ASCII whitespace only | ISO 10303-21 §6：空格、水平制表、垂直制表、换页、回车、换行 | 低 — VT/FF不在正常STEP中 |
| 行号跟踪 | line/col 跟踪 | 同 | ✅ |
| 编码回退 | ISO-8859-1 fallback | ISO 10303-21 默认 Latin-1 | ✅ |

### 需要添加的测试

```rust
#[test] fn lex_all_token_types()         // 所有 token type round-trip
#[test] fn lex_escaped_apostrophe()       // 'it''s' → "it's"
#[test] fn lex_negative_real()            // -1.5e+3
#[test] fn lex_block_comment_in_data()    // /* comment */ should be skipped
#[test] fn lex_iso_8859_string()          // Latin-1 chars in strings
#[test] fn lex_binary_literal()           // "01AB" → Token::Binary (new)
```

## L1: 语法分析

### 当前状态
- 参数解析: parse_param_list_at 处理 nested lists、typed params、refs、strings、enums、numbers、omits
- Entity 检测: `rest.starts_with('(')` → subsuper, otherwise → simple
- sub-entity 标识: `find_outer_paren` 找到匹配的右括号

### 差距 vs OCC/规范

| 项 | 当前 | OCC/规范 | 严重度 |
|----|------|---------|--------|
| `find_outer_paren` | 单字符遍历计数深度 | OCC用栈跟踪嵌套层 | 中 — 对合法STEP正确，但边缘case可能出错 |
| 逗号规则 | 严格要求逗号分隔 | ISO允许尾部逗号(trailing comma) | 低 |
| 实体结束符 | 必须有 `;` | 同 | ✅ |
| 字符串转义 | 仅处理 `''` → `'` | ISO允许 `\x` 等转义？| 低 — ISO part21不用 `\x` |
| Typed parameter | `KEYWORD(list)` 形式 | 同，OCC称为 defined data | ✅ |

### 需要添加的测试

```rust
#[test] fn parse_entity_with_trailing_comma()    // (#1,(1.0,2.0,),)
#[test] fn parse_deeply_nested_param_list()      // 5层嵌套
#[test] fn parse_typed_param_nested()            // LENGTH_MEASURE(0.001)  
#[test] fn parse_enum_with_dots()               // .UNSPECIFIED. .T. .F.
#[test] fn parse_mixed_types_in_list()           // (#1, 2.0, 'str', .T., $)
```

## L2: Subsuper 处理

### 当前状态
- Internal: 所有记录在括号内 → `parse_subsuper_internal` → `leaf_index = len-1`
- External: 记录在括号外有 keyword → `parse_subsuper_external_prefix` → 排序 → `leaf_index = len-1`
- 参数合并: CompatMerge / StrictFidelity

### 已修复项
- ✅ `select_primary_record_structured`: 优先级表 + 结构性检查 (commit fcd40e5)
- ✅ `merge_all_params_structured`: 移除错误去重 (commit 308728a)

### 剩余差距

| 项 | 当前 | OCC/规范 | 严重度 |
|----|------|---------|--------|
| `sort_records_alphabetically` | 在 External 中排序 | OCC 不排序，保持原始顺序 | 已修复为 Internal 映射 |
| External/Internal 判定 | 基于 `after_starts_keyword` | OCC 检查 `)` 后是否有更多 keyword | 中 — 边界 case |
| wrapped entity ref 中 `#id` | 处理正确 | 同 | ✅ |

### 需要添加的测试

```rust
#[test] fn subsuper_internal_bspline_surface()  // BoundedSurface包裹BSpline
#[test] fn subsuper_external_entity()           // (SUPER())ENTITY(params)
#[test] fn subsuper_3_level_nesting()          // (GRANDCHILD() CHILD() PARENT())
#[test] fn subsuper_with_omitted_params()       // ($, *, .T.)
```

## L3: 类型映射

### 当前状态
- EntityType 枚举: 107 个变体
- `from_name`: 字符串 → EntityType 映射
- `build_surface`: 9 种表面类型 + supertype unwrap

### 差距 vs OCC/规范

| 项 | 当前 | OCC/规范 | 严重度 |
|----|------|---------|--------|
| ELEMENTARY_SURFACE | 已处理（unwrap to basis） | OCC同 | ✅ |
| SWEPT_SURFACE | 已处理（unwrap to basis） | OCC同 | ✅ |
| SURFACE | 已处理（unwrap via entity ref） | OCC同 | ✅ |
| 实体名称 vs EntityType | 不一致（e.g. `BOUNDED_CURVE`是"curve bounded"但名字可以是"BOUNDED_CURVE"从subsuper） | OCC用类型优先级 | 已修复 |
| build_curve 的 subsuper | `SURFACE_CURVE`/`SEAM_CURVE` unwrap，但 `BOUNDED_CURVE` subsuper 可能也被命名为上级类型 | OCC递归unwrap | 中 |

### 需要添加的测试

现有的 step_files.rs 集成测试已覆盖大多数类型。需要增加：
```rust
#[test] fn build_surface_from_elementary_surface_wrapper()
#[test] fn build_curve_from_bounded_curve_wrapper()
```

## L4: 参数合并

### 当前状态
- merge_all_params_structured: 串联所有记录参数，跳过 Omitted，去重已移除
- CompatMerge vs StrictFidelity 模式

### 已修复项
- ✅ 去重逻辑已移除 (commit 308728a)

### 剩余差距

| 项 | 当前 | OCC/规范 | 严重度 |
|----|------|---------|--------|
| 参数顺序 | 按记录顺序串联 | OCC 按 subtype→supertype 顺序 | 中 — 可能导致参数位置偏移 |
| 缺失 knot/weight 检测 | 硬编码 off+7 读 knot，失败则无回退 | 需动态定位 | 中 |

### 需要添加的测试

```rust
#[test] fn merge_params_bspline_surface_subsurface()
#[test] fn merge_params_omitted_flags_handling()
#[test] fn merge_params_order_preserved()
```

## 实施策略

5 层独立审查，每层一个任务：

| 任务 | 层级 | 预计影响 |
|------|------|---------|
| T1: 词法测试 | L0 | 低风险——只加测试 |
| T2: 语法测试 | L1 | 低风险——只加测试 |
| T3: Subsuper 边缘 case | L2 | 中风险——可能需要修复 |
| T4: 类型映射完善 | L3 | 中风险——build_curve 补充 |
| T5: 参数合并改进 | L4 | 中风险——scan_knot_data 回退 |

每个任务: **加测试 → 修复 → 验证 → 提交**
